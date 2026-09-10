"""Case-script formatting and export helpers."""

from __future__ import annotations

import ast
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from core._00_case_management.constants import (
    GUI_CASE_FILENAME,
    OUTPUT_COMMENTS,
    OUTPUT_FIELD_ORDER,
    SECTION_COMMENTS,
    SECTION_FIELD_ORDER,
    SECTION_HEADING_GROUPS,
)
from core._00_case_management.normalize import flatten_sections, normalize_case_spec, safe_run_name


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _to_offset(offsets: list[int], lineno: int, col_offset: int) -> int:
    return offsets[lineno - 1] + col_offset


def _format_py_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, list):
        return "[" + ", ".join(_format_py_value(item) for item in value) + "]"
    if isinstance(value, tuple):
        inner = ", ".join(_format_py_value(item) for item in value)
        suffix = "," if len(value) == 1 else ""
        return f"({inner}{suffix})"
    if isinstance(value, dict):
        parts = [f"{_format_py_value(k)}: {_format_py_value(v)}" for k, v in value.items()]
        return "{" + ", ".join(parts) + "}"
    return repr(value)


def _order_fields_for_export(section_name: str, values: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in SECTION_FIELD_ORDER.get(section_name, []):
        if key in values:
            ordered[key] = values[key]
    for key, value in values.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _shape_specific_particle_comments(values: dict[str, Any]) -> dict[str, str]:
    comments = dict(SECTION_COMMENTS.get("PARTICLE_SETTINGS", {}))
    shape = str(values.get("shape", "")).strip().lower()
    if shape == "sphere":
        comments["L"] = "sphere diameter D [m]"
        comments["I"] = "same as D for internal sphere handling"
        comments["S"] = "same as D for internal sphere handling"
    elif shape == "cylinder":
        comments["L"] = "cylinder length L [m]"
        comments["S"] = "cylinder diameter D [m]"
        comments["I"] = "same as D for internal cylinder handling"
    elif shape == "disk":
        comments["L"] = "disk diameter D [m]"
        comments["S"] = "disk thickness S [m]"
        comments["I"] = "same as D for internal disk handling"
    elif shape == "ellipsoid":
        comments["L"] = "ellipsoid long axis L [m]"
        comments["I"] = "ellipsoid intermediate axis I [m]"
        comments["S"] = "ellipsoid short axis S [m]"
    elif shape == "prism":
        comments["L"] = "prism long axis L [m]"
        comments["I"] = "prism intermediate axis I [m]"
        comments["S"] = "prism short axis S [m]"
    return comments


def _export_particle_settings(values: dict[str, Any]) -> dict[str, Any]:
    exported = dict(values)
    shape = str(exported.get("shape", "")).strip().lower()
    if shape == "sphere":
        exported.pop("I", None)
        exported.pop("S", None)
    elif shape in {"cylinder", "disk"}:
        exported.pop("I", None)
    return exported


def _format_commented_dict(name: str, values: dict[str, Any], *, case_type: str | None = None) -> str:
    if name == "PARTICLE_SETTINGS":
        values = _export_particle_settings(values)
    values = _order_fields_for_export(name, values)
    comments = SECTION_COMMENTS.get(name, {})
    if name == "PARTICLE_SETTINGS":
        comments = _shape_specific_particle_comments(values)
    elif name == "TRANSPORT_SETTINGS" and case_type in {"basement", "hecras"}:
        comments = dict(comments)
        comments["transportVelocityMode"] = '"depth_averaged"'
    heading_lookup: dict[str, str] = {}
    for heading, keys in SECTION_HEADING_GROUPS.get(name, []):
        for key in keys:
            heading_lookup.setdefault(key, heading)
    lines = ["{"]
    items = list(values.items())
    previous_heading: str | None = None
    for index, (key, value) in enumerate(items):
        heading = heading_lookup.get(key)
        if heading and heading != previous_heading:
            if len(lines) > 1:
                lines.append("")
            lines.append(f"    # {heading}")
            previous_heading = heading
        comma = "," if index < len(items) - 1 else ""
        line = f'    "{key}": {_format_py_value(value)}{comma}'
        comment = comments.get(key)
        if comment:
            line += f"  # {comment}"
        lines.append(line)
    lines.append("}")
    return "\n".join(lines)


def _format_settings_dict(section_names: list[str]) -> str:
    lines = ["{"]
    lines.extend(f"    **{section_name}," for section_name in section_names)
    lines.append("    **OUTPUT_SETTINGS,")
    lines.append("}")
    return "\n".join(lines)


def _format_output_settings(merged: dict[str, Any]) -> str:
    values: dict[str, Any] = {"export_data": bool(merged.get("export_data", False))}
    for key in OUTPUT_FIELD_ORDER:
        if key != "export_data" and key in merged.get("plots", {}):
            values[key] = merged["plots"][key]
    for key, value in merged.get("plots", {}).items():
        if key not in values:
            values[key] = value
    ordered: dict[str, Any] = {}
    for key in OUTPUT_FIELD_ORDER:
        if key in values:
            ordered[key] = values[key]
    for key, value in values.items():
        if key not in ordered:
            ordered[key] = value
    lines = ["{"]
    items = list(ordered.items())
    heading_lookup: dict[str, str] = {}
    for heading, keys in SECTION_HEADING_GROUPS.get("OUTPUT_SETTINGS", []):
        for key in keys:
            heading_lookup.setdefault(key, heading)
    previous_heading: str | None = None
    for index, (key, value) in enumerate(items):
        heading = heading_lookup.get(key)
        if heading and heading != previous_heading:
            if len(lines) > 1:
                lines.append("")
            lines.append(f"    # {heading}")
            previous_heading = heading
        comma = "," if index < len(items) - 1 else ""
        line = f'    "{key}": {_format_py_value(value)}{comma}'
        comment = OUTPUT_COMMENTS.get(key)
        if comment:
            line += f"  # {comment}"
        lines.append(line)
    lines.append("}")
    return "\n".join(lines)


def _format_assignment_value(name: str, merged: dict[str, Any]) -> str:
    if name == "SETTINGS":
        return _format_settings_dict(list(merged.get("sections", {})))
    if name == "OUTPUT_SETTINGS":
        return _format_output_settings(merged)
    if name == "XDMF_PATH":
        return _format_py_value(merged.get("xdmf_path", ""))
    if name == "PLAN_HDF_PATH":
        return _format_py_value(merged.get("xdmf_path", ""))
    if name == "USER_NOTES":
        notes = str(merged.get("user_notes", "")).strip("\n")
        return f'"""\n{notes}\n"""'
    return _format_commented_dict(name, merged["sections"][name], case_type=merged.get("case_type"))


def _write_text_file_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path_str = tempfile.mkstemp(prefix=f"{path.stem}_", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _should_replace_assignment_name(name: str, merged: dict[str, Any]) -> bool:
    """Return whether a top-level assignment belongs to the GUI-managed block."""
    managed_names = set(merged.get("sections", {}))
    managed_names.update({"OUTPUT_SETTINGS", "SETTINGS", "PLOTS", "USER_NOTES"})
    if merged.get("case_type") == "basement":
        managed_names.add("XDMF_PATH")
    if merged.get("case_type") == "hecras":
        managed_names.add("PLAN_HDF_PATH")
    return name in managed_names or name.endswith("_SETTINGS")


def _rewrite_case_script(base_script_path: Path, target_path: Path, merged: dict[str, Any]) -> None:
    source = Path(base_script_path).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    offsets = _line_offsets(source)
    section_names = list(merged.get("sections", {}))
    insert_at: int | None = None
    block_start: int | None = None
    block_end: int | None = None
    for node in tree.body:
        if insert_at is None and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            insert_at = _to_offset(offsets, node.lineno, node.col_offset)
        name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            name = node.target.id
            value_node = node.value
        if name is None or value_node is None or not _should_replace_assignment_name(name, merged):
            continue
        start = _to_offset(offsets, node.lineno, node.col_offset)
        end = _to_offset(offsets, node.end_lineno, node.end_col_offset)
        block_start = start if block_start is None else min(block_start, start)
        block_end = end if block_end is None else max(block_end, end)
    if insert_at is None:
        insert_at = len(source)
    ordered_block_names = list(section_names) + ["OUTPUT_SETTINGS"]
    if merged.get("case_type") == "basement":
        ordered_block_names.append("XDMF_PATH")
    if merged.get("case_type") == "hecras":
        ordered_block_names.append("PLAN_HDF_PATH")
    ordered_block_names.extend(["USER_NOTES", "SETTINGS"])
    replacement_block = "\n\n".join(
        f"{name} = {_format_assignment_value(name, merged)}" for name in ordered_block_names
    ) + "\n\n"
    updated_source = source[:insert_at] + replacement_block + source[insert_at:] if block_start is None or block_end is None else source[:block_start] + replacement_block + source[block_end:]
    updated_source = updated_source.replace(
        "run_synthetic_case(__file__, settings=SETTINGS, plots=PLOTS)",
        "run_synthetic_case(__file__, settings=SETTINGS)",
    )
    updated_source = updated_source.replace(
        "run_basement_case(__file__, settings=SETTINGS, plots=PLOTS, xdmf_path=XDMF_PATH)",
        "run_basement_case(__file__, settings=SETTINGS, xdmf_path=XDMF_PATH)",
    )
    updated_source = updated_source.replace(
        "run_basement_case(__file__, settings=SETTINGS, plots=PLOTS)",
        "run_basement_case(__file__, settings=SETTINGS, xdmf_path=XDMF_PATH)",
    )
    updated_source = updated_source.replace(
        "run_hecras_case(__file__, settings=SETTINGS, plots=PLOTS, plan_hdf_path=PLAN_HDF_PATH)",
        "run_hecras_case(__file__, settings=SETTINGS, plan_hdf_path=PLAN_HDF_PATH)",
    )
    updated_source = updated_source.replace(
        "run_hecras_case(__file__, settings=SETTINGS, plots=PLOTS)",
        "run_hecras_case(__file__, settings=SETTINGS, plan_hdf_path=PLAN_HDF_PATH)",
    )
    _write_text_file_atomic(Path(target_path), updated_source)


def _build_export_payload(spec: dict[str, Any]) -> tuple[Path, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    output_dir = Path(spec["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = safe_run_name(spec.get("run_name", "gui_case"))
    merged = normalize_case_spec(spec)
    settings = flatten_sections(merged["sections"])
    settings["export_data"] = bool(merged.get("export_data", False))
    plots = dict(merged.get("plots", {}))
    script_path = Path(spec["base_script_path"]).resolve() if spec.get("base_script_path") else output_dir / GUI_CASE_FILENAME
    return output_dir, run_name, merged, settings, {"script_path": script_path, "plots": plots}


def save_case_file(spec: dict[str, Any]) -> Path:
    _output_dir, _run_name, merged, _settings, export_info = _build_export_payload(spec)
    script_path = export_info["script_path"]
    sections_text = [f"{name} = {_format_commented_dict(name, values)}\n" for name, values in merged["sections"].items()]
    if spec.get("base_script_path"):
        _rewrite_case_script(Path(spec["base_script_path"]).resolve(), script_path, merged)
        return script_path
    if merged["case_type"] == "synthetic":
        body = (
            "from __future__ import annotations\n\n"
            "import sys\nfrom pathlib import Path\n\n"
            "ROOT = Path(__file__).resolve().parents[1]\n"
            "if str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n\n"
            "from core._01_hydraulic_input_and_adapter.synthetic import run_synthetic_case\n\n"
            + "".join(sections_text)
            + f"OUTPUT_SETTINGS = {_format_output_settings(merged)}\n\n"
            + f"SETTINGS = {_format_settings_dict(list(merged['sections']))}\n\n"
            + f"USER_NOTES = {_format_assignment_value('USER_NOTES', merged)}\n\n"
            + "def main() -> None:\n    run_synthetic_case(__file__, settings=SETTINGS)\n\n"
            + "if __name__ == '__main__':\n    main()\n"
        )
    elif merged["case_type"] == "basement":
        body = (
            "from __future__ import annotations\n\n"
            "import sys\nfrom pathlib import Path\n\n"
            "ROOT = Path(__file__).resolve().parents[1]\n"
            "if str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n\n"
            "from core._01_hydraulic_input_and_adapter.basement import run_basement_case\n\n"
            + "".join(sections_text)
            + f"OUTPUT_SETTINGS = {_format_output_settings(merged)}\n\n"
            + f"SETTINGS = {_format_settings_dict(list(merged['sections']))}\n\n"
            + f"XDMF_PATH = {_format_py_value(merged.get('xdmf_path', ''))}\n\n"
            + f"USER_NOTES = {_format_assignment_value('USER_NOTES', merged)}\n\n"
            + "def main() -> None:\n    run_basement_case(__file__, settings=SETTINGS, xdmf_path=XDMF_PATH)\n\n"
            + "if __name__ == '__main__':\n    main()\n"
        )
    elif merged["case_type"] == "hecras":
        body = (
            "from __future__ import annotations\n\n"
            "import sys\nfrom pathlib import Path\n\n"
            "ROOT = Path(__file__).resolve().parents[1]\n"
            "if str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n\n"
            "from core._01_hydraulic_input_and_adapter.hecras import run_hecras_case\n\n"
            + "".join(sections_text)
            + f"OUTPUT_SETTINGS = {_format_output_settings(merged)}\n\n"
            + f"SETTINGS = {_format_settings_dict(list(merged['sections']))}\n\n"
            + f"PLAN_HDF_PATH = {_format_py_value(merged.get('xdmf_path', ''))}\n\n"
            + f"USER_NOTES = {_format_assignment_value('USER_NOTES', merged)}\n\n"
            + "def main() -> None:\n    run_hecras_case(__file__, settings=SETTINGS, plan_hdf_path=PLAN_HDF_PATH)\n\n"
            + "if __name__ == '__main__':\n    main()\n"
        )
    else:
        raise ValueError(f"Unsupported case type: {merged['case_type']}")
    _write_text_file_atomic(script_path, body)
    return script_path


def generate_case_files(spec: dict[str, Any]) -> tuple[Path | None, Path | None, Path | None]:
    script_path = save_case_file(spec)
    return script_path, None, None
