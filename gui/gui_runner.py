"""CLI-style runner used by the GUI for subprocess execution of case specs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from core.app import execute_case_spec
from core._00_case_management import (
    apply_overrides_to_module,
    load_case_definition,
)


def _configure_line_buffered_streams() -> None:
    """Force line-buffered stdout/stderr for responsive GUI log capture."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def _load_spec(spec_path: Path) -> dict[str, Any]:
    """Load a GUI case specification from JSON."""
    return json.loads(spec_path.read_text(encoding="utf-8"))


def _run_spec_file(spec_path: Path) -> int:
    """Execute a serialized case spec JSON file."""
    spec = _load_spec(spec_path)
    export_case_files = bool(spec.get("export_case_files", False))
    show_plots = bool(spec.get("show_plots", True))
    try:
        generated_path, config_json_path, config_txt_path, _run_info = execute_case_spec(
            spec,
            show_plots=show_plots,
            block_on_plots=False,
            export_case_files=export_case_files,
        )
    except Exception as exc:
        print(exc)
        return 1

    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    if export_case_files:
        if generated_path is not None:
            print(f"Generated file: {generated_path}")

    if show_plots:
        print("Opening plots...")
        plt.show()
        print("All figures closed")
    return 0


def _run_script_with_optional_overrides(
    script_path: Path,
    overrides_path: Path | None,
) -> int:
    """Execute a Python case script, optionally with JSON overrides applied."""
    case = load_case_definition(script_path)
    module = case["module"]

    if overrides_path is not None and overrides_path.exists():
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
        apply_overrides_to_module(module, overrides)

    os.chdir(script_path.parent)
    module.main()
    return 0


def main() -> int:
    """Run a HydroLPT case from either a spec JSON or a Python case script."""
    _configure_line_buffered_streams()

    if len(sys.argv) < 2:
        print("Usage: python -m gui.gui_runner <script_path|spec_json> [overrides_json]")
        return 1

    first_arg = Path(sys.argv[1]).resolve()
    overrides_path = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else None

    if first_arg.suffix.lower() == ".json" and overrides_path is None:
        return _run_spec_file(first_arg)

    return _run_script_with_optional_overrides(first_arg, overrides_path)


if __name__ == "__main__":
    raise SystemExit(main())

