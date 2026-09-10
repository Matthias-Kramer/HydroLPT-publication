"""Case discovery and loading helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from core._00_case_management.config import get_default_paths
from core._00_case_management.constants import GUI_CASE_FILENAME, RUNS_DIR
from core._00_case_management.models import case_schema_from_payload, case_schema_to_payload
from core._02_particle_initialisation import complete_particle_dimensions


def discover_run_scripts(root: Path | None = None) -> list[Path]:
    """Discover GUI-exported run scripts below the runs tree."""
    search_root = RUNS_DIR if root is None else Path(root)
    return sorted(search_root.rglob(GUI_CASE_FILENAME))


def get_case_type(module: ModuleType) -> str:
    """Infer case type from which runner symbol the module exposes."""
    if hasattr(module, "run_synthetic_case"):
        return "synthetic"
    if hasattr(module, "run_hecras_case"):
        return "hecras"
    if hasattr(module, "run_basement_case"):
        return "basement"
    return "unknown"


def _load_module_from_path(script_path: Path) -> ModuleType:
    """Load a case script as a Python module without importing by package name."""
    script_path = Path(script_path).resolve()
    module_name = f"_hydrolpt_gui_{script_path.stem}_{abs(hash(script_path))}"
    source = script_path.read_text(encoding="utf-8-sig")
    code = compile(source, str(script_path), "exec")
    module = ModuleType(module_name)
    module.__file__ = str(script_path)
    module.__package__ = None
    module.__spec__ = importlib.util.spec_from_file_location(module_name, script_path)
    exec(code, module.__dict__)
    return module


def _resolve_loaded_xdmf_path(script_path: Path, module: ModuleType) -> str:
    """Resolve the XDMF path for loaded BASEMENT cases."""
    raw_xdmf_path = getattr(module, "XDMF_PATH", "")
    default_xdmf = get_default_paths(script_path)["xdmf"]
    if raw_xdmf_path:
        candidate = Path(str(raw_xdmf_path)).expanduser()
        candidate = (script_path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if candidate.exists():
            return str(candidate)
    if default_xdmf.exists():
        return str(default_xdmf)
    return str(raw_xdmf_path)


def _resolve_loaded_plan_hdf_path(script_path: Path, module: ModuleType) -> str:
    """Resolve the HEC-RAS plan HDF path for loaded HEC-RAS cases."""
    raw_plan_path = getattr(module, "PLAN_HDF_PATH", "")
    if raw_plan_path:
        candidate = Path(str(raw_plan_path)).expanduser()
        candidate = (script_path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if candidate.exists():
            return str(candidate)
    return str(raw_plan_path)


def load_case_definition(script_path: Path) -> dict[str, Any]:
    """Load a case script and extract GUI-relevant globals from it."""
    script_path = Path(script_path).resolve()
    module = _load_module_from_path(script_path)
    sections: dict[str, dict[str, Any]] = {}
    for name, value in vars(module).items():
        if (name.endswith("_SETTINGS") or name == "HYDRAULIC_INPUT") and isinstance(value, dict):
            sections[name] = dict(value)
    particle_settings = sections.get("PARTICLE_SETTINGS")
    if isinstance(particle_settings, dict) and "shape" in particle_settings and "L" in particle_settings:
        shape = str(particle_settings.get("shape", "")).strip().lower()
        try:
            l_val, i_val, s_val = complete_particle_dimensions(
                shape,
                particle_settings.get("L"),
                particle_settings.get("I"),
                particle_settings.get("S"),
            )
            particle_settings["L"] = l_val
            particle_settings["I"] = i_val
            particle_settings["S"] = s_val
        except Exception:
            pass
    settings = dict(getattr(module, "SETTINGS", {}))
    output_settings = getattr(module, "OUTPUT_SETTINGS", {})
    plots: dict[str, Any] = {}
    if isinstance(output_settings, dict):
        plots.update({key: value for key, value in output_settings.items() if key != "export_data"})
        if "export_data" not in settings:
            settings["export_data"] = bool(output_settings.get("export_data", False))
    legacy_plots = getattr(module, "PLOTS", {})
    if isinstance(legacy_plots, dict):
        plots.update(legacy_plots)
    xdmf_path = getattr(module, "XDMF_PATH", "")
    module_case_type = get_case_type(module)
    if module_case_type == "basement":
        xdmf_path = _resolve_loaded_xdmf_path(script_path, module)
    elif module_case_type == "hecras":
        xdmf_path = _resolve_loaded_plan_hdf_path(script_path, module)
    schema = case_schema_from_payload(
        case_type=module_case_type,
        sections=sections,
        plots=plots,
        export_data=bool(settings.get("export_data", False)),
        xdmf_path=xdmf_path,
        user_notes=getattr(module, "USER_NOTES", ""),
    )
    payload = case_schema_to_payload(schema)
    payload.update({
        "script_path": script_path,
        "module": module,
        "settings": settings,
    })
    return payload


def apply_overrides_to_module(module: ModuleType, overrides: dict[str, Any]) -> None:
    """Apply GUI override dictionaries directly onto an imported case module."""
    settings_override = dict(overrides.get("settings", {}))
    plots_override = dict(overrides.get("plots", {}))
    if hasattr(module, "SETTINGS") and isinstance(module.SETTINGS, dict):
        module.SETTINGS = dict(module.SETTINGS)
        module.SETTINGS.update(settings_override)
        module.SETTINGS.update(plots_override)
    if hasattr(module, "OUTPUT_SETTINGS") and isinstance(module.OUTPUT_SETTINGS, dict):
        module.OUTPUT_SETTINGS = dict(module.OUTPUT_SETTINGS)
        module.OUTPUT_SETTINGS.update(plots_override)
    elif hasattr(module, "PLOTS") and isinstance(module.PLOTS, dict):
        module.PLOTS = dict(module.PLOTS)
        module.PLOTS.update(plots_override)
    for name, section_override in overrides.get("sections", {}).items():
        current = getattr(module, name, None)
        if isinstance(current, dict):
            updated = dict(current)
            updated.update(section_override)
            setattr(module, name, updated)
