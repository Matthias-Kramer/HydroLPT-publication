"""Normalization and merge logic for case specs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core._00_case_management.constants import SECTION_EXPORT_ORDER, SECTION_FIELD_ORDER
from core._00_case_management.loading import load_case_definition
from core._00_case_management.models import case_schema_from_payload, case_schema_to_payload


def flatten_sections(sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Flatten nested section dictionaries into one settings dictionary."""
    settings: dict[str, Any] = {}
    for section in sections.values():
        settings.update(section)
    return settings


def write_preset(path: Path, payload: dict[str, Any]) -> None:
    """Write a GUI preset JSON file."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_preset(path: Path) -> dict[str, Any]:
    """Read a GUI preset JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def safe_run_name(name: str) -> str:
    """Return a filesystem-safe run name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("._") or "gui_case"


def _order_fields_for_export(section_name: str, values: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of one section with GUI-ordered keys first."""
    ordered: dict[str, Any] = {}
    for key in SECTION_FIELD_ORDER.get(section_name, []):
        if key in values:
            ordered[key] = values[key]
    for key, value in values.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _order_sections_for_export(sections: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return sections ordered like the GUI tabs, with extras appended."""
    ordered: dict[str, dict[str, Any]] = {}
    for name in SECTION_EXPORT_ORDER:
        if name in sections:
            ordered[name] = _order_fields_for_export(name, dict(sections[name]))
    for name, values in sections.items():
        if name not in ordered:
            ordered[name] = _order_fields_for_export(name, dict(values))
    return ordered


def _normalize_sections_for_export(case_type: str, sections: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Remove inactive GUI fields so saved scripts reflect the current choices."""
    normalized = {name: dict(values) for name, values in sections.items()}
    hydraulic_settings = normalized.get("HYDRAULIC_INPUT", {})
    run_mode = str(hydraulic_settings.get("run_mode", "steady")).strip().lower()
    is_transient = run_mode == "transient"
    transport_settings = normalized.get("TRANSPORT_SETTINGS", {})
    particle_settings = normalized.get("PARTICLE_SETTINGS", {})
    if "rng_seed" in particle_settings and "rng_seed" not in transport_settings:
        transport_settings["rng_seed"] = particle_settings["rng_seed"]
    particle_settings.pop("rng_seed", None)
    if not is_transient:
        transport_settings.pop("tRelease", None)
        for key in ("hyd_time_start", "hyd_dt", "hyd_time_end"):
            hydraulic_settings.pop(key, None)
    location_mode = str(particle_settings.get("location_mode", "coordinates")).strip().lower()
    if location_mode == "click":
        particle_settings.pop("release_x", None)
        particle_settings.pop("release_y", None)
    if str(particle_settings.get("release_mode", "bulk")).strip().lower() != "continuous":
        particle_settings.pop("release_dt", None)
    if str(transport_settings.get("transportModel", "random_walk")).strip().lower() != "langevin":
        transport_settings.pop("TL_horizontal", None)
        transport_settings.pop("TL_vertical", None)
    particle_evo_settings = normalized.get("PARTICLE_EVO_SETTINGS", {})
    biofouling_on = str(particle_evo_settings.get("biofouling", "off")).strip().lower() == "on"
    degradation_on = str(particle_evo_settings.get("degradation", "off")).strip().lower() == "on"
    if biofouling_on and degradation_on:
        particle_evo_settings["degradation"] = "off"
        degradation_on = False
    if not biofouling_on:
        for key in ("BT0", "BR", "rho_biofilm"):
            particle_evo_settings.pop(key, None)
    if not degradation_on:
        particle_evo_settings.pop("DR", None)
    boundary_settings = normalized.get("BOUNDARY_SETTINGS", {})
    if str(boundary_settings.get("surfacePolicy", "always_reflect")).strip().lower() != "probabilistic_detachment":
        boundary_settings.pop("surfaceDetachmentSigmaStar", None)
    if str(boundary_settings.get("bedPolicy", "always_reflect")).strip().lower() != "probabilistic_entrainment":
        boundary_settings.pop("bedEntrainmentSigmaStar", None)
    if str(boundary_settings.get("uphillPolicy", "off")).strip().lower() != "stop":
        boundary_settings.pop("dzUpMax", None)
    shape = str(particle_settings.get("shape", "")).strip().lower()
    if shape == "sphere":
        particle_settings.pop("I", None)
        particle_settings.pop("S", None)
    elif shape in {"cylinder", "disk"}:
        particle_settings.pop("I", None)
    if case_type == "synthetic":
        hydraulic = normalized.get("HYDRAULIC_INPUT", {})
        hydraulic.setdefault("hydraulicClosureModel", "standard_z0")
        hydraulic["hydraulicClosureModel"] = "standard_z0"
        hydraulic["hydraulicPrimaryVariable"] = "U"
        hydraulic.pop("ks", None)
        if is_transient:
            for key in ("U", "V", "ustar"):
                hydraulic.pop(key, None)
        else:
            for key in ("transient_shape", "switch_fraction", "U0", "U1", "ustar0", "ustar1", "V0", "V1"):
                hydraulic.pop(key, None)
        for key in ("ustar", "ustar0", "ustar1"):
            hydraulic.pop(key, None)
    elif case_type in {"basement", "hecras"}:
        hydraulic = normalized.setdefault("HYDRAULIC_INPUT", {})
        hydraulic["hydraulicClosureModel"] = "external_adapter"
        transport_settings["transportVelocityMode"] = "depth_averaged"
        for key in ("hydraulicPrimaryVariable", "U", "V", "ustar", "z0", "transient_shape", "switch_fraction", "U0", "U1", "ustar0", "ustar1", "V0", "V1"):
            hydraulic.pop(key, None)
    return normalized


def merge_case_definition_with_spec(case: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Merge a loaded base case with GUI spec overrides."""
    merged_sections = _normalize_sections_for_export(case["case_type"], spec.get("sections", {}))
    merged_plots = dict(case.get("plots", {}))
    merged_plots.update(spec.get("plots", {}))
    schema = case_schema_from_payload(
        case_type=case["case_type"],
        sections=_order_sections_for_export(merged_sections),
        plots=merged_plots,
        export_data=bool(spec.get("export_data", case.get("settings", {}).get("export_data", False))),
        xdmf_path=spec.get("xdmf_path") or case.get("xdmf_path", ""),
        user_notes=case.get("user_notes", ""),
    )
    return case_schema_to_payload(schema) | {
        "case_type": case["case_type"],
    }


def normalize_case_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical normalized case payload for save/export/run flows."""
    if spec.get("base_script_path"):
        case = load_case_definition(Path(spec["base_script_path"]))
        return merge_case_definition_with_spec(case, spec)
    schema = case_schema_from_payload(
        case_type=spec["case_type"],
        sections=_order_sections_for_export(_normalize_sections_for_export(spec["case_type"], spec["sections"])),
        plots=dict(spec.get("plots", {})),
        export_data=bool(spec.get("export_data", False)),
        xdmf_path=spec.get("xdmf_path", ""),
        user_notes=(
            "Anything written here is preserved by the GUI exporter.\n\n"
            "Use this block for reminders, assumptions, calibration notes, warnings, or TODOs."
        ),
    )
    return case_schema_to_payload(schema)
