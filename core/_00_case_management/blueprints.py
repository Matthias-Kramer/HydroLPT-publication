"""Default GUI blueprints and case-to-blueprint conversion."""

from __future__ import annotations

import copy
from typing import Any


def default_case_blueprints() -> dict[str, dict[str, Any]]:
    """Return default GUI blueprints for synthetic, BASEMENT, and HEC-RAS cases."""
    synthetic_sections = {
        "HYDRAULIC_INPUT": {
            "run_mode": "steady",
            "length": 20.0,
            "domain_width": 1.0,
            "wetted_width": 0.6,
            "water_depth": 0.5,
            "rho_f": 1000.0,
            "nu": 1e-6,
            "g": 9.81,
            "hydraulicClosureModel": "standard_z0",
            "hydraulicPrimaryVariable": "U",
            "U": 0.3,
            "V": 0.0,
            "z0": 5e-4,
            "transient_shape": "step",
            "switch_fraction": 0.5,
            "U0": 0.03,
            "U1": 0.30,
            "hyd_time_start": 0.0,
            "hyd_dt": 1.0,
            "hyd_time_end": 100.0,
            "V0": 0.0,
            "V1": 0.0,
        },
        "PARTICLE_SETTINGS": {
            "location_mode": "click",
            "shape": "sphere",
            "rho_p": 1010.0,
            "L": 5e-3,
            "I": 5e-3,
            "S": 5e-3,
            "release_x": 1.0,
            "release_y": 0.5,
            "release_mode": "bulk",
            "n_per_center": 300,
            "release_dt": 1.0,
            "nTrack": 100,
            "releaseSigma": 0.01,
            "zFrac": 0.5,
        },
        "PARTICLE_EVO_SETTINGS": {
            "biofouling": "off",
            "BT0": 0.0,
            "BR": 0.0,
            "rho_biofilm": 1388.0,
            "degradation": "off",
            "DR": 0.0,
        },
        "TRANSPORT_SETTINGS": {
            "dt": 0.1,
            "tTrack": 60.0,
            "tRelease": 0.0,
            "hydraulicFieldMode": "cellwise",
            "transportVelocityMode": "depth_averaged",
            "transportModel": "random_walk",
            "rng_seed": None,
            "useRWx": True,
            "useRWy": True,
            "useRWz": True,
            "betaKh": 0.6,
            "KhMax": 5.0,
            "rAniso": 3.0,
            "alphaKz": 1.0,
            "KzMax": 0.3,
            "TL_horizontal": 2.0,
            "TL_vertical": 0.5,
        },
        "BOUNDARY_SETTINGS": {
            "surfacePolicy": "always_reflect",
            "bedPolicy": "always_reflect",
            "surfaceVerticalDragCoeff": 2.0,
            "surfaceContactAngleDeg": 105.0,
            "surfaceDetachmentSigmaStar": 0.2,
            "tanphi_ratio": 0.55,
            "bedEntrainmentSigmaStar": 0.2,
            "outsidePolicy": "stop",
            "dryPolicy": "reflect",
            "hmin": 1e-2,
            "uphillPolicy": "off",
            "dzUpMax": 0.4,
        },
    }
    basement_sections = {
        "HYDRAULIC_INPUT": {
            "run_mode": "steady",
            "rho_f": 1000.0,
            "nu": 1e-6,
            "g": 9.81,
            "hyd_time_start": 0.0,
            "hyd_dt": 1.0,
            "hyd_time_end": 100.0,
            "hydraulicClosureModel": "external_adapter",
        },
        "PARTICLE_SETTINGS": {
            "location_mode": "click",
            "shape": "sphere",
            "rho_p": 1010.0,
            "L": 5e-3,
            "I": 5e-3,
            "S": 5e-3,
            "release_x": 787340.0,
            "release_y": 155115.0,
            "release_mode": "bulk",
            "n_per_center": 300,
            "release_dt": 1.0,
            "releaseSigma": 2.0,
            "zFrac": 0.9,
            "nTrack": 6,
        },
        "PARTICLE_EVO_SETTINGS": {
            "biofouling": "off",
            "BT0": 0.0,
            "BR": 0.0,
            "rho_biofilm": 1388.0,
            "degradation": "off",
            "DR": 0.0,
        },
        "TRANSPORT_SETTINGS": {
            "dt": 0.25,
            "tTrack": 500.0,
            "tRelease": 100.0,
            "hydraulicFieldMode": "cellwise",
            "transportVelocityMode": "depth_averaged",
            "transportModel": "random_walk",
            "rng_seed": None,
            "useRWx": True,
            "useRWy": True,
            "useRWz": True,
            "betaKh": 0.6,
            "KhMax": 5.0,
            "rAniso": 3.0,
            "alphaKz": 1.0,
            "KzMax": 0.3,
            "TL_horizontal": 2.0,
            "TL_vertical": 0.5,
        },
        "BOUNDARY_SETTINGS": {
            "surfacePolicy": "always_reflect",
            "bedPolicy": "reflect_if_ustar_gt_crit",
            "surfaceVerticalDragCoeff": 2.0,
            "surfaceContactAngleDeg": 105.0,
            "surfaceDetachmentSigmaStar": 0.2,
            "tanphi_ratio": 0.55,
            "bedEntrainmentSigmaStar": 0.2,
            "outsidePolicy": "stop",
            "dryPolicy": "reflect",
            "hmin": 1e-2,
            "uphillPolicy": "off",
            "dzUpMax": 0.4,
        },
    }
    hecras_sections = copy.deepcopy(basement_sections)
    hecras_sections["HYDRAULIC_INPUT"].update(
        {
            "run_mode": "transient",
            "hyd_time_start": 0.0,
            "hyd_dt": 300.0,
            "hyd_time_end": 86400.0,
            "hydraulicClosureModel": "external_adapter",
        }
    )
    hecras_sections["PARTICLE_SETTINGS"].update(
        {
            "release_x": 0.0,
            "release_y": 0.0,
            "releaseSigma": 1.0,
        }
    )
    hecras_sections["TRANSPORT_SETTINGS"].update(
        {
            "dt": 1.0,
            "tTrack": 3600.0,
            "tRelease": 18000.0,
        }
    )
    return {
        "synthetic": {
            "case_type": "synthetic",
            "export_data": False,
            "sections": synthetic_sections,
            "plots": {"output_dt": 0.1, "binary_map": True, "binary_map_marker_size": 5.0, "binary_map_mesh_overlay": True, "trajectories": True},
        },
        "basement": {
            "case_type": "basement",
            "export_data": False,
            "sections": basement_sections,
            "plots": {"output_dt": 0.25, "binary_map": True, "binary_map_marker_size": 5.0, "binary_map_mesh_overlay": True, "trajectories": True},
        },
        "hecras": {
            "case_type": "hecras",
            "export_data": False,
            "sections": hecras_sections,
            "plots": {"output_dt": 1.0, "binary_map": True, "binary_map_marker_size": 5.0, "binary_map_mesh_overlay": True, "trajectories": True},
        },
    }


def build_blueprint_from_case(case: dict[str, Any]) -> dict[str, Any]:
    """Build a GUI-editable blueprint from an existing case definition."""
    case_type = case["case_type"]
    blueprint = copy.deepcopy(default_case_blueprints()[case_type])
    sections = case.get("sections", {})
    for section_name, section_values in blueprint["sections"].items():
        source_values = dict(sections.get(section_name, {}))
        for key in list(section_values):
            if key in source_values:
                section_values[key] = source_values[key]
    hydraulic_settings = blueprint["sections"].get("HYDRAULIC_INPUT", {})
    particle_settings = blueprint["sections"].get("PARTICLE_SETTINGS", {})
    transport_settings = blueprint["sections"].get("TRANSPORT_SETTINGS", {})
    source_particle_settings = sections.get("PARTICLE_SETTINGS", {})
    source_transport_settings = sections.get("TRANSPORT_SETTINGS", {})
    case_plots = dict(case.get("plots", {}))
    if "rng_seed" in source_transport_settings:
        transport_settings["rng_seed"] = source_transport_settings["rng_seed"]
    elif "rng_seed" in source_particle_settings:
        transport_settings["rng_seed"] = source_particle_settings["rng_seed"]
    if "rAniso" not in source_transport_settings and "rAniso" in case_plots:
        transport_settings["rAniso"] = case_plots["rAniso"]
    particle_settings.pop("rng_seed", None)
    explicit_run_mode = str(hydraulic_settings.get("run_mode", "")).strip().lower()
    if not explicit_run_mode:
        explicit_run_mode = str(case.get("settings", {}).get("run_mode", "")).strip().lower()
    if explicit_run_mode in {"steady", "transient"}:
        hydraulic_settings["run_mode"] = explicit_run_mode
    else:
        hyd_time_end = hydraulic_settings.get("hyd_time_end")
        hyd_time_start = hydraulic_settings.get("hyd_time_start")
        try:
            if hyd_time_end is not None and hyd_time_start is not None and float(hyd_time_end) > float(hyd_time_start):
                hydraulic_settings["run_mode"] = "transient"
        except (TypeError, ValueError):
            pass
    if particle_settings.get("location_mode") == "manual":
        particle_settings["location_mode"] = "coordinates"
    manual_centers = sections.get("PARTICLE_SETTINGS", {}).get("manual_centers", [])
    if manual_centers:
        particle_settings["release_x"] = ", ".join(str(center[0]) for center in manual_centers)
        particle_settings["release_y"] = ", ".join(str(center[1]) for center in manual_centers)
    elif particle_settings.get("location_mode") == "click":
        particle_settings["release_x"] = ""
        particle_settings["release_y"] = ""
    for key in list(blueprint["plots"]):
        if key in case_plots:
            blueprint["plots"][key] = case_plots[key]
    for key, value in case_plots.items():
        if key not in blueprint["plots"]:
            blueprint["plots"][key] = value
    blueprint["script_path"] = str(case["script_path"])
    blueprint["export_data"] = bool(case.get("settings", {}).get("export_data", False))
    if case_type in {"basement", "hecras"}:
        blueprint["xdmf_path"] = case.get("xdmf_path", "")
    return blueprint
