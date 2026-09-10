"""Typed schema models for HydroLPT case definitions."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(slots=True)
class HydraulicInput:
    length: Any = None
    domain_width: Any = None
    wetted_width: Any = None
    water_depth: Any = None
    run_mode: Any = None
    hydraulicPrimaryVariable: Any = None
    U: Any = None
    V: Any = None
    hydraulicClosureModel: Any = None
    z0: Any = None
    transient_shape: Any = None
    switch_fraction: Any = None
    U0: Any = None
    V0: Any = None
    U1: Any = None
    V1: Any = None
    hyd_time_start: Any = None
    hyd_dt: Any = None
    hyd_time_end: Any = None
    rho_f: Any = None
    nu: Any = None
    g: Any = None
    nx: Any = None
    ny: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParticleSettings:
    shape: Any = None
    L: Any = None
    I: Any = None
    S: Any = None
    rho_p: Any = None
    release_mode: Any = None
    release_dt: Any = None
    location_mode: Any = None
    release_x: Any = None
    release_y: Any = None
    manual_centers: Any = None
    n_per_center: Any = None
    nTrack: Any = None
    releaseSigma: Any = None
    zFrac: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParticleEvolutionSettings:
    biofouling: Any = None
    BT0: Any = None
    BR: Any = None
    rho_biofilm: Any = None
    degradation: Any = None
    DR: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TransportSettings:
    tRelease: Any = None
    dt: Any = None
    tTrack: Any = None
    hydraulicFieldMode: Any = None
    transportVelocityMode: Any = None
    transportModel: Any = None
    rng_seed: Any = None
    useRWx: Any = None
    useRWy: Any = None
    useRWz: Any = None
    betaKh: Any = None
    KhMax: Any = None
    rAniso: Any = None
    alphaKz: Any = None
    KzMax: Any = None
    TL_horizontal: Any = None
    TL_vertical: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BoundarySettings:
    surfacePolicy: Any = None
    bedPolicy: Any = None
    surfaceVerticalDragCoeff: Any = None
    surfaceContactAngleDeg: Any = None
    surfaceDetachmentSigmaStar: Any = None
    tanphi_ratio: Any = None
    bedEntrainmentSigmaStar: Any = None
    dryPolicy: Any = None
    hmin: Any = None
    outsidePolicy: Any = None
    uphillPolicy: Any = None
    dzUpMax: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OutputSettings:
    export_data: bool = False
    output_dt: Any = None
    binary_map: Any = None
    trajectories: Any = None
    binary_map_marker_size: Any = None
    binary_map_mesh_overlay: Any = None
    advection_diffusion: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CaseSchema:
    case_type: str
    hydraulic_input: HydraulicInput = field(default_factory=HydraulicInput)
    particle_settings: ParticleSettings = field(default_factory=ParticleSettings)
    particle_evo_settings: ParticleEvolutionSettings = field(default_factory=ParticleEvolutionSettings)
    transport_settings: TransportSettings = field(default_factory=TransportSettings)
    boundary_settings: BoundarySettings = field(default_factory=BoundarySettings)
    output_settings: OutputSettings = field(default_factory=OutputSettings)
    xdmf_path: str = ""
    user_notes: str = ""


def model_from_dict(model_type: type, values: dict[str, Any]) -> Any:
    """Build a section model from a dict, preserving unknown keys in extras."""
    values = dict(values)
    known_names = {item.name for item in fields(model_type) if item.name != "extras"}
    kwargs = {name: values.pop(name, None) for name in known_names}
    kwargs["extras"] = values
    return model_type(**kwargs)


def model_to_dict(model: Any) -> dict[str, Any]:
    """Convert a section model back into a plain dict including extras."""
    data: dict[str, Any] = {}
    extras = getattr(model, "extras", {})
    for item in fields(model):
        if item.name == "extras":
            continue
        value = getattr(model, item.name)
        if value is not None:
            data[item.name] = value
    data.update(dict(extras))
    return data


def case_schema_from_payload(
    *,
    case_type: str,
    sections: dict[str, dict[str, Any]],
    plots: dict[str, Any],
    export_data: bool,
    xdmf_path: str = "",
    user_notes: str = "",
) -> CaseSchema:
    """Build a typed schema from the current dict-based payload."""
    output_settings = model_from_dict(
        OutputSettings,
        {"export_data": bool(export_data), **dict(plots)},
    )
    return CaseSchema(
        case_type=case_type,
        hydraulic_input=model_from_dict(HydraulicInput, sections.get("HYDRAULIC_INPUT", {})),
        particle_settings=model_from_dict(ParticleSettings, sections.get("PARTICLE_SETTINGS", {})),
        particle_evo_settings=model_from_dict(ParticleEvolutionSettings, sections.get("PARTICLE_EVO_SETTINGS", {})),
        transport_settings=model_from_dict(TransportSettings, sections.get("TRANSPORT_SETTINGS", {})),
        boundary_settings=model_from_dict(BoundarySettings, sections.get("BOUNDARY_SETTINGS", {})),
        output_settings=output_settings,
        xdmf_path=xdmf_path,
        user_notes=user_notes,
    )


def case_schema_to_payload(schema: CaseSchema) -> dict[str, Any]:
    """Convert the typed schema back to the dict payload used by the app."""
    output_dict = model_to_dict(schema.output_settings)
    export_data = bool(output_dict.pop("export_data", False))
    return {
        "case_type": schema.case_type,
        "export_data": export_data,
        "sections": {
            "HYDRAULIC_INPUT": model_to_dict(schema.hydraulic_input),
            "PARTICLE_SETTINGS": model_to_dict(schema.particle_settings),
            "PARTICLE_EVO_SETTINGS": model_to_dict(schema.particle_evo_settings),
            "TRANSPORT_SETTINGS": model_to_dict(schema.transport_settings),
            "BOUNDARY_SETTINGS": model_to_dict(schema.boundary_settings),
        },
        "plots": output_dict,
        "xdmf_path": schema.xdmf_path,
        "user_notes": schema.user_notes,
    }
