"""Case configuration, discovery, normalization, and export helpers."""

from core._00_case_management.blueprints import (
    build_blueprint_from_case,
    default_case_blueprints,
)
from core._00_case_management.constants import GUI_CASE_FILENAME
from core._00_case_management.exporter import generate_case_files, save_case_file
from core._00_case_management.loading import (
    apply_overrides_to_module,
    discover_run_scripts,
    load_case_definition,
)
from core._00_case_management.models import (
    BoundarySettings,
    CaseSchema,
    HydraulicInput,
    OutputSettings,
    ParticleEvolutionSettings,
    ParticleSettings,
    TransportSettings,
    case_schema_from_payload,
    case_schema_to_payload,
)
from core._00_case_management.normalize import (
    flatten_sections,
    normalize_case_spec,
    read_preset,
    safe_run_name,
    write_preset,
)

__all__ = [
    "GUI_CASE_FILENAME",
    "BoundarySettings",
    "CaseSchema",
    "HydraulicInput",
    "OutputSettings",
    "ParticleEvolutionSettings",
    "ParticleSettings",
    "TransportSettings",
    "apply_overrides_to_module",
    "build_blueprint_from_case",
    "case_schema_from_payload",
    "case_schema_to_payload",
    "default_case_blueprints",
    "discover_run_scripts",
    "flatten_sections",
    "generate_case_files",
    "load_case_definition",
    "normalize_case_spec",
    "read_preset",
    "safe_run_name",
    "save_case_file",
    "write_preset",
]
