"""Particle-evolution helpers such as simple biofouling models."""

from core._03_particle_evolution.biofouling import (
    BiofoulingConfig,
    biofilm_thickness_from_age,
    biofouling_config_from_settings,
    compute_biofouled_state,
    compute_degraded_state,
)
from core._03_particle_evolution.degradation import (
    DegradationConfig,
    degradation_config_from_settings,
    degradation_scale_from_age,
)

__all__ = [
    "BiofoulingConfig",
    "DegradationConfig",
    "biofilm_thickness_from_age",
    "biofouling_config_from_settings",
    "compute_biofouled_state",
    "compute_degraded_state",
    "degradation_config_from_settings",
    "degradation_scale_from_age",
]
