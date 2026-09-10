from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DegradationConfig:
    """Simple isotropic size-degradation model configuration."""

    enabled: bool = False
    DR: float = 0.0
    min_scale: float = 1.0e-6


def degradation_config_from_settings(settings: dict) -> DegradationConfig:
    """Build the degradation configuration from flattened case settings."""
    raw_toggle = str(settings.get("degradation", "off")).strip().lower()
    return DegradationConfig(
        enabled=raw_toggle == "on",
        DR=float(settings.get("DR", 0.0) or 0.0),
    )


def degradation_scale_from_age(age_s: np.ndarray | float, config: DegradationConfig) -> np.ndarray:
    """Return the isotropic size-reduction factor from particle age [s]."""
    age = np.asarray(age_s, dtype=float)
    if not config.enabled:
        return np.ones_like(age, dtype=float)
    scale = 1.0 - float(config.DR) * np.maximum(age, 0.0) / 100.0
    return np.maximum(scale, float(config.min_scale))
