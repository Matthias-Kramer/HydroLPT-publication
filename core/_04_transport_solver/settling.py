"""Settling-velocity relations for HydroLPT particles."""

from __future__ import annotations

import numpy as np


def wsettling(
    DV,
    DA,
    rho_p: float,
    rho_f: float = 1000.0,
    nu: float = 1e-6,
    g: float = 9.81,
    kN=1.0,
) -> np.ndarray:
    """Return terminal settling velocity ``ws`` [m/s].

    Positive values indicate gravitational settling (`rho_p > rho_f`).
    Negative values indicate buoyant rise (`rho_p < rho_f`).

    Parameters
    ----------
    DV
        Volume-equivalent particle diameter [m].
    DA
        Area-equivalent particle diameter [m].
    rho_p
        Particle density [kg/m^3].
    rho_f
        Fluid density [kg/m^3].
    nu
        Fluid kinematic viscosity [m^2/s].
    g
        Gravitational acceleration [m/s^2].
    kN
        Newton drag correction factor.

    Returns
    -------
    np.ndarray
        Settling velocity [m/s].
    """
    dv = np.asarray(DV, dtype=float)
    da = np.asarray(DA, dtype=float)
    drag_correction = np.asarray(kN, dtype=float)

    density_ratio = rho_p / rho_f
    density_contrast = density_ratio - 1.0
    settling_sign = np.sign(density_contrast)

    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = 18.0 * nu + np.sqrt(
            (3.0 / 4.0)
            * drag_correction
            * 0.46
            * abs(density_contrast)
            * g
            * dv**3
        )
        settling_velocity = (
            settling_sign
            * g
            * dv**2
            * abs(density_contrast)
            / denominator
            * (dv / da)
        )

    return settling_velocity

