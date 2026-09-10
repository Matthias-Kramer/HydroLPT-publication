from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core._02_particle_initialisation.properties import (
    beta_p_from_shape,
    kN_from_shape,
)
from core._04_transport_solver.settling import wsettling


@dataclass(slots=True)
class BiofoulingConfig:
    """Simple constant-rate biofilm growth model configuration."""

    enabled: bool = False
    BT0: float = 0.0
    BR: float = 0.0
    rho_biofilm: float = 1388.0


def biofouling_config_from_settings(settings: dict) -> BiofoulingConfig:
    """Build the biofouling configuration from flattened case settings."""
    raw_toggle = str(settings.get("biofouling", "off")).strip().lower()
    return BiofoulingConfig(
        enabled=raw_toggle == "on",
        BT0=float(settings.get("BT0", 0.0) or 0.0),
        BR=float(settings.get("BR", 0.0) or 0.0),
        rho_biofilm=float(settings.get("rho_biofilm", 1388.0) or 1388.0),
    )


def biofilm_thickness_from_age(age_s: np.ndarray | float, config: BiofoulingConfig) -> np.ndarray:
    """Return biofilm thickness [m] from particle age [s]."""
    age = np.asarray(age_s, dtype=float)
    if not config.enabled:
        return np.zeros_like(age, dtype=float)
    return np.maximum(float(config.BT0) + float(config.BR) * np.maximum(age, 0.0), 0.0)


def _coated_dimensions(
    shape: str,
    *,
    L: float,
    I: float,
    S: float,
    bt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coated dimensions using simple shape-specific thickness assumptions."""
    shape = str(shape).strip().lower()
    bt = np.asarray(bt, dtype=float)

    if shape == "sphere":
        D = float(L) + 2.0 * bt
        return D, D, D
    if shape == "cylinder":
        D = float(S) + 2.0 * bt
        Lc = float(L) + 2.0 * bt
        return Lc, D, D
    if shape == "disk":
        D = float(L) + 2.0 * bt
        T = float(S) + 2.0 * bt
        return D, D, T
    if shape == "ellipsoid":
        return float(L) + 2.0 * bt, float(I) + 2.0 * bt, float(S) + 2.0 * bt
    if shape == "prism":
        return float(L) + 2.0 * bt, float(I) + 2.0 * bt, float(S) + 2.0 * bt
    raise ValueError(f"Unsupported particle shape for biofouling evolution: {shape!r}")


def degraded_dimensions(
    shape: str,
    *,
    L: float,
    I: float,
    S: float,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return degraded dimensions after isotropic size shrinkage."""
    shape = str(shape).strip().lower()
    scale = np.asarray(scale, dtype=float)
    if shape == "sphere":
        D = float(L) * scale
        return D, D, D
    if shape == "cylinder":
        return float(L) * scale, float(S) * scale, float(S) * scale
    if shape == "disk":
        D = float(L) * scale
        T = float(S) * scale
        return D, D, T
    if shape in {"ellipsoid", "prism"}:
        return float(L) * scale, float(I) * scale, float(S) * scale
    raise ValueError(f"Unsupported particle shape for degradation evolution: {shape!r}")


def _volume_and_projected_area(
    shape: str,
    *,
    L: np.ndarray,
    I: np.ndarray,
    S: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return coated particle volume and maximum projected area."""
    shape = str(shape).strip().lower()

    if shape == "sphere":
        Vp = np.pi * L**3 / 6.0
        Aplus = np.pi * L**2 / 4.0
        return Vp, Aplus
    if shape == "ellipsoid":
        Vp = np.pi * L * I * S / 6.0
        Aplus = np.pi * L * I / 4.0
        return Vp, Aplus
    if shape == "cylinder":
        Vp = np.pi * S**2 * L / 4.0
        Aplus = L * S
        return Vp, Aplus
    if shape == "disk":
        Vp = np.pi * L**2 * S / 4.0
        Aplus = np.pi * L**2 / 4.0
        return Vp, Aplus
    if shape == "prism":
        Vp = L * I * S
        Aplus = L * I
        return Vp, Aplus
    raise ValueError(f"Unsupported particle shape for biofouling evolution: {shape!r}")


def compute_biofouled_state(
    *,
    shape: str,
    rho_p0: float,
    L0: float,
    I0: float,
    S0: float,
    rho_f: float,
    nu: float,
    g: float,
    bt: np.ndarray,
    rho_biofilm: float,
) -> dict[str, np.ndarray]:
    """Return evolved density and settling properties for one-shape particles."""
    bt = np.asarray(bt, dtype=float)
    Lc, Ic, Sc = _coated_dimensions(shape, L=L0, I=I0, S=S0, bt=bt)
    Vp0, _ = _volume_and_projected_area(
        shape,
        L=np.full_like(bt, float(L0), dtype=float),
        I=np.full_like(bt, float(I0), dtype=float),
        S=np.full_like(bt, float(S0), dtype=float),
    )
    Vp, Aplus = _volume_and_projected_area(shape, L=Lc, I=Ic, S=Sc)

    shell_volume = np.maximum(Vp - Vp0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_p = (float(rho_p0) * Vp0 + float(rho_biofilm) * shell_volume) / np.maximum(Vp, 1.0e-30)

    DV = (6.0 * Vp / np.pi) ** (1.0 / 3.0)
    DA = np.sqrt(4.0 * Aplus / np.pi)
    beta_p = np.asarray(beta_p_from_shape(shape, Lc, Sc, Vp), dtype=float)
    kN = np.asarray(kN_from_shape(shape, DV, DA), dtype=float)
    ws = np.asarray(
        wsettling(DV, DA, rho_p=rho_p, rho_f=rho_f, nu=nu, g=g, kN=kN),
        dtype=float,
    )

    return {
        "L": Lc,
        "I": Ic,
        "S": Sc,
        "Vp": Vp,
        "Aplus": Aplus,
        "DV": DV,
        "DA": DA,
        "Dp": DV,
        "beta_p": beta_p,
        "kN": kN,
        "rho_p": rho_p,
        "ws": ws,
    }


def compute_degraded_state(
    *,
    shape: str,
    rho_p0: float,
    L0: float,
    I0: float,
    S0: float,
    rho_f: float,
    nu: float,
    g: float,
    scale: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return degraded geometry and settling properties with constant density."""
    scale = np.asarray(scale, dtype=float)
    Lc, Ic, Sc = degraded_dimensions(shape, L=L0, I=I0, S=S0, scale=scale)
    Vp, Aplus = _volume_and_projected_area(shape, L=Lc, I=Ic, S=Sc)
    DV = (6.0 * Vp / np.pi) ** (1.0 / 3.0)
    DA = np.sqrt(4.0 * Aplus / np.pi)
    beta_p = np.asarray(beta_p_from_shape(shape, Lc, Sc, Vp), dtype=float)
    kN = np.asarray(kN_from_shape(shape, DV, DA), dtype=float)
    rho_p = np.full_like(scale, float(rho_p0), dtype=float)
    ws = np.asarray(
        wsettling(DV, DA, rho_p=rho_p, rho_f=rho_f, nu=nu, g=g, kN=kN),
        dtype=float,
    )

    return {
        "L": Lc,
        "I": Ic,
        "S": Sc,
        "Vp": Vp,
        "Aplus": Aplus,
        "DV": DV,
        "DA": DA,
        "Dp": DV,
        "beta_p": beta_p,
        "kN": kN,
        "rho_p": rho_p,
        "ws": ws,
    }
