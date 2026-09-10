from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core._04_transport_solver.settling import wsettling


@dataclass
class ParticleProps:
    """Derived particle properties used by HydroLPT transport physics."""

    shape: str
    rho_p: float
    L: float
    I: float
    S: float
    Vp: float
    Aplus: float
    DV: float
    DA: float
    Dp: float
    beta_p: float
    kN: float
    ws: float


def _scalar(x) -> float:
    """Convert scalar-like input to Python float."""
    return float(np.asarray(x).ravel()[0])


def complete_particle_dimensions(
    shape: str,
    L: float | None,
    I: float | None = None,
    S: float | None = None,
) -> tuple[float, float, float]:
    """Expand shape-specific independent dimensions into a full (L, I, S) tuple."""
    shape = str(shape).strip().lower()

    if L is None:
        raise ValueError("Particle dimension L must be provided.")
    L = float(L)

    if shape == "sphere":
        return L, L, L

    if shape == "cylinder":
        if S is None and I is None:
            raise ValueError("Cylinder particles require S (diameter) or I.")
        diameter = float(S if S is not None else I)
        return L, diameter, diameter

    if shape == "disk":
        if S is None:
            raise ValueError("Disk particles require S (thickness).")
        thickness = float(S)
        return L, L, thickness

    if shape in ("ellipsoid", "prism"):
        if I is None or S is None:
            raise ValueError(f"{shape.capitalize()} particles require both I and S.")
        return L, float(I), float(S)

    raise ValueError(f"Unknown shape '{shape}'")


def Vp_Aplus_from_shape(shape: str, L: float, I: float, S: float):
    """Compute particle volume and maximum projected area from shape dimensions."""
    shape = shape.lower()

    if shape == "sphere":
        if not (np.isclose(L, I) and np.isclose(I, S)):
            raise ValueError("For a sphere, L = I = S (diameter).")
        D = float(L)
        Vp = np.pi * D**3 / 6.0
        Aplus = np.pi * D**2 / 4.0
        return Vp, Aplus

    if not (L >= I >= S):
        raise ValueError(f"Dimensions must satisfy L >= I >= S (got L={L}, I={I}, S={S}).")

    L = float(L)
    I = float(I)
    S = float(S)

    if shape == "ellipsoid":
        Vp = np.pi * L * I * S / 6.0
        Aplus = np.pi * L * I / 4.0
    elif shape == "cylinder":
        D = S
        Vp = np.pi * D**2 * L / 4.0
        Aplus = L * D
    elif shape == "disk":
        D = L
        Vp = np.pi * D**2 * S / 4.0
        Aplus = np.pi * D**2 / 4.0
    elif shape == "prism":
        Vp = L * I * S
        Aplus = L * I
    else:
        raise ValueError(f"Unknown shape '{shape}'")

    return Vp, Aplus


def DV_from_Vp(Vp):
    """Volume-equivalent diameter DV [m]."""
    Vp = np.asarray(Vp, dtype=float)
    return (6.0 * Vp / np.pi) ** (1.0 / 3.0)


def DA_from_Aplus(Aplus):
    """Area-equivalent diameter DA [m]."""
    Aplus = np.asarray(Aplus, dtype=float)
    return np.sqrt(4.0 * Aplus / np.pi)


def beta_p_from_shape(shape: str, L, S, Vp):
    """Compute beta_p from projected area and volume-equivalent diameter."""
    shape = shape.lower()
    L = np.asarray(L, dtype=float)
    S = np.asarray(S, dtype=float)
    Vp = np.asarray(Vp, dtype=float)

    DV = DV_from_Vp(Vp)
    Aproj = np.pi * L**2 / 4.0 if shape == "sphere" else L * S

    beta = np.full_like(DV, np.nan, dtype=float)
    ok = (Vp > 0) & np.isfinite(Vp) & np.isfinite(Aproj)
    beta[ok] = (Aproj[ok] * DV[ok]) / Vp[ok]
    return beta


def kN_from_shape(shape: str, DV, DA):
    """Return Newton correction factor kN from particle shape."""
    shape = shape.lower()
    DV = np.asarray(DV, dtype=float)
    DA = np.asarray(DA, dtype=float)

    if shape in ("ellipsoid", "prism"):
        return np.full_like(DV, 1.7, dtype=float)
    if shape in ("cylinder", "disk"):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (DV / DA) ** (-1.4)
    if shape == "sphere":
        return np.ones_like(DV, dtype=float)
    raise ValueError(f"Unknown shape '{shape}'")


def build_particle_properties(
    shape: str,
    rho_p: float,
    L: float,
    I: float | None,
    S: float | None,
    rho_f: float = 1000.0,
    nu: float = 1e-6,
    g: float = 9.81,
) -> ParticleProps:
    """Build derived particle properties for settling and entrainment models."""
    L, I, S = complete_particle_dimensions(shape, L, I, S)
    Vp, Aplus = Vp_Aplus_from_shape(shape, L, I, S)

    DV = _scalar(DV_from_Vp(Vp))
    DA = _scalar(DA_from_Aplus(Aplus))
    Dp = DV
    beta_p = _scalar(beta_p_from_shape(shape, L, S, Vp))
    kN = _scalar(kN_from_shape(shape, DV, DA))
    ws = _scalar(wsettling(DV, DA, rho_p=rho_p, rho_f=rho_f, nu=nu, g=g, kN=kN))

    return ParticleProps(
        shape=shape,
        rho_p=rho_p,
        L=L,
        I=I,
        S=S,
        Vp=_scalar(Vp),
        Aplus=_scalar(Aplus),
        DV=DV,
        DA=DA,
        Dp=Dp,
        beta_p=beta_p,
        kN=kN,
        ws=ws,
    )


def print_particle_info(p: ParticleProps):
    """Print a compact summary of derived particle properties."""
    print("\nParticle (derived):")
    print(f"  shape        : {p.shape}")
    print(f"  rho_p        : {p.rho_p:.0f} kg/m^3")
    print(f"  L, I, S      : {p.L*1e3:.2f}, {p.I*1e3:.2f}, {p.S*1e3:.2f} mm")
    print(f"  DV (vol-eq)  : {p.DV*1e3:.2f} mm")
    print(f"  DA (area-eq) : {p.DA*1e3:.2f} mm")
    print(f"  beta_p       : {p.beta_p:.3f} [-]")
    print(f"  kN           : {p.kN:.3f} [-]")
    print(f"  ws           : {p.ws:.4f} m/s  ({p.ws*100:.2f} cm/s)")
