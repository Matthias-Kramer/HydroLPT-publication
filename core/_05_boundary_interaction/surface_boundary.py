from __future__ import annotations

import numpy as np

from core._05_boundary_interaction.bed_boundary import sample_entrainment


SURFACE_POLICIES = (
    "always_stick",
    "always_reflect",
    "deterministic_detachment",
    "probabilistic_detachment",
)


def _ellipse_perimeter(a: float, b: float) -> float:
    """Approximate ellipse perimeter using Ramanujan's second formula."""
    a = float(max(a, 0.0))
    b = float(max(b, 0.0))
    if a <= 0.0 or b <= 0.0:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return float(np.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h))))


def _critical_detachment_index(
    v_sub: np.ndarray,
    l_sigma: np.ndarray,
    *,
    rho_f: float,
    g: float,
    sigma: float,
    contact_angle_deg: float,
) -> int:
    """Select the immersion state that maximizes the resisting force at detachment."""
    sin_omega = np.sin(np.deg2rad(float(contact_angle_deg)))
    force_resisting = float(rho_f) * float(g) * v_sub + float(sigma) * sin_omega * l_sigma
    return int(np.argmax(force_resisting))


def _sphere_detachment_geometry(
    diameter: float,
    *,
    rho_f: float,
    g: float,
    sigma: float,
    contact_angle_deg: float,
    n_samples: int = 1024,
) -> tuple[float, float]:
    d = float(diameter)
    if d <= 0.0:
        return np.nan, np.nan
    r = 0.5 * d
    h = np.linspace(0.5 * d, d, max(int(n_samples), 16))
    v_sub = (np.pi * h * h * (3.0 * r - h)) / 3.0
    l_sigma = 2.0 * np.pi * np.sqrt(np.maximum(0.0, 2.0 * h * r - h * h))
    idx = _critical_detachment_index(
        v_sub,
        l_sigma,
        rho_f=rho_f,
        g=g,
        sigma=sigma,
        contact_angle_deg=contact_angle_deg,
    )
    return float(v_sub[idx]), float(l_sigma[idx])


def _ellipsoid_detachment_geometry(
    long_axis: float,
    intermediate_axis: float,
    short_axis: float,
    *,
    rho_f: float,
    g: float,
    sigma: float,
    contact_angle_deg: float,
    n_samples: int = 1024,
) -> tuple[float, float]:
    a = 0.5 * float(long_axis)
    b = 0.5 * float(intermediate_axis)
    c = 0.5 * float(short_axis)
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        return np.nan, np.nan
    h = np.linspace(c, 2.0 * c, max(int(n_samples), 16))
    t = (h / c) - 1.0
    v_sub = np.pi * a * b * c * (t - (t**3) / 3.0 + 2.0 / 3.0)
    scale = np.sqrt(np.maximum(0.0, 1.0 - t * t))
    l_sigma = scale * _ellipse_perimeter(a, b)
    idx = _critical_detachment_index(
        v_sub,
        l_sigma,
        rho_f=rho_f,
        g=g,
        sigma=sigma,
        contact_angle_deg=contact_angle_deg,
    )
    return float(v_sub[idx]), float(l_sigma[idx])


def _cylinder_detachment_geometry(
    length: float,
    diameter: float,
    *,
    rho_f: float,
    g: float,
    sigma: float,
    contact_angle_deg: float,
    n_samples: int = 1024,
) -> tuple[float, float]:
    length = float(length)
    diameter = float(diameter)
    if length <= 0.0 or diameter <= 0.0:
        return np.nan, np.nan
    r = 0.5 * diameter
    h = np.linspace(r, 2.0 * r, max(int(n_samples), 16))
    y = np.sqrt(np.maximum(0.0, 2.0 * h * r - h * h))
    area_sub = (r * r * np.arccos(np.clip((r - h) / r, -1.0, 1.0))) - ((r - h) * y)
    v_sub = length * area_sub
    l_sigma = 2.0 * length + 4.0 * y
    idx = _critical_detachment_index(
        v_sub,
        l_sigma,
        rho_f=rho_f,
        g=g,
        sigma=sigma,
        contact_angle_deg=contact_angle_deg,
    )
    return float(v_sub[idx]), float(l_sigma[idx])


def _disk_detachment_geometry(
    diameter: float,
    thickness: float,
    *,
    rho_f: float,
    g: float,
    sigma: float,
    contact_angle_deg: float,
    n_samples: int = 1024,
) -> tuple[float, float]:
    diameter = float(diameter)
    thickness = float(thickness)
    if diameter <= 0.0 or thickness <= 0.0:
        return np.nan, np.nan
    area_face = np.pi * diameter * diameter / 4.0
    h = np.linspace(0.5 * thickness, thickness, max(int(n_samples), 16))
    v_sub = area_face * h
    l_sigma = np.full_like(h, np.pi * diameter)
    idx = _critical_detachment_index(
        v_sub,
        l_sigma,
        rho_f=rho_f,
        g=g,
        sigma=sigma,
        contact_angle_deg=contact_angle_deg,
    )
    return float(v_sub[idx]), float(l_sigma[idx])


def _prism_detachment_geometry(
    long_axis: float,
    intermediate_axis: float,
    short_axis: float,
    *,
    rho_f: float,
    g: float,
    sigma: float,
    contact_angle_deg: float,
    n_samples: int = 1024,
) -> tuple[float, float]:
    length = float(long_axis)
    intermediate = float(intermediate_axis)
    short = float(short_axis)
    if length <= 0.0 or intermediate <= 0.0 or short <= 0.0:
        return np.nan, np.nan
    area_face = length * intermediate
    h = np.linspace(0.5 * short, short, max(int(n_samples), 16))
    v_sub = area_face * h
    l_sigma = np.full_like(h, 2.0 * (length + intermediate))
    idx = _critical_detachment_index(
        v_sub,
        l_sigma,
        rho_f=rho_f,
        g=g,
        sigma=sigma,
        contact_angle_deg=contact_angle_deg,
    )
    return float(v_sub[idx]), float(l_sigma[idx])


def projected_area_surface_detachment_from_shape(shape: str, L: float, I: float, S: float) -> float:
    """Return the projected area used by free-surface detachment drag."""
    shape = str(shape).strip().lower()
    L = float(L)
    I = float(I)
    S = float(S)
    if L <= 0.0 or I <= 0.0 or S <= 0.0:
        return np.nan
    if shape == "sphere":
        return float(np.pi * L * L / 4.0)
    if shape == "ellipsoid":
        return float(np.pi * L * I / 4.0)
    if shape == "cylinder":
        return float(L * S)
    if shape == "disk":
        return float(np.pi * L * L / 4.0)
    if shape == "prism":
        return float(L * I)
    raise ValueError(f"Unsupported particle shape for surface detachment: {shape!r}")


def surface_detachment_geometry(
    shape: str,
    L: float,
    I: float,
    S: float,
    Vp: float,
    *,
    rho_f: float,
    g: float,
    sigma: float,
    contact_angle_deg: float,
) -> tuple[float, float, float]:
    """Return ``(Vpw_crit, Lsigma_crit, Aproj_surface)`` for surface detachment."""
    shape = str(shape).strip().lower()
    L = float(L)
    I = float(I)
    S = float(S)
    _ = float(Vp)
    aproj_surface = projected_area_surface_detachment_from_shape(shape, L, I, S)
    if shape == "sphere":
        vpw_crit, lsigma_crit = _sphere_detachment_geometry(
            L, rho_f=rho_f, g=g, sigma=sigma, contact_angle_deg=contact_angle_deg
        )
        return vpw_crit, lsigma_crit, aproj_surface
    if shape == "disk":
        vpw_crit, lsigma_crit = _disk_detachment_geometry(
            L, S, rho_f=rho_f, g=g, sigma=sigma, contact_angle_deg=contact_angle_deg
        )
        return vpw_crit, float(lsigma_crit), aproj_surface
    if shape == "ellipsoid":
        vpw_crit, lsigma_crit = _ellipsoid_detachment_geometry(
            L, I, S, rho_f=rho_f, g=g, sigma=sigma, contact_angle_deg=contact_angle_deg
        )
        return vpw_crit, float(lsigma_crit), aproj_surface
    if shape == "prism":
        vpw_crit, lsigma_crit = _prism_detachment_geometry(
            L, I, S, rho_f=rho_f, g=g, sigma=sigma, contact_angle_deg=contact_angle_deg
        )
        return vpw_crit, float(lsigma_crit), aproj_surface
    if shape == "cylinder":
        vpw_crit, lsigma_crit = _cylinder_detachment_geometry(
            L, S, rho_f=rho_f, g=g, sigma=sigma, contact_angle_deg=contact_angle_deg
        )
        return vpw_crit, float(lsigma_crit), aproj_surface
    raise ValueError(f"Unsupported particle shape for surface detachment: {shape!r}")


def ustarcrit_surface_from_particle(
    *,
    shape: str,
    rho_p: float,
    L: float,
    I: float,
    S: float,
    Vp: float,
    rho_f: float = 1000.0,
    g: float = 9.81,
    sigma: float = 0.072,
    contact_angle_deg: float = 105.0,
    drag_coeff_vertical: float = 2.0,
) -> float:
    """Return the Kramer-style critical shear velocity for surface detachment."""
    vpw_crit, lsigma_crit, aproj_fw = surface_detachment_geometry(
        shape, L, I, S, Vp, rho_f=rho_f, g=g, sigma=sigma, contact_angle_deg=contact_angle_deg
    )
    if (
        not np.isfinite(vpw_crit)
        or not np.isfinite(lsigma_crit)
        or not np.isfinite(aproj_fw)
        or vpw_crit <= 0.0
        or aproj_fw <= 0.0
        or drag_coeff_vertical <= 0.0
    ):
        return np.nan
    sin_omega = np.sin(np.deg2rad(float(contact_angle_deg)))
    numerator = rho_f * g * vpw_crit - rho_p * g * Vp + lsigma_crit * sigma * sin_omega
    denominator = 0.5 * rho_f * drag_coeff_vertical * aproj_fw
    if numerator <= 0.0 or denominator <= 0.0:
        return np.nan
    return float((1.0 / 0.467) * np.sqrt(numerator / denominator))


def resolve_surface_detachment(
    *,
    surface_policy: str,
    tid: np.ndarray,
    wet_cell: np.ndarray,
    can_detach_surface: np.ndarray,
    ustar_c: np.ndarray,
    surface_ustar_crit_c: np.ndarray,
    rng: np.random.Generator,
    entrainment_prob_mode: str,
    entrainment_prob_sigma_star: float = 0.2,
    entrainment_prob_k: float | None = None,
) -> np.ndarray:
    """Return a detach mask for surfaced particles in the provided cells."""
    tid = np.asarray(tid, dtype=int).ravel()
    if tid.size == 0:
        return np.zeros(0, dtype=bool)
    if surface_policy == "always_stick":
        return np.zeros(tid.size, dtype=bool)
    if surface_policy == "always_reflect":
        return np.ones(tid.size, dtype=bool)
    if surface_policy == "deterministic_detachment":
        can_detach_arr = np.asarray(can_detach_surface, dtype=bool).ravel()
        if can_detach_arr.size == tid.size:
            return can_detach_arr
        return np.asarray(can_detach_arr[tid], dtype=bool)
    if surface_policy != "probabilistic_detachment":
        raise ValueError(
            'surfacePolicy must be '
            '"always_stick", "always_reflect", '
            '"deterministic_detachment", or "probabilistic_detachment".'
        )
    valid = np.asarray(wet_cell[tid], dtype=bool)
    detach = np.zeros(tid.size, dtype=bool)
    if np.any(valid):
        surface_crit_arr = np.asarray(surface_ustar_crit_c, dtype=float).ravel()
        if surface_crit_arr.size == tid.size:
            surface_crit_valid = surface_crit_arr[valid]
        else:
            surface_crit_valid = surface_crit_arr[tid[valid]]
        detach_valid, _ = sample_entrainment(
            ustar=ustar_c[tid[valid]],
            ustar_crit=surface_crit_valid,
            rng=rng,
            sigma_star=entrainment_prob_sigma_star,
            mode=entrainment_prob_mode,
            k=entrainment_prob_k,
        )
        detach[valid] = detach_valid
    return detach


def resolve_surface_contact(
    *,
    surface_policy: str,
    tid: np.ndarray,
    wet_cell: np.ndarray,
    can_detach_surface: np.ndarray,
    ustar_c: np.ndarray,
    surface_ustar_crit_c: np.ndarray,
    rng: np.random.Generator,
    entrainment_prob_mode: str,
    entrainment_prob_sigma_star: float = 0.2,
    entrainment_prob_k: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reflect and stick masks for particles that hit the free surface."""
    reflect = resolve_surface_detachment(
        surface_policy=surface_policy,
        tid=tid,
        wet_cell=wet_cell,
        can_detach_surface=can_detach_surface,
        ustar_c=ustar_c,
        surface_ustar_crit_c=surface_ustar_crit_c,
        rng=rng,
        entrainment_prob_mode=entrainment_prob_mode,
        entrainment_prob_sigma_star=entrainment_prob_sigma_star,
        entrainment_prob_k=entrainment_prob_k,
    )
    return reflect, ~reflect
