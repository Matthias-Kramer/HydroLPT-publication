from __future__ import annotations

import numpy as np
from scipy.special import erf


BED_POLICIES = (
    "always_deposit",
    "always_reflect",
    "reflect_if_ustar_gt_crit",
    "probabilistic_entrainment",
)


def theta_cr_sui(Re_star):
    """Return the critical Shields parameter from a Sui-type relation."""
    Re_star = np.asarray(Re_star, dtype=float)
    return (
        0.165 * (Re_star + 0.6) ** (-0.8)
        + 0.045 * np.exp(-40.0 * Re_star ** (-1.3))
    )


def ustarcrit_from_eq20(
    Dp,
    d50,
    rho_p,
    beta_p,
    tanphi_ratio,
    rho_w=1000.0,
    nu=1e-6,
    g=9.81,
    beta_s=1.5,
    c2=-0.6,
    u0=0.005,
    n_iter=30,
    tol=1e-8,
):
    """Solve critical shear velocity ``u*_crit`` from the particle-force balance."""
    if rho_p <= rho_w or d50 <= 0 or Dp <= 0 or beta_p <= 0:
        return np.nan

    u = float(u0)
    for _ in range(n_iter):
        re_star = u * d50 / nu
        theta_s = theta_cr_sui(re_star)
        theta_p = theta_s * (beta_s / beta_p) * tanphi_ratio * (Dp / d50) ** c2
        u_new = np.sqrt(theta_p * ((rho_p - rho_w) / rho_w) * g * Dp)
        if not np.isfinite(u_new):
            return np.nan
        if abs(u_new - u) < tol * max(1.0, abs(u)):
            return float(u_new)
        u = u_new
    return float(u)


def ustarcrit_cells_from_d50(
    d50_c,
    Dp,
    rho_p,
    beta_p,
    tanphi_ratio,
    rho_w=1000.0,
    nu=1e-6,
    g=9.81,
    beta_s=1.5,
    c2=-0.6,
    u0=0.005,
    n_iter=30,
    tol=1e-8,
):
    """Compute cellwise critical shear velocity from cellwise representative d50."""
    d50_c = np.asarray(d50_c, dtype=float).ravel()
    ustarcrit_c = np.full_like(d50_c, np.nan, dtype=float)
    valid = np.isfinite(d50_c) & (d50_c > 0)
    for i in np.flatnonzero(valid):
        ustarcrit_c[i] = ustarcrit_from_eq20(
            Dp=Dp,
            d50=d50_c[i],
            rho_p=rho_p,
            beta_p=beta_p,
            tanphi_ratio=tanphi_ratio,
            rho_w=rho_w,
            nu=nu,
            g=g,
            beta_s=beta_s,
            c2=c2,
            u0=u0,
            n_iter=n_iter,
            tol=tol,
        )
    return ustarcrit_c


def d50_from_chezy_loglaw(
    chezy,
    h_c,
    hmin,
    nk=2.5,
    alpha_d90_d50=1.5,
    chezy_min=None,
    ks_cap_ratio=0.99,
    d50_clip=(1e-5, 0.2),
    verbose=False,
):
    """Estimate representative bed ``d50`` from logarithmic Chezy roughness."""
    chezy = np.asarray(chezy, dtype=float).ravel()
    h_c = np.asarray(h_c, dtype=float).ravel()

    if chezy_min is None:
        chezy_min = 5.75 * np.log10(12.0)

    d50 = np.full_like(h_c, np.nan, dtype=float)
    ks = np.full_like(h_c, np.nan, dtype=float)

    valid = np.isfinite(h_c) & (h_c > hmin) & np.isfinite(chezy) & (chezy > 0)
    chezy_eff = chezy.copy()
    chezy_eff[valid] = np.maximum(chezy_eff[valid], chezy_min)

    ks[valid] = 12.0 * h_c[valid] / (10.0 ** (chezy_eff[valid] / 5.75))
    ks[valid] = np.minimum(ks[valid], ks_cap_ratio * h_c[valid])

    d90 = ks / nk
    d50[valid] = d90[valid] / alpha_d90_d50

    if d50_clip is not None:
        d50 = np.clip(d50, d50_clip[0], d50_clip[1])

    if verbose:
        wet = valid & np.isfinite(d50)
        print("d50_from_chezy_loglaw diagnostics:")
        if np.any(wet):
            print("  h stats (wet)      [m]:", np.nanmin(h_c[wet]), np.nanmedian(h_c[wet]), np.nanmax(h_c[wet]))
            print("  Chezy stats (wet) [-]:", np.nanmin(chezy[wet]), np.nanmedian(chezy[wet]), np.nanmax(chezy[wet]))
            print("  Ks stats (wet)     [m]:", np.nanmin(ks[wet]), np.nanmedian(ks[wet]), np.nanmax(ks[wet]))
            print("  d50 stats (wet)    [m]:", np.nanmin(d50[wet]), np.nanmedian(d50[wet]), np.nanmax(d50[wet]))
        else:
            print("  no valid wet cells")
        print("  cells with Chezy < Chezy_min:", np.count_nonzero(valid & (chezy < chezy_min)))

    return d50


def entrainment_probability(
    ustar: np.ndarray,
    ustar_crit: np.ndarray,
    sigma_star: float = 0.2,
    mode: str = "gaussian_threshold",
    k: float | None = None,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Compute probabilistic entrainment from local shear velocity."""
    ustar = np.asarray(ustar, dtype=float)
    ustar_crit = np.asarray(ustar_crit, dtype=float)
    ratio = ustar / np.maximum(ustar_crit, eps)

    if mode in ("gaussian_threshold", "normal_threshold", "erf"):
        sigma_eff = max(float(sigma_star), eps)
        p = 0.5 * (1.0 + erf((ratio - 1.0) / (np.sqrt(2.0) * sigma_eff)))
    elif mode == "logistic":
        k_eff = 8.0 if k is None else float(k)
        p = 1.0 / (1.0 + np.exp(-k_eff * (ratio - 1.0)))
    elif mode == "linear":
        r = ustar_crit / np.maximum(ustar, eps)
        p = np.clip(1.0 - r, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown entrainment probability mode: {mode!r}")
    return np.clip(p, 0.0, 1.0)


def sample_entrainment(
    ustar: np.ndarray,
    ustar_crit: np.ndarray,
    rng: np.random.Generator,
    sigma_star: float = 0.2,
    mode: str = "gaussian_threshold",
    k: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample entrainment events from the computed probability field."""
    probability = entrainment_probability(
        ustar=ustar,
        ustar_crit=ustar_crit,
        sigma_star=sigma_star,
        mode=mode,
        k=k,
    )
    entrain = rng.random(probability.shape) < probability
    return entrain, probability


def resolve_bed_entrainment(
    *,
    bed_policy: str,
    tid: np.ndarray,
    wet_cell: np.ndarray,
    can_reentrain: np.ndarray,
    ustar_c: np.ndarray,
    ustar_crit_c: np.ndarray,
    rng: np.random.Generator,
    entrainment_prob_mode: str,
    entrainment_prob_sigma_star: float = 0.2,
    entrainment_prob_k: float | None = None,
) -> np.ndarray:
    """Return a pickup/reflect mask for deposited or bed-contact particles."""
    tid = np.asarray(tid, dtype=int).ravel()
    if tid.size == 0:
        return np.zeros(0, dtype=bool)

    if bed_policy == "always_deposit":
        return np.zeros(tid.size, dtype=bool)
    if bed_policy == "always_reflect":
        return np.ones(tid.size, dtype=bool)
    if bed_policy == "reflect_if_ustar_gt_crit":
        can_reentrain_arr = np.asarray(can_reentrain, dtype=bool).ravel()
        if can_reentrain_arr.size == tid.size:
            return can_reentrain_arr
        return np.asarray(can_reentrain_arr[tid], dtype=bool)
    if bed_policy != "probabilistic_entrainment":
        raise ValueError(
            'bedPolicy must be '
            '"always_deposit", "always_reflect", '
            '"reflect_if_ustar_gt_crit", or "probabilistic_entrainment".'
        )

    valid = np.asarray(wet_cell[tid], dtype=bool)
    pickup = np.zeros(tid.size, dtype=bool)
    if np.any(valid):
        ustar_crit_arr = np.asarray(ustar_crit_c, dtype=float).ravel()
        if ustar_crit_arr.size == tid.size:
            ustar_crit_valid = ustar_crit_arr[valid]
        else:
            ustar_crit_valid = ustar_crit_arr[tid[valid]]
        pickup_valid, _ = sample_entrainment(
            ustar=ustar_c[tid[valid]],
            ustar_crit=ustar_crit_valid,
            rng=rng,
            sigma_star=entrainment_prob_sigma_star,
            mode=entrainment_prob_mode,
            k=entrainment_prob_k,
        )
        pickup[valid] = pickup_valid
    return pickup


def resolve_bed_contact(
    *,
    bed_policy: str,
    tid: np.ndarray,
    wet_cell: np.ndarray,
    can_reentrain: np.ndarray,
    ustar_c: np.ndarray,
    ustar_crit_c: np.ndarray,
    rng: np.random.Generator,
    entrainment_prob_mode: str,
    entrainment_prob_sigma_star: float = 0.2,
    entrainment_prob_k: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reflect and deposit masks for particles that hit the bed."""
    reflect = resolve_bed_entrainment(
        bed_policy=bed_policy,
        tid=tid,
        wet_cell=wet_cell,
        can_reentrain=can_reentrain,
        ustar_c=ustar_c,
        ustar_crit_c=ustar_crit_c,
        rng=rng,
        entrainment_prob_mode=entrainment_prob_mode,
        entrainment_prob_sigma_star=entrainment_prob_sigma_star,
        entrainment_prob_k=entrainment_prob_k,
    )
    return reflect, ~reflect
