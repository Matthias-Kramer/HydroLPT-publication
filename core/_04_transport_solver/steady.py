"""Core particle-transport integrators for steady hydraulic conditions."""

from __future__ import annotations

import os
from time import perf_counter

import numpy as np

try:
    from numba import njit, prange
except ImportError:
    njit = None
    prange = range

from core._03_particle_evolution import (
    biofilm_thickness_from_age,
    compute_biofouled_state,
    compute_degraded_state,
    degradation_scale_from_age,
)
from core._05_boundary_interaction import (
    apply_dry_stop,
    apply_outside_stop,
    resolve_bed_contact,
    resolve_bed_entrainment,
    tangential_slide_wetdry_subset_fullcache,
    resolve_surface_contact,
    resolve_surface_detachment,
    ustarcrit_from_eq20,
    ustarcrit_surface_from_particle,
    validate_boundary_policies,
    wet_dry_stop_contact_points,
)


def _langevin_ou_step(
    current: np.ndarray,
    sigma: np.ndarray,
    tau_l: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Advance one Ornstein-Uhlenbeck fluctuation step."""
    current = np.asarray(current, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if current.size == 0:
        return current

    tau_eff = max(float(tau_l), 1.0e-9)
    decay = np.exp(-dt / tau_eff)
    return decay * current + sigma * np.sqrt(max(0.0, 1.0 - decay * decay)) * rng.standard_normal(current.size)


def _profile_add(profile: dict | None, key: str, dt_seconds: float) -> None:
    """Accumulate section timings into a mutable profiling dictionary."""
    if profile is None:
        return
    sections = profile.setdefault("sections", {})
    sections[key] = sections.get(key, 0.0) + float(dt_seconds)


def _profile_increment(profile: dict | None, key: str, amount: int = 1) -> None:
    """Accumulate integer counters into a mutable profiling dictionary."""
    if profile is None:
        return
    counters = profile.setdefault("counts", {})
    counters[key] = counters.get(key, 0) + int(amount)


def _progress_chunk_enabled() -> bool:
    """Return whether extra per-progress-chunk timing should be printed."""
    raw = os.getenv("HYDROLPT_PROGRESS_PROFILE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


if njit is not None:
    @njit(cache=True, parallel=True)
    def _local_idw_sample_fields_numba(
        triangle_ids,
        points_xy,
        cell_centers,
        neighbors,
        field_matrix,
        field_indices,
        valid_cell_mask,
        power,
        eps,
    ):
        n_points = triangle_ids.size
        n_fields = field_indices.size
        n_neighbor = neighbors.shape[1]
        out = np.empty((n_fields, n_points), dtype=np.float64)
        for field_pos in prange(n_fields):
            for point_pos in range(n_points):
                out[field_pos, point_pos] = np.nan

        for point_pos in prange(n_points):
            tid = triangle_ids[point_pos]
            if tid < 0 or tid >= cell_centers.shape[0]:
                continue

            px = points_xy[point_pos, 0]
            py = points_xy[point_pos, 1]
            denom = np.zeros(n_fields, dtype=np.float64)
            numer = np.zeros(n_fields, dtype=np.float64)

            for candidate_pos in range(n_neighbor + 1):
                if candidate_pos == 0:
                    cell_id = tid
                else:
                    cell_id = neighbors[tid, candidate_pos - 1]
                if cell_id < 0 or cell_id >= cell_centers.shape[0]:
                    continue
                if not valid_cell_mask[cell_id]:
                    continue

                dx = cell_centers[cell_id, 0] - px
                dy = cell_centers[cell_id, 1] - py
                dist = (dx * dx + dy * dy) ** 0.5
                weight = 1.0 / (max(dist, eps) ** power)

                for field_pos in range(n_fields):
                    value = field_matrix[field_indices[field_pos], cell_id]
                    if np.isfinite(value):
                        denom[field_pos] += weight
                        numer[field_pos] += value * weight

            for field_pos in range(n_fields):
                if denom[field_pos] > 0.0:
                    out[field_pos, point_pos] = numer[field_pos] / denom[field_pos]
        return out
else:
    _local_idw_sample_fields_numba = None


class _LocalCellIDWFieldSampler:
    """Sample cellwise fields directly at points using local cell-center IDW."""

    def __init__(
        self,
        mesh,
        field_map: dict[str, np.ndarray | None],
        power: float = 2.0,
        valid_cell_mask: np.ndarray | None = None,
    ) -> None:
        self.mesh = mesh
        if hasattr(mesh, "cell_centers"):
            self.cell_centers = np.asarray(mesh.cell_centers, dtype=float)
        elif all(hasattr(mesh, name) for name in ("V0", "V1", "V2")):
            self.cell_centers = (mesh.V0 + mesh.V1 + mesh.V2) / 3.0
        else:
            raise ValueError("Mesh must expose cell_centers or triangular vertices for local IDW.")
        self.neighbors = np.asarray(mesh.neighbors, dtype=int)
        self.power = float(power)
        self.eps = 1.0e-12
        if valid_cell_mask is None:
            self.valid_cell_mask = np.ones(self.cell_centers.shape[0], dtype=bool)
        else:
            self.valid_cell_mask = np.asarray(valid_cell_mask, dtype=bool).ravel()
            if self.valid_cell_mask.size != self.cell_centers.shape[0]:
                raise ValueError("valid_cell_mask must have size Nc")
        self._fields: dict[str, np.ndarray] = {}
        self._field_index: dict[str, int] = {}
        for name, values in field_map.items():
            if values is None:
                continue
            arr = np.asarray(values, dtype=float).ravel()
            if arr.size != self.cell_centers.shape[0]:
                raise ValueError("Local IDW field values must have size Nc")
            self._fields[name] = arr
            self._field_index[name] = len(self._field_index)

        if self._fields:
            self._field_matrix = np.vstack([self._fields[name] for name in self._field_index])
        else:
            self._field_matrix = np.empty((0, self.cell_centers.shape[0]), dtype=float)

    def sample_fields(
        self,
        tid: np.ndarray,
        points_xy: np.ndarray,
        names: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        """Sample multiple named fields at known triangle ids and xy locations."""
        triangle_ids = np.asarray(tid, dtype=int).ravel()
        points = np.asarray(points_xy, dtype=float)
        out = {name: np.full(triangle_ids.shape, np.nan, dtype=float) for name in names}
        valid = triangle_ids >= 0
        if not np.any(valid):
            return out

        if _local_idw_sample_fields_numba is not None and self._field_matrix.size:
            present_names = [name for name in names if name in self._field_index]
            if present_names:
                field_indices = np.asarray([self._field_index[name] for name in present_names], dtype=np.int64)
                sampled = _local_idw_sample_fields_numba(
                    triangle_ids[valid].astype(np.int64, copy=False),
                    points[valid],
                    self.cell_centers,
                    self.neighbors,
                    self._field_matrix,
                    field_indices,
                    self.valid_cell_mask,
                    self.power,
                    self.eps,
                )
                for field_pos, name in enumerate(present_names):
                    out[name][valid] = sampled[field_pos]
            return out

        candidate_cells = np.column_stack([triangle_ids[valid], self.neighbors[triangle_ids[valid]]])
        candidate_ok = candidate_cells >= 0
        candidate_cells_safe = np.where(candidate_ok, candidate_cells, 0)
        candidate_ok &= self.valid_cell_mask[candidate_cells_safe]
        centers = self.cell_centers[candidate_cells_safe]
        delta = centers - points[valid, None, :]
        dist = np.linalg.norm(delta, axis=2)
        base_weights = np.where(candidate_ok, 1.0 / np.maximum(dist, self.eps) ** self.power, 0.0)

        for name in names:
            values = self._fields.get(name)
            if values is None:
                continue
            local_values = values[candidate_cells_safe]
            finite = np.isfinite(local_values)
            weights = np.where(finite, base_weights, 0.0)
            denom = np.sum(weights, axis=1)
            sampled = np.full(np.count_nonzero(valid), np.nan, dtype=float)
            good = denom > 0.0
            if np.any(good):
                sampled[good] = np.sum(np.where(finite, local_values * weights, 0.0), axis=1)[good] / denom[good]
            out[name][valid] = sampled

        return out


def _sample_transport_velocity(
    ubar: np.ndarray,
    vbar: np.ndarray,
    ustar: np.ndarray | None,
    h: np.ndarray,
    zb: np.ndarray,
    zp: np.ndarray,
    ks: np.ndarray | None,
    mode: str,
    hydraulic_closure_model: str = "external_adapter",
    z0: np.ndarray | None = None,
    z_frac: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample horizontal transport velocity from depth-averaged hydraulics.

    Parameters
    ----------
    ubar, vbar : (N,)
        Depth-averaged cell velocities.
    h : (N,)
        Water depth.
    zb : (N,)
        Bed elevation.
    zp : (N,)
        Particle elevation.
    ks : (N,) or None
        Roughness height for log-law mode.
    mode : str
        "depth_averaged" or "loglaw_vertical"

    Returns
    -------
    u, v : (N,)
        Sampled horizontal velocities for transport.
    """
    ubar = np.asarray(ubar, dtype=float)
    vbar = np.asarray(vbar, dtype=float)
    h = np.asarray(h, dtype=float)
    zb = np.asarray(zb, dtype=float)
    zp = np.asarray(zp, dtype=float)
    if ustar is not None:
        ustar = np.asarray(ustar, dtype=float)

    u = ubar.copy()
    v = vbar.copy()
    hydraulic_closure_model = str(hydraulic_closure_model).strip().lower()

    if mode == "depth_averaged":
        return u, v

    if mode == "depth_averaged_above_release":
        if hydraulic_closure_model not in ("external_adapter", "standard_z0"):
            raise ValueError("hydraulicClosureModel must be 'external_adapter' or 'standard_z0'")
        if hydraulic_closure_model == "external_adapter":
            if ks is None:
                raise ValueError(
                    "ks array must be provided when "
                    "transportVelocityMode='depth_averaged_above_release' "
                    "with hydraulicClosureModel='external_adapter'"
                )
            ks = np.asarray(ks, dtype=float)
        elif ks is not None:
            ks = np.asarray(ks, dtype=float)

        if z0 is not None:
            z0 = np.asarray(z0, dtype=float)

        good = (
            np.isfinite(ubar)
            & np.isfinite(vbar)
            & np.isfinite(h)
            & (h > 0.0)
        )
        if hydraulic_closure_model == "external_adapter":
            good &= np.isfinite(ks) & (ks > 0.0)
        elif z0 is not None:
            good &= np.isfinite(z0) & (z0 > 0.0)
        if not np.any(good):
            return u, v

        speed_bar = np.hypot(ubar[good], vbar[good])
        ex = np.zeros_like(speed_bar)
        ey = np.zeros_like(speed_bar)
        has_dir = np.isfinite(speed_bar) & (speed_bar > 1e-12)
        ex[has_dir] = ubar[good][has_dir] / speed_bar[has_dir]
        ey[has_dir] = vbar[good][has_dir] / speed_bar[has_dir]

        h_good = h[good]
        z_rel_release = np.clip(float(z_frac), 0.0, 1.0) * h_good

        speed = np.zeros_like(speed_bar)

        if hydraulic_closure_model == "external_adapter":
            ks_eff = np.maximum(ks[good], 1e-12)
            z1 = np.maximum(z_rel_release, ks_eff)
            z1 = np.minimum(z1, h_good)
            layer = np.maximum(h_good - z1, 1e-12)
            denom = np.log(np.maximum(12.0 * h_good / ks_eff, 12.0))
            numer_int = (
                h_good * np.log(np.maximum(12.0 * h_good / ks_eff, 12.0))
                - z1 * np.log(np.maximum(12.0 * z1 / ks_eff, 12.0))
                - layer
            )
            fac = numer_int / np.maximum(layer * np.maximum(denom, 1e-12), 1e-12)
            speed = speed_bar * fac
        else:
            if ustar is None:
                raise ValueError("ustar array must be provided for hydraulicClosureModel='standard_z0'")
            if z0 is None:
                raise ValueError(
                    "z0 array must be provided when "
                    "transportVelocityMode='depth_averaged_above_release' "
                    "with hydraulicClosureModel='standard_z0'"
                )
            z0_eff = np.maximum(z0[good], 1e-12)

            z1 = np.maximum(z_rel_release, z0_eff * (1.0 + 1e-12))
            z1 = np.minimum(z1, h_good)
            layer = np.maximum(h_good - z1, 1e-12)

            valid_speed = np.isfinite(ustar[good]) & (ustar[good] > 0.0)
            speed[valid_speed] = (
                ustar[good][valid_speed] / 0.41
            ) * (
                h_good[valid_speed] * np.log(np.maximum(h_good[valid_speed] / z0_eff[valid_speed], 1.0))
                - z1[valid_speed] * np.log(np.maximum(z1[valid_speed] / z0_eff[valid_speed], 1.0))
                - layer[valid_speed]
            ) / layer[valid_speed]

        u[good] = speed * ex
        v[good] = speed * ey
        return u, v

    if mode != "loglaw_vertical":
        raise ValueError(
            "transportVelocityMode must be 'depth_averaged', "
            "'depth_averaged_above_release', or 'loglaw_vertical'"
        )
    if hydraulic_closure_model not in ("external_adapter", "standard_z0"):
        raise ValueError("hydraulicClosureModel must be 'external_adapter' or 'standard_z0'")

    if ks is None:
        raise ValueError("ks array must be provided when transportVelocityMode='loglaw_vertical'")

    ks = np.asarray(ks, dtype=float)
    if z0 is not None:
        z0 = np.asarray(z0, dtype=float)

    good = (
        np.isfinite(ubar)
        & np.isfinite(vbar)
        & np.isfinite(h)
        & np.isfinite(zb)
        & np.isfinite(zp)
        & np.isfinite(ks)
        & (h > 0.0)
        & (ks > 0.0)
    )

    if not np.any(good):
        return u, v

    ks_eff = np.maximum(ks[good], 1e-12)
    zrel = zp[good] - zb[good]

    if hydraulic_closure_model == "external_adapter":
        # Historical HYDROLPT / BASEMENT-compatible profile:
        # scale the depth-averaged velocity by ln(12 z / ks) / ln(12 h / ks)
        # with a near-bed floor at z = ks.
        z_eff = np.maximum(zrel, ks_eff)
        z_eff = np.minimum(z_eff, h[good])

        denom = np.log(np.maximum(12.0 * h[good] / ks_eff, 12.0))
        numer = np.log(np.maximum(12.0 * z_eff / ks_eff, 12.0))
        fac = numer / np.maximum(denom, 1e-12)

        u[good] = ubar[good] * fac
        v[good] = vbar[good] * fac
        return u, v

    if ustar is None:
        raise ValueError("ustar array must be provided for hydraulicClosureModel='standard_z0'")

    if z0 is None:
        z0_eff = ks_eff / 30.0
    else:
        z0_eff = np.maximum(z0[good], 1e-12)

    z_eff = np.maximum(zrel, z0_eff * (1.0 + 1e-12))
    z_eff = np.minimum(z_eff, h[good])

    speed_bar = np.hypot(ubar[good], vbar[good])
    ex = np.zeros_like(speed_bar)
    ey = np.zeros_like(speed_bar)
    has_dir = np.isfinite(speed_bar) & (speed_bar > 1e-12)
    ex[has_dir] = ubar[good][has_dir] / speed_bar[has_dir]
    ey[has_dir] = vbar[good][has_dir] / speed_bar[has_dir]

    speed = np.zeros_like(speed_bar)
    valid_speed = np.isfinite(ustar[good]) & (ustar[good] > 0.0)
    speed[valid_speed] = (ustar[good][valid_speed] / 0.41) * np.log(
        np.maximum(z_eff[valid_speed] / z0_eff[valid_speed], 1.0)
    )

    u[good] = speed * ex
    v[good] = speed * ey
    return u, v


def advect_with_frame(
    mesh,
    frame,
    zb_c,
    ustar_c,
    ustarCrit_c,
    surfaceUstarCrit,
    bedPolicy,
    P0,
    t0,
    t1,
    dt,
    outsidePolicy,
    dryPolicy,
    hmin,
    useRWx,
    useRWy,
    betaKh,
    KhMax,
    ws,
    zFrac,
    surfacePolicy,
    dzUpMax,
    uphillPolicy,
    useRWz,
    alphaKz,
    KzMax,
    rng=None,
    nTrack=0,
    track_idx=None,
    showProgress=True,
    flags0=None,    
    transportModel="random_walk",
    hydraulicFieldMode="cellwise",
    transportVelocityMode="depth_averaged",
    hydraulicClosureModel="external_adapter",
    d50_c=None,
    rAniso=3.0, # hard coded here
    entrainmentProbMode="gaussian_threshold",
    entrainmentProbSigmaStar=0.2,
    entrainmentProbK=None,
    surfaceDetachmentProbMode="gaussian_threshold",
    surfaceDetachmentProbSigmaStar=0.2,
    surfaceDetachmentProbK=None,
    TL_horizontal=2.0,
    TL_vertical=0.5,
    release_times=None,
    transport_state0=None,
    particleEvolution=None,
    trackOutputStride=1,
):
    """Dispatch steady advection using a single hydraulic frame object."""
    ks_c = getattr(frame, "ks_c", None)
    z0_c = getattr(frame, "z0_c", None)

  
    return advect_particles_euler_cell_fast_fullcache_3d_ustar(
        mesh,
        frame.U_c,
        frame.V_c,
        frame.h_c,
        zb_c,
        frame.wse_c,
        ustar_c,
        ustarCrit_c,
        surfaceUstarCrit,
        bedPolicy,
        P0,
        t0,
        t1,
        dt,
        outsidePolicy,
        dryPolicy,
        hmin,
        useRWx,
        useRWy,
        betaKh,
        KhMax,
        ws,
        zFrac,
        surfacePolicy,
        dzUpMax,
        uphillPolicy,
        useRWz,
        alphaKz,
        KzMax,
        rng=rng,
        nTrack=nTrack,
        track_idx=track_idx,
        showProgress=showProgress,
        flags0=flags0,       
        depositedAlive=True,
        transportModel=transportModel,
        hydraulicFieldMode=hydraulicFieldMode,
        transportVelocityMode=transportVelocityMode,
        hydraulicClosureModel=hydraulicClosureModel,
        d50_c=d50_c,
        ks_c=ks_c,
        z0_c=z0_c,
        rAniso=rAniso,
        entrainmentProbMode=entrainmentProbMode,
        entrainmentProbSigmaStar=entrainmentProbSigmaStar,
        entrainmentProbK=entrainmentProbK,
        surfaceDetachmentProbMode=surfaceDetachmentProbMode,
        surfaceDetachmentProbSigmaStar=surfaceDetachmentProbSigmaStar,
        surfaceDetachmentProbK=surfaceDetachmentProbK,
        TL_horizontal=TL_horizontal,
        TL_vertical=TL_vertical,
        release_times=release_times,
        transport_state0=transport_state0,
        particleEvolution=particleEvolution,
        trackOutputStride=trackOutputStride,
    )


def advect_particles_euler_cell_fast_fullcache_3d_ustar(
    mesh,
    u_c,
    v_c,
    h_c,
    zb_c,
    wse_c,
    ustar_c,
    ustarCrit,
    surfaceUstarCrit,
    bedPolicy,
    P0,
    t0,
    t1,
    dt,
    outsidePolicy,
    dryPolicy,
    hmin,
    useRWx,
    useRWy,
    betaKh,
    KhMax,
    ws,
    zFrac,
    surfacePolicy,
    dzUpMax,
    uphillPolicy,
    useRWz,
    alphaKz,
    KzMax,
    rng=None,
    nTrack=6,
    track_idx=None,
    showProgress=True,
    flags0=None,
    depositedAlive=True, # hard coded here, deposited particles are kept alive
    transportModel="random_walk",
    hydraulicFieldMode="cellwise",
    transportVelocityMode="depth_averaged",
    hydraulicClosureModel="external_adapter",
    d50_c=None,
    ks_c=None,
    z0_c=None,
    rAniso=3.0,
    entrainmentProbMode="gaussian_threshold",
    entrainmentProbSigmaStar=0.2,
    entrainmentProbK=None,
    surfaceDetachmentProbMode="gaussian_threshold",
    surfaceDetachmentProbSigmaStar=0.2,
    surfaceDetachmentProbK=None,
    TL_horizontal=2.0,
    TL_vertical=0.5,
    tid0=None,
    positionsInitialized=False,
    countResidence=True,
    release_times=None,
    stopWhenAllParticlesSurfaced=False,
    transport_state0=None,
    particleEvolution=None,
    profile=None,
    fieldSamplerOverride=None,
    trackOutputStride=1,
):
    """
    Cell-centered Euler advection with:
      - horizontal random walk
      - vertical settling / RWz
      - wet/dry handling
      - deposition / reflection at the bed
      - surface stick / reflection / detachment
      - uphill limiter

    Horizontal RW switches
    ----------------------
    useRWx : bool
        Streamwise random walk along local flow direction.
    useRWy : bool
        Transverse random walk normal to local flow direction.

    depositedAlive
    --------------
    False:
        Deposited particles become terminal and no longer evolve.

    True:
        Deposited particles remain in the simulation, contribute to cell
        residence/occupancy statistics, may be re-entrained, but do not move
        horizontally while deposited.

    Surface interaction
    -------------------
    Surface policies can either always reflect, always stick, or use a
    Kramer-style detachment threshold to decide whether surfaced particles
    re-enter the water column.

    Bed policies
    ------------
    "always_deposit"
        Always deposit at the bed.

    "always_reflect"
        Always reflect at the bed.

    "reflect_if_ustar_gt_crit"
        Old deterministic option using strict threshold:
        ustar > ustarCrit.

    "probabilistic_entrainment"
        New probabilistic option using sample_entrainment(...) from the
        boundary-interaction framework.

    Notes
    -----
    - Residence counts particle presence in a cell, including deposited particles
      when depositedAlive=True.
    - When depositedAlive=True, deposited particles may resuspend depending on
      the selected bedPolicy.
    """

    # =========================================================
    # checks / setup
    # =========================================================
    validate_boundary_policies(
        surface_policy=surfacePolicy,
        bed_policy=bedPolicy,
        dry_policy=dryPolicy,
        outside_policy=outsidePolicy,
    )
    if float(ws) < 0.0:
        bedPolicy = "always_reflect"
    elif float(ws) > 0.0:
        surfacePolicy = "always_reflect"
    elif float(ws) == 0.0:
        bedPolicy = "always_reflect"
        surfacePolicy = "always_reflect"
    if uphillPolicy not in ("off", "stop"):
        raise ValueError('uphillPolicy must be "off" or "stop".')
    if transportVelocityMode not in ("depth_averaged", "depth_averaged_above_release", "loglaw_vertical"):
        raise ValueError(
            "transportVelocityMode must be 'depth_averaged', "
            "'depth_averaged_above_release', or 'loglaw_vertical'"
        )
    if hydraulicFieldMode not in ("cellwise", "local_idw"):
        raise ValueError("hydraulicFieldMode must be 'cellwise' or 'local_idw'")
    if transportModel not in ("random_walk", "langevin"):
        raise ValueError("transportModel must be 'random_walk' or 'langevin'")
    hydraulicClosureModel = str(hydraulicClosureModel).strip().lower()
    if hydraulicClosureModel not in ("external_adapter", "standard_z0"):
        raise ValueError("hydraulicClosureModel must be 'external_adapter' or 'standard_z0'")
    if transportModel == "langevin":
        if float(TL_horizontal) <= 0.0:
            raise ValueError("TL_horizontal must be > 0 for Langevin transport.")
        if float(TL_vertical) <= 0.0:
            raise ValueError("TL_vertical must be > 0 for Langevin transport.")

    if rng is None:
        rng = np.random.default_rng(1)

    P0 = np.asarray(P0, dtype=float)
    if P0.ndim != 2 or P0.shape[1] != 3:
        raise ValueError("P0 must be (Np,3) with columns [x, y, z]")

    u_c = np.asarray(u_c, dtype=float).ravel()
    v_c = np.asarray(v_c, dtype=float).ravel()
    h_c = np.asarray(h_c, dtype=float).ravel()
    zb_c = np.asarray(zb_c, dtype=float).ravel()
    wse_c = np.asarray(wse_c, dtype=float).ravel()
    ustar_c = np.asarray(ustar_c, dtype=float).ravel()

    Np = P0.shape[0]
    Nc = u_c.size

    if v_c.size != Nc or h_c.size != Nc or zb_c.size != Nc or wse_c.size != Nc or ustar_c.size != Nc:
        raise ValueError("Hydraulic arrays must all have size Nc")

    if transportVelocityMode == "loglaw_vertical":
        if ks_c is None:
            raise ValueError("ks_c must be provided for loglaw_vertical transport")
        ks_c = np.asarray(ks_c, dtype=float).ravel()
        if ks_c.size != Nc:
            raise ValueError("ks_c must have size Nc")
        if z0_c is not None:
            z0_c = np.asarray(z0_c, dtype=float).ravel()
            if z0_c.size != Nc:
                raise ValueError("z0_c must have size Nc")

    if np.ndim(ustarCrit) == 0:
        ustarCrit_c = np.full(Nc, float(ustarCrit), dtype=float)
    else:
        ustarCrit_c = np.asarray(ustarCrit, dtype=float).ravel()
        if ustarCrit_c.size != Nc:
            raise ValueError("ustarCrit array must have size Nc")

    if surfaceUstarCrit is None:
        surfaceUstarCrit_c = ustarCrit_c.copy()
    elif np.ndim(surfaceUstarCrit) == 0:
        surfaceUstarCrit_c = np.full(Nc, float(surfaceUstarCrit), dtype=float)
    else:
        surfaceUstarCrit_c = np.asarray(surfaceUstarCrit, dtype=float).ravel()
        if surfaceUstarCrit_c.size != Nc:
            raise ValueError("surfaceUstarCrit array must have size Nc")
    if d50_c is not None:
        d50_c = np.asarray(d50_c, dtype=float).ravel()
        if d50_c.size != Nc:
            raise ValueError("d50_c array must have size Nc")

    Nt = int(round((t1 - t0) / dt)) + 1
    T = t0 + dt * np.arange(Nt, dtype=float)
    trackOutputStride = max(1, int(trackOutputStride))
    Ttrack = T[::trackOutputStride]
    if release_times is None:
        release_times_arr = None
        all_released_from_start = True
    else:
        release_times_arr = np.asarray(release_times, dtype=float).ravel()
        if release_times_arr.size != Np:
            raise ValueError("release_times must have size Np")
        all_released_from_start = bool(np.all(release_times_arr <= (T[0] + 1e-12)))

    if release_times_arr is None:
        particle_release_times = np.full(Np, T[0], dtype=float)
    else:
        particle_release_times = release_times_arr.copy()

    tolBC = 1e-9
    progEvery = max(1, Nt // 100)
    progress_chunk_profile = _progress_chunk_enabled()

    # Precompute steady cellwise quantities
    wet_cell = np.isfinite(h_c) & (h_c > hmin)

    ustar_safe = np.where(np.isfinite(ustar_c), ustar_c, 0.0)
    ustar_field = np.where(wet_cell & np.isfinite(ustar_c), ustar_c, np.nan)
    ustarCrit_safe = np.where(np.isfinite(ustarCrit_c), ustarCrit_c, np.inf)
    surfaceUstarCrit_safe = np.where(np.isfinite(surfaceUstarCrit_c), surfaceUstarCrit_c, np.inf)
    h_safe = np.where(np.isfinite(h_c), h_c, 0.0)

    Kh_c = betaKh * ustar_safe * h_safe
    Kh_c[(~np.isfinite(Kh_c)) | (Kh_c < 0.0)] = 0.0
    Kh_c = np.minimum(Kh_c, KhMax)

    Kpar_c = np.minimum(rAniso * Kh_c, KhMax)
    Kperp_c = np.minimum(Kh_c / rAniso, KhMax)

    if not useRWx:
        Kpar_c[:] = 0.0
    if not useRWy:
        Kperp_c[:] = 0.0

    field_sampler = None
    if hydraulicFieldMode == "local_idw":
        if fieldSamplerOverride is not None:
            field_sampler = fieldSamplerOverride
        else:
            t_sampler = perf_counter()
            field_sampler = _LocalCellIDWFieldSampler(
                mesh,
                {
                    "u": u_c,
                    "v": v_c,
                    "h": h_c,
                    "zb": zb_c,
                    "wse": wse_c,
                    "ustar": ustar_field,
                    "ustarCrit": ustarCrit_c,
                    "surfaceUstarCrit": surfaceUstarCrit_c,
                    "ks": ks_c,
                    "z0": z0_c,
                    "kpar": Kpar_c,
                    "kperp": Kperp_c,
                },
                valid_cell_mask=wet_cell,
            )
            _profile_add(profile, "field_sampler_build_s", perf_counter() - t_sampler)

    def _sample_fields_at_indices(
        indices: np.ndarray,
        names: tuple[str, ...],
        *,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        tid_arr: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample selected fields at particle indices, with cellwise fallback."""
        idx = np.asarray(indices, dtype=int).ravel()
        out = {name: np.full(idx.shape, np.nan, dtype=float) for name in names}
        if idx.size == 0:
            return out

        triangle_ids = tidPrev[idx] if tid_arr is None else np.asarray(tid_arr, dtype=int).ravel()
        if triangle_ids.size != idx.size:
            raise ValueError("tid_arr must have the same size as indices")
        valid = triangle_ids >= 0
        if not np.any(valid):
            return out

        if field_sampler is not None:
            points_xy = np.empty((np.count_nonzero(valid), 2), dtype=float)
            points_xy[:, 0] = x_arr[idx[valid]]
            points_xy[:, 1] = y_arr[idx[valid]]
            sampled = field_sampler.sample_fields(triangle_ids[valid], points_xy, names)
            for name in names:
                out[name][valid] = sampled[name]
            return out

        source_map = {
            "u": u_c,
            "v": v_c,
            "h": h_c,
            "zb": zb_c,
            "wse": wse_c,
            "ustar": ustar_c,
            "ustarCrit": ustarCrit_c,
            "surfaceUstarCrit": surfaceUstarCrit_c,
            "ks": ks_c,
            "z0": z0_c,
            "kpar": Kpar_c,
            "kperp": Kperp_c,
        }
        for name in names:
            source = source_map.get(name)
            if source is None:
                continue
            out[name][valid] = np.asarray(source, dtype=float).ravel()[triangle_ids[valid]]
        return out

    def _sample_is_wet_at_indices(
        indices: np.ndarray,
        *,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        tid_arr: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a hybrid wetness test that avoids over-stopping near wet/dry fronts."""
        idx = np.asarray(indices, dtype=int).ravel()
        is_wet = np.zeros(idx.shape, dtype=bool)
        if idx.size == 0:
            return is_wet

        triangle_ids = tidPrev[idx] if tid_arr is None else np.asarray(tid_arr, dtype=int).ravel()
        if triangle_ids.size != idx.size:
            raise ValueError("tid_arr must have the same size as indices")
        valid = triangle_ids >= 0
        if not np.any(valid):
            return is_wet

        is_wet[valid] = wet_cell[triangle_ids[valid]]
        return is_wet

    def _contains_points_in_cells(cell_ids: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        """Return whether points remain in their current cells."""
        if hasattr(mesh, "contains_points"):
            return np.asarray(mesh.contains_points(cell_ids, points_xy, tol=tolBC), dtype=bool)
        barycentric_coords = mesh.barycentric(cell_ids, points_xy)
        _profile_increment(profile, "mesh_barycentric_calls", 1)
        return np.all(
            (barycentric_coords >= -tolBC) & (barycentric_coords <= 1.0 + tolBC),
            axis=1,
        )

    # Old deterministic threshold by design: equality does not entrain / reflect.
    can_reentrain_c = wet_cell & (ustar_safe > ustarCrit_safe)
    if float(ws) < 0.0:
        can_reentrain_c = wet_cell.copy()
    can_detach_surface_c = wet_cell & (ustar_safe > surfaceUstarCrit_safe)
    if float(ws) > 0.0:
        can_detach_surface_c = wet_cell.copy()

    # Cheap constants
    two_dt = 2.0 * dt
    kappa = 0.41
    z_eps_pickup = max(1.0e-4, 5 * abs(ws) * dt)

    biofouling_bundle = particleEvolution.get("biofouling") if particleEvolution else None
    degradation_bundle = particleEvolution.get("degradation") if particleEvolution else None
    biofouling_enabled = bool(biofouling_bundle and biofouling_bundle.get("config") is not None)
    degradation_enabled = bool(degradation_bundle and degradation_bundle.get("config") is not None)
    current_rho_p = np.full(Np, np.nan, dtype=float)
    current_ws = np.full(Np, float(ws), dtype=float)
    current_L = np.full(Np, np.nan, dtype=float)
    current_I = np.full(Np, np.nan, dtype=float)
    current_S = np.full(Np, np.nan, dtype=float)
    current_Vp = np.full(Np, np.nan, dtype=float)
    current_Dp = np.full(Np, np.nan, dtype=float)
    current_beta_p = np.full(Np, np.nan, dtype=float)

    def update_particle_evolution_state(t_now: float) -> None:
        if not biofouling_enabled and not degradation_enabled:
            current_rho_p.fill(np.nan)
            current_ws.fill(float(ws))
            current_L.fill(np.nan)
            current_I.fill(np.nan)
            current_S.fill(np.nan)
            current_Vp.fill(np.nan)
            current_Dp.fill(np.nan)
            current_beta_p.fill(np.nan)
            return

        ages = np.maximum(float(t_now) - particle_release_times, 0.0)
        if biofouling_enabled:
            bt = biofilm_thickness_from_age(ages, biofouling_bundle["config"])
            evolved = compute_biofouled_state(
                shape=biofouling_bundle["shape"],
                rho_p0=biofouling_bundle["rho_p0"],
                L0=biofouling_bundle["L"],
                I0=biofouling_bundle["I"],
                S0=biofouling_bundle["S"],
                rho_f=biofouling_bundle["rho_f"],
                nu=biofouling_bundle["nu"],
                g=biofouling_bundle["g"],
                bt=bt,
                rho_biofilm=biofouling_bundle["config"].rho_biofilm,
            )
        else:
            scale = degradation_scale_from_age(ages, degradation_bundle["config"])
            evolved = compute_degraded_state(
                shape=degradation_bundle["shape"],
                rho_p0=degradation_bundle["rho_p0"],
                L0=degradation_bundle["L"],
                I0=degradation_bundle["I"],
                S0=degradation_bundle["S"],
                rho_f=degradation_bundle["rho_f"],
                nu=degradation_bundle["nu"],
                g=degradation_bundle["g"],
                scale=scale,
            )
        current_L[:] = evolved["L"]
        current_I[:] = evolved["I"]
        current_S[:] = evolved["S"]
        current_Vp[:] = evolved["Vp"]
        current_Dp[:] = evolved["Dp"]
        current_beta_p[:] = evolved["beta_p"]
        current_rho_p[:] = evolved["rho_p"]
        current_ws[:] = evolved["ws"]

    def current_surface_crit_for_indices(indices: np.ndarray) -> np.ndarray:
        if (not biofouling_enabled and not degradation_enabled) or indices.size == 0:
            return np.zeros(indices.size, dtype=float)
        return np.asarray(
            [
                ustarcrit_surface_from_particle(
                    shape=(biofouling_bundle or degradation_bundle)["shape"],
                    rho_p=float(current_rho_p[i]),
                    L=float(current_L[i]),
                    I=float(current_I[i]),
                    S=float(current_S[i]),
                    Vp=float(current_Vp[i]),
                    rho_f=(biofouling_bundle or degradation_bundle)["rho_f"],
                    g=(biofouling_bundle or degradation_bundle)["g"],
                )
                for i in indices
            ],
            dtype=float,
        )

    def current_can_detach_surface_for_indices(
        indices: np.ndarray,
        tid_local: np.ndarray,
        *,
        x_arr: np.ndarray | None = None,
        y_arr: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return whether the selected particles are currently releasable from the free surface."""
        indices = np.asarray(indices, dtype=int).ravel()
        tid_local = np.asarray(tid_local, dtype=int).ravel()
        if indices.size == 0:
            return np.zeros(0, dtype=bool)
        if (not biofouling_enabled and not degradation_enabled):
            if float(ws) > 0.0:
                if field_sampler is None or x_arr is None or y_arr is None:
                    return np.asarray(wet_cell[tid_local], dtype=bool)
                return _sample_is_wet_at_indices(indices, x_arr=x_arr, y_arr=y_arr, tid_arr=tid_local)
            if field_sampler is None or x_arr is None or y_arr is None:
                return np.asarray(can_detach_surface_c[tid_local], dtype=bool)
            sampled = _sample_fields_at_indices(
                indices,
                ("ustar", "surfaceUstarCrit"),
                x_arr=x_arr,
                y_arr=y_arr,
                tid_arr=tid_local,
            )
            wet_local = _sample_is_wet_at_indices(indices, x_arr=x_arr, y_arr=y_arr, tid_arr=tid_local)
            return (
                wet_local
                & np.isfinite(sampled["ustar"])
                & np.isfinite(sampled["surfaceUstarCrit"])
                & (sampled["ustar"] > sampled["surfaceUstarCrit"])
            )

        rho_f_current = float((biofouling_bundle or degradation_bundle)["rho_f"])
        naturally_releasable = np.asarray(current_rho_p[indices], dtype=float) > rho_f_current
        if field_sampler is None or x_arr is None or y_arr is None:
            wet_local = np.asarray(wet_cell[tid_local], dtype=bool)
            ustar_local = ustar_c[tid_local]
        else:
            wet_local = _sample_is_wet_at_indices(indices, x_arr=x_arr, y_arr=y_arr, tid_arr=tid_local)
            ustar_local = _sample_fields_at_indices(
                indices,
                ("ustar",),
                x_arr=x_arr,
                y_arr=y_arr,
                tid_arr=tid_local,
            )["ustar"]
        return wet_local & (
            naturally_releasable | np.asarray(ustar_local > current_surface_crit_for_indices(indices), dtype=bool)
        )

    def current_bed_crit_for_indices(indices: np.ndarray, tid_local: np.ndarray) -> np.ndarray:
        if (not biofouling_enabled and not degradation_enabled) or indices.size == 0 or d50_c is None:
            return np.zeros(indices.size, dtype=float)
        return np.asarray(
            [
                ustarcrit_from_eq20(
                    Dp=float(current_Dp[i]),
                    d50=float(d50_c[tid_j]),
                    rho_p=float(current_rho_p[i]),
                    beta_p=float(current_beta_p[i]),
                    tanphi_ratio=(biofouling_bundle or degradation_bundle)["tanphi_ratio"],
                    rho_w=(biofouling_bundle or degradation_bundle)["rho_f"],
                    nu=(biofouling_bundle or degradation_bundle)["nu"],
                    g=(biofouling_bundle or degradation_bundle)["g"],
                )
                for i, tid_j in zip(indices, tid_local, strict=False)
            ],
            dtype=float,
        )

    def current_can_reentrain_for_indices(
        indices: np.ndarray,
        tid_local: np.ndarray,
        *,
        x_arr: np.ndarray | None = None,
        y_arr: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return whether the selected particles are currently releasable from the bed."""
        indices = np.asarray(indices, dtype=int).ravel()
        tid_local = np.asarray(tid_local, dtype=int).ravel()
        if indices.size == 0:
            return np.zeros(0, dtype=bool)
        if (not biofouling_enabled and not degradation_enabled):
            if float(ws) < 0.0:
                if field_sampler is None or x_arr is None or y_arr is None:
                    return np.asarray(wet_cell[tid_local], dtype=bool)
                return _sample_is_wet_at_indices(indices, x_arr=x_arr, y_arr=y_arr, tid_arr=tid_local)
            if field_sampler is None or x_arr is None or y_arr is None:
                return np.asarray(can_reentrain_c[tid_local], dtype=bool)
            sampled = _sample_fields_at_indices(
                indices,
                ("ustar", "ustarCrit"),
                x_arr=x_arr,
                y_arr=y_arr,
                tid_arr=tid_local,
            )
            wet_local = _sample_is_wet_at_indices(indices, x_arr=x_arr, y_arr=y_arr, tid_arr=tid_local)
            return (
                wet_local
                & np.isfinite(sampled["ustar"])
                & np.isfinite(sampled["ustarCrit"])
                & (sampled["ustar"] > sampled["ustarCrit"])
            )

        rho_f_current = float((biofouling_bundle or degradation_bundle)["rho_f"])
        naturally_releasable = np.asarray(current_rho_p[indices], dtype=float) < rho_f_current
        if field_sampler is None or x_arr is None or y_arr is None:
            wet_local = np.asarray(wet_cell[tid_local], dtype=bool)
            ustar_local = ustar_c[tid_local]
        else:
            wet_local = _sample_is_wet_at_indices(indices, x_arr=x_arr, y_arr=y_arr, tid_arr=tid_local)
            ustar_local = _sample_fields_at_indices(
                indices,
                ("ustar",),
                x_arr=x_arr,
                y_arr=y_arr,
                tid_arr=tid_local,
            )["ustar"]
        return wet_local & (
            naturally_releasable | np.asarray(ustar_local > current_bed_crit_for_indices(indices, tid_local), dtype=bool)
        )

    # =========================================================
    # tracked particles
    # =========================================================
    if track_idx is None:
        if Np == 0:
            track_idx = np.empty(0, dtype=int)
        else:
            nsel = int(min(max(0, nTrack), Np))
            track_idx = np.sort(rng.choice(Np, size=nsel, replace=False).astype(int))
    else:
        track_idx = np.sort(np.asarray(track_idx, dtype=int).ravel())
        if track_idx.size and ((track_idx.min() < 0) or (track_idx.max() >= Np)):
            raise ValueError("track_idx out of bounds for P0")

    Ntrack = track_idx.size
    Ntrack_times = Ttrack.size
    Ptrack = np.full((Ntrack, 3, Ntrack_times), np.nan, dtype=float)
    tidTr_track = np.full((Ntrack, Ntrack_times), -1, dtype=np.int32)
    stateCountHist = np.zeros((Ntrack_times, 10), dtype=np.int64)

    def write_state_counts(n):
        if n % trackOutputStride != 0:
            return
        k = n // trackOutputStride
        if k >= Ntrack_times:
            return
        if all_released_from_start:
            released = np.ones(Np, dtype=bool)
        else:
            released = release_times_arr <= (T[n] + 1e-12)
        active = released & (~stoppedByLimiter) & (~stoppedByDry) & (~stoppedByOutside)
        deposited_active = active & isDeposited
        surfaced_active = active & isSurfaced
        mobile_active = active & (~isDeposited) & (~isSurfaced)
        in_domain_mobile = mobile_active & (tidPrev >= 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

        near_bed = np.zeros(Np, dtype=bool)
        near_surface = np.zeros(Np, dtype=bool)
        if np.any(in_domain_mobile):
            ids = np.where(in_domain_mobile)[0]
            sampled_state = _sample_fields_at_indices(
                ids,
                ("zb", "wse", "h"),
                x_arr=x,
                y_arr=y,
            )
            ztol = max(1.0e-9, 0.5 * float(abs(ws)) * float(dt))
            wet_here = np.isfinite(sampled_state["h"]) & (sampled_state["h"] > hmin)
            near_bed_ids = wet_here & np.isfinite(sampled_state["zb"]) & (z[ids] <= sampled_state["zb"] + ztol)
            near_surface_ids = wet_here & np.isfinite(sampled_state["wse"]) & (z[ids] >= sampled_state["wse"] - ztol)
            near_bed[ids] = near_bed_ids
            near_surface[ids] = near_surface_ids

            overlap = near_bed & near_surface
            if np.any(overlap):
                if ws < 0.0:
                    near_bed[overlap] = False
                elif ws > 0.0:
                    near_surface[overlap] = False
                else:
                    near_bed[overlap] = False
                    near_surface[overlap] = False

        suspended = mobile_active & (~near_bed) & (~near_surface)
        mobile_count = int(np.count_nonzero(suspended | near_bed | near_surface | surfaced_active))
        stateCountHist[k] = (
            int(np.count_nonzero(released)),
            int(np.count_nonzero(stoppedByOutside)),
            int(np.count_nonzero(stoppedByDry)),
            int(np.count_nonzero(stoppedByLimiter)),
            int(np.count_nonzero(deposited_active)),
            int(np.count_nonzero(surfaced_active)),
            int(np.count_nonzero(near_bed)),
            int(np.count_nonzero(near_surface)),
            int(np.count_nonzero(suspended)),
            mobile_count,
        )

    def write_tracked(n, x, y, z, tid):
        if Ntrack == 0:
            return
        if n % trackOutputStride != 0:
            return
        k = n // trackOutputStride
        if k >= Ntrack_times:
            return
        if all_released_from_start:
            Ptrack[:, 0, k] = x[track_idx]
            Ptrack[:, 1, k] = y[track_idx]
            Ptrack[:, 2, k] = z[track_idx]
            tidTr_track[:, k] = tid[track_idx]
            return
        released = release_times_arr[track_idx] <= (T[n] + 1e-12)
        Ptrack[:, 0, k] = np.where(released, x[track_idx], np.nan)
        Ptrack[:, 1, k] = np.where(released, y[track_idx], np.nan)
        Ptrack[:, 2, k] = np.where(released, z[track_idx], np.nan)
        tidTr_track[:, k] = np.where(released, tid[track_idx], -1)

    def print_progress(n):
        if not showProgress:
            return
        if (n % progEvery == 0) or (n == 1) or (n == Nt - 1):
            pct = 100.0 * n / (Nt - 1) if Nt > 1 else 100.0
            print(f"Progress: {pct:5.1f} %  (step {n+1} / {Nt})", flush=True)

    def activate_particles_between(t_prev, t_now, x_arr, y_arr, z_arr):
        if all_released_from_start:
            return
        newly_released = (release_times_arr > t_prev + 1e-12) & (release_times_arr <= t_now + 1e-12)
        if not np.any(newly_released):
            return

        idx_new = np.where(newly_released & (~stoppedByOutside) & (~stoppedByLimiter) & (~stoppedByDry))[0]
        if idx_new.size == 0:
            return

        tid_new = mesh.point_location(x_arr[idx_new], y_arr[idx_new])
        ok_new = tid_new >= 0
        tidPrev[idx_new[ok_new]] = tid_new[ok_new]

        if np.any(ok_new):
            good_idx = idx_new[ok_new]
            tid_good = tid_new[ok_new]
            sampled_new = _sample_fields_at_indices(
                good_idx,
                ("zb", "wse"),
                x_arr=x_arr,
                y_arr=y_arr,
                tid_arr=tid_good,
            )
            zb_new = sampled_new["zb"]
            wse_new = sampled_new["wse"]
            z_here = z_arr[good_idx].copy()
            badz = ~np.isfinite(z_here)
            if np.any(badz):
                z_here[badz] = zb_new[badz] + zFrac * (wse_new[badz] - zb_new[badz])
            z_here = np.minimum(z_here, wse_new)
            z_here = np.maximum(z_here, zb_new)
            z_arr[good_idx] = z_here

        bad_idx = idx_new[~ok_new]
        if bad_idx.size:
            stoppedByOutside[bad_idx] = True
            new = np.isnan(tStopOutside[bad_idx])
            tStopOutside[bad_idx[new]] = t_now

    # =========================================================
    # state arrays
    # =========================================================
    x = P0[:, 0].copy()
    y = P0[:, 1].copy()
    z = P0[:, 2].copy()
    tidPrev = np.full(Np, -1, dtype=np.int32)
    if tid0 is not None:
        tid0 = np.asarray(tid0, dtype=np.int32).ravel()
        if tid0.size != Np:
            raise ValueError("tid0 must have size Np")
        tidPrev[:] = tid0

    cellCount = np.zeros(Nc, dtype=np.uint32)

    # ---------------------------------------------------------
    # initialize flags from incoming state or fresh
    # ---------------------------------------------------------
    def _get_bool_flag(name):
        if flags0 is None or name not in flags0:
            return np.zeros(Np, dtype=bool)
        arr = np.asarray(flags0[name], dtype=bool).ravel()
        if arr.size != Np:
            raise ValueError(f"flags0['{name}'] must have size Np")
        return arr.copy()

    def _get_float_flag(name):
        if flags0 is None or name not in flags0:
            return np.full(Np, np.nan, dtype=float)
        arr = np.asarray(flags0[name], dtype=float).ravel()
        if arr.size != Np:
            raise ValueError(f"flags0['{name}'] must have size Np")
        return arr.copy()

    isDeposited = _get_bool_flag("isDeposited")
    isSurfaced = _get_bool_flag("isSurfaced")
    limiterHitEver = _get_bool_flag("limiterHitEver")
    stoppedByLimiter = _get_bool_flag("stoppedByLimiter")
    stoppedByOutside = _get_bool_flag("stoppedByOutside")

    stoppedByDry = _get_bool_flag("stoppedByDry")
    tStopDry = _get_float_flag("tStopDry")

    tStopLimiter = _get_float_flag("tStopLimiter")
    tStopOutside = _get_float_flag("tStopOutside")
    tDeposit = _get_float_flag("tDeposit")
    tSurfaceHit = _get_float_flag("tSurfaceHit")
    xDryHit = _get_float_flag("xDryHit")
    yDryHit = _get_float_flag("yDryHit")
    dryStopAtStepStart = _get_bool_flag("dryStopAtStepStart")
    dryStopAtCrossing = _get_bool_flag("dryStopAtCrossing")
    xSurfaceHit = _get_float_flag("xSurfaceHit")
    ySurfaceHit = _get_float_flag("ySurfaceHit")
    def _get_transport_state(name):
        if transport_state0 is None or name not in transport_state0:
            return np.zeros(Np, dtype=float)
        arr = np.asarray(transport_state0[name], dtype=float).ravel()
        if arr.size != Np:
            raise ValueError(f"transport_state0['{name}'] must have size Np")
        return arr.copy()

    ufluc_par = _get_transport_state("ufluc_par")
    ufluc_perp = _get_transport_state("ufluc_perp")
    wfluc = _get_transport_state("wfluc")

    def _all_released_particles_done(t_now: float) -> bool:
        if not stopWhenAllParticlesSurfaced:
            return False

        if release_times_arr is None:
            released = np.ones(Np, dtype=bool)
        else:
            released = release_times_arr <= (t_now + 1e-12)
            if not np.all(released):
                return False

        if not np.any(released):
            return False

        surfaced_done = isSurfaced if surfacePolicy == "always_stick" else np.zeros(Np, dtype=bool)
        done = surfaced_done | stoppedByOutside | stoppedByLimiter | stoppedByDry
        return bool(np.all(done[released]))

    # =========================================================
    # initial cell location
    # =========================================================
    if not positionsInitialized:
        alive_init = np.isfinite(x) & np.isfinite(y) & (~stoppedByOutside) & (~stoppedByLimiter)
        if not all_released_from_start:
            alive_init &= release_times_arr <= (T[0] + 1e-12)
        alive_init &= (~stoppedByDry)

        if np.any(alive_init):
            idx_alive = np.where(alive_init)[0]
            tid_init = mesh.point_location(x[idx_alive], y[idx_alive])

            ok0 = tid_init >= 0
            tidPrev[idx_alive[ok0]] = tid_init[ok0]

            idx_bad0 = idx_alive[~ok0]
            if idx_bad0.size:
                stoppedByOutside[idx_bad0] = True
                new = np.isnan(tStopOutside[idx_bad0])
                tStopOutside[idx_bad0[new]] = T[0]

    # =========================================================
    # initial vertical coordinate
    # =========================================================
    if not positionsInitialized:
        ok_init = (
            (tidPrev >= 0)
            & np.isfinite(x)
            & np.isfinite(y)
            & (~stoppedByOutside)
            & (~stoppedByLimiter)
        )
        if not all_released_from_start:
            ok_init &= release_times_arr <= (T[0] + 1e-12)
        ok_init &= (~stoppedByDry)

        if np.any(ok_init):
            idx_ok = np.where(ok_init)[0]
            tidI = tidPrev[idx_ok]
            sampled_init = _sample_fields_at_indices(idx_ok, ("zb", "wse"), x_arr=x, y_arr=y, tid_arr=tidI)
            zbI = sampled_init["zb"]
            wseI = sampled_init["wse"]

            z_here = z[idx_ok].copy()
            badz = ~np.isfinite(z_here)

            if np.any(badz):
                z_here[badz] = zbI[badz] + zFrac * (wseI[badz] - zbI[badz])

            z_here = np.minimum(z_here, wseI)
            z_here = np.maximum(z_here, zbI)
            z[idx_ok] = z_here

        if depositedAlive:
            idx_dep0 = np.where(isDeposited & (tidPrev >= 0))[0]
            if idx_dep0.size:
                z[idx_dep0] = _sample_fields_at_indices(idx_dep0, ("zb",), x_arr=x, y_arr=y)["zb"]

        idx_surf0 = np.where(isSurfaced & (tidPrev >= 0))[0]
        if idx_surf0.size:
            z[idx_surf0] = _sample_fields_at_indices(idx_surf0, ("wse",), x_arr=x, y_arr=y)["wse"]

    # =========================================================
    # initial residence count
    # =========================================================
    ok_tid_init = (
        (tidPrev >= 0)
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (~stoppedByOutside)
        & (~stoppedByLimiter)
    )
    if not all_released_from_start:
        ok_tid_init &= release_times_arr <= (T[0] + 1e-12)

    if not depositedAlive:
        ok_tid_init &= (~isDeposited)
    ok_tid_init &= (~stoppedByDry)

    if countResidence:
        tid_init = tidPrev[ok_tid_init]
        if tid_init.size:
            cellCount += np.bincount(tid_init, minlength=Nc).astype(np.uint32)

    write_tracked(0, x, y, z, tidPrev)
    write_state_counts(0)
    _profile_increment(profile, "solver_steps", max(Nt - 1, 0))
    _profile_increment(profile, "particle_updates", Np * max(Nt - 1, 0))

    chunk_start_wall = perf_counter()
    chunk_start_step = 1
    chunk_crossing_candidates = 0
    chunk_crossings_resolved = 0
    chunk_outside_after_cross = 0
    chunk_wetdry_tangential = 0
    chunk_alive_move_peak = 0
    chunk_alive_move_sum = 0
    chunk_active_samples = 0

    def flush_progress_chunk(step_index: int) -> None:
        nonlocal chunk_start_wall
        nonlocal chunk_start_step
        nonlocal chunk_crossing_candidates
        nonlocal chunk_crossings_resolved
        nonlocal chunk_outside_after_cross
        nonlocal chunk_wetdry_tangential
        nonlocal chunk_alive_move_peak
        nonlocal chunk_alive_move_sum
        nonlocal chunk_active_samples

        if not progress_chunk_profile or step_index < chunk_start_step:
            return

        chunk_elapsed_s = perf_counter() - chunk_start_wall
        n_steps = step_index - chunk_start_step + 1
        avg_alive_move = (
            chunk_alive_move_sum / chunk_active_samples
            if chunk_active_samples > 0
            else 0.0
        )
        pct_end = 100.0 * step_index / (Nt - 1) if Nt > 1 else 100.0
        pct_start = 100.0 * (chunk_start_step - 1) / (Nt - 1) if Nt > 1 else 0.0
        print(
            "  Chunk "
            f"{pct_start:5.1f}%->{pct_end:5.1f}%: "
            f"{chunk_elapsed_s:.3f} s total, "
            f"{(1000.0 * chunk_elapsed_s / max(n_steps, 1)):.3f} ms/step, "
            f"alive_move avg {avg_alive_move:.0f}, peak {chunk_alive_move_peak}, "
            f"crossings {chunk_crossings_resolved}/{chunk_crossing_candidates}, "
            f"outside_after_cross {chunk_outside_after_cross}, "
            f"wetdry_tangential {chunk_wetdry_tangential}",
            flush=True,
        )

        chunk_start_wall = perf_counter()
        chunk_start_step = step_index + 1
        chunk_crossing_candidates = 0
        chunk_crossings_resolved = 0
        chunk_outside_after_cross = 0
        chunk_wetdry_tangential = 0
        chunk_alive_move_peak = 0
        chunk_alive_move_sum = 0
        chunk_active_samples = 0

    # =========================================================
    # main loop
    # =========================================================
    last_completed_step = 0
    for n in range(1, Nt):
        print_progress(n)
        t_now = T[n]
        update_particle_evolution_state(t_now)

        # -----------------------------------------------------
        # optional re-entrainment of deposited particles
        # -----------------------------------------------------
        t_section = perf_counter()
        if depositedAlive:
            can_check_dep = (
                isDeposited
                & (~stoppedByLimiter)
                & (~stoppedByOutside)
                & (~stoppedByDry)
                & (tidPrev >= 0)
            )

            idx_dep = np.where(can_check_dep)[0]
            if idx_dep.size:
                tid_dep = tidPrev[idx_dep]
                if biofouling_enabled and d50_c is not None:
                    bed_crit_local = current_bed_crit_for_indices(idx_dep, tid_dep)
                    can_reentrain_local = current_can_reentrain_for_indices(
                        idx_dep,
                        tid_dep,
                        x_arr=x,
                        y_arr=y,
                    )
                    ustar_crit_for_pickup = bed_crit_local
                else:
                    if field_sampler is None:
                        can_reentrain_local = can_reentrain_c
                    else:
                        can_reentrain_local = current_can_reentrain_for_indices(
                            idx_dep,
                            tid_dep,
                            x_arr=x,
                            y_arr=y,
                        )
                    ustar_crit_for_pickup = ustarCrit_c

                pickup = resolve_bed_entrainment(
                    bed_policy=bedPolicy,
                    tid=tid_dep,
                    wet_cell=wet_cell,
                    can_reentrain=can_reentrain_local,
                    ustar_c=ustar_c,
                    ustar_crit_c=ustar_crit_for_pickup,
                    rng=rng,
                    entrainment_prob_mode=entrainmentProbMode,
                    entrainment_prob_sigma_star=entrainmentProbSigmaStar,
                    entrainment_prob_k=entrainmentProbK,
                )

                if np.any(pickup):
                    id_pick = idx_dep[pickup]

                    isDeposited[id_pick] = False
                    tDeposit[id_pick] = np.nan

                    isSurfaced[id_pick] = False
                    z[id_pick] = _sample_fields_at_indices(id_pick, ("zb",), x_arr=x, y_arr=y)["zb"] + z_eps_pickup
        _profile_add(profile, "reentrainment_s", perf_counter() - t_section)

        t_section = perf_counter()
        can_check_surf = (
            isSurfaced
            & (~stoppedByLimiter)
            & (~stoppedByOutside)
            & (~stoppedByDry)
            & (tidPrev >= 0)
        )
        idx_surf = np.where(can_check_surf)[0]
        if idx_surf.size:
            tid_surf = tidPrev[idx_surf]
            if biofouling_enabled:
                surface_crit_local = current_surface_crit_for_indices(idx_surf)
                can_detach_local = current_can_detach_surface_for_indices(
                    idx_surf,
                    tid_surf,
                    x_arr=x,
                    y_arr=y,
                )
                surface_crit_for_detach = surface_crit_local
            else:
                if field_sampler is None:
                    can_detach_local = can_detach_surface_c
                else:
                    can_detach_local = current_can_detach_surface_for_indices(
                        idx_surf,
                        tid_surf,
                        x_arr=x,
                        y_arr=y,
                    )
                surface_crit_for_detach = surfaceUstarCrit_c

            detach = resolve_surface_detachment(
                surface_policy=surfacePolicy,
                tid=tid_surf,
                wet_cell=wet_cell,
                can_detach_surface=can_detach_local,
                ustar_c=ustar_c,
                surface_ustar_crit_c=surface_crit_for_detach,
                rng=rng,
                entrainment_prob_mode=surfaceDetachmentProbMode,
                entrainment_prob_sigma_star=surfaceDetachmentProbSigmaStar,
                entrainment_prob_k=surfaceDetachmentProbK,
            )

            if np.any(detach):
                id_detach = idx_surf[detach]
                tid_detach = tidPrev[id_detach]
                isSurfaced[id_detach] = False
                sampled_detach = _sample_fields_at_indices(
                    id_detach,
                    ("zb", "wse"),
                    x_arr=x,
                    y_arr=y,
                    tid_arr=tid_detach,
                )
                z[id_detach] = np.maximum(sampled_detach["zb"], sampled_detach["wse"] - z_eps_pickup)
        _profile_add(profile, "surface_detachment_s", perf_counter() - t_section)

        x_prev = x
        y_prev = y
        z_prev = z
        tidAtPn = tidPrev.copy()

        alive_count = (
            np.isfinite(x_prev)
            & np.isfinite(y_prev)
            & np.isfinite(z_prev)
            & (tidPrev >= 0)
        )

        dead_terminal = stoppedByLimiter | stoppedByOutside | stoppedByDry
        if not depositedAlive:
            dead_terminal |= isDeposited

        # Residence counts presence in a cell, including deposited particles
        # when depositedAlive=True. Surface-stuck particles are also counted.
        alive_for_residence = alive_count & (~dead_terminal)
        alive_move = alive_for_residence & (~isDeposited)
        alive_move_count = int(np.count_nonzero(alive_move))
        chunk_alive_move_peak = max(chunk_alive_move_peak, alive_move_count)
        chunk_alive_move_sum += alive_move_count
        chunk_active_samples += 1

        if not np.any(alive_count):
            if not all_released_from_start:
                activate_particles_between(T[n - 1], t_now, x, y, z)
            write_tracked(n, x, y, z, tidPrev)
            write_state_counts(n)
            last_completed_step = n
            continue

        # -----------------------------------------------------
        # count residence
        # -----------------------------------------------------
        t_section = perf_counter()
        if countResidence:
            idxC = np.where(alive_for_residence)[0]
            if idxC.size:
                tidC = tidPrev[idxC]
                tidC = tidC[tidC >= 0]
                if tidC.size:
                    cellCount += np.bincount(tidC, minlength=Nc).astype(np.uint32)
        _profile_add(profile, "residence_count_s", perf_counter() - t_section)

        x_new = x.copy()
        y_new = y.copy()
        z_new = z.copy()

        if depositedAlive:
            idx_dep_hold = np.where(isDeposited & (tidPrev >= 0) & (~dead_terminal))[0]
            if idx_dep_hold.size:
                z_new[idx_dep_hold] = _sample_fields_at_indices(
                    idx_dep_hold,
                    ("zb",),
                    x_arr=x_new,
                    y_arr=y_new,
                )["zb"]

        idx_surf_hold = np.where(
            isSurfaced & (tidPrev >= 0) & (~stoppedByLimiter) & (~stoppedByOutside)
        )[0]
        if idx_surf_hold.size:
            z_new[idx_surf_hold] = _sample_fields_at_indices(
                idx_surf_hold,
                ("wse",),
                x_arr=x_new,
                y_arr=y_new,
            )["wse"]

        if not np.any(alive_move):
            x, y, z = x_new, y_new, z_new
            write_tracked(n, x, y, z, tidPrev)
            write_state_counts(n)
            last_completed_step = n
            continue

        # -----------------------------------------------------
        # horizontal deterministic move + optional RW
        # -----------------------------------------------------
        t_section = perf_counter()
        t_subsection = perf_counter()
        idx_alive = np.where(alive_move)[0]
        tidA = tidPrev[idx_alive]
        valid_tid = tidA >= 0

        u = np.full(idx_alive.size, np.nan, dtype=float)
        v = np.full(idx_alive.size, np.nan, dtype=float)
        h = np.full(idx_alive.size, np.nan, dtype=float)

        if np.any(valid_tid):
            t = tidA[valid_tid]
            xy_valid = np.empty((np.count_nonzero(valid_tid), 2), dtype=float)
            xy_valid[:, 0] = x_prev[idx_alive[valid_tid]]
            xy_valid[:, 1] = y_prev[idx_alive[valid_tid]]

            if field_sampler is not None:
                sampled_start = field_sampler.sample_fields(
                    t,
                    xy_valid,
                    ("h", "u", "v", "ustar", "zb", "ks", "z0"),
                )
                h_valid = sampled_start["h"]
                ubar_start = sampled_start["u"]
                vbar_start = sampled_start["v"]
                ustar_start = sampled_start["ustar"]
                zb_start = sampled_start["zb"]
                ks_start = sampled_start["ks"]
                z0_start = sampled_start["z0"]
            else:
                h_valid = h_c[t]
                ubar_start = u_c[t]
                vbar_start = v_c[t]
                ustar_start = ustar_c[t]
                zb_start = zb_c[t]
                ks_start = None if ks_c is None else ks_c[t]
                z0_start = None if z0_c is None else z0_c[t]
            h[valid_tid] = h_valid

            if transportVelocityMode == "depth_averaged":
                u_samp = ubar_start
                v_samp = vbar_start
            else:
                ks_loc = None
                if transportVelocityMode == "loglaw_vertical":
                    ks_loc = ks_start

                u_samp, v_samp = _sample_transport_velocity(
                    ubar=ubar_start,
                    vbar=vbar_start,
                    ustar=ustar_start,
                    h=h_valid,
                    zb=zb_start,
                    zp=z_prev[idx_alive[valid_tid]],
                    ks=ks_loc,
                    z0=z0_start,
                    mode=transportVelocityMode,
                    hydraulic_closure_model=hydraulicClosureModel,
                    z_frac=zFrac,
                )
            u[valid_tid] = u_samp
            v[valid_tid] = v_samp
        _profile_add(profile, "horizontal_velocity_sampling_s", perf_counter() - t_subsection)

        good = valid_tid & np.isfinite(u) & np.isfinite(v)
        is_wet_here = _sample_is_wet_at_indices(idx_alive, x_arr=x_prev, y_arr=y_prev)
        wet_move = good & is_wet_here

        idx_bad = idx_alive[~good]
        if idx_bad.size:
            apply_outside_stop(
                idx_bad,
                t_now=t_now,
                x_new=x_new,
                y_new=y_new,
                z_new=z_new,
                tid_prev=tidPrev,
                x_prev=x_prev,
                y_prev=y_prev,
                z_prev=z_prev,
                tid_at_prev=tidAtPn,
                stopped_by_outside=stoppedByOutside,
                t_stop_outside=tStopOutside,
            )

        idx_dry_start = idx_alive[good & ~wet_move]
        if idx_dry_start.size:
            if dryPolicy == "stop":
                first_dry = np.isnan(tStopDry[idx_dry_start])
                if np.any(first_dry):
                    id_first = idx_dry_start[first_dry]
                    xDryHit[id_first] = x_prev[id_first]
                    yDryHit[id_first] = y_prev[id_first]
                    dryStopAtStepStart[id_first] = True
                apply_dry_stop(
                    idx_dry_start,
                    t_now=t_now,
                    x_new=x_new,
                    y_new=y_new,
                    z_new=z_new,
                    tid_prev=tidPrev,
                    x_prev=x_prev,
                    y_prev=y_prev,
                    z_prev=z_prev,
                    tid_at_prev=tidAtPn,
                    stopped_by_dry=stoppedByDry,
                    t_stop_dry=tStopDry,
                )

        idx_wet = idx_alive[wet_move]
        if idx_wet.size:
            tidW0 = tidPrev[idx_wet]

            # -------------------------------------------------
            # 1) start-of-step sampled transport velocity
            # -------------------------------------------------
            uw0 = u[wet_move]
            vw0 = v[wet_move]

            # midpoint predictor
            x_mid = x_prev[idx_wet] + 0.5 * dt * uw0
            y_mid = y_prev[idx_wet] + 0.5 * dt * vw0
            z_mid = z_prev[idx_wet]

            # midpoint cell location: keep particles in their current cell
            # on the cheap, and only do a broader search for crossings.
            tidW = tidW0.copy()
            xy_mid = np.empty((idx_wet.size, 2), dtype=float)
            xy_mid[:, 0] = x_mid
            xy_mid[:, 1] = y_mid

            t_subsection = perf_counter()
            inside_mid = _contains_points_in_cells(tidW0, xy_mid)
            ok_mid = inside_mid.copy()

            if np.any(~inside_mid):
                mid_cross = np.where(~inside_mid)[0]
                tid_mid = mesh.point_location_local(
                    x_mid[mid_cross],
                    y_mid[mid_cross],
                    tidW0[mid_cross],
                )
                _profile_increment(profile, "mesh_point_location_calls", 1)
                ok_cross_mid = tid_mid >= 0
                if np.any(ok_cross_mid):
                    tidW[mid_cross[ok_cross_mid]] = tid_mid[ok_cross_mid]
                    ok_mid[mid_cross[ok_cross_mid]] = True
            _profile_add(profile, "horizontal_midpoint_mesh_s", perf_counter() - t_subsection)

            # -------------------------------------------------
            # 2) midpoint sampled transport velocity
            # -------------------------------------------------
            uw = uw0.copy()
            vw = vw0.copy()
            Kpar_mid = None
            Kperp_mid = None

            if np.any(ok_mid):
                t_subsection = perf_counter()
                xy_mid_ok = xy_mid[ok_mid]
                if transportVelocityMode == "depth_averaged":
                    if field_sampler is not None:
                        midpoint_names = ("u", "v", "kpar", "kperp") if (useRWx or useRWy) else ("u", "v")
                        sampled_mid = field_sampler.sample_fields(tidW[ok_mid], xy_mid_ok, midpoint_names)
                        uw_mid = sampled_mid["u"]
                        vw_mid = sampled_mid["v"]
                        if useRWx or useRWy:
                            Kpar_mid = np.zeros(idx_wet.size, dtype=float)
                            Kperp_mid = np.zeros(idx_wet.size, dtype=float)
                            Kpar_mid[ok_mid] = np.where(np.isfinite(sampled_mid["kpar"]), sampled_mid["kpar"], 0.0)
                            Kperp_mid[ok_mid] = np.where(np.isfinite(sampled_mid["kperp"]), sampled_mid["kperp"], 0.0)
                            if np.any(~ok_mid):
                                sampled_mid_rw = field_sampler.sample_fields(
                                    tidW[~ok_mid],
                                    xy_mid[~ok_mid],
                                    ("kpar", "kperp"),
                                )
                                Kpar_mid[~ok_mid] = np.where(
                                    np.isfinite(sampled_mid_rw["kpar"]),
                                    sampled_mid_rw["kpar"],
                                    0.0,
                                )
                                Kperp_mid[~ok_mid] = np.where(
                                    np.isfinite(sampled_mid_rw["kperp"]),
                                    sampled_mid_rw["kperp"],
                                    0.0,
                                )
                    else:
                        uw_mid = u_c[tidW[ok_mid]]
                        vw_mid = v_c[tidW[ok_mid]]
                else:
                    if field_sampler is not None:
                        midpoint_names = ("u", "v", "ustar", "h", "zb", "ks", "z0")
                        if useRWx or useRWy:
                            midpoint_names = midpoint_names + ("kpar", "kperp")
                        sampled_mid = field_sampler.sample_fields(
                            tidW[ok_mid],
                            xy_mid_ok,
                            midpoint_names,
                        )
                        ks_mid = sampled_mid["ks"] if transportVelocityMode == "loglaw_vertical" else None
                        ubar_mid = sampled_mid["u"]
                        vbar_mid = sampled_mid["v"]
                        ustar_mid = sampled_mid["ustar"]
                        h_mid = sampled_mid["h"]
                        zb_mid = sampled_mid["zb"]
                        z0_mid = sampled_mid["z0"]
                        if useRWx or useRWy:
                            Kpar_mid = np.zeros(idx_wet.size, dtype=float)
                            Kperp_mid = np.zeros(idx_wet.size, dtype=float)
                            Kpar_mid[ok_mid] = np.where(np.isfinite(sampled_mid["kpar"]), sampled_mid["kpar"], 0.0)
                            Kperp_mid[ok_mid] = np.where(np.isfinite(sampled_mid["kperp"]), sampled_mid["kperp"], 0.0)
                            if np.any(~ok_mid):
                                sampled_mid_rw = field_sampler.sample_fields(
                                    tidW[~ok_mid],
                                    xy_mid[~ok_mid],
                                    ("kpar", "kperp"),
                                )
                                Kpar_mid[~ok_mid] = np.where(
                                    np.isfinite(sampled_mid_rw["kpar"]),
                                    sampled_mid_rw["kpar"],
                                    0.0,
                                )
                                Kperp_mid[~ok_mid] = np.where(
                                    np.isfinite(sampled_mid_rw["kperp"]),
                                    sampled_mid_rw["kperp"],
                                    0.0,
                                )
                    else:
                        ks_mid = None
                        if transportVelocityMode == "loglaw_vertical":
                            ks_mid = ks_c[tidW[ok_mid]]
                        ubar_mid = u_c[tidW[ok_mid]]
                        vbar_mid = v_c[tidW[ok_mid]]
                        ustar_mid = ustar_c[tidW[ok_mid]]
                        h_mid = h_c[tidW[ok_mid]]
                        zb_mid = zb_c[tidW[ok_mid]]
                        z0_mid = None if z0_c is None else z0_c[tidW[ok_mid]]

                    uw_mid, vw_mid = _sample_transport_velocity(
                        ubar=ubar_mid,
                        vbar=vbar_mid,
                        ustar=ustar_mid,
                        h=h_mid,
                        zb=zb_mid,
                        zp=z_mid[ok_mid],
                        ks=ks_mid,
                        z0=z0_mid,
                        mode=transportVelocityMode,
                        hydraulic_closure_model=hydraulicClosureModel,
                        z_frac=zFrac,
                    )
                uw[ok_mid] = uw_mid
                vw[ok_mid] = vw_mid
                _profile_add(profile, "horizontal_midpoint_velocity_s", perf_counter() - t_subsection)

            # deterministic full step with midpoint velocity
            x_new[idx_wet] = x_prev[idx_wet] + dt * uw
            y_new[idx_wet] = y_prev[idx_wet] + dt * vw

            # -------------------------------------------------
            # 3) stochastic horizontal transport
            # -------------------------------------------------
            if useRWx or useRWy:
                t_subsection = perf_counter()
                if Kpar_mid is not None and Kperp_mid is not None:
                    Kpar = Kpar_mid
                    Kperp = Kperp_mid
                elif field_sampler is not None:
                    sampled_rw = field_sampler.sample_fields(tidW, xy_mid, ("kpar", "kperp"))
                    Kpar = np.where(np.isfinite(sampled_rw["kpar"]), sampled_rw["kpar"], 0.0)
                    Kperp = np.where(np.isfinite(sampled_rw["kperp"]), sampled_rw["kperp"], 0.0)
                else:
                    Kpar = Kpar_c[tidW]
                    Kperp = Kperp_c[tidW]

                Umag = np.hypot(uw, vw)
                ex = np.zeros_like(uw)
                ey = np.zeros_like(vw)
                goodU = np.isfinite(Umag) & (Umag > 1e-12)
                ex[goodU] = uw[goodU] / Umag[goodU]
                ey[goodU] = vw[goodU] / Umag[goodU]
                if np.any(~goodU):
                    ex[~goodU] = 1.0
                    ey[~goodU] = 0.0

                nx = -ey
                ny = ex

                if transportModel == "random_walk":
                    sigPar = np.sqrt(two_dt * Kpar)
                    sigPerp = np.sqrt(two_dt * Kperp)

                    xiPar = rng.standard_normal(idx_wet.size)
                    xiPerp = rng.standard_normal(idx_wet.size)

                    dx = sigPar * xiPar * ex + sigPerp * xiPerp * nx
                    dy = sigPar * xiPar * ey + sigPerp * xiPerp * ny

                    if np.any(~goodU):
                        rr = np.where(~goodU)[0]

                        if useRWx and useRWy:
                            sigIso = np.sqrt(two_dt * Kh_c[tidW[rr]])
                            dx[rr] = sigIso * rng.standard_normal(rr.size)
                            dy[rr] = sigIso * rng.standard_normal(rr.size)

                        elif useRWx and (not useRWy):
                            sigIso = np.sqrt(two_dt * Kpar[rr])
                            dx[rr] = sigIso * rng.standard_normal(rr.size)
                            dy[rr] = 0.0

                        elif useRWy and (not useRWx):
                            sigIso = np.sqrt(two_dt * Kperp[rr])
                            dx[rr] = 0.0
                            dy[rr] = sigIso * rng.standard_normal(rr.size)

                        else:
                            dx[rr] = 0.0
                            dy[rr] = 0.0

                    x_new[idx_wet] += dx
                    y_new[idx_wet] += dy
                else:
                    sigma_par = np.sqrt(np.maximum(0.0, np.where(useRWx, Kpar, 0.0) / float(TL_horizontal)))
                    sigma_perp = np.sqrt(np.maximum(0.0, np.where(useRWy, Kperp, 0.0) / float(TL_horizontal)))

                    upar_new = _langevin_ou_step(ufluc_par[idx_wet], sigma_par, TL_horizontal, dt, rng)
                    uperp_new = _langevin_ou_step(ufluc_perp[idx_wet], sigma_perp, TL_horizontal, dt, rng)
                    ufluc_par[idx_wet] = upar_new
                    ufluc_perp[idx_wet] = uperp_new

                    x_new[idx_wet] += dt * (upar_new * ex + uperp_new * nx)
                    y_new[idx_wet] += dt * (upar_new * ey + uperp_new * ny)
                _profile_add(profile, "horizontal_stochastic_s", perf_counter() - t_subsection)

            t_subsection = perf_counter()
            tid_wet = tidPrev[idx_wet]
            xy_wet = np.empty((idx_wet.size, 2), dtype=float)
            xy_wet[:, 0] = x_new[idx_wet]
            xy_wet[:, 1] = y_new[idx_wet]

            inside_same = _contains_points_in_cells(tid_wet, xy_wet)

            idx_cross = idx_wet[~inside_same]
            chunk_crossing_candidates += int(idx_cross.size)
            if idx_cross.size:
                tid_cross = mesh.point_location_local(
                    x_new[idx_cross],
                    y_new[idx_cross],
                    tid_wet[~inside_same],
                )
                _profile_increment(profile, "mesh_point_location_calls", 1)
                ok_cross = tid_cross >= 0
                chunk_crossings_resolved += int(np.count_nonzero(ok_cross))
                chunk_outside_after_cross += int(np.count_nonzero(~ok_cross))

                if np.any(~ok_cross):
                    idx_out = idx_cross[~ok_cross]
                    apply_outside_stop(
                        idx_out,
                        t_now=t_now,
                        x_new=x_new,
                        y_new=y_new,
                        z_new=z_new,
                        tid_prev=tidPrev,
                        x_prev=x_prev,
                        y_prev=y_prev,
                        z_prev=z_prev,
                        tid_at_prev=tidAtPn,
                        stopped_by_outside=stoppedByOutside,
                        t_stop_outside=tStopOutside,
                    )

                idx_in = idx_cross[ok_cross]
                tid_in = tid_cross[ok_cross]

                if idx_in.size:
                    is_wet_in = _sample_is_wet_at_indices(
                        idx_in,
                        x_arr=x_new,
                        y_arr=y_new,
                        tid_arr=tid_in,
                    )
                    dry2 = ~is_wet_in

                    if np.any(dry2):
                        idx_dry = idx_in[dry2]
                        if dryPolicy == "tangential":
                            chunk_wetdry_tangential += int(idx_dry.size)

                        if dryPolicy in ("stop", "stick_active"):
                            Pprop_stop = np.empty((idx_dry.size, 2), dtype=float)
                            Pdet_stop = np.empty((idx_dry.size, 2), dtype=float)
                            Pprop_stop[:, 0] = x_new[idx_dry]
                            Pprop_stop[:, 1] = y_new[idx_dry]
                            Pdet_stop[:, 0] = x_prev[idx_dry]
                            Pdet_stop[:, 1] = y_prev[idx_dry]
                            z_prop_stop = z_new[idx_dry].copy()
                            t_wetdry = perf_counter()
                            Pcontact_stop, tid_contact_stop, alpha_contact_all = wet_dry_stop_contact_points(
                                Pprop_sub=Pprop_stop,
                                Pdet_sub=Pdet_stop,
                                mesh=mesh,
                                zb_c=zb_c,
                                wse_c=wse_c,
                                hmin=hmin,
                                tid_prev_sub=tidAtPn[idx_dry],
                                tid_dry_sub=tid_in[dry2],
                                wet_cell_mask=wet_cell,
                            )
                            if dryPolicy == "stick_active":
                                _profile_add(profile, "wetdry_stick_active_s", perf_counter() - t_wetdry)
                            else:
                                _profile_add(profile, "wetdry_stop_contact_s", perf_counter() - t_wetdry)
                            if dryPolicy == "stop":
                                first_dry = np.isnan(tStopDry[idx_dry])
                                if np.any(first_dry):
                                    id_first = idx_dry[first_dry]
                                    xDryHit[id_first] = Pcontact_stop[first_dry, 0]
                                    yDryHit[id_first] = Pcontact_stop[first_dry, 1]
                                    dryStopAtCrossing[id_first] = True
                                apply_dry_stop(
                                    idx_dry,
                                    t_now=t_now,
                                    x_new=x_new,
                                    y_new=y_new,
                                    z_new=z_new,
                                    tid_prev=tidPrev,
                                    x_prev=x_prev,
                                    y_prev=y_prev,
                                    z_prev=z_prev,
                                    tid_at_prev=tidAtPn,
                                    stopped_by_dry=stoppedByDry,
                                    t_stop_dry=tStopDry,
                                )
                            else:
                                first_dry_contact = np.isnan(xDryHit[idx_dry]) | np.isnan(yDryHit[idx_dry])
                                if np.any(first_dry_contact):
                                    id_first = idx_dry[first_dry_contact]
                                    xDryHit[id_first] = Pcontact_stop[first_dry_contact, 0]
                                    yDryHit[id_first] = Pcontact_stop[first_dry_contact, 1]
                            contact_valid = tid_contact_stop >= 0
                            if np.any(contact_valid):
                                idx_contact = idx_dry[contact_valid]
                                x_new[idx_contact] = Pcontact_stop[contact_valid, 0]
                                y_new[idx_contact] = Pcontact_stop[contact_valid, 1]
                                tidPrev[idx_contact] = tid_contact_stop[contact_valid]
                                alpha_contact = np.where(
                                    np.isfinite(alpha_contact_all[contact_valid]),
                                    alpha_contact_all[contact_valid],
                                    0.0,
                                )
                                z_new[idx_contact] = (
                                    z_prev[idx_contact]
                                    + alpha_contact * (z_prop_stop[contact_valid] - z_prev[idx_contact])
                                )

                        else:
                            Pprop_sub = np.empty((idx_dry.size, 2), dtype=float)
                            Pdet_sub = np.empty((idx_dry.size, 2), dtype=float)

                            Pprop_sub[:, 0] = x_new[idx_dry]
                            Pprop_sub[:, 1] = y_new[idx_dry]
                            Pdet_sub[:, 0] = x_prev[idx_dry]
                            Pdet_sub[:, 1] = y_prev[idx_dry]

                            t_wetdry = perf_counter()
                            Pcorr_sub = tangential_slide_wetdry_subset_fullcache(
                                Pprop_sub=Pprop_sub,
                                Pdet_sub=Pdet_sub,
                                mesh=mesh,
                                u_c=u_c,
                                v_c=v_c,
                                zb_c=zb_c,
                                wse_c=wse_c,
                                hmin=hmin,
                                dt=dt,
                                tid_prev_sub=tidAtPn[idx_dry],
                            )
                            _profile_add(profile, "wetdry_tangential_s", perf_counter() - t_wetdry)

                            x_new[idx_dry] = Pcorr_sub[:, 0]
                            y_new[idx_dry] = Pcorr_sub[:, 1]
                            z_new[idx_dry] = z_prev[idx_dry]

                            tid_ref = mesh.point_location_local(
                                x_new[idx_dry],
                                y_new[idx_dry],
                                tidAtPn[idx_dry],
                            )
                            _profile_increment(profile, "mesh_point_location_calls", 1)
                            ok_ref = tid_ref >= 0

                            if np.any(~ok_ref):
                                idx_bad_ref = idx_dry[~ok_ref]
                                apply_outside_stop(
                                    idx_bad_ref,
                                    t_now=t_now,
                                    x_new=x_new,
                                    y_new=y_new,
                                    z_new=z_new,
                                    tid_prev=tidPrev,
                                    x_prev=x_prev,
                                    y_prev=y_prev,
                                    z_prev=z_prev,
                                    tid_at_prev=tidAtPn,
                                    stopped_by_outside=stoppedByOutside,
                                    t_stop_outside=tStopOutside,
                                )

                            if np.any(ok_ref):
                                tidPrev[idx_dry[ok_ref]] = tid_ref[ok_ref]

                    idx_move = idx_in[~dry2]
                    if idx_move.size:
                        tidPrev[idx_move] = tid_in[~dry2]
            _profile_add(profile, "horizontal_crossing_s", perf_counter() - t_subsection)
        _profile_add(profile, "horizontal_transport_s", perf_counter() - t_section)

        # -----------------------------------------------------
        # uphill limiter
        # -----------------------------------------------------
        t_section = perf_counter()
        if uphillPolicy != "off":
            cand = (
                np.isfinite(x_new)
                & np.isfinite(y_new)
                & np.isfinite(z_new)
                & (tidAtPn >= 0)
                & (tidPrev >= 0)
                & (~isDeposited)
                & (~stoppedByLimiter)
                & (~stoppedByOutside)
                & (~stoppedByDry)
            )

            idxL = np.where(cand)[0]
            if idxL.size:
                zb_old = zb_c[tidAtPn[idxL]]
                zb_new_cell = zb_c[tidPrev[idxL]]
                dzUp = zb_new_cell - zb_old

                hit = np.isfinite(dzUp) & (dzUp > dzUpMax)
                if np.any(hit):
                    idx_hit = idxL[hit]
                    limiterHitEver[idx_hit] = True
                    stoppedByLimiter[idx_hit] = True

                    new = np.isnan(tStopLimiter[idx_hit])
                    tStopLimiter[idx_hit[new]] = t_now

                    x_new[idx_hit] = x_prev[idx_hit]
                    y_new[idx_hit] = y_prev[idx_hit]
                    z_new[idx_hit] = z_prev[idx_hit]
                    tidPrev[idx_hit] = tidAtPn[idx_hit]
        _profile_add(profile, "uphill_limiter_s", perf_counter() - t_section)

        # -----------------------------------------------------
        # vertical update
        # -----------------------------------------------------
        t_section = perf_counter()
        alive_v = (
            np.isfinite(x_new)
            & np.isfinite(y_new)
            & np.isfinite(z_new)
            & (tidPrev >= 0)
            & (~isDeposited)
            & (~isSurfaced)
            & (~stoppedByLimiter)
            & (~stoppedByOutside)
            & (~stoppedByDry)
        )

        idxV = np.where(alive_v)[0]
        if idxV.size:
            m = idxV.size

            zloc = z_new[idxV].copy()

            # `alive_v` already guarantees valid horizontal-stage cell ids, so
            # the vertical update can stay entirely on cached cell-centered data.
            tidZ = tidPrev[idxV].copy()
            sampled_vertical = _sample_fields_at_indices(
                idxV,
                ("zb", "wse", "ustar"),
                x_arr=x_new,
                y_arr=y_new,
                tid_arr=tidZ,
            )
            zBed = sampled_vertical["zb"]
            zSurf = sampled_vertical["wse"]
            ustar_vertical = sampled_vertical["ustar"]
            depthV = zSurf - zBed
            wetV = _sample_is_wet_at_indices(idxV, x_arr=x_new, y_arr=y_new, tid_arr=tidZ)

            under_jump = wetV & (zloc < zBed)
            zloc[under_jump] = zBed[under_jump]

            over_jump = wetV & (zloc > zSurf)
            zloc[over_jump] = zSurf[over_jump]

            # deterministic settling
            zTrial = zloc.copy()
            if np.any(wetV):
                ws_dt_local = current_ws[idxV[wetV]] * dt
                zTrial[wetV] = zloc[wetV] - ws_dt_local

            # stochastic vertical transport
            if useRWz:
                KzLoc = np.zeros(m, dtype=float)
                dKz_dz = np.zeros(m, dtype=float)

                if np.any(wetV):
                    hLoc = depthV[wetV]
                    zLoc = zloc[wetV]
                    zbLoc = zBed[wetV]

                    zprime = zLoc - zbLoc
                    zprime = np.clip(zprime, 0.0, hLoc)

                    ustarLoc = ustar_vertical[wetV]
                    C = alphaKz * kappa * ustarLoc

                    KzTemp = C * zprime * (1.0 - zprime / hLoc)
                    KzTemp = np.clip(KzTemp, 0.0, KzMax)
                    KzLoc[wetV] = KzTemp

                    dKzTemp = C * (1.0 - 2.0 * zprime / hLoc)

                    at_bed = zprime <= 0.0
                    at_top = zprime >= hLoc
                    dKzTemp[at_bed | at_top] = 0.0

                    dKz_dz[wetV] = dKzTemp

                if transportModel == "random_walk":
                    zTrial[wetV] += dKz_dz[wetV] * dt

                    nn = np.count_nonzero(wetV)
                    if nn:
                        sigZ = np.sqrt(two_dt * KzLoc[wetV])
                        zTrial[wetV] += sigZ * rng.standard_normal(nn)
                else:
                    sigma_w = np.sqrt(np.maximum(0.0, KzLoc / float(TL_vertical)))
                    w_new = _langevin_ou_step(wfluc[idxV], sigma_w, TL_vertical, dt, rng)
                    wfluc[idxV] = w_new
                    zTrial[wetV] += dKz_dz[wetV] * dt + w_new[wetV] * dt

            # ---- surface boundary ----
            hitSurf = wetV & np.isfinite(zSurf) & (zTrial >= zSurf)
            if np.any(hitSurf):
                tid_hit = tidZ[hitSurf]
                j_hit = np.where(hitSurf)[0]
                if biofouling_enabled:
                    idx_hit_global = idxV[j_hit]
                    surface_crit_local = current_surface_crit_for_indices(idx_hit_global)
                    can_detach_local = current_can_detach_surface_for_indices(
                        idx_hit_global,
                        tid_hit,
                        x_arr=x_new,
                        y_arr=y_new,
                    )
                    surface_crit_for_contact = surface_crit_local
                else:
                    if field_sampler is None:
                        can_detach_local = can_detach_surface_c
                    else:
                        idx_hit_global = idxV[j_hit]
                        can_detach_local = current_can_detach_surface_for_indices(
                            idx_hit_global,
                            tid_hit,
                            x_arr=x_new,
                            y_arr=y_new,
                        )
                    surface_crit_for_contact = surfaceUstarCrit_c

                doReflect, doStick = resolve_surface_contact(
                    surface_policy=surfacePolicy,
                    tid=tid_hit,
                    wet_cell=wet_cell,
                    can_detach_surface=can_detach_local,
                    ustar_c=ustar_c,
                    surface_ustar_crit_c=surface_crit_for_contact,
                    rng=rng,
                    entrainment_prob_mode=surfaceDetachmentProbMode,
                    entrainment_prob_sigma_star=surfaceDetachmentProbSigmaStar,
                    entrainment_prob_k=surfaceDetachmentProbK,
                )

                if np.any(doReflect):
                    j_reflect = j_hit[doReflect]
                    zTrial[j_reflect] = 2.0 * zSurf[j_reflect] - zTrial[j_reflect]

                if np.any(doStick):
                    j_surf = j_hit[doStick]
                    id_surf = idxV[j_surf]
                    zTrial[j_surf] = zSurf[j_surf]
                    zloc[j_surf] = zSurf[j_surf]
                    first_hit = np.isnan(tSurfaceHit[id_surf])
                    if np.any(first_hit):
                        id_first = id_surf[first_hit]
                        tSurfaceHit[id_first] = t_now
                        xSurfaceHit[id_first] = x_new[id_first]
                        ySurfaceHit[id_first] = y_new[id_first]
                    isSurfaced[id_surf] = True

            # ---- bed boundary ----
            hitBed = wetV & np.isfinite(zBed) & (zTrial <= zBed)
            if np.any(hitBed):
                tid_hit = tidZ[hitBed]
                j_hit = np.where(hitBed)[0]
                if biofouling_enabled and d50_c is not None:
                    idx_hit_global = idxV[j_hit]
                    bed_crit_local = current_bed_crit_for_indices(idx_hit_global, tid_hit)
                    can_reentrain_local = current_can_reentrain_for_indices(
                        idx_hit_global,
                        tid_hit,
                        x_arr=x_new,
                        y_arr=y_new,
                    )
                    ustar_crit_for_contact = bed_crit_local
                else:
                    if field_sampler is None:
                        can_reentrain_local = can_reentrain_c
                    else:
                        idx_hit_global = idxV[j_hit]
                        can_reentrain_local = current_can_reentrain_for_indices(
                            idx_hit_global,
                            tid_hit,
                            x_arr=x_new,
                            y_arr=y_new,
                        )
                    ustar_crit_for_contact = ustarCrit_c

                doReflect, doDeposit = resolve_bed_contact(
                    bed_policy=bedPolicy,
                    tid=tid_hit,
                    wet_cell=wet_cell,
                    can_reentrain=can_reentrain_local,
                    ustar_c=ustar_c,
                    ustar_crit_c=ustar_crit_for_contact,
                    rng=rng,
                    entrainment_prob_mode=entrainmentProbMode,
                    entrainment_prob_sigma_star=entrainmentProbSigmaStar,
                    entrainment_prob_k=entrainmentProbK,
                )

                j_ref = j_hit[doReflect]
                j_dep = j_hit[doDeposit]

                if j_ref.size:
                    zTrial[j_ref] = 2.0 * zBed[j_ref] - zTrial[j_ref]

                if j_dep.size:
                    id_dep = idxV[j_dep]
                    tid_dep = tidZ[j_dep]

                    tidPrev[id_dep] = tid_dep
                    zTrial[j_dep] = zBed[j_dep]
                    zloc[j_dep] = zBed[j_dep]
                    isDeposited[id_dep] = True

                    new = np.isnan(tDeposit[id_dep])
                    tDeposit[id_dep[new]] = t_now

            dep_local = isDeposited[idxV]
            surf_local = isSurfaced[idxV]

            clamp_ok = wetV & (~dep_local) & (~surf_local) & np.isfinite(zTrial)
            if np.any(clamp_ok):
                zTrial[clamp_ok] = np.minimum(zTrial[clamp_ok], zSurf[clamp_ok])
                zTrial[clamp_ok] = np.maximum(zTrial[clamp_ok], zBed[clamp_ok])

            if np.any(dep_local):
                jD = np.where(dep_local)[0]
                idD = idxV[jD]
                ok_tidD = tidPrev[idD] >= 0
                if np.any(ok_tidD):
                    tD = tidPrev[idD[ok_tidD]]
                    sampled_dep = _sample_fields_at_indices(
                        idD[ok_tidD],
                        ("zb",),
                        x_arr=x_new,
                        y_arr=y_new,
                        tid_arr=tD,
                    )["zb"]
                    zloc[jD[ok_tidD]] = sampled_dep
                    zTrial[jD[ok_tidD]] = sampled_dep

            if np.any(surf_local):
                jS = np.where(surf_local)[0]
                idS = idxV[jS]
                ok_tidS = tidPrev[idS] >= 0
                if np.any(ok_tidS):
                    tS = tidPrev[idS[ok_tidS]]
                    sampled_surf = _sample_fields_at_indices(
                        idS[ok_tidS],
                        ("wse",),
                        x_arr=x_new,
                        y_arr=y_new,
                        tid_arr=tS,
                    )["wse"]
                    zloc[jS[ok_tidS]] = sampled_surf
                    zTrial[jS[ok_tidS]] = sampled_surf

            write_trial = (~dep_local) & (~surf_local) & np.isfinite(zTrial)
            zloc[write_trial] = zTrial[write_trial]

            z_new[idxV] = zloc
        _profile_add(profile, "vertical_transport_s", perf_counter() - t_section)

        # Re-snap surface-stuck particles to the free surface of their final cell
        t_section = perf_counter()
        idx_surf_final = np.where(
            isSurfaced
            & (tidPrev >= 0)
            & np.isfinite(x_new)
            & np.isfinite(y_new)
            & (~stoppedByLimiter)
            & (~stoppedByOutside)
            & (~stoppedByDry)
        )[0]
        if idx_surf_final.size:
            z_new[idx_surf_final] = _sample_fields_at_indices(
                idx_surf_final,
                ("wse",),
                x_arr=x_new,
                y_arr=y_new,
            )["wse"]
        _profile_add(profile, "surface_resnap_s", perf_counter() - t_section)

        # -----------------------------------------------------
        # commit
        # -----------------------------------------------------
        t_section = perf_counter()
        x, y, z = x_new, y_new, z_new
        if not all_released_from_start:
            activate_particles_between(T[n - 1], t_now, x, y, z)
        write_tracked(n, x, y, z, tidPrev)
        write_state_counts(n)
        last_completed_step = n
        _profile_add(profile, "commit_tracking_s", perf_counter() - t_section)
        if progress_chunk_profile and ((n % progEvery == 0) or (n == Nt - 1)):
            flush_progress_chunk(n)

        if _all_released_particles_done(t_now):
            print(
                "Stopping early because all released particles have already "
                "surfaced or otherwise terminated."
            )
            flush_progress_chunk(n)
            break

    # =========================================================
    # final diagnostics
    # =========================================================
    T = T[: last_completed_step + 1]
    n_tracked_time_samples = last_completed_step // trackOutputStride + 1
    Ttrack = Ttrack[:n_tracked_time_samples]
    Ptrack = Ptrack[:, :, :n_tracked_time_samples]
    tidTr_track = tidTr_track[:, :n_tracked_time_samples]
    stateCountHist = stateCountHist[:n_tracked_time_samples]
    cellResidenceTime = cellCount.astype(float) * dt
    n_time_samples = last_completed_step + 1
    if Np > 0 and n_time_samples > 0:
        cellOccupancyFraction = cellCount.astype(float) / (float(Np) * float(n_time_samples))
    else:
        cellOccupancyFraction = np.zeros_like(cellCount, dtype=float)

    flags = {
        "isDeposited": isDeposited,
        "isSurfaced": isSurfaced,
        "limiterHitEver": limiterHitEver,
        "stoppedByLimiter": stoppedByLimiter,
        "stoppedByOutside": stoppedByOutside,
        "stoppedByDry": stoppedByDry,
        "tStopLimiter": tStopLimiter,
        "tStopDry": tStopDry,
        "tStopOutside": tStopOutside,
        "tDeposit": tDeposit,
        "tSurfaceHit": tSurfaceHit,
        "xDryHit": xDryHit,
        "yDryHit": yDryHit,
        "dryStopAtStepStart": dryStopAtStepStart,
        "dryStopAtCrossing": dryStopAtCrossing,
        "xSurfaceHit": xSurfaceHit,
        "ySurfaceHit": ySurfaceHit,
    }

    transport_state = {
        "ufluc_par": ufluc_par,
        "ufluc_perp": ufluc_perp,
        "wfluc": wfluc,
    }

    xEnd = x
    yEnd = y
    zEnd = z
    tidEnd = tidPrev

    return (
        Ptrack,
        Ttrack,
        tidTr_track,
        track_idx,
        stateCountHist,
        xEnd,
        yEnd,
        zEnd,
        tidEnd,
        cellCount,
        cellResidenceTime,
        cellOccupancyFraction,
        flags,
        transport_state,
    )

