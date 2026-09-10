"""Transient hydraulic interpolation and segmented transient advection."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from core._04_transport_solver.steady import (
    _LocalCellIDWFieldSampler,
    advect_particles_euler_cell_fast_fullcache_3d_ustar,
)
from core._05_boundary_interaction import d50_from_chezy_loglaw, ustarcrit_cells_from_d50


def build_hydro_cache(
    adapter,
    particle,
    hmin,
    tanphi_ratio,
    rho_f,
    nu,
    g,
    tHyd0,
    dtHyd,
    tHydEnd,
    fixed_ustarcrit=True,
    cache_time_start=None,
    cache_time_end=None,
):
    """Build a frame-wise hydraulic cache for transient particle advection."""
    n_available_frames = adapter.last_index() + 1
    n_expected = int(round((tHydEnd - tHyd0) / dtHyd)) + 1

    if n_available_frames != n_expected:
        print(
            f"Warning: number of frames in file ({n_available_frames}) does not match "
            f"expected number from timing ({n_expected}). Using actual frames."
        )

    k_start = 0
    k_end = n_available_frames - 1
    if cache_time_start is not None or cache_time_end is not None:
        t_start = tHyd0 if cache_time_start is None else float(cache_time_start)
        t_end = tHydEnd if cache_time_end is None else float(cache_time_end)
        k_start = int(np.floor((t_start - tHyd0) / dtHyd + 1.0e-12))
        k_end = int(np.ceil((t_end - tHyd0) / dtHyd - 1.0e-12))
        k_start = int(np.clip(k_start, 0, n_available_frames - 1))
        k_end = int(np.clip(k_end, k_start, n_available_frames - 1))

    print(
        "Building transient hydraulic cache: "
        f"frames {k_start}..{k_end} ({k_end - k_start + 1} of {n_available_frames})"
    )

    hydro_cache = []
    ustarCrit_fixed = None

    def _has_finite_values(arr) -> bool:
        arr = np.asarray(arr, dtype=float)
        return bool(np.any(np.isfinite(arr)))

    for k in range(k_start, k_end + 1):
        frame = adapter.load_frame(k)

        if frame.Chezy_c is None or frame.ustar_c is None:
            raise RuntimeError(
                f"Chezy_c / ustar_c not found in frame {k}. "
                f"Make sure the adapter attaches helper fields."
            )

        U_c = np.asarray(frame.U_c, dtype=float).ravel()
        V_c = np.asarray(frame.V_c, dtype=float).ravel()
        h_c = np.asarray(frame.h_c, dtype=float).ravel()
        wse_c = np.asarray(frame.wse_c, dtype=float).ravel()
        Chezy_c = np.array(frame.Chezy_c, dtype=float, copy=True).ravel()
        ustar_c = np.array(frame.ustar_c, dtype=float, copy=True).ravel()

        ks_c = None
        if frame.ks_c is not None:
            ks_c = np.array(frame.ks_c, dtype=float, copy=True).ravel()
        z0_c = None
        if getattr(frame, "z0_c", None) is not None:
            z0_c = np.array(frame.z0_c, dtype=float, copy=True).ravel()

        Chezy_c[(~np.isfinite(Chezy_c)) | (Chezy_c <= 0)] = np.nan
        ustar_c[~np.isfinite(ustar_c)] = 0.0

        d50_c = d50_from_chezy_loglaw(Chezy_c, h_c, hmin, verbose=False)

        if fixed_ustarcrit and ustarCrit_fixed is not None:
            ustarCrit_c = ustarCrit_fixed
        else:
            ustarCrit_c = ustarcrit_cells_from_d50(
                d50_c=d50_c,
                Dp=particle.Dp,
                rho_p=particle.rho_p,
                beta_p=particle.beta_p,
                tanphi_ratio=tanphi_ratio,
                rho_w=rho_f,
                nu=nu,
                g=g,
                beta_s=1.5,
                c2=-0.6,
                u0=0.005,
                n_iter=30,
                tol=1e-8,
            )
            if fixed_ustarcrit and ustarCrit_fixed is None and _has_finite_values(ustarCrit_c):
                ustarCrit_fixed = ustarCrit_c.copy()

        hydro_cache.append(
            {
                "k": k,
                "time": tHyd0 + k * dtHyd,
                "step": frame.step,
                "U_c": U_c,
                "V_c": V_c,
                "h_c": h_c,
                "wse_c": wse_c,
                "Chezy_c": Chezy_c,
                "ks_c": ks_c,
                "z0_c": z0_c,
                "d50_c": d50_c,
                "ustar_c": ustar_c,
                "ustarCrit_c": ustarCrit_c,
            }
        )

    return hydro_cache


def _init_merged_flags(flags_template, Np):
    """Initialize merged transient flags using template dtypes."""
    out = {}
    for key, arr in flags_template.items():
        arr = np.asarray(arr)
        if arr.dtype == bool:
            out[key] = np.zeros(Np, dtype=bool)
        elif np.issubdtype(arr.dtype, np.integer):
            out[key] = np.full(Np, -1, dtype=int)
        else:
            out[key] = np.full(Np, np.nan, dtype=float)
    return out


def _sample_tracked_bed_surface(
    mesh,
    hydraulic_field_mode,
    points_xy,
    triangle_ids,
    zb_c,
    wse_c,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample bed and water surface at tracked particle positions."""
    points = np.asarray(points_xy, dtype=float)
    tid = np.asarray(triangle_ids, dtype=int).ravel()
    bed = np.full(tid.shape, np.nan, dtype=float)
    surface = np.full(tid.shape, np.nan, dtype=float)
    valid = (tid >= 0) & np.all(np.isfinite(points), axis=1)
    if not np.any(valid):
        return bed, surface

    mode = str(hydraulic_field_mode or "cellwise").strip().lower()
    if mode == "local_idw":
        sampler = _LocalCellIDWFieldSampler(mesh, {"zb": zb_c, "wse": wse_c})
        sampled = sampler.sample_fields(tid[valid], points[valid], ("zb", "wse"))
        bed[valid] = sampled["zb"]
        surface[valid] = sampled["wse"]
        return bed, surface

    zb_arr = np.asarray(zb_c, dtype=float).ravel()
    wse_arr = np.asarray(wse_c, dtype=float).ravel()
    in_range = valid & (tid < zb_arr.size) & (tid < wse_arr.size)
    bed[in_range] = zb_arr[tid[in_range]]
    surface[in_range] = wse_arr[tid[in_range]]
    return bed, surface


def _merge_flags(global_flags, seg_flags, history_offset):
    """
    Merge segment flags into global flags.

    Rules
    -----
    - isDeposited is CURRENT STATE -> overwrite with latest segment
    - stoppedByOutside / stoppedByLimiter / limiterHitEver / stoppedByDry
      are EVER-HAPPENED diagnostics -> OR them
    - time arrays keep earliest occurrence in absolute simulation time
    """
    index_like_float_keys = {"tStopLimiter", "tStopDry", "tStopOutside", "tDeposit", "tSurfaceHit"}
    first_value_float_keys = {"xDryHit", "yDryHit", "xSurfaceHit", "ySurfaceHit"}

    for key, arr in seg_flags.items():
        arr = np.asarray(arr)

        if arr.dtype == bool:
            g = global_flags[key]

            if key == "isDeposited":
                g[:] = arr
            else:
                g |= arr

        elif np.issubdtype(arr.dtype, np.integer):
            valid = arr >= 0
            shifted = np.full(arr.shape, -1, dtype=int)
            shifted[valid] = arr[valid] + history_offset

            g = global_flags[key]
            write_new = valid & (g < 0)
            g[write_new] = shifted[write_new]

            both = valid & (g >= 0)
            g[both] = np.minimum(g[both], shifted[both])

        elif key in index_like_float_keys:
            valid = np.isfinite(arr)
            g = global_flags[key]
            write_new = valid & (~np.isfinite(g))
            g[write_new] = arr[write_new]

            both = valid & np.isfinite(g)
            g[both] = np.minimum(g[both], arr[both])

        elif key in first_value_float_keys:
            valid = np.isfinite(arr)
            g = global_flags[key]

            write_new = valid & (~np.isfinite(g))
            g[write_new] = arr[write_new]

        else:
            valid = np.isfinite(arr)
            g = global_flags[key]

            write_new = valid & (~np.isfinite(g))
            g[write_new] = arr[write_new]

            both = valid & np.isfinite(g)
            g[both] = np.minimum(g[both], arr[both])


def interpolate_hydro_fields(hydro_cache, t, hmin, interpolation_mode="linear"):
    """Interpolate hydraulic fields at a target time ``t``."""
    times = np.array([H["time"] for H in hydro_cache], dtype=float)
    exact_idx = np.where(np.isclose(times, t, rtol=0.0, atol=1e-12))[0]

    if exact_idx.size:
        H0 = hydro_cache[int(exact_idx[0])]
        H = {
            "time": float(t),
            "U_c": H0["U_c"].copy(),
            "V_c": H0["V_c"].copy(),
            "h_c": H0["h_c"].copy(),
            "wse_c": H0["wse_c"].copy(),
            "Chezy_c": H0["Chezy_c"].copy(),
            "ks_c": None if H0["ks_c"] is None else H0["ks_c"].copy(),
            "z0_c": None if H0.get("z0_c") is None else H0["z0_c"].copy(),
            "d50_c": H0["d50_c"].copy(),
            "ustar_c": H0["ustar_c"].copy(),
            "ustarCrit_c": H0["ustarCrit_c"].copy(),
        }
    elif t <= times[0]:
        H0 = hydro_cache[0]
        H = {
            "time": float(t),
            "U_c": H0["U_c"].copy(),
            "V_c": H0["V_c"].copy(),
            "h_c": H0["h_c"].copy(),
            "wse_c": H0["wse_c"].copy(),
            "Chezy_c": H0["Chezy_c"].copy(),
            "ks_c": None if H0["ks_c"] is None else H0["ks_c"].copy(),
            "z0_c": None if H0.get("z0_c") is None else H0["z0_c"].copy(),
            "d50_c": H0["d50_c"].copy(),
            "ustar_c": H0["ustar_c"].copy(),
            "ustarCrit_c": H0["ustarCrit_c"].copy(),
        }
    elif t >= times[-1]:
        H1 = hydro_cache[-1]
        H = {
            "time": float(t),
            "U_c": H1["U_c"].copy(),
            "V_c": H1["V_c"].copy(),
            "h_c": H1["h_c"].copy(),
            "wse_c": H1["wse_c"].copy(),
            "Chezy_c": H1["Chezy_c"].copy(),
            "ks_c": None if H1["ks_c"] is None else H1["ks_c"].copy(),
            "z0_c": None if H1.get("z0_c") is None else H1["z0_c"].copy(),
            "d50_c": H1["d50_c"].copy(),
            "ustar_c": H1["ustar_c"].copy(),
            "ustarCrit_c": H1["ustarCrit_c"].copy(),
        }
    else:
        k = np.searchsorted(times, t) - 1
        k = max(0, min(k, len(times) - 2))

        H0 = hydro_cache[k]
        H1 = hydro_cache[k + 1]

        t0 = H0["time"]
        t1 = H1["time"]
        if interpolation_mode == "step":
            alpha = 0.0
        else:
            alpha = (t - t0) / (t1 - t0)

        def lerp(a, b):
            return (1.0 - alpha) * a + alpha * b

        H_crit = H0 if alpha < 0.5 else H1

        H = {
            "time": float(t),
            "U_c": lerp(H0["U_c"], H1["U_c"]),
            "V_c": lerp(H0["V_c"], H1["V_c"]),
            "h_c": lerp(H0["h_c"], H1["h_c"]),
            "wse_c": lerp(H0["wse_c"], H1["wse_c"]),
            "Chezy_c": lerp(H0["Chezy_c"], H1["Chezy_c"]),
            "ks_c": None if (H0["ks_c"] is None or H1["ks_c"] is None) else lerp(H0["ks_c"], H1["ks_c"]),
            "z0_c": None if (H0.get("z0_c") is None or H1.get("z0_c") is None) else lerp(H0["z0_c"], H1["z0_c"]),
            "d50_c": lerp(H0["d50_c"], H1["d50_c"]),
            "ustar_c": lerp(H0["ustar_c"], H1["ustar_c"]),
            "ustarCrit_c": H_crit["ustarCrit_c"].copy(),
        }

    wet = np.isfinite(H["h_c"]) & (H["h_c"] > hmin)

    for key in ("U_c", "V_c", "ustar_c"):
        arr = np.asarray(H[key], dtype=float).copy()
        arr[~np.isfinite(arr)] = 0.0
        arr[~wet] = 0.0
        H[key] = arr

    H["h_c"] = np.asarray(H["h_c"], dtype=float).copy()
    H["wse_c"] = np.asarray(H["wse_c"], dtype=float).copy()
    H["Chezy_c"] = np.asarray(H["Chezy_c"], dtype=float).copy()
    H["d50_c"] = np.asarray(H["d50_c"], dtype=float).copy()
    H["ustarCrit_c"] = np.asarray(H["ustarCrit_c"], dtype=float).copy()

    if H["ks_c"] is not None:
        H["ks_c"] = np.asarray(H["ks_c"], dtype=float).copy()
        H["ks_c"][~np.isfinite(H["ks_c"])] = np.nan
        H["ks_c"][~wet] = np.nan
    if H.get("z0_c") is not None:
        H["z0_c"] = np.asarray(H["z0_c"], dtype=float).copy()
        H["z0_c"][~np.isfinite(H["z0_c"])] = np.nan
        H["z0_c"][~wet] = np.nan

    return H


class _HydroIntervalInterpolator:
    """Reuse frame-pair deltas and output buffers for transient interpolation."""

    _ARRAY_KEYS = ("U_c", "V_c", "h_c", "wse_c", "Chezy_c", "d50_c", "ustar_c")

    def __init__(self, hydro_cache, hmin, interpolation_mode="linear"):
        self.hydro_cache = hydro_cache
        self.hmin = hmin
        self.interpolation_mode = interpolation_mode
        self.times = np.array([H["time"] for H in hydro_cache], dtype=float)
        self.interval_index = None
        self.h0 = None
        self.h1 = None
        self.t0 = None
        self.t1 = None
        self.delta = {}
        self.buffers = {
            key: np.empty_like(np.asarray(hydro_cache[0][key], dtype=float))
            for key in self._ARRAY_KEYS
        }
        self.optional_buffers = {
            key: np.empty_like(np.asarray(hydro_cache[0][key], dtype=float))
            for key in ("ks_c", "z0_c")
            if hydro_cache[0].get(key) is not None
        }
        self._result = {
            "time": float(self.times[0]),
            **self.buffers,
            "ks_c": self.optional_buffers.get("ks_c"),
            "z0_c": self.optional_buffers.get("z0_c"),
            "ustarCrit_c": np.asarray(hydro_cache[0]["ustarCrit_c"], dtype=float),
        }

    def _interval_for_time(self, t):
        if t <= self.times[0]:
            return 0, 0.0
        if t >= self.times[-1]:
            return len(self.times) - 2, 1.0
        k = int(np.searchsorted(self.times, t) - 1)
        k = max(0, min(k, len(self.times) - 2))
        if self.interpolation_mode == "step":
            return k, 0.0
        t0 = self.times[k]
        t1 = self.times[k + 1]
        return k, float((t - t0) / (t1 - t0))

    def _set_interval(self, k):
        if self.interval_index == k:
            return
        self.interval_index = k
        self.h0 = self.hydro_cache[k]
        self.h1 = self.hydro_cache[k + 1]
        self.t0 = self.times[k]
        self.t1 = self.times[k + 1]
        for key in self._ARRAY_KEYS:
            self.delta[key] = np.asarray(self.h1[key], dtype=float) - np.asarray(self.h0[key], dtype=float)
        for key in self.optional_buffers:
            if self.h0.get(key) is None or self.h1.get(key) is None:
                self.delta[key] = None
            else:
                self.delta[key] = np.asarray(self.h1[key], dtype=float) - np.asarray(self.h0[key], dtype=float)

    def sample(self, t):
        k, alpha = self._interval_for_time(t)
        self._set_interval(k)

        self._result["time"] = float(t)
        wet = None
        for key in self._ARRAY_KEYS:
            np.multiply(self.delta[key], alpha, out=self.buffers[key])
            self.buffers[key] += np.asarray(self.h0[key], dtype=float)
            if key == "h_c":
                wet = np.isfinite(self.buffers[key]) & (self.buffers[key] > self.hmin)

        assert wet is not None
        for key in ("U_c", "V_c", "ustar_c"):
            arr = self.buffers[key]
            arr[~np.isfinite(arr)] = 0.0
            arr[~wet] = 0.0

        for key in self.optional_buffers:
            if self.delta.get(key) is None:
                self._result[key] = None
                continue
            arr = self.optional_buffers[key]
            np.multiply(self.delta[key], alpha, out=arr)
            arr += np.asarray(self.h0[key], dtype=float)
            arr[~np.isfinite(arr)] = np.nan
            arr[~wet] = np.nan
            self._result[key] = arr

        self._result["ks_c"] = self.optional_buffers.get("ks_c") if self.delta.get("ks_c") is not None else None
        self._result["z0_c"] = self.optional_buffers.get("z0_c") if self.delta.get("z0_c") is not None else None
        self._result["ustarCrit_c"] = (
            self.h0["ustarCrit_c"] if alpha < 0.5 else self.h1["ustarCrit_c"]
        )
        return self._result


def advect_particles_euler_cell_transient_3d_ustar(
    mesh,
    hydro_cache,
    zb_c,
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
    surfaceUstarCrit,
    dzUpMax,
    uphillPolicy,
    useRWz,
    alphaKz,
    KzMax,
    bedPolicy,
    rng=None,
    nTrack=0,
    track_idx=None,
    showProgress=True,
    hydraulicFieldMode="cellwise",
    transportVelocityMode="depth_averaged",
    hydraulicClosureModel="external_adapter",
    rAniso=3.0,
    release_times=None,
    stopWhenAllParticlesSurfaced=False,
    hydraulicInterpolationMode="linear",
    transportModel="random_walk",
    entrainmentProbMode="gaussian_threshold",
    entrainmentProbSigmaStar=0.2,
    entrainmentProbK=None,
    surfaceDetachmentProbMode="gaussian_threshold",
    surfaceDetachmentProbSigmaStar=0.2,
    surfaceDetachmentProbK=None,
    TL_horizontal=2.0,
    TL_vertical=0.5,
    particleEvolution=None,
    profile=None,
    trackOutputStride=1,
):
    """
    Transient driver with linear interpolation of hydraulic fields
    at each particle timestep.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(hydro_cache) < 2:
        raise ValueError("hydro_cache must contain at least 2 frames.")

    if transportVelocityMode not in ("depth_averaged", "loglaw_vertical"):
        raise ValueError("transportVelocityMode must be 'depth_averaged' or 'loglaw_vertical'")
    if hydraulicFieldMode not in ("cellwise", "local_idw"):
        raise ValueError("hydraulicFieldMode must be 'cellwise' or 'local_idw'")
    hydraulicClosureModel = str(hydraulicClosureModel).strip().lower()
    if hydraulicClosureModel not in ("external_adapter", "standard_z0"):
        raise ValueError("hydraulicClosureModel must be 'external_adapter' or 'standard_z0'")
    if hydraulicInterpolationMode not in ("linear", "step"):
        raise ValueError("hydraulicInterpolationMode must be 'linear' or 'step'")

    P0 = np.asarray(P0, dtype=float)
    zb_c = np.asarray(zb_c, dtype=float).ravel()

    if P0.ndim != 2 or P0.shape[1] != 3:
        raise ValueError("P0 must be (Np,3) with columns [x, y, z]")

    frame_times = np.array([H["time"] for H in hydro_cache], dtype=float)

    t0 = max(float(t0), frame_times[0])
    t1 = min(float(t1), frame_times[-1])

    if t1 <= t0:
        raise ValueError("Need t1 > t0 within hydro time range.")

    n_steps = int(np.round((t1 - t0) / dt))
    tHist = t0 + np.arange(n_steps + 1, dtype=float) * dt
    Nt = len(tHist)
    trackOutputStride = max(1, int(trackOutputStride))

    if Nt < 2:
        raise ValueError("Need at least 2 particle times.")

    Np = P0.shape[0]
    release_times_arr = None if release_times is None else np.asarray(release_times, dtype=float).ravel()

    n_updates = Nt - 1
    print_every = max(1, n_updates // 100)

    def print_global_progress(step_done):
        if not showProgress:
            return
        pct = 100.0 if n_updates <= 0 else 100.0 * step_done / n_updates
        print(f"Progress: {pct:5.1f} %  (global step {step_done} / {n_updates})", flush=True)

    Pcur = P0.copy()
    last_track_idx = track_idx

    Ptrack_all = None
    tidTr_all = None
    flags_all = None
    flags_state = None
    transport_state = None

    xEnd = Pcur[:, 0].copy()
    yEnd = Pcur[:, 1].copy()
    zEnd = Pcur[:, 2].copy()
    tidEnd = np.full(Np, -1, dtype=int)

    zbTr_list = []
    wseTr_list = []

    H_init = interpolate_hydro_fields(hydro_cache, tHist[0], hmin, interpolation_mode=hydraulicInterpolationMode)

    if transportVelocityMode == "loglaw_vertical" and H_init["ks_c"] is None:
        raise RuntimeError(
            "transportVelocityMode='loglaw_vertical' requires ks_c in hydro_cache, "
            "but ks_c is missing."
        )

    hydro_interpolator = _HydroIntervalInterpolator(
        hydro_cache,
        hmin,
        interpolation_mode=hydraulicInterpolationMode,
    )
    for it in range(Nt - 1):
        ta = tHist[it]
        tb = tHist[it + 1]
        tm = 0.5 * (ta + tb)

        t_section = perf_counter()
        H = hydro_interpolator.sample(tm)
        if profile is not None:
            sections = profile.setdefault("sections", {})
            sections["hydro_interpolation_s"] = sections.get("hydro_interpolation_s", 0.0) + (perf_counter() - t_section)

        t_section = perf_counter()
        out = advect_particles_euler_cell_fast_fullcache_3d_ustar(
            mesh,
            H["U_c"],
            H["V_c"],
            H["h_c"],
            zb_c,
            H["wse_c"],
            H["ustar_c"],
            H["ustarCrit_c"],
            surfaceUstarCrit,
            bedPolicy,
            Pcur,
            ta,
            tb,
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
            track_idx=last_track_idx,
            showProgress=False,
            flags0=flags_state,
            depositedAlive=True,
            hydraulicFieldMode=hydraulicFieldMode,
            transportVelocityMode=transportVelocityMode,
            hydraulicClosureModel=hydraulicClosureModel,
            d50_c=H["d50_c"],
            ks_c=H["ks_c"],
            z0_c=H.get("z0_c"),
            rAniso=rAniso,
            entrainmentProbMode=entrainmentProbMode,
            entrainmentProbSigmaStar=entrainmentProbSigmaStar,
            entrainmentProbK=entrainmentProbK,
            surfaceDetachmentProbMode=surfaceDetachmentProbMode,
            surfaceDetachmentProbSigmaStar=surfaceDetachmentProbSigmaStar,
            surfaceDetachmentProbK=surfaceDetachmentProbK,
            tid0=tidEnd if flags_state is not None else None,
            positionsInitialized=(flags_state is not None),
            countResidence=False,
            release_times=release_times,
            stopWhenAllParticlesSurfaced=stopWhenAllParticlesSurfaced,
            transportModel=transportModel,
            TL_horizontal=TL_horizontal,
            TL_vertical=TL_vertical,
            transport_state0=transport_state,
            particleEvolution=particleEvolution,
            profile=profile,
            fieldSamplerOverride=None,
        )
        if profile is not None:
            sections = profile.setdefault("sections", {})
            sections["transient_particle_step_s"] = sections.get("transient_particle_step_s", 0.0) + (perf_counter() - t_section)

        (
            Ptrack_seg,
            _,
            tidTr_seg,
            last_track_idx,
            xEnd,
            yEnd,
            zEnd,
            tidEnd,
            _,
            _,
            _,
            flags_seg,
            transport_state_seg,
        ) = out

        t_section = perf_counter()
        if flags_all is None:
            flags_all = _init_merged_flags(flags_seg, Np)

        history_offset = it
        _merge_flags(flags_all, flags_seg, history_offset)

        flags_state = {kk: np.asarray(vv).copy() for kk, vv in flags_seg.items()}
        transport_state = {kk: np.asarray(vv).copy() for kk, vv in transport_state_seg.items()}

        tid_now = tidTr_seg[:, -1]
        xy_now = Ptrack_seg[:, :2, -1]
        zb_now, wse_now = _sample_tracked_bed_surface(
            mesh,
            hydraulicFieldMode,
            xy_now,
            tid_now,
            zb_c,
            H["wse_c"],
        )

        if it == 0:
            tid0 = tidTr_seg[:, 0]
            xy0 = Ptrack_seg[:, :2, 0]
            zb0, wse0 = _sample_tracked_bed_surface(
                mesh,
                hydraulicFieldMode,
                xy0,
                tid0,
                zb_c,
                H_init["wse_c"],
            )

            zbTr_list.append(zb0)
            wseTr_list.append(wse0)

        if Ptrack_all is None:
            Ptrack_all = Ptrack_seg[:, :, :1].copy()
            tidTr_all = tidTr_seg[:, :1].copy()

        step_done = it + 1
        store_current_step = (step_done % trackOutputStride) == 0
        if store_current_step:
            Ptrack_all = np.concatenate([Ptrack_all, Ptrack_seg[:, :, -1:]], axis=2)
            tidTr_all = np.concatenate([tidTr_all, tidTr_seg[:, -1:]], axis=1)
            zbTr_list.append(zb_now)
            wseTr_list.append(wse_now)

        Pcur = np.column_stack([xEnd, yEnd, zEnd])
        if profile is not None:
            sections = profile.setdefault("sections", {})
            sections["transient_merge_s"] = sections.get("transient_merge_s", 0.0) + (perf_counter() - t_section)

        if (step_done % print_every == 0) or (step_done == n_updates):
            print_global_progress(step_done)

        if stopWhenAllParticlesSurfaced and flags_all is not None:
            if release_times_arr is None:
                released = np.ones(Np, dtype=bool)
            else:
                released = release_times_arr <= (tb + 1e-12)

            if np.any(released) and np.all(released):
                done = (
                    np.asarray(flags_all["isSurfaced"], dtype=bool)
                    | np.asarray(flags_all["stoppedByOutside"], dtype=bool)
                    | np.asarray(flags_all["stoppedByLimiter"], dtype=bool)
                    | np.asarray(flags_all["stoppedByDry"], dtype=bool)
                )
                if np.all(done[released]):
                    print(
                        "Stopping transient run early because all released particles "
                        "have already surfaced or otherwise terminated."
                    )
                    break

    if Ptrack_all is None:
        raise RuntimeError("No transient timestep was advanced.")

    n_stored = Ptrack_all.shape[2]
    tHist_track = tHist[::trackOutputStride][:n_stored]
    xTr = Ptrack_all[:, 0, :]
    yTr = Ptrack_all[:, 1, :]
    zTr = Ptrack_all[:, 2, :]

    zbTr_track = np.column_stack(zbTr_list)
    wseTr_track = np.column_stack(wseTr_list)

    return {
        "Ptrack": Ptrack_all,
        "xTr": xTr,
        "yTr": yTr,
        "zTr": zTr,
        "tidTr_track": tidTr_all,
        "track_idx": last_track_idx,
        "xEnd": xEnd,
        "yEnd": yEnd,
        "zEnd": zEnd,
        "tidEnd": tidEnd,
        "flags": flags_all,
        "zbTr_track": zbTr_track,
        "wseTr_track": wseTr_track,
        "tHist": tHist_track,
    }
