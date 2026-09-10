"""Export utilities for HydroLPT trajectory, status, and run metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_output_folder(outdir: str | Path) -> Path:
    """Create and return the output directory for exported run products."""
    output_directory = Path(outdir)
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def _validate_matching_shapes(*arrays: np.ndarray) -> None:
    """Validate that all provided arrays share the same shape."""
    shapes = [array.shape for array in arrays]
    if len(set(shapes)) > 1:
        raise ValueError(f"Arrays must share the same shape, got {shapes}")


def _to_1d(a: Any, n: int | None = None, dtype=None, fill_value=np.nan) -> np.ndarray:
    """Convert input to a 1D array with optional broadcasting and filling."""
    if a is None:
        if n is None:
            raise ValueError("n must be provided when input is None")
        return np.full(n, fill_value, dtype=dtype if dtype is not None else float)

    arr = np.asarray(a, dtype=dtype)
    if arr.ndim == 0:
        if n is None:
            return arr.reshape(1)
        return np.full(n, arr.item(), dtype=arr.dtype)

    arr = arr.ravel()

    if n is not None and arr.size != n:
        raise ValueError(f"Array length {arr.size} does not match expected length {n}")

    return arr


def _state_to_text(final_state: Any, n: int) -> np.ndarray:
    """Convert a final-state container into a string array of length ``n``."""
    if final_state is None:
        return np.full(n, "", dtype=object)

    arr = np.asarray(final_state)

    if arr.ndim == 0:
        return np.full(n, str(arr.item()), dtype=object)

    arr = arr.ravel()
    if arr.size != n:
        raise ValueError(f"final_state length {arr.size} != expected {n}")

    return arr.astype(str)


def _event_time_from_flags(flags: dict[str, Any], key: str, n: int) -> np.ndarray:
    """Return event time values if present, otherwise a NaN array."""
    if flags is None or key not in flags:
        return np.full(n, np.nan, dtype=float)
    return _to_1d(flags[key], n=n, dtype=float, fill_value=np.nan)


def _event_bool_from_flags(flags: dict[str, Any], key: str, n: int) -> np.ndarray:
    """Return boolean event flags if present, otherwise a False array."""
    if flags is None or key not in flags:
        return np.zeros(n, dtype=bool)
    return _to_1d(flags[key], n=n, dtype=bool, fill_value=False)


def _sanitize_state_text(state_value: Any) -> str:
    """Return a CSV-safe state label."""
    return str(state_value).replace(",", ";")


def export_tracked_trajectories(
    xTr: np.ndarray,
    yTr: np.ndarray,
    zTr: np.ndarray,
    *,
    dt: float,
    outdir: str | Path,
    tidTr: np.ndarray | None = None,
    track_idx: np.ndarray | None = None,
    time0: float = 0.0,
    filename: str = "trajectories_tracked.csv",
) -> Path:
    """
    Export tracked particle trajectories.

    Parameters
    ----------
    xTr, yTr, zTr : (nTrack, Nt)
        Tracked particle coordinates through time.
    dt : float
        Output timestep between stored trajectory points.
    outdir : str or Path
        Output directory, typically run_folder / "output".
    tidTr : (nTrack, Nt), optional
        Cell id history for tracked particles.
    track_idx : (nTrack,), optional
        Global particle ids corresponding to tracked subset.
        If omitted, local ids [0..nTrack-1] are used.
    time0 : float, optional
        Initial output time.
    filename : str, optional
        CSV filename.

    Returns
    -------
    Path
        Path to written CSV file.
    """
    outdir = ensure_output_folder(outdir)

    xTr = np.asarray(xTr, dtype=float)
    yTr = np.asarray(yTr, dtype=float)
    zTr = np.asarray(zTr, dtype=float)

    _validate_matching_shapes(xTr, yTr, zTr)

    nTrack, Nt = xTr.shape
    times = time0 + np.arange(Nt, dtype=float) * float(dt)

    if track_idx is None:
        track_idx = np.arange(nTrack, dtype=int)
    else:
        track_idx = _to_1d(track_idx, n=nTrack, dtype=int)

    if tidTr is not None:
        tidTr = np.asarray(tidTr, dtype=int)
        if tidTr.shape != xTr.shape:
            raise ValueError("tidTr must have the same shape as xTr")
    else:
        tidTr = np.full_like(xTr, -1, dtype=int)

    rows: list[list[Any]] = []

    for i in range(nTrack):
        pid = int(track_idx[i])
        for k in range(Nt):
            x = xTr[i, k]
            y = yTr[i, k]
            z = zTr[i, k]

            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue

            rows.append([
                pid,
                float(times[k]),
                float(x),
                float(y),
                float(z),
                int(tidTr[i, k]),
            ])

    file = outdir / filename

    if rows:
        data = np.asarray(rows, dtype=object)
        np.savetxt(
            file,
            data,
            delimiter=",",
            fmt=["%d", "%.10g", "%.10g", "%.10g", "%.10g", "%d"],
            header="particle_id,time,x,y,z,cell_id",
            comments="",
        )
    else:
        file.write_text("particle_id,time,x,y,z,cell_id\n", encoding="utf-8")

    print(f"\nExported tracked trajectories -> {file}")
    return file


def export_final_particles(
    xEnd: np.ndarray,
    yEnd: np.ndarray,
    zEnd: np.ndarray,
    *,
    outdir: str | Path,
    tidEnd: np.ndarray | None = None,
    final_state: Any = None,
    filename: str = "particles_final.csv",
) -> Path:
    """
    Export final particle locations for all particles.

    Parameters
    ----------
    xEnd, yEnd, zEnd : (Np,)
        Final particle coordinates.
    outdir : str or Path
        Output directory.
    tidEnd : (Np,), optional
        Final cell id for each particle.
    final_state : (Np,), optional
        Final state label per particle, e.g. deposited/mobile/outside.
    filename : str, optional
        CSV filename.

    Returns
    -------
    Path
        Path to written CSV file.
    """
    outdir = ensure_output_folder(outdir)

    xEnd = np.asarray(xEnd, dtype=float).ravel()
    yEnd = np.asarray(yEnd, dtype=float).ravel()
    zEnd = np.asarray(zEnd, dtype=float).ravel()

    if not (xEnd.size == yEnd.size == zEnd.size):
        raise ValueError("xEnd, yEnd, zEnd must have the same length")

    n = xEnd.size
    pid = np.arange(n, dtype=int)

    if tidEnd is None:
        tidEnd = np.full(n, -1, dtype=int)
    else:
        tidEnd = _to_1d(tidEnd, n=n, dtype=int, fill_value=-1)

    final_state_txt = _state_to_text(final_state, n)

    file = outdir / filename

    with file.open("w", encoding="utf-8", newline="") as f:
        f.write("particle_id,x,y,z,cell_id,final_state\n")
        for i in range(n):
            state_i = _sanitize_state_text(final_state_txt[i])
            f.write(
                f"{pid[i]},{xEnd[i]:.10g},{yEnd[i]:.10g},{zEnd[i]:.10g},{tidEnd[i]},{state_i}\n"
            )

    print(f"Exported final particle locations -> {file}")
    return file


def export_particle_status(
    *,
    outdir: str | Path,
    n_particles: int,
    flags: dict[str, Any] | None = None,
    filename: str = "particle_status.csv",
) -> Path:
    """
    Export particle event/status information for all particles.

    Recognized boolean flag keys:
      - isDeposited
      - stoppedByLimiter
      - stoppedByDry
      - stoppedByOutside

    Recognized time keys:
      - tDeposit
      - tStopLimiter
      - tStopDry
      - tStopOutside
      - tSurfaceHit

    Recognized first-hit location keys:
      - xSurfaceHit
      - ySurfaceHit

    Any missing key is filled with False or NaN.

    Returns
    -------
    Path
        Path to written CSV file.
    """
    outdir = ensure_output_folder(outdir)
    n = int(n_particles)

    pid = np.arange(n, dtype=int)

    is_deposited = _event_bool_from_flags(flags, "isDeposited", n)
    stopped_limiter = _event_bool_from_flags(flags, "stoppedByLimiter", n)
    stopped_dry = _event_bool_from_flags(flags, "stoppedByDry", n)
    stopped_outside = _event_bool_from_flags(flags, "stoppedByOutside", n)

    t_deposit = _event_time_from_flags(flags, "tDeposit", n)
    t_stop_limiter = _event_time_from_flags(flags, "tStopLimiter", n)
    t_stop_dry = _event_time_from_flags(flags, "tStopDry", n)
    t_stop_outside = _event_time_from_flags(flags, "tStopOutside", n)
    t_surface_hit = _event_time_from_flags(flags, "tSurfaceHit", n)
    x_surface_hit = _event_time_from_flags(flags, "xSurfaceHit", n)
    y_surface_hit = _event_time_from_flags(flags, "ySurfaceHit", n)

    file = outdir / filename

    header = (
        "particle_id,is_deposited,stopped_by_limiter,stopped_by_dry,stopped_by_outside,"
        "t_deposit,t_stop_limiter,t_stop_dry,t_stop_outside,t_surface_hit,x_surface_hit,y_surface_hit"
    )

    data = np.column_stack([
        pid,
        is_deposited.astype(int),
        stopped_limiter.astype(int),
        stopped_dry.astype(int),
        stopped_outside.astype(int),
        t_deposit,
        t_stop_limiter,
        t_stop_dry,
        t_stop_outside,
        t_surface_hit,
        x_surface_hit,
        y_surface_hit,
    ])

    np.savetxt(
        file,
        data,
        delimiter=",",
        fmt=["%d", "%d", "%d", "%d", "%d", "%.10g", "%.10g", "%.10g", "%.10g", "%.10g", "%.10g", "%.10g"],
        header=header,
        comments="",
    )

    print(f"Exported particle status -> {file}")
    return file


def export_particle_state_counts(
    *,
    outdir: str | Path,
    t: np.ndarray,
    state_counts: np.ndarray,
    filename: str = "particle_state_counts.csv",
) -> Path:
    """Export population state counts at each stored output time."""
    outdir = ensure_output_folder(outdir)
    times = np.asarray(t, dtype=float).ravel()
    counts = np.asarray(state_counts, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[1] != 10:
        raise ValueError("state_counts must have shape (n_times, 10)")
    if counts.shape[0] != times.size:
        raise ValueError("state_counts and t must have the same number of rows")

    n_particles = np.maximum(counts[:, 0], 1)
    data = np.column_stack(
        [
            times,
            times / 86400.0,
            counts,
            counts[:, 1] / n_particles,
            counts[:, 4] / n_particles,
            counts[:, 9] / n_particles,
            counts[:, 6] / n_particles,
            counts[:, 8] / n_particles,
            counts[:, 7] / n_particles,
            counts[:, 5] / n_particles,
        ]
    )
    header = (
        "time_s,time_days,released_count,outside_count,dry_count,limiter_count,"
        "deposited_count,surfaced_count,near_bed_mobile_count,near_surface_mobile_count,"
        "suspended_count,mobile_count,outside_fraction,deposited_fraction,mobile_fraction,"
        "near_bed_mobile_fraction,suspended_fraction,near_surface_mobile_fraction,surfaced_fraction"
    )
    file = outdir / filename
    np.savetxt(
        file,
        data,
        delimiter=",",
        fmt=(
            ["%.10g", "%.10g"]
            + ["%d"] * 10
            + ["%.10g"] * 7
        ),
        header=header,
        comments="",
    )
    print(f"Exported particle state counts -> {file}")
    return file


def export_run_info(
    *,
    outdir: str | Path,
    hydrolpt_version: str = "HydroLPT v0.9",
    adapter_name: str = "BASEHPC",
    n_particles: int | None = None,
    n_tracked: int | None = None,
    n_frames: int | None = None,
    dt: float | None = None,
    extra: dict[str, Any] | None = None,
    filename_txt: str = "run_info.txt",
    filename_json: str = "run_info.json",
) -> tuple[Path, Path]:
    """
    Export simple run metadata to TXT and JSON.
    """
    outdir = ensure_output_folder(outdir)

    info: dict[str, Any] = {
        "hydrolpt_version": hydrolpt_version,
        "adapter_name": adapter_name,
        "n_particles": n_particles,
        "n_tracked": n_tracked,
        "n_frames": n_frames,
        "dt": dt,
    }

    if extra:
        info.update(extra)

    json_file = outdir / filename_json

    txt_file = outdir / filename_txt
    with txt_file.open("w", encoding="utf-8") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")

    with json_file.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"Exported run info -> {txt_file}")
    print(f"Exported run info -> {json_file}")
    return txt_file, json_file


def export_all(
    *,
    outdir: str | Path,
    xTr: np.ndarray | None = None,
    yTr: np.ndarray | None = None,
    zTr: np.ndarray | None = None,
    tidTr: np.ndarray | None = None,
    track_idx: np.ndarray | None = None,
    dt: float | None = None,
    xEnd: np.ndarray,
    yEnd: np.ndarray,
    zEnd: np.ndarray,
    tidEnd: np.ndarray | None = None,
    final_state: Any = None,
    flags: dict[str, Any] | None = None,
    state_counts: np.ndarray | None = None,
    state_count_times: np.ndarray | None = None,
    hydrolpt_version: str = "HydroLPT v0.9",
    adapter_name: str = "BASEHPC",
    extra_run_info: dict[str, Any] | None = None,
) -> None:
    """
    Convenience wrapper to export all standard HydroLPT outputs.

    Required:
      - xEnd, yEnd, zEnd

    Optional:
      - tracked trajectories xTr, yTr, zTr, dt
      - tidTr, track_idx
      - tidEnd, final_state
      - flags
      - run metadata

    Output files:
      - trajectories_tracked.csv      (if tracked trajectories provided)
      - particles_final.csv
      - particle_status.csv
      - run_info.txt
      - run_info.json
    """
    outdir = ensure_output_folder(outdir)

    xEnd = np.asarray(xEnd, dtype=float).ravel()
    yEnd = np.asarray(yEnd, dtype=float).ravel()
    zEnd = np.asarray(zEnd, dtype=float).ravel()

    if not (xEnd.size == yEnd.size == zEnd.size):
        raise ValueError("xEnd, yEnd, zEnd must have the same length")

    n_particles = xEnd.size
    n_tracked = None
    n_frames = None

    if xTr is not None or yTr is not None or zTr is not None:
        if xTr is None or yTr is None or zTr is None:
            raise ValueError("xTr, yTr, zTr must either all be provided or all be omitted")
        if dt is None:
            raise ValueError("dt must be provided when exporting tracked trajectories")

        xTr = np.asarray(xTr, dtype=float)
        yTr = np.asarray(yTr, dtype=float)
        zTr = np.asarray(zTr, dtype=float)

        n_tracked, n_frames = xTr.shape

        export_tracked_trajectories(
            xTr=xTr,
            yTr=yTr,
            zTr=zTr,
            tidTr=tidTr,
            track_idx=track_idx,
            dt=float(dt),
            outdir=outdir,
        )

    export_final_particles(
        xEnd=xEnd,
        yEnd=yEnd,
        zEnd=zEnd,
        tidEnd=tidEnd,
        final_state=final_state,
        outdir=outdir,
    )

    export_particle_status(
        outdir=outdir,
        n_particles=n_particles,
        flags=flags,
    )

    if state_counts is not None:
        if state_count_times is None:
            raise ValueError("state_count_times must be provided with state_counts")
        export_particle_state_counts(
            outdir=outdir,
            t=state_count_times,
            state_counts=state_counts,
        )

    export_run_info(
        outdir=outdir,
        hydrolpt_version=hydrolpt_version,
        adapter_name=adapter_name,
        n_particles=n_particles,
        n_tracked=n_tracked,
        n_frames=n_frames,
        dt=dt,
        extra=extra_run_info,
    )

