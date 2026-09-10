from __future__ import annotations

from typing import Any

import numpy as np


def _validate_flag_lengths(
    flags: dict[str, Any],
    n_particles: int,
) -> dict[str, np.ndarray]:
    """Return authoritative end-state flags with validated lengths."""
    validated_flags = {
        "isDeposited": np.asarray(flags["isDeposited"], dtype=bool),
        "isSurfaced": np.asarray(flags["isSurfaced"], dtype=bool),
        "stoppedByLimiter": np.asarray(flags["stoppedByLimiter"], dtype=bool),
        "stoppedByDry": np.asarray(flags["stoppedByDry"], dtype=bool),
        "stoppedByOutside": np.asarray(flags["stoppedByOutside"], dtype=bool),
    }

    for name, values in validated_flags.items():
        if values.size != n_particles:
            raise ValueError(f"flags['{name}'] must have length {n_particles}.")

    return validated_flags


def _map_cell_values_to_particles(
    particle_cell_ids: np.ndarray,
    cell_values: np.ndarray,
) -> np.ndarray:
    """Map cell-centered values to particle-aligned arrays using cell ids."""
    mapped_values = np.full(particle_cell_ids.size, np.nan, dtype=float)
    in_domain = particle_cell_ids >= 0
    if np.any(in_domain):
        mapped_values[in_domain] = cell_values[particle_cell_ids[in_domain]]
    return mapped_values


def classify_end_state_all(
    xEnd,
    yEnd,
    zEnd,
    tidEnd,
    flags,
    zb_c,
    wse_c,
    hmin,
    zClassTol,
    *,
    zb_end=None,
    wse_end=None,
    h_end=None,
    vertical_overlap_preference: str = "bed",
):
    """
    Classify final particle states using end positions only.

    Parameters
    ----------
    xEnd, yEnd, zEnd : ndarray, shape (Np,)
        Final particle coordinates.
    tidEnd : ndarray, shape (Np,)
        Final cell id for each particle. Values < 0 indicate outside/unknown.
    flags : dict
        Dictionary containing boolean arrays of shape (Np,):
            - isDeposited
            - stoppedByLimiter
            - stoppedByDry
            - stoppedByOutside
    zb_c, wse_c : ndarray, shape (Nc,)
        Cell-centered bed elevation and free-surface elevation.
    hmin : float
        Minimum water depth threshold for a cell to be considered wet.
    zClassTol : float
        Classification tolerance used only for identifying particles
        near the bed or near the surface. This does not affect transport.
    zb_end, wse_end, h_end : ndarray, shape (Np,), optional
        Particle-aligned final bed, water-surface, and depth values. When
        provided, these override cell-centered mapping. This is used for
        local-IDW hydraulic-field runs, where final classification must use
        the same spatial sampling convention as transport.
    vertical_overlap_preference : {"bed", "surface", "closest"}
        Rule used when shallow water makes the near-bed and near-surface
        tolerance zones overlap. Use "bed" for negatively buoyant particles,
        "surface" for positively buoyant particles, and "closest" for
        neutral particles.

    Returns
    -------
    dict
        Boolean arrays of shape (Np,) with keys:
            inDomain, finiteXYZ, wet,
            isDeposited, stoppedByLimiter, stoppedByDry, stoppedByOutside,
            isSurfaced, isNearSurfaceMobile,
            isNearBedMobile, isSuspended
    """
    x_end = np.asarray(xEnd, dtype=float).ravel()
    y_end = np.asarray(yEnd, dtype=float).ravel()
    z_end = np.asarray(zEnd, dtype=float).ravel()
    cell_ids_end = np.asarray(tidEnd, dtype=int).ravel()

    n_particles = x_end.size
    if y_end.size != n_particles or z_end.size != n_particles or cell_ids_end.size != n_particles:
        raise ValueError("xEnd, yEnd, zEnd, and tidEnd must all have the same length.")

    bed_elevation = np.asarray(zb_c, dtype=float)
    water_surface_elevation = np.asarray(wse_c, dtype=float)

    if zClassTol < 0.0:
        raise ValueError("zClassTol must be non-negative.")
    overlap_preference = str(vertical_overlap_preference).strip().lower()
    if overlap_preference not in {"bed", "surface", "closest"}:
        raise ValueError("vertical_overlap_preference must be 'bed', 'surface', or 'closest'.")

    # ------------------------------------------------------------------
    # Basic validity masks
    # ------------------------------------------------------------------
    in_domain = cell_ids_end >= 0
    finite_xyz = np.isfinite(x_end) & np.isfinite(y_end) & np.isfinite(z_end)

    if zb_end is None:
        bed_elevation_end = _map_cell_values_to_particles(cell_ids_end, bed_elevation)
    else:
        bed_elevation_end = np.asarray(zb_end, dtype=float).ravel()
        if bed_elevation_end.size != n_particles:
            raise ValueError("zb_end must have length Np.")

    if wse_end is None:
        water_surface_end = _map_cell_values_to_particles(cell_ids_end, water_surface_elevation)
    else:
        water_surface_end = np.asarray(wse_end, dtype=float).ravel()
        if water_surface_end.size != n_particles:
            raise ValueError("wse_end must have length Np.")

    if h_end is None:
        water_depth_end = water_surface_end - bed_elevation_end
    else:
        water_depth_end = np.asarray(h_end, dtype=float).ravel()
        if water_depth_end.size != n_particles:
            raise ValueError("h_end must have length Np.")
    wet = np.isfinite(water_depth_end) & (water_depth_end > hmin)

    end_flags = _validate_flag_lengths(flags, n_particles)
    is_deposited = end_flags["isDeposited"]
    is_surfaced = end_flags["isSurfaced"]
    stopped_by_limiter = end_flags["stoppedByLimiter"]
    stopped_by_dry = end_flags["stoppedByDry"]
    stopped_by_outside = end_flags["stoppedByOutside"]

    is_near_surface_mobile = np.zeros(n_particles, dtype=bool)
    is_near_bed_mobile = np.zeros(n_particles, dtype=bool)
    is_suspended = np.zeros(n_particles, dtype=bool)

    valid_vertical_state = (
        finite_xyz
        & in_domain
        & wet
        & (~is_surfaced)
        & (~stopped_by_limiter)
        & (~stopped_by_dry)
        & (~stopped_by_outside)
        & np.isfinite(bed_elevation_end)
        & np.isfinite(water_surface_end)
    )

    if np.any(valid_vertical_state):
        near_surface = z_end >= (water_surface_end - zClassTol)
        near_bed = z_end <= (bed_elevation_end + zClassTol)

        is_near_surface_mobile[valid_vertical_state] = (
            ~is_deposited[valid_vertical_state]
            & near_surface[valid_vertical_state]
        )

        is_near_bed_mobile[valid_vertical_state] = (
            ~is_deposited[valid_vertical_state]
            & near_bed[valid_vertical_state]
        )

        overlap = is_near_bed_mobile & is_near_surface_mobile
        if np.any(overlap):
            if overlap_preference == "bed":
                is_near_surface_mobile[overlap] = False
            elif overlap_preference == "surface":
                is_near_bed_mobile[overlap] = False
            else:
                dist_to_bed = np.abs(z_end - bed_elevation_end)
                dist_to_surface = np.abs(water_surface_end - z_end)
                prefer_bed = overlap & (dist_to_bed < dist_to_surface)
                prefer_surface = overlap & (dist_to_surface < dist_to_bed)
                tied = overlap & ~(prefer_bed | prefer_surface)
                is_near_surface_mobile[prefer_bed | tied] = False
                is_near_bed_mobile[prefer_surface | tied] = False
                is_suspended[tied] = True

        is_suspended[valid_vertical_state] = (
            ~is_deposited[valid_vertical_state]
            & ~is_near_bed_mobile[valid_vertical_state]
            & ~is_near_surface_mobile[valid_vertical_state]
        )

    return {
        "inDomain": in_domain,
        "finiteXYZ": finite_xyz,
        "wet": wet,
        "isDeposited": is_deposited,
        "isSurfaced": is_surfaced,
        "stoppedByLimiter": stopped_by_limiter,
        "stoppedByDry": stopped_by_dry,
        "stoppedByOutside": stopped_by_outside,
        "isNearSurfaceMobile": is_near_surface_mobile,
        "isNearBedMobile": is_near_bed_mobile,
        "isSuspended": is_suspended,
    }
