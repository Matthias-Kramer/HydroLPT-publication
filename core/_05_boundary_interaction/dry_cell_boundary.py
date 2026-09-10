from __future__ import annotations

import os
from typing import Callable

import numpy as np


DRY_POLICIES = ("stop", "reflect", "stick_active")


def apply_dry_stop(
    indices: np.ndarray,
    *,
    t_now: float,
    x_new: np.ndarray,
    y_new: np.ndarray,
    z_new: np.ndarray,
    tid_prev: np.ndarray,
    x_prev: np.ndarray,
    y_prev: np.ndarray,
    z_prev: np.ndarray,
    tid_at_prev: np.ndarray,
    stopped_by_dry: np.ndarray,
    t_stop_dry: np.ndarray,
) -> None:
    """Mark particles as dry-cell terminated and restore prior state."""
    idx = np.asarray(indices, dtype=int).ravel()
    if idx.size == 0:
        return

    stopped_by_dry[idx] = True
    new = np.isnan(t_stop_dry[idx])
    t_stop_dry[idx[new]] = t_now

    x_new[idx] = x_prev[idx]
    y_new[idx] = y_prev[idx]
    z_new[idx] = z_prev[idx]
    tid_prev[idx] = tid_at_prev[idx]


def validate_dry_policy(dry_policy: str) -> None:
    """Validate the dry-cell interaction policy."""
    if dry_policy not in DRY_POLICIES:
        raise ValueError('dryPolicy must be "stop", "reflect", or "stick_active".')


def _validate_subset_inputs(
    proposed_positions: np.ndarray,
    previous_positions: np.ndarray,
) -> None:
    """Validate wet/dry correction subset arrays."""
    if proposed_positions.ndim != 2 or proposed_positions.shape[1] != 2:
        raise ValueError("Pprop_sub must have shape (N, 2)")
    if previous_positions.shape != proposed_positions.shape:
        raise ValueError("Pdet_sub must have the same shape as Pprop_sub")


def _build_wet_cell_mask(
    bed_elevation: np.ndarray,
    water_surface_elevation: np.ndarray,
    hmin: float,
) -> np.ndarray:
    """Return a boolean mask of wet mesh cells."""
    return (
        np.isfinite(water_surface_elevation)
        & np.isfinite(bed_elevation)
        & ((water_surface_elevation - bed_elevation) > hmin)
    )


def _build_point_location_helpers(
    mesh,
    wet_cell_mask: np.ndarray,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray | None], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Create helper closures for point location and wetness checks."""

    def point_triangle_ids_global(points: np.ndarray) -> np.ndarray:
        return mesh.point_location(points[:, 0], points[:, 1])

    def point_triangle_ids(points: np.ndarray, start_triangle_ids: np.ndarray | None = None) -> np.ndarray:
        if start_triangle_ids is None:
            return point_triangle_ids_global(points)

        start_triangle_ids = np.asarray(start_triangle_ids, dtype=int).ravel()
        if start_triangle_ids.size != points.shape[0]:
            raise ValueError("start_triangle_ids must have the same length as points")

        barycentric_coords = mesh.barycentric(start_triangle_ids, points)
        inside_start = np.all(
            (barycentric_coords >= -1e-9) & (barycentric_coords <= 1.0 + 1e-9),
            axis=1,
        )
        triangle_ids = np.full(start_triangle_ids.shape, -1, dtype=int)
        if np.any(inside_start):
            triangle_ids[inside_start] = start_triangle_ids[inside_start]
        if np.any(~inside_start):
            unresolved = ~inside_start
            triangle_ids[unresolved] = mesh.point_location_local(
                points[unresolved, 0],
                points[unresolved, 1],
                start_triangle_ids[unresolved],
            )
        return triangle_ids

    def wet_from_triangle_ids(triangle_ids: np.ndarray) -> np.ndarray:
        triangle_ids = np.asarray(triangle_ids, dtype=int)
        is_wet = np.zeros(triangle_ids.shape, dtype=bool)
        valid = triangle_ids >= 0
        if np.any(valid):
            is_wet[valid] = wet_cell_mask[triangle_ids[valid]]
        return is_wet

    return point_triangle_ids_global, point_triangle_ids, wet_from_triangle_ids


def _resolve_wetdry_bisection_steps(n_bisect: int | None) -> int:
    """Return the configured wet/dry bisection count."""
    if n_bisect is not None:
        return max(1, int(n_bisect))

    raw = os.getenv("HYDROLPT_WETDRY_N_BISECT", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass

    return 12


def _bisect_wet_dry_contact(
    previous_wet_positions: np.ndarray,
    previous_wet_triangle_ids: np.ndarray,
    proposed_dry_positions: np.ndarray,
    point_triangle_ids: Callable[[np.ndarray, np.ndarray | None], np.ndarray],
    wet_from_triangle_ids: Callable[[np.ndarray], np.ndarray],
    n_bisect: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate wet/dry contact points by repeated bisection."""
    wet_side = previous_wet_positions.copy()
    wet_side_triangle_ids = np.asarray(previous_wet_triangle_ids, dtype=int).copy()
    dry_side = proposed_dry_positions.copy()

    for _ in range(n_bisect):
        midpoint = 0.5 * (wet_side + dry_side)
        midpoint_triangle_ids = point_triangle_ids(midpoint, wet_side_triangle_ids)
        midpoint_is_wet = wet_from_triangle_ids(midpoint_triangle_ids)

        if np.any(midpoint_is_wet):
            wet_side[midpoint_is_wet] = midpoint[midpoint_is_wet]
            wet_side_triangle_ids[midpoint_is_wet] = midpoint_triangle_ids[midpoint_is_wet]
        if np.any(~midpoint_is_wet):
            dry_side[~midpoint_is_wet] = midpoint[~midpoint_is_wet]

    return wet_side, wet_side_triangle_ids, dry_side


def _wet_dry_boundary_edges(triangles: np.ndarray, wet_cell_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mesh edges separating wet cells from dry/outside cells and their wet-side cell ids."""
    tri = np.asarray(triangles, dtype=int)
    wet = np.asarray(wet_cell_mask, dtype=bool).ravel()
    edge_owner: dict[tuple[int, int], tuple[int, int]] = {}
    boundary_edges = []
    boundary_wet_tids = []
    edge_node_cols = ((1, 2), (2, 0), (0, 1))

    for tid, nodes in enumerate(tri):
        for edge_id, cols in enumerate(edge_node_cols):
            pair = (int(nodes[cols[0]]), int(nodes[cols[1]]))
            key = tuple(sorted(pair))
            previous = edge_owner.pop(key, None)
            if previous is None:
                edge_owner[key] = (tid, edge_id)
                continue

            other_tid, other_edge_id = previous
            this_wet = bool(wet[tid])
            other_wet = bool(wet[other_tid])
            if this_wet == other_wet:
                continue
            wet_tid = tid if this_wet else other_tid
            wet_edge_id = edge_id if this_wet else other_edge_id
            wet_cols = edge_node_cols[wet_edge_id]
            wet_nodes = tri[wet_tid]
            boundary_edges.append((int(wet_nodes[wet_cols[0]]), int(wet_nodes[wet_cols[1]])))
            boundary_wet_tids.append(int(wet_tid))

    for _key, (tid, edge_id) in edge_owner.items():
        if bool(wet[tid]):
            cols = edge_node_cols[edge_id]
            nodes = tri[tid]
            boundary_edges.append((int(nodes[cols[0]]), int(nodes[cols[1]])))
            boundary_wet_tids.append(int(tid))

    if not boundary_edges:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=int)
    return np.asarray(boundary_edges, dtype=int), np.asarray(boundary_wet_tids, dtype=int)


def _first_segment_boundary_intersections(
    previous_positions: np.ndarray,
    proposed_positions: np.ndarray,
    edge_xy0: np.ndarray,
    edge_xy1: np.ndarray,
    edge_wet_tids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return first segment intersections with wet/dry boundary edges."""
    p0 = np.asarray(previous_positions, dtype=float)
    p1 = np.asarray(proposed_positions, dtype=float)
    out = np.full_like(p0, np.nan, dtype=float)
    out_tid = np.full(p0.shape[0], -1, dtype=int)
    out_alpha = np.full(p0.shape[0], np.nan, dtype=float)
    if p0.size == 0 or edge_xy0.size == 0:
        return out, out_tid, out_alpha

    edges0 = np.asarray(edge_xy0, dtype=float)
    edge_vec = np.asarray(edge_xy1, dtype=float) - edges0
    eps = 1.0e-12

    for i in range(p0.shape[0]):
        r = p1[i] - p0[i]
        if not np.all(np.isfinite(r)) or float(np.dot(r, r)) <= eps:
            continue

        qp = edges0 - p0[i]
        denom = r[0] * edge_vec[:, 1] - r[1] * edge_vec[:, 0]
        parallel = np.abs(denom) <= eps
        alpha = np.full(denom.shape, np.nan, dtype=float)
        beta = np.full(denom.shape, np.nan, dtype=float)
        good = ~parallel
        if np.any(good):
            alpha[good] = (qp[good, 0] * edge_vec[good, 1] - qp[good, 1] * edge_vec[good, 0]) / denom[good]
            beta[good] = (qp[good, 0] * r[1] - qp[good, 1] * r[0]) / denom[good]

        hit = (
            np.isfinite(alpha)
            & np.isfinite(beta)
            & (alpha >= -1.0e-10)
            & (alpha <= 1.0 + 1.0e-10)
            & (beta >= -1.0e-10)
            & (beta <= 1.0 + 1.0e-10)
        )
        if not np.any(hit):
            continue

        hit_indices = np.flatnonzero(hit)
        best = hit_indices[int(np.argmin(alpha[hit]))]
        alpha_best = float(np.clip(alpha[best], 0.0, 1.0))
        out[i] = p0[i] + alpha_best * r
        out_tid[i] = int(edge_wet_tids[best])
        out_alpha[i] = alpha_best

    return out, out_tid, out_alpha


def wet_dry_stop_contact_points(
    Pprop_sub,
    Pdet_sub,
    mesh,
    zb_c,
    wse_c,
    hmin,
    tid_prev_sub=None,
    tid_dry_sub=None,
    n_bisect=None,
    wet_cell_mask=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return wet-side contact points for particles that stop at a wet/dry crossing.

    The returned positions are the last wet points found by bisection between the
    previous wet position and the proposed dry position. Particles whose previous
    position is not wet fall back to the previous position.
    """
    proposed_positions = np.asarray(Pprop_sub, dtype=float)
    previous_positions = np.asarray(Pdet_sub, dtype=float)
    bed_elevation = np.asarray(zb_c, dtype=float).ravel()
    water_surface_elevation = np.asarray(wse_c, dtype=float).ravel()

    _validate_subset_inputs(proposed_positions, previous_positions)

    n_particles = proposed_positions.shape[0]
    contact_positions = previous_positions.copy()
    contact_triangle_ids = np.full(n_particles, -1, dtype=int)
    if n_particles == 0:
        return contact_positions, contact_triangle_ids, np.full(n_particles, np.nan, dtype=float)

    if wet_cell_mask is None:
        wet_mask = _build_wet_cell_mask(bed_elevation, water_surface_elevation, hmin)
    else:
        wet_mask = np.asarray(wet_cell_mask, dtype=bool).ravel()
        if wet_mask.size != bed_elevation.size:
            raise ValueError("wet_cell_mask must have the same length as zb_c")
    point_triangle_ids_global, point_triangle_ids, wet_from_triangle_ids = _build_point_location_helpers(mesh, wet_mask)

    if tid_prev_sub is None:
        previous_triangle_ids = point_triangle_ids_global(previous_positions)
    else:
        previous_triangle_ids = np.asarray(tid_prev_sub, dtype=int).ravel()
        if previous_triangle_ids.size != n_particles:
            raise ValueError("tid_prev_sub must have the same length as Pprop_sub")
    dry_triangle_ids = None
    if tid_dry_sub is not None:
        dry_triangle_ids = np.asarray(tid_dry_sub, dtype=int).ravel()
        if dry_triangle_ids.size != n_particles:
            raise ValueError("tid_dry_sub must have the same length as Pprop_sub")

    contact_triangle_ids[:] = previous_triangle_ids
    previous_is_wet = wet_from_triangle_ids(previous_triangle_ids)
    if not np.any(previous_is_wet):
        return contact_positions, contact_triangle_ids, np.full(n_particles, np.nan, dtype=float)

    boundary_alpha = np.full(n_particles, np.nan, dtype=float)
    mesh_tri = np.asarray(getattr(mesh, "tri", np.empty((0, 3), dtype=int)), dtype=int)
    if mesh_tri.ndim == 2 and mesh_tri.shape[1] == 3 and mesh_tri.shape[0] == wet_mask.size:
        boundary_edges, boundary_wet_tids = _wet_dry_boundary_edges(mesh_tri, wet_mask)
    else:
        boundary_edges = np.empty((0, 2), dtype=int)
        boundary_wet_tids = np.empty(0, dtype=int)
    if boundary_edges.size:
        edge_a = np.asarray(mesh.xy, dtype=float)[boundary_edges[:, 0]]
        edge_b = np.asarray(mesh.xy, dtype=float)[boundary_edges[:, 1]]
        active_indices = np.where(previous_is_wet)[0]
        edge_hits, edge_hit_tids, edge_hit_alpha = _first_segment_boundary_intersections(
            previous_positions[active_indices],
            proposed_positions[active_indices],
            edge_a,
            edge_b,
            boundary_wet_tids,
        )
        edge_hit_ok = edge_hit_tids >= 0
        if np.any(edge_hit_ok):
            hit_indices = active_indices[edge_hit_ok]
            contact_positions[hit_indices] = edge_hits[edge_hit_ok]
            contact_triangle_ids[hit_indices] = edge_hit_tids[edge_hit_ok]
            boundary_alpha[hit_indices] = edge_hit_alpha[edge_hit_ok]

    if (
        dry_triangle_ids is not None
        and hasattr(mesh, "neighbors")
        and mesh_tri.ndim == 2
        and mesh_tri.shape[1] == 3
        and mesh_tri.shape[0] == wet_mask.size
    ):
        tri = mesh_tri
        xy = np.asarray(mesh.xy, dtype=float)
        neighbors = np.asarray(mesh.neighbors, dtype=int)
        edge_node_cols = ((1, 2), (2, 0), (0, 1))
        for i in np.where(previous_is_wet & ~np.isfinite(boundary_alpha))[0]:
            dry_tid = int(dry_triangle_ids[i])
            if dry_tid < 0 or dry_tid >= neighbors.shape[0]:
                continue

            p0 = previous_positions[i]
            p1 = proposed_positions[i]
            r = p1 - p0
            r_len2 = float(np.dot(r, r))
            if r_len2 <= 1.0e-30:
                continue

            best_alpha = np.inf
            best_point = None
            best_tid = -1
            best_dist2 = np.inf
            for edge_id, wet_tid in enumerate(neighbors[dry_tid]):
                if wet_tid < 0 or wet_tid >= wet_mask.size or not bool(wet_mask[wet_tid]):
                    continue

                cols = edge_node_cols[edge_id]
                a = xy[tri[dry_tid, cols[0]]]
                b = xy[tri[dry_tid, cols[1]]]
                edge_vec = b - a
                denom = r[0] * edge_vec[1] - r[1] * edge_vec[0]
                if abs(float(denom)) > 1.0e-12:
                    qp = a - p0
                    alpha = float((qp[0] * edge_vec[1] - qp[1] * edge_vec[0]) / denom)
                    beta = float((qp[0] * r[1] - qp[1] * r[0]) / denom)
                    if -1.0e-10 <= alpha <= 1.0 + 1.0e-10 and -1.0e-10 <= beta <= 1.0 + 1.0e-10:
                        alpha_clipped = float(np.clip(alpha, 0.0, 1.0))
                        if alpha_clipped < best_alpha:
                            best_alpha = alpha_clipped
                            best_point = p0 + alpha_clipped * r
                            best_tid = int(wet_tid)
                            continue

                edge_len2 = float(np.dot(edge_vec, edge_vec))
                if edge_len2 <= 1.0e-30:
                    continue
                beta_nearest = float(np.clip(np.dot(p0 - a, edge_vec) / edge_len2, 0.0, 1.0))
                candidate = a + beta_nearest * edge_vec
                dist2 = float(np.sum((candidate - p0) ** 2))
                if best_point is None and dist2 < best_dist2:
                    alpha_nearest = float(np.clip(np.dot(candidate - p0, r) / r_len2, 0.0, 1.0))
                    best_dist2 = dist2
                    best_alpha = alpha_nearest
                    best_point = candidate
                    best_tid = int(wet_tid)

            if best_point is not None:
                contact_positions[i] = best_point
                contact_triangle_ids[i] = best_tid
                boundary_alpha[i] = best_alpha

    n_bisect_resolved = _resolve_wetdry_bisection_steps(n_bisect)
    need_fallback = previous_is_wet & ~np.isfinite(boundary_alpha)
    if not np.any(need_fallback):
        return contact_positions, contact_triangle_ids, boundary_alpha

    wet_side, wet_side_triangle_ids, _dry_side = _bisect_wet_dry_contact(
        previous_positions[need_fallback],
        previous_triangle_ids[need_fallback],
        proposed_positions[need_fallback],
        point_triangle_ids,
        wet_from_triangle_ids,
        n_bisect_resolved,
    )

    active_indices = np.where(need_fallback)[0]
    segment = proposed_positions[active_indices] - previous_positions[active_indices]
    contact_points_fallback = wet_side.copy()
    contact_tids_fallback = wet_side_triangle_ids.copy()

    if boundary_edges.size:
        edge_a = np.asarray(mesh.xy, dtype=float)[boundary_edges[:, 0]]
        edge_b = np.asarray(mesh.xy, dtype=float)[boundary_edges[:, 1]]
        edge_ab = edge_b - edge_a
        edge_len2 = np.sum(edge_ab * edge_ab, axis=1)
        valid_edges = edge_len2 > 1.0e-30
        if np.any(valid_edges):
            for local_i, point in enumerate(wet_side):
                t_edge = np.sum((point - edge_a[valid_edges]) * edge_ab[valid_edges], axis=1) / edge_len2[valid_edges]
                t_edge = np.clip(t_edge, 0.0, 1.0)
                candidates = edge_a[valid_edges] + t_edge[:, None] * edge_ab[valid_edges]
                dist2 = np.sum((candidates - point) ** 2, axis=1)
                best_local = int(np.argmin(dist2))
                valid_edge_indices = np.flatnonzero(valid_edges)
                best_edge = int(valid_edge_indices[best_local])
                contact_points_fallback[local_i] = candidates[best_local]
                contact_tids_fallback[local_i] = boundary_wet_tids[best_edge]

    contact_positions[active_indices] = contact_points_fallback
    contact_triangle_ids[active_indices] = contact_tids_fallback
    contact = contact_points_fallback - previous_positions[active_indices]
    segment_len2 = np.sum(segment * segment, axis=1)
    good_segment = segment_len2 > 1.0e-30
    fallback_alpha = np.zeros(active_indices.size, dtype=float)
    fallback_alpha[good_segment] = np.sum(contact[good_segment] * segment[good_segment], axis=1) / segment_len2[good_segment]
    boundary_alpha[active_indices] = np.clip(fallback_alpha, 0.0, 1.0)
    return contact_positions, contact_triangle_ids, boundary_alpha


def reflect_slide_wetdry_subset_fullcache(
    Pprop_sub,
    Pdet_sub,
    mesh,
    u_c,
    v_c,
    zb_c,
    wse_c,
    hmin,
    dt,
    tid_prev_sub=None,
    eps_push=1e-3,
    n_bisect=None,
    tol_bc=1e-9,
):
    """
    Project particles that cross from wet into dry regions back to the wet/dry
    interface, then slide them tangentially along the boundary using the local
    cell-centered velocity.
    """
    proposed_positions = np.asarray(Pprop_sub, dtype=float)
    previous_positions = np.asarray(Pdet_sub, dtype=float)
    velocity_u = np.asarray(u_c, dtype=float).ravel()
    velocity_v = np.asarray(v_c, dtype=float).ravel()
    bed_elevation = np.asarray(zb_c, dtype=float).ravel()
    water_surface_elevation = np.asarray(wse_c, dtype=float).ravel()

    _validate_subset_inputs(proposed_positions, previous_positions)

    n_particles = proposed_positions.shape[0]
    if n_particles == 0:
        return proposed_positions.copy()

    corrected_positions = proposed_positions.copy()
    wet_cell_mask = _build_wet_cell_mask(bed_elevation, water_surface_elevation, hmin)
    point_triangle_ids_global, point_triangle_ids, wet_from_triangle_ids = _build_point_location_helpers(mesh, wet_cell_mask)

    if tid_prev_sub is None:
        previous_triangle_ids = point_triangle_ids_global(previous_positions)
    else:
        previous_triangle_ids = np.asarray(tid_prev_sub, dtype=int).ravel()
        if previous_triangle_ids.size != n_particles:
            raise ValueError("tid_prev_sub must have the same length as Pprop_sub")
    previous_is_wet = wet_from_triangle_ids(previous_triangle_ids)

    invalid_previous_positions = ~previous_is_wet
    if np.any(invalid_previous_positions):
        corrected_positions[invalid_previous_positions] = previous_positions[invalid_previous_positions]

    active_particles = previous_is_wet
    if not np.any(active_particles):
        return corrected_positions

    previous_wet_positions = previous_positions[active_particles].copy()
    previous_wet_triangle_ids = previous_triangle_ids[active_particles].copy()
    proposed_dry_positions = proposed_positions[active_particles].copy()
    n_bisect_resolved = _resolve_wetdry_bisection_steps(n_bisect)
    wet_side, wet_side_triangle_ids, dry_side = _bisect_wet_dry_contact(
        previous_wet_positions,
        previous_wet_triangle_ids,
        proposed_dry_positions,
        point_triangle_ids,
        wet_from_triangle_ids,
        n_bisect_resolved,
    )

    interface_normal = dry_side - wet_side
    interface_normal_norm = np.linalg.norm(interface_normal, axis=1)
    active_indices = np.where(active_particles)[0]

    degenerate = interface_normal_norm < 1e-12
    if np.any(degenerate):
        corrected_positions[active_indices[degenerate]] = wet_side[degenerate]

    valid_normals = ~degenerate
    if not np.any(valid_normals):
        return corrected_positions

    wet_side_valid = wet_side[valid_normals]
    interface_normal_valid = interface_normal[valid_normals] / interface_normal_norm[valid_normals, None]
    wet_side_triangle_ids_valid = wet_side_triangle_ids[valid_normals]

    contact_points = wet_side_valid - eps_push * interface_normal_valid
    contact_triangle_ids = point_triangle_ids(contact_points, wet_side_triangle_ids_valid)
    contact_is_wet = wet_from_triangle_ids(contact_triangle_ids)

    valid_indices = active_indices[valid_normals]
    if np.any(~contact_is_wet):
        corrected_positions[valid_indices[~contact_is_wet]] = wet_side_valid[~contact_is_wet]

    movable = contact_is_wet
    if not np.any(movable):
        return corrected_positions

    contact_points_movable = contact_points[movable]
    interface_normal_movable = interface_normal_valid[movable]
    contact_triangle_ids_movable = contact_triangle_ids[movable]
    movable_indices = valid_indices[movable]

    local_velocity = np.empty((contact_triangle_ids_movable.size, 2), dtype=float)
    local_velocity[:, 0] = velocity_u[contact_triangle_ids_movable]
    local_velocity[:, 1] = velocity_v[contact_triangle_ids_movable]

    velocity_dot_normal = np.sum(local_velocity * interface_normal_movable, axis=1)
    tangent_velocity = local_velocity - velocity_dot_normal[:, None] * interface_normal_movable

    tangent_speed = np.linalg.norm(tangent_velocity, axis=1)
    updated_positions = contact_points_movable.copy()

    can_slide = tangent_speed >= 1e-10
    if np.any(can_slide):
        updated_positions[can_slide] = (
            contact_points_movable[can_slide] + dt * tangent_velocity[can_slide]
        )

    barycentric_coords = mesh.barycentric(contact_triangle_ids_movable, updated_positions)
    inside_contact_cell = np.all(
        (barycentric_coords >= -tol_bc) & (barycentric_coords <= 1.0 + tol_bc),
        axis=1,
    )

    updated_triangle_ids = np.full(contact_triangle_ids_movable.shape, -1, dtype=int)
    if np.any(inside_contact_cell):
        updated_triangle_ids[inside_contact_cell] = contact_triangle_ids_movable[inside_contact_cell]
    if np.any(~inside_contact_cell):
        updated_triangle_ids[~inside_contact_cell] = point_triangle_ids(
            updated_positions[~inside_contact_cell],
            contact_triangle_ids_movable[~inside_contact_cell],
        )

    updated_is_wet = wet_from_triangle_ids(updated_triangle_ids)
    if np.any(~updated_is_wet):
        updated_positions[~updated_is_wet] = contact_points_movable[~updated_is_wet]

    corrected_positions[movable_indices] = updated_positions
    return corrected_positions
