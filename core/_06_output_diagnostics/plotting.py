"""Plotting utilities used by HydroLPT solver and GUI workflows."""

from __future__ import annotations

import os
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter1d


def _configure_plot_fonts() -> None:
    """Use a UI font that matches the Studio panel as closely as possible."""
    if os.name == "nt":
        plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    else:
        plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]


_configure_plot_fonts()


LEGEND_STYLE = {
    "ncol": 1,
    "frameon": True,
    "fontsize": 8.5,
    "title": "Markers",
    "borderaxespad": 0.0,
    "handlelength": 1.2,
    "handletextpad": 0.6,
    "borderpad": 0.35,
    "labelspacing": 0.35,
}

FIGURE1_TITLE_FONTSIZE = 10.5


def _final_state_masks(xEnd, yEnd, flags, state):
    """
    Build mutually exclusive final-state masks with priority:
    outside > limiter > dry > deposited > surfaced 
    > near bed mobile > near surface mobile > suspended
    """
    ok_end = np.isfinite(xEnd) & np.isfinite(yEnd)

    masks = {
        "outside": ok_end & np.asarray(flags["stoppedByOutside"], dtype=bool),
        "limiter": ok_end & np.asarray(flags["stoppedByLimiter"], dtype=bool),
        "dry": ok_end & np.asarray(flags["stoppedByDry"], dtype=bool),
        "deposited": ok_end & np.asarray(flags["isDeposited"], dtype=bool),
        "surfaced": ok_end & np.asarray(state["isSurfaced"], dtype=bool),
        "near_bed_mobile": ok_end & np.asarray(state["isNearBedMobile"], dtype=bool),
        "near_surface_mobile": ok_end & np.asarray(state["isNearSurfaceMobile"], dtype=bool),
        "suspended": ok_end & np.asarray(state["isSuspended"], dtype=bool),
    }

    used = np.zeros_like(ok_end, dtype=bool)
    order = [
        "outside",
        "limiter",
        "dry",
        "deposited",
        "surfaced",
        "near_bed_mobile",
        "near_surface_mobile",
        "suspended",
    ]
    for key in order:
        masks[key] &= ~used
        used |= masks[key]

    return masks


def _triangle_nodal_idw(triangles, xy, values, node_count: int, power: float = 2.0) -> np.ndarray:
    """Reconstruct triangle-centered values at nodes using inverse-distance weights."""
    tri = np.asarray(triangles, dtype=int)
    xy_arr = np.asarray(xy, dtype=float)
    cell_values = np.asarray(values, dtype=float).ravel()
    numerator = np.zeros(int(node_count), dtype=float)
    denominator = np.zeros(int(node_count), dtype=float)

    valid = np.isfinite(cell_values)
    if (
        tri.ndim != 2
        or tri.shape[1] != 3
        or xy_arr.ndim != 2
        or xy_arr.shape[1] != 2
        or cell_values.size != tri.shape[0]
    ):
        return np.full(int(node_count), np.nan, dtype=float)
    if not np.any(valid):
        return np.full(int(node_count), np.nan, dtype=float)

    centers = np.mean(xy_arr[tri], axis=1)
    eps = 1.0e-12
    for local_col in range(3):
        node_ids = tri[:, local_col]
        dist = np.linalg.norm(xy_arr[node_ids] - centers, axis=1)
        weights = np.where(valid, 1.0 / np.maximum(dist, eps) ** float(power), 0.0)
        np.add.at(numerator, node_ids, np.where(valid, cell_values, 0.0) * weights)
        np.add.at(denominator, node_ids, weights)

    out = np.full(int(node_count), np.nan, dtype=float)
    good = denominator > 0.0
    out[good] = numerator[good] / denominator[good]
    return out


def _refined_triangle_grid(xy, triangles, max_points: int = 1_200_000):
    """Return refined triangle points, connectivity, and source triangle ids."""
    xy_arr = np.asarray(xy, dtype=float)
    tri_arr = np.asarray(triangles, dtype=int)
    if tri_arr.ndim != 2 or tri_arr.shape[1] != 3:
        return xy_arr, tri_arr, None, None

    n_tri = max(int(tri_arr.shape[0]), 1)
    subdiv = 4
    while subdiv > 1 and n_tri * ((subdiv + 1) * (subdiv + 2) // 2) > int(max_points):
        subdiv -= 1
    if subdiv <= 1:
        return xy_arr, tri_arr, None, None

    bary_coords = []
    ij_to_local = {}
    for i in range(subdiv + 1):
        for j in range(subdiv + 1 - i):
            local_id = len(bary_coords)
            ij_to_local[(i, j)] = local_id
            w1 = i / subdiv
            w2 = j / subdiv
            w0 = 1.0 - w1 - w2
            bary_coords.append((w0, w1, w2))
    bary = np.asarray(bary_coords, dtype=float)

    local_triangles = []
    for i in range(subdiv):
        for j in range(subdiv - i):
            a = ij_to_local[(i, j)]
            b = ij_to_local[(i + 1, j)]
            c = ij_to_local[(i, j + 1)]
            local_triangles.append((a, b, c))
            if j < subdiv - i - 1:
                d = ij_to_local[(i + 1, j + 1)]
                local_triangles.append((b, d, c))
    local_triangles = np.asarray(local_triangles, dtype=int)

    points_per_tri = bary.shape[0]
    refined_points = np.empty((tri_arr.shape[0] * points_per_tri, 2), dtype=float)
    refined_triangles = np.empty((tri_arr.shape[0] * local_triangles.shape[0], 3), dtype=int)
    source_triangles = np.repeat(np.arange(tri_arr.shape[0], dtype=int), points_per_tri)
    refined_bary = np.tile(bary, (tri_arr.shape[0], 1))

    for tri_id, node_ids in enumerate(tri_arr):
        p0, p1, p2 = xy_arr[node_ids]
        point_offset = tri_id * points_per_tri
        tri_offset = tri_id * local_triangles.shape[0]
        refined_points[point_offset:point_offset + points_per_tri] = (
            bary[:, [0]] * p0 + bary[:, [1]] * p1 + bary[:, [2]] * p2
        )
        refined_triangles[tri_offset:tri_offset + local_triangles.shape[0]] = local_triangles + point_offset

    return refined_points, refined_triangles, source_triangles, refined_bary


def _triangle_neighbors(triangles) -> np.ndarray:
    """Build edge-neighbor triangle ids for a triangular mesh."""
    tri = np.asarray(triangles, dtype=int)
    neighbors = np.full((tri.shape[0], 3), -1, dtype=int)
    edge_owner: dict[tuple[int, int], tuple[int, int]] = {}
    for tri_id, nodes in enumerate(tri):
        for edge_id, pair in enumerate(((nodes[1], nodes[2]), (nodes[2], nodes[0]), (nodes[0], nodes[1]))):
            key = tuple(sorted((int(pair[0]), int(pair[1]))))
            other = edge_owner.pop(key, None)
            if other is None:
                edge_owner[key] = (tri_id, edge_id)
            else:
                other_tri, other_edge = other
                neighbors[tri_id, edge_id] = other_tri
                neighbors[other_tri, other_edge] = tri_id
    return neighbors


def _native_polygon_collection(mesh, values, *, cmap, alpha: float = 1.0, zorder: float = 1.0):
    """Return a PolyCollection for native polygon-cell meshes."""
    polygons = []
    if not hasattr(mesh, "_polygons") or not hasattr(mesh, "xy"):
        return None
    xy = np.asarray(mesh.xy, dtype=float)
    for poly_ids in mesh._polygons:
        ids = np.asarray(poly_ids, dtype=int)
        if ids.size >= 3:
            polygons.append(xy[ids])
    if not polygons:
        return None
    arr = np.ma.masked_invalid(np.asarray(values, dtype=float).ravel()[: len(polygons)])
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=(0.74, 0.74, 0.74, 1.0))
    collection = PolyCollection(
        polygons,
        array=arr,
        cmap=cmap_obj,
        edgecolors="face",
        linewidths=0.0,
        alpha=alpha,
        antialiaseds=False,
        zorder=zorder,
    )
    collection.set_rasterized(True)
    return collection


def _native_mask_polygon_collection(mesh, mask, *, facecolor, alpha: float = 1.0, zorder: float = 1.0):
    """Return a constant-color PolyCollection for selected native mesh cells."""
    if not hasattr(mesh, "_polygons") or not hasattr(mesh, "xy"):
        return None
    mask_arr = np.asarray(mask, dtype=bool).ravel()
    xy = np.asarray(mesh.xy, dtype=float)
    polygons = []
    for cell_id, poly_ids in enumerate(mesh._polygons):
        if cell_id >= mask_arr.size or not bool(mask_arr[cell_id]):
            continue
        ids = np.asarray(poly_ids, dtype=int)
        if ids.size >= 3:
            polygons.append(xy[ids])
    if not polygons:
        return None
    collection = PolyCollection(
        polygons,
        facecolors=facecolor,
        edgecolors="face",
        linewidths=0.0,
        alpha=alpha,
        antialiaseds=False,
        zorder=zorder,
    )
    collection.set_rasterized(True)
    return collection


def _native_mask_clip_patch(mesh, mask, ax):
    """Return a clip patch following selected native mesh polygons."""
    if not hasattr(mesh, "_polygons") or not hasattr(mesh, "xy"):
        return None
    mask_arr = np.asarray(mask, dtype=bool).ravel()
    xy = np.asarray(mesh.xy, dtype=float)
    vertices = []
    codes = []
    for cell_id, poly_ids in enumerate(mesh._polygons):
        if cell_id >= mask_arr.size or not bool(mask_arr[cell_id]):
            continue
        ids = np.asarray(poly_ids, dtype=int)
        if ids.size < 3:
            continue
        pts = xy[ids]
        vertices.extend(pts.tolist())
        codes.extend([Path.MOVETO] + [Path.LINETO] * (pts.shape[0] - 1))
        vertices.append(pts[0].tolist())
        codes.append(Path.CLOSEPOLY)
    if not vertices:
        return None
    return PathPatch(
        Path(np.asarray(vertices, dtype=float), np.asarray(codes, dtype=np.uint8)),
        transform=ax.transData,
        facecolor="none",
        edgecolor="none",
    )


def _edge_line_collection(xy, edges, *, color, linewidth, alpha, zorder):
    """Return a LineCollection for node-indexed edge segments."""
    edge_arr = np.asarray(edges, dtype=int)
    xy_arr = np.asarray(xy, dtype=float)
    if edge_arr.ndim != 2 or edge_arr.shape[1] != 2 or edge_arr.size == 0:
        return None
    valid = (
        (edge_arr[:, 0] >= 0)
        & (edge_arr[:, 1] >= 0)
        & (edge_arr[:, 0] < xy_arr.shape[0])
        & (edge_arr[:, 1] < xy_arr.shape[0])
    )
    if not np.any(valid):
        return None
    segments = xy_arr[edge_arr[valid]]
    return LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
    )


def _sample_mobility_values_at_points(
    points,
    cell_ids,
    *,
    mesh,
    ustar_source,
    bed_crit,
    surface_crit,
    scalar_c,
    h_c,
    hmin: float,
    rho_p: float,
    rho_f: float,
):
    """Sample the plotted mobility/shear scalar at known native cell ids."""
    points_arr = np.asarray(points, dtype=float)
    tid = np.asarray(cell_ids, dtype=int).ravel()
    out = np.full(tid.shape, np.nan, dtype=float)
    h_arr = np.asarray(h_c, dtype=float).ravel()
    valid = (
        (tid >= 0)
        & (tid < h_arr.size)
        & np.all(np.isfinite(points_arr), axis=1)
        & np.isfinite(h_arr[np.where((tid >= 0) & (tid < h_arr.size), tid, 0)])
        & (h_arr[np.where((tid >= 0) & (tid < h_arr.size), tid, 0)] > hmin)
    )
    if not np.any(valid):
        return out
    centers = np.asarray(mesh.cell_centers, dtype=float)
    neighbors = np.asarray(mesh.neighbors, dtype=int)
    if rho_p > rho_f:
        ustar_q = _local_cell_idw_sample(points_arr, tid, centers, neighbors, ustar_source)
        crit_q = _local_cell_idw_sample(points_arr, tid, centers, neighbors, bed_crit)
        ok = valid & np.isfinite(ustar_q) & np.isfinite(crit_q) & (crit_q > 0.0)
        out[ok] = np.clip(ustar_q[ok] / crit_q[ok], 0.0, 2.0)
    elif rho_p < rho_f:
        ustar_q = _local_cell_idw_sample(points_arr, tid, centers, neighbors, ustar_source)
        crit_q = _local_cell_idw_sample(points_arr, tid, centers, neighbors, surface_crit)
        ok = valid & np.isfinite(ustar_q) & np.isfinite(crit_q) & (crit_q > 0.0)
        out[ok] = np.clip(ustar_q[ok] / crit_q[ok], 0.0, 2.0)
    else:
        sampled = _local_cell_idw_sample(points_arr, tid, centers, neighbors, scalar_c)
        out[valid] = sampled[valid]
    return out


def _native_idw_grid(
    mesh,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    h_c,
    hmin: float,
    ustar_source,
    bed_crit,
    surface_crit,
    scalar_c,
    rho_p: float,
    rho_f: float,
    max_pixels: int = 2_000_000,
):
    """Build a smooth native-mesh IDW raster for HEC-RAS polygon plots."""
    span_x = max(float(x_max) - float(x_min), 1.0e-12)
    span_y = max(float(y_max) - float(y_min), 1.0e-12)
    max_pixels = max(int(max_pixels), 10_000)
    nx = max(64, int(np.sqrt(max_pixels * span_x / span_y)))
    ny = max(64, int(np.ceil(max_pixels / nx)))
    if nx * ny > max_pixels:
        scale = np.sqrt(max_pixels / float(nx * ny))
        nx = max(64, int(nx * scale))
        ny = max(64, int(ny * scale))

    xs = np.linspace(float(x_min), float(x_max), nx, dtype=float)
    ys = np.linspace(float(y_min), float(y_max), ny, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    tid = np.asarray(mesh.point_location(points[:, 0], points[:, 1]), dtype=int)
    valid = (tid >= 0) & (tid < np.asarray(h_c).size)
    wet = np.zeros(tid.shape, dtype=bool)
    h_arr = np.asarray(h_c, dtype=float).ravel()
    wet[valid] = np.isfinite(h_arr[tid[valid]]) & (h_arr[tid[valid]] > hmin)

    centers = np.asarray(mesh.cell_centers, dtype=float)
    neighbors = np.asarray(mesh.neighbors, dtype=int)
    if rho_p > rho_f:
        ustar_q = _local_cell_idw_sample(points, tid, centers, neighbors, ustar_source)
        crit_q = _local_cell_idw_sample(points, tid, centers, neighbors, bed_crit)
        values = np.full(tid.shape, np.nan, dtype=float)
        ok = wet & np.isfinite(ustar_q) & np.isfinite(crit_q) & (crit_q > 0.0)
        values[ok] = np.clip(ustar_q[ok] / crit_q[ok], 0.0, 2.0)
    elif rho_p < rho_f:
        ustar_q = _local_cell_idw_sample(points, tid, centers, neighbors, ustar_source)
        crit_q = _local_cell_idw_sample(points, tid, centers, neighbors, surface_crit)
        values = np.full(tid.shape, np.nan, dtype=float)
        ok = wet & np.isfinite(ustar_q) & np.isfinite(crit_q) & (crit_q > 0.0)
        values[ok] = np.clip(ustar_q[ok] / crit_q[ok], 0.0, 2.0)
    else:
        values = _local_cell_idw_sample(points, tid, centers, neighbors, scalar_c)
        values[~wet] = np.nan

    values[~valid] = np.nan
    values[~wet] = np.nan
    return xs, ys, values.reshape((ny, nx))


def _mask_boundary_edges(triangles, mask) -> np.ndarray:
    """Return edges around the True region of a triangle mask."""
    tri = np.asarray(triangles, dtype=int)
    mask_arr = np.asarray(mask, dtype=bool).ravel()
    if tri.ndim != 2 or tri.shape[1] != 3 or mask_arr.size != tri.shape[0]:
        return np.empty((0, 2), dtype=int)

    edge_owner: dict[tuple[int, int], tuple[bool, bool]] = {}
    for tri_id, nodes in enumerate(tri):
        is_true = bool(mask_arr[tri_id])
        for pair in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            key = tuple(sorted((int(pair[0]), int(pair[1]))))
            previous = edge_owner.get(key)
            if previous is None:
                edge_owner[key] = (is_true, False)
            else:
                edge_owner[key] = (previous[0], is_true)

    edges = [
        edge
        for edge, (left_true, right_true) in edge_owner.items()
        if left_true != right_true or (left_true and not right_true)
    ]
    return np.asarray(edges, dtype=int).reshape((-1, 2)) if edges else np.empty((0, 2), dtype=int)


def _local_cell_idw_sample(points, source_triangles, cell_centers, neighbors, values, power: float = 2.0) -> np.ndarray:
    """Sample cell-centered values at points from containing cells plus edge neighbors."""
    points_arr = np.asarray(points, dtype=float)
    source = np.asarray(source_triangles, dtype=int).ravel()
    centers = np.asarray(cell_centers, dtype=float)
    nbr = np.asarray(neighbors, dtype=int)
    cell_values = np.asarray(values, dtype=float).ravel()
    out = np.full(source.shape, np.nan, dtype=float)
    valid = source >= 0
    if not np.any(valid):
        return out

    candidate_cells = np.column_stack([source[valid], nbr[source[valid]]])
    candidate_ok = candidate_cells >= 0
    candidate_cells_safe = np.where(candidate_ok, candidate_cells, 0)
    candidate_centers = centers[candidate_cells_safe]
    dist = np.linalg.norm(candidate_centers - points_arr[valid, None, :], axis=2)
    candidate_values = cell_values[candidate_cells_safe]
    finite = np.isfinite(candidate_values)
    weights = np.where(
        candidate_ok & finite,
        1.0 / np.maximum(dist, 1.0e-12) ** float(power),
        0.0,
    )
    denom = np.sum(weights, axis=1)
    good = denom > 0.0
    sampled = np.full(np.count_nonzero(valid), np.nan, dtype=float)
    if np.any(good):
        sampled[good] = np.sum(np.where(finite, candidate_values * weights, 0.0), axis=1)[good] / denom[good]
    out[valid] = sampled
    return out


def _barycentric_from_xy(triangle_xy, points) -> np.ndarray:
    """Compute barycentric weights for points in matching 2D triangles."""
    tri_xy = np.asarray(triangle_xy, dtype=float)
    pts = np.asarray(points, dtype=float)
    out = np.full((pts.shape[0], 3), np.nan, dtype=float)
    if tri_xy.ndim != 3 or tri_xy.shape[1:] != (3, 2) or pts.ndim != 2 or pts.shape[1] != 2:
        return out

    a = tri_xy[:, 0, :]
    b = tri_xy[:, 1, :]
    c = tri_xy[:, 2, :]
    v0 = b - a
    v1 = c - a
    v2 = pts - a
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)
    denom = d00 * d11 - d01 * d01
    good = np.abs(denom) > 1.0e-30
    if np.any(good):
        w1 = (d11[good] * d20[good] - d01[good] * d21[good]) / denom[good]
        w2 = (d00[good] * d21[good] - d01[good] * d20[good]) / denom[good]
        w0 = 1.0 - w1 - w2
        out[good, 0] = w0
        out[good, 1] = w1
        out[good, 2] = w2
    return out


def _sample_plot_field_at_points(mesh, hydraulic_field_mode, points, triangle_ids, values) -> np.ndarray:
    """Sample a cell-centered plot field using the selected hydraulic field mode."""
    points_arr = np.asarray(points, dtype=float)
    tid = np.asarray(triangle_ids, dtype=int).ravel()
    cell_values = np.asarray(values, dtype=float).ravel()
    out = np.full(tid.shape, np.nan, dtype=float)
    valid = (tid >= 0) & np.all(np.isfinite(points_arr), axis=1) & (tid < cell_values.size)
    if not np.any(valid):
        return out

    mode = str(hydraulic_field_mode or "cellwise").strip().lower()
    if mesh is None or mode == "cellwise":
        out[valid] = cell_values[tid[valid]]
        return out

    if mode == "local_idw" and hasattr(mesh, "neighbors"):
        if hasattr(mesh, "cell_centers"):
            centers = np.asarray(mesh.cell_centers, dtype=float)
        elif all(hasattr(mesh, name) for name in ("V0", "V1", "V2")):
            centers = (mesh.V0 + mesh.V1 + mesh.V2) / 3.0
        else:
            out[valid] = cell_values[tid[valid]]
            return out
        out[valid] = _local_cell_idw_sample(
            points_arr[valid],
            tid[valid],
            centers,
            np.asarray(mesh.neighbors, dtype=int),
            cell_values,
        )
        return out

    out[valid] = cell_values[tid[valid]]
    return out


def _compact_tick(value, _pos):
    """Format ticks without unnecessary trailing zeros."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _finite_span(values):
    """Return the finite data span, guarded away from zero."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1e-12
    return max(float(np.ptp(finite)), 1e-12)


def _axis_bounds(values) -> tuple[float, float]:
    """Return finite min/max axis bounds with a safe fallback."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.min(finite)), float(np.max(finite))


def _flume_x_tick_values(x_min: float, x_max: float) -> np.ndarray:
    """Build readable x ticks that always include the flume end."""
    span = max(float(x_max) - float(x_min), 0.0)
    if span <= 0.0:
        return np.asarray([x_min], dtype=float)

    step = 5.0 if span > 10.0 else max(1.0, span / 4.0)
    start = step * np.floor(x_min / step)
    stop = step * np.ceil(x_max / step)
    ticks = np.arange(start, stop + 0.5 * step, step, dtype=float)

    if ticks.size == 0 or abs(ticks[0] - x_min) > 1e-9:
        ticks = np.insert(ticks, 0, x_min)
    if abs(ticks[-1] - x_max) > 1e-9:
        ticks = np.append(ticks, x_max)

    ticks = np.unique(np.round(ticks, 10))
    return ticks[(ticks >= x_min - 1e-9) & (ticks <= x_max + 1e-9)]


def _binary_map_figure_size(x_coords, y_coords):
    """Choose a binary-map figure layout from the domain aspect ratio."""
    Lx = _finite_span(x_coords)
    Ly = _finite_span(y_coords)
    use_wide_flume_layout = (Lx / Ly) >= 8.0
    tall_planform = (Ly / Lx) > 1.3 if not use_wide_flume_layout else False

    if use_wide_flume_layout:
        figsize = (7.0, 8.2)
    else:
        figsize = (8.4, 9.0) if tall_planform else (10.5, 8.0)

    return use_wide_flume_layout, tall_planform, figsize, Lx, Ly
def _estimate_text_width_inches(text_items, fontsize: float, extra_pad: float = 0.0) -> float:
    """Estimate the width needed to render a short list of labels."""

    prop = FontProperties(
        family=plt.rcParams.get("font.family", ["DejaVu Sans"]),
        size=fontsize,
    )
    widths = []
    for item in text_items:
        text = str(item or "")
        if not text:
            continue
        widths.append(TextPath((0, 0), text, prop=prop).get_extents().width / 72.0)
    return (max(widths) if widths else 0.0) + extra_pad


def _estimate_text_block_height_inches(text: str, fontsize: float, line_spacing: float = 1.25, pad: float = 0.18) -> float:
    """Estimate the vertical space needed for a wrapped title block."""

    n_lines = max(1, len(str(text).splitlines()))
    return n_lines * (fontsize / 72.0) * line_spacing + pad


def _estimate_legend_width_inches(
    labels: list[str] | None,
    *,
    title: str = "Markers",
    fontsize: float = 8.5,
) -> float:
    """Estimate legend width from an actual rendered legend box."""

    labels = list(labels or [])
    fig = plt.figure(figsize=(2.0, 2.0))
    ax = fig.add_subplot(111)
    ax.axis("off")
    proxy_handles = [
        Line2D([0], [0], linestyle="-", linewidth=1.6, color="k")
        for _ in labels
    ]
    legend = ax.legend(
        proxy_handles,
        labels,
        loc="upper left",
        ncol=LEGEND_STYLE["ncol"],
        frameon=LEGEND_STYLE["frameon"],
        fontsize=fontsize,
        title=title,
        borderaxespad=LEGEND_STYLE["borderaxespad"],
        handlelength=LEGEND_STYLE["handlelength"],
        handletextpad=LEGEND_STYLE["handletextpad"],
        borderpad=LEGEND_STYLE["borderpad"],
        labelspacing=LEGEND_STYLE["labelspacing"],
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer=renderer)
    width_inches = bbox.width / float(fig.dpi) if fig.dpi else 0.0
    plt.close(fig)
    return width_inches


def _display_height_inches(fig, fallback_inches: float) -> float:
    """Estimate the available display height in inches from the current backend window."""

    dpi = float(fig.get_dpi()) if fig.get_dpi() else 100.0
    manager = getattr(fig.canvas, "manager", None)
    window = getattr(manager, "window", None)

    try:
        if window is not None and hasattr(window, "winfo_screenheight"):
            screen_h = float(window.winfo_screenheight())
            if screen_h > 0:
                return screen_h / dpi
    except Exception:
        pass

    try:
        if window is not None and hasattr(window, "screen"):
            screen = window.screen()
            if screen is not None and hasattr(screen, "availableGeometry"):
                geom = screen.availableGeometry()
                screen_h = float(geom.height())
                if screen_h > 0:
                    return screen_h / dpi
    except Exception:
        pass

    return float(fallback_inches)


def _binary_map_layout(
    x_plot,
    y_plot,
    use_wide_flume_layout: bool,
    tall_planform: bool,
    *,
    display_height_inches: float | None = None,
    legend_labels: list[str] | None = None,
    title_text: str | None = None,
):
    """Build a universal 2-row, 3-column layout from actual content and display size."""

    if use_wide_flume_layout:
        fallback_display_height = 8.2
        display_aspect = 0.22
    elif tall_planform:
        fallback_display_height = 9.0
        display_aspect = 1.0
    else:
        fallback_display_height = 8.0
        display_aspect = 1.0

    display_height_inches = float(display_height_inches or fallback_display_height)
    grid_height = 0.65 * display_height_inches
    title_height = _estimate_text_block_height_inches(title_text or "", fontsize=FIGURE1_TITLE_FONTSIZE)
    outer_pad_inches = 0.12 * display_height_inches
    title_gap_inches = 0.03 * display_height_inches
    title_width = _estimate_text_width_inches(
        ["X"],
        fontsize=FIGURE1_TITLE_FONTSIZE,
        extra_pad=0.05,
    )
    legend_width = max(
        0.80,
        _estimate_legend_width_inches(legend_labels, title="Markers", fontsize=8.5) + 0.05,
    )
    # Keep the colorbar thickness proportional to the map-row height.
    cbar_width = 0.02 * grid_height
    display_ratio = (_finite_span(y_plot) / _finite_span(x_plot)) * display_aspect
    map_width = grid_height / max(display_ratio, 1e-12)

    fig_width = max(legend_width + map_width + cbar_width, title_width, 4.8)
    fig_height = grid_height + title_height + title_gap_inches + 2.0 * outer_pad_inches

    figsize = (fig_width, fig_height)
    width_ratios = [legend_width, map_width, cbar_width]
    height_ratios = [title_height, grid_height]
    return figsize, width_ratios, height_ratios


def _style_grid_frame(ax, color: str) -> None:
    """Show a lightweight debug frame around a grid cell."""

    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0.0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _shrink_legend_column_to_content(fig, legend_ax, legend, axes_to_shift, pad_inches: float = 0.04) -> None:
    """Shrink the legend column to the rendered legend width and shift neighbors left."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer=renderer)
    dpi = float(fig.dpi) if fig.dpi else 100.0
    fig_width = max(float(fig.get_size_inches()[0]), 1e-12)

    desired_width_frac = min(
        (bbox.width / dpi + pad_inches) / fig_width,
        legend_ax.get_position().width,
    )
    pos = legend_ax.get_position()
    delta = pos.width - desired_width_frac
    if delta <= 1e-6:
        return

    legend_ax.set_position([pos.x0, pos.y0, desired_width_frac, pos.height])
    for other_ax in axes_to_shift:
        other_pos = other_ax.get_position()
        other_ax.set_position([other_pos.x0 - delta, other_pos.y0, other_pos.width, other_pos.height])


def _fit_map_axis_to_cell_height(
    fig,
    ax,
    cell_ax,
    *,
    display_ratio: float,
) -> None:
    """Resize the map axis so the displayed map fills the grid cell height."""

    fig.canvas.draw()
    cell_pos = cell_ax.get_position()
    fig_width_in, fig_height_in = fig.get_size_inches()
    if fig_width_in <= 0 or fig_height_in <= 0:
        return

    cell_height_in = cell_pos.height * fig_height_in
    desired_width_in = cell_height_in / max(float(display_ratio), 1e-12)
    desired_width_frac = desired_width_in / fig_width_in
    width_frac = min(cell_pos.width, desired_width_frac)
    x0 = cell_pos.x0 + 0.5 * (cell_pos.width - width_frac)
    ax.set_position([x0, cell_pos.y0, width_frac, cell_pos.height])


def _enforce_main_row_layout(
    fig,
    *,
    title_ax,
    legend_ax,
    map_ax,
    map_frame_ax,
    cax,
    cbar_frame_ax,
    title_text: str,
    legend,
    display_ratio: float,
    row2_height_inches: float,
    outer_pad_frac: float = 0.012,
    inter_col_gap_frac: float = 0.04,
    title_gap_frac: float = 0.003,
    cbar_right_pad_frac: float = 0.01,
    cbar_bottom_pad_frac: float = 0.012,
) -> None:
    """Lay out Figure 1 explicitly so the map/colorbar row keeps the target height."""

    fig.canvas.draw()
    fig_w, fig_h = fig.get_size_inches()
    if fig_w <= 0 or fig_h <= 0:
        return

    title_h_in = _estimate_text_block_height_inches(title_text or "", fontsize=FIGURE1_TITLE_FONTSIZE)
    title_h_frac = title_h_in / fig_h
    row2_h_frac = row2_height_inches / fig_h
    row2_y0 = outer_pad_frac
    title_y0 = row2_y0 + row2_h_frac + title_gap_frac

    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer=renderer)
    legend_w_frac = min(max((legend_bbox.width / fig.dpi + 0.03) / fig_w, 0.11), 0.28)

    cbar_w_frac = min(max((0.03 * row2_h_frac * fig_h) / fig_w, 0.04), 0.075)
    x_left = outer_pad_frac
    x_right = 1.0 - outer_pad_frac - cbar_right_pad_frac
    available_w = x_right - x_left
    map_w_frac = max(
        available_w - legend_w_frac - cbar_w_frac - 2.0 * inter_col_gap_frac,
        0.12,
    )
    map_target_w = min(map_w_frac, (row2_h_frac * fig_h) / max(display_ratio, 1e-12) / fig_w)
    map_x0 = x_left + legend_w_frac + inter_col_gap_frac + 0.5 * (map_w_frac - map_target_w)

    title_ax.set_position([x_left, title_y0, available_w, title_h_frac])
    legend_ax.set_position([x_left, row2_y0, legend_w_frac, row2_h_frac])
    map_frame_ax.set_position([map_x0, row2_y0, map_target_w, row2_h_frac])
    map_ax.set_position([map_x0, row2_y0, map_target_w, row2_h_frac])
    cbar_x0 = min(map_x0 + map_target_w + inter_col_gap_frac + 0.03, 1.0 - cbar_right_pad_frac - cbar_w_frac)
    cbar_y0 = row2_y0 + cbar_bottom_pad_frac
    cbar_h = max(row2_h_frac - cbar_bottom_pad_frac, 0.1)
    cbar_frame_ax.set_position([cbar_x0, cbar_y0, cbar_w_frac, cbar_h])
    cax.set_position([cbar_x0, cbar_y0, cbar_w_frac, cbar_h])


def _map_display_ratio_from_limits(ax, *, use_wide_flume_layout: bool) -> float:
    """Return the rendered map height/width ratio implied by the current data limits."""

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_span = max(abs(float(x1) - float(x0)), 1e-12)
    y_span = max(abs(float(y1) - float(y0)), 1e-12)
    aspect_scale = 0.22 if use_wide_flume_layout else 1.0
    return (y_span / x_span) * aspect_scale


def _sync_frame_to_axis(fig, frame_ax, source_ax) -> None:
    """Make a debug frame use the exact final position of another axis."""

    fig.canvas.draw()
    frame_ax.set_position(source_ax.get_position())


def _stabilize_window_width(fig, delta_px: int = 2) -> None:
    """Force one width relayout cycle to settle backend window geometry."""

    manager = getattr(fig.canvas, "manager", None)
    if manager is None or not hasattr(manager, "resize"):
        return

    try:
        dpi = float(fig.get_dpi()) if fig.get_dpi() else 100.0
        width_in, height_in = fig.get_size_inches()
        width_px = int(round(width_in * dpi))
        height_px = int(round(height_in * dpi))
        manager.resize(width_px + delta_px, height_px)
        manager.resize(width_px, height_px)
        fig.canvas.draw_idle()
    except Exception:
        return


def _schedule_post_show_stabilization(
    fig,
    *,
    ax,
    map_frame_ax,
    yaxis_frame_ax,
    left_pad_inches: float,
    bottom_pad_inches: float,
    right_pad_inches: float,
    map_width_inches: float,
    map_height_inches: float,
) -> None:
    """Run one deferred layout pass after the GUI window is actually shown."""

    manager = getattr(fig.canvas, "manager", None)
    window = getattr(manager, "window", None)
    if window is None or not hasattr(window, "after"):
        return

    state = {"done": False}

    def _finalize() -> None:
        if state["done"] or not plt.fignum_exists(fig.number):
            return
        state["done"] = True
        try:
            fig.canvas.draw()
            ytick_texts = [tick.get_text() for tick in ax.get_yticklabels()]
            tick_label_width_inches = _estimate_text_width_inches(ytick_texts, fontsize=10.0, extra_pad=0.0)
            ylabel = ax.get_ylabel()
            yaxis_label_width_inches = (10.0 / 72.0) * 1.6 if ylabel else 0.0
            yaxis_cell_width_inches = tick_label_width_inches + yaxis_label_width_inches + 0.18

            fig_width = left_pad_inches + map_width_inches + yaxis_cell_width_inches + right_pad_inches
            fig_height = map_height_inches + 0.8
            fig.set_size_inches(fig_width, fig_height, forward=True)
            if manager is not None and hasattr(manager, "resize"):
                dpi = float(fig.get_dpi()) if fig.get_dpi() else 100.0
                width_px = int(round(fig_width * dpi))
                height_px = int(round(fig_height * dpi))
                manager.resize(width_px + 2, height_px)
                manager.resize(width_px, height_px)

            map_pos = [
                left_pad_inches / fig_width,
                bottom_pad_inches / fig_height,
                map_width_inches / fig_width,
                map_height_inches / fig_height,
            ]
            yaxis_cell_pos = [
                (left_pad_inches + map_width_inches) / fig_width,
                bottom_pad_inches / fig_height,
                yaxis_cell_width_inches / fig_width,
                map_height_inches / fig_height,
            ]
            ax.set_position(map_pos)
            map_frame_ax.set_position(map_pos)
            yaxis_frame_ax.set_position(yaxis_cell_pos)
            fig.canvas.draw_idle()
        except Exception:
            return

    try:
        window.after(50, _finalize)
    except Exception:
        return


def _match_axis_height_to_reference(target_ax, reference_ax, *, keep_x0: bool = True) -> None:
    """Match an axis height and vertical position to a reference axis."""

    ref_pos = reference_ax.get_position()
    target_pos = target_ax.get_position()
    x0 = target_pos.x0 if keep_x0 else ref_pos.x0
    target_ax.set_position([x0, ref_pos.y0, target_pos.width, ref_pos.height])


def _place_title_close_to_columns(
    fig,
    title_ax,
    *,
    reference_axes,
    title_artist,
    gap_frac: float = 0.0015,
    top_pad_frac: float = 0.001,
) -> None:
    """Place the title block directly above the actual column boxes."""

    fig.canvas.draw()
    fig_bbox = fig.bbox
    if fig_bbox.width <= 0 or fig_bbox.height <= 0:
        return
    renderer = fig.canvas.get_renderer()
    text_bbox = title_artist.get_window_extent(renderer=renderer)
    title_h_frac = min(max(text_bbox.height / fig_bbox.height + 0.003, 0.03), 0.10)

    left = min(ax.get_position().x0 for ax in reference_axes)
    right = max(ax.get_position().x1 for ax in reference_axes)
    top_of_columns = max(ax.get_position().y1 for ax in reference_axes)
    y0 = min(top_of_columns + gap_frac, 1.0 - top_pad_frac - title_h_frac)
    title_ax.set_position([left, y0, right - left, title_h_frac])
    title_artist.set_position((0.5, 0.5))


def make_figure_shields(
    d50_c,
    ustar_c,
    ustarCrit_c,
    h_c,
    hmin,
    nu,
    rho_f,
    rho_p,
    g,
    Dp,
    beta_p,
    tanphi_ratio,
):
    """Create the Shields-style mobility diagnostic plot."""

    beta_s = 1.5
    c2 = -0.6
    density_contrast = abs(rho_p - rho_f)

    d50_c = np.asarray(d50_c, dtype=float).ravel()
    ustar_c = np.asarray(ustar_c, dtype=float).ravel()
    ustarCrit_c = np.asarray(ustarCrit_c, dtype=float).ravel()
    h_c = np.asarray(h_c, dtype=float).ravel()

    Re_line = np.logspace(-1, 4, 400)

    theta_cr_s_curve = (
        0.165 * (Re_line + 0.6) ** (-0.8)
        + 0.045 * np.exp(-40.0 * Re_line ** (-1.3))
    )

    # ---------------------------------------------------------
    # Current condition
    # ---------------------------------------------------------
    ok_u = (
        (h_c > hmin)
        & np.isfinite(d50_c) & (d50_c > 0)
        & np.isfinite(ustar_c) & (ustar_c > 0)
    )

    ReStar_u = np.full_like(d50_c, np.nan)
    theta_p_u = np.full_like(d50_c, np.nan)
    factor_u = np.full_like(d50_c, np.nan)
    theta_s_eq_u = np.full_like(d50_c, np.nan)

    ReStar_u[ok_u] = (ustar_c[ok_u] * d50_c[ok_u]) / nu
    theta_p_u[ok_u] = (rho_f * ustar_c[ok_u] ** 2) / (density_contrast * g * Dp)

    factor_u[ok_u] = (
        (beta_s / beta_p)
        * tanphi_ratio
        * (Dp / d50_c[ok_u]) ** c2
    )

    theta_s_eq_u[ok_u] = theta_p_u[ok_u] / factor_u[ok_u]

    # ---------------------------------------------------------
    # Critical condition
    # ---------------------------------------------------------
    ok_cr = (
        (h_c > hmin)
        & np.isfinite(d50_c) & (d50_c > 0)
        & np.isfinite(ustarCrit_c) & (ustarCrit_c > 0)
    )

    ReStar_cr = np.full_like(d50_c, np.nan)
    theta_p_cr = np.full_like(d50_c, np.nan)
    factor_cr = np.full_like(d50_c, np.nan)
    theta_s_eq_cr = np.full_like(d50_c, np.nan)

    ReStar_cr[ok_cr] = (ustarCrit_c[ok_cr] * d50_c[ok_cr]) / nu
    theta_p_cr[ok_cr] = (rho_f * ustarCrit_c[ok_cr] ** 2) / (density_contrast * g * Dp)

    factor_cr[ok_cr] = (
        (beta_s / beta_p)
        * tanphi_ratio
        * (Dp / d50_c[ok_cr]) ** c2
    )

    theta_s_eq_cr[ok_cr] = theta_p_cr[ok_cr] / factor_cr[ok_cr]

    # ---------------------------------------------------------
    # Mobility number
    # ---------------------------------------------------------
    ok_mob = (
        (h_c > hmin)
        & np.isfinite(ustar_c) & (ustar_c > 0)
        & np.isfinite(ustarCrit_c) & (ustarCrit_c > 0)
    )

    mobility = np.full_like(d50_c, np.nan)
    mobility[ok_mob] = ustar_c[ok_mob] / ustarCrit_c[ok_mob]

    valid = (
        ok_u
        & np.isfinite(ReStar_u)
        & np.isfinite(theta_s_eq_u)
        & np.isfinite(mobility)
        & (ReStar_u > 0)
        & (theta_s_eq_u > 0)
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)

    mobility_cmap = "RdYlGn"
    mobility_vmin = 0.0
    mobility_vmax = 2.0

    sc = ax.scatter(
        ReStar_u[valid],
        theta_s_eq_u[valid],
        c=mobility[valid],
        cmap=mobility_cmap,
        vmin=mobility_vmin,
        vmax=mobility_vmax,
        s=30,
        alpha=0.8,
        edgecolors="none",
        zorder=3,
        label=r"$\theta_{s,eq}$",
    )

    ax.loglog(
        ReStar_cr[ok_cr],
        theta_s_eq_cr[ok_cr],
        ".",
        color="black",
        markersize=7,
        alpha=0.9,
        zorder=4,
        label=r"$\theta_{s,eq,cr}$",
    )

    ax.loglog(
        Re_line,
        theta_cr_s_curve,
        "-",
        color="black",
        linewidth=2,
        zorder=5,
        label=r"$\theta_{s,cr}$ (sediment Shields curve)",
    )

    ax.axvline(5.0, color="black", linestyle="--", linewidth=1.5, zorder=2)
    ax.axvline(70.0, color="black", linestyle=":", linewidth=1.8, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$Re_* = u_* d_{50} / \nu$")
    ax.set_ylabel(
        r"$\theta_{s,eq}=\dfrac{\theta_p}{\left(\dfrac{\beta_s}{\beta_p}\right)\,\left(\dfrac{\tan\phi_p}{\tan\phi_s}\right)\,\left(\dfrac{D_p}{d_{50}}\right)^{c_2}}$"
    )

    ax.set_xlim(1e-1, 1e4)
    ax.set_ylim(1e-3, 10)

    ax.grid(True, which="major", linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.5, zorder=0)

    ax.set_title("Plastic-Shields diagram")

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(
        r"$M=\dfrac{u_*}{u_{*,cr}}=\sqrt{\dfrac{\theta_{s,eq}}{\theta_{s,eq,cr}}}$"
    )
    cb.set_ticks([0, 0.5, 1.0, 1.5, 2.0])

    theta_marker = Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markerfacecolor="gray",
        markeredgecolor="k",
        markersize=5,
        label=r"$\theta_{s,eq}$"
    )

    theta_cr_marker = Line2D(
        [0], [0],
        marker=".",
        linestyle="None",
        color="black",
        markersize=7,
        label=r"$\theta_{s,eq,cr}$"
    )

    shields_line = Line2D(
        [0], [0],
        color="black",
        linewidth=2,
        label="$\\theta_{s,cr}$\n(sediment Shields curve)"
    )

    re5_line = Line2D(
        [0], [0],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=r"$Re_* = 5$"
    )

    re70_line = Line2D(
        [0], [0],
        color="black",
        linestyle=":",
        linewidth=1.8,
        label=r"$Re_* = 70$"
    )

    ax.legend(
        handles=[
            theta_marker,
            theta_cr_marker,
            shields_line,
            re5_line,
            re70_line,
        ],
        loc="best"
    )

    fig.tight_layout()
    return fig


def _alive_event_index(
    gi,
    flags,
    tHist,
    transient_deposition_alive=False,
    transient_dry_alive=True,
    Nt=None,
):
    """
    Return the first index where the particle becomes not alive.

    Event times stored in flags are physical times, not array indices.
    Therefore convert the earliest event time to the nearest index in tHist.
    """
    t_candidates = []

    if np.isfinite(flags["tStopLimiter"][gi]):
        t_candidates.append(flags["tStopLimiter"][gi])

    if np.isfinite(flags["tStopOutside"][gi]):
        t_candidates.append(flags["tStopOutside"][gi])

    if (not transient_dry_alive) and np.isfinite(flags["tStopDry"][gi]):
        t_candidates.append(flags["tStopDry"][gi])

    if (not transient_deposition_alive) and np.isfinite(flags["tDeposit"][gi]):
        t_candidates.append(flags["tDeposit"][gi])

    if not t_candidates:
        return None

    if tHist is None:
        raise ValueError("tHist is required to convert event times to indices.")

    t_event = float(min(t_candidates))
    k_event = int(np.argmin(np.abs(tHist - t_event)))

    if Nt is not None:
        k_event = int(np.clip(k_event, 0, Nt - 1))

    return k_event


def _final_marker(ax, s_end, z_end, gi, flags, state, marker_size=5.0):
    """
    Plot final marker for tracked particle gi.
    Returns (handle, label).
    """
    base_marker_size = max(float(marker_size), 1e-6)

    if bool(flags["stoppedByLimiter"][gi]):
        return ax.plot(s_end, z_end, "x", linewidth=2, markersize=base_marker_size * (10.0 / 8.0), color="r")[0], "Stopped by limiter"

    if bool(flags["stoppedByDry"][gi]):
        return ax.plot(s_end, z_end, "*", linewidth=1.5, markersize=base_marker_size * (10.0 / 8.0), color="b")[0], "Dry stop"

    if bool(flags["stoppedByOutside"][gi]):
        return ax.plot(
            s_end, z_end, "p",
            markeredgecolor="k",
            markerfacecolor=(1.0, 0.6, 0.0),
            markersize=base_marker_size * (10.0 / 8.0),
        )[0], "Stopped by outside"

    if bool(flags["isDeposited"][gi]):
        return ax.plot(
            s_end, z_end, "s",
            markeredgecolor="k",
            markerfacecolor="r",
            markersize=base_marker_size * (9.0 / 8.0),
        )[0], "Deposited"

    if bool(state["isSurfaced"][gi]):
        return ax.plot(
            s_end, z_end, "^",
            markeredgecolor="k",
            markerfacecolor="c",
            markersize=base_marker_size * (9.0 / 8.0),
        )[0], "Surfaced"

    if bool(state["isNearBedMobile"][gi]):
        return ax.plot(
            s_end, z_end, "d",
            markeredgecolor="k",
            markerfacecolor="g",
            markersize=base_marker_size * (9.0 / 8.0),
        )[0], "Near bed mobile"

    if bool(state["isNearSurfaceMobile"][gi]):
        return ax.plot(
            s_end, z_end, "^",
            markeredgecolor="k",
            markerfacecolor=(0.4, 1.0, 1.0),
            markersize=base_marker_size * (9.0 / 8.0),
        )[0], "Near surface mobile"

    if bool(state["isSuspended"][gi]):
        return ax.plot(s_end, z_end, ".", color="k", markersize=base_marker_size * (15.0 / 8.0))[0], "Suspended"

    return ax.plot(s_end, z_end, ".", color="k", markersize=base_marker_size * (15.0 / 8.0))[0], "Final"


def make_figure_trajectories_tracked(
    xTr,
    yTr,
    zTr,
    tidTr,
    zb_c,
    wse_c,
    flags,
    state,
    track_idx=None,
    nPlot=None,
    zbTr_track=None,
    wseTr_track=None,
    tHist=None,
    xAxisMode="distance",
    depositedAliveInPlot=False,
    dryAliveInPlot=False,
    xEnd=None,
    yEnd=None,
    zEnd=None,
    marker_size=5.0,
    mesh=None,
    hydraulicFieldMode="cellwise",
):
    """
    Plot tracked particle vertical histories.

    If xEnd/yEnd and zEnd are provided, final markers are drawn from the true final
    particle state rather than from the last visible tracked point.
    """
    xTr = np.asarray(xTr, dtype=float)
    yTr = np.asarray(yTr, dtype=float)
    zTr = np.asarray(zTr, dtype=float)
    tidTr = np.asarray(tidTr, dtype=int)
    zb_c = np.asarray(zb_c, dtype=float).ravel()
    wse_c = np.asarray(wse_c, dtype=float).ravel()

    if xEnd is not None:
        xEnd = np.asarray(xEnd, dtype=float).ravel()
    if yEnd is not None:
        yEnd = np.asarray(yEnd, dtype=float).ravel()
    if zEnd is not None:
        zEnd = np.asarray(zEnd, dtype=float).ravel()

    if zbTr_track is not None:
        zbTr_track = np.asarray(zbTr_track, dtype=float)
    if wseTr_track is not None:
        wseTr_track = np.asarray(wseTr_track, dtype=float)
    if tHist is not None:
        tHist = np.asarray(tHist, dtype=float).ravel()

    if xAxisMode not in ("distance", "time"):
        raise ValueError("xAxisMode must be 'distance' or 'time'")

    if tHist is None:
        raise ValueError("tHist is required for trajectory plotting.")

    nTrack, Nt = xTr.shape

    if nPlot is None:
        nUse = nTrack
    else:
        nUse = min(int(nPlot), nTrack)

    idx_pick = np.arange(nUse)

    colors = plt.cm.viridis(np.linspace(0, 1, max(nTrack, 2)))
    base_marker_size = max(float(marker_size), 1e-6)

    ncols = 1 if nUse == 1 else 2
    nrows = int(np.ceil(nUse / ncols))
    use_wide_flume_layout, tall_planform, figsize, _, _ = _binary_map_figure_size(xTr, yTr)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    axs = np.atleast_1d(axs).ravel()

    title_x = "s"
    track_idx_arr = None if track_idx is None else np.asarray(track_idx)

    for ax, i in zip(axs, idx_pick):
        xi = xTr[i]
        yi = yTr[i]
        zi = zTr[i]

        gi = int(track_idx_arr[i]) if track_idx_arr is not None else i
        color = colors[i]

        ok_xyz = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(zi)
        if np.count_nonzero(ok_xyz) < 2:
            ax.set_title(f"Particle {gi}: insufficient data")
            ax.axis("off")
            continue

        idx_ok = np.where(ok_xyz)[0]
        k0 = int(idx_ok[0])

        k_event = _alive_event_index(
            gi,
            flags,
            tHist=tHist,
            transient_deposition_alive=depositedAliveInPlot,
            transient_dry_alive=dryAliveInPlot,
            Nt=Nt,
        )
        if k_event is None:
            k_event = Nt - 1

        # -----------------------------------------------------
        # Build x-axis coordinate
        # -----------------------------------------------------
        if xAxisMode == "time":
            s = np.full(Nt, np.nan, dtype=float)
            ncopy = min(Nt, tHist.size)
            s[:ncopy] = tHist[:ncopy]

            if k_event < Nt - 1:
                s[k_event + 1:] = np.nan

            xlabel = "time (s)"
            title_x = "t"

        else:
            s = np.full(Nt, np.nan, dtype=float)
            s[k0] = 0.0

            for k in range(k0 + 1, Nt):
                if k > k_event:
                    break
                if not (
                    np.isfinite(xi[k - 1]) and np.isfinite(yi[k - 1])
                    and np.isfinite(xi[k]) and np.isfinite(yi[k])
                ):
                    break
                s[k] = s[k - 1] + np.hypot(xi[k] - xi[k - 1], yi[k] - yi[k - 1])

            xlabel = "travelled distance s (m)"
            title_x = "s"

        seg = np.isfinite(s) & np.isfinite(zi)
        seg_idx = np.where(seg)[0]
        if seg_idx.size < 2:
            ax.set_title(f"Particle {gi}: insufficient plotted segment")
            ax.axis("off")
            continue

        k_end = int(seg_idx[-1])

        # -----------------------------------------------------
        # Bed / surface lines
        # -----------------------------------------------------
        if zbTr_track is not None:
            bed_line = np.full(Nt, np.nan, dtype=float)
            ncopy = min(Nt, zbTr_track.shape[1])
            bed_line[:ncopy] = zbTr_track[i, :ncopy]
        else:
            bed_line = np.full(Nt, np.nan, dtype=float)
            tid_seg = tidTr[i, seg]
            in_dom = tid_seg >= 0
            if np.any(in_dom):
                seg_idx_in = seg_idx[in_dom]
                tid_in = tid_seg[in_dom]
                points_in = np.column_stack([xi[seg][in_dom], yi[seg][in_dom]])
                bed_line[seg_idx_in] = _sample_plot_field_at_points(
                    mesh,
                    hydraulicFieldMode,
                    points_in,
                    tid_in,
                    zb_c,
                )

        if wseTr_track is not None:
            wse_line = np.full(Nt, np.nan, dtype=float)
            ncopy = min(Nt, wseTr_track.shape[1])
            wse_line[:ncopy] = wseTr_track[i, :ncopy]
        else:
            wse_line = np.full(Nt, np.nan, dtype=float)
            tid_seg = tidTr[i, seg]
            in_dom = tid_seg >= 0
            if np.any(in_dom):
                seg_idx_in = seg_idx[in_dom]
                tid_in = tid_seg[in_dom]
                points_in = np.column_stack([xi[seg][in_dom], yi[seg][in_dom]])
                wse_line[seg_idx_in] = _sample_plot_field_at_points(
                    mesh,
                    hydraulicFieldMode,
                    points_in,
                    tid_in,
                    wse_c,
                )

        # -----------------------------------------------------
        # Plot lines
        # -----------------------------------------------------
        ax.grid(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("z (m)")

        h_bed = ax.plot(s[seg], bed_line[seg], "k-", linewidth=1.8)[0]
        h_wse = ax.plot(s[seg], wse_line[seg], "b-", linewidth=1.8)[0]
        h_traj = ax.plot(s[seg], zi[seg], "-", color=color, linewidth=1.6)[0]

        ax.plot(
            s[k0],
            zi[k0],
            "o",
            markeredgecolor="k",
            markerfacecolor="y",
            markersize=base_marker_size,
        )

        # -----------------------------------------------------
        # Final marker position
        # Prefer true final state if provided.
        # -----------------------------------------------------
        if (
            xEnd is not None
            and yEnd is not None
            and zEnd is not None
            and 0 <= gi < len(xEnd)
            and 0 <= gi < len(yEnd)
            and 0 <= gi < len(zEnd)
            and np.isfinite(xEnd[gi])
            and np.isfinite(yEnd[gi])
            and np.isfinite(zEnd[gi])
        ):
            z_marker = zEnd[gi]

            if xAxisMode == "time":
                s_marker = tHist[k_event] if (0 <= k_event < len(tHist)) else s[k_end]
            else:
                # For distance plots, place the marker at the travelled distance
                # implied by the true final horizontal position.
                xy_final = np.asarray([xEnd[gi], yEnd[gi]], dtype=float)
                if (
                    0 <= k_event < len(s)
                    and np.isfinite(s[k_event])
                    and np.isfinite(xi[k_event])
                    and np.isfinite(yi[k_event])
                    and np.hypot(xi[k_event] - xy_final[0], yi[k_event] - xy_final[1]) <= 1.0e-8
                ):
                    s_marker = s[k_event]
                else:
                    prior = seg_idx[seg_idx < max(k_event, 0)]
                    k_anchor = int(prior[-1]) if prior.size else k_end
                    if np.isfinite(s[k_anchor]) and np.isfinite(xi[k_anchor]) and np.isfinite(yi[k_anchor]):
                        s_marker = s[k_anchor] + np.hypot(xy_final[0] - xi[k_anchor], xy_final[1] - yi[k_anchor])
                    elif 0 <= k_event < len(s) and np.isfinite(s[k_event]):
                        s_marker = s[k_event]
                    else:
                        s_marker = s[k_end]
        else:
            # fallback to tracked history point
            k_marker = k_end
            if k_event is not None:
                if 0 <= k_event < Nt:
                    if np.isfinite(zi[k_event]) and np.isfinite(s[k_event]):
                        k_marker = k_event

            s_marker = s[k_marker]
            z_marker = zi[k_marker]

        h_final, final_label = _final_marker(
            ax, s_marker, z_marker, gi, flags, state, marker_size=base_marker_size
        )

        ax.set_title(f"Particle {gi}: z vs {title_x}")

        ax.legend(
            [h_bed, h_wse, h_traj, h_final],
            ["bed $z_b$", "free surface wse", f"particle $z({title_x})$", final_label],
            loc="upper right",
            frameon=True,
            fontsize=8,
        )

        s_valid = s[seg]
        if np.any(np.isfinite(s_valid)):
            smin = np.nanmin(s_valid)
            smax = np.nanmax(s_valid)
            if np.isfinite(s_marker):
                smin = min(float(smin), float(s_marker))
                smax = max(float(smax), float(s_marker))
            if np.isfinite(smin) and np.isfinite(smax) and (smax > smin):
                ax.set_xlim(smin, smax)

        x_release = xi[k0] if 0 <= k0 < len(xi) else np.nan
        x_final = xEnd[gi] if xEnd is not None and 0 <= gi < len(xEnd) else np.nan
        travel_distance = float(np.nanmax(s)) if np.any(np.isfinite(s)) else np.nan

        debug_lines = [
            f"x0 = {x_release:.2f} m",
            f"x_end = {x_final:.2f} m",
            f"s_path = {travel_distance:.2f} m",
        ]
        if xAxisMode == "time":
            t_final = s_marker if np.isfinite(s_marker) else np.nan
            debug_lines.append(f"t_end = {t_final:.2f} s")
        else:
            s_final = s_marker if np.isfinite(s_marker) else np.nan
            debug_lines.append(f"s_end = {s_final:.2f} m")
        debug_text = "\n".join(debug_lines)
        ax.text(
            0.02,
            0.98,
            debug_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85, "boxstyle": "round,pad=0.25"},
        )

    for j in range(nUse, len(axs)):
        axs[j].axis("off")

    return fig


def make_figure_binary_map(
    mesh,
    XY,
    tri0,
    U_c,
    V_c,
    ustar_c,
    ustarCrit_c,
    surfaceUstarCrit_c,
    h_c,
    hmin,
    rho_f,
    rho_p,
    P0,
    state,
    xEnd,
    yEnd,
    flags,
    tidEnd=None,
    xTr_track=None,
    yTr_track=None,
    track_idx=None,
    boundary_edges=None,
    marker_size=5.0,
    window_anchor: dict | None = None,
    hydraulicFieldMode: str = "cellwise",
    mesh_overlay: bool | None = None,
):
    """
    Boundary-mobility map + release points + end states + tracked trajectories.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    mobility_cmap = LinearSegmentedColormap.from_list(
        "hydrolptMobilitySoft",
        [
            (140 / 255.0, 81 / 255.0, 10 / 255.0),
            (216 / 255.0, 179 / 255.0, 101 / 255.0),
            (245 / 255.0, 245 / 255.0, 220 / 255.0),
            (90 / 255.0, 180 / 255.0, 172 / 255.0),
            (1 / 255.0, 102 / 255.0, 94 / 255.0),
        ],
    )

    P0 = np.asarray(P0, dtype=float)
    xEnd = np.asarray(xEnd, dtype=float).ravel()
    yEnd = np.asarray(yEnd, dtype=float).ravel()
    tidEnd = None if tidEnd is None else np.asarray(tidEnd, dtype=int).ravel()
    U_c = np.asarray(U_c, dtype=float).ravel()
    V_c = np.asarray(V_c, dtype=float).ravel()
    ustar_c = np.asarray(ustar_c, dtype=float).ravel()
    ustarCrit_c = np.asarray(ustarCrit_c, dtype=float).ravel()
    surfaceUstarCrit_c = np.asarray(surfaceUstarCrit_c, dtype=float).ravel()
    h_c = np.asarray(h_c, dtype=float).ravel()

    XY = np.asarray(XY, dtype=float)
    x_nodes = XY[:, 0]
    y_nodes = XY[:, 1]

    use_wide_flume_layout, tall_planform, _figsize, Lx, Ly = _binary_map_figure_size(x_nodes, y_nodes)

    if use_wide_flume_layout:
        x_plot = x_nodes
        y_plot = y_nodes
        p0x = P0[:, 0]
        p0y = P0[:, 1]
        x_end_plot = xEnd
        y_end_plot = yEnd
        xlabel = "x (m)"
        ylabel = "y (m)"
    else:
        x_plot = x_nodes
        y_plot = y_nodes
        p0x = P0[:, 0]
        p0y = P0[:, 1]
        x_end_plot = xEnd
        y_end_plot = yEnd
        xlabel = "x"
        ylabel = "y"
    masks = _final_state_masks(xEnd, yEnd, flags, state)
    base_marker_size = max(float(marker_size), 1e-6)
    styles = [
        ("deposited", "Deposited", dict(marker="s", linestyle="None", markeredgecolor="k", markerfacecolor="r", markersize=base_marker_size * (9.0 / 8.0), zorder=40)),
        ("near_bed_mobile", "Near bed mobile", dict(marker="d", linestyle="None", markeredgecolor="k", markerfacecolor="g", markersize=base_marker_size * (9.0 / 8.0), zorder=40)),
        ("suspended", "Suspended", dict(marker=".", linestyle="None", color="k", markersize=base_marker_size * (15.0 / 8.0), zorder=40)),
        ("near_surface_mobile", "Near surface mobile", dict(marker="^", linestyle="None", markeredgecolor="k", markerfacecolor=(0.4, 1.0, 1.0), markersize=base_marker_size * (9.0 / 8.0), zorder=40)),
        ("surfaced", "Surfaced", dict(marker="^", linestyle="None", markeredgecolor="k", markerfacecolor="c", markersize=base_marker_size * (9.0 / 8.0), zorder=40)),
        ("dry", "Dry stop", dict(marker="*", linestyle="None", color="b", linewidth=1.5, markersize=base_marker_size * (10.0 / 8.0), zorder=40)),
        ("outside", "Stopped by outside", dict(marker="p", linestyle="None", markeredgecolor="k", markerfacecolor=(1.0, 0.6, 0.0), markersize=base_marker_size * (10.0 / 8.0), zorder=40)),
        ("limiter", "Stopped by limiter", dict(marker="x", linestyle="None", color="r", linewidth=2, markersize=base_marker_size * (10.0 / 8.0), zorder=40)),
    ]

    legend_labels = ["Release"]
    for key, label, _style in styles:
        if np.any(masks[key]):
            legend_labels.append(label)

    xTr_track = None if xTr_track is None else np.asarray(xTr_track, dtype=float)
    yTr_track = None if yTr_track is None else np.asarray(yTr_track, dtype=float)
    track_labels = []
    if xTr_track is not None and yTr_track is not None and xTr_track.ndim == 2 and yTr_track.ndim == 2:
        nTrack = xTr_track.shape[0]
        for k in range(nTrack):
            good = np.isfinite(xTr_track[k]) & np.isfinite(yTr_track[k])
            if np.any(good):
                pid = int(track_idx[k]) if track_idx is not None else k
                track_labels.append(f"Particle {pid}")
    legend_labels.extend(track_labels)

    summary = (
        f"dep={np.count_nonzero(masks['deposited'])}, "
        f"nbed={np.count_nonzero(masks['near_bed_mobile'])}, "
        f"sus={np.count_nonzero(masks['suspended'])}, "
        f"nsurf={np.count_nonzero(masks['near_surface_mobile'])}, "
        f"surf={np.count_nonzero(masks['surfaced'])}, "
        f"dry={np.count_nonzero(masks['dry'])}, "
        f"out={np.count_nonzero(masks['outside'])}"
    )
    title = "Particle end states over boundary mobility field"
    summary_wrapped = textwrap.fill(summary, width=34, break_long_words=False)
    title_text = f"{title}\n{summary_wrapped}"

    speed_ok = (
        (h_c > hmin)
        & np.isfinite(U_c)
        & np.isfinite(V_c)
    )
    ustar_plot_source = np.where(speed_ok & np.isfinite(ustar_c), ustar_c, np.nan)
    scalar_c = np.full_like(ustar_c, np.nan, dtype=float)
    cbar_label_text = r"Shear velocity $u_*$ (m/s)"

    mobility_bed = np.full_like(ustar_c, np.nan, dtype=float)
    mobility_bed_ok = (
        speed_ok
        & np.isfinite(ustar_plot_source)
        & np.isfinite(ustarCrit_c)
        & (ustarCrit_c > 0.0)
    )
    mobility_bed[mobility_bed_ok] = np.clip(
        ustar_plot_source[mobility_bed_ok] / ustarCrit_c[mobility_bed_ok],
        0.0,
        2.0,
    )

    mobility_surface = np.full_like(ustar_c, np.nan, dtype=float)
    mobility_surface_ok = (
        speed_ok
        & np.isfinite(ustar_plot_source)
        & np.isfinite(surfaceUstarCrit_c)
        & (surfaceUstarCrit_c > 0.0)
    )
    mobility_surface[mobility_surface_ok] = np.clip(
        ustar_plot_source[mobility_surface_ok] / surfaceUstarCrit_c[mobility_surface_ok],
        0.0,
        2.0,
    )

    if rho_p > rho_f:
        scalar_c[:] = mobility_bed
        cbar_label_text = r"$M=\dfrac{u_*}{u_{*,\mathrm{crit,bed}}}$, clipped to $[0,2]$"
        scalar_cmap = mobility_cmap
    elif rho_p < rho_f:
        scalar_c[:] = mobility_surface
        cbar_label_text = r"$M=\dfrac{u_*}{u_{*,\mathrm{crit,surf}}}$, clipped to $[0,2]$"
        scalar_cmap = mobility_cmap
    else:
        scalar_c[speed_ok] = ustar_plot_source[speed_ok]
        cbar_label_text = r"Neutral case: shear velocity $u_*$ (m/s)"
        scalar_cmap = "viridis"
    hydraulicFieldMode = str(hydraulicFieldMode or "cellwise").strip().lower()
    native_polygon_mesh = bool(getattr(mesh, "is_native_hecras", False))
    native_idw_plot = native_polygon_mesh and hydraulicFieldMode == "local_idw"
    if mesh_overlay is None:
        mesh_overlay = native_polygon_mesh or (hydraulicFieldMode == "local_idw")
    tri_arr_base = np.asarray(tri0, dtype=int)
    xy_arr_base = np.asarray(XY, dtype=float)
    if native_polygon_mesh and hasattr(mesh, "cell_centers"):
        cell_centers_base = np.asarray(mesh.cell_centers, dtype=float)
    elif tri_arr_base.ndim == 2 and tri_arr_base.shape[1] == 3 and tri_arr_base.size:
        cell_centers_base = np.mean(xy_arr_base[tri_arr_base], axis=1)
    else:
        cell_centers_base = np.empty((0, 2), dtype=float)
    mesh_neighbors_base = (
        np.asarray(mesh.neighbors, dtype=int)
        if mesh is not None and hasattr(mesh, "neighbors")
        else _triangle_neighbors(tri_arr_base)
    )

    def _sample_plot_mobility_at_points(points_xy, point_tid):
        points = np.asarray(points_xy, dtype=float)
        tid_local = np.asarray(point_tid, dtype=int).ravel()
        out = np.full(tid_local.shape, np.nan, dtype=float)
        valid = tid_local >= 0
        if not np.any(valid) or rho_p == rho_f:
            return out
        crit_source = ustarCrit_c if rho_p > rho_f else surfaceUstarCrit_c

        if hydraulicFieldMode == "local_idw":
            ustar_q = _local_cell_idw_sample(points, tid_local, cell_centers_base, mesh_neighbors_base, ustar_plot_source)
            crit_q = _local_cell_idw_sample(points, tid_local, cell_centers_base, mesh_neighbors_base, crit_source)
        else:
            ustar_q = np.full(tid_local.shape, np.nan, dtype=float)
            crit_q = np.full(tid_local.shape, np.nan, dtype=float)
            ustar_q[valid] = ustar_plot_source[tid_local[valid]]
            crit_q[valid] = crit_source[tid_local[valid]]

        cellwise_wet = np.zeros(tid_local.shape, dtype=bool)
        cellwise_wet[valid] = np.isfinite(h_c[tid_local[valid]]) & (h_c[tid_local[valid]] > hmin)
        ok = valid & cellwise_wet & np.isfinite(ustar_q) & np.isfinite(crit_q) & (crit_q > 0.0)
        out[ok] = ustar_q[ok] / crit_q[ok]
        return out

    dep_mask = np.asarray(masks["deposited"], dtype=bool)
    dep_plot_m = None
    if np.any(dep_mask) and rho_p != rho_f:
        dep_points = np.column_stack([x_end_plot[dep_mask], y_end_plot[dep_mask]])
        if tidEnd is not None and tidEnd.shape == xEnd.shape:
            dep_tid = tidEnd[dep_mask]
        elif mesh is not None and hasattr(mesh, "point_location"):
            dep_tid = mesh.point_location(dep_points[:, 0], dep_points[:, 1])
        else:
            dep_tid = np.full(dep_points.shape[0], -1, dtype=int)
        dep_m = _sample_plot_mobility_at_points(dep_points, dep_tid)
        dep_plot_m = dep_m
    h_nodes_plot = None
    scalar_nodes_plot = None
    native_grid_plot = None
    field_x_plot = x_plot
    field_y_plot = y_plot
    field_tri_plot = tri0
    field_scalar_values = None
    if native_idw_plot:
        field_scalar_values = scalar_c
    elif hydraulicFieldMode == "local_idw":
        tri_arr = np.asarray(tri0, dtype=int)
        xy_arr = np.asarray(XY, dtype=float)
        refined_xy, refined_tri, source_tri, _refined_bary = _refined_triangle_grid(xy_arr, tri_arr)
        if source_tri is None:
            refined_xy = xy_arr
            refined_tri = tri_arr
            source_tri = np.asarray(mesh.point_location(refined_xy[:, 0], refined_xy[:, 1]), dtype=int)
        if mesh is not None and hasattr(mesh, "neighbors"):
            neighbors = np.asarray(mesh.neighbors, dtype=int)
        else:
            neighbors = _triangle_neighbors(tri_arr)
        cell_centers = np.mean(xy_arr[tri_arr], axis=1)
        h_refined = _local_cell_idw_sample(refined_xy, source_tri, cell_centers, neighbors, h_c)
        wet_refined = (
            (source_tri >= 0)
            & np.isfinite(h_c[np.where(source_tri >= 0, source_tri, 0)])
            & (h_c[np.where(source_tri >= 0, source_tri, 0)] > hmin)
        )
        if rho_p > rho_f:
            ustar_refined = _local_cell_idw_sample(refined_xy, source_tri, cell_centers, neighbors, ustar_plot_source)
            crit_refined = _local_cell_idw_sample(refined_xy, source_tri, cell_centers, neighbors, ustarCrit_c)
            scalar_refined = np.full_like(ustar_refined, np.nan, dtype=float)
            ok_refined = np.isfinite(ustar_refined) & np.isfinite(crit_refined) & (crit_refined > 0.0)
            scalar_refined[ok_refined] = np.clip(ustar_refined[ok_refined] / crit_refined[ok_refined], 0.0, 2.0)
        elif rho_p < rho_f:
            ustar_refined = _local_cell_idw_sample(refined_xy, source_tri, cell_centers, neighbors, ustar_plot_source)
            crit_refined = _local_cell_idw_sample(refined_xy, source_tri, cell_centers, neighbors, surfaceUstarCrit_c)
            scalar_refined = np.full_like(ustar_refined, np.nan, dtype=float)
            ok_refined = np.isfinite(ustar_refined) & np.isfinite(crit_refined) & (crit_refined > 0.0)
            scalar_refined[ok_refined] = np.clip(ustar_refined[ok_refined] / crit_refined[ok_refined], 0.0, 2.0)
        else:
            scalar_refined = _local_cell_idw_sample(refined_xy, source_tri, cell_centers, neighbors, scalar_c)
        h_refined[~wet_refined] = np.nan
        scalar_refined[~wet_refined] = np.nan
        scalar_nodes_plot = np.ma.masked_where(
            ~np.isfinite(scalar_refined),
            scalar_refined,
        )
        field_x_plot = refined_xy[:, 0]
        field_y_plot = refined_xy[:, 1]
        field_tri_plot = refined_tri
        field_scalar_values = scalar_refined
    else:
        scalar_plot = np.ma.masked_where(~np.isfinite(scalar_c), scalar_c)
        field_scalar_values = scalar_c
    fig = plt.figure(dpi=100)
    if use_wide_flume_layout:
        map_width_inches = 6.4
        map_height_inches = 2.4
        x_min, x_max = _axis_bounds(x_plot)
        y_min = float(np.nanmin(y_plot))
        y_max = float(np.nanmax(y_plot))
        y_tick_values = np.asarray([0.0, 0.2, 0.8, 1.0], dtype=float)
        x_tick_values = _flume_x_tick_values(x_min, x_max)
    else:
        map_height_inches = 5.8
        display_aspect = 1.0
        display_ratio = (_finite_span(y_plot) / _finite_span(x_plot)) * display_aspect
        map_width_inches = map_height_inches / max(display_ratio, 1e-12)
        y_min = float(np.nanmin(y_plot))
        y_max = float(np.nanmax(y_plot))
        y_tick_values = np.linspace(y_min, y_max, 6)
        x_min = float(np.nanmin(x_plot))
        x_max = float(np.nanmax(x_plot))
        x_tick_values = np.linspace(x_min, x_max, 6)
    pad_inches = 0.4
    y_tick_labels = [_compact_tick(v, None) for v in y_tick_values]
    x_tick_labels = [_compact_tick(v, None) for v in x_tick_values]
    legend_cell_width_inches = max(
        0.7,
        _estimate_legend_width_inches(legend_labels, title="Markers", fontsize=LEGEND_STYLE["fontsize"]) + 0.02,
    )
    legend_gap_inches = 0.10
    tick_width_inches = _estimate_text_width_inches(y_tick_labels, fontsize=10.0, extra_pad=0.08)
    # A rotated y-label uses roughly one line height horizontally, not its full text width.
    ylabel_width_inches = (10.0 / 72.0) * 1.6
    yaxis_cell_width_inches = tick_width_inches + ylabel_width_inches + 0.12
    xtick_height_inches = (10.0 / 72.0) * 1.25
    xlabel_height_inches = _estimate_text_block_height_inches(xlabel, fontsize=10.0, line_spacing=1.0, pad=0.02)
    xaxis_cell_height_inches = xtick_height_inches + xlabel_height_inches + 0.08
    title_cell_height_inches = _estimate_text_block_height_inches(title_text, fontsize=FIGURE1_TITLE_FONTSIZE, line_spacing=1.0, pad=0.04)
    title_gap_inches = 0.10
    spacer_cell_height_inches = 0.12
    cbar_tick_height_inches = (10.0 / 72.0) * 1.25
    cbar_label_height_inches = _estimate_text_block_height_inches(cbar_label_text, fontsize=10.0, line_spacing=1.0, pad=0.02)
    cbar_cell_height_inches = 0.22 + cbar_tick_height_inches + cbar_label_height_inches + 0.06
    cbar_bar_cell_height_inches = 0.5 * cbar_cell_height_inches
    cbar_text_cell_height_inches = cbar_tick_height_inches + cbar_label_height_inches + 0.08
    fig_height = map_height_inches + title_gap_inches + title_cell_height_inches + xaxis_cell_height_inches + spacer_cell_height_inches + cbar_bar_cell_height_inches + cbar_text_cell_height_inches + 2.0 * pad_inches
    if use_wide_flume_layout:
        side_pad_inches = 0.4
        fig_width = legend_cell_width_inches + legend_gap_inches + map_width_inches + yaxis_cell_width_inches + 2.0 * side_pad_inches
    else:
        side_pad_inches = pad_inches
        fig_width = legend_cell_width_inches + legend_gap_inches + map_width_inches + yaxis_cell_width_inches + 2.0 * side_pad_inches
    fig.set_size_inches(fig_width, fig_height, forward=True)

    title_pos = [
        (side_pad_inches + legend_cell_width_inches + legend_gap_inches) / fig_width,
        (pad_inches + cbar_text_cell_height_inches + cbar_bar_cell_height_inches + spacer_cell_height_inches + xaxis_cell_height_inches + map_height_inches + title_gap_inches) / fig_height,
        map_width_inches / fig_width,
        title_cell_height_inches / fig_height,
    ]
    legend_pos = [
        side_pad_inches / fig_width,
        (pad_inches + cbar_text_cell_height_inches + cbar_bar_cell_height_inches + spacer_cell_height_inches + xaxis_cell_height_inches) / fig_height,
        legend_cell_width_inches / fig_width,
        map_height_inches / fig_height,
    ]
    map_pos = [
        (side_pad_inches + legend_cell_width_inches + legend_gap_inches) / fig_width,
        (pad_inches + cbar_text_cell_height_inches + cbar_bar_cell_height_inches + spacer_cell_height_inches + xaxis_cell_height_inches) / fig_height,
        map_width_inches / fig_width,
        map_height_inches / fig_height,
    ]
    yaxis_pos = [
        (side_pad_inches + legend_cell_width_inches + legend_gap_inches + map_width_inches) / fig_width,
        (pad_inches + cbar_text_cell_height_inches + cbar_bar_cell_height_inches + spacer_cell_height_inches + xaxis_cell_height_inches) / fig_height,
        yaxis_cell_width_inches / fig_width,
        map_height_inches / fig_height,
    ]
    xaxis_pos = [
        (side_pad_inches + legend_cell_width_inches + legend_gap_inches) / fig_width,
        (pad_inches + cbar_text_cell_height_inches + cbar_bar_cell_height_inches + spacer_cell_height_inches) / fig_height,
        map_width_inches / fig_width,
        xaxis_cell_height_inches / fig_height,
    ]
    spacer_pos = [
        (side_pad_inches + legend_cell_width_inches + legend_gap_inches) / fig_width,
        (pad_inches + cbar_text_cell_height_inches + cbar_bar_cell_height_inches) / fig_height,
        map_width_inches / fig_width,
        spacer_cell_height_inches / fig_height,
    ]
    cbar_pos = [
        (side_pad_inches + legend_cell_width_inches + legend_gap_inches) / fig_width,
        (pad_inches + cbar_text_cell_height_inches) / fig_height,
        map_width_inches / fig_width,
        cbar_bar_cell_height_inches / fig_height,
    ]
    cbar_label_pos = [
        (side_pad_inches + legend_cell_width_inches + legend_gap_inches) / fig_width,
        pad_inches / fig_height,
        map_width_inches / fig_width,
        cbar_text_cell_height_inches / fig_height,
    ]
    ax = fig.add_axes(map_pos)
    ax.set_zorder(1)
    ax.set_navigate(True)
    title_ax = fig.add_axes(title_pos)
    title_ax.set_zorder(2)
    title_ax.set_navigate(False)
    title_ax.patch.set_alpha(0.0)
    title_ax.axis("off")
    legend_ax = fig.add_axes(legend_pos)
    legend_ax.set_zorder(2)
    legend_ax.set_navigate(False)
    legend_ax.patch.set_alpha(0.0)
    legend_ax.axis("off")
    yaxis_ax = fig.add_axes(yaxis_pos, sharey=ax, label=f"_yaxis_only_{id(fig)}")
    yaxis_ax.set_zorder(2)
    yaxis_ax.set_navigate(False)
    yaxis_ax.patch.set_alpha(0.0)
    yaxis_ax.xaxis.set_visible(False)
    yaxis_ax.yaxis.tick_right()
    yaxis_ax.yaxis.set_label_position("right")
    yaxis_ax.spines["top"].set_visible(False)
    yaxis_ax.spines["bottom"].set_visible(False)
    yaxis_ax.spines["left"].set_visible(False)
    yaxis_ax.spines["right"].set_position(("axes", 0.0))
    yaxis_ax.spines["right"].set_visible(False)

    if use_wide_flume_layout:
        ax.set_aspect("auto")
    else:
        ax.set_aspect("equal", adjustable="datalim")

    native_collection = None
    if native_idw_plot:
        _grid_x, _grid_y, native_grid_plot = _native_idw_grid(
            mesh,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            h_c=h_c,
            hmin=hmin,
            ustar_source=ustar_plot_source,
            bed_crit=ustarCrit_c,
            surface_crit=surfaceUstarCrit_c,
            scalar_c=scalar_c,
            rho_p=rho_p,
            rho_f=rho_f,
        )
        native_underlay = _native_polygon_collection(
            mesh,
            field_scalar_values,
            cmap=scalar_cmap,
            alpha=1.0,
            zorder=0.8,
        )
        if native_underlay is not None:
            if rho_p != rho_f:
                native_underlay.set_clim(0.0, 2.0)
            ax.add_collection(native_underlay)
    if native_polygon_mesh and not native_idw_plot:
        native_collection = _native_polygon_collection(mesh, field_scalar_values, cmap=scalar_cmap, alpha=1.0)

    if native_grid_plot is not None:
        cmap_obj = plt.get_cmap(scalar_cmap).copy()
        cmap_obj.set_bad(color=(0.74, 0.74, 0.74, 0.0))
        tpc = ax.imshow(
            np.ma.masked_invalid(native_grid_plot),
            extent=(x_min, x_max, y_min, y_max),
            origin="lower",
            interpolation="nearest",
            cmap=cmap_obj,
            alpha=1.0,
            zorder=1,
        )
        wet_clip = _native_mask_clip_patch(mesh, np.isfinite(h_c) & (h_c > hmin), ax)
        if wet_clip is not None:
            tpc.set_clip_path(wet_clip.get_path(), wet_clip.get_transform())
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        dry_overlay = _native_mask_polygon_collection(
            mesh,
            ~(np.isfinite(h_c) & (h_c > hmin)),
            facecolor=(0.74, 0.74, 0.74, 1.0),
            alpha=1.0,
            zorder=2,
        )
        if dry_overlay is not None:
            ax.add_collection(dry_overlay)
    elif native_collection is not None:
        ax.add_collection(native_collection)
        tpc = native_collection
    elif scalar_nodes_plot is not None:
        tpc = ax.tripcolor(
            field_x_plot,
            field_y_plot,
            field_tri_plot,
            scalar_nodes_plot,
            shading="gouraud",
            cmap=scalar_cmap,
            alpha=0.9,
        )
    else:
        tpc = ax.tripcolor(
            x_plot,
            y_plot,
            tri0,
            facecolors=scalar_plot,
            shading="flat",
            cmap=scalar_cmap,
            alpha=0.9,
        )
    if mesh_overlay and native_polygon_mesh and hasattr(mesh, "all_edges"):
        mesh_edges = mesh.all_edges()
        mesh_lines = _edge_line_collection(
            XY,
            mesh_edges,
            color="k",
            linewidth=0.14,
            alpha=0.24,
            zorder=28,
        )
        if mesh_lines is not None:
            ax.add_collection(mesh_lines)
    elif mesh_overlay:
        ax.triplot(
            x_plot,
            y_plot,
            tri0,
            color="k",
            linewidth=0.18,
            alpha=0.18,
            zorder=28,
        )
    wet_mask_plot = np.isfinite(h_c) & (h_c > hmin)
    if native_polygon_mesh and hasattr(mesh, "wet_dry_boundary_edges"):
        if native_idw_plot and hasattr(mesh, "wet_dry_boundary_edges_with_wet_cells"):
            wet_mesh_edges, wet_side_cells = mesh.wet_dry_boundary_edges_with_wet_cells(wet_mask_plot)
        else:
            wet_mesh_edges = mesh.wet_dry_boundary_edges(wet_mask_plot)
            wet_side_cells = None
    else:
        wet_mesh_edges = _mask_boundary_edges(tri0, wet_mask_plot)
        wet_side_cells = None
    if wet_mesh_edges.size:
        wet_lines = _edge_line_collection(
            XY,
            wet_mesh_edges,
            color="black",
            linewidth=1.6 if native_idw_plot else 1.2,
            alpha=0.95,
            zorder=34,
        )
        if wet_lines is not None:
            ax.add_collection(wet_lines)
    if rho_p != rho_f:
        tpc.set_clim(0.0, 2.0)
    cbar_ax = fig.add_axes(cbar_pos)
    cbar_ax.set_zorder(2)
    cbar_ax.set_navigate(False)
    cb = fig.colorbar(tpc, cax=cbar_ax, orientation="horizontal")
    cb.set_label(cbar_label_text)
    cb.ax.tick_params(axis="x", bottom=True, labelbottom=True)
    cbar_label_ax = fig.add_axes(cbar_label_pos)
    cbar_label_ax.set_zorder(2)
    cbar_label_ax.set_navigate(False)
    cbar_label_ax.patch.set_alpha(0.0)
    cbar_label_ax.axis("off")

    if boundary_edges is None:
        if mesh is None:
            raise ValueError("boundary_edges must be provided when mesh is None.")
        bd = mesh.free_boundary_edges()
    else:
        bd = np.asarray(boundary_edges, dtype=int)
    if bd.ndim == 2 and bd.shape[1] == 2 and bd.size:
        boundary_lines = _edge_line_collection(
            XY,
            bd,
            color="k",
            linewidth=1.0,
            alpha=1.0,
            zorder=35,
        )
        if boundary_lines is not None:
            ax.add_collection(boundary_lines)

    handles = []
    labels = []
    h_rel = ax.plot(
        p0x, p0y,
        marker="o", linestyle="None",
        markeredgecolor="k", markerfacecolor="y",
        markersize=base_marker_size, zorder=30,
    )[0]
    handles.append(h_rel)
    labels.append("Release")

    for key, label, style in styles:
        mask = masks[key]
        if np.any(mask):
            x_marker = x_end_plot
            y_marker = y_end_plot
            h = ax.plot(x_marker[mask], y_marker[mask], **style)[0]
            handles.append(h)
            labels.append(label)

    if xTr_track is not None and yTr_track is not None:
        if xTr_track.ndim == 2 and yTr_track.ndim == 2:
            nTrack = xTr_track.shape[0]
            colors = plt.cm.viridis(np.linspace(0, 1, max(nTrack, 2)))

            for k in range(nTrack):
                good = np.isfinite(xTr_track[k]) & np.isfinite(yTr_track[k])
                if not np.any(good):
                    continue

                pid = int(track_idx[k]) if track_idx is not None else k
                x_track_plot = xTr_track[k, good]
                y_track_plot = yTr_track[k, good]
                h = ax.plot(
                    x_track_plot,
                    y_track_plot,
                    "-",
                    linewidth=1.6,
                    color=colors[k],
                    alpha=0.85,
                    zorder=60,
                )[0]
                handles.append(h)
                labels.append(f"Particle {pid}")
    legend_ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 1.0), **LEGEND_STYLE)
    title_ax.text(0.5, 0.5, title_text, ha="center", va="center", fontsize=FIGURE1_TITLE_FONTSIZE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    yaxis_ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(FuncFormatter(_compact_tick))
    yaxis_ax.yaxis.set_major_formatter(FuncFormatter(_compact_tick))
    ax.set_xticks(x_tick_values)
    yaxis_ax.set_yticks(y_tick_values)
    ax.margins(x=0.0, y=0.0)
    if use_wide_flume_layout:
        ax.set_xlim(x_tick_values[0], x_tick_values[-1])
        ax.set_ylim(0.0, 1.0)

    return fig


def make_figure_rouse_profile_section(
    xTr,
    zTr,
    tidTr,
    zb_c,
    wse_c,
    ustar_c,
    ws,
    hmin,
    flume_length,
    x_offset_from_end,
    half_window,
    nbins,
    a_ref,
    u_c=None,
    z_frac=0.5,
    normalize="max",   # "max" for C/Ca-type plots, "pdf" for normalized density
    x_scan_step=0.1,
    x_release=None,
    surface_cdf_target=0.95,
    dcdx_convergence_fraction=0.02,
):
    """
    Plot five empirical vertical particle profiles along the flume and compare
    each against a theoretical Rouse-type profile. Also generates a
    longitudinal convergence figure based on the section-to-section change in
    the normalized empirical profile.

    Supports both:
    - settling particles   : ws > 0  -> bed-concentrated profile
    - buoyant particles    : ws < 0  -> surface-concentrated profile

    The figure always reserves five panels. The first section is placed at the
    release location, and the remaining four are distributed evenly to the
    requested downstream location. Panels
    without enough valid samples are kept and marked as unavailable.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    xTr = np.asarray(xTr, dtype=float)
    zTr = np.asarray(zTr, dtype=float)
    tidTr = np.asarray(tidTr, dtype=int)
    zb_c = np.asarray(zb_c, dtype=float).ravel()
    wse_c = np.asarray(wse_c, dtype=float).ravel()
    ustar_c = np.asarray(ustar_c, dtype=float).ravel()
    u_c = None if u_c is None else np.asarray(u_c, dtype=float).ravel()

    if xTr.shape != zTr.shape or xTr.shape != tidTr.shape:
        raise ValueError("xTr, zTr, and tidTr must have the same shape.")

    if nbins < 5:
        raise ValueError("nbins should be at least 5.")

    if normalize not in ("pdf", "max"):
        raise ValueError("normalize must be 'pdf' or 'max'.")
    if x_scan_step <= 0.0:
        raise ValueError("x_scan_step must be > 0.")
    if not (0.0 <= float(z_frac) <= 1.0):
        raise ValueError("z_frac must be between 0 and 1.")
    if not (0.0 < surface_cdf_target <= 1.0):
        raise ValueError("surface_cdf_target must be in (0, 1].")
    if dcdx_convergence_fraction <= 0.0:
        raise ValueError("dcdx_convergence_fraction must be > 0.")

    # ---------------------------------------------------------
    # Basic valid mask for section selection
    # ---------------------------------------------------------
    valid = np.isfinite(xTr) & np.isfinite(zTr) & (tidTr >= 0)

    if not np.any(valid):
        print("Warning: no valid trajectory samples available for Rouse plot. Skipping figure.")
        return None

    x_valid = xTr[valid]
    x_min_occ = float(np.nanmin(x_valid))
    x_max_occ = float(np.nanmax(x_valid))
    x_section_req = float(flume_length) - float(x_offset_from_end)
    x_section_target = max(x_section_req, x_min_occ)

    kappa = 0.41
    eta_compare = np.linspace(0.0, 1.0, max(int(nbins), 50), dtype=float)

    def _section_mask(x_section_use):
        x_lo_use = x_section_use - float(half_window)
        x_hi_use = x_section_use + float(half_window)
        in_section_use = valid & (xTr >= x_lo_use) & (xTr <= x_hi_use)
        return in_section_use, x_lo_use, x_hi_use

    def _evaluate_section(x_section_use):
        in_section_use, x_lo_use, x_hi_use = _section_mask(x_section_use)
        if not np.any(in_section_use):
            return None

        tid_s = tidTr[in_section_use]
        z_s = zTr[in_section_use]

        zb_s = zb_c[tid_s]
        wse_s = wse_c[tid_s]
        h_s = wse_s - zb_s
        ustar_s = ustar_c[tid_s]
        u_s = None if u_c is None else u_c[tid_s]

        wet = (
            np.isfinite(zb_s)
            & np.isfinite(wse_s)
            & np.isfinite(h_s)
            & (h_s > hmin)
            & np.isfinite(z_s)
            & np.isfinite(ustar_s)
            & (ustar_s > 0.0)
        )
        if not np.any(wet):
            return None

        z_s = z_s[wet]
        zb_s = zb_s[wet]
        h_s = h_s[wet]
        ustar_s = ustar_s[wet]
        if u_s is not None:
            u_s = u_s[wet]

        eta = (z_s - zb_s) / h_s
        good_eta = np.isfinite(eta) & (eta >= 0.0) & (eta <= 1.0)
        eta = eta[good_eta]
        h_s = h_s[good_eta]
        ustar_s = ustar_s[good_eta]
        if u_s is not None:
            u_s = u_s[good_eta]
        eta_all = eta.copy()

        if eta.size < 10:
            return None

        ustar_mean = np.mean(ustar_s)
        h_mean = np.mean(h_s)
        u_mean = np.nan if u_s is None else float(np.nanmean(np.abs(u_s)))
        if not np.isfinite(ustar_mean) or ustar_mean <= 0.0:
            return None
        if not np.isfinite(h_mean) or h_mean <= hmin:
            return None

        a_ref_use = a_ref
        if a_ref_use is None:
            a_ref_use = max(0.02 * h_mean, 3.0 * hmin)
        a_ref_use = float(a_ref_use)
        a_ref_use = min(max(a_ref_use, 1e-6), 0.95 * h_mean)
        eta_a = a_ref_use / h_mean

        P = abs(float(ws)) / (kappa * ustar_mean)

        edges_all = np.linspace(0.0, 1.0, nbins + 1)
        centers_all = 0.5 * (edges_all[:-1] + edges_all[1:])
        d_eta_all = edges_all[1] - edges_all[0]
        eta_all_hist = np.clip(eta_all, edges_all[0], np.nextafter(edges_all[-1], edges_all[0]))
        counts_all, _ = np.histogram(eta_all_hist, bins=edges_all)
        empirical_all = counts_all.astype(float)

        if ws >= 0.0:
            eta_min = eta_a
            eta_max = 1.0
            eta_emp = eta[(eta >= eta_min) & (eta <= eta_max)]
            if eta_emp.size < 10:
                return None

            edges = np.linspace(eta_min, eta_max, nbins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            d_eta = edges[1] - edges[0]
            eta_th = np.linspace(max(eta_a, 1e-6), 0.999, 400)
            rouse = (
                ((1.0 - eta_th) / eta_th)
                * (eta_a / (1.0 - eta_a))
            ) ** P
            ref_label = "Reference level\n" + fr"$a/H={eta_a:.3f}$"
            ref_y = eta_a
            profile_kind = "settling"
        else:
            eta_min = 0.0
            eta_max = 1.0 - eta_a
            eta_emp = eta[(eta >= eta_min) & (eta <= eta_max)]
            if eta_emp.size < 10:
                return None

            edges = np.linspace(eta_min, eta_max, nbins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            d_eta = edges[1] - edges[0]
            eta_th = np.linspace(1e-3, min(eta_max, 0.999), 400)
            rouse = (
                (eta_th / (1.0 - eta_th))
                * (eta_a / (1.0 - eta_a))
            ) ** P
            ref_label = "Surface reference\n" + fr"$(H-a)/H={1.0 - eta_a:.3f}$"
            ref_y = 1.0 - eta_a
            profile_kind = "buoyant"

        eta_emp_hist = np.clip(eta_emp, edges[0], np.nextafter(edges[-1], edges[0]))
        counts, _ = np.histogram(eta_emp_hist, bins=edges)
        empirical = counts.astype(float)

        if normalize == "pdf":
            area_emp = np.sum(empirical) * d_eta
            if area_emp > 0.0:
                empirical /= area_emp
            area_emp_all = np.sum(empirical_all) * d_eta_all
            if area_emp_all > 0.0:
                empirical_all /= area_emp_all
            area_th = np.trapezoid(rouse, eta_th)
            if area_th > 0.0:
                rouse /= area_th
        else:
            max_emp = np.max(empirical)
            if max_emp > 0.0:
                empirical /= max_emp
            max_emp_all = np.max(empirical_all)
            if max_emp_all > 0.0:
                empirical_all /= max_emp_all
            max_th = np.max(rouse)
            if max_th > 0.0:
                rouse /= max_th

        return {
            "x_section": float(x_section_use),
            "x_lo": float(x_lo_use),
            "x_hi": float(x_hi_use),
            "eta_emp_n": int(eta_emp.size),
            "P": float(P),
            "h_mean": float(h_mean),
            "u_mean": float(u_mean),
            "profile_kind": profile_kind,
            "centers": centers,
            "empirical": empirical,
            "empirical_compare": np.interp(eta_compare, centers_all, empirical_all, left=0.0, right=0.0),
            "empirical_compare_rouse": np.interp(eta_compare, centers, empirical, left=0.0, right=0.0),
            "eta_th": eta_th,
            "rouse": rouse,
            "ref_y": float(ref_y),
            "ref_label": ref_label,
        }

    if x_release is None:
        x_start = x_min_occ
    else:
        x_start = float(np.asarray(x_release, dtype=float).ravel()[0])
    x_start = max(x_start, x_min_occ)
    x_stop = x_section_target
    if x_stop < x_start:
        x_stop = x_start
    x_centers = np.linspace(x_start, x_stop, 5, dtype=float)

    section_results = [_evaluate_section(xc) for xc in x_centers]
    if not any(result is not None for result in section_results):
        print("Warning: no valid samples found for any Rouse profile section. Skipping figure.")
        return None

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    legend_ax = fig.add_subplot(gs[1, 2])
    legend_ax.axis("off")
    if normalize == "max":
        for ax in axes[1:]:
            ax.sharex(axes[0])
    for ax in axes[1:]:
        ax.sharey(axes[0])

    legend_handles = None
    legend_labels = None
    x_label = r"$C/C_{\mathrm{ref}}$" if normalize == "max" else r"$p(\eta)$"

    for idx, (ax, result) in enumerate(zip(axes, section_results)):
        if result is None:
            ax.text(
                0.5,
                0.5,
                "Insufficient\nsamples",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.set_title(f"x = {x_centers[idx]:.2f} m")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.35)
        else:
            line_rouse, = ax.plot(
                result["rouse"],
                result["eta_th"],
                "-",
                linewidth=2.0,
                color="tab:orange",
                label="Rouse profile",
            )
            line_emp, = ax.plot(
                result["empirical"],
                result["centers"],
                "o",
                markersize=3.5,
                color="black",
                alpha=0.7,
                label="Particle profile",
            )
            line_ref = ax.axhline(
                result["ref_y"],
                linestyle="--",
                linewidth=1.5,
                color="gray",
                label=result["ref_label"],
            )
            if legend_handles is None:
                legend_handles = [line_rouse, line_emp, line_ref]
                legend_labels = ["Rouse profile", "Particle profile", result["ref_label"]]

            ax.set_title(
                f"x = {result['x_section']:.2f} m\n"
                f"N = {result['eta_emp_n']}, |P| = {result['P']:.3f}"
            )
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.35)

        if normalize == "max":
            ax.set_xlim(0.0, 1.05)
        ax.set_xlabel(x_label)

    axes[0].set_ylabel(r"$\eta=(z-z_b)/H$")
    axes[3].set_ylabel(r"$\eta=(z-z_b)/H$")

    if legend_handles is not None and legend_labels is not None:
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            frameon=True,
        )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95)

    x_scan = np.arange(x_min_occ, x_section_target + 0.5 * x_scan_step, x_scan_step, dtype=float)
    scan_results = [_evaluate_section(xc) for xc in x_scan]
    x_conv = []
    dcdx = []
    dcdx_rouse = []
    for prev_result, curr_result in zip(scan_results[:-1], scan_results[1:]):
        if prev_result is None or curr_result is None:
            continue

        dx = curr_result["x_section"] - prev_result["x_section"]
        if not np.isfinite(dx) or dx <= 0.0:
            continue

        profile_delta = np.abs(curr_result["empirical_compare"] - prev_result["empirical_compare"])
        dcdx.append(100.0 * np.trapezoid(profile_delta, eta_compare) / dx)
        profile_delta_rouse = np.abs(
            curr_result["empirical_compare_rouse"] - prev_result["empirical_compare_rouse"]
        )
        dcdx_rouse.append(100.0 * np.trapezoid(profile_delta_rouse, eta_compare) / dx)
        x_conv.append(0.5 * (prev_result["x_section"] + curr_result["x_section"]))

    ws_value = float(ws)
    if ws_value > 0.0:
        boundary_modes = ["bed"]
    elif ws_value < 0.0:
        boundary_modes = ["surface"]
    else:
        boundary_modes = ["surface", "bed"]

    first_boundary_x = {mode: [] for mode in boundary_modes}
    if xTr.ndim == 2 and zTr.ndim == 2 and tidTr.ndim == 2:
        for particle_idx in range(xTr.shape[0]):
            x_row = np.asarray(xTr[particle_idx], dtype=float)
            z_row = np.asarray(zTr[particle_idx], dtype=float)
            tid_row = np.asarray(tidTr[particle_idx], dtype=int)

            valid_row = np.isfinite(x_row) & np.isfinite(z_row) & (tid_row >= 0)
            if not np.any(valid_row):
                continue

            tid_valid = tid_row[valid_row]
            x_valid_row = x_row[valid_row]
            z_valid_row = z_row[valid_row]
            zb_row = zb_c[tid_valid]
            wse_row = wse_c[tid_valid]
            depth_row = wse_row - zb_row

            wet_row = (
                np.isfinite(zb_row)
                & np.isfinite(wse_row)
                & np.isfinite(depth_row)
                & (depth_row > hmin)
            )
            if not np.any(wet_row):
                continue

            x_valid_row = x_valid_row[wet_row]
            z_valid_row = z_valid_row[wet_row]
            zb_row = zb_row[wet_row]
            wse_row = wse_row[wet_row]
            depth_row = depth_row[wet_row]

            z_tol = np.maximum(1e-6, 1e-3 * depth_row)
            hit_surface = z_valid_row >= (wse_row - z_tol)
            hit_bed = z_valid_row <= (zb_row + z_tol)
            hit_any = hit_surface | hit_bed
            if not np.any(hit_any):
                continue

            first_hit_idx = int(np.argmax(hit_any))
            first_hits_surface = hit_surface[first_hit_idx] and (not hit_bed[first_hit_idx])
            first_hits_bed = hit_bed[first_hit_idx] and (not hit_surface[first_hit_idx])
            if first_hits_surface and "surface" in first_boundary_x:
                first_boundary_x["surface"].append(float(x_valid_row[first_hit_idx]))
            elif first_hits_bed and "bed" in first_boundary_x:
                first_boundary_x["bed"].append(float(x_valid_row[first_hit_idx]))

    has_conv = bool(x_conv)
    has_boundary_cdf = any(bool(values) for values in first_boundary_x.values())
    valid_results = [result for result in scan_results if result is not None]
    reference_length = np.nan
    if valid_results and np.isfinite(ws) and abs(float(ws)) > 0.0:
        h_ref = float(np.nanmean([result["h_mean"] for result in valid_results]))
        u_ref = float(np.nanmean([result["u_mean"] for result in valid_results]))
        if np.isfinite(h_ref) and np.isfinite(u_ref):
            z_release = float(z_frac) * h_ref
            if float(ws) > 0.0:
                reference_length = u_ref * z_release / abs(float(ws))
            else:
                reference_length = u_ref * (h_ref - z_release) / abs(float(ws))

    if has_conv or has_boundary_cdf:
        conv_fig, conv_ax = plt.subplots(figsize=(7.2, 4.8))
        handles = []
        labels = []

        if has_conv:
            conv_line, = conv_ax.plot(
                x_conv,
                dcdx,
                "-o",
                color="tab:blue",
                linewidth=1.8,
                markersize=3.5,
                label=r"$\Delta C / \Delta x$",
            )
            handles.append(conv_line)
            labels.append(r"$\Delta C / \Delta x$ full depth")
            conv_line_rouse, = conv_ax.plot(
                x_conv,
                dcdx_rouse,
                "--s",
                color="tab:orange",
                linewidth=1.6,
                markersize=3.2,
                label=r"$\Delta C / \Delta x$ Rouse support",
            )
            handles.append(conv_line_rouse)
            labels.append(r"$\Delta C / \Delta x$ Rouse support")
        conv_ax.set_xlabel("x (m)")
        conv_ax.set_ylabel(r"$\Delta C / \Delta x$ (%)")
        conv_ax.set_ylim(0.0, 100.0)
        if len(boundary_modes) == 2:
            boundary_title = "surface/bed"
        else:
            boundary_title = boundary_modes[0]
        conv_ax.set_title(f"Longitudinal convergence and first {boundary_title}-hit CDF")
        conv_ax.set_xlim(x_min_occ, x_section_target)
        conv_ax.grid(True, alpha=0.35)

        if np.isfinite(reference_length):
            length_line = conv_ax.axvline(
                reference_length,
                color="tab:purple",
                linestyle="-.",
                linewidth=1.4,
                alpha=0.9,
            )
            handles.append(length_line)
            if float(ws) > 0.0:
                labels.append(fr"$U z_{{rel}} / |w_s| = {reference_length:.2f}$ m")
            else:
                labels.append(fr"$U (H-z_{{rel}}) / |w_s| = {reference_length:.2f}$ m")

        if has_conv:
            dcdx_arr = np.asarray(dcdx, dtype=float)
            x_conv_arr = np.asarray(x_conv, dtype=float)
            finite_conv = np.isfinite(dcdx_arr)
            if np.any(finite_conv):
                dcdx_peak = float(np.nanmax(dcdx_arr[finite_conv]))
                dcdx_threshold = dcdx_convergence_fraction * dcdx_peak
                below = dcdx_arr <= dcdx_threshold
                conv_idx = None
                for idx in range(dcdx_arr.size):
                    if below[idx] and np.all(below[idx:]):
                        conv_idx = idx
                        break
                if conv_idx is not None:
                    x_conv_mark = float(x_conv_arr[conv_idx])
                    conv_vline = conv_ax.axvline(
                        x_conv_mark,
                        color="tab:blue",
                        linestyle="--",
                        linewidth=1.4,
                        alpha=0.9,
                    )
                    handles.append(conv_vline)
                    labels.append(
                        f"dC/dx <= {dcdx_convergence_fraction:.0%} of peak at x={x_conv_mark:.2f} m"
                    )

        if has_boundary_cdf:
            surf_ax = conv_ax.twinx()
            boundary_styles = {
                "surface": {"color": "tab:red", "linestyle": "-", "label": "Surface-first CDF"},
                "bed": {"color": "tab:green", "linestyle": "-", "label": "Bed-first CDF"},
            }
            surf_ax.set_ylabel("CDF of first boundary hit")
            surf_ax.set_ylim(0.0, 1.0)
            for boundary_mode in boundary_modes:
                boundary_x = first_boundary_x.get(boundary_mode, [])
                if not boundary_x:
                    continue
                style = boundary_styles[boundary_mode]
                x_boundary_sorted = np.sort(np.asarray(boundary_x, dtype=float))
                cdf_boundary = np.arange(1, x_boundary_sorted.size + 1, dtype=float) / x_boundary_sorted.size
                surf_line, = surf_ax.step(
                    x_boundary_sorted,
                    cdf_boundary,
                    where="post",
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    label=style["label"],
                )
                handles.append(surf_line)
                labels.append(style["label"])
                surf_idx = int(np.searchsorted(cdf_boundary, surface_cdf_target, side="left"))
                if surf_idx < x_boundary_sorted.size:
                    x_surface_mark = float(x_boundary_sorted[surf_idx])
                    surf_vline = conv_ax.axvline(
                        x_surface_mark,
                        color=style["color"],
                        linestyle=":",
                        linewidth=1.4,
                        alpha=0.9,
                    )
                    handles.append(surf_vline)
                    labels.append(f"{surface_cdf_target:.0%} {boundary_mode} hit at x={x_surface_mark:.2f} m")
            surf_ax.grid(False)

        if handles:
            conv_ax.legend(handles, labels, loc="best", frameon=True)
        conv_fig.tight_layout()
    else:
        print("Warning: insufficient valid data to build the longitudinal convergence / first-boundary-hit diagnostics.")

    return fig

def make_figure_advection_diffusion(
    xTr,
    tHist,
    U,
    Kx,
    transport_model="random_walk",
    TL_horizontal=2.0,
    t_eval=None,
    nbins="sturges",
):
    """
    Advection–diffusion validation figure.

    Panel 1 : your Gaussian plume plot
    Panel 2 : mean position vs time
    Panel 3 : variance growth vs time
    """

    import numpy as np
    import matplotlib.pyplot as plt

    xTr = np.asarray(xTr, dtype=float)
    tHist = np.asarray(tHist, dtype=float)

    _Np, Nt = xTr.shape
    transport_model = str(transport_model).strip().lower()

    # ---------------------------------------------------------
    # particle statistics
    # ---------------------------------------------------------

    mu = np.zeros(Nt)
    var = np.zeros(Nt)

    for i in range(Nt):
        x = xTr[:, i]
        x = x[np.isfinite(x)]

        if x.size > 1:
            mu[i] = np.mean(x)
            var[i] = np.var(x)
        else:
            mu[i] = np.nan
            var[i] = np.nan

    x0 = mu[0]
    dt_hist = tHist - tHist[0]

    def _theory_variance(delta_t):
        delta_t = np.asarray(delta_t, dtype=float)
        if transport_model == "langevin":
            tau_l = max(float(TL_horizontal), 1.0e-12)
            return 2.0 * Kx * (delta_t - tau_l * (1.0 - np.exp(-delta_t / tau_l)))
        return 2.0 * Kx * delta_t

    mu_th = x0 + U * dt_hist
    var_th = _theory_variance(dt_hist)
    theory_label = "Langevin theory" if transport_model == "langevin" else "Advection-diffusion theory"

    # ---------------------------------------------------------
    # snapshot index (same logic as your function)
    # ---------------------------------------------------------

    if t_eval is None:
        it = -1
        t_eval = tHist[-1]
    else:
        it = np.argmin(np.abs(tHist - t_eval))
        t_eval = tHist[it]

    dt_eval = t_eval - tHist[0]

    x = xTr[:, it]
    x = x[np.isfinite(x)]

    histogram_bins = nbins
    if isinstance(histogram_bins, str):
        if histogram_bins.strip().lower() != "sturges":
            raise ValueError("nbins must be an integer or 'sturges'.")
        histogram_bins = max(5, int(np.ceil(np.log2(max(x.size, 1))) + 1))
    else:
        histogram_bins = int(histogram_bins)
        if histogram_bins < 5:
            raise ValueError("nbins should be at least 5.")

    hist, edges = np.histogram(x, bins=histogram_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    x_th = np.linspace(edges[0], edges[-1], 500)

    mu_snap = x0 + U * dt_eval
    var_snap = max(float(_theory_variance(dt_eval)), 1.0e-12)

    C_th = (
        1.0
        / np.sqrt(2.0 * np.pi * var_snap)
        * np.exp(-(x_th - mu_snap) ** 2 / (2.0 * var_snap))
    )

    # ---------------------------------------------------------
    # figure
    # ---------------------------------------------------------

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # =========================================================
    # PANEL 1 — your Gaussian plume
    # =========================================================

    ax[0].step(
        centers,
        hist,
        where="mid",
        linewidth=2.0,
        color="black",
        label="Particle distribution",
    )

    ax[0].plot(
        centers,
        hist,
        "o",
        markersize=4,
        color="black",
        alpha=0.6,
    )

    ax[0].plot(
        x_th,
        C_th,
        "-",
        linewidth=2.0,
        color="tab:orange",
        label="Advection–diffusion theory",
    )

    ax[0].lines[-1].set_label(theory_label)
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylabel("Probability density")
    ax[0].set_title(fr"Gaussian plume ($t={t_eval:.1f}$ s)")
    ax[0].legend()
    ax[0].grid(True)

    # =========================================================
    # PANEL 2 — mean position
    # =========================================================

    ax[1].plot(
        tHist,
        mu,
        "o",
        markersize=4,
        color="black",
        alpha=0.6,
        label="Particles",
    )

    ax[1].plot(
        tHist,
        mu_th,
        "-",
        linewidth=2.5,
        color="tab:orange",
        label="Theory",
    )

    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Mean position (m)")
    ax[1].set_title("Mean position evolution")
    ax[1].grid(True)
    ax[1].legend()

    # =========================================================
    # PANEL 3 — variance growth
    # =========================================================

    ax[2].plot(
        tHist,
        var,
        "o",
        markersize=4,
        color="black",
        alpha=0.6,
        label="Particles",
    )

    ax[2].plot(
        tHist,
        var_th,
        "-",
        linewidth=2.5,
        color="tab:orange",
        label="Theory",
    )

    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel(r"$\sigma_x^2$ (m$^2$)")
    ax[2].set_title("Variance growth")
    ax[2].grid(True)
    ax[2].legend()

    fig.tight_layout()

    return fig

