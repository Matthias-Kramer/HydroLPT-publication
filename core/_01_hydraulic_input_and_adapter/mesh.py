from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

try:
    from numba import njit, prange
except ImportError:
    njit = None
    prange = range


if njit is not None:
    @njit(cache=True)
    def _barycentric_in_triangle_numba(px, py, ax, ay, bx, by, cx, cy, tol):
        edge_ab_x = bx - ax
        edge_ab_y = by - ay
        edge_ac_x = cx - ax
        edge_ac_y = cy - ay
        offset_x = px - ax
        offset_y = py - ay

        dot_aa = edge_ab_x * edge_ab_x + edge_ab_y * edge_ab_y
        dot_ab = edge_ab_x * edge_ac_x + edge_ab_y * edge_ac_y
        dot_bb = edge_ac_x * edge_ac_x + edge_ac_y * edge_ac_y
        dot_pa = offset_x * edge_ab_x + offset_y * edge_ab_y
        dot_pb = offset_x * edge_ac_x + offset_y * edge_ac_y

        denominator = dot_aa * dot_bb - dot_ab * dot_ab
        if abs(denominator) < 1e-30:
            nan = np.nan
            return False, nan, nan, nan

        weight_v = (dot_bb * dot_pa - dot_ab * dot_pb) / denominator
        weight_w = (dot_aa * dot_pb - dot_ab * dot_pa) / denominator
        weight_u = 1.0 - weight_v - weight_w
        is_inside = (weight_u >= -tol) and (weight_v >= -tol) and (weight_w >= -tol)
        return is_inside, weight_u, weight_v, weight_w


    @njit(cache=True, parallel=True)
    def _barycentric_batch_numba(triangle_ids, points, v0, v1, v2):
        out = np.full((points.shape[0], 3), np.nan, dtype=np.float64)
        for i in prange(points.shape[0]):
            triangle_id = triangle_ids[i]
            if triangle_id < 0:
                continue

            _, wu, wv, ww = _barycentric_in_triangle_numba(
                points[i, 0],
                points[i, 1],
                v0[triangle_id, 0],
                v0[triangle_id, 1],
                v1[triangle_id, 0],
                v1[triangle_id, 1],
                v2[triangle_id, 0],
                v2[triangle_id, 1],
                1e-12,
            )
            out[i, 0] = wu
            out[i, 1] = wv
            out[i, 2] = ww
        return out


    @njit(cache=True, parallel=True)
    def _point_location_candidates_numba(points, v0, v1, v2, candidate_ids, tol):
        out = np.full(points.shape[0], -1, dtype=np.int64)
        for i in prange(points.shape[0]):
            px = points[i, 0]
            py = points[i, 1]
            for j in range(candidate_ids.shape[1]):
                triangle_id = candidate_ids[i, j]
                if triangle_id < 0:
                    continue

                is_inside, _, _, _ = _barycentric_in_triangle_numba(
                    px,
                    py,
                    v0[triangle_id, 0],
                    v0[triangle_id, 1],
                    v1[triangle_id, 0],
                    v1[triangle_id, 1],
                    v2[triangle_id, 0],
                    v2[triangle_id, 1],
                    tol,
                )
                if is_inside:
                    out[i] = triangle_id
                    break
        return out


    @njit(cache=True, parallel=True)
    def _point_location_local_walk_numba(points, start_triangle_ids, v0, v1, v2, neighbors, tol, max_steps):
        out = np.full(points.shape[0], -1, dtype=np.int64)
        for i in prange(points.shape[0]):
            triangle_id = start_triangle_ids[i]
            if triangle_id < 0:
                continue

            px = points[i, 0]
            py = points[i, 1]

            for _ in range(max_steps):
                is_inside, wu, wv, ww = _barycentric_in_triangle_numba(
                    px,
                    py,
                    v0[triangle_id, 0],
                    v0[triangle_id, 1],
                    v1[triangle_id, 0],
                    v1[triangle_id, 1],
                    v2[triangle_id, 0],
                    v2[triangle_id, 1],
                    tol,
                )
                if is_inside:
                    out[i] = triangle_id
                    break

                edge_index = 0
                min_weight = wu
                if wv < min_weight:
                    edge_index = 1
                    min_weight = wv
                if ww < min_weight:
                    edge_index = 2

                next_triangle = neighbors[triangle_id, edge_index]
                if next_triangle < 0 or next_triangle == triangle_id:
                    break
                triangle_id = next_triangle
        return out


    @njit(cache=True)
    def _point_in_native_cell_numba(px, py, cell_id, xy, cell_facepoint_indexes, bboxes, tol):
        if cell_id < 0 or cell_id >= cell_facepoint_indexes.shape[0]:
            return False
        xmin = bboxes[cell_id, 0]
        xmax = bboxes[cell_id, 1]
        ymin = bboxes[cell_id, 2]
        ymax = bboxes[cell_id, 3]
        if not np.isfinite(xmin):
            return False
        if px < xmin - tol or px > xmax + tol or py < ymin - tol or py > ymax + tol:
            return False

        inside = False
        prev = -1
        first = -1
        last = -1
        for col in range(cell_facepoint_indexes.shape[1]):
            node = cell_facepoint_indexes[cell_id, col]
            if node < 0 or node >= xy.shape[0]:
                continue
            if first < 0:
                first = node
            if prev >= 0 and node == prev:
                continue
            if last >= 0:
                xi = xy[last, 0]
                yi = xy[last, 1]
                xj = xy[node, 0]
                yj = xy[node, 1]
                dx = xj - xi
                dy = yj - yi
                seg_len2 = dx * dx + dy * dy
                if seg_len2 > 0.0:
                    t = ((px - xi) * dx + (py - yi) * dy) / seg_len2
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                    qx = xi + t * dx
                    qy = yi + t * dy
                    ddx = px - qx
                    ddy = py - qy
                    if ddx * ddx + ddy * ddy <= tol * tol:
                        return True
                if ((yi > py) != (yj > py)) and (
                    px < (xj - xi) * (py - yi) / (yj - yi + 1.0e-300) + xi
                ):
                    inside = not inside
            prev = node
            last = node

        if first >= 0 and last >= 0 and first != last:
            xi = xy[last, 0]
            yi = xy[last, 1]
            xj = xy[first, 0]
            yj = xy[first, 1]
            dx = xj - xi
            dy = yj - yi
            seg_len2 = dx * dx + dy * dy
            if seg_len2 > 0.0:
                t = ((px - xi) * dx + (py - yi) * dy) / seg_len2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                qx = xi + t * dx
                qy = yi + t * dy
                ddx = px - qx
                ddy = py - qy
                if ddx * ddx + ddy * ddy <= tol * tol:
                    return True
            if ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (yj - yi + 1.0e-300) + xi
            ):
                inside = not inside
        return inside


    @njit(cache=True, parallel=True)
    def _native_point_location_candidates_numba(points, xy, cell_facepoint_indexes, bboxes, candidate_ids, tol):
        out = np.full(points.shape[0], -1, dtype=np.int64)
        for i in prange(points.shape[0]):
            px = points[i, 0]
            py = points[i, 1]
            for j in range(candidate_ids.shape[1]):
                cell_id = candidate_ids[i, j]
                if _point_in_native_cell_numba(px, py, cell_id, xy, cell_facepoint_indexes, bboxes, tol):
                    out[i] = cell_id
                    break
        return out


    @njit(cache=True, parallel=True)
    def _native_point_location_local_numba(points, start_cell_ids, xy, cell_facepoint_indexes, bboxes, neighbors, tol, max_rings):
        out = np.full(points.shape[0], -1, dtype=np.int64)
        for i in prange(points.shape[0]):
            px = points[i, 0]
            py = points[i, 1]
            start = start_cell_ids[i]
            if _point_in_native_cell_numba(px, py, start, xy, cell_facepoint_indexes, bboxes, tol):
                out[i] = start
                continue
            if start < 0 or start >= neighbors.shape[0]:
                continue
            for j in range(neighbors.shape[1]):
                cell_id = neighbors[start, j]
                if _point_in_native_cell_numba(px, py, cell_id, xy, cell_facepoint_indexes, bboxes, tol):
                    out[i] = cell_id
                    break
            if out[i] >= 0 or max_rings <= 1:
                continue
            for j in range(neighbors.shape[1]):
                first = neighbors[start, j]
                if first < 0 or first >= neighbors.shape[0]:
                    continue
                for k in range(neighbors.shape[1]):
                    cell_id = neighbors[first, k]
                    if cell_id == start:
                        continue
                    if _point_in_native_cell_numba(px, py, cell_id, xy, cell_facepoint_indexes, bboxes, tol):
                        out[i] = cell_id
                        break
                if out[i] >= 0:
                    break
            if out[i] >= 0 or max_rings <= 2:
                continue
            for j in range(neighbors.shape[1]):
                first = neighbors[start, j]
                if first < 0 or first >= neighbors.shape[0]:
                    continue
                for k in range(neighbors.shape[1]):
                    second = neighbors[first, k]
                    if second < 0 or second >= neighbors.shape[0] or second == start:
                        continue
                    for m in range(neighbors.shape[1]):
                        cell_id = neighbors[second, m]
                        if cell_id == start or cell_id == first:
                            continue
                        if _point_in_native_cell_numba(px, py, cell_id, xy, cell_facepoint_indexes, bboxes, tol):
                            out[i] = cell_id
                            break
                    if out[i] >= 0:
                        break
                if out[i] >= 0:
                    break
        return out


    @njit(cache=True, parallel=True)
    def _native_inside_marker_numba(cell_ids, points, xy, cell_facepoint_indexes, bboxes, tol):
        out = np.full((points.shape[0], 3), np.nan, dtype=np.float64)
        for i in prange(points.shape[0]):
            cell_id = cell_ids[i]
            if _point_in_native_cell_numba(
                points[i, 0],
                points[i, 1],
                cell_id,
                xy,
                cell_facepoint_indexes,
                bboxes,
                tol,
            ):
                out[i, 0] = 1.0 / 3.0
                out[i, 1] = 1.0 / 3.0
                out[i, 2] = 1.0 / 3.0
            else:
                out[i, 0] = -1.0
        return out


    @njit(cache=True, parallel=True)
    def _native_contains_points_numba(cell_ids, points, xy, cell_facepoint_indexes, bboxes, tol):
        out = np.zeros(points.shape[0], dtype=np.bool_)
        for i in prange(points.shape[0]):
            out[i] = _point_in_native_cell_numba(
                points[i, 0],
                points[i, 1],
                cell_ids[i],
                xy,
                cell_facepoint_indexes,
                bboxes,
                tol,
            )
        return out


else:
    _barycentric_batch_numba = None
    _point_location_candidates_numba = None
    _point_location_local_walk_numba = None
    _native_point_location_candidates_numba = None
    _native_point_location_local_numba = None
    _native_inside_marker_numba = None
    _native_contains_points_numba = None


class TriMesh:
    """Triangular mesh helper for point location and barycentric coordinates."""

    def __init__(self, xy, tri) -> None:
        self.xy = np.asarray(xy, dtype=float)
        self.tri = np.asarray(tri, dtype=int)

        self._validate_inputs()

        self.Nc = self.tri.shape[0]
        self.V0 = self.xy[self.tri[:, 0]]
        self.V1 = self.xy[self.tri[:, 1]]
        self.V2 = self.xy[self.tri[:, 2]]
        self._xmin = float(np.min(self.xy[:, 0]))
        self._xmax = float(np.max(self.xy[:, 0]))
        self._ymin = float(np.min(self.xy[:, 1]))
        self._ymax = float(np.max(self.xy[:, 1]))
        self.neighbors = self._build_triangle_neighbors()
        centroids = (self.V0 + self.V1 + self.V2) / 3.0
        self.cell_centers = centroids
        self._centroid_tree = cKDTree(centroids)
        self._search_radius = self._estimate_search_radius()
        self._query_workers = self._resolve_query_workers()
        self._local_walk_steps = self._resolve_local_walk_steps()

    def _validate_inputs(self) -> None:
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            raise ValueError(f"xy must have shape (Nn, 2), got {self.xy.shape}")
        if self.tri.ndim != 2 or self.tri.shape[1] != 3:
            raise ValueError(f"tri must have shape (Nc, 3), got {self.tri.shape}")

    def _estimate_search_radius(self) -> float:
        edge_01 = np.linalg.norm(self.V1 - self.V0, axis=1)
        edge_12 = np.linalg.norm(self.V2 - self.V1, axis=1)
        edge_20 = np.linalg.norm(self.V0 - self.V2, axis=1)
        return float(np.median(np.maximum.reduce([edge_01, edge_12, edge_20])))

    def _build_triangle_neighbors(self) -> np.ndarray:
        """Build triangle adjacency indexed by opposite local vertex."""
        neighbors = np.full((self.Nc, 3), -1, dtype=int)
        edge_to_owner: dict[tuple[int, int], tuple[int, int]] = {}

        # local edge index 0/1/2 is opposite local vertex 0/1/2
        local_edges = ((1, 2), (2, 0), (0, 1))
        for tri_id in range(self.Nc):
            tri_nodes = self.tri[tri_id]
            for edge_index, (a_local, b_local) in enumerate(local_edges):
                edge = tuple(sorted((int(tri_nodes[a_local]), int(tri_nodes[b_local]))))
                other = edge_to_owner.get(edge)
                if other is None:
                    edge_to_owner[edge] = (tri_id, edge_index)
                    continue
                other_tri_id, other_edge_index = other
                neighbors[tri_id, edge_index] = other_tri_id
                neighbors[other_tri_id, other_edge_index] = tri_id

        return neighbors

    @staticmethod
    def _resolve_query_workers() -> int | None:
        """Return the cKDTree batch-query worker count, if configured."""
        raw = os.getenv("HYDROLPT_MESH_WORKERS", "-1").strip()
        if not raw:
            return -1
        try:
            workers = int(raw)
        except ValueError:
            return -1
        if workers == 0:
            return None
        return workers

    @staticmethod
    def _resolve_local_walk_steps() -> int:
        """Return the default number of neighbor-walk steps before global fallback."""
        raw = os.getenv("HYDROLPT_MESH_LOCAL_WALK_STEPS", "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
        return 16

    def _query_candidate_ids(self, points: np.ndarray, k: int) -> np.ndarray:
        """Batch-query nearest triangle centroids, using multicore SciPy when available."""
        kwargs = {"k": min(k, self.Nc)}
        if self._query_workers is not None:
            kwargs["workers"] = self._query_workers
        try:
            _, candidate_ids = self._centroid_tree.query(points, **kwargs)
        except TypeError:
            kwargs.pop("workers", None)
            _, candidate_ids = self._centroid_tree.query(points, **kwargs)
        if np.ndim(candidate_ids) == 1:
            candidate_ids = candidate_ids[:, None]
        return candidate_ids

    def _inside_domain_bounds(self, points: np.ndarray) -> np.ndarray:
        """Return a cheap axis-aligned precheck for points inside the mesh extent."""
        return (
            (points[:, 0] >= self._xmin)
            & (points[:, 0] <= self._xmax)
            & (points[:, 1] >= self._ymin)
            & (points[:, 1] <= self._ymax)
        )

    def _point_location_global(self, points: np.ndarray, k: int = 64) -> np.ndarray:
        """Global point-location fallback using KD-tree candidate search."""
        triangle_ids = np.full(points.shape[0], -1, dtype=int)
        if points.shape[0] == 0:
            return triangle_ids

        inside_bounds = self._inside_domain_bounds(points)
        if not np.any(inside_bounds):
            return triangle_ids

        active_points = points[inside_bounds]
        candidate_ids = self._query_candidate_ids(active_points, k)

        if _point_location_candidates_numba is not None:
            active_triangle_ids = _point_location_candidates_numba(
                active_points,
                self.V0,
                self.V1,
                self.V2,
                candidate_ids,
                1e-12,
            )

            missing = active_triangle_ids < 0
            if np.any(missing):
                missing_idx = np.where(missing)[0]
                for point_index in missing_idx:
                    active_triangle_ids[point_index] = self._find_triangle_in_candidates(
                        active_points[point_index],
                        self._centroid_tree.query_ball_point(
                            active_points[point_index],
                            r=10.0 * self._search_radius,
                        ),
                    )
            triangle_ids[inside_bounds] = active_triangle_ids
            return triangle_ids

        active_triangle_ids = np.full(active_points.shape[0], -1, dtype=int)
        for point_index, point in enumerate(active_points):
            triangle_id = self._find_triangle_in_candidates(point, candidate_ids[point_index])
            if triangle_id >= 0:
                active_triangle_ids[point_index] = triangle_id
                continue

            fallback_candidates = self._centroid_tree.query_ball_point(
                point,
                r=10.0 * self._search_radius,
            )
            active_triangle_ids[point_index] = self._find_triangle_in_candidates(
                point,
                fallback_candidates,
            )

        triangle_ids[inside_bounds] = active_triangle_ids
        return triangle_ids

    def point_location_local(self, x, y, start_tid, max_steps: int = 8, k: int = 64) -> np.ndarray:
        """Find containing triangles by walking through neighbors before global fallback."""
        points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
        start_triangle_ids = np.asarray(start_tid, dtype=int).ravel()
        if start_triangle_ids.shape[0] != points.shape[0]:
            raise ValueError("start_tid and points must have the same length")
        if max_steps == 8:
            max_steps = self._local_walk_steps

        if _point_location_local_walk_numba is not None:
            triangle_ids = _point_location_local_walk_numba(
                points,
                start_triangle_ids,
                self.V0,
                self.V1,
                self.V2,
                self.neighbors,
                1e-12,
                max(1, int(max_steps)),
            )
        else:
            triangle_ids = np.full(points.shape[0], -1, dtype=int)

        unresolved = triangle_ids < 0
        if np.any(unresolved):
            triangle_ids[unresolved] = self._point_location_global(points[unresolved], k=k)
        return triangle_ids

    @staticmethod
    def _barycentric_in_triangle(
        point: np.ndarray,
        vertex_a: np.ndarray,
        vertex_b: np.ndarray,
        vertex_c: np.ndarray,
        tol: float = 1e-12,
    ) -> tuple[bool, tuple[float, float, float]]:
        """Return whether a point lies in a triangle and its barycentric weights."""
        edge_ab = vertex_b - vertex_a
        edge_ac = vertex_c - vertex_a
        offset = point - vertex_a

        dot_aa = np.dot(edge_ab, edge_ab)
        dot_ab = np.dot(edge_ab, edge_ac)
        dot_bb = np.dot(edge_ac, edge_ac)
        dot_pa = np.dot(offset, edge_ab)
        dot_pb = np.dot(offset, edge_ac)

        denominator = dot_aa * dot_bb - dot_ab * dot_ab
        if abs(denominator) < 1e-30:
            return False, (np.nan, np.nan, np.nan)

        weight_v = (dot_bb * dot_pa - dot_ab * dot_pb) / denominator
        weight_w = (dot_aa * dot_pb - dot_ab * dot_pa) / denominator
        weight_u = 1.0 - weight_v - weight_w

        is_inside = (
            weight_u >= -tol
            and weight_v >= -tol
            and weight_w >= -tol
        )
        return is_inside, (weight_u, weight_v, weight_w)

    def _find_triangle_in_candidates(
        self,
        point: np.ndarray,
        candidate_ids: Iterable[int],
        tol: float = 1e-12,
    ) -> int:
        """Return the first containing triangle id or `-1` if none match."""
        for triangle_id in candidate_ids:
            vertex_a = self.V0[triangle_id]
            vertex_b = self.V1[triangle_id]
            vertex_c = self.V2[triangle_id]
            is_inside, _ = self._barycentric_in_triangle(
                point,
                vertex_a,
                vertex_b,
                vertex_c,
                tol=tol,
            )
            if is_inside:
                return int(triangle_id)
        return -1

    def point_location(self, x, y, k: int = 64) -> np.ndarray:
        """Find the containing triangle id for each point."""
        points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
        return self._point_location_global(points, k=k)

    def barycentric(self, tid, pts_xy) -> np.ndarray:
        """Compute barycentric coordinates for points in known triangle ids."""
        points = np.asarray(pts_xy, dtype=float)
        triangle_ids = np.asarray(tid, dtype=int)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"pts_xy must have shape (N, 2), got {points.shape}")
        if triangle_ids.shape[0] != points.shape[0]:
            raise ValueError("tid and pts_xy must have the same length")

        if _barycentric_batch_numba is not None:
            return _barycentric_batch_numba(
                triangle_ids,
                points,
                self.V0,
                self.V1,
                self.V2,
            )

        barycentric_coords = np.full((points.shape[0], 3), np.nan, dtype=float)

        for index, triangle_id in enumerate(triangle_ids):
            if triangle_id < 0:
                continue
            vertex_a = self.V0[triangle_id]
            vertex_b = self.V1[triangle_id]
            vertex_c = self.V2[triangle_id]
            _, barycentric_weights = self._barycentric_in_triangle(
                points[index],
                vertex_a,
                vertex_b,
                vertex_c,
            )
            barycentric_coords[index] = barycentric_weights

        return barycentric_coords

    def contains_points(self, tid, pts_xy, tol: float = 1.0e-9) -> np.ndarray:
        """Return whether each point lies inside the corresponding triangle."""
        barycentric_coords = self.barycentric(tid, pts_xy)
        return np.all(
            (barycentric_coords >= -float(tol)) & (barycentric_coords <= 1.0 + float(tol)),
            axis=1,
        )


    def free_boundary_edges(self) -> np.ndarray:
        """Return edges that belong to exactly one triangle."""
        edges = np.vstack(
            [
                self.tri[:, [0, 1]],
                self.tri[:, [1, 2]],
                self.tri[:, [2, 0]],
            ]
        )
        edges = np.sort(edges, axis=1)

        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        return unique_edges[counts == 1]


class HecRasNativeMesh:
    """Native HEC-RAS polygon mesh helper.

    The public lookup methods intentionally mirror ``TriMesh`` so the transport
    solver can work with cell ids without knowing whether the backend is
    triangular or polygonal.
    """

    is_native_hecras = True

    def __init__(
        self,
        *,
        facepoints_xy,
        cell_facepoint_indexes,
        cell_centers_xy,
        faces_cell_indexes=None,
        faces_facepoint_indexes=None,
        facepoints_is_perimeter=None,
    ) -> None:
        self.xy = np.asarray(facepoints_xy, dtype=float)
        self.cell_facepoint_indexes = np.asarray(cell_facepoint_indexes, dtype=int)
        self.cell_centers = np.asarray(cell_centers_xy, dtype=float)
        self.faces_cell_indexes = (
            None if faces_cell_indexes is None else np.asarray(faces_cell_indexes, dtype=int)
        )
        self.faces_facepoint_indexes = (
            None if faces_facepoint_indexes is None else np.asarray(faces_facepoint_indexes, dtype=int)
        )
        self.facepoints_is_perimeter = (
            None if facepoints_is_perimeter is None else np.asarray(facepoints_is_perimeter, dtype=bool).ravel()
        )

        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            raise ValueError(f"facepoints_xy must have shape (N, 2), got {self.xy.shape}")
        if self.cell_facepoint_indexes.ndim != 2:
            raise ValueError("cell_facepoint_indexes must be a 2D padded array")
        if self.cell_centers.ndim != 2 or self.cell_centers.shape[1] != 2:
            raise ValueError(f"cell_centers_xy must have shape (Nc, 2), got {self.cell_centers.shape}")
        if self.cell_centers.shape[0] != self.cell_facepoint_indexes.shape[0]:
            raise ValueError("cell_centers_xy and cell_facepoint_indexes disagree on Nc")

        self.Nc = int(self.cell_centers.shape[0])
        self.tri = np.empty((0, 3), dtype=int)
        self._xmin = float(np.min(self.xy[:, 0]))
        self._xmax = float(np.max(self.xy[:, 0]))
        self._ymin = float(np.min(self.xy[:, 1]))
        self._ymax = float(np.max(self.xy[:, 1]))
        self._polygons = [self._clean_polygon(row) for row in self.cell_facepoint_indexes]
        self._bbox = self._build_cell_bboxes()
        self._cell_facepoint_indexes_clean = self._build_clean_cell_facepoint_array()
        self.neighbors = self._build_cell_neighbors()
        self._centroid_tree = cKDTree(self.cell_centers)
        self._query_workers = TriMesh._resolve_query_workers()
        self._local_walk_steps = TriMesh._resolve_local_walk_steps()

    def _clean_polygon(self, row: np.ndarray) -> np.ndarray:
        valid = np.asarray(row, dtype=int).ravel()
        valid = valid[(valid >= 0) & (valid < self.xy.shape[0])]
        if valid.size >= 2 and valid[0] == valid[-1]:
            valid = valid[:-1]
        if valid.size <= 1:
            return valid
        keep = [int(valid[0])]
        for value in valid[1:]:
            if int(value) != keep[-1]:
                keep.append(int(value))
        if len(keep) >= 2 and keep[0] == keep[-1]:
            keep.pop()
        return np.asarray(keep, dtype=int)

    def _build_cell_bboxes(self) -> np.ndarray:
        bboxes = np.full((self.Nc, 4), np.nan, dtype=float)
        for cell_id, poly_ids in enumerate(self._polygons):
            if poly_ids.size == 0:
                continue
            pts = self.xy[poly_ids]
            bboxes[cell_id] = [
                float(np.min(pts[:, 0])),
                float(np.max(pts[:, 0])),
                float(np.min(pts[:, 1])),
                float(np.max(pts[:, 1])),
            ]
        return bboxes

    def _build_clean_cell_facepoint_array(self) -> np.ndarray:
        max_vertices = max((poly.size for poly in self._polygons), default=0)
        clean = np.full((self.Nc, max(max_vertices, 1)), -1, dtype=int)
        for cell_id, poly_ids in enumerate(self._polygons):
            if poly_ids.size:
                clean[cell_id, : poly_ids.size] = poly_ids
        return clean

    def _build_cell_neighbors(self) -> np.ndarray:
        neighbor_sets: list[set[int]] = [set() for _ in range(self.Nc)]
        if self.faces_cell_indexes is not None:
            for row in self.faces_cell_indexes:
                cells = [int(c) for c in np.asarray(row).ravel() if 0 <= int(c) < self.Nc]
                if len(cells) >= 2 and cells[0] != cells[1]:
                    a, b = cells[:2]
                    neighbor_sets[a].add(b)
                    neighbor_sets[b].add(a)
        else:
            edge_owner: dict[tuple[int, int], int] = {}
            for cell_id, poly_ids in enumerate(self._polygons):
                for a, b in self._polygon_edges(poly_ids):
                    key = tuple(sorted((int(a), int(b))))
                    other = edge_owner.get(key)
                    if other is None:
                        edge_owner[key] = cell_id
                    elif other != cell_id:
                        neighbor_sets[cell_id].add(other)
                        neighbor_sets[other].add(cell_id)

        max_neighbors = max((len(values) for values in neighbor_sets), default=0)
        neighbors = np.full((self.Nc, max(max_neighbors, 1)), -1, dtype=int)
        for cell_id, values in enumerate(neighbor_sets):
            if values:
                ordered = sorted(values)
                neighbors[cell_id, : len(ordered)] = ordered
        return neighbors

    @staticmethod
    def _polygon_edges(poly_ids: np.ndarray) -> list[tuple[int, int]]:
        if poly_ids.size < 2:
            return []
        return [
            (int(poly_ids[i]), int(poly_ids[(i + 1) % poly_ids.size]))
            for i in range(poly_ids.size)
        ]

    def _inside_domain_bounds(self, points: np.ndarray) -> np.ndarray:
        return (
            (points[:, 0] >= self._xmin)
            & (points[:, 0] <= self._xmax)
            & (points[:, 1] >= self._ymin)
            & (points[:, 1] <= self._ymax)
        )

    def _query_candidate_ids(self, points: np.ndarray, k: int) -> np.ndarray:
        kwargs = {"k": min(max(int(k), 1), self.Nc)}
        if self._query_workers is not None:
            kwargs["workers"] = self._query_workers
        try:
            _, candidate_ids = self._centroid_tree.query(points, **kwargs)
        except TypeError:
            kwargs.pop("workers", None)
            _, candidate_ids = self._centroid_tree.query(points, **kwargs)
        if np.ndim(candidate_ids) == 1:
            candidate_ids = candidate_ids[:, None]
        return candidate_ids

    def _point_in_cell(self, point: np.ndarray, cell_id: int, tol: float = 1.0e-10) -> bool:
        if cell_id < 0 or cell_id >= self.Nc:
            return False
        xmin, xmax, ymin, ymax = self._bbox[cell_id]
        if not np.isfinite(xmin):
            return False
        x = float(point[0])
        y = float(point[1])
        if x < xmin - tol or x > xmax + tol or y < ymin - tol or y > ymax + tol:
            return False
        poly_ids = self._polygons[cell_id]
        if poly_ids.size < 3:
            return False
        poly = self.xy[poly_ids]
        inside = False
        j = poly.shape[0] - 1
        for i in range(poly.shape[0]):
            xi, yi = poly[i]
            xj, yj = poly[j]
            dx = xj - xi
            dy = yj - yi
            seg_len2 = dx * dx + dy * dy
            if seg_len2 > 0.0:
                t = np.clip(((x - xi) * dx + (y - yi) * dy) / seg_len2, 0.0, 1.0)
                px = xi + t * dx
                py = yi + t * dy
                if (x - px) * (x - px) + (y - py) * (y - py) <= tol * tol:
                    return True
            crosses = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1.0e-300) + xi
            )
            if crosses:
                inside = not inside
            j = i
        return inside

    def _find_cell_in_candidates(self, point: np.ndarray, candidate_ids) -> int:
        for cell_id in np.asarray(candidate_ids, dtype=int).ravel():
            if self._point_in_cell(point, int(cell_id)):
                return int(cell_id)
        return -1

    def point_location(self, x, y, k: int = 64) -> np.ndarray:
        points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
        cell_ids = np.full(points.shape[0], -1, dtype=int)
        if points.shape[0] == 0:
            return cell_ids
        inside_bounds = self._inside_domain_bounds(points)
        if not np.any(inside_bounds):
            return cell_ids
        active = points[inside_bounds]
        candidate_ids = self._query_candidate_ids(active, k)
        if _native_point_location_candidates_numba is not None:
            active_ids = _native_point_location_candidates_numba(
                active,
                self.xy,
                self._cell_facepoint_indexes_clean,
                self._bbox,
                candidate_ids,
                1.0e-10,
            )
            missing = active_ids < 0
            if np.any(missing):
                expanded = self._query_candidate_ids(active[missing], max(int(k) * 4, 256))
                active_ids[missing] = _native_point_location_candidates_numba(
                    active[missing],
                    self.xy,
                    self._cell_facepoint_indexes_clean,
                    self._bbox,
                    expanded,
                    1.0e-10,
                )
        else:
            active_ids = np.full(active.shape[0], -1, dtype=int)
            for i, point in enumerate(active):
                found = self._find_cell_in_candidates(point, candidate_ids[i])
                if found < 0:
                    expanded = self._query_candidate_ids(point[None, :], max(int(k) * 4, 256))[0]
                    found = self._find_cell_in_candidates(point, expanded)
                active_ids[i] = found
        cell_ids[inside_bounds] = active_ids
        return cell_ids

    def point_location_local(self, x, y, start_tid, max_steps: int = 8, k: int = 64) -> np.ndarray:
        points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
        start_ids = np.asarray(start_tid, dtype=int).ravel()
        if start_ids.shape[0] != points.shape[0]:
            raise ValueError("start_tid and points must have the same length")
        if _native_point_location_local_numba is not None:
            out = _native_point_location_local_numba(
                points,
                start_ids,
                self.xy,
                self._cell_facepoint_indexes_clean,
                self._bbox,
                self.neighbors,
                1.0e-10,
                max(1, min(int(max_steps), 3)),
            )
            unresolved = out < 0
            if np.any(unresolved):
                out[unresolved] = self.point_location(points[unresolved, 0], points[unresolved, 1], k=k)
            return out
        out = np.full(points.shape[0], -1, dtype=int)
        for i, point in enumerate(points):
            start = int(start_ids[i])
            candidates: list[int] = []
            if 0 <= start < self.Nc:
                candidates.append(start)
                frontier = [start]
                seen = {start}
                for _ in range(max(1, int(max_steps))):
                    next_frontier: list[int] = []
                    for cell_id in frontier:
                        for nbr in self.neighbors[cell_id]:
                            nbr_i = int(nbr)
                            if nbr_i >= 0 and nbr_i not in seen:
                                candidates.append(nbr_i)
                                next_frontier.append(nbr_i)
                                seen.add(nbr_i)
                    if not next_frontier:
                        break
                    frontier = next_frontier
            found = self._find_cell_in_candidates(point, candidates)
            if found < 0:
                found = self.point_location([point[0]], [point[1]], k=k)[0]
            out[i] = found
        return out

    def barycentric(self, tid, pts_xy) -> np.ndarray:
        """Return triangle-like inside markers for polygon cells.

        Existing solver code only checks whether all three values lie in
        ``[0, 1]``. For native polygons we return a valid triplet for inside
        points and invalid values for outside points.
        """
        points = np.asarray(pts_xy, dtype=float)
        cell_ids = np.asarray(tid, dtype=int).ravel()
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"pts_xy must have shape (N, 2), got {points.shape}")
        if cell_ids.shape[0] != points.shape[0]:
            raise ValueError("tid and pts_xy must have the same length")
        if _native_inside_marker_numba is not None:
            return _native_inside_marker_numba(
                cell_ids,
                points,
                self.xy,
                self._cell_facepoint_indexes_clean,
                self._bbox,
                1.0e-10,
            )
        out = np.full((points.shape[0], 3), np.nan, dtype=float)
        for i, cell_id in enumerate(cell_ids):
            if self._point_in_cell(points[i], int(cell_id)):
                out[i] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
            else:
                out[i] = (-1.0, np.nan, np.nan)
        return out

    def contains_points(self, tid, pts_xy, tol: float = 1.0e-10) -> np.ndarray:
        """Return whether each point lies inside the corresponding native cell."""
        points = np.asarray(pts_xy, dtype=float)
        cell_ids = np.asarray(tid, dtype=int).ravel()
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"pts_xy must have shape (N, 2), got {points.shape}")
        if cell_ids.shape[0] != points.shape[0]:
            raise ValueError("tid and pts_xy must have the same length")
        if _native_contains_points_numba is not None:
            return _native_contains_points_numba(
                cell_ids,
                points,
                self.xy,
                self._cell_facepoint_indexes_clean,
                self._bbox,
                float(tol),
            )
        out = np.zeros(points.shape[0], dtype=bool)
        for i, cell_id in enumerate(cell_ids):
            out[i] = self._point_in_cell(points[i], int(cell_id), tol=float(tol))
        return out

    def free_boundary_edges(self) -> np.ndarray:
        if self.faces_facepoint_indexes is not None and self.facepoints_is_perimeter is not None:
            edge_ids = np.asarray(self.faces_facepoint_indexes, dtype=int)
            valid = (
                (edge_ids[:, 0] >= 0)
                & (edge_ids[:, 1] >= 0)
                & (edge_ids[:, 0] < self.facepoints_is_perimeter.size)
                & (edge_ids[:, 1] < self.facepoints_is_perimeter.size)
            )
            perimeter = np.zeros(edge_ids.shape[0], dtype=bool)
            perimeter[valid] = (
                self.facepoints_is_perimeter[edge_ids[valid, 0]]
                & self.facepoints_is_perimeter[edge_ids[valid, 1]]
            )
            return np.asarray(edge_ids[perimeter], dtype=int)

        if self.faces_cell_indexes is None or self.faces_facepoint_indexes is None:
            edge_counts: dict[tuple[int, int], int] = {}
            for poly_ids in self._polygons:
                for edge in self._polygon_edges(poly_ids):
                    key = tuple(sorted(edge))
                    edge_counts[key] = edge_counts.get(key, 0) + 1
            return np.asarray([edge for edge, count in edge_counts.items() if count == 1], dtype=int)
        boundary = []
        for face_id, cells in enumerate(self.faces_cell_indexes):
            valid = [int(c) for c in np.asarray(cells).ravel() if 0 <= int(c) < self.Nc]
            if len(valid) < 2:
                boundary.append(self.faces_facepoint_indexes[face_id])
        if not boundary:
            return np.empty((0, 2), dtype=int)
        return np.asarray(boundary, dtype=int)

    def all_edges(self) -> np.ndarray:
        """Return native face edges for plotting the complete underlying mesh."""
        if self.faces_facepoint_indexes is not None:
            return np.asarray(self.faces_facepoint_indexes, dtype=int)
        edges = set()
        for poly_ids in self._polygons:
            for edge in self._polygon_edges(poly_ids):
                edges.add(tuple(sorted(edge)))
        if not edges:
            return np.empty((0, 2), dtype=int)
        return np.asarray(sorted(edges), dtype=int)

    def wet_dry_boundary_edges(self, wet_mask) -> np.ndarray:
        """Return native face edges separating wet and dry cells."""
        edges, _wet_cells = self.wet_dry_boundary_edges_with_wet_cells(wet_mask)
        return edges

    def wet_dry_boundary_edges_with_wet_cells(self, wet_mask) -> tuple[np.ndarray, np.ndarray]:
        """Return wet/dry face edges and the wet-side cell id for each edge."""
        wet = np.asarray(wet_mask, dtype=bool).ravel()
        if wet.size != self.Nc:
            return np.empty((0, 2), dtype=int), np.empty(0, dtype=int)
        if self.faces_cell_indexes is not None and self.faces_facepoint_indexes is not None:
            cells = np.asarray(self.faces_cell_indexes, dtype=int)
            valid = (
                (cells[:, 0] >= 0)
                & (cells[:, 1] >= 0)
                & (cells[:, 0] < wet.size)
                & (cells[:, 1] < wet.size)
                & (wet[cells[:, 0]] != wet[cells[:, 1]])
            )
            edge_cells = cells[valid]
            wet_side = np.where(wet[edge_cells[:, 0]], edge_cells[:, 0], edge_cells[:, 1])
            return np.asarray(self.faces_facepoint_indexes[valid], dtype=int), np.asarray(wet_side, dtype=int)

        edge_owner: dict[tuple[int, int], tuple[int, bool]] = {}
        wet_edges: list[tuple[int, int]] = []
        wet_cells: list[int] = []
        for cell_id, poly_ids in enumerate(self._polygons):
            for edge in self._polygon_edges(poly_ids):
                key = tuple(sorted(edge))
                previous = edge_owner.get(key)
                if previous is None:
                    edge_owner[key] = (cell_id, bool(wet[cell_id]))
                elif previous[1] != bool(wet[cell_id]):
                    wet_edges.append(key)
                    wet_cells.append(cell_id if bool(wet[cell_id]) else previous[0])
        if not wet_edges:
            return np.empty((0, 2), dtype=int), np.empty(0, dtype=int)
        return np.asarray(wet_edges, dtype=int), np.asarray(wet_cells, dtype=int)

