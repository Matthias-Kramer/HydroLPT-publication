"""Adapter for HEC-RAS 2D plan HDF hydraulic results.

This adapter reads a HEC-RAS unsteady plan HDF file (for example
``Muncie.p04.hdf``) and exposes the same meta/frame interface used by the
existing HydroLPT backends.

Implementation notes
--------------------
- HEC-RAS stores polygonal 2D cells. HydroLPT currently expects triangular
  cells, so the adapter triangulates each polygon internally for point
  location while repeating parent-cell hydraulic values onto the generated
  sub-triangles.
- HEC-RAS stores face-normal velocity magnitudes rather than direct
  cell-centred ``U``/``V``. We reconstruct approximate cell-centred velocity
  vectors with a weighted least-squares fit against the adjacent face normals.
- When the HEC-RAS file is in US customary units, geometry and hydraulic
  outputs are converted to SI by default so they remain compatible with the
  rest of HydroLPT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from core._01_hydraulic_input_and_adapter.basement import BasementFrame, BasementMeta
from core._01_hydraulic_input_and_adapter.mesh import HecRasNativeMesh
from core._04_transport_solver.runner import run_hydraulic_case

FT_TO_M = 0.3048
PSF_TO_PA = 47.88025898033584
GRAVITY_SI = 9.80665
GRAVITY_US = 32.174048556
SECONDS_PER_DAY = 86400.0


def _decode_attr(value: Any) -> str:
    """Decode HDF5 byte/string attributes into stripped Python strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return " ".join(
            item.decode("utf-8", errors="replace").strip()
            for item in value.tolist()
        ).strip()
    return str(value).strip()


def _to_bool_attr(value: Any) -> bool:
    """Interpret HDF attributes such as ``b'True'`` robustly."""
    return _decode_attr(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_dataset(group: h5py.File | h5py.Group, path: str) -> np.ndarray:
    """Return a dataset as a NumPy array with a clear missing-path error."""
    if path not in group:
        raise KeyError(f"Dataset not found in HEC-RAS HDF: {path}")
    return np.asarray(group[path][()])


def _optional_dataset_path(group: h5py.File | h5py.Group, candidates: list[str]) -> str | None:
    """Return the first existing HDF5 dataset path from a candidate list."""
    for path in candidates:
        if path in group:
            return path
    return None


def _pick_first_2d_flow_area(h5file: h5py.File) -> str:
    """Return the first available HEC-RAS 2D flow area name."""
    group_path = "Geometry/2D Flow Areas"
    if group_path not in h5file:
        raise KeyError("Missing HEC-RAS geometry group: Geometry/2D Flow Areas")
    names = [
        key for key in h5file[group_path].keys()
        if key not in {"Attributes", "Cell Info", "Cell Points", "Polygon Info", "Polygon Parts", "Polygon Points"}
    ]
    if not names:
        raise KeyError("No 2D flow areas found in HEC-RAS geometry.")
    return names[0]


def _polygon_signed_area(poly_xy: np.ndarray) -> float:
    """Return the signed area of a polygon ring."""
    poly_xy = np.asarray(poly_xy, dtype=float)
    if poly_xy.ndim != 2 or poly_xy.shape[0] < 3 or poly_xy.shape[1] != 2:
        return 0.0
    x = poly_xy[:, 0]
    y = poly_xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _cross_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the signed z-component of the 2D cross product for points a-b-c."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _point_in_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, tol: float = 1.0e-12) -> bool:
    """Return whether a point lies inside or on a triangle."""
    p = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    c0 = _cross_2d(a, b, p)
    c1 = _cross_2d(b, c, p)
    c2 = _cross_2d(c, a, p)
    has_neg = (c0 < -tol) or (c1 < -tol) or (c2 < -tol)
    has_pos = (c0 > tol) or (c1 > tol) or (c2 > tol)
    return not (has_neg and has_pos)


def _clean_polygon_vertices(vertex_ids: list[int], xy: np.ndarray, tol: float = 1.0e-10) -> list[int]:
    """Remove duplicate and nearly-collinear vertices from a polygon ring."""
    if len(vertex_ids) < 3:
        return []

    cleaned: list[int] = []
    for vertex in vertex_ids:
        if not cleaned or cleaned[-1] != vertex:
            cleaned.append(int(vertex))
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if len(cleaned) < 3:
        return []

    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        next_cleaned: list[int] = []
        n = len(cleaned)
        for i, vid in enumerate(cleaned):
            prev_vid = cleaned[(i - 1) % n]
            next_vid = cleaned[(i + 1) % n]
            a = xy[prev_vid]
            b = xy[vid]
            c = xy[next_vid]
            if np.linalg.norm(b - a) <= tol or np.linalg.norm(c - b) <= tol:
                changed = True
                continue
            if abs(_cross_2d(a, b, c)) <= tol:
                changed = True
                continue
            next_cleaned.append(vid)
        cleaned = next_cleaned

    if len(cleaned) < 3:
        return []
    return cleaned


def _triangulate_polygon_center_fan(
    vertex_ids: list[int],
    xy: np.ndarray,
    center_xy: np.ndarray | None = None,
) -> tuple[list[list[int]], np.ndarray | None]:
    """Triangulate a polygon by connecting each edge to a centre point."""
    if len(vertex_ids) < 3:
        return [], None
    if len(vertex_ids) == 3:
        return [[int(vertex_ids[0]), int(vertex_ids[1]), int(vertex_ids[2])]], None

    ordered = list(vertex_ids)
    poly_xy = xy[ordered]
    if _polygon_signed_area(poly_xy) < 0.0:
        ordered.reverse()
        poly_xy = xy[ordered]

    if center_xy is None:
        center = np.mean(poly_xy, axis=0)
    else:
        center = np.asarray(center_xy, dtype=float)

    triangles: list[list[int]] = []
    center_node_id = xy.shape[0]
    for local_id in range(len(ordered)):
        v0 = ordered[local_id]
        v1 = ordered[(local_id + 1) % len(ordered)]
        if v0 == v1:
            continue
        triangles.append([int(v0), int(v1), int(center_node_id)])

    return triangles, center


def _triangulate_polygon_ear_clip(vertex_ids: list[int], xy: np.ndarray) -> list[list[int]]:
    """Triangulate a simple polygon using basic ear clipping."""
    if len(vertex_ids) < 3:
        return []
    if len(vertex_ids) == 3:
        return [[int(vertex_ids[0]), int(vertex_ids[1]), int(vertex_ids[2])]]

    ordered = list(vertex_ids)
    poly_xy = xy[ordered]
    if _polygon_signed_area(poly_xy) < 0.0:
        ordered.reverse()

    remaining = ordered.copy()
    triangles: list[list[int]] = []
    max_iter = len(remaining) * len(remaining)
    iterations = 0

    while len(remaining) > 3 and iterations < max_iter:
        iterations += 1
        ear_found = False
        n = len(remaining)
        for i in range(n):
            prev_vid = remaining[(i - 1) % n]
            curr_vid = remaining[i]
            next_vid = remaining[(i + 1) % n]

            a = xy[prev_vid]
            b = xy[curr_vid]
            c = xy[next_vid]

            if _cross_2d(a, b, c) <= 1.0e-12:
                continue

            contains_vertex = False
            for other_vid in remaining:
                if other_vid in {prev_vid, curr_vid, next_vid}:
                    continue
                if _point_in_triangle(xy[other_vid], a, b, c):
                    contains_vertex = True
                    break
            if contains_vertex:
                continue

            triangles.append([int(prev_vid), int(curr_vid), int(next_vid)])
            remaining.pop(i)
            ear_found = True
            break

        if not ear_found:
            return []

    if len(remaining) == 3:
        triangles.append([int(remaining[0]), int(remaining[1]), int(remaining[2])])
    return triangles


def _triangulate_cells(
    facepoints_xy: np.ndarray,
    cell_centers_xy: np.ndarray,
    cell_facepoint_indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate polygon cells with a centre-fan default and ear-clipping fallback."""
    xy = np.asarray(facepoints_xy, dtype=float)
    centers = np.asarray(cell_centers_xy, dtype=float)
    cell_facepoint_indexes = np.asarray(cell_facepoint_indexes, dtype=int)

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Face-point coordinates must have shape (N,2), got {xy.shape}")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"Cell-centre coordinates must have shape (N,2), got {centers.shape}")

    xy_out = np.array(xy, dtype=float, copy=True)
    tri: list[list[int]] = []
    parent_cell_index: list[int] = []

    for cell_id, raw_vertices in enumerate(cell_facepoint_indexes):
        vertex_ids = [int(v) for v in raw_vertices if int(v) >= 0]
        cleaned = _clean_polygon_vertices(vertex_ids, xy_out)
        if len(cleaned) < 3:
            continue

        cell_triangles, center_point = _triangulate_polygon_center_fan(
            cleaned,
            xy_out,
            centers[cell_id],
        )
        if not cell_triangles:
            cell_triangles = _triangulate_polygon_ear_clip(cleaned, xy_out)
            center_point = None
        if center_point is not None:
            center_node_id = xy_out.shape[0]
            xy_out = np.vstack([xy_out, center_point])
            for triangle in cell_triangles:
                triangle[2] = center_node_id

        tri.extend(cell_triangles)
        parent_cell_index.extend([cell_id] * len(cell_triangles))

    if not tri:
        raise ValueError("Could not triangulate any HEC-RAS 2D cells.")

    return (
        np.asarray(xy_out, dtype=float),
        np.asarray(tri, dtype=int),
        np.asarray(parent_cell_index, dtype=int),
    )


def _fit_cell_velocity_vectors(
    face_velocity: np.ndarray,
    face_normals_and_length: np.ndarray,
    cells_face_info: np.ndarray,
    cells_face_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate cell-centred velocity vectors from face-normal velocities."""
    face_velocity = np.asarray(face_velocity, dtype=float).ravel()
    face_normals_and_length = np.asarray(face_normals_and_length, dtype=float)
    cells_face_info = np.asarray(cells_face_info, dtype=int)
    cells_face_values = np.asarray(cells_face_values, dtype=int)

    n_cells = cells_face_info.shape[0]
    U_c = np.zeros(n_cells, dtype=float)
    V_c = np.zeros(n_cells, dtype=float)

    normals = face_normals_and_length[:, :2]
    lengths = face_normals_and_length[:, 2]

    for cell_id in range(n_cells):
        start, count = cells_face_info[cell_id]
        if count <= 0:
            continue

        cell_rows = cells_face_values[start:start + count]
        face_ids = cell_rows[:, 0].astype(int, copy=False)
        valid = (
            (face_ids >= 0)
            & (face_ids < face_velocity.size)
            & np.isfinite(face_velocity[face_ids])
        )
        if not np.any(valid):
            continue

        face_ids = face_ids[valid]
        n = normals[face_ids]
        b = face_velocity[face_ids]
        w = np.maximum(lengths[face_ids], 1e-12)

        finite = np.isfinite(n[:, 0]) & np.isfinite(n[:, 1]) & np.isfinite(b) & np.isfinite(w)
        if not np.any(finite):
            continue

        n = n[finite]
        b = b[finite]
        w = w[finite]

        if n.shape[0] == 1:
            U_c[cell_id] = b[0] * n[0, 0]
            V_c[cell_id] = b[0] * n[0, 1]
            continue

        a = n * np.sqrt(w)[:, None]
        rhs = b * np.sqrt(w)
        try:
            uv, *_ = np.linalg.lstsq(a, rhs, rcond=None)
        except np.linalg.LinAlgError:
            uv = np.zeros(2, dtype=float)
        U_c[cell_id] = float(uv[0])
        V_c[cell_id] = float(uv[1])

    return U_c, V_c


def _average_face_values_to_cells(
    face_values: np.ndarray,
    cells_face_info: np.ndarray,
    cells_face_values: np.ndarray,
    face_normals_and_length: np.ndarray,
) -> np.ndarray:
    """Average a HEC-RAS face-centred scalar onto parent 2D cells."""
    face_values = np.asarray(face_values, dtype=float).ravel()
    cells_face_info = np.asarray(cells_face_info, dtype=int)
    cells_face_values = np.asarray(cells_face_values, dtype=int)
    face_normals_and_length = np.asarray(face_normals_and_length, dtype=float)

    n_cells = cells_face_info.shape[0]
    cell_values = np.full(n_cells, np.nan, dtype=float)
    lengths = face_normals_and_length[:, 2] if face_normals_and_length.ndim == 2 else np.array([])

    for cell_id in range(n_cells):
        start, count = cells_face_info[cell_id]
        if count <= 0:
            continue

        cell_rows = cells_face_values[start:start + count]
        face_ids = cell_rows[:, 0].astype(int, copy=False)
        valid = (
            (face_ids >= 0)
            & (face_ids < face_values.size)
            & np.isfinite(face_values[face_ids])
        )
        if not np.any(valid):
            continue

        face_ids = face_ids[valid]
        values = face_values[face_ids]
        if lengths.size > int(np.max(face_ids)):
            weights = np.maximum(lengths[face_ids], 1e-12)
            finite = np.isfinite(values) & np.isfinite(weights)
            if np.any(finite):
                cell_values[cell_id] = float(np.average(values[finite], weights=weights[finite]))
        else:
            cell_values[cell_id] = float(np.nanmean(values))

    return cell_values


def _face_water_surface_from_cells(wse_parent: np.ndarray, faces_cell_indexes: np.ndarray) -> np.ndarray:
    """Estimate face water surface from adjacent HEC-RAS cell water surfaces."""
    wse_parent = np.asarray(wse_parent, dtype=float).ravel()
    faces_cell_indexes = np.asarray(faces_cell_indexes, dtype=int)
    wse_face = np.full(faces_cell_indexes.shape[0], np.nan, dtype=float)

    for face_id, cell_ids in enumerate(faces_cell_indexes):
        valid = [
            int(cell_id)
            for cell_id in cell_ids
            if 0 <= int(cell_id) < wse_parent.size and np.isfinite(wse_parent[int(cell_id)])
        ]
        if valid:
            wse_face[face_id] = float(np.mean(wse_parent[valid]))

    return wse_face


def _interpolate_face_hydraulic_properties(
    wse_face: np.ndarray,
    face_area_elevation_info: np.ndarray,
    face_area_elevation_values: np.ndarray,
    *,
    length_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate face hydraulic radius and Manning's n from HEC-RAS face tables."""
    wse_face = np.asarray(wse_face, dtype=float).ravel()
    face_area_elevation_info = np.asarray(face_area_elevation_info, dtype=int)
    face_area_elevation_values = np.asarray(face_area_elevation_values, dtype=float)

    n_faces = face_area_elevation_info.shape[0]
    hydraulic_radius = np.full(n_faces, np.nan, dtype=float)
    manning_n = np.full(n_faces, np.nan, dtype=float)

    for face_id in range(n_faces):
        start, count = face_area_elevation_info[face_id]
        if count <= 0 or not np.isfinite(wse_face[face_id]):
            continue

        rows = face_area_elevation_values[start:start + count]
        if rows.shape[0] == 0 or rows.shape[1] < 4:
            continue

        z = rows[:, 0] * length_scale
        area = rows[:, 1] * (length_scale ** 2)
        wetted_perimeter = rows[:, 2] * length_scale
        n_values = rows[:, 3]

        if wse_face[face_id] < z[0]:
            continue

        area_face = float(np.interp(wse_face[face_id], z, area, left=area[0], right=area[-1]))
        perimeter_face = float(
            np.interp(wse_face[face_id], z, wetted_perimeter, left=wetted_perimeter[0], right=wetted_perimeter[-1])
        )
        n_face = float(np.interp(wse_face[face_id], z, n_values, left=n_values[0], right=n_values[-1]))
        if area_face > 0.0 and perimeter_face > 0.0 and n_face > 0.0:
            hydraulic_radius[face_id] = area_face / perimeter_face
            manning_n[face_id] = n_face

    return hydraulic_radius, manning_n


def _ustar_from_face_manning(
    face_velocity: np.ndarray,
    wse_parent: np.ndarray,
    faces_cell_indexes: np.ndarray,
    face_area_elevation_info: np.ndarray,
    face_area_elevation_values: np.ndarray,
    cells_face_info: np.ndarray,
    cells_face_values: np.ndarray,
    face_normals_and_length: np.ndarray,
    *,
    length_scale: float,
    gravity: float,
) -> np.ndarray:
    """Fallback HEC-RAS u* from tau = gamma R Sf with Manning-derived Sf."""
    wse_face = _face_water_surface_from_cells(wse_parent, faces_cell_indexes)
    hydraulic_radius, manning_n = _interpolate_face_hydraulic_properties(
        wse_face,
        face_area_elevation_info,
        face_area_elevation_values,
        length_scale=length_scale,
    )
    return _ustar_from_face_hydraulic_properties(
        face_velocity=face_velocity,
        hydraulic_radius=hydraulic_radius,
        manning_n=manning_n,
        cells_face_info=cells_face_info,
        cells_face_values=cells_face_values,
        face_normals_and_length=face_normals_and_length,
        gravity=gravity,
    )


def _ustar_from_face_hydraulic_properties(
    face_velocity: np.ndarray,
    hydraulic_radius: np.ndarray,
    manning_n: np.ndarray,
    cells_face_info: np.ndarray,
    cells_face_values: np.ndarray,
    face_normals_and_length: np.ndarray,
    *,
    gravity: float,
) -> np.ndarray:
    """Fallback HEC-RAS cell u* from precomputed face R, n, and velocity."""
    face_velocity = np.asarray(face_velocity, dtype=float).ravel()
    hydraulic_radius = np.asarray(hydraulic_radius, dtype=float).ravel()
    manning_n = np.asarray(manning_n, dtype=float).ravel()
    face_tau_over_rho = np.full_like(hydraulic_radius, np.nan, dtype=float)
    good = (
        np.isfinite(hydraulic_radius)
        & np.isfinite(manning_n)
        & np.isfinite(face_velocity)
        & (hydraulic_radius > 0.0)
        & (manning_n > 0.0)
    )
    if np.any(good):
        friction_slope = (
            (manning_n[good] ** 2)
            * (face_velocity[good] ** 2)
            / np.power(hydraulic_radius[good], 4.0 / 3.0)
        )
        face_tau_over_rho[good] = gravity * hydraulic_radius[good] * friction_slope

    cell_tau_over_rho = _average_face_values_to_cells(
        face_values=face_tau_over_rho,
        cells_face_info=cells_face_info,
        cells_face_values=cells_face_values,
        face_normals_and_length=face_normals_and_length,
    )
    ustar = np.full_like(cell_tau_over_rho, np.nan, dtype=float)
    good_cell = np.isfinite(cell_tau_over_rho) & (cell_tau_over_rho >= 0.0)
    if np.any(good_cell):
        ustar[good_cell] = np.sqrt(cell_tau_over_rho[good_cell])
    return ustar


def _manning_friction_ratio(radius_c: np.ndarray, n_c: np.ndarray, *, gravity: float) -> np.ndarray:
    """Convert Manning's n and hydraulic radius to dimensionless U/u*."""
    radius_c = np.asarray(radius_c, dtype=float)
    n_c = np.asarray(n_c, dtype=float)
    friction_ratio = np.full_like(radius_c, np.nan, dtype=float)
    good = (
        np.isfinite(radius_c)
        & np.isfinite(n_c)
        & (radius_c > 0.0)
        & (n_c > 0.0)
        & np.isfinite(gravity)
        & (gravity > 0.0)
    )
    if np.any(good):
        friction_ratio[good] = np.power(radius_c[good], 1.0 / 6.0) / (n_c[good] * np.sqrt(gravity))
    return friction_ratio


@dataclass
class HecRasPaths:
    area_name: str
    geometry_attrs_path: str
    cells_center_path: str
    cells_manning_path: str
    cells_min_elevation_path: str
    cell_facepoint_indexes_path: str
    cells_face_info_path: str
    cells_face_values_path: str
    face_area_elevation_info_path: str | None
    face_area_elevation_values_path: str | None
    faces_cell_indexes_path: str | None
    facepoints_coord_path: str
    face_normals_path: str
    water_surface_path: str
    face_velocity_path: str
    face_shear_stress_path: str | None
    time_path: str


class HecRasAdapter:
    """Hydraulic adapter for HEC-RAS 2D unsteady plan HDF files."""

    adapter_name = "HEC-RAS"

    def __init__(
        self,
        plan_hdf_path: str | Path,
        *,
        area_name: str | None = None,
        convert_to_si: bool = True,
        hmin_fallback: float = 1e-2,
        rho_f: float = 1000.0,
    ) -> None:
        self.plan_hdf_path = Path(plan_hdf_path)
        if not self.plan_hdf_path.exists():
            raise FileNotFoundError(f"HEC-RAS plan HDF not found: {self.plan_hdf_path}")

        self.area_name = area_name
        self.convert_to_si = bool(convert_to_si)
        self.hmin_fallback = float(hmin_fallback)
        self.rho_f = float(rho_f)
        if self.rho_f <= 0.0:
            raise ValueError("rho_f must be positive for HEC-RAS shear-stress conversion.")

        self.meta: BasementMeta | None = None
        self.paths: HecRasPaths | None = None
        self.parent_cell_index: np.ndarray | None = None
        self._native_facepoints_xy: np.ndarray | None = None
        self._native_cell_facepoint_indexes: np.ndarray | None = None
        self._native_cell_centers_xy: np.ndarray | None = None
        self._native_faces_cell_indexes: np.ndarray | None = None
        self._native_faces_facepoint_indexes: np.ndarray | None = None
        self._native_facepoints_is_perimeter: np.ndarray | None = None
        self._time_values_s: np.ndarray | None = None
        self._si_units: bool | None = None
        self._native_length_unit: str | None = None
        self._frame_cache: dict[int, BasementFrame] = {}

    def _build_paths(self, h5file: h5py.File) -> HecRasPaths:
        area_name = self.area_name or _pick_first_2d_flow_area(h5file)
        area_prefix = f"Geometry/2D Flow Areas/{area_name}"
        results_prefix = (
            "Results/Unsteady/Output/Output Blocks/Base Output/"
            f"Unsteady Time Series/2D Flow Areas/{area_name}"
        )
        face_shear_stress_path = _optional_dataset_path(
            h5file,
            [
                f"{results_prefix}/Face Shear Stress",
                f"{results_prefix}/Face Shear",
                f"{results_prefix}/Shear Stress",
                f"{results_prefix}/Tau_0",
                f"{results_prefix}/Tau0",
            ],
        )
        face_area_elevation_info_path = _optional_dataset_path(
            h5file,
            [f"{area_prefix}/Faces Area Elevation Info"],
        )
        face_area_elevation_values_path = _optional_dataset_path(
            h5file,
            [f"{area_prefix}/Faces Area Elevation Values"],
        )
        faces_cell_indexes_path = _optional_dataset_path(
            h5file,
            [f"{area_prefix}/Faces Cell Indexes"],
        )
        return HecRasPaths(
            area_name=area_name,
            geometry_attrs_path="Geometry",
            cells_center_path=f"{area_prefix}/Cells Center Coordinate",
            cells_manning_path=f"{area_prefix}/Cells Center Manning's n",
            cells_min_elevation_path=f"{area_prefix}/Cells Minimum Elevation",
            cell_facepoint_indexes_path=f"{area_prefix}/Cells FacePoint Indexes",
            cells_face_info_path=f"{area_prefix}/Cells Face and Orientation Info",
            cells_face_values_path=f"{area_prefix}/Cells Face and Orientation Values",
            face_area_elevation_info_path=face_area_elevation_info_path,
            face_area_elevation_values_path=face_area_elevation_values_path,
            faces_cell_indexes_path=faces_cell_indexes_path,
            facepoints_coord_path=f"{area_prefix}/FacePoints Coordinate",
            face_normals_path=f"{area_prefix}/Faces NormalUnitVector and Length",
            water_surface_path=f"{results_prefix}/Water Surface",
            face_velocity_path=f"{results_prefix}/Face Velocity",
            face_shear_stress_path=face_shear_stress_path,
            time_path="Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time",
        )

    def _infer_units(self, h5file: h5py.File) -> tuple[bool, str]:
        geometry_attrs = dict(h5file["Geometry"].attrs)
        si_units = _to_bool_attr(geometry_attrs.get("SI Units", b"False"))
        native_length_unit = "m" if si_units else "ft"

        if self.paths is not None and self.paths.water_surface_path in h5file:
            units_attr = h5file[self.paths.water_surface_path].attrs.get("Units")
            if units_attr is not None:
                native_length_unit = _decode_attr(units_attr)

        return si_units, native_length_unit

    def _length_scale(self) -> float:
        if not self.convert_to_si:
            return 1.0
        if self._si_units:
            return 1.0
        return FT_TO_M

    def _velocity_scale(self) -> float:
        if not self.convert_to_si:
            return 1.0
        if self._si_units:
            return 1.0
        return FT_TO_M

    def _gravity_acceleration(self) -> float:
        if self.convert_to_si or self._si_units:
            return GRAVITY_SI
        return GRAVITY_US

    def _shear_stress_scale_to_si(self, h5file: h5py.File) -> float | None:
        if self.paths is None or self.paths.face_shear_stress_path is None:
            return None
        if not self.convert_to_si and not self._si_units:
            return None

        units_attr = h5file[self.paths.face_shear_stress_path].attrs.get("Units")
        units = _decode_attr(units_attr).lower() if units_attr is not None else ""
        if any(token in units for token in ("lb/ft", "psf")):
            return PSF_TO_PA
        return 1.0

    def _read_time_values_s(self, h5file: h5py.File) -> np.ndarray:
        time_days = np.asarray(_safe_dataset(h5file, self.paths.time_path), dtype=float).ravel()
        if time_days.size == 0:
            raise ValueError("HEC-RAS time array is empty.")
        time_s = (time_days - time_days[0]) * SECONDS_PER_DAY
        return time_s

    def validate_outputs(self, *, verbose: bool = True, raise_on_error: bool = False) -> dict[str, Any]:
        """Validate the required HEC-RAS datasets and return a report."""
        report: dict[str, Any] = {
            "ok": True,
            "errors": [],
            "warnings": [],
            "files": {},
            "paths": {},
            "variables": {},
        }

        def add_error(message: str) -> None:
            report["ok"] = False
            report["errors"].append(message)
            if verbose:
                print("ERROR:", message)

        def add_warning(message: str) -> None:
            report["warnings"].append(message)
            if verbose:
                print("WARNING:", message)

        def add_info(message: str) -> None:
            if verbose:
                print(message)

        report["files"]["plan_hdf"] = {
            "exists": self.plan_hdf_path.exists(),
            "size_bytes": self.plan_hdf_path.stat().st_size if self.plan_hdf_path.exists() else None,
        }
        if not self.plan_hdf_path.exists():
            add_error(f"HEC-RAS plan HDF not found: {self.plan_hdf_path}")
            if raise_on_error:
                raise RuntimeError("\n".join(report["errors"]))
            return report

        try:
            with h5py.File(self.plan_hdf_path, "r") as h5file:
                self.paths = self._build_paths(h5file)
                self._si_units, self._native_length_unit = self._infer_units(h5file)
                report["paths"]["area_name"] = self.paths.area_name
                report["paths"]["water_surface_path"] = self.paths.water_surface_path
                report["paths"]["face_velocity_path"] = self.paths.face_velocity_path
                report["paths"]["face_shear_stress_path"] = self.paths.face_shear_stress_path
                report["paths"]["face_area_elevation_info_path"] = self.paths.face_area_elevation_info_path
                report["paths"]["face_area_elevation_values_path"] = self.paths.face_area_elevation_values_path
                report["paths"]["faces_cell_indexes_path"] = self.paths.faces_cell_indexes_path

                required_paths = {
                    "cells_center": self.paths.cells_center_path,
                    "cells_manning": self.paths.cells_manning_path,
                    "cells_min_elevation": self.paths.cells_min_elevation_path,
                    "cell_facepoint_indexes": self.paths.cell_facepoint_indexes_path,
                    "cells_face_info": self.paths.cells_face_info_path,
                    "cells_face_values": self.paths.cells_face_values_path,
                    "facepoints_coord": self.paths.facepoints_coord_path,
                    "face_normals": self.paths.face_normals_path,
                    "water_surface": self.paths.water_surface_path,
                    "face_velocity": self.paths.face_velocity_path,
                    "time": self.paths.time_path,
                }

                for name, path in required_paths.items():
                    if path not in h5file:
                        add_error(f"Missing HEC-RAS dataset/group: {path}")
                        continue
                    obj = h5file[path]
                    report["variables"][name] = {
                        "exists": True,
                        "shape": getattr(obj, "shape", None),
                    }

                if self.paths.face_shear_stress_path is not None:
                    obj = h5file[self.paths.face_shear_stress_path]
                    report["variables"]["face_shear_stress"] = {
                        "exists": True,
                        "shape": getattr(obj, "shape", None),
                        "units": _decode_attr(obj.attrs.get("Units", "")),
                    }

                optional_paths = {
                    "face_area_elevation_info": self.paths.face_area_elevation_info_path,
                    "face_area_elevation_values": self.paths.face_area_elevation_values_path,
                    "faces_cell_indexes": self.paths.faces_cell_indexes_path,
                }
                for name, path in optional_paths.items():
                    if path is None:
                        continue
                    obj = h5file[path]
                    report["variables"][name] = {
                        "exists": True,
                        "shape": getattr(obj, "shape", None),
                    }

                if report["errors"]:
                    if raise_on_error:
                        raise RuntimeError("\n".join(report["errors"]))
                    return report

                cell_centers = np.asarray(h5file[self.paths.cells_center_path][()], dtype=float)
                wse = np.asarray(h5file[self.paths.water_surface_path][()], dtype=float)
                face_v = np.asarray(h5file[self.paths.face_velocity_path][()], dtype=float)
                time_values_s = self._read_time_values_s(h5file)

                if cell_centers.ndim != 2 or cell_centers.shape[1] != 2:
                    add_error(f"HEC-RAS cell centres must be (Nc,2), got {cell_centers.shape}")
                if wse.ndim != 2:
                    add_error(f"HEC-RAS water-surface array must be 2D, got {wse.shape}")
                if face_v.ndim != 2:
                    add_error(f"HEC-RAS face-velocity array must be 2D, got {face_v.shape}")
                if wse.shape[0] != time_values_s.size:
                    add_warning(
                        f"Water-surface time count ({wse.shape[0]}) differs from time array length ({time_values_s.size})."
                    )
                add_info(f"HEC-RAS area: {self.paths.area_name}")
                add_info(f"Geometry unit system: {'SI' if self._si_units else 'US customary'}")
                add_info(f"2D cells: {cell_centers.shape[0]}")
                add_info(f"Time frames: {time_values_s.size}")
                if self.paths.face_shear_stress_path is not None:
                    add_info(f"Face shear stress: {self.paths.face_shear_stress_path}")

        except Exception as exc:
            add_error(f"HEC-RAS HDF validation failed: {exc}")

        if raise_on_error and not report["ok"]:
            raise RuntimeError("\n".join(report["errors"]))

        return report

    def load_meta(self) -> BasementMeta:
        """Load native HEC-RAS polygon mesh metadata."""
        if self.meta is not None:
            return self.meta

        with h5py.File(self.plan_hdf_path, "r") as h5file:
            self.paths = self._build_paths(h5file)
            self._si_units, self._native_length_unit = self._infer_units(h5file)
            assert self.paths is not None
            cell_centers_xy = np.asarray(_safe_dataset(h5file, self.paths.cells_center_path), dtype=float)
            facepoints_xy = np.asarray(_safe_dataset(h5file, self.paths.facepoints_coord_path), dtype=float)
            cell_facepoint_indexes = np.asarray(
                _safe_dataset(h5file, self.paths.cell_facepoint_indexes_path),
                dtype=int,
            )
            zb_parent = np.asarray(_safe_dataset(h5file, self.paths.cells_min_elevation_path), dtype=float).ravel()
            faces_cell_indexes = (
                None
                if self.paths.faces_cell_indexes_path is None
                else np.asarray(_safe_dataset(h5file, self.paths.faces_cell_indexes_path), dtype=int)
            )
            faces_facepoint_indexes = np.asarray(
                _safe_dataset(h5file, f"Geometry/2D Flow Areas/{self.paths.area_name}/Faces FacePoint Indexes"),
                dtype=int,
            )
            facepoints_is_perimeter = np.asarray(
                _safe_dataset(h5file, f"Geometry/2D Flow Areas/{self.paths.area_name}/FacePoints Is Perimeter"),
                dtype=bool,
            ).ravel()
            time_values_s = self._read_time_values_s(h5file)

        scale = self._length_scale()
        cell_centers_xy = cell_centers_xy * scale
        facepoints_xy = facepoints_xy * scale
        zb_parent = zb_parent * scale

        self.parent_cell_index = np.arange(cell_centers_xy.shape[0], dtype=int)
        self._native_facepoints_xy = facepoints_xy
        self._native_cell_facepoint_indexes = cell_facepoint_indexes
        self._native_cell_centers_xy = cell_centers_xy
        self._native_faces_cell_indexes = faces_cell_indexes
        self._native_faces_facepoint_indexes = faces_facepoint_indexes
        self._native_facepoints_is_perimeter = facepoints_is_perimeter
        self._time_values_s = time_values_s

        hmin = self.hmin_fallback
        steps = [str(i) for i in range(time_values_s.size)]

        self.meta = BasementMeta(
            XY=facepoints_xy,
            tri=np.empty((0, 3), dtype=int),
            zb_c=zb_parent,
            hmin=hmin,
            steps_hyd=steps,
            steps_vel=steps,
            Nc=int(cell_centers_xy.shape[0]),
            Nn=int(facepoints_xy.shape[0]),
        )
        return self.meta

    def build_mesh(self) -> HecRasNativeMesh:
        """Return a native polygon mesh helper for the HEC-RAS area."""
        if self.meta is None:
            self.load_meta()
        assert self._native_facepoints_xy is not None
        assert self._native_cell_facepoint_indexes is not None
        assert self._native_cell_centers_xy is not None
        return HecRasNativeMesh(
            facepoints_xy=self._native_facepoints_xy,
            cell_facepoint_indexes=self._native_cell_facepoint_indexes,
            cell_centers_xy=self._native_cell_centers_xy,
            faces_cell_indexes=self._native_faces_cell_indexes,
            faces_facepoint_indexes=self._native_faces_facepoint_indexes,
            facepoints_is_perimeter=self._native_facepoints_is_perimeter,
        )

    def last_index(self) -> int:
        if self.meta is None:
            self.load_meta()
        assert self.meta is not None
        return len(self.meta.steps_hyd) - 1

    def native_time_values_s(self) -> np.ndarray:
        """Return the HEC-RAS output times in seconds relative to the first frame."""
        if self._time_values_s is None:
            self.load_meta()
        assert self._time_values_s is not None
        return self._time_values_s.copy()

    def load_frame(self, k: int | None = None) -> BasementFrame:
        """Load one HEC-RAS output frame on the native parent-cell mesh."""
        if self.meta is None or self.parent_cell_index is None:
            self.load_meta()

        assert self.meta is not None
        assert self.parent_cell_index is not None
        assert self.paths is not None

        if k is None:
            k = self.last_index()
        k = int(np.clip(k, 0, self.last_index()))
        cached = self._frame_cache.get(k)
        if cached is not None:
            return cached

        with h5py.File(self.plan_hdf_path, "r") as h5file:
            wse_parent = np.asarray(_safe_dataset(h5file, self.paths.water_surface_path), dtype=float)[k].ravel()
            face_velocity = np.asarray(_safe_dataset(h5file, self.paths.face_velocity_path), dtype=float)[k].ravel()
            manning_parent = np.asarray(_safe_dataset(h5file, self.paths.cells_manning_path), dtype=float).ravel()
            zb_parent = np.asarray(_safe_dataset(h5file, self.paths.cells_min_elevation_path), dtype=float).ravel()
            face_normals = np.asarray(_safe_dataset(h5file, self.paths.face_normals_path), dtype=float)
            cells_face_info = np.asarray(_safe_dataset(h5file, self.paths.cells_face_info_path), dtype=int)
            cells_face_values = np.asarray(_safe_dataset(h5file, self.paths.cells_face_values_path), dtype=int)
            face_shear_stress = None
            shear_scale = self._shear_stress_scale_to_si(h5file)
            if self.paths.face_shear_stress_path is not None and shear_scale is not None:
                face_shear_stress = np.asarray(
                    _safe_dataset(h5file, self.paths.face_shear_stress_path),
                    dtype=float,
                )[k].ravel()
            face_area_elevation_info = None
            face_area_elevation_values = None
            faces_cell_indexes = None
            if (
                self.paths.face_area_elevation_info_path is not None
                and self.paths.face_area_elevation_values_path is not None
                and self.paths.faces_cell_indexes_path is not None
            ):
                face_area_elevation_info = np.asarray(
                    _safe_dataset(h5file, self.paths.face_area_elevation_info_path),
                    dtype=int,
                )
                face_area_elevation_values = np.asarray(
                    _safe_dataset(h5file, self.paths.face_area_elevation_values_path),
                    dtype=float,
                )
                faces_cell_indexes = np.asarray(
                    _safe_dataset(h5file, self.paths.faces_cell_indexes_path),
                    dtype=int,
                )

        length_scale = self._length_scale()
        velocity_scale = self._velocity_scale()

        wse_parent = wse_parent * length_scale
        zb_parent = zb_parent * length_scale
        h_parent = np.maximum(wse_parent - zb_parent, 0.0)
        face_velocity = face_velocity * velocity_scale
        face_normals = face_normals.copy()
        face_normals[:, 2] = face_normals[:, 2] * length_scale
        face_hydraulic_radius = None
        face_manning_n = None
        if (
            face_area_elevation_info is not None
            and face_area_elevation_values is not None
            and faces_cell_indexes is not None
        ):
            wse_face = _face_water_surface_from_cells(wse_parent, faces_cell_indexes)
            face_hydraulic_radius, face_manning_n = _interpolate_face_hydraulic_properties(
                wse_face,
                face_area_elevation_info,
                face_area_elevation_values,
                length_scale=length_scale,
            )

        U_parent, V_parent = _fit_cell_velocity_vectors(
            face_velocity=face_velocity,
            face_normals_and_length=face_normals,
            cells_face_info=cells_face_info,
            cells_face_values=cells_face_values,
        )
        # HEC-RAS face hydraulic radii describe face conveyance sections, not
        # a representative cell bed roughness radius. For 2D cells we use the
        # standard wide-channel approximation R ~= H.
        radius_parent = h_parent
        chezy_parent = _manning_friction_ratio(
            radius_parent,
            manning_parent,
            gravity=self._gravity_acceleration(),
        )
        ustar_parent = np.full_like(h_parent, np.nan, dtype=float)
        if face_shear_stress is not None and shear_scale is not None:
            tau_parent = _average_face_values_to_cells(
                face_values=np.abs(face_shear_stress) * shear_scale,
                cells_face_info=cells_face_info,
                cells_face_values=cells_face_values,
                face_normals_and_length=face_normals,
            )
            good_tau = np.isfinite(tau_parent)
            if np.any(good_tau):
                ustar_parent[good_tau] = np.sqrt(np.maximum(tau_parent[good_tau], 0.0) / self.rho_f)

        missing_ustar = (~np.isfinite(ustar_parent)) & (face_shear_stress is None or shear_scale is None)
        if (
            np.any(missing_ustar)
            and face_hydraulic_radius is not None
            and face_manning_n is not None
        ):
            face_manning_ustar = _ustar_from_face_hydraulic_properties(
                face_velocity=face_velocity,
                hydraulic_radius=face_hydraulic_radius,
                manning_n=face_manning_n,
                cells_face_info=cells_face_info,
                cells_face_values=cells_face_values,
                face_normals_and_length=face_normals,
                gravity=self._gravity_acceleration(),
            )
            good_face = missing_ustar & np.isfinite(face_manning_ustar)
            if np.any(good_face):
                ustar_parent[good_face] = face_manning_ustar[good_face]

        wet_c = np.isfinite(h_parent) & (h_parent > self.meta.hmin)

        frame = BasementFrame(
            step=str(k),
            step_vel=str(k),
            wse_c=wse_parent,
            h_c=h_parent,
            U_c=U_parent,
            V_c=V_parent,
            Chezy_c=chezy_parent,
            wet_c=wet_c,
            ks_c=None,
            ustar_c=ustar_parent,
        )
        frame.face_hydraulic_radius = face_hydraulic_radius
        frame.face_manning_n = face_manning_n
        self._frame_cache[k] = frame
        return frame

    def close(self) -> None:
        """Mirror the adapter interface used by other backends."""
        return


def run_hecras_case(
    run_file: str | Path,
    *,
    settings: dict,
    plan_hdf_path: str | Path,
    plots: dict | None = None,
    area_name: str | None = None,
    convert_to_si: bool = True,
    show_plots: bool = True,
    block_on_plots: bool = True,
) -> np.ndarray:
    """Run a HydroLPT case against a HEC-RAS 2D plan HDF file."""
    run_file = Path(run_file).resolve()
    plan_hdf_path = Path(plan_hdf_path).resolve()
    settings = dict(settings)
    if str(settings.get("transportVelocityMode", "depth_averaged")).strip().lower() != "depth_averaged":
        raise ValueError("HEC-RAS cases support only transportVelocityMode='depth_averaged'.")

    adapter_preview = HecRasAdapter(
        plan_hdf_path,
        area_name=area_name,
        convert_to_si=convert_to_si,
        hmin_fallback=float(settings.get("hmin", 1e-2)),
        rho_f=float(settings.get("rho_f", 1000.0)),
    )
    file_times_s = adapter_preview.native_time_values_s()

    if str(settings.get("run_mode", "steady")).strip().lower() == "transient":
        if file_times_s.size < 2:
            raise RuntimeError("Transient HEC-RAS run requires at least two output frames.")
        dt_h = float(np.median(np.diff(file_times_s)))
        if dt_h <= 0.0:
            raise RuntimeError("Could not infer a positive HEC-RAS hydraulic time step.")

        existing = {
            "hyd_time_start": settings.get("hyd_time_start"),
            "hyd_dt": settings.get("hyd_dt"),
            "hyd_time_end": settings.get("hyd_time_end"),
        }
        inferred = {
            "hyd_time_start": float(file_times_s[0]),
            "hyd_dt": float(dt_h),
            "hyd_time_end": float(file_times_s[-1]),
        }
        for key, value in inferred.items():
            raw = existing.get(key)
            if raw is None:
                settings[key] = value
                continue
            try:
                current = float(raw)
            except (TypeError, ValueError):
                settings[key] = value
                continue
            if not np.isclose(current, value, rtol=0.0, atol=1e-9):
                print(
                    f"Warning: overriding {key}={current:g} with HEC-RAS file timing {value:g}."
                )
                settings[key] = value

    startup_lines = [
        f"Using HEC-RAS plan HDF: {plan_hdf_path}",
        f"HEC-RAS 2D area: {adapter_preview.paths.area_name}",
        (
            "Units converted to SI"
            if convert_to_si and not adapter_preview._si_units
            else "Using native SI units from HEC-RAS"
            if adapter_preview._si_units
            else "Using native HEC-RAS units without conversion"
        ),
        "Using only native 2D flow area",
    ]
    if adapter_preview.paths is not None and adapter_preview.paths.face_shear_stress_path is not None:
        startup_lines.append("Using HEC-RAS Face Shear Stress for u* where available")
    else:
        startup_lines.append("Using Manning-based HEC-RAS shear-stress fallback for u*")

    return run_hydraulic_case(
        run_file,
        settings=settings,
        plots=plots,
        adapter_factory=lambda: HecRasAdapter(
            plan_hdf_path,
            area_name=area_name,
            convert_to_si=convert_to_si,
            hmin_fallback=float(settings.get("hmin", 1e-2)),
            rho_f=float(settings.get("rho_f", 1000.0)),
        ),
        startup_lines=startup_lines,
        classification_tol_default=5.0,
        show_plots=show_plots,
        block_on_plots=block_on_plots,
    )
