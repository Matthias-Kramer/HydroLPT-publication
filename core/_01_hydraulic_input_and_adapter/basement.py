"""Adapter for BASEMENT XDMF/HDF5 hydraulic results.

This module converts externally produced hydraulic outputs into the common
HydroLPT frame/meta representation consumed by the particle solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from core._00_case_management.config import get_default_paths
from core._04_transport_solver.runner import run_hydraulic_case


def list_datasets(h5file, group_path: str) -> list[str]:
    """Return dataset names below an HDF5 group path."""
    if group_path not in h5file:
        raise KeyError(f"Group not found in HDF5: {group_path}")
    return list(h5file[group_path].keys())


def _norm_tri(tri: np.ndarray) -> np.ndarray:
    """Normalize triangle connectivity to zero-based ``(Nc, 3)`` layout."""
    tri = np.asarray(tri, dtype=int)

    if tri.ndim == 1:
        tri = tri.reshape(3, -1).T
    elif tri.ndim == 2 and tri.shape[0] == 3 and tri.shape[1] != 3:
        tri = tri.T

    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError(f"Topology expected (Nc,3), got {tri.shape}")

    if tri.min() == 1:
        tri = tri - 1

    return tri


def _strip_step_zeros(step: str) -> str:
    """Normalize timestep labels by removing leading zeros."""
    return str(int(step))


def _sort_steps_numeric(steps: list[str]) -> list[str]:
    """Sort string-encoded timesteps numerically instead of lexically."""
    return sorted(steps, key=lambda s: int(s))


@dataclass
class BasementMeta:
    XY: np.ndarray
    tri: np.ndarray
    zb_c: np.ndarray
    hmin: float
    steps_hyd: list[str]
    steps_vel: list[str]
    Nc: int
    Nn: int


@dataclass
class BasementFrame:
    step: str
    step_vel: str
    wse_c: np.ndarray
    h_c: np.ndarray
    U_c: np.ndarray
    V_c: np.ndarray
    Chezy_c: np.ndarray | None
    wet_c: np.ndarray

    # Optional helper fields for transport
    ks_c: np.ndarray | None = None
    ustar_c: np.ndarray | None = None


@dataclass
class XdmfPaths:
    water_surface_file: str
    water_surface_root: str
    water_surface_col: int

    velocity_file: str
    velocity_root: str

    chezy_file: str | None
    chezy_root: str | None

    bottom_file: str
    bottom_path: str


class BasementAdapter:
    """Hydraulic adapter that exposes BASEMENT results through a common API."""

    adapter_name = "BASEHPC"

    def __init__(self, results_xdmf: str | Path, *, hmin_fallback: float = 1e-2):
        self.results_xdmf = Path(results_xdmf)
        if not self.results_xdmf.exists():
            raise FileNotFoundError(f"XDMF file not found: {self.results_xdmf}")

        self.hmin_fallback = float(hmin_fallback)

        self.meta: BasementMeta | None = None
        self.paths: XdmfPaths | None = None

        self._f_hyd: h5py.File | None = None
        self._f_vel: h5py.File | None = None
        self._f_chezy: h5py.File | None = None
        self._f_bottom: h5py.File | None = None

    @staticmethod
    def _local(tag: str) -> str:
        return tag.split("}", 1)[-1]

    @staticmethod
    def _split_hdf_ref(text: str) -> tuple[str, str]:
        text = text.strip()
        if ":" not in text:
            raise ValueError(f"Not an HDF reference: {text}")
        fname, path = text.split(":", 1)
        if not path.startswith("/"):
            path = "/" + path
        return fname.strip(), path.strip()

    @staticmethod
    def _parent_path(path: str) -> str:
        path = path.rstrip("/")
        if "/" not in path:
            return path
        return path[: path.rfind("/")]

    @staticmethod
    def _first_dataitem_with_hdf_text(elem: ET.Element) -> str | None:
        for child in elem.iter():
            if BasementAdapter._local(child.tag) != "DataItem":
                continue
            txt = (child.text or "").strip()
            if ".h5:" in txt:
                return txt
        return None

    @staticmethod
    def _find_attribute(root: ET.Element, name: str) -> ET.Element:
        for elem in root.iter():
            if BasementAdapter._local(elem.tag) == "Attribute" and elem.attrib.get("Name") == name:
                return elem
        raise KeyError(f'Attribute Name="{name}" not found in XDMF')

    @staticmethod
    def _extract_hyperslab_column(attribute_elem: ET.Element) -> int:
        for elem in attribute_elem.iter():
            if BasementAdapter._local(elem.tag) != "DataItem":
                continue
            if elem.attrib.get("ItemType") != "HyperSlab":
                continue

            children = [c for c in list(elem) if BasementAdapter._local(c.tag) == "DataItem"]
            if len(children) < 2:
                continue

            slab_txt = " ".join((children[0].text or "").split())
            nums = [int(float(v)) for v in slab_txt.split()]
            if len(nums) >= 2:
                return nums[1]
        return 0

    def _parse_xdmf_paths(self) -> XdmfPaths:
        root = ET.parse(self.results_xdmf).getroot()

        attr_wse = self._find_attribute(root, "water_surface")
        attr_vel = self._find_attribute(root, "flow_velocity")
        attr_zb = self._find_attribute(root, "bottom_elevation")

        try:
            attr_chezy = self._find_attribute(root, "friction_chezy")
        except KeyError:
            attr_chezy = None

        wse_ref = self._first_dataitem_with_hdf_text(attr_wse)
        if wse_ref is None:
            raise ValueError('Could not resolve HDF5 reference for "water_surface"')
        wse_file, wse_path = self._split_hdf_ref(wse_ref)

        vel_ref = self._first_dataitem_with_hdf_text(attr_vel)
        if vel_ref is None:
            raise ValueError('Could not resolve HDF5 reference for "flow_velocity"')
        vel_file, vel_path = self._split_hdf_ref(vel_ref)

        zb_ref = self._first_dataitem_with_hdf_text(attr_zb)
        if zb_ref is None:
            raise ValueError('Could not resolve HDF5 reference for "bottom_elevation"')
        zb_file, zb_path = self._split_hdf_ref(zb_ref)

        chezy_file = None
        chezy_root = None
        if attr_chezy is not None:
            chezy_ref = self._first_dataitem_with_hdf_text(attr_chezy)
            if chezy_ref is not None:
                chezy_file, chezy_path = self._split_hdf_ref(chezy_ref)
                chezy_root = self._parent_path(chezy_path)

        return XdmfPaths(
            water_surface_file=wse_file,
            water_surface_root=self._parent_path(wse_path),
            water_surface_col=self._extract_hyperslab_column(attr_wse),
            velocity_file=vel_file,
            velocity_root=self._parent_path(vel_path),
            chezy_file=chezy_file,
            chezy_root=chezy_root,
            bottom_file=zb_file,
            bottom_path=zb_path,
        )

    def _resolve_h5_path(self, filename_from_xdmf: str) -> Path:
        p = Path(filename_from_xdmf)
        if not p.is_absolute():
            p = self.results_xdmf.parent / p
        return p

    def _check_path_exists(self, h5file: h5py.File, path: str) -> bool:
        return path in h5file

    def _find_group_with_numeric_children(self, h5file: h5py.File, preferred: str | None = None) -> str:
        if preferred is not None and preferred in h5file:
            return preferred

        candidates: list[tuple[str, int, int]] = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Group):
                keys = list(obj.keys())
                if not keys:
                    return
                n_numeric = sum(k.isdigit() for k in keys)
                if n_numeric > 0:
                    candidates.append((name, n_numeric, len(keys)))

        h5file.visititems(visitor)

        if not candidates:
            raise KeyError("Could not find any group with numeric timestep children in HDF5 file.")

        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return "/" + candidates[0][0].lstrip("/")

    def validate_outputs(self, *, verbose: bool = True, raise_on_error: bool = False) -> dict:
        """
        Validate all expected variables and return a structured report.
        """
        report: dict[str, object] = {
            "ok": True,
            "errors": [],
            "warnings": [],
            "files": {},
            "paths": {},
            "variables": {},
        }

        def add_error(msg: str):
            report["ok"] = False
            report["errors"].append(msg)
            if verbose:
                print("ERROR:", msg)

        def add_warning(msg: str):
            report["warnings"].append(msg)
            if verbose:
                print("WARNING:", msg)

        def add_info(msg: str):
            if verbose:
                print(msg)

        try:
            self.paths = self._parse_xdmf_paths()
        except Exception as e:
            add_error(f"Failed to parse XDMF: {e}")
            if raise_on_error:
                raise RuntimeError("\n".join(report["errors"]))
            return report

        assert self.paths is not None

        report["paths"] = {
            "water_surface_root": self.paths.water_surface_root,
            "water_surface_col": self.paths.water_surface_col,
            "velocity_root": self.paths.velocity_root,
            "chezy_root": self.paths.chezy_root,
            "bottom_path": self.paths.bottom_path,
        }

        hyd_path = self._resolve_h5_path(self.paths.water_surface_file)
        vel_path = self._resolve_h5_path(self.paths.velocity_file)
        zb_path = self._resolve_h5_path(self.paths.bottom_file)
        chezy_path = self._resolve_h5_path(self.paths.chezy_file) if self.paths.chezy_file else None

        file_paths = {
            "hydraulic_h5": hyd_path,
            "velocity_h5": vel_path,
            "bottom_h5": zb_path,
            "chezy_h5": chezy_path,
        }

        for name, p in file_paths.items():
            if p is None:
                report["files"][name] = {"exists": False, "size_bytes": None}
                continue
            exists = p.exists()
            size = p.stat().st_size if exists else None
            report["files"][name] = {"exists": exists, "size_bytes": size}
            if not exists:
                add_error(f"{name} not found: {p}")
            elif size is not None and size < 2048:
                add_warning(f"{name} is very small ({size} bytes): {p}")

        if report["errors"]:
            if raise_on_error:
                raise RuntimeError("\n".join(report["errors"]))
            return report

        try:
            with h5py.File(hyd_path, "r") as f_hyd, \
                 h5py.File(vel_path, "r") as f_vel, \
                 h5py.File(zb_path, "r") as f_zb:

                f_chezy = h5py.File(chezy_path, "r") if chezy_path is not None and chezy_path.exists() else None
                try:
                    mesh_ok = True
                    if "/NodesAll/Coordnts" not in f_zb:
                        add_error("Mesh node coordinates missing: /NodesAll/Coordnts")
                        mesh_ok = False
                    if "/CellsAll/Topology" not in f_zb:
                        add_error("Mesh topology missing: /CellsAll/Topology")
                        mesh_ok = False

                    if mesh_ok:
                        XY = np.asarray(f_zb["/NodesAll/Coordnts"][()], dtype=float)
                        tri = _norm_tri(f_zb["/CellsAll/Topology"][()])
                        report["variables"]["mesh"] = {
                            "exists": True,
                            "XY_shape": tuple(XY.shape),
                            "tri_shape": tuple(tri.shape),
                        }
                        add_info(f"Mesh OK: XY {XY.shape}, tri {tri.shape}")

                    if self.paths.bottom_path not in f_zb:
                        add_error(f'Bottom elevation path missing: {self.paths.bottom_path}')
                    else:
                        zb = np.asarray(f_zb[self.paths.bottom_path][()], dtype=float).ravel()
                        report["variables"]["bottom_elevation"] = {
                            "exists": True,
                            "path": self.paths.bottom_path,
                            "shape": tuple(zb.shape),
                        }
                        add_info(f"bottom_elevation OK: {self.paths.bottom_path} shape {zb.shape}")

                    hyd_root = self.paths.water_surface_root
                    if hyd_root not in f_hyd:
                        add_warning(f'water_surface root from XDMF not found: {hyd_root}')
                        try:
                            hyd_root = self._find_group_with_numeric_children(f_hyd, None)
                            add_info(f"Recovered hydraulic root: {hyd_root}")
                        except Exception as e:
                            add_error(f"Could not recover hydraulic root: {e}")

                    if hyd_root in f_hyd:
                        hyd_steps = _sort_steps_numeric(list_datasets(f_hyd, hyd_root))
                        if not hyd_steps:
                            add_error(f"No hydraulic steps under {hyd_root}")
                        else:
                            s0 = hyd_steps[0]
                            Hraw = np.asarray(f_hyd[f"{hyd_root}/{s0}"][()], dtype=float)
                            report["variables"]["water_surface"] = {
                                "exists": True,
                                "root": hyd_root,
                                "column": self.paths.water_surface_col,
                                "n_steps": len(hyd_steps),
                                "first_step": s0,
                                "sample_shape": tuple(Hraw.shape),
                            }
                            add_info(
                                f"water_surface OK: root={hyd_root}, first_step={s0}, "
                                f"shape={Hraw.shape}, col={self.paths.water_surface_col}"
                            )

                    if self.paths.chezy_root is None or f_chezy is None:
                        add_warning("friction_chezy not referenced or file not available")
                        report["variables"]["friction_chezy"] = {"exists": False}
                    else:
                        chezy_root = self.paths.chezy_root
                        if chezy_root not in f_chezy:
                            add_error(f"friction_chezy root missing: {chezy_root}")
                            report["variables"]["friction_chezy"] = {
                                "exists": False,
                                "root": chezy_root,
                            }
                        else:
                            chezy_steps = _sort_steps_numeric(list_datasets(f_chezy, chezy_root))
                            if not chezy_steps:
                                add_error(f"No Chezy steps under {chezy_root}")
                            else:
                                s0 = chezy_steps[0]
                                Carr = np.asarray(f_chezy[f"{chezy_root}/{s0}"][()], dtype=float).ravel()
                                report["variables"]["friction_chezy"] = {
                                    "exists": True,
                                    "root": chezy_root,
                                    "n_steps": len(chezy_steps),
                                    "first_step": s0,
                                    "sample_shape": tuple(Carr.shape),
                                }
                                add_info(
                                    f"friction_chezy OK: root={chezy_root}, first_step={s0}, shape={Carr.shape}"
                                )

                    vel_root = self.paths.velocity_root
                    if vel_root not in f_vel:
                        add_warning(f'flow_velocity root from XDMF not found: {vel_root}')
                        try:
                            vel_root = self._find_group_with_numeric_children(f_vel, None)
                            add_info(f"Recovered velocity root: {vel_root}")
                        except Exception as e:
                            add_error(f"Could not recover velocity root: {e}")

                    if vel_root in f_vel:
                        vel_steps = _sort_steps_numeric(list_datasets(f_vel, vel_root))
                        if not vel_steps:
                            add_error(f"No velocity steps under {vel_root}")
                        else:
                            s0 = vel_steps[0]
                            UV = np.asarray(f_vel[f"{vel_root}/{s0}"][()], dtype=float)
                            report["variables"]["flow_velocity"] = {
                                "exists": True,
                                "root": vel_root,
                                "n_steps": len(vel_steps),
                                "first_step": s0,
                                "sample_shape": tuple(UV.shape),
                            }
                            add_info(
                                f"flow_velocity OK: root={vel_root}, first_step={s0}, shape={UV.shape}"
                            )

                finally:
                    if f_chezy is not None:
                        f_chezy.close()

        except Exception as e:
            add_error(f"HDF5 validation failed: {e}")

        if raise_on_error and not report["ok"]:
            raise RuntimeError("\n".join(report["errors"]))

        return report

    def load_meta(self) -> BasementMeta:
        rep = self.validate_outputs(verbose=False, raise_on_error=True)

        assert self.paths is not None
        hyd_path = self._resolve_h5_path(self.paths.water_surface_file)
        vel_path = self._resolve_h5_path(self.paths.velocity_file)
        zb_path = self._resolve_h5_path(self.paths.bottom_file)

        with h5py.File(zb_path, "r") as f_zb, h5py.File(hyd_path, "r") as f_hyd, h5py.File(vel_path, "r") as f_vel:
            XY = np.asarray(f_zb["/NodesAll/Coordnts"][()], dtype=float)
            if XY.ndim != 2:
                raise ValueError(f"/NodesAll/Coordnts expected 2D, got {XY.shape}")
            if XY.shape[1] != 3 and XY.shape[0] == 3:
                XY = XY.T
            XY = XY[:, :2]
            Nn = int(XY.shape[0])

            tri = _norm_tri(f_zb["/CellsAll/Topology"][()])
            Nc = int(tri.shape[0])

            zb_c = np.asarray(f_zb[self.paths.bottom_path][()], dtype=float).ravel()
            if zb_c.size != Nc:
                raise ValueError(f"Bottom elevation size {zb_c.size} != Nc {Nc}")

            if "/Parameters/MinWaterDepth" in f_zb:
                hmin = float(np.asarray(f_zb["/Parameters/MinWaterDepth"][()]).ravel()[0])
            else:
                hmin = self.hmin_fallback

            hyd_root = rep["variables"]["water_surface"]["root"]
            vel_root = rep["variables"]["flow_velocity"]["root"]

            steps_hyd = _sort_steps_numeric(list_datasets(f_hyd, hyd_root))
            steps_vel = _sort_steps_numeric(list_datasets(f_vel, vel_root))

        self.paths.water_surface_root = hyd_root
        self.paths.velocity_root = vel_root

        self.meta = BasementMeta(
            XY=XY,
            tri=tri,
            zb_c=zb_c,
            hmin=hmin,
            steps_hyd=steps_hyd,
            steps_vel=steps_vel,
            Nc=Nc,
            Nn=Nn,
        )
        return self.meta

    @staticmethod
    def _pick_step(steps: list[str], k: int | None) -> str:
        if not steps:
            raise ValueError("Empty step list")
        if k is None:
            return steps[-1]
        k = int(np.clip(k, 0, len(steps) - 1))
        return steps[k]

    @staticmethod
    def _as_nc_nvar(arr: np.ndarray, Nc: int, label: str) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)

        if arr.ndim == 1:
            if arr.size != Nc:
                raise ValueError(f"{label} length {arr.size} != Nc {Nc}")
            return arr.reshape(Nc, 1)

        if arr.ndim != 2:
            raise ValueError(f"{label} expected 1D/2D, got {arr.shape}")

        if arr.shape[0] == Nc:
            return arr

        if arr.shape[1] == Nc:
            return arr.T

        raise ValueError(f"{label} shape {arr.shape} incompatible with Nc={Nc}")

    def last_index(self) -> int:
        if self.meta is None:
            self.load_meta()
        assert self.meta is not None
        return len(self.meta.steps_hyd) - 1

    def load_frame(self, k: int | None = None) -> BasementFrame:
        if self.meta is None:
            self.load_meta()

        assert self.meta is not None
        assert self.paths is not None

        meta = self.meta
        step = self._pick_step(meta.steps_hyd, k)
        step_vel = _strip_step_zeros(step)

        hyd_path = self._resolve_h5_path(self.paths.water_surface_file)
        vel_path = self._resolve_h5_path(self.paths.velocity_file)
        chezy_path = self._resolve_h5_path(self.paths.chezy_file) if self.paths.chezy_file else None

        with h5py.File(hyd_path, "r") as f_hyd, h5py.File(vel_path, "r") as f_vel:
            Hraw = f_hyd[f"{self.paths.water_surface_root}/{step}"][()]
            H = self._as_nc_nvar(Hraw, meta.Nc, f"HydState/{step}")

            if self.paths.water_surface_col >= H.shape[1]:
                raise ValueError(
                    f"water_surface column {self.paths.water_surface_col} out of bounds "
                    f"for HydState shape {H.shape}"
                )

            wse_c = np.asarray(H[:, self.paths.water_surface_col], dtype=float).ravel()
            h_c = np.maximum(wse_c - meta.zb_c, 0.0)

            UVraw = f_vel[f"{self.paths.velocity_root}/{step_vel}"][()]
            UV = np.asarray(UVraw, dtype=float)

            if UV.ndim == 2 and UV.shape[0] != meta.Nc and UV.shape[1] == meta.Nc:
                UV = UV.T

            if UV.shape != (meta.Nc, 2):
                raise ValueError(
                    f"Velocity/{step_vel} shape {UV.shape} != (Nc,2) with Nc={meta.Nc}"
                )

            U_c = UV[:, 0]
            V_c = UV[:, 1]

        Chezy_c = None
        ustar_c = None

        if chezy_path is not None and self.paths.chezy_root is not None and chezy_path.exists():
            with h5py.File(chezy_path, "r") as f_chezy:
                chezy_ds = f"{self.paths.chezy_root}/{step}"
                if chezy_ds in f_chezy:
                    Chezy_c = np.sqrt(np.asarray(f_chezy[chezy_ds][()], dtype=float).ravel())

                    if Chezy_c.size != meta.Nc:
                        raise ValueError(f"Chezy size {Chezy_c.size} != Nc {meta.Nc}")

                    Umag_c = np.hypot(U_c, V_c)
                    ustar_c = np.full(meta.Nc, np.nan, dtype=float)

                    # HydroLPT/BASEMENT convention:
                    # u* = |U| / Chezy
                    good = np.isfinite(Umag_c) & np.isfinite(Chezy_c) & (Chezy_c > 0.0)
                    if np.any(good):
                        ustar_c[good] = Umag_c[good] / Chezy_c[good]

        wet_c = (h_c > meta.hmin) & np.isfinite(h_c)

        return BasementFrame(
            step=step,
            step_vel=step_vel,
            wse_c=wse_c,
            h_c=h_c,
            U_c=U_c,
            V_c=V_c,
            Chezy_c=Chezy_c,
            wet_c=wet_c,
            ks_c=None,
            ustar_c=ustar_c,
        )

    def close(self):
        """Close any open HDF5 file handles."""
        for attr in ("_f_hyd", "_f_vel", "_f_chezy", "_f_bottom"):
            f = getattr(self, attr, None)
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
                setattr(self, attr, None)


def run_basement_case(
    run_file: str | Path,
    *,
    settings: dict,
    plots: dict | None = None,
    xdmf_path: str | Path | None = None,
    show_plots: bool = True,
    block_on_plots: bool = True,
) -> np.ndarray:
    run_file = Path(run_file).resolve()
    if str(settings.get("transportVelocityMode", "depth_averaged")).strip().lower() != "depth_averaged":
        raise ValueError("BASEMENT cases support only transportVelocityMode='depth_averaged'.")
    xdmf_path = Path(xdmf_path) if xdmf_path is not None else get_default_paths(run_file)["xdmf"]
    return run_hydraulic_case(
        run_file,
        settings=settings,
        plots=plots,
        adapter_factory=lambda: BasementAdapter(xdmf_path),
        startup_lines=[f"Using XDMF: {xdmf_path}"],
        classification_tol_default=5.0,
        show_plots=show_plots,
        block_on_plots=block_on_plots,
    )

