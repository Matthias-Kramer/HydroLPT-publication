"""Synthetic hydraulic adapter for idealized HydroLPT verification cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from core._01_hydraulic_input_and_adapter.basement import BasementMeta, BasementFrame
from core._04_transport_solver.runner import run_hydraulic_case


def _rect_tri_mesh(L: float, W: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a rectangular triangular mesh over [0, L] x [0, W]
    by splitting each background rectangle into 4 triangles
    using a cell-center node.
    """
    x = np.linspace(0.0, L, nx + 1)
    y = np.linspace(0.0, W, ny + 1)

    XX, YY = np.meshgrid(x, y, indexing="xy")
    XY_corner = np.column_stack([XX.ravel(), YY.ravel()]).astype(float)

    def nid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    centers = []
    tri = []
    base_center_id = XY_corner.shape[0]

    for j in range(ny):
        for i in range(nx):
            n00 = nid(i, j)
            n10 = nid(i + 1, j)
            n01 = nid(i, j + 1)
            n11 = nid(i + 1, j + 1)

            xc = 0.5 * (x[i] + x[i + 1])
            yc = 0.5 * (y[j] + y[j + 1])

            nc = base_center_id + len(centers)
            centers.append([xc, yc])

            tri.append([n00, n10, nc])
            tri.append([n10, n11, nc])
            tri.append([n11, n01, nc])
            tri.append([n01, n00, nc])

    XY = np.vstack([XY_corner, np.asarray(centers, dtype=float)]) if centers else XY_corner
    return XY, np.asarray(tri, dtype=int)


def _cell_centers(XY: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-center coordinates from triangle vertices."""
    pts = XY[tri]
    xc = pts[:, :, 0].mean(axis=1)
    yc = pts[:, :, 1].mean(axis=1)
    return xc, yc


def _standard_loglaw_mean_factor(h: np.ndarray, z0: float | np.ndarray, kappa: float = 0.41) -> np.ndarray:
    """Depth-averaged factor relating u* and U for a standard z0 log law."""
    h = np.asarray(h, dtype=float)
    z0 = np.asarray(z0, dtype=float)
    z0_eff = np.maximum(z0, 1e-12)
    h_eff = np.maximum(h, z0_eff * (1.0 + 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.log(h_eff / z0_eff) - 1.0 + (z0_eff / h_eff)) / kappa


def _standard_loglaw_ustar_from_speed(
    h: np.ndarray,
    speed: float | np.ndarray,
    z0: float | np.ndarray,
    kappa: float = 0.41,
) -> np.ndarray:
    """Convert depth-averaged speed to u* using the standard z0 log law."""
    factor = np.maximum(_standard_loglaw_mean_factor(h, z0, kappa=kappa), 1e-12)
    return np.asarray(speed, dtype=float) / factor


@dataclass
class SyntheticConfig:
    # Geometry
    length: float
    domain_width: float
    wetted_width: float
    water_depth: float

    # Mesh
    nx: int
    ny: int

    # Hydraulic timing
    hyd_time_start: float | None = None
    hyd_dt: float | None = None
    hyd_time_end: float | None = None

    # Transport / velocity mode
    transportVelocityMode: str = "depth_averaged"  # "depth_averaged", "depth_averaged_above_release", or "loglaw_vertical"

    # Hydraulic closure
    hydraulicClosureModel: str = "standard_z0"

    # Steady prescribed mean velocity
    U: float | None = None
    V: float | None = None

    # Roughness
    z0: float | None = None

    # Run setup
    run_mode: str = "steady"         # "steady" or "transient"
    transient_shape: str = "step"    # "step" or "linear"

    # Two-stage mean velocities
    U0: float | None = None
    U1: float | None = None
    V0: float = 0.0
    V1: float = 0.0

    # Location of switch or end of ramp in normalized frame time [0,1]
    switch_fraction: float = 0.5

    # Wet threshold
    hmin: float = 1e-6


class SyntheticAdapter:
    """
    Synthetic HydroLPT adapter for idealized verification cases.

    Notes
    -----
    - frame.Chezy_c stores the HydroLPT friction coefficient
    - frame.ustar_c is attached as a synthetic convenience field
    - frame.ks_c is attached for transport velocity sampling
    """

    adapter_name = "synthetic"

    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.meta: BasementMeta | None = None

        self._times: np.ndarray | None = None
        self._XY: np.ndarray | None = None
        self._tri: np.ndarray | None = None
        self._xc: np.ndarray | None = None
        self._yc: np.ndarray | None = None
        self._zb_c: np.ndarray | None = None
        self._wet_corridor: np.ndarray | None = None

    def _ensure_meta(self) -> None:
        if self.meta is None:
            self.load_meta()

    def _hyd_times(self) -> np.ndarray:
        cfg = self.config

        if cfg.run_mode == "steady":
            if cfg.hyd_time_start is None or cfg.hyd_dt is None or cfg.hyd_time_end is None:
                return np.array([0.0], dtype=float)

        if cfg.hyd_dt <= 0:
            raise ValueError("hyd_dt must be > 0")
        if cfg.hyd_time_end < cfg.hyd_time_start:
            raise ValueError("hyd_time_end must be >= hyd_time_start")

        n_frames = int(round((cfg.hyd_time_end - cfg.hyd_time_start) / cfg.hyd_dt)) + 1
        times = cfg.hyd_time_start + np.arange(n_frames, dtype=float) * cfg.hyd_dt

        expected_end = cfg.hyd_time_start + (n_frames - 1) * cfg.hyd_dt
        if not np.isclose(expected_end, cfg.hyd_time_end, atol=1e-12, rtol=1e-9):
            raise ValueError(
                "hyd_time_end is not consistent with hyd_time_start and hyd_dt. "
                f"Expected {expected_end}, got {cfg.hyd_time_end}."
            )

        return times

    def _build_wet_corridor(self, yc: np.ndarray) -> np.ndarray:
        cfg = self.config
        y_mid = 0.5 * cfg.domain_width
        half_wet = 0.5 * cfg.wetted_width
        return (yc >= y_mid - half_wet) & (yc <= y_mid + half_wet)

    def _build_bed_elevation(self, xc: np.ndarray, wet_corridor: np.ndarray) -> np.ndarray:
        cfg = self.config
        # Synthetic flume uses a flat channel bed with dry banks outside the wetted corridor.
        zb_c = np.full_like(xc, cfg.water_depth + 1.0, dtype=float)
        zb_c[wet_corridor] = 0.0
        return zb_c

    def _build_depth_and_wse(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.meta is not None
        assert self._zb_c is not None

        cfg = self.config
        wse_ref = cfg.water_depth
        wse_c = np.full(self.meta.Nc, wse_ref, dtype=float)
        h_c = np.maximum(wse_c - self._zb_c, 0.0)
        return wse_c, h_c

    def _frame_alpha(self, k: int) -> float:
        cfg = self.config
        times = self._hyd_times()
        n_frames = len(times)

        if n_frames <= 1:
            return 0.0

        s = k / (n_frames - 1)

        if cfg.run_mode == "steady":
            return 0.0

        sf = float(np.clip(cfg.switch_fraction, 0.0, 1.0))

        if cfg.transient_shape == "step":
            return 0.0 if s < sf else 1.0

        if cfg.transient_shape == "linear":
            if sf <= 0.0:
                return 1.0
            if sf >= 1.0:
                return 0.0
            return float(np.clip(s / sf, 0.0, 1.0))

        raise ValueError(f"Unknown transient_shape: {cfg.transient_shape}")

    def _frame_velocity_values(self, k: int) -> tuple[float, float]:
        cfg = self.config

        if cfg.run_mode == "steady":
            return float(cfg.U), float(cfg.V)

        a = self._frame_alpha(k)
        U = (1.0 - a) * float(cfg.U0) + a * float(cfg.U1)
        V = (1.0 - a) * float(cfg.V0) + a * float(cfg.V1)
        return U, V

    def _build_hydraulic_fields(
        self, h_c: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.meta is not None
        cfg = self.config

        U_c = np.zeros(self.meta.Nc, dtype=float)
        V_c = np.zeros(self.meta.Nc, dtype=float)
        ustar_c = np.full(self.meta.Nc, np.nan, dtype=float)
        chezy_c = np.full(self.meta.Nc, np.nan, dtype=float)
        ks_c = np.full(self.meta.Nc, np.nan, dtype=float)
        z0_c = np.full(self.meta.Nc, np.nan, dtype=float)

        wet = h_c > self.meta.hmin
        if not np.any(wet):
            return U_c, V_c, ustar_c, chezy_c, ks_c, z0_c

        if cfg.transportVelocityMode not in ("depth_averaged", "depth_averaged_above_release", "loglaw_vertical"):
            raise ValueError(
                "transportVelocityMode must be 'depth_averaged', "
                "'depth_averaged_above_release', or 'loglaw_vertical'"
            )
        if cfg.hydraulicClosureModel != "standard_z0":
            raise ValueError("Synthetic flume cases support only hydraulicClosureModel='standard_z0'")

        U_val, V_val = self._frame_velocity_values(k)
        U_c[wet] = U_val
        V_c[wet] = V_val
        speed = np.hypot(U_c[wet], V_c[wet])

        z0 = float(cfg.z0)
        z0_c[wet] = z0
        ks_c[wet] = 30.0 * z0
        chezy_c[wet] = _standard_loglaw_mean_factor(h_c[wet], z0)
        ustar_c[wet] = _standard_loglaw_ustar_from_speed(h_c[wet], speed, z0)

        return U_c, V_c, ustar_c, chezy_c, ks_c, z0_c

    def validate_outputs(self, *, verbose: bool = True, raise_on_error: bool = False) -> dict:
        cfg = self.config
        errors: list[str] = []
        warnings: list[str] = []

        # Geometry
        if cfg.length <= 0:
            errors.append("length must be > 0")
        if cfg.domain_width <= 0:
            errors.append("domain_width must be > 0")
        if cfg.wetted_width <= 0:
            errors.append("wetted_width must be > 0")
        if cfg.wetted_width >= cfg.domain_width:
            errors.append("wetted_width must be smaller than domain_width")
        if cfg.water_depth <= 0:
            errors.append("water_depth must be > 0")
        if cfg.water_depth > 1.0:
            errors.append("water_depth must be <= 1.0 for the synthetic flume case")

        # Mesh
        if cfg.nx < 1 or cfg.ny < 1:
            errors.append("nx and ny must be >= 1")

        # Hydraulic timing
        if cfg.run_mode == "steady":
            if not (
                cfg.hyd_time_start is None
                and cfg.hyd_dt is None
                and cfg.hyd_time_end is None
            ):
                if cfg.hyd_time_start is None or cfg.hyd_dt is None or cfg.hyd_time_end is None:
                    errors.append(
                        "For steady synthetic cases, hyd_time_start, hyd_dt, and hyd_time_end "
                        "must either all be omitted or all be provided."
                    )
                else:
                    if cfg.hyd_dt <= 0:
                        errors.append("hyd_dt must be > 0")
                    if cfg.hyd_time_end < cfg.hyd_time_start:
                        errors.append("hyd_time_end must be >= hyd_time_start")
                    else:
                        span = cfg.hyd_time_end - cfg.hyd_time_start
                        n_frames = int(round(span / cfg.hyd_dt)) + 1
                        reconstructed_end = cfg.hyd_time_start + max(n_frames - 1, 0) * cfg.hyd_dt
                        if not np.isclose(reconstructed_end, cfg.hyd_time_end, atol=1e-12, rtol=1e-9):
                            errors.append("hyd_time_end must be consistent with hyd_time_start and hyd_dt")
        else:
            if cfg.hyd_time_start is None or cfg.hyd_dt is None or cfg.hyd_time_end is None:
                errors.append("Transient synthetic cases require hyd_time_start, hyd_dt, and hyd_time_end")
            else:
                if cfg.hyd_dt <= 0:
                    errors.append("hyd_dt must be > 0")
                if cfg.hyd_time_end < cfg.hyd_time_start:
                    errors.append("hyd_time_end must be >= hyd_time_start")
                else:
                    span = cfg.hyd_time_end - cfg.hyd_time_start
                    n_frames = int(round(span / cfg.hyd_dt)) + 1
                    reconstructed_end = cfg.hyd_time_start + max(n_frames - 1, 0) * cfg.hyd_dt
                    if not np.isclose(reconstructed_end, cfg.hyd_time_end, atol=1e-12, rtol=1e-9):
                        errors.append("hyd_time_end must be consistent with hyd_time_start and hyd_dt")

        # General hydraulics
        if cfg.transportVelocityMode not in ("depth_averaged", "depth_averaged_above_release", "loglaw_vertical"):
            errors.append(
                "transportVelocityMode must be 'depth_averaged', "
                "'depth_averaged_above_release', or 'loglaw_vertical'"
            )
        if cfg.hydraulicClosureModel != "standard_z0":
            errors.append("Synthetic flume cases support only hydraulicClosureModel='standard_z0'")
        if cfg.hmin < 0:
            errors.append("hmin must be >= 0")
        if cfg.run_mode not in ("steady", "transient"):
            errors.append("run_mode must be 'steady' or 'transient'")
        if cfg.transient_shape not in ("step", "linear"):
            errors.append("transient_shape must be 'step' or 'linear'")
        if not (0.0 <= cfg.switch_fraction <= 1.0):
            errors.append("switch_fraction must be between 0 and 1")

        if cfg.z0 is None or not np.isfinite(cfg.z0) or cfg.z0 <= 0:
            errors.append("z0 must be finite and > 0 for synthetic flume cases")
        elif cfg.z0 >= cfg.water_depth:
            errors.append("z0 must be smaller than water_depth for synthetic flume cases")

        # Mode-specific validation
        if cfg.run_mode == "steady":
            if cfg.U is None or cfg.V is None:
                errors.append("U and V must be provided in steady synthetic mode")
            else:
                if not np.isfinite(cfg.U) or not np.isfinite(cfg.V):
                    errors.append("U and V must be finite in steady synthetic mode")
        else:
            if cfg.U0 is None or cfg.U1 is None:
                errors.append("U0 and U1 must be provided in transient synthetic mode")
            if not np.isfinite(cfg.V0) or not np.isfinite(cfg.V1):
                errors.append("V0 and V1 must be finite in transient synthetic mode")

        n_frames = 0
        if (
            cfg.hyd_time_start is not None
            and cfg.hyd_dt is not None
            and cfg.hyd_time_end is not None
            and cfg.hyd_dt > 0
            and cfg.hyd_time_end >= cfg.hyd_time_start
        ):
            n_frames = int(round((cfg.hyd_time_end - cfg.hyd_time_start) / cfg.hyd_dt)) + 1

        if cfg.run_mode == "transient" and n_frames < 2:
            warnings.append("Transient mode is more meaningful with at least 2 hydraulic frames")

        if n_frames == 1:
            warnings.append("Single-frame hydraulic setup: effectively steady.")

        ok = len(errors) == 0

        if verbose:
            print("\nSyntheticAdapter validation summary:")
            print("OK:", ok)
            print("Errors:", errors)
            print("Warnings:", warnings)

        if raise_on_error and not ok:
            raise RuntimeError("\n".join(errors))

        return {
            "ok": ok,
            "errors": errors,
            "warnings": warnings,
            "files": {},
            "paths": {},
            "variables": {
                "mesh": True,
                "bottom_elevation": True,
                "water_surface": True,
                "flow_velocity": True,
                "friction_chezy": True,
                "ustar_synthetic": True,
                "ks_synthetic": False,
            },
        }

    def load_meta(self) -> BasementMeta:
        cfg = self.config

        XY, tri = _rect_tri_mesh(cfg.length, cfg.domain_width, cfg.nx, cfg.ny)
        xc, yc = _cell_centers(XY, tri)
        wet_corridor = self._build_wet_corridor(yc)
        zb_c = self._build_bed_elevation(xc, wet_corridor)

        times = self._hyd_times()
        steps = [str(i) for i in range(len(times))]

        self._times = times
        self._XY = XY
        self._tri = tri
        self._xc = xc
        self._yc = yc
        self._zb_c = zb_c
        self._wet_corridor = wet_corridor

        self.meta = BasementMeta(
            XY=XY,
            tri=tri,
            zb_c=zb_c,
            hmin=float(cfg.hmin),
            steps_hyd=steps,
            steps_vel=steps,
            Nc=int(tri.shape[0]),
            Nn=int(XY.shape[0]),
        )
        return self.meta

    def last_index(self) -> int:
        self._ensure_meta()
        assert self.meta is not None
        return len(self.meta.steps_hyd) - 1

    def default_release_centers(self, n: int = 1, x: float | None = None) -> np.ndarray:
        self._ensure_meta()
        cfg = self.config

        if x is None:
            x = 0.05 * cfg.length

        y_mid = 0.5 * cfg.domain_width
        half_wet = 0.5 * cfg.wetted_width

        if n == 1:
            return np.array([[x, y_mid]], dtype=float)

        y = np.linspace(
            y_mid - 0.8 * half_wet,
            y_mid + 0.8 * half_wet,
            int(n),
        )
        xarr = np.full_like(y, x, dtype=float)
        return np.column_stack([xarr, y])

    def load_frame(self, k: int | None = None) -> BasementFrame:
        self._ensure_meta()
        assert self.meta is not None

        if k is None:
            k = self.last_index()
        k = int(np.clip(k, 0, self.last_index()))

        step = str(k)
        step_vel = str(k)

        wse_c, h_c = self._build_depth_and_wse()
        U_c, V_c, ustar_c, chezy_c, ks_c, z0_c = self._build_hydraulic_fields(h_c, k)
        wet_c = (h_c > self.meta.hmin) & np.isfinite(h_c)

        frame = BasementFrame(
            step=step,
            step_vel=step_vel,
            wse_c=wse_c,
            h_c=h_c,
            U_c=U_c,
            V_c=V_c,
            Chezy_c=chezy_c,
            wet_c=wet_c,
        )

        frame.ustar_c = ustar_c
        frame.ks_c = ks_c
        frame.z0_c = z0_c
        return frame

    def close(self):
        pass


def _build_cfg(settings: dict) -> SyntheticConfig:
    hydraulic_closure = settings.get("hydraulicClosureModel")
    if hydraulic_closure is None:
        hydraulic_closure = "standard_z0"
    if str(hydraulic_closure).strip().lower() != "standard_z0":
        raise ValueError("Synthetic flume cases support only hydraulicClosureModel='standard_z0'")
    hydraulic_primary = settings.get("hydraulicPrimaryVariable", "U")
    if hydraulic_primary is None:
        hydraulic_primary = "U"
    if str(hydraulic_primary).strip().lower() != "u":
        raise ValueError(
            "Synthetic flume cases no longer support prescribed ustar as a primary "
            "hydraulic input. Set hydraulicPrimaryVariable='U' and prescribe U instead."
        )

    z0_value = settings.get("z0")

    cfg_kwargs = {
        "length": settings.get("length", 40.0),
        "domain_width": settings.get("domain_width", 1.0),
        "wetted_width": settings.get("wetted_width", 0.6),
        "water_depth": settings.get("water_depth", 0.5),
        "nx": settings.get("nx", 800),
        "ny": settings.get("ny", 20),
        "hyd_time_start": settings.get("hyd_time_start"),
        "hyd_dt": settings.get("hyd_dt"),
        "hyd_time_end": settings.get("hyd_time_end"),
        "transportVelocityMode": settings["transportVelocityMode"],
        "hydraulicClosureModel": "standard_z0",
        "z0": z0_value,
        "hmin": settings["hmin"],
    }

    if settings["run_mode"] == "steady":
        cfg_kwargs.update(
            run_mode="steady",
            U=settings.get("U", 0.3),
            V=settings.get("V", 0.0),
        )
    else:
        cfg_kwargs.update(
            run_mode="transient",
            transient_shape=settings.get("transient_shape", "step"),
            switch_fraction=settings.get("switch_fraction", 0.5),
            U0=settings.get("U0", 0.03),
            U1=settings.get("U1", 0.30),
            V0=settings.get("V0", 0.0),
            V1=settings.get("V1", 0.0),
            U=settings.get("U"),
            V=settings.get("V", 0.0),
        )

    return SyntheticConfig(**cfg_kwargs)


def run_synthetic_case(
    run_file: str | Path,
    *,
    settings: dict,
    plots: dict | None = None,
    show_plots: bool = True,
    block_on_plots: bool = True,
) -> np.ndarray:
    cfg = _build_cfg(settings)
    return run_hydraulic_case(
        run_file,
        settings=settings,
        plots=plots,
        adapter_factory=lambda: SyntheticAdapter(cfg),
        startup_lines=["Using synthetic adapter: flume test"],
        classification_tol_default=3.0,
        enable_rouse=True,
        enable_advection_diffusion=True,
        show_plots=show_plots,
        block_on_plots=block_on_plots,
    )

