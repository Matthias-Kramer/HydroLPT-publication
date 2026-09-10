"""Shared HydroLPT execution workflow for synthetic and BASEMENT cases."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from core._05_boundary_interaction import (
    d50_from_chezy_loglaw,
    ustarcrit_cells_from_d50,
    ustarcrit_surface_from_particle,
)
from core._03_particle_evolution import (
    biofouling_config_from_settings,
    degradation_config_from_settings,
)
from core._02_particle_initialisation import (
    build_particle_properties,
    build_release_clouds,
    build_release_positions_3d_gaussian_vertical,
    build_release_schedule_per_center,
    filter_release_centers,
    get_release_centers,
    print_particle_info,
)
from core._01_hydraulic_input_and_adapter.mesh import TriMesh
from core._04_transport_solver.transient import (
    advect_particles_euler_cell_transient_3d_ustar,
    build_hydro_cache,
    interpolate_hydro_fields,
)
from core._04_transport_solver.steady import (
    _LocalCellIDWFieldSampler,
    advect_particles_euler_cell_fast_fullcache_3d_ustar,
)
from core._06_output_diagnostics.export import export_all
from core._06_output_diagnostics.plotting import (
    make_figure_advection_diffusion,
    make_figure_binary_map,
    make_figure_rouse_profile_section,
    make_figure_shields,
    make_figure_trajectories_tracked,
)
from core._06_output_diagnostics.utils import classify_end_state_all

ICON_PNG_PATH = Path("assets") / "hydrolpt_icon_option2c.png"
ICON_ICO_PATH = Path("assets") / "hydrolpt_icon_option2c.ico"


def _runtime_asset_path(relative_path: Path) -> Path:
    """Resolve an asset path for both source and frozen application runs."""
    if getattr(sys, "frozen", False):
        runtime_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    else:
        runtime_root = Path(__file__).resolve().parents[1]
    return runtime_root / relative_path


def _apply_plot_window_icons() -> None:
    """Apply the HydroLPT icon to Tk-backed Matplotlib figure windows."""
    png_path = _runtime_asset_path(ICON_PNG_PATH)
    ico_path = _runtime_asset_path(ICON_ICO_PATH)

    for num in plt.get_fignums():
        manager = plt.figure(num).canvas.manager
        window = getattr(manager, "window", None)
        if window is None:
            continue

        try:
            if png_path.exists():
                import tkinter as tk

                icon_image = tk.PhotoImage(file=str(png_path))
                window.iconphoto(True, icon_image)
                setattr(window, "_hydrolpt_plot_icon_image", icon_image)
        except Exception:
            pass

        try:
            if ico_path.exists() and hasattr(window, "iconbitmap"):
                window.iconbitmap(str(ico_path))
        except Exception:
            pass

def _position_plot_windows(window_anchor: dict | None) -> None:
    """Offset plot windows away from the GUI to reduce overlap."""
    if not window_anchor:
        return

    base_x = int(window_anchor.get("x", 100))
    base_y = int(window_anchor.get("y", 100))
    base_w = int(window_anchor.get("width", 900))

    for idx, num in enumerate(plt.get_fignums()):
        manager = plt.figure(num).canvas.manager
        window = getattr(manager, "window", None)
        if window is None or not hasattr(window, "wm_geometry"):
            continue

        offset_x = base_x + 40 + (idx % 3) * 30
        offset_y = base_y + 40 + (idx % 3) * 30
        if idx > 0:
            offset_x = min(offset_x + 80, base_x + max(base_w - 220, 100))
        try:
            window.wm_geometry(f"+{offset_x}+{offset_y}")
        except Exception:
            continue


def _apply_plot_window_size(window_anchor: dict | None) -> None:
    """Resize plot windows using the current screen size heuristics."""
    if not window_anchor:
        return

    screen_width = int(window_anchor.get("screen_width") or 0)
    screen_height = int(window_anchor.get("screen_height") or 0)
    if screen_width <= 0 or screen_height <= 0:
        return

    target_width_px = max(int(round(0.35 * screen_width)), 640)
    target_height_px = max(int(round(0.65 * screen_height)), 480)

    for num in plt.get_fignums():
        fig = plt.figure(num)
        dpi = float(fig.get_dpi()) if fig.get_dpi() else 100.0
        if num == 1:
            continue
        target_height_in = target_height_px / dpi
        target_width_in = target_width_px / dpi
        fig.set_size_inches(target_width_in, target_height_in, forward=True)

def _resolve_time_window(settings: dict) -> tuple[float, float, float]:
    """Resolve start, end, and release times from solver settings."""
    run_mode = settings["run_mode"]
    t_track = float(settings["tTrack"])
    t_release = float(settings.get("tRelease", 0.0))

    if run_mode == "steady":
        return 0.0, t_track, t_release
    if run_mode == "transient":
        hyd_time_end = float(settings["hyd_time_end"])
        return t_release, min(t_release + t_track, hyd_time_end), t_release
    raise ValueError("run_mode must be 'steady' or 'transient'.")


def _validate_continuous_release_capacity(settings: dict) -> None:
    """Cap continuous release so it fits inside the tracking duration."""
    release_mode = str(settings.get("release_mode", "bulk")).lower()
    if release_mode != "continuous":
        return

    t_track = float(settings["tTrack"])
    release_dt = float(settings.get("release_dt", 1.0))
    n_per_center = int(settings["n_per_center"])

    if release_dt <= 0.0:
        raise ValueError("release_dt must be > 0 for continuous release.")

    max_n_per_center = max(1, int(np.floor(t_track / release_dt)) + 1)
    if n_per_center > max_n_per_center:
        print(
            "Total particles reduced because the maximum possible number "
            f"for continuous release was exceeded: N reduced from {n_per_center} "
            f"to {max_n_per_center} per center for tTrack={t_track:g} s "
            f"and release_dt={release_dt:g} s."
        )
        settings["n_per_center"] = max_n_per_center


def _validate_hmin_particle_scale(settings: dict, particle) -> None:
    """Reject wet-depth thresholds that are too large for the selected particle."""
    hmin = float(settings["hmin"])
    dv = float(particle.DV)
    if not np.isfinite(hmin) or hmin < 0.0:
        raise ValueError(f"hmin must be a finite non-negative value (got {hmin}).")
    if not np.isfinite(dv) or dv <= 0.0:
        raise ValueError(f"Particle volumetric diameter DV must be > 0 (got {dv}).")

    hmin_max = 2.0 * dv
    if hmin > hmin_max:
        raise ValueError(
            "hmin must be <= 2 * DV for physically meaningful wet/dry handling "
            f"(hmin={hmin:g} m, DV={dv:g} m, 2*DV={hmin_max:g} m)."
        )


def _resolve_output_timestep(settings: dict) -> tuple[float, int]:
    """Resolve exported trajectory cadence as an integer multiple of particle dt."""
    dt = float(settings["dt"])
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be a finite positive value (got {dt}).")

    raw_output_dt = settings.get("output_dt", None)
    if raw_output_dt in (None, ""):
        output_dt = dt
    else:
        output_dt = float(raw_output_dt)

    if not np.isfinite(output_dt) or output_dt <= 0.0:
        raise ValueError(f"output_dt must be a finite positive value (got {output_dt}).")
    if output_dt < dt - 1e-12:
        raise ValueError(f"output_dt must be >= dt (output_dt={output_dt:g} s, dt={dt:g} s).")

    ratio = output_dt / dt
    stride = int(round(ratio))
    if stride < 1 or not np.isclose(ratio, stride, rtol=1e-9, atol=1e-12):
        raise ValueError(
            "output_dt must be an integer multiple of the particle tracking timestep dt "
            f"(output_dt={output_dt:g} s, dt={dt:g} s, ratio={ratio:g})."
        )

    output_dt = float(stride * dt)
    settings["output_dt"] = output_dt
    return output_dt, stride


def _build_particle(settings: dict):
    """Build and print the derived particle property bundle."""
    particle = build_particle_properties(
        shape=settings["shape"],
        rho_p=settings["rho_p"],
        L=settings["L"],
        I=settings.get("I"),
        S=settings.get("S"),
        rho_f=settings["rho_f"],
        nu=settings["nu"],
        g=settings["g"],
    )
    imposed_ws = settings.get("imposed_ws")
    if imposed_ws is not None:
        particle.ws = float(imposed_ws)
    print_particle_info(particle)
    if imposed_ws is not None:
        print("  note         : ws was imposed directly by the run settings")
    return particle


def _build_biofouling(settings: dict, particle) -> dict | None:
    """Build a simple biofouling-evolution bundle from case settings."""
    config = biofouling_config_from_settings(settings)
    if not config.enabled:
        return None
    return {
        "config": config,
        "shape": particle.shape,
        "rho_p0": float(particle.rho_p),
        "L": float(particle.L),
        "I": float(particle.I),
        "S": float(particle.S),
        "rho_f": float(settings["rho_f"]),
        "nu": float(settings["nu"]),
        "g": float(settings["g"]),
        "tanphi_ratio": float(settings["tanphi_ratio"]),
    }


def _build_degradation(settings: dict, particle) -> dict | None:
    """Build a simple degradation-evolution bundle from case settings."""
    config = degradation_config_from_settings(settings)
    if not config.enabled:
        return None
    return {
        "config": config,
        "shape": particle.shape,
        "rho_p0": float(particle.rho_p),
        "L": float(particle.L),
        "I": float(particle.I),
        "S": float(particle.S),
        "rho_f": float(settings["rho_f"]),
        "nu": float(settings["nu"]),
        "g": float(settings["g"]),
        "tanphi_ratio": float(settings["tanphi_ratio"]),
    }


def _prepare_steady_fields(frame, particle, settings: dict):
    """Prepare steady-frame roughness and critical shear fields."""
    hmin = settings["hmin"]
    chezy_c = np.asarray(frame.Chezy_c, dtype=float).ravel()
    ustar_c = np.asarray(frame.ustar_c, dtype=float).ravel()

    chezy_c[(~np.isfinite(chezy_c)) | (chezy_c <= 0)] = np.nan
    ustar_c[~np.isfinite(ustar_c)] = 0.0

    if settings["transportVelocityMode"] == "loglaw_vertical" and frame.ks_c is None:
        raise RuntimeError("transportVelocityMode='loglaw_vertical' requires frame.ks_c.")

    d50_c = d50_from_chezy_loglaw(chezy_c, frame.h_c, hmin, verbose=False)
    ustar_crit_c = ustarcrit_cells_from_d50(
        d50_c=d50_c,
        Dp=particle.Dp,
        rho_p=particle.rho_p,
        beta_p=particle.beta_p,
        tanphi_ratio=settings["tanphi_ratio"],
        rho_w=settings["rho_f"],
        nu=settings["nu"],
        g=settings["g"],
        beta_s=1.5,
        c2=-0.6,
        u0=0.005,
        n_iter=30,
        tol=1e-8,
    )
    return ustar_c, ustar_crit_c, d50_c


def _compute_surface_ustar_crit(particle, settings: dict) -> float:
    """Compute a particle-specific free-surface detachment threshold."""
    return ustarcrit_surface_from_particle(
        shape=particle.shape,
        rho_p=particle.rho_p,
        L=particle.L,
        I=particle.I,
        S=particle.S,
        Vp=particle.Vp,
        rho_f=settings["rho_f"],
        g=settings["g"],
        sigma=float(settings.get("surfaceTension", 0.072)),
        contact_angle_deg=float(settings.get("surfaceContactAngleDeg", 105.0)),
        drag_coeff_vertical=float(settings.get("surfaceVerticalDragCoeff", 2.0)),
    )


def _effective_boundary_policies(ws: float, bed_policy: str, surface_policy: str) -> tuple[str, str]:
    """Lock vertical boundary policies from the exact settling-velocity sign."""
    ws = float(ws)
    bed_policy = str(bed_policy).strip().lower()
    surface_policy = str(surface_policy).strip().lower()
    if ws < 0.0:
        return "always_reflect", surface_policy
    if ws > 0.0:
        return bed_policy, surface_policy
    return "always_reflect", "always_reflect"


def _vertical_overlap_preference(particle, settings: dict) -> str:
    """Return the preferred mobile-state label when shallow-water zones overlap."""
    rho_f = float(settings["rho_f"])
    rho_p = float(particle.rho_p)
    if rho_p == rho_f:
        return "closest"
    return "bed" if rho_p > rho_f else "surface"


def _build_final_state(state: dict, n_particles: int) -> np.ndarray:
    """Collapse solver state flags into a single final-state label array."""
    final_state = np.full(n_particles, "unknown", dtype=object)
    final_state[state["stoppedByOutside"]] = "outside"
    final_state[state["stoppedByLimiter"]] = "limiter"
    final_state[state["stoppedByDry"]] = "dry"
    final_state[state["isDeposited"]] = "deposited"
    final_state[state["isSurfaced"]] = "surfaced"
    mask = state["isNearBedMobile"] & (final_state == "unknown")
    final_state[mask] = "near_bed_mobile"
    mask = state["isNearSurfaceMobile"] & (final_state == "unknown")
    final_state[mask] = "near_surface_mobile"
    mask = state["isSuspended"] & (final_state == "unknown")
    final_state[mask] = "suspended"
    return final_state


def _print_vertical_stats(z_end: np.ndarray) -> None:
    z_valid = z_end[np.isfinite(z_end)]
    if z_valid.size == 0:
        print("\nNo valid z values found.")
        return
    print("\nVertical distribution statistics:")
    print(f"z mean   = {np.mean(z_valid):.6f} m")
    print(f"z median = {np.median(z_valid):.6f} m")
    print(f"z min    = {np.min(z_valid):.6f} m")
    print(f"z max    = {np.max(z_valid):.6f} m")


def _compute_kx(h_plot, ustar_plot, beta_kh, kh_max, r_aniso):
    """Estimate an effective along-flow diffusivity for diagnostics."""
    wet = h_plot > 0.0
    if not np.any(wet):
        return 0.0
    kh = beta_kh * ustar_plot[wet] * h_plot[wet]
    kh = np.minimum(kh, kh_max)
    kpar = np.minimum(r_aniso * kh, kh_max)
    return float(np.mean(kpar)) if kpar.size else 0.0


def _summarize_solver_profile(profile: dict | None, solver_elapsed_s: float) -> dict | None:
    """Build a compact solver profiling summary suitable for printing/export."""
    if not profile:
        return None

    sections = {
        str(key): float(value)
        for key, value in profile.get("sections", {}).items()
        if float(value) > 0.0
    }
    counts = {
        str(key): int(value)
        for key, value in profile.get("counts", {}).items()
    }

    if not sections and not counts:
        return None

    top_sections = sorted(sections.items(), key=lambda item: item[1], reverse=True)[:8]
    return {
        "solver_elapsed_s": float(solver_elapsed_s),
        "counts": counts,
        "sections_s": sections,
        "top_sections": [
            {
                "name": name,
                "seconds": float(seconds),
                "percent_of_solver": float(100.0 * seconds / solver_elapsed_s) if solver_elapsed_s > 0.0 else 0.0,
            }
            for name, seconds in top_sections
        ],
    }


def run_hydraulic_case(
    run_file: str | Path,
    *,
    settings: dict,
    plots: dict | None = None,
    adapter_factory: Callable[[], object],
    startup_lines: list[str] | None = None,
    classification_tol_default: float = 3.0,
    enable_rouse: bool = False,
    enable_advection_diffusion: bool = False,
    show_plots: bool = True,
    block_on_plots: bool = True,
) -> np.ndarray:
    """Run a full HydroLPT case from hydraulic adapter to exported outputs."""
    plots = dict(settings) if plots is None else {**dict(settings), **dict(plots)}
    run_file = Path(run_file).resolve()
    adapter = None

    try:
        print("\nHydroLPT v0.9")
        for line in startup_lines or []:
            print(f"\n{line}")

        z_frac = float(settings["zFrac"])
        if not (0.0 <= z_frac <= 1.0):
            raise ValueError(f"zFrac must be between 0 and 1 inclusive (got {z_frac}).")

        _validate_continuous_release_capacity(settings)
        t0, t1, t_release = _resolve_time_window(settings)
        print(f"Run mode: {settings['run_mode']}")
        print(f"Transport model: {settings.get('transportModel', 'random_walk')}")
        print(f"Hydraulic field mode: {settings.get('hydraulicFieldMode', 'cellwise')}")
        print(f"Transport velocity mode: {settings['transportVelocityMode']}")
        print(f"Particle simulation window: {t0:.1f} s -> {t1:.1f} s")

        transport_model = str(settings.get("transportModel", "random_walk")).strip().lower()
        hydraulic_field_mode = str(settings.get("hydraulicFieldMode", "cellwise")).strip().lower()

        rng = np.random.default_rng(settings.get("rng_seed"))
        n_track = int(settings.get("nTrack", settings.get("n_per_center", 6)))
        particle = _build_particle(settings)
        _validate_hmin_particle_scale(settings, particle)
        selected_bed_policy = str(settings["bedPolicy"]).strip().lower()
        selected_surface_policy = str(settings["surfacePolicy"]).strip().lower()
        effective_bed_policy, effective_surface_policy = _effective_boundary_policies(
            particle.ws,
            selected_bed_policy,
            selected_surface_policy,
        )
        if effective_bed_policy != selected_bed_policy:
            print(
                "Bed policy locked to always_reflect because the particle settling "
                f"velocity is ws={particle.ws:g} m/s."
            )
        if effective_surface_policy != selected_surface_policy:
            print(
                "Surface policy locked to always_reflect because the particle settling "
                f"velocity is ws={particle.ws:g} m/s."
            )
        settings["bedPolicy"] = effective_bed_policy
        settings["surfacePolicy"] = effective_surface_policy
        output_dt, output_stride = _resolve_output_timestep(settings)
        if output_stride > 1:
            print(
                f"Tracked trajectory export cadence: every {output_dt:g} s "
                f"({output_stride} particle timesteps)."
            )
        else:
            print(f"Tracked trajectory export cadence: every particle timestep ({output_dt:g} s).")
        bed_entrainment_sigma_star = float(settings.get("bedEntrainmentSigmaStar", 0.2) or 0.2)
        uses_probabilistic_threshold = settings["bedPolicy"] == "probabilistic_entrainment"
        if uses_probabilistic_threshold and (
            not np.isfinite(bed_entrainment_sigma_star) or bed_entrainment_sigma_star <= 0.0
        ):
            raise ValueError(
                "bedEntrainmentSigmaStar must be a finite positive value "
                f"(got {bed_entrainment_sigma_star})."
            )
        surface_detachment_sigma_star = float(settings.get("surfaceDetachmentSigmaStar", 0.2) or 0.2)
        if settings["surfacePolicy"] == "probabilistic_detachment" and (
            not np.isfinite(surface_detachment_sigma_star) or surface_detachment_sigma_star <= 0.0
        ):
            raise ValueError(
                "surfaceDetachmentSigmaStar must be a finite positive value "
                f"(got {surface_detachment_sigma_star})."
            )
        particle_evolution = {
            "biofouling": _build_biofouling(settings, particle),
            "degradation": _build_degradation(settings, particle),
        }
        if particle_evolution["biofouling"] is not None:
            print("Particle evolution: simple biofouling enabled")
        if particle_evolution["degradation"] is not None:
            print("Particle evolution: simple degradation enabled")
        surface_ustar_crit = _compute_surface_ustar_crit(particle, settings)
        adapter = adapter_factory()
        hydraulic_closure_model = settings.get("hydraulicClosureModel")
        if hydraulic_closure_model is None:
            raise ValueError(
                "hydraulicClosureModel must be declared explicitly in the case settings. "
                'Use "standard_z0" or "external_adapter".'
            )
        hydraulic_closure_model = str(hydraulic_closure_model).strip().lower()

        if plots.get("debug_hydraulics"):
            print("\nHydraulic frames from adapter:")
            for k in range(adapter.last_index() + 1):
                fr = adapter.load_frame(k)
                mean_u = np.nanmean(fr.U_c[fr.wet_c])
                if "hyd_time_start" in settings and "hyd_dt" in settings:
                    t_frame = settings["hyd_time_start"] + k * settings["hyd_dt"]
                    print(f"k={k:3d}, time={t_frame:6.1f} s, meanU={mean_u:.3f}")
                else:
                    print(f"k={k:3d}, meanU={mean_u:.3f}")

        report = adapter.validate_outputs(verbose=True, raise_on_error=False)
        print("\nValidation summary:")
        print("OK:", report["ok"])
        print("Errors:", report["errors"])
        print("Warnings:", report["warnings"])

        meta = adapter.load_meta()
        print("Number of hydraulic frames:", adapter.last_index() + 1)

        frame0 = adapter.load_frame(0)
        frame_last = adapter.load_frame(adapter.last_index())
        print("First frame step:", frame0.step)
        print("Last frame step:", frame_last.step)
        if plots.get("print_mean_u"):
            print("Mean U first frame:", np.nanmean(frame0.U_c[frame0.wet_c]))
            print("Mean U last frame: ", np.nanmean(frame_last.U_c[frame_last.wet_c]))

        mesh = adapter.build_mesh() if hasattr(adapter, "build_mesh") else TriMesh(meta.XY, meta.tri)
        xy = meta.XY
        tri0 = meta.tri
        zb_c = meta.zb_c

        frame = None
        hydro_cache = None
        h_rel = None
        h_final = None
        zb_tr_track = None
        wse_tr_track = None
        simulation_time_per_timestep_ms = None
        solver_profile = {"sections": {}, "counts": {}}
        solver_profile_summary = None

        if settings["run_mode"] == "steady":
            print("\nRunning steady-state particle tracking")
            frame = adapter.load_frame(adapter.last_index())
            umag = np.hypot(frame.U_c, frame.V_c)
            print(f"\nFrame step: {frame.step}")
            print(f"WSE (cell):    min {np.nanmin(frame.wse_c):.4g} m   max {np.nanmax(frame.wse_c):.4g} m")
            print(f"Depth (cell):  min {np.nanmin(frame.h_c):.4g} m   max {np.nanmax(frame.h_c):.4g} m")
            print(f"|U| (cell):    min {np.nanmin(umag):.4g} m/s   max {np.nanmax(umag):.4g} m/s")

            if frame.Chezy_c is None or frame.ustar_c is None:
                raise RuntimeError("Adapter did not provide Chezy_c / ustar_c.")

            ustar_c, ustar_crit_c, d50_c = _prepare_steady_fields(frame, particle, settings)
            h_for_release = frame.h_c
            wse_for_release = frame.wse_c
            wse_for_classification = frame.wse_c
        else:
            print("\nRunning transient particle tracking")
            hydro_cache = build_hydro_cache(
                adapter=adapter,
                particle=particle,
                hmin=settings["hmin"],
                tanphi_ratio=settings["tanphi_ratio"],
                rho_f=settings["rho_f"],
                nu=settings["nu"],
                g=settings["g"],
                tHyd0=settings["hyd_time_start"],
                dtHyd=settings["hyd_dt"],
                tHydEnd=settings["hyd_time_end"],
                cache_time_start=t0,
                cache_time_end=t1,
            )
            if len(hydro_cache) < 2:
                raise RuntimeError("Need at least 2 hydraulic frames for transient run.")

            h_rel = interpolate_hydro_fields(hydro_cache, t_release, settings["hmin"])
            umag_rel = np.hypot(h_rel["U_c"], h_rel["V_c"])
            print(f"\nRequested release time: {t_release:.1f} s")
            print(f"Interpolated release time used: {h_rel['time']:.1f} s")
            print(f"WSE (cell):    min {np.nanmin(h_rel['wse_c']):.4g} m   max {np.nanmax(h_rel['wse_c']):.4g} m")
            print(f"Depth (cell):  min {np.nanmin(h_rel['h_c']):.4g} m   max {np.nanmax(h_rel['h_c']):.4g} m")
            print(f"|U| (cell):    min {np.nanmin(umag_rel):.4g} m/s   max {np.nanmax(umag_rel):.4g} m/s")

            ustar_c = h_rel["ustar_c"]
            ustar_crit_c = h_rel["ustarCrit_c"]
            d50_c = h_rel["d50_c"]
            h_for_release = h_rel["h_c"]
            wse_for_release = h_rel["wse_c"]
            wse_for_classification = None

        if settings["location_mode"] == "click":
            centers = get_release_centers(
                mesh,
                scalar_c=ustar_c,
                scalar_name="Shear velocity u* (m/s)",
                XY=xy,
                tri0=tri0,
                h_c=h_for_release,
                hmin=settings["hmin"],
                window_anchor=plots.get("_window_anchor"),
            )
        else:
            centers = filter_release_centers(
                mesh=mesh,
                h_c=h_for_release,
                hmin=settings["hmin"],
                pts=np.asarray(settings["manual_centers"], dtype=float),
            )
            print(f"\nUsing {centers.shape[0]} valid predefined release center(s)")
            for center in centers:
                print(f"  x = {center[0]:.3f}   y = {center[1]:.3f}")

        if centers.shape[0] == 0:
            raise ValueError("No valid release points selected. Please try again and select release points.")

        pxy = build_release_clouds(
            centers,
            n_per_center=settings["n_per_center"],
            sigma=settings["releaseSigma"],
            rng=rng,
        )
        p0 = build_release_positions_3d_gaussian_vertical(
            pxy,
            mesh=mesh,
            zb_c=zb_c,
            wse_c=wse_for_release,
            hmin=settings["hmin"],
            z_frac=settings["zFrac"],
            sigma_z=settings["releaseSigma"],
            rng=rng,
        )
        release_mode = str(settings.get("release_mode", "bulk")).lower()
        release_start_time = t_release if settings["run_mode"] == "transient" else t0
        delayed_release = (
            release_mode == "continuous"
            or release_start_time > (t0 + 1e-12)
        )
        release_times = None
        if delayed_release:
            release_times = build_release_schedule_per_center(
                centers.shape[0],
                settings["n_per_center"],
                t_release=release_start_time,
                release_mode=release_mode,
                release_dt=settings.get("release_dt", 1.0),
            )
        nparticles = p0.shape[0]
        print(f"\nSelected {centers.shape[0]} release center(s). Total particles: {nparticles}")
        if n_track > nparticles:
            print(
                "Tracked particles reduced because the requested number exceeded "
                f"the released particles: N_track reduced from {n_track} to {nparticles}."
            )
            n_track = nparticles

        track_idx = rng.choice(nparticles, size=n_track, replace=False)
        track_idx.sort()
        r_aniso = float(settings.get("rAniso", plots.get("rAniso", 3.0)))
        if r_aniso <= 0.0:
            raise ValueError(f"rAniso must be positive (got {r_aniso}).")

        if settings["run_mode"] == "steady":
            solver_start = perf_counter()
            out = advect_particles_euler_cell_fast_fullcache_3d_ustar(
                mesh,
                frame.U_c,
                frame.V_c,
                frame.h_c,
                zb_c,
                frame.wse_c,
                ustar_c,
                ustar_crit_c,
                surface_ustar_crit,
                settings["bedPolicy"],
                p0,
                t0,
                t1,
                settings["dt"],
                settings["outsidePolicy"],
                settings["dryPolicy"],
                settings["hmin"],
                settings["useRWx"],
                settings["useRWy"],
                settings["betaKh"],
                settings["KhMax"],
                particle.ws,
                settings["zFrac"],
                settings["surfacePolicy"],
                settings.get("dzUpMax", 0.4),
                settings["uphillPolicy"],
                settings["useRWz"],
                settings["alphaKz"],
                settings["KzMax"],
                rng=rng,
                nTrack=n_track,
                track_idx=track_idx,
                showProgress=True,
                transportModel=transport_model,
                hydraulicFieldMode=hydraulic_field_mode,
                transportVelocityMode=settings["transportVelocityMode"],
                hydraulicClosureModel=hydraulic_closure_model,
                d50_c=d50_c,
                ks_c=frame.ks_c,
                z0_c=getattr(frame, "z0_c", None),
                rAniso=r_aniso,
                entrainmentProbMode="gaussian_threshold",
                entrainmentProbSigmaStar=bed_entrainment_sigma_star,
                surfaceDetachmentProbMode="gaussian_threshold",
                surfaceDetachmentProbSigmaStar=surface_detachment_sigma_star,
                TL_horizontal=float(settings.get("TL_horizontal", 2.0)),
                TL_vertical=float(settings.get("TL_vertical", 0.5)),
                release_times=release_times,
                stopWhenAllParticlesSurfaced=bool(settings.get("stopWhenAllParticlesSurfaced", False)),
                particleEvolution=particle_evolution,
                profile=solver_profile,
                trackOutputStride=output_stride,
            )
            solver_elapsed_s = perf_counter() - solver_start
            (
                ptrack,
                t_hist,
                tid_tr_track,
                track_idx,
                state_counts,
                x_end,
                y_end,
                z_end,
                tid_end,
                _,
                _,
                _,
                flags,
                _transport_state,
            ) = out
            x_tr = ptrack[:, 0, :]
            y_tr = ptrack[:, 1, :]
            z_tr = ptrack[:, 2, :]
        else:
            solver_start = perf_counter()
            out = advect_particles_euler_cell_transient_3d_ustar(
                mesh=mesh,
                hydro_cache=hydro_cache,
                zb_c=zb_c,
                P0=p0,
                t0=t0,
                t1=t1,
                dt=settings["dt"],
                outsidePolicy=settings["outsidePolicy"],
                dryPolicy=settings["dryPolicy"],
                hmin=settings["hmin"],
                useRWx=settings["useRWx"],
                useRWy=settings["useRWy"],
                betaKh=settings["betaKh"],
                KhMax=settings["KhMax"],
                ws=particle.ws,
                zFrac=settings["zFrac"],
                surfacePolicy=settings["surfacePolicy"],
                surfaceUstarCrit=surface_ustar_crit,
                dzUpMax=settings.get("dzUpMax", 0.4),
                uphillPolicy=settings["uphillPolicy"],
                useRWz=settings["useRWz"],
                alphaKz=settings["alphaKz"],
                KzMax=settings["KzMax"],
                bedPolicy=settings["bedPolicy"],
                rng=rng,
                nTrack=n_track,
                track_idx=track_idx,
                showProgress=True,
                hydraulicFieldMode=hydraulic_field_mode,
                transportVelocityMode=settings["transportVelocityMode"],
                hydraulicClosureModel=hydraulic_closure_model,
                hydraulicInterpolationMode=(
                    "step"
                    if adapter.adapter_name == "synthetic"
                    and settings.get("transient_shape") == "step"
                    else "linear"
                ),
                rAniso=r_aniso,
                entrainmentProbMode="gaussian_threshold",
                entrainmentProbSigmaStar=bed_entrainment_sigma_star,
                surfaceDetachmentProbMode="gaussian_threshold",
                surfaceDetachmentProbSigmaStar=surface_detachment_sigma_star,
                release_times=release_times,
                stopWhenAllParticlesSurfaced=bool(settings.get("stopWhenAllParticlesSurfaced", False)),
                transportModel=transport_model,
                TL_horizontal=float(settings.get("TL_horizontal", 2.0)),
                TL_vertical=float(settings.get("TL_vertical", 0.5)),
                particleEvolution=particle_evolution,
                profile=solver_profile,
                trackOutputStride=output_stride,
            )
            solver_elapsed_s = perf_counter() - solver_start
            x_tr = out["xTr"]
            y_tr = out["yTr"]
            z_tr = out["zTr"]
            tid_tr_track = out["tidTr_track"]
            track_idx = out["track_idx"]
            x_end = out["xEnd"]
            y_end = out["yEnd"]
            z_end = out["zEnd"]
            tid_end = out["tidEnd"]
            flags = out["flags"]
            zb_tr_track = out["zbTr_track"]
            wse_tr_track = out["wseTr_track"]
            t_hist = out["tHist"]
            state_counts = None
            h_final = interpolate_hydro_fields(hydro_cache, t1, settings["hmin"])
            wse_for_classification = h_final["wse_c"]

        n_solver_steps = max(int(round((t1 - t0) / float(settings["dt"]))), 0)
        solver_profile_summary = _summarize_solver_profile(solver_profile, solver_elapsed_s)
        if n_solver_steps > 0:
            simulation_time_per_timestep_ms = 1000.0 * solver_elapsed_s / n_solver_steps
            print(f"\nSimulation time per time step: {simulation_time_per_timestep_ms:.3f} ms")
        else:
            print("\nSimulation time per time step: n/a (no solver steps)")
        if solver_profile_summary and solver_profile_summary["top_sections"]:
            print("Solver profile summary:")
            for item in solver_profile_summary["top_sections"][:5]:
                print(
                    f"  {item['name']}: {item['seconds']:.3f} s "
                    f"({item['percent_of_solver']:.1f} % of solver)"
                )

        _print_vertical_stats(z_end)

        classification_geometry = {}
        if hydraulic_field_mode == "local_idw":
            final_points = np.column_stack([x_end, y_end])
            final_sampler = _LocalCellIDWFieldSampler(
                mesh,
                {
                    "zb": zb_c,
                    "wse": wse_for_classification,
                    "h": wse_for_classification - zb_c,
                },
            )
            sampled_final = final_sampler.sample_fields(tid_end, final_points, ("zb", "wse", "h"))
            classification_geometry = {
                "zb_end": sampled_final["zb"],
                "wse_end": sampled_final["wse"],
                "h_end": sampled_final["h"],
            }

        state = classify_end_state_all(
            x_end,
            y_end,
            z_end,
            tid_end,
            flags,
            zb_c,
            wse_for_classification,
            settings["hmin"],
            zClassTol=settings.get("classification_tol_factor", classification_tol_default) * particle.Dp,
            vertical_overlap_preference=_vertical_overlap_preference(particle, settings),
            **classification_geometry,
        )
        final_state = _build_final_state(state, len(x_end))

        if settings.get("export_data", True):
            export_all(
                outdir=run_file.parent / "output",
                xTr=x_tr,
                yTr=y_tr,
                zTr=z_tr,
                tidTr=tid_tr_track,
                track_idx=track_idx,
                dt=output_dt,
                xEnd=x_end,
                yEnd=y_end,
                zEnd=z_end,
                tidEnd=tid_end,
                final_state=final_state,
                flags=flags,
                state_counts=state_counts,
                state_count_times=t_hist if state_counts is not None else None,
                hydrolpt_version="HydroLPT v0.9",
                adapter_name=adapter.adapter_name,
                extra_run_info={
                    "run_mode": settings["run_mode"],
                    "transportVelocityMode": settings["transportVelocityMode"],
                    "n_release_centers": int(centers.shape[0]),
                    "n_particles_total": int(nparticles),
                    "t0": float(t0),
                    "t1": float(t1),
                    "particle_tracking_dt": float(settings["dt"]),
                    "output_dt": float(output_dt),
                    "output_stride_steps": int(output_stride),
                    "selectedBedPolicy": selected_bed_policy,
                    "bedPolicy": settings["bedPolicy"],
                    "bedEntrainmentSigmaStar": bed_entrainment_sigma_star,
                    "selectedSurfacePolicy": selected_surface_policy,
                    "surfacePolicy": settings["surfacePolicy"],
                    "surfaceDetachmentSigmaStar": surface_detachment_sigma_star,
                    "simulation_time_per_timestep_ms": simulation_time_per_timestep_ms,
                    "solver_profile": solver_profile_summary,
                },
            )
        else:
            print("\nData export disabled; skipping CSV/JSON output files.")

        if settings["run_mode"] == "steady":
            h_plot = frame.h_c
            ustar_plot = ustar_c
            ustar_crit_plot = ustar_crit_c
            wse_plot = frame.wse_c
        else:
            h_plot = h_rel["h_c"]
            ustar_plot = h_rel["ustar_c"]
            ustar_crit_plot = h_rel["ustarCrit_c"]
            wse_plot = h_final["wse_c"]

        x_axis_mode = plots.get("xAxisMode", "time" if settings["run_mode"] == "transient" else "distance")
        n_plot = int(plots.get("nPlot", 6))
        plot_sel = rng.choice(x_tr.shape[0], size=min(n_plot, x_tr.shape[0]), replace=False)
        plot_sel.sort()
        x_tr_plot = x_tr[plot_sel, :]
        y_tr_plot = y_tr[plot_sel, :]
        z_tr_plot = z_tr[plot_sel, :]
        tid_tr_plot = tid_tr_track[plot_sel, :]
        track_idx_plot = track_idx[plot_sel]

        if plots.get("binary_map", True):
            print("\nBuilding binary map figure...")
            if np.ndim(surface_ustar_crit) == 0:
                surface_ustar_crit_plot = np.full_like(ustar_plot, float(surface_ustar_crit), dtype=float)
            else:
                surface_ustar_crit_plot = np.asarray(surface_ustar_crit, dtype=float).ravel()
            make_figure_binary_map(
                mesh,
                xy,
                tri0,
                frame.U_c if settings["run_mode"] == "steady" else h_rel["U_c"],
                frame.V_c if settings["run_mode"] == "steady" else h_rel["V_c"],
                ustar_plot,
                ustar_crit_plot,
                surface_ustar_crit_plot,
                h_plot,
                settings["hmin"],
                settings["rho_f"],
                particle.rho_p,
                p0,
                state,
                x_end,
                y_end,
                flags,
                tidEnd=tid_end,
                xTr_track=x_tr_plot,
                yTr_track=y_tr_plot,
                track_idx=track_idx_plot,
                marker_size=float(plots.get("binary_map_marker_size", plots.get("binary_map_marker_scale", 5.0))),
                window_anchor=plots.get("_window_anchor"),
                hydraulicFieldMode=hydraulic_field_mode,
                mesh_overlay=plots.get("binary_map_mesh_overlay"),
            )

        if plots.get("shields", False):
            print("\nBuilding Shields figure...")
            make_figure_shields(
                d50_c,
                ustar_plot,
                ustar_crit_plot,
                h_plot,
                settings["hmin"],
                settings["nu"],
                settings["rho_f"],
                particle.rho_p,
                settings["g"],
                particle.Dp,
                particle.beta_p,
                settings["tanphi_ratio"],
            )

        if plots.get("trajectories", True):
            print("\nBuilding trajectory figure...")
            shared_marker_size = float(
                plots.get("binary_map_marker_size", plots.get("binary_map_marker_scale", 5.0))
            )
            make_figure_trajectories_tracked(
                x_tr_plot,
                y_tr_plot,
                z_tr_plot,
                tid_tr_plot,
                zb_c,
                wse_plot,
                flags,
                state,
                track_idx=track_idx_plot,
                nPlot=n_plot,
                zbTr_track=zb_tr_track[plot_sel, :] if zb_tr_track is not None else None,
                wseTr_track=wse_tr_track[plot_sel, :] if wse_tr_track is not None else None,
                tHist=t_hist,
                xAxisMode=x_axis_mode,
                depositedAliveInPlot=(settings["run_mode"] == "transient"),
                dryAliveInPlot=(settings["run_mode"] == "transient"),
                xEnd=x_end,
                yEnd=y_end,
                zEnd=z_end,
                marker_size=shared_marker_size,
                mesh=mesh,
                hydraulicFieldMode=hydraulic_field_mode,
            )

        if enable_rouse:
            rouse_cfg = plots.get("rouse_profile")
            if rouse_cfg:
                print("\nBuilding Rouse profile and convergence figures...")
                rouse_cfg = dict(rouse_cfg)
                if "length" in settings:
                    rouse_cfg["flume_length"] = float(settings["length"])
                make_figure_rouse_profile_section(
                    xTr=x_tr,
                    zTr=z_tr,
                    tidTr=tid_tr_track,
                    zb_c=zb_c,
                    wse_c=wse_plot,
                    ustar_c=ustar_plot,
                    u_c=frame.U_c if settings["run_mode"] == "steady" else h_rel["U_c"],
                    ws=particle.ws,
                    hmin=settings["hmin"],
                    z_frac=settings["zFrac"],
                    x_release=float(np.mean(centers[:, 0])),
                    **rouse_cfg,
                )

        if enable_advection_diffusion:
            ad_cfg = plots.get("advection_diffusion")
            if ad_cfg:
                print("\nBuilding advection-diffusion diagnostics...")
                kx = _compute_kx(h_plot, ustar_plot, settings["betaKh"], settings["KhMax"], r_aniso)
                mean_u = ad_cfg.get("U", settings.get("U", 0.0))
                make_figure_advection_diffusion(
                    xTr=x_tr,
                    tHist=t_hist,
                    U=mean_u,
                    Kx=ad_cfg.get("Kx", kx),
                    transport_model=settings.get("transportModel", "random_walk"),
                    TL_horizontal=settings.get("TL_horizontal", 2.0),
                    t_eval=ad_cfg["t_eval"],
                    nbins=ad_cfg.get("nbins", "sturges"),
                )

        print("\nSimulation finished")
        if show_plots:
            _apply_plot_window_icons()
            _apply_plot_window_size(plots.get("_window_anchor"))
            _position_plot_windows(plots.get("_window_anchor"))
            if block_on_plots:
                print("Opening plots...")
                plt.show()
                print("All figures closed")
        else:
            print("Plot display skipped by caller.")
        return np.asarray(centers, dtype=float)
    finally:
        if adapter is not None and hasattr(adapter, "close"):
            adapter.close()

