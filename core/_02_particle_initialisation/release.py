from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from core._06_output_diagnostics.plotting import (
    FIGURE1_TITLE_FONTSIZE,
    LEGEND_STYLE,
    _axis_bounds,
    _binary_map_figure_size,
    _compact_tick,
    _estimate_legend_width_inches,
    _estimate_text_block_height_inches,
    _estimate_text_width_inches,
    _flume_x_tick_values,
)


ICON_PNG_PATH = Path("assets") / "hydrolpt_icon_option2c.png"
ICON_ICO_PATH = Path("assets") / "hydrolpt_icon_option2c.ico"


def _runtime_asset_path(relative_path: Path) -> Path:
    """Resolve an asset path for both source and frozen application runs."""
    if getattr(sys, "frozen", False):
        runtime_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    else:
        runtime_root = Path(__file__).resolve().parents[2]
    return runtime_root / relative_path


def _apply_release_picker_window_icon(fig) -> None:
    """Apply the HydroLPT icon to the interactive release picker window."""
    ico_path = _runtime_asset_path(ICON_ICO_PATH)
    manager = getattr(fig.canvas, "manager", None)
    window = getattr(manager, "window", None)
    if window is None:
        return
    try:
        if ico_path.exists() and hasattr(window, "iconbitmap"):
            window.iconbitmap(str(ico_path))
    except Exception:
        pass


def pick_points_interactive(fig, ax, max_points: int | None = None):
    """Interactively pick horizontal release points from a figure."""
    pts = []
    cancelled = {"flag": False}
    done = {"flag": False}

    sc = ax.scatter([], [], s=60, marker="o", edgecolors="k", facecolors="y")

    def _on_click(ev):
        if done["flag"] or cancelled["flag"]:
            return
        if ev.inaxes != ax:
            return
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return
        if ev.button == 1 and ev.xdata is not None and ev.ydata is not None:
            pts.append((float(ev.xdata), float(ev.ydata)))
            sc.set_offsets(np.asarray(pts, dtype=float))
            fig.canvas.draw_idle()
            if max_points is not None and len(pts) >= max_points:
                done["flag"] = True

    def _on_key(ev):
        if ev.key == "enter":
            done["flag"] = True
        elif ev.key == "escape":
            cancelled["flag"] = True
            done["flag"] = True
        elif ev.key == "backspace" and pts:
            pts.pop()
            sc.set_offsets(np.asarray(pts, dtype=float) if pts else np.empty((0, 2)))
            fig.canvas.draw_idle()

    def _on_close(_ev):
        cancelled["flag"] = True
        done["flag"] = True

    cid_click = fig.canvas.mpl_connect("button_press_event", _on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", _on_key)
    cid_close = fig.canvas.mpl_connect("close_event", _on_close)

    while not done["flag"]:
        if not plt.fignum_exists(fig.number):
            cancelled["flag"] = True
            done["flag"] = True
            break
        plt.pause(0.05)

    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)
    fig.canvas.mpl_disconnect(cid_close)

    if cancelled["flag"] or not pts:
        return np.empty((0, 2), dtype=float)
    return np.asarray(pts, dtype=float)


def filter_release_centers(mesh, h_c, hmin: float, pts):
    """Keep only release points that lie inside the mesh and in wet cells."""
    pts = np.asarray(pts, dtype=float)
    if pts.size == 0:
        return np.empty((0, 2), dtype=float)

    tid = mesh.point_location(pts[:, 0], pts[:, 1])
    inside = tid >= 0
    pts = pts[inside]
    tid = tid[inside]

    if pts.size == 0:
        return np.empty((0, 2), dtype=float)

    h_c = np.asarray(h_c, dtype=float)
    wet = np.isfinite(h_c[tid]) & (h_c[tid] > hmin)
    return pts[wet]


def build_release_clouds(centers_xy, n_per_center=100, sigma=2.0, rng=None, seed=1):
    """Build Gaussian particle clouds around each release center."""
    centers_xy = np.asarray(centers_xy, dtype=float)
    if centers_xy.size == 0:
        return np.empty((0, 2), dtype=float)
    if rng is None:
        rng = np.random.default_rng(seed)
    clouds = [
        center + sigma * rng.standard_normal((n_per_center, 2))
        for center in centers_xy
    ]
    return np.vstack(clouds)


def build_release_positions_3d(Pxy, z0=np.nan):
    """Convert horizontal release coordinates into full particle positions."""
    Pxy = np.asarray(Pxy, dtype=float)
    if Pxy.ndim != 2 or Pxy.shape[1] != 2:
        raise ValueError("Pxy must be (Np,2)")

    Np = Pxy.shape[0]
    if Np == 0:
        return np.empty((0, 3), dtype=float)

    if np.isscalar(z0):
        z = np.full(Np, float(z0), dtype=float)
    else:
        z = np.asarray(z0, dtype=float).ravel()
        if z.size != Np:
            raise ValueError("z0 array must have length Np")

    return np.column_stack([Pxy[:, 0], Pxy[:, 1], z])


def build_release_positions_3d_gaussian_vertical(
    Pxy,
    *,
    mesh,
    zb_c,
    wse_c,
    hmin: float,
    z_frac: float,
    sigma_z: float,
    rng=None,
    seed=1,
):
    """Build 3D release positions with a Gaussian vertical spread."""
    Pxy = np.asarray(Pxy, dtype=float)
    if Pxy.ndim != 2 or Pxy.shape[1] != 2:
        raise ValueError("Pxy must be (Np,2)")
    if Pxy.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    if not (0.0 <= float(z_frac) <= 1.0):
        raise ValueError("z_frac must be between 0 and 1.")
    if float(sigma_z) < 0.0:
        raise ValueError("sigma_z must be >= 0.")

    if rng is None:
        rng = np.random.default_rng(seed)

    tid = mesh.point_location(Pxy[:, 0], Pxy[:, 1])
    if np.any(tid < 0):
        raise ValueError("All release points must lie inside the mesh for vertical initialization.")

    zb_c = np.asarray(zb_c, dtype=float).ravel()
    wse_c = np.asarray(wse_c, dtype=float).ravel()
    zb = zb_c[tid]
    wse = wse_c[tid]
    depth = wse - zb

    valid_depth = np.isfinite(depth) & (depth > float(hmin)) & np.isfinite(zb) & np.isfinite(wse)
    if not np.all(valid_depth):
        raise ValueError("All release points must lie in wet cells for Gaussian vertical initialization.")

    z_center = zb + float(z_frac) * depth
    z = z_center if float(sigma_z) == 0.0 else z_center + float(sigma_z) * rng.standard_normal(Pxy.shape[0])

    z_min = zb + 1e-6 * depth
    z_max = wse - 1e-6 * depth
    z = np.clip(z, z_min, z_max)
    return np.column_stack([Pxy[:, 0], Pxy[:, 1], z])


def build_release_schedule_per_center(
    n_centers: int,
    n_per_center: int,
    *,
    t_release: float,
    release_mode: str = "bulk",
    release_dt: float = 1.0,
):
    """Build one release time per particle, grouped by release center."""
    n_centers = int(n_centers)
    n_per_center = int(n_per_center)
    t_release = float(t_release)
    release_dt = float(release_dt)
    release_mode = str(release_mode).strip().lower()
    if release_mode == "instantaneous":
        release_mode = "bulk"

    if n_centers <= 0 or n_per_center <= 0:
        return np.empty(0, dtype=float)
    if release_mode == "bulk":
        return np.full(n_centers * n_per_center, t_release, dtype=float)
    if release_mode != "continuous":
        raise ValueError("release_mode must be 'bulk' or 'continuous'")
    if release_dt <= 0.0:
        raise ValueError("release_dt must be > 0 for continuous release")

    one_center = t_release + release_dt * np.arange(n_per_center, dtype=float)
    return np.tile(one_center, n_centers)


def get_release_centers(mesh, scalar_c, scalar_name: str, XY, tri0, h_c, hmin: float, window_anchor: dict | None = None):
    """Pick release centers interactively from a map and keep only wet in-domain points."""
    XY = np.asarray(getattr(mesh, "xy", XY), dtype=float)
    x_nodes = XY[:, 0]
    y_nodes = XY[:, 1]
    use_wide_flume_layout, _tall_planform, _figsize, _Lx, _Ly = _binary_map_figure_size(x_nodes, y_nodes)

    if use_wide_flume_layout:
        x_plot = x_nodes
        y_plot = y_nodes
        xlabel = "x (m)"
        ylabel = "y (m)"
    else:
        x_plot = x_nodes
        y_plot = y_nodes
        xlabel = "x"
        ylabel = "y"

    title_text = "Pick release locations\nLeft click = add | ENTER = finish | ESC = cancel"
    legend_labels = ["Release point"]

    fig = plt.figure(dpi=100)
    if use_wide_flume_layout:
        map_width_inches = 6.4
        map_height_inches = 2.4
        x_min, x_max = _axis_bounds(x_plot)
        y_tick_values = np.asarray([0.0, 0.2, 0.8, 1.0], dtype=float)
        x_tick_values = _flume_x_tick_values(x_min, x_max)
    else:
        map_height_inches = 5.8
        display_ratio = max(float(np.ptp(y_plot[np.isfinite(y_plot)])), 1e-12) / max(float(np.ptp(x_plot[np.isfinite(x_plot)])), 1e-12)
        map_width_inches = map_height_inches / max(display_ratio, 1e-12)
        y_min = float(np.nanmin(y_plot))
        y_max = float(np.nanmax(y_plot))
        y_tick_values = np.linspace(y_min, y_max, 6)
        x_min = float(np.nanmin(x_plot))
        x_max = float(np.nanmax(x_plot))
        x_tick_values = np.linspace(x_min, x_max, 6)

    pad_inches = 0.4
    y_tick_labels = [_compact_tick(v, None) for v in y_tick_values]
    legend_cell_width_inches = max(
        0.7,
        _estimate_legend_width_inches(legend_labels, title="Markers", fontsize=LEGEND_STYLE["fontsize"]) + 0.02,
    )
    legend_gap_inches = 0.10
    tick_width_inches = _estimate_text_width_inches(y_tick_labels, fontsize=10.0, extra_pad=0.08)
    ylabel_width_inches = (10.0 / 72.0) * 1.6
    yaxis_cell_width_inches = tick_width_inches + ylabel_width_inches + 0.12
    xtick_height_inches = (10.0 / 72.0) * 1.25
    xlabel_height_inches = _estimate_text_block_height_inches(xlabel, fontsize=10.0, line_spacing=1.0, pad=0.02)
    xaxis_cell_height_inches = xtick_height_inches + xlabel_height_inches + 0.08
    title_cell_height_inches = _estimate_text_block_height_inches(title_text, fontsize=FIGURE1_TITLE_FONTSIZE, line_spacing=1.0, pad=0.04)
    title_gap_inches = 0.10
    spacer_cell_height_inches = 0.12
    cbar_label_text = scalar_name
    cbar_tick_height_inches = (10.0 / 72.0) * 1.25
    cbar_label_height_inches = _estimate_text_block_height_inches(cbar_label_text, fontsize=10.0, line_spacing=1.0, pad=0.02)
    cbar_cell_height_inches = 0.22 + cbar_tick_height_inches + cbar_label_height_inches + 0.06
    cbar_bar_cell_height_inches = 0.5 * cbar_cell_height_inches
    cbar_text_cell_height_inches = cbar_tick_height_inches + cbar_label_height_inches + 0.08
    fig_height = map_height_inches + title_gap_inches + title_cell_height_inches + xaxis_cell_height_inches + spacer_cell_height_inches + cbar_bar_cell_height_inches + cbar_text_cell_height_inches + 2.0 * pad_inches
    side_pad_inches = 0.4 if use_wide_flume_layout else pad_inches
    fig_width = legend_cell_width_inches + legend_gap_inches + map_width_inches + yaxis_cell_width_inches + 2.0 * side_pad_inches
    fig.set_size_inches(fig_width, fig_height, forward=True)

    _apply_release_picker_window_icon(fig)

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
    title_ax = fig.add_axes(title_pos)
    legend_ax = fig.add_axes(legend_pos)
    yaxis_ax = fig.add_axes(yaxis_pos, sharey=ax)
    cbar_ax = fig.add_axes(cbar_pos)
    cbar_label_ax = fig.add_axes(cbar_label_pos)

    title_ax.patch.set_alpha(0.0)
    title_ax.axis("off")
    legend_ax.patch.set_alpha(0.0)
    legend_ax.axis("off")
    yaxis_ax.patch.set_alpha(0.0)
    yaxis_ax.xaxis.set_visible(False)
    yaxis_ax.yaxis.tick_right()
    yaxis_ax.yaxis.set_label_position("right")
    yaxis_ax.spines["top"].set_visible(False)
    yaxis_ax.spines["bottom"].set_visible(False)
    yaxis_ax.spines["left"].set_visible(False)
    yaxis_ax.spines["right"].set_position(("axes", 0.0))
    yaxis_ax.spines["right"].set_visible(False)
    cbar_label_ax.patch.set_alpha(0.0)
    cbar_label_ax.axis("off")

    ax.set_aspect("auto" if use_wide_flume_layout else "equal", adjustable="datalim")

    plot_c = np.asarray(scalar_c, dtype=float).copy()
    plot_c[(~np.isfinite(plot_c)) | (np.asarray(h_c, dtype=float) <= hmin)] = np.nan
    if getattr(mesh, "is_native_hecras", False):
        polygons = []
        polygon_values = []
        for cell_id, poly_ids in enumerate(getattr(mesh, "_polygons", [])):
            poly_ids = np.asarray(poly_ids, dtype=int)
            if poly_ids.size < 3:
                continue
            polygons.append(XY[poly_ids])
            polygon_values.append(plot_c[cell_id] if cell_id < plot_c.size else np.nan)
        if not polygons:
            raise ValueError("Cannot draw release picker: HEC-RAS mesh has no valid cell polygons.")
        tpc = PolyCollection(polygons, array=np.asarray(polygon_values, dtype=float), edgecolors="none")
        ax.add_collection(tpc)
        x_min, x_max = _axis_bounds(x_plot)
        y_min, y_max = _axis_bounds(y_plot)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        tpc = ax.tripcolor(x_plot, y_plot, tri0, facecolors=plot_c, shading="flat")
    cb = fig.colorbar(tpc, cax=cbar_ax, orientation="horizontal")
    cb.set_label(cbar_label_text)
    cb.ax.tick_params(axis="x", bottom=True, labelbottom=True)

    bd = mesh.free_boundary_edges()
    bdxy0 = XY[bd[:, 0]]
    bdxy1 = XY[bd[:, 1]]
    for a, b in zip(bdxy0, bdxy1):
        ax.plot([a[0], b[0]], [a[1], b[1]], "k-", linewidth=1.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    yaxis_ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(FuncFormatter(_compact_tick))
    yaxis_ax.yaxis.set_major_formatter(FuncFormatter(_compact_tick))
    release_handle = Line2D([0], [0], marker="o", linestyle="None", markeredgecolor="k", markerfacecolor="y", markersize=8, label="Release point")
    legend_ax.legend(handles=[release_handle], loc="upper left", bbox_to_anchor=(0.0, 1.0), **LEGEND_STYLE)
    title_ax.text(0.5, 0.5, title_text, ha="center", va="center", fontsize=FIGURE1_TITLE_FONTSIZE)
    ax.set_xticks(x_tick_values)
    yaxis_ax.set_yticks(y_tick_values)
    ax.margins(x=0.0, y=0.0)
    if use_wide_flume_layout:
        ax.set_xlim(x_tick_values[0], x_tick_values[-1])
        ax.set_ylim(0.0, 1.0)

    pts = pick_points_interactive(fig, ax)
    plt.close(fig)
    pts = filter_release_centers(mesh, h_c, hmin, pts)
    print(f"Accepted release centers: {pts.shape[0]}")
    return pts
