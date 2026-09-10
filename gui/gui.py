"""Tkinter desktop GUI for configuring and launching HydroLPT cases."""

from __future__ import annotations

import ast
import copy
import ctypes
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from tkinter import font as tkfont
from typing import Any

from core.app import default_output_dir, execute_case_spec
from core._00_case_management import (
    GUI_CASE_FILENAME,
    CaseSchema,
    build_blueprint_from_case,
    case_schema_from_payload,
    case_schema_to_payload,
    default_case_blueprints,
    load_case_definition,
    normalize_case_spec,
    save_case_file,
)
from core._02_particle_initialisation import build_particle_properties
from core._01_hydraulic_input_and_adapter.basement import BasementAdapter
from core._01_hydraulic_input_and_adapter.hecras import HecRasAdapter


RUN_SPEC_FLAG = "--run-gui-spec"
ICON_PNG_PATH = Path("assets") / "hydrolpt_icon_option2c.png"
ICON_ICO_PATH = Path("assets") / "hydrolpt_icon_option2c.ico"
WINDOWS_APP_ID = "HydroLPT.v0.9"
SELECTOR_CONTROL_WIDTH = 20
PANEL_RIGHT_MARGIN = 14
ACTION_BAR_RIGHT_MARGIN = 17
CASE_LABELS = {
    "synthetic": "Synthetic Flume",
    "basement": "BASEMENT HPC",
    "hecras": "HEC-RAS 2D",
}
WORKFLOW_LABELS = {
    "new": "New Simulation",
    "base": "Existing Simulation",
}
CLICK_RELEASE_PLACEHOLDER = "click in plot"

LOCKED_TRANSPORT_KEYS = {"KhMax", "KzMax"}
LOCKED_FLUID_KEYS = {"rho_f", "nu", "g"}
LOCKED_GEOMETRY_KEYS = {"domain_width", "wetted_width"}
LOCKED_RELEASE_KEYS: set[str] = set()
HIDDEN_SECTION_FIELDS = {
    "PARTICLE_SETTINGS": {"rng_seed"},
}
HYDRAULIC_INPUT_KEYS = {
    "length",
    "domain_width",
    "wetted_width",
    "water_depth",
    "nx",
    "ny",
    "hydraulicPrimaryVariable",
    "U",
    "V",
    "z0",
    "transient_shape",
    "switch_fraction",
    "U0",
    "U1",
    "V0",
    "V1",
}
SYNTHETIC_LENGTH_OPTIONS = ["10", "15", "20", "25", "30", "40", "50"]
SYNTHETIC_WATER_DEPTH_OPTIONS = [f"{0.1 * k:.1f}" for k in range(1, 11)]
FIXED_SYNTHETIC_CELL_SIZE_X = 40.0 / 800.0
FIXED_SYNTHETIC_CELL_SIZE_Y = 1.0 / 20.0
PARTICLE_DIMENSION_FIELDS = ("L", "I", "S")
PARTICLE_SHAPE_DIMENSIONS = {
    "sphere": ("L",),
    "ellipsoid": ("L", "I", "S"),
    "cylinder": ("L", "S"),
    "disk": ("L", "S"),
    "prism": ("L", "I", "S"),
}
PARTICLE_DIMENSION_LABELS = {
    "sphere": {
        "L": "Diameter D (m)",
        "I": "Intermediate axis I (m)",
        "S": "Short axis S (m)",
    },
    "ellipsoid": {
        "L": "Long axis L (m)",
        "I": "Intermediate axis I (m)",
        "S": "Short axis S (m)",
    },
    "cylinder": {
        "L": "Length L (m)",
        "I": "Intermediate axis I (m)",
        "S": "Diameter S (m)",
    },
    "disk": {
        "L": "Diameter L (m)",
        "I": "Intermediate axis I (m)",
        "S": "Thickness S (m)",
    },
    "prism": {
        "L": "Length L (m)",
        "I": "Intermediate axis I (m)",
        "S": "Short axis S (m)",
    },
}

OPTION_VALUES = {
    "run_mode": ["steady", "transient"],
    "location_mode": ["coordinates", "click"],
    "release_mode": ["bulk", "continuous"],
    "hydraulicFieldMode": ["cellwise", "local_idw"],
    "transportModel": ["random_walk", "langevin"],
    "rng_seed": ["None", "1"],
    "transportVelocityMode": ["depth_averaged", "loglaw_vertical"],
    "hydraulicPrimaryVariable": ["U"],
    "shape": ["sphere", "ellipsoid", "cylinder", "disk", "prism"],
    "surfacePolicy": [
        "always_reflect",
        "always_stick",
        "deterministic_detachment",
        "probabilistic_detachment",
    ],
    "bedPolicy": [
        "always_reflect",
        "always_deposit",
        "reflect_if_ustar_gt_crit",
        "probabilistic_entrainment",
    ],
    "outsidePolicy": ["stop"],
    "dryPolicy": ["reflect", "stop", "stick_active"],
    "uphillPolicy": ["off", "stop"],
    "transient_shape": ["step", "linear"],
    "biofouling": ["off", "on"],
    "degradation": ["off", "on"],
}

OPTION_DISPLAY_LABELS = {
    "bedPolicy": {
        "always_reflect": "always reflect",
        "always_deposit": "always deposit",
        "reflect_if_ustar_gt_crit": "deterministic entrainment",
        "probabilistic_entrainment": "probabilistic entrainment",
    },
    "dryPolicy": {
        "reflect": "reflect",
        "stop": "stop",
        "stick_active": "stick active",
    }
}

FIELD_LABELS = {
    "xdmf_path": "Hydraulic results file",
    "run_mode": "Run mode",
    "location_mode": "Location mode",
    "release_mode": "Release mode",
    "transportVelocityMode": "Velocity mode",
    "hydraulicPrimaryVariable": "Primary hydraulic input",
    "hydraulicClosureModel": "Hydraulic closure model",
    "dt": "Time step dt (s)",
    "tTrack": "Tracking duration T_track (s)",
    "tRelease": "Release time t_release (s)",
    "output_dt": "Output timestep dt_out (s)",
    "hydraulicFieldMode": "Hydraulic field mode",
    "hyd_time_start": "Start time t_start (s)",
    "hyd_dt": "Time step dt_h (s)",
    "hyd_time_end": "End time t_end (s)",
    "length": "Flume length L_f (m)",
    "domain_width": "Domain width B (m)",
    "wetted_width": "Wetted width B_w (m)",
    "water_depth": "Water depth h (m)",
    "nx": "Cells in x: n_x",
    "ny": "Cells in y: n_y",
    "rho_f": "Fluid density ρ_f (kg/m³)",
    "nu": "Kinematic viscosity ν (m²/s)",
    "g": "Gravity g (m/s²)",
    "shape": "Particle shape",
    "L": "Length L (m)",
    "I": "Intermediate axis I (m)",
    "S": "Short axis S (m)",
    "rho_p": "Particle density ρ_p (kg/m³)",
    "tanphi_ratio": "Slope ratio tanφ_p / tanφ_s",
    "release_x": "Release x-coordinate(s) x_r (m)",
    "release_y": "Release y-coordinate(s) y_r (m)",
    "n_per_center": "Number of particles N (per release)",
    "release_dt": "Inter-release time dT (s)",
    "nTrack": "Tracked particles N_track (per release)",
    "rng_seed": "Random seed",
    "releaseSigma": "Release spread σ_r (m)",
    "zFrac": "Release elevation z/H",
    "useRWx": "Random walk in x",
    "useRWy": "Random walk in y",
    "useRWz": "Random walk in z",
    "betaKh": "Horizontal mixing β_Kh",
    "KhMax": "Max horizontal diffusivity K_h,max (m²/s)",
    "rAniso": "Horizontal anisotropy ratio r_aniso",
    "alphaKz": "Vertical mixing α_Kz",
    "KzMax": "Max vertical diffusivity K_z,max (m²/s)",
    "surfacePolicy": "Surface policy",
    "bedPolicy": "Bed policy",
    "surfaceVerticalDragCoeff": "Surface drag coefficient C_f (-)",
    "surfaceContactAngleDeg": "Contact angle Omega (deg)",
    "surfaceDetachmentSigmaStar": "Surface detachment sigma* (-)",
    "bedEntrainmentSigmaStar": "Bed entrainment sigma* (-)",
    "dryPolicy": "Dry-cell policy",
    "outsidePolicy": "Outside-domain policy",
    "uphillPolicy": "Uphill policy",
    "hmin": "Wet threshold h_min (m)",
    "dzUpMax": "Max uphill step Δz_up,max (m)",
    "U": "Mean Velocity U (m/s)",
    "V": "Mean Velocity V (m/s)",
    "z0": "Roughness height z0 (m)",
    "transient_shape": "Transition shape",
    "switch_fraction": "Switch fraction t/T",
    "U0": "Initial U_0 (m/s)",
    "U1": "Final U_1 (m/s)",
    "V0": "Initial V_0 (m/s)",
    "V1": "Final V_1 (m/s)",
    "binary_map": "Particle map",
    "binary_map_marker_size": "Marker size",
    "binary_map_mesh_overlay": "Mesh overlay",
    "trajectories": "Trajectories",
    "biofouling": "Biofouling",
    "BT0": "Initial biofilm thickness BT_0 (m)",
    "BR": "Biofilm growth rate BR (m/s)",
    "rho_biofilm": "Biofilm density rho_b (kg/m^3)",
    "degradation": "Degradation",
    "DR": "Degradation rate DR (%/s)",
}

FIELD_LABELS.update(
    {
        "transportModel": "Transport model",
        "rho_f": "Fluid density rho_f (kg/m^3)",
        "nu": "Kinematic viscosity nu (m^2/s)",
        "g": "Gravity g (m/s^2)",
        "rho_p": "Particle density rho_p (kg/m^3)",
        "tanphi_ratio": "Slope ratio tanphi_p / tanphi_s",
        "releaseSigma": "Release spread sigma_r (m)",
        "betaKh": "Horizontal mixing beta_Kh",
        "KhMax": "Max horizontal diffusivity K_h,max (m^2/s)",
        "rAniso": "Horizontal anisotropy ratio r_aniso",
        "alphaKz": "Vertical mixing alpha_Kz",
        "KzMax": "Max vertical diffusivity K_z,max (m^2/s)",
        "TL_horizontal": "Langevin T_L,h (s)",
        "TL_vertical": "Langevin T_L,v (s)",
        "dzUpMax": "Max uphill step dz_up,max (m)",
    }
)

FIELD_HELPERS = {
    ("HYDRAULIC_INPUT", "z0"): "Typical 1e-4-1e-2 m",
    ("TRANSPORT_SETTINGS", "dt"): "Typical 0.01-1.0 s",
    ("PLOTS", "output_dt"): "Must be dt, 2*dt, 3*dt, ...",
    ("TRANSPORT_SETTINGS", "hydraulicFieldMode"): "cellwise or local_idw",
    ("TRANSPORT_SETTINGS", "rng_seed"): "None gives a fresh run; use an integer to reproduce release and random-walk paths",
    ("PARTICLE_SETTINGS", "releaseSigma"): "Typical 0-0.1 m",
    ("TRANSPORT_SETTINGS", "betaKh"): "Typical 0.1-1.0",
    ("TRANSPORT_SETTINGS", "KhMax"): "Typical 0.01-5.0 m^2/s",
    ("TRANSPORT_SETTINGS", "rAniso"): "1.0 isotropic; 3.0 gives stronger along-flow than cross-flow diffusion",
    ("TRANSPORT_SETTINGS", "alphaKz"): "Typical 0.1-1.0",
    ("TRANSPORT_SETTINGS", "KzMax"): "Typical 0.001-0.3 m^2/s",
    ("TRANSPORT_SETTINGS", "TL_horizontal"): "Typical 1-10 s",
    ("TRANSPORT_SETTINGS", "TL_vertical"): "Typical 0.1-2 s",
    ("BOUNDARY_SETTINGS", "tanphi_ratio"): "Typical 0.55",
    ("BOUNDARY_SETTINGS", "surfaceDetachmentSigmaStar"): "Default 0.2; spread of particle-scale surface resistance thresholds",
    ("BOUNDARY_SETTINGS", "bedEntrainmentSigmaStar"): "Default 0.2; spread of particle-scale bed resistance thresholds",
    ("BOUNDARY_SETTINGS", "surfaceVerticalDragCoeff"): "Typical 0.5-2.0",
    ("BOUNDARY_SETTINGS", "surfaceContactAngleDeg"): "Default 105 deg; typical hydrophobic contact angle",
    ("BOUNDARY_SETTINGS", "hmin"): "Typical 1e-4-1e-2 m",
    ("BOUNDARY_SETTINGS", "dzUpMax"): "Typical 0.01-0.5 m",
    ("PARTICLE_EVO_SETTINGS", "BT0"): "Typical 0-1e-4 m",
    ("PARTICLE_EVO_SETTINGS", "BR"): "Typical 1e-10-1e-7 m/s",
    ("PARTICLE_EVO_SETTINGS", "rho_biofilm"): "Typical 1000-1500 kg/m^3",
    ("PARTICLE_EVO_SETTINGS", "DR"): "Typical 1e-6-1e-2 %/s",
}

SECTION_LAYOUTS = {
    "HYDRAULIC_INPUT": [
        ("header", "Geometry"),
        ("field", "length"),
        ("field", "domain_width"),
        ("field", "wetted_width"),
        ("field", "water_depth"),
        ("header", "Hydraulic Inputs"),
        ("field", "run_mode"),
        ("field", "hydraulicPrimaryVariable"),
        ("field", "U"),
        ("field", "hydraulicClosureModel"),
        ("field", "z0"),
        ("field", "transient_shape"),
        ("field", "switch_fraction"),
        ("field", "U0"),
        ("field", "U1"),
        ("header", "Time settings"),
        ("field", "hyd_time_start"),
        ("field", "hyd_dt"),
        ("field", "hyd_time_end"),
        ("header", "Fluid properties"),
        ("field", "rho_f"),
        ("field", "nu"),
        ("field", "g"),
        ("header", "Mesh properties"),
        ("field", "nx"),
        ("field", "ny"),
        ("field", "V0"),
        ("field", "V1"),
    ],
    "PARTICLE_SETTINGS": [
        ("header", "Particle properties"),
        ("field", "shape"),
        ("field", "L"),
        ("field", "I"),
        ("field", "S"),
        ("field", "rho_p"),
        ("header", "Particle release"),
        ("field", "release_mode"),
        ("field", "release_dt"),
        ("field", "location_mode"),
        ("field", "release_x"),
        ("field", "release_y"),
        ("field", "n_per_center"),
        ("field", "nTrack"),
        ("field", "releaseSigma"),
        ("field", "zFrac"),
    ],
    "PARTICLE_EVO_SETTINGS": [
        ("header", "Particle evolution"),
        ("field", "biofouling"),
        ("field", "BT0"),
        ("field", "BR"),
        ("field", "rho_biofilm"),
        ("field", "degradation"),
        ("field", "DR"),
    ],
    "TRANSPORT_SETTINGS": [
        ("header", "Time settings"),
        ("field", "tRelease"),
        ("field", "dt"),
        ("field", "tTrack"),
        ("header", "Advection and dispersion"),
        ("field", "hydraulicFieldMode"),
        ("field", "transportVelocityMode"),
        ("field", "transportModel"),
        ("field", "rng_seed"),
        ("field", "useRWx"),
        ("field", "useRWy"),
        ("field", "useRWz"),
        ("field", "betaKh"),
        ("field", "KhMax"),
        ("field", "rAniso"),
        ("field", "alphaKz"),
        ("field", "KzMax"),
        ("field", "TL_horizontal"),
        ("field", "TL_vertical"),
    ],
    "BOUNDARY_SETTINGS": [
        ("header", "Boundary interaction framework"),
        ("field", "surfacePolicy"),
        ("field", "surfaceDetachmentSigmaStar"),
        ("field", "surfaceVerticalDragCoeff"),
        ("field", "surfaceContactAngleDeg"),
        ("field", "bedPolicy"),
        ("field", "bedEntrainmentSigmaStar"),
        ("field", "tanphi_ratio"),
        ("field", "dryPolicy"),
        ("field", "hmin"),
        ("field", "outsidePolicy"),
        ("field", "uphillPolicy"),
        ("field", "dzUpMax"),
    ],
    "PLOTS": [
        ("field", "output_dt"),
        ("header", "Plotting"),
        ("field", "binary_map"),
        ("field", "trajectories"),
        ("field", "binary_map_marker_size"),
        ("field", "binary_map_mesh_overlay"),
    ],
}


def _parse_value(raw: str) -> Any:
    """Parse a user-entered string into a Python scalar or literal."""
    text = raw.strip()
    if text == "":
        return ""
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _runtime_asset_path(relative_path: Path) -> Path:
    """Resolve an asset path for both source and frozen application runs."""
    if getattr(sys, "frozen", False):
        runtime_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    else:
        runtime_root = Path(__file__).resolve().parents[1]
    return runtime_root / relative_path


def _display_path(path_value: str | Path) -> str:
    """Normalize displayed paths to native Windows-style separators."""
    text = str(path_value).strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser())
    except (OSError, ValueError):
        return text.replace("/", "\\")


def _normalize_float_text(raw: str) -> str | None:
    """Format integer-like numeric text as a float display string."""
    text = raw.strip()
    if text == "":
        return None
    try:
        value = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"{float(value):.1f}"
    return None


def _format_editor_value(value: Any, option_values: list[str] | None = None, key: str | None = None) -> str:
    """Format a Python value for display in a Tk editor variable."""
    if isinstance(value, str):
        return _to_display_option_value(key, value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if key == "rng_seed" and value is None:
        return "None"
    if option_values and value is None:
        return "None"
    if value == "":
        return ""
    return repr(value)


def _to_display_option_value(key: str | None, value: str) -> str:
    """Map internal option values to user-facing labels where needed."""
    if key is None:
        return value
    return OPTION_DISPLAY_LABELS.get(key, {}).get(value, value)


def _from_display_option_value(key: str, value: Any) -> Any:
    """Map user-facing option labels back to internal option values."""
    if not isinstance(value, str):
        return value
    labels = OPTION_DISPLAY_LABELS.get(key, {})
    for internal_value, display_label in labels.items():
        if value == display_label:
            return internal_value
    return value


def _configure_windows_app_id() -> None:
    """Set a stable Windows AppUserModelID for the main HydroLPT GUI window."""
    if os.name != "nt":
        return

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except Exception:
        pass

class HydroLPTGUI(tk.Tk):
    """Main desktop application for creating and running HydroLPT cases."""

    def __init__(self) -> None:
        super().__init__()
        self.title("HydroLPT v0.9 Studio")
        self.geometry("820x860")
        self.minsize(760, 740)

        self.blueprints = default_case_blueprints()
        self.case_schemas: dict[str, CaseSchema] = {
            case_type: case_schema_from_payload(
                case_type=blueprint["case_type"],
                sections=blueprint["sections"],
                plots=blueprint["plots"],
                export_data=bool(blueprint.get("export_data", False)),
                xdmf_path=str(blueprint.get("xdmf_path", "")),
            )
            for case_type, blueprint in self.blueprints.items()
        }
        self.current_schema: CaseSchema = copy.deepcopy(self.case_schemas["synthetic"])
        self.case_type = "synthetic"
        self.editor_vars: dict[str, dict[str, tk.StringVar]] = {}
        self.section_widgets: dict[str, dict[str, list[tk.Widget]]] = {}
        self.section_entries: dict[str, dict[str, tk.Widget]] = {}
        self.section_canvases: dict[str, tk.Canvas] = {}
        self.last_manual_release_coords: dict[str, tuple[str, str]] = {}
        self.run_thread: threading.Thread | None = None
        self.process: subprocess.Popen[str] | None = None
        self._suppress_next_run_exit_message = False
        self._pending_run_spec: dict[str, Any] | None = None
        self.temp_spec_path: Path | None = None
        self.stop_requested = False
        self.controls_locked_for_run = False
        self._run_token_counter = 0
        self._active_run_token: int | None = None
        self._plots_opened_tokens: set[int] = set()
        self._lingering_plot_process: subprocess.Popen[bytes] | None = None
        self.base_case_path: Path | None = None
        self.loaded_base_xdmf_path = ""
        self.workflow_mode = "new"
        self.output_dir_locked = False
        self.export_data_var = tk.BooleanVar(value=False)
        self.save_click_points_var = tk.BooleanVar(value=True)
        self._window_icon_image: tk.PhotoImage | None = None
        self._header_logo_source: tk.PhotoImage | None = None
        self._header_logo_image: tk.PhotoImage | None = None
        self._header_brand_frame: ttk.Frame | None = None
        self._mesh_count_refresh_after_id: str | None = None
        self._mesh_count_cache: dict[tuple[str, str], tuple[str, str]] = {}
        self._mesh_count_request_id = 0
        self.xdmf_var = tk.StringVar()
        self.hydraulic_mesh_nodes_var = tk.StringVar(value="")
        self.hydraulic_mesh_cells_var = tk.StringVar(value="")
        self.case_folder_ref_var = tk.StringVar()
        self.load_case_ref_var = tk.StringVar()
        self._last_browse_dir = default_output_dir().resolve()
        self.style = ttk.Style(self)
        self.ui_font = tkfont.nametofont("TkDefaultFont")
        self.xdmf_var.trace_add("write", lambda *_args: self._schedule_loaded_mesh_count_refresh())

        self._configure_window_icon()
        self._configure_styles()
        self._build_layout()
        self._set_case_type("synthetic")
        self._apply_workflow_mode()
        self._initialize_window_layout()

    def _set_current_schema(self, payload: dict[str, Any]) -> None:
        """Update the active typed case schema from a dict payload."""
        self.current_schema = case_schema_from_payload(
            case_type=payload["case_type"],
            sections=payload["sections"],
            plots=payload.get("plots", {}),
            export_data=bool(payload.get("export_data", False)),
            xdmf_path=str(payload.get("xdmf_path", "")),
            user_notes=str(payload.get("user_notes", "")),
        )

    def _initialize_window_layout(self) -> None:
        """Finalize startup sizing and logo placement after widgets exist."""
        self._fit_window_to_content_width()
        self._place_header_logo()
        self.after_idle(self._focus_notebook)

    def _normalize_float_entry_var(self, var: tk.StringVar) -> None:
        """Apply one-decimal formatting to integer-like values in float fields."""
        normalized = _normalize_float_text(var.get())
        if normalized is not None and var.get().strip() != normalized:
            var.set(normalized)

    def _apply_normalized_case_sections(self, normalized_case: dict[str, Any]) -> None:
        """Refresh editor fields from a normalized case payload."""
        self._set_current_schema(normalized_case)
        blueprint = build_blueprint_from_case(
            {
                "case_type": normalized_case["case_type"],
                "sections": normalized_case["sections"],
                "plots": normalized_case.get("plots", {}),
                "script_path": "",
                "xdmf_path": normalized_case.get("xdmf_path", ""),
            }
        )

        for section_name, vars_for_section in self.editor_vars.items():
            if section_name == "PLOTS":
                section_values = blueprint.get("plots", {})
            else:
                section_values = blueprint["sections"].get(section_name, {})
            for key, var in vars_for_section.items():
                if key in section_values:
                    current_value = section_values[key]
                else:
                    current_value = self.blueprints[self.case_type]["sections"].get(section_name, {}).get(key, "")
                option_values = self._get_option_values(key, current_value)
                var.set(_format_editor_value(current_value, option_values=option_values, key=key))

        if self._uses_external_hydraulics():
            self.loaded_base_xdmf_path = _display_path(normalized_case.get("xdmf_path", ""))
            if self.workflow_mode != "base":
                self.xdmf_var.set(self.loaded_base_xdmf_path)

        self._refresh_loaded_mesh_counts()
        self._update_conditional_fields()

    def _focus_notebook(self, _event: tk.Event | None = None) -> None:
        """Keep focus off the first editable field when tabs are shown."""
        try:
            self.notebook.focus_set()
        except Exception:
            pass

    def _configure_window_icon(self) -> None:
        """Apply the HydroLPT application icon to the Tkinter window."""
        png_path = _runtime_asset_path(ICON_PNG_PATH)
        ico_path = _runtime_asset_path(ICON_ICO_PATH)

        try:
            if png_path.exists():
                self._window_icon_image = tk.PhotoImage(file=str(png_path))
                self.iconphoto(True, self._window_icon_image)
        except Exception:
            self._window_icon_image = None

        if os.name == "nt":
            try:
                if ico_path.exists():
                    self.iconbitmap(str(ico_path))
            except Exception:
                pass

    def _configure_styles(self) -> None:
        """Configure styles used by the loader buttons."""
        self.header_font = self.ui_font.copy()
        self.header_font.configure(weight="bold")
        self.style.configure("SectionHeader.TLabel", font=self.header_font)
        self.style.configure(
            "ActiveLoad.TButton",
            background="#4fa3ff",
            foreground="#06284a",
        )
        self.style.map(
            "ActiveLoad.TButton",
            background=[
                ("disabled", "#f0f0f0"),
                ("pressed", "#2d86eb"),
                ("active", "#69b2ff"),
            ],
            foreground=[
                ("disabled", "#808080"),
                ("pressed", "#06284a"),
                ("active", "#06284a"),
            ],
        )

    def _fit_window_to_content_width(self) -> None:
        """Reduce the initial window width to the actual content edge."""
        self.update_idletasks()
        content_widgets = [self.hint_text, self.notebook, self.log_text]
        content_right = max(widget.winfo_x() + widget.winfo_width() for widget in content_widgets)
        target_width = max(700, int((content_right + 28) * 0.8))
        target_height = max(740, self.winfo_height())
        self.minsize(target_width, 740)
        self.geometry(f"{target_width}x{target_height}")

    def _place_header_logo(self, _event: tk.Event | None = None) -> None:
        """Align the brand block with the right edge of the top input boxes."""
        self.update_idletasks()
        selector_right = self.selector_entry.winfo_rootx() + self.selector_entry.winfo_width()
        panel_left = self.top_panel.winfo_rootx()
        right_edge = selector_right - panel_left
        if self._header_brand_frame is not None:
            self._header_brand_frame.place(x=right_edge, y=0, anchor="ne", bordermode="ignore")

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)

        self.top_panel = ttk.Frame(self, padding=14)
        self.top_panel.grid(row=0, column=0, sticky="ew")
        self.top_panel.columnconfigure(0, minsize=120)
        self.top_panel.columnconfigure(1, weight=1)

        ttk.Label(self.top_panel, text="Case Type").grid(row=0, column=0, sticky="w")
        self.case_type_var = tk.StringVar()
        self.case_type_combo = ttk.Combobox(
            self.top_panel,
            state="readonly",
            textvariable=self.case_type_var,
            values=[CASE_LABELS["synthetic"], CASE_LABELS["basement"], CASE_LABELS["hecras"]],
            width=SELECTOR_CONTROL_WIDTH,
        )
        self.case_type_combo.grid(row=0, column=1, sticky="w")
        self.case_type_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_case_type_change())

        ttk.Label(self.top_panel, text="Workflow").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.workflow_var = tk.StringVar(value=WORKFLOW_LABELS["new"])
        self.workflow_combo = ttk.Combobox(
            self.top_panel,
            state="readonly",
            textvariable=self.workflow_var,
            values=[WORKFLOW_LABELS["new"], WORKFLOW_LABELS["base"]],
            width=SELECTOR_CONTROL_WIDTH,
        )
        self.workflow_combo.grid(row=1, column=1, sticky="w", pady=(8, 0))
        self.workflow_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_workflow_mode_change())

        logo_path = _runtime_asset_path(ICON_PNG_PATH)
        self._header_brand_frame = ttk.Frame(self.top_panel)
        self._header_brand_frame.columnconfigure(0, weight=0)
        self._header_brand_frame.columnconfigure(1, weight=0)
        brand_text = ttk.Frame(self._header_brand_frame)
        brand_text.grid(row=0, column=0, sticky="ne", padx=(0, 10), pady=(4, 0))
        ttk.Label(brand_text, text="HydroLPT", font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, sticky="e")
        ttk.Label(brand_text, text="Lagrangian Particle Tracking for").grid(row=1, column=0, sticky="e")
        ttk.Label(brand_text, text="Hydro-Environmental Systems").grid(row=2, column=0, sticky="e")
        self.logo_label = ttk.Label(self._header_brand_frame)
        if logo_path.exists():
            try:
                self._header_logo_source = tk.PhotoImage(file=str(logo_path))
                self._header_logo_image = self._header_logo_source.zoom(4, 4).subsample(20, 20)
                self.logo_label.configure(image=self._header_logo_image)
            except Exception:
                self._header_logo_source = None
                self._header_logo_image = None
        self.logo_label.grid(row=0, column=1, sticky="ne", pady=(5, 0))
        self._header_brand_frame.place(x=0, y=0, anchor="ne")
        self.top_panel.bind("<Configure>", self._place_header_logo)

        action_bar = ttk.Frame(self, padding=(14, 0, 14, 0))
        action_bar.grid(row=1, column=0, sticky="ew", padx=(0, ACTION_BAR_RIGHT_MARGIN))
        action_bar.columnconfigure(0, minsize=120)
        action_bar.columnconfigure(1, weight=3)

        self.xdmf_button = ttk.Button(
            action_bar,
            text="XDMF File",
            command=self._pick_top_level_xdmf_file,
            width=14,
        )
        self.xdmf_entry = ttk.Entry(action_bar, textvariable=self.xdmf_var)
        self.xdmf_button.grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
        self.xdmf_entry.grid(row=0, column=1, sticky="ew", pady=(0, 8))

        self.output_dir_var = tk.StringVar(value=str(default_output_dir().resolve()))
        self.base_case_var = tk.StringVar()
        self.selector_entry = ttk.Entry(action_bar, textvariable=self.output_dir_var)
        self.selector_button = ttk.Button(
            action_bar,
            text="Case Folder",
            command=self._choose_output_dir,
            width=14,
        )
        self.selector_button.grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.selector_entry.grid(row=1, column=1, sticky="ew")

        self.load_case_entry = ttk.Entry(action_bar, textvariable=self.base_case_var)
        self.load_case_button = ttk.Button(
            action_bar,
            text="Load Case",
            command=self._choose_base_case,
            width=14,
        )
        self.load_case_button.grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.load_case_entry.grid(row=2, column=1, sticky="ew", pady=(8, 0))

        self.status_var = tk.StringVar(value="Ready")

        self.hint_text = ScrolledText(self, wrap="word", height=8)
        self.hint_text.grid(row=2, column=0, sticky="ew", padx=(14, PANEL_RIGHT_MARGIN), pady=(10, 0))
        self.hint_text.configure(font=self.ui_font)
        self.hint_text.tag_configure(
            "bold",
            font=(self.ui_font.actual("family"), self.ui_font.actual("size"), "bold"),
        )
        self.hint_text.configure(state="disabled")

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=3, column=0, sticky="nsew", padx=(14, PANEL_RIGHT_MARGIN), pady=(10, 0))
        self.notebook.bind("<<NotebookTabChanged>>", self._focus_notebook)

        bottom = ttk.Frame(self, padding=(14, 16, PANEL_RIGHT_MARGIN, 14))
        bottom.grid(row=4, column=0, sticky="nsew")
        bottom.columnconfigure(0, weight=1)
        bottom.rowconfigure(1, weight=1)

        controls = ttk.Frame(bottom)
        controls.grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.run_case_button = ttk.Button(controls, text="Run Case", command=self._run_case)
        self.run_case_button.pack(side="left")
        self.save_case_button = ttk.Button(controls, text="Save", command=self._save_case)
        self.save_case_button.pack(side="left", padx=(8, 0))
        self.stop_case_button = ttk.Button(controls, text="Stop Run", command=self._stop_case)
        self.stop_case_button.pack(side="left", padx=(8, 0))
        self.stop_case_button.configure(state="disabled")
        self.export_files_var = tk.BooleanVar(value=True)
        self.export_files_check: ttk.Checkbutton | None = None
        self.export_data_check: ttk.Checkbutton | None = None
        self.save_click_points_check: ttk.Checkbutton | None = None

        self.log_text = ScrolledText(bottom, wrap="word", height=8)
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=(2, 0))
        self.log_text.configure(font=self.ui_font)

        self.output_dir_var.trace_add("write", lambda *_args: self._sync_reference_paths())

    def _set_hint_text(self, text: str) -> None:
        self.hint_text.configure(state="normal")
        self.hint_text.delete("1.0", tk.END)
        parts = text.split("**")
        for index, part in enumerate(parts):
            if not part:
                continue
            if index % 2 == 1:
                self.hint_text.insert(tk.END, part, "bold")
            else:
                self.hint_text.insert(tk.END, part)
        self.hint_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)

    def _append_log_threadsafe(self, text: str) -> None:
        self.after(0, self._append_log, text)

    def _set_widget_state(self, widget: tk.Widget, state: str) -> None:
        """Set widget state when supported, ignoring widgets without a state option."""
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass

    def _set_editor_locked_state(self, locked: bool) -> None:
        """Grey out editor controls while leaving the current tabs visible."""
        for section_name, entry_map in self.section_entries.items():
            vars_for_section = self.editor_vars.get(section_name, {})
            for key, widget in entry_map.items():
                if locked:
                    self._set_widget_state(widget, "disabled")
                else:
                    current_var = vars_for_section.get(key)
                    current_value = _parse_value(current_var.get()) if current_var is not None else None
                    has_options = self._get_option_values(key, current_value) is not None
                    editable = self._is_field_editable(section_name, key)

                    if isinstance(widget, ttk.Combobox):
                        self._set_widget_state(widget, "readonly" if editable and has_options else "disabled")
                    elif isinstance(widget, ttk.Entry):
                        self._set_widget_state(widget, "normal" if editable else "readonly")
                    else:
                        self._set_widget_state(widget, "normal" if editable else "disabled")

    def _set_run_ui_state(self, running: bool) -> None:
        """Grey out editor tabs during a run and leave only Stop Run enabled."""
        self.controls_locked_for_run = running

        combo_state = "disabled" if running else "readonly"
        entry_state = "disabled" if running else "normal"
        button_state = "disabled" if running else "normal"

        self._set_widget_state(self.case_type_combo, combo_state)
        self._set_widget_state(self.workflow_combo, combo_state)
        self._set_widget_state(self.xdmf_button, button_state)
        self._set_widget_state(self.xdmf_entry, entry_state)
        self._set_widget_state(self.selector_button, button_state)
        self._set_widget_state(self.selector_entry, entry_state)
        self._set_widget_state(self.load_case_button, button_state)
        self._set_widget_state(self.load_case_entry, entry_state)
        self._set_widget_state(self.run_case_button, button_state)
        self._set_widget_state(self.save_case_button, button_state)
        if self.export_files_check is not None:
            self._set_widget_state(self.export_files_check, "disabled")
        if self.export_data_check is not None:
            self._set_widget_state(self.export_data_check, button_state)
        if self.save_click_points_check is not None:
            self._set_widget_state(self.save_click_points_check, button_state)
        self._set_widget_state(self.stop_case_button, "normal" if running else "disabled")
        self._set_editor_locked_state(running)

        if not running:
            self._sync_xdmf_controls()
            self._update_conditional_fields()

    def _close_lingering_plot_process(self) -> None:
        """Close any still-open plot windows from a previous completed run."""
        process = self._lingering_plot_process
        if process is None:
            return
        if process.poll() is None:
            self._append_log("\nClosing figures from the previous run...\n")
            process.terminate()
        self._lingering_plot_process = None

    def _format_release_coordinate_texts(self, centers: list[list[float]]) -> tuple[str, str]:
        """Format release center coordinates for the GUI text fields."""
        xs = ", ".join(f"{float(center[0]):g}" for center in centers)
        ys = ", ".join(f"{float(center[1]):g}" for center in centers)
        return xs, ys

    def _adopt_manual_release_centers(self, centers: list[list[float]]) -> None:
        """Store accepted click-picked release centers and switch back to coordinate mode."""
        if not centers:
            return

        release_vars = self.editor_vars.get("PARTICLE_SETTINGS", {})
        release_x_var = release_vars.get("release_x")
        release_y_var = release_vars.get("release_y")
        location_mode_var = self.editor_vars.get("PARTICLE_SETTINGS", {}).get("location_mode")
        if release_x_var is None or release_y_var is None:
            return

        release_x_text, release_y_text = self._format_release_coordinate_texts(centers)
        self.last_manual_release_coords[self.case_type] = (release_x_text, release_y_text)
        release_x_var.set(release_x_text)
        release_y_var.set(release_y_text)
        if location_mode_var is not None and location_mode_var.get() == "click":
            location_mode_var.set("coordinates")
            self._update_conditional_fields()

    def _apply_saved_click_points_from_temp_spec(self, spec_path: Path | None = None) -> None:
        """Pull persisted click-picked release points back from the subprocess spec file."""
        target_path = spec_path if spec_path is not None else self.temp_spec_path
        if target_path is None or not target_path.exists():
            return

        try:
            spec = json.loads(target_path.read_text(encoding="utf-8"))
        except Exception:
            return

        if not spec.get("preserve_clicked_release_points"):
            return

        sections = spec.get("sections", {})
        release_settings = sections.get("PARTICLE_SETTINGS", {})
        manual_centers = release_settings.get("manual_centers", [])
        if not manual_centers:
            return

        self._adopt_manual_release_centers(manual_centers)

    def _clamp_release_tracking_count(
        self,
        sections: dict[str, dict[str, Any]],
        *,
        duration_limited: bool = False,
    ) -> str | None:
        """Ensure N_track never exceeds N and keep the GUI fields in sync."""
        release_settings = sections.get("PARTICLE_SETTINGS", {})
        if not release_settings:
            return None

        try:
            n_per_center = int(release_settings.get("n_per_center", 0))
            n_track = int(release_settings.get("nTrack", 0))
        except (TypeError, ValueError):
            return None

        if n_track <= n_per_center:
            return None

        release_settings["nTrack"] = n_per_center
        track_var = self.editor_vars.get("PARTICLE_SETTINGS", {}).get("nTrack")
        if track_var is not None:
            track_var.set(str(n_per_center))
        if duration_limited:
            return (
                "Tracked particles reduced after applying the continuous-release "
                f"duration limit: N_track reduced from {n_track} to {n_per_center}.\n"
            )
        return (
            "Tracked particles reduced because the requested number exceeded "
            f"the released particles: N_track reduced from {n_track} to {n_per_center}.\n"
        )

    def _sync_save_click_points_control(self) -> None:
        """Enable the click-save option only while click location mode is active."""
        if self.save_click_points_check is None:
            return
        location_mode_var = self.editor_vars.get("PARTICLE_SETTINGS", {}).get("location_mode")
        location_mode = location_mode_var.get() if location_mode_var is not None else "coordinates"
        if location_mode == "click":
            self.save_click_points_check.configure(state="normal")
            return

        self.save_click_points_check.configure(state="disabled")

    def _on_workflow_mode_change(self) -> None:
        label = self.workflow_var.get()
        for key, value in WORKFLOW_LABELS.items():
            if value == label:
                self.workflow_mode = key
                break
        self._apply_workflow_mode()

    def _apply_workflow_mode(self) -> None:
        is_base_mode = self.workflow_mode == "base"
        self.workflow_var.set(WORKFLOW_LABELS[self.workflow_mode])

        if is_base_mode:
            self.case_type_combo.configure(state="readonly")
            if self._uses_external_hydraulics():
                self.selector_entry.configure(textvariable=self.base_case_var)
                self.selector_button.configure(command=self._choose_base_case)
            else:
                self.selector_entry.configure(textvariable=self.case_folder_ref_var)
                self.selector_button.configure(command=self._choose_output_dir)
                self.load_case_entry.configure(textvariable=self.base_case_var)
            self.selector_entry.configure(state="normal")
            if self.base_case_path is None:
                self.base_case_var.set("")
            self.output_dir_var.set("")
            if self.base_case_path is not None:
                target_output_dir = self._default_output_dir_for_base_case(self.base_case_path)
                self._set_output_dir(target_output_dir, locked=self._should_lock_output_dir(self.base_case_path))
        else:
            self.case_type_combo.configure(state="readonly")
            if self._uses_external_hydraulics():
                self.selector_entry.configure(textvariable=self.load_case_ref_var)
                self.selector_button.configure(command=self._choose_base_case)
            else:
                self.selector_entry.configure(textvariable=self.output_dir_var)
                self.selector_button.configure(command=self._choose_output_dir)
                self.load_case_entry.configure(textvariable=self.load_case_ref_var)
            if self.base_case_path is not None:
                current_case_type = self.case_type
                self.base_case_path = None
                self.loaded_base_xdmf_path = ""
                self.base_case_var.set("")
                self._set_case_type(current_case_type)
            self.output_dir_var.set("")
            self._set_output_controls_locked(False)
        self._sync_reference_paths()
        self._sync_action_button_labels()
        self._sync_aux_action_rows()
        self._sync_selector_state()
        self._set_hint_text(self._hint_for_case_type(self.case_type))

    def _set_output_controls_locked(self, locked: bool) -> None:
        self.output_dir_locked = locked
        self._sync_selector_state()

    def _sync_action_button_labels(self) -> None:
        if self.case_type == "synthetic":
            self.selector_button.configure(text="Case Folder" if self.workflow_mode == "new" else "Case Folder Ref.")
            self.load_case_button.configure(text="Load Case" if self.workflow_mode == "base" else "Load Case Ref.")
            self.selector_button.configure(style="ActiveLoad.TButton" if self.workflow_mode == "new" else "TButton")
            self.load_case_button.configure(style="ActiveLoad.TButton" if self.workflow_mode == "base" else "TButton")
            return

        self.xdmf_button.configure(
            text=self._hydraulic_file_label() if self.workflow_mode == "new" else self._hydraulic_file_ref_label()
        )
        self.selector_button.configure(text="Load Case" if self.workflow_mode == "base" else "Load Case Ref.")
        self.xdmf_button.configure(style="ActiveLoad.TButton" if self.workflow_mode == "new" else "TButton")
        self.selector_button.configure(style="ActiveLoad.TButton" if self.workflow_mode == "base" else "TButton")
        self.load_case_button.configure(style="TButton")

    def _predicted_case_script_path(self) -> str:
        output_dir = self.output_dir_var.get().strip()
        if not output_dir:
            return ""
        try:
            return _display_path(Path(output_dir).expanduser().resolve() / GUI_CASE_FILENAME)
        except OSError:
            return ""

    def _existing_simulation_folder(self) -> str:
        if self.base_case_path is None:
            return ""
        try:
            return _display_path(Path(self.base_case_path).expanduser().resolve().parent)
        except OSError:
            return ""

    def _sync_reference_paths(self) -> None:
        self.load_case_ref_var.set(self._predicted_case_script_path())
        self.case_folder_ref_var.set(self._existing_simulation_folder())
        if self._uses_external_hydraulics() and self.workflow_mode == "base":
            self.xdmf_var.set(self.loaded_base_xdmf_path)

    def _sync_selector_state(self) -> None:
        if self._uses_external_hydraulics():
            if self.workflow_mode == "base":
                self.xdmf_entry.configure(state="disabled")
                self.xdmf_button.configure(state="disabled")
            else:
                xdmf_locked = self._should_lock_xdmf_field(self.xdmf_var.get())
                self.xdmf_entry.configure(state="readonly" if xdmf_locked else "normal")
                self.xdmf_button.configure(state="disabled" if xdmf_locked else "normal")
        else:
            self.xdmf_entry.configure(state="disabled")
            self.xdmf_button.configure(state="disabled")

        if self.case_type == "synthetic":
            if self.workflow_mode == "base":
                self.load_case_entry.configure(state="readonly")
                self.load_case_button.configure(state="normal")
                self.selector_entry.configure(state="readonly")
                self.selector_button.configure(state="disabled")
            else:
                self.load_case_entry.configure(state="readonly")
                self.load_case_button.configure(state="disabled")
                self.selector_entry.configure(state="readonly" if self.output_dir_locked else "normal")
                self.selector_button.configure(state="disabled" if self.output_dir_locked else "normal")
            return

        if self.workflow_mode == "new" and self._uses_external_hydraulics():
            self.selector_entry.configure(state="readonly")
            self.selector_button.configure(state="disabled")
            return

        if self.workflow_mode == "base":
            self.selector_entry.configure(state="normal")
            self.selector_button.configure(state="normal")
            return
        self.selector_entry.configure(state="readonly" if self.output_dir_locked else "normal")
        self.selector_button.configure(state="disabled" if self.output_dir_locked else "normal")

    def _set_output_dir(self, path: Path, *, locked: bool) -> None:
        resolved = Path(path).expanduser().resolve()
        self.output_dir_var.set(_display_path(resolved))
        self._last_browse_dir = resolved
        self._set_output_controls_locked(locked)

    def _browse_initial_dir(self, preferred: Path | None = None) -> str:
        candidates = [preferred, self.base_case_path.parent if self.base_case_path is not None else None, self._last_browse_dir]
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                resolved = Path(candidate).expanduser().resolve()
            except OSError:
                continue
            if resolved.exists():
                return str(resolved)
        return str(default_output_dir().resolve())

    def _sync_aux_action_rows(self) -> None:
        if self.case_type == "synthetic":
            self.xdmf_button.grid_remove()
            self.xdmf_entry.grid_remove()
            self.load_case_button.grid()
            self.load_case_entry.grid()
        else:
            self.xdmf_button.grid()
            self.xdmf_entry.grid()
            self.load_case_button.grid_remove()
            self.load_case_entry.grid_remove()

    def _default_output_dir_for_base_case(self, script_path: Path) -> Path:
        resolved = Path(script_path).expanduser().resolve()
        return resolved.parent

    def _should_lock_output_dir(self, script_path: Path) -> bool:
        return True

    def _current_run_name(self) -> str:
        if self.base_case_path is not None:
            return self.base_case_path.stem

        output_dir = self.output_dir_var.get().strip()
        if output_dir:
            return Path(output_dir).expanduser().resolve().name or "gui_case"

        return "gui_case"

    def _uses_external_hydraulics(self, case_type: str | None = None) -> bool:
        """Return whether the selected case type loads external hydraulic files."""
        target = self.case_type if case_type is None else case_type
        return target in {"basement", "hecras"}

    def _hydraulic_file_label(self, case_type: str | None = None) -> str:
        """Return the short label used for the external hydraulic file selector."""
        target = self.case_type if case_type is None else case_type
        if target == "hecras":
            return "Plan HDF"
        return "XDMF File"

    def _hydraulic_file_ref_label(self, case_type: str | None = None) -> str:
        """Return the short label used for the read-only hydraulic file reference."""
        target = self.case_type if case_type is None else case_type
        if target == "hecras":
            return "Plan Ref."
        return "XDMF Ref."

    def _hydraulic_file_prompt(self, case_type: str | None = None) -> str:
        """Return the validation message for the selected hydraulic file type."""
        target = self.case_type if case_type is None else case_type
        if target == "hecras":
            return "Please choose a valid HEC-RAS plan HDF file first."
        return "Please choose a valid XDMF file first."

    def _hydraulic_file_dialog_title(self, case_type: str | None = None) -> str:
        """Return the file dialog title for the selected hydraulic backend."""
        target = self.case_type if case_type is None else case_type
        if target == "hecras":
            return "Select HEC-RAS plan HDF"
        return "Select BASEMENT results.xdmf"

    def _hydraulic_filetypes(self, case_type: str | None = None) -> list[tuple[str, str]]:
        """Return filetype filters for the selected hydraulic backend."""
        target = self.case_type if case_type is None else case_type
        if target == "hecras":
            return [("HDF files", "*.hdf"), ("All files", "*.*")]
        return [("XDMF files", "*.xdmf"), ("All files", "*.*")]

    def _is_valid_hydraulic_file_path(self, path: Path | None, case_type: str | None = None) -> bool:
        """Return whether a path matches the expected external hydraulic file type."""
        if path is None or not path.is_file():
            return False
        target = self.case_type if case_type is None else case_type
        suffix = path.suffix.lower()
        if target == "hecras":
            return suffix == ".hdf"
        return suffix == ".xdmf"

    def _on_case_type_change(self) -> None:
        label = self.case_type_var.get()
        for key, value in CASE_LABELS.items():
            if value == label:
                if self.workflow_mode == "base":
                    self.base_case_path = None
                    self.loaded_base_xdmf_path = ""
                    self.base_case_var.set("")
                self._set_case_type(key)
                self._apply_workflow_mode()
                break

    def _set_case_type(self, case_type: str) -> None:
        self.case_type = case_type
        self.current_schema = copy.deepcopy(self.case_schemas[case_type])
        self.case_type_var.set(CASE_LABELS[case_type])
        self._populate_from_blueprint(self.blueprints[case_type])
        release_defaults = self.blueprints[case_type]["sections"].get("PARTICLE_SETTINGS", {})
        self.last_manual_release_coords[case_type] = (
            str(release_defaults.get("release_x", "")),
            str(release_defaults.get("release_y", "")),
        )
        self._set_hint_text(self._hint_for_case_type(case_type))

    def _hint_for_case_type(self, case_type: str) -> str:
        if case_type == "synthetic" and self.workflow_mode == "base":
            return (
                "Synthetic Flume mode can reopen an existing HydroLPT case.\n\n"
                "- Use Load Case to select the existing `HydroLPT.py` (or renamed) file.\n"
                "- Loaded case settings will populate the editable fields shown here.\n"
                "- Runs are saved after `Run Case` or using the `Save` function.\n"
                "- Outputs, including particle status, final locations, and selected trajectories, are saved to the `output` folder." 
            )
        if case_type == "synthetic":
            return (
                "Synthetic Flume mode builds an idealized hydraulic case directly in HydroLPT.\n\n"
                "- Use the `HYDRAULIC_INPUT` tab to specify flume geometry and hydraulic inputs.\n"
                "- The width and the mesh size are fixed for this demonstration.\n"
                "- Use the `PARTICLE_INI` tab to switch between click and coordinate release locations.\n"
                "- For a new simulation, choose a run folder before starting so the GUI can write the HydroLPT.py case file there.\n" 
                "- Outputs, including particle status, final locations, and selected trajectories, are saved to the `output` folder."                
            )
        if case_type == "hecras" and self.workflow_mode == "base":
            return (
                "HEC-RAS 2D mode can reopen an existing HydroLPT HEC-RAS case.\n\n"
                "- Use Load Case to select the existing `HydroLPT.py` (or renamed) file.\n"
                "- Loaded case settings will populate the editable fields shown here.\n"
                "- Runs are saved after `Run Case` or using the `Save` function.\n"
                "- Outputs, including particle status, final locations, and selected trajectories, are saved to the `output` folder."
            )
        if case_type == "hecras":
            return (
                "HEC-RAS 2D mode runs HydroLPT against a HEC-RAS unsteady plan HDF.\n\n"
                "- For a new simulation, pick the plan HDF file with the Plan HDF selector at the top.\n"
                "- HydroLPT reads cell water surface and face velocities from the HEC-RAS results file.\n"
                "- Use the `PARTICLE_INI` tab to switch between click and coordinate release locations.\n"
                "- Outputs, including particle status, final locations, and selected trajectories, are saved to the `output` folder."
            )
        if self.workflow_mode == "base":
            return (
                "BASEMENT HPC mode runs HydroLPT against exported BASEMENT results.\n\n"
                "- Use Load Case to select the existing `HydroLPT.py` (or renamed) file.\n"
                "- Loaded case settings will populate the editable fields shown here.\n"
                "- Runs are saved after `Run Case` or using the `Save` function.\n"
                "- Outputs, including particle status, final locations, and selected trajectories, are saved to the `output` folder."             )
        return (
            "BASEMENT HPC mode runs HydroLPT against exported BASEMENT results.\n\n"
            "**IMPORTANT**: HydroLPT requires the outputs `flow_velocity`, `water_surface`, `bottom_elevation`, and `friction_chezy` to be written by BASEMENT into the XDMF file.\n\n"
            "- For a new simulation, pick the `results.xdmf` file with the XDMF File selector at the top.\n"
            "- Use the `PARTICLE_INI` tab to switch between click and coordinate release locations.\n"  
            "- Outputs, including particle status, final locations, and selected trajectories, are saved to the `output` folder." 
        )

    def _choose_base_case(self) -> None:
        if self.workflow_mode != "base":
            return
        path = filedialog.askopenfilename(
            title="Select HydroLPT case script",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            initialdir=self._browse_initial_dir(self.base_case_path.parent if self.base_case_path is not None else None),
        )
        if not path:
            return
        try:
            self._load_base_case(Path(path))
        except Exception as exc:
            self.status_var.set("Failed")
            self._append_log(f"\nLoad failed: Could not load case file: {exc}\n")

    def _load_base_case(self, script_path: Path) -> None:
        case = load_case_definition(script_path)
        if case["case_type"] not in CASE_LABELS:
            raise ValueError(f"Unsupported case type in {script_path}")

        blueprint = build_blueprint_from_case(case)
        self._set_current_schema(
            {
                "case_type": case["case_type"],
                "sections": case["sections"],
                "plots": case.get("plots", {}),
                "export_data": bool(case.get("settings", {}).get("export_data", False)),
                "xdmf_path": case.get("xdmf_path", ""),
                "user_notes": case.get("user_notes", ""),
            }
        )
        self.base_case_path = Path(script_path).resolve()
        self._last_browse_dir = self.base_case_path.parent
        self.loaded_base_xdmf_path = _display_path(case.get("xdmf_path", ""))
        self.base_case_var.set(_display_path(self.base_case_path))
        self.case_type = case["case_type"]
        self.case_type_var.set(CASE_LABELS[self.case_type])
        self._populate_from_blueprint(blueprint)
        self._set_output_dir(
            self._default_output_dir_for_base_case(self.base_case_path),
            locked=self._should_lock_output_dir(self.base_case_path),
        )
        self._sync_reference_paths()
        self._set_hint_text(self._hint_for_case_type(self.case_type))
        self.status_var.set("Loaded base case")

    def _clear_base_case(self) -> None:
        self.base_case_path = None
        self.loaded_base_xdmf_path = ""
        self.base_case_var.set("")
        self._sync_reference_paths()
        self._set_case_type(self.case_type)
        if self.workflow_mode == "base":
            self._set_output_dir(default_output_dir().resolve(), locked=False)

    def _populate_from_blueprint(self, blueprint: dict[str, Any]) -> None:
        self.editor_vars.clear()
        self.section_widgets.clear()
        self.section_entries.clear()
        self.section_canvases.clear()
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)

        if self._uses_external_hydraulics(blueprint["case_type"]) and self.workflow_mode != "base":
            self.xdmf_var.set(str(blueprint.get("xdmf_path", "")))
        else:
            self.xdmf_var.set("")
        self._sync_xdmf_controls()
        self.export_data_var.set(bool(blueprint.get("export_data", False)))
        self.save_click_points_var.set(self.workflow_mode == "new")

        section_items = list(blueprint["sections"].items())
        if blueprint["case_type"] == "synthetic":
            priority = {
                "HYDRAULIC_INPUT": 0,
            }
            section_items.sort(key=lambda item: (priority.get(item[0], 1),))

        for section_name, section_values in section_items:
            if not section_values and section_name != "PARTICLE_EVO_SETTINGS":
                continue
            self._add_section_tab(section_name, dict(section_values))

        self._add_section_tab("PLOTS", blueprint["plots"])
        self._update_conditional_fields()
        if self.controls_locked_for_run:
            self._set_editor_locked_state(True)

    def _add_section_tab(self, title: str, values: dict[str, Any], allow_picker: bool = False) -> None:
        outer = ttk.Frame(self.notebook, padding=8)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        if title == "PARTICLE_SETTINGS":
            inner = ttk.Frame(canvas, padding=4)
            inner.columnconfigure(0, weight=1)
            form_parent = inner
        else:
            inner = ttk.Frame(canvas, padding=4)
            form_parent = inner

        inner.bind("<Configure>", lambda _event, c=canvas: c.configure(scrollregion=c.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        vars_for_section: dict[str, tk.StringVar] = {}
        widgets_for_section: dict[str, list[tk.Widget]] = {}
        entries_for_section: dict[str, tk.Widget] = {}
        preview: tk.Canvas | None = None
        row = 0
        if title == "PLOTS":
            export_header = ttk.Label(form_parent, text="Export", style="SectionHeader.TLabel")
            export_header.grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 4))
            export_separator = ttk.Separator(form_parent, orient="horizontal")
            export_separator.grid(row=row + 1, column=0, columnspan=3, sticky="ew", pady=(0, 6))
            widgets_for_section["__export_header__"] = [export_header, export_separator]
            row += 2
            export_data_label = ttk.Label(form_parent, text="Export data")
            export_data_label.grid(row=row, column=0, sticky="nw", padx=(0, 12), pady=4)
            export_data_check = ttk.Checkbutton(
                form_parent,
                variable=self.export_data_var,
            )
            export_data_check.grid(row=row, column=1, sticky="w", pady=4)
            self.export_data_check = export_data_check
            widgets_for_section["__export_data__"] = [export_data_label, export_data_check]
            entries_for_section["__export_data__"] = export_data_check
            row += 1
            export_files_label = ttk.Label(form_parent, text="Export case file")
            export_files_label.grid(row=row, column=0, sticky="nw", padx=(0, 12), pady=4)
            export_files_check = ttk.Checkbutton(
                form_parent,
                variable=self.export_files_var,
            )
            export_files_check.grid(row=row, column=1, sticky="w", pady=4)
            export_files_check.configure(state="disabled")
            self.export_files_check = export_files_check
            widgets_for_section["__export_files__"] = [export_files_label, export_files_check]
            entries_for_section["__export_files__"] = export_files_check
            row += 1
        for item_type, value_key in self._ordered_section_items(title, values):
            if item_type == "header":
                if title == "PARTICLE_SETTINGS" and value_key == "Particle release" and preview is None:
                    preview = tk.Canvas(form_parent, width=220, height=130, highlightthickness=0, bd=0)
                    preview.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))
                    self.section_canvases[title] = preview
                    row += 1
                header_text = value_key
                if title == "HYDRAULIC_INPUT" and self._uses_external_hydraulics() and value_key == "Geometry":
                    header_text = "Hydraulic Inputs"
                header = ttk.Label(form_parent, text=header_text, style="SectionHeader.TLabel")
                header.grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 4))
                separator = ttk.Separator(form_parent, orient="horizontal")
                separator.grid(row=row + 1, column=0, columnspan=3, sticky="ew", pady=(0, 6))
                if title == "HYDRAULIC_INPUT" and value_key == "Steady Flow":
                    widgets_for_section["__steady_header__"] = [header, separator]
                if title == "HYDRAULIC_INPUT" and value_key == "Transient Flow":
                    widgets_for_section["__transient_header__"] = [header, separator]
                if title == "HYDRAULIC_INPUT" and value_key == "Time settings":
                    widgets_for_section["__hydraulic_time_header__"] = [header, separator]
                row += 2
                continue

            key = value_key
            label = ttk.Label(form_parent, text=self._field_label(key))
            label.grid(row=row, column=0, sticky="nw", padx=(0, 12), pady=4)
            value = values[key]
            display_value = repr(value) if value != "" else ""
            option_values = self._get_option_values(key, value)
            if isinstance(value, str):
                display_value = _to_display_option_value(key, value)
            elif isinstance(value, bool):
                display_value = "True" if value else "False"
            elif key == "rng_seed" and value is None:
                display_value = "None"
            elif option_values and value is None:
                display_value = "None"

            var = tk.StringVar(value=display_value)
            if option_values:
                entry_state = "readonly" if self._is_field_editable(title, key) else "disabled"
                entry = ttk.Combobox(
                    form_parent,
                    textvariable=var,
                    values=option_values,
                    state=entry_state,
                    width=SELECTOR_CONTROL_WIDTH,
                )
            else:
                entry_width = SELECTOR_CONTROL_WIDTH
                entry = ttk.Entry(form_parent, textvariable=var, width=entry_width)
                if title == "HYDRAULIC_INPUT" and key == "hydraulicClosureModel":
                    entry.configure(state="disabled")
                elif not self._is_field_editable(title, key):
                    entry.configure(state="readonly")
                elif allow_picker and key == "xdmf_path" and self._should_lock_xdmf_field(value):
                    entry.configure(state="disabled")
                elif isinstance(value, float):
                    entry.bind("<FocusOut>", lambda _event, target_var=var: self._normalize_float_entry_var(target_var))
                    entry.bind("<Return>", lambda _event, target_var=var: self._normalize_float_entry_var(target_var))
            entry.grid(row=row, column=1, sticky="w", pady=4)
            row_widgets: list[tk.Widget] = [label, entry]
            if allow_picker and key == "xdmf_path" and not self._should_lock_xdmf_field(value):
                button = ttk.Button(
                    form_parent,
                    text="Browse",
                    command=lambda target_var=var: self._pick_xdmf_file(target_var),
                )
                button.grid(row=row, column=2, padx=(8, 0), pady=4)
                row_widgets.append(button)
            elif title == "PARTICLE_SETTINGS" and key == "release_x":
                helper = ttk.Label(form_parent, text="Comma-separated values supported")
                helper.grid(row=row, column=2, padx=(8, 0), pady=4, sticky="w")
                row_widgets.append(helper)
            elif title == "PARTICLE_SETTINGS" and key == "release_y":
                helper = ttk.Label(form_parent, text="Comma-separated values supported")
                helper.grid(row=row, column=2, padx=(8, 0), pady=4, sticky="w")
                row_widgets.append(helper)
            elif title == "PARTICLE_SETTINGS" and key == "nTrack":
                helper = ttk.Label(form_parent, text="Trajectories saved in `output`")
                helper.grid(row=row, column=2, padx=(8, 0), pady=4, sticky="w")
                row_widgets.append(helper)
            else:
                helper_text = FIELD_HELPERS.get((title, key))
                if helper_text:
                    helper = ttk.Label(form_parent, text=helper_text)
                    helper.grid(row=row, column=2, padx=(8, 0), pady=4, sticky="w")
                    row_widgets.append(helper)
            if title == "HYDRAULIC_INPUT" and key == "run_mode" and isinstance(entry, ttk.Combobox):
                entry.bind("<<ComboboxSelected>>", lambda _event: self._update_conditional_fields())
            if title == "PARTICLE_SETTINGS" and key == "location_mode" and isinstance(entry, ttk.Combobox):
                entry.bind("<<ComboboxSelected>>", lambda _event: self._update_conditional_fields())
                save_click_points_label = ttk.Label(form_parent, text="Save click coordinates")
                save_click_points_label.grid(row=row + 1, column=0, sticky="nw", padx=(0, 12), pady=4)
                save_click_points_check = ttk.Checkbutton(
                    form_parent,
                    variable=self.save_click_points_var,
                )
                save_click_points_check.grid(row=row + 1, column=1, sticky="w", pady=4)
                self.save_click_points_check = save_click_points_check
                widgets_for_section["__save_click_points__"] = [save_click_points_label, save_click_points_check]
                entries_for_section["__save_click_points__"] = save_click_points_check
                row += 1
            if title == "PARTICLE_SETTINGS" and key == "release_mode" and isinstance(entry, ttk.Combobox):
                entry.bind("<<ComboboxSelected>>", lambda _event: self._update_conditional_fields())
            if title == "PARTICLE_SETTINGS" and key == "shape" and isinstance(entry, ttk.Combobox):
                entry.bind("<<ComboboxSelected>>", lambda _event: self._update_conditional_fields())
            if title == "HYDRAULIC_INPUT" and key == "hydraulicPrimaryVariable" and isinstance(entry, ttk.Combobox):
                entry.bind("<<ComboboxSelected>>", lambda _event: self._update_conditional_fields())
            if title == "TRANSPORT_SETTINGS" and key == "transportModel" and isinstance(entry, ttk.Combobox):
                entry.bind("<<ComboboxSelected>>", lambda _event: self._update_conditional_fields())
            if title == "BOUNDARY_SETTINGS" and key == "surfacePolicy" and isinstance(entry, ttk.Combobox):
                entry.bind("<<ComboboxSelected>>", lambda _event: self._update_conditional_fields())
            if title == "BOUNDARY_SETTINGS" and key == "bedPolicy" and isinstance(entry, ttk.Combobox):
                entry.bind("<<ComboboxSelected>>", lambda _event: self._update_conditional_fields())
            if title == "BOUNDARY_SETTINGS" and key == "uphillPolicy" and isinstance(entry, ttk.Combobox):
                entry.bind("<<ComboboxSelected>>", lambda _event: self._update_conditional_fields())
            if title == "PARTICLE_EVO_SETTINGS" and key == "biofouling" and isinstance(entry, ttk.Combobox):
                entry.bind("<<ComboboxSelected>>", lambda _event: self._update_conditional_fields("biofouling"))
            if title == "PARTICLE_EVO_SETTINGS" and key == "degradation" and isinstance(entry, ttk.Combobox):
                entry.bind("<<ComboboxSelected>>", lambda _event: self._update_conditional_fields("degradation"))
            vars_for_section[key] = var
            widgets_for_section[key] = row_widgets
            entries_for_section[key] = entry
            if title == "HYDRAULIC_INPUT" and key in {"length", "domain_width", "nx", "ny"}:
                var.trace_add("write", lambda *_args: self._schedule_loaded_mesh_count_refresh())
            if (
                (title == "HYDRAULIC_INPUT" and key in {"rho_f", "nu", "g"})
                or (title == "PARTICLE_SETTINGS" and key in {"shape", "L", "I", "S", "rho_p"})
            ):
                var.trace_add("write", lambda *_args: self._update_conditional_fields())
            row += 1

        if title == "HYDRAULIC_INPUT":
            nodes_label = ttk.Label(form_parent, text="Loaded mesh nodes")
            nodes_label.grid(row=row, column=0, sticky="nw", padx=(0, 12), pady=4)
            nodes_entry = ttk.Entry(
                form_parent,
                textvariable=self.hydraulic_mesh_nodes_var,
                width=SELECTOR_CONTROL_WIDTH,
                state="readonly",
            )
            nodes_entry.grid(row=row, column=1, sticky="w", pady=4)
            widgets_for_section["__loaded_nodes__"] = [nodes_label, nodes_entry]
            entries_for_section["__loaded_nodes__"] = nodes_entry
            row += 1

            cells_label = ttk.Label(form_parent, text="Loaded mesh cells")
            cells_label.grid(row=row, column=0, sticky="nw", padx=(0, 12), pady=4)
            cells_entry = ttk.Entry(
                form_parent,
                textvariable=self.hydraulic_mesh_cells_var,
                width=SELECTOR_CONTROL_WIDTH,
                state="readonly",
            )
            cells_entry.grid(row=row, column=1, sticky="w", pady=4)
            cells_helper = ttk.Label(form_parent, text="Read-only mesh counts")
            cells_helper.grid(row=row, column=2, padx=(8, 0), pady=4, sticky="w")
            widgets_for_section["__loaded_cells__"] = [cells_label, cells_entry, cells_helper]
            entries_for_section["__loaded_cells__"] = cells_entry

        form_parent.columnconfigure(1, weight=1)
        self.editor_vars[title] = vars_for_section
        self.section_widgets[title] = widgets_for_section
        self.section_entries[title] = entries_for_section
        if title == "HYDRAULIC_INPUT":
            label = "HYDRAULIC_INPUT"
        elif title == "PARTICLE_SETTINGS":
            label = "PARTICLE_INI"
        elif title == "PARTICLE_EVO_SETTINGS":
            label = "PARTICLE_EVO"
        elif title == "TRANSPORT_SETTINGS":
            label = "TRANSPORT"
        elif title == "BOUNDARY_SETTINGS":
            label = "BOUNDARY"
        elif title == "PLOTS":
            label = "OUTPUT"
        else:
            label = title.replace("_SETTINGS", "").title()
        self.notebook.add(outer, text=label)

    def _sync_xdmf_controls(self) -> None:
        self._sync_aux_action_rows()
        self._sync_selector_state()

    def _ordered_section_items(self, title: str, values: dict[str, Any]) -> list[tuple[str, str]]:
        layout = SECTION_LAYOUTS.get(title, [])
        ordered_items: list[tuple[str, str]] = []
        seen_keys: set[str] = set()
        hidden_keys = HIDDEN_SECTION_FIELDS.get(title, set())
        skip_hydraulic_headers = (
            title == "HYDRAULIC_INPUT"
            and self.case_type != "synthetic"
            and {"hydraulicPrimaryVariable", "U", "transient_shape", "switch_fraction", "U0", "U1"}.isdisjoint(values)
        )

        for item_type, value_key in layout:
            if item_type == "header":
                if skip_hydraulic_headers and value_key in {"Hydraulic Inputs", "Steady Flow", "Transient Flow"}:
                    continue
                ordered_items.append((item_type, value_key))
                continue
            if value_key in values and value_key not in hidden_keys:
                ordered_items.append((item_type, value_key))
                seen_keys.add(value_key)

        for key in values:
            if key not in seen_keys and key not in hidden_keys:
                ordered_items.append(("field", key))

        return ordered_items

    def _field_label(self, key: str) -> str:
        return FIELD_LABELS.get(key, key)

    def _set_loaded_mesh_count_display(self, nodes_text: str, cells_text: str) -> None:
        """Update the read-only mesh count display."""
        self.hydraulic_mesh_nodes_var.set(nodes_text)
        self.hydraulic_mesh_cells_var.set(cells_text)

    def _refresh_synthetic_mesh_counts(self) -> None:
        """Refresh synthetic mesh node/cell counts from the current geometry."""
        geometry_vars = self.editor_vars.get("HYDRAULIC_INPUT", {})
        length_var = geometry_vars.get("length")
        width_var = geometry_vars.get("domain_width")
        if length_var is None or width_var is None:
            self._set_loaded_mesh_count_display("", "")
            return

        try:
            length = float(_parse_value(length_var.get()))
            domain_width = float(_parse_value(width_var.get()))
            nx = max(1, int(round(length / FIXED_SYNTHETIC_CELL_SIZE_X)))
            ny = max(1, int(round(domain_width / FIXED_SYNTHETIC_CELL_SIZE_Y)))
        except (TypeError, ValueError):
            self._set_loaded_mesh_count_display("Unavailable", "Unavailable")
            return

        total_nodes = (nx + 1) * (ny + 1) + (nx * ny)
        total_cells = 4 * nx * ny
        self._set_loaded_mesh_count_display(f"{total_nodes:,}", f"{total_cells:,}")

    def _schedule_loaded_mesh_count_refresh(self) -> None:
        """Debounce mesh count refresh while the XDMF path is being edited."""
        if self._mesh_count_refresh_after_id is not None:
            try:
                self.after_cancel(self._mesh_count_refresh_after_id)
            except Exception:
                pass
        self._mesh_count_refresh_after_id = self.after(250, self._refresh_loaded_mesh_counts)

    def _refresh_loaded_mesh_counts(self) -> None:
        """Refresh mesh node/cell counts for the active case."""
        self._mesh_count_refresh_after_id = None
        if self.case_type == "synthetic":
            self._refresh_synthetic_mesh_counts()
            return
        if not self._uses_external_hydraulics():
            self._set_loaded_mesh_count_display("", "")
            return

        if self.workflow_mode == "base":
            xdmf_text = self.loaded_base_xdmf_path.strip()
        else:
            xdmf_text = str(_parse_value(self.xdmf_var.get())).strip().strip("'\"")

        if not xdmf_text:
            self._set_loaded_mesh_count_display("No hydraulic file selected", "No hydraulic file selected")
            return

        try:
            xdmf_path = Path(xdmf_text).expanduser().resolve()
        except OSError:
            self._set_loaded_mesh_count_display("Invalid hydraulic path", "Invalid hydraulic path")
            return

        if not self._is_valid_hydraulic_file_path(xdmf_path):
            self._set_loaded_mesh_count_display("Hydraulic file not found", "Hydraulic file not found")
            return

        cache_key = (self.case_type, str(xdmf_path))
        cached = self._mesh_count_cache.get(cache_key)
        if cached is not None:
            self._set_loaded_mesh_count_display(*cached)
            return

        self._mesh_count_request_id += 1
        request_id = self._mesh_count_request_id
        self._set_loaded_mesh_count_display("Loading...", "Loading...")

        def worker() -> None:
            try:
                if self.case_type == "hecras":
                    meta = HecRasAdapter(xdmf_path).load_meta()
                else:
                    meta = BasementAdapter(xdmf_path).load_meta()
                result = (f"{meta.Nn:,}", f"{meta.Nc:,}")
            except Exception:
                result = ("Unavailable", "Unavailable")
            self.after(0, self._apply_loaded_mesh_count_result, request_id, cache_key, result)

        threading.Thread(target=worker, daemon=True).start()

    def _apply_loaded_mesh_count_result(
        self,
        request_id: int,
        cache_key: tuple[str, str],
        result: tuple[str, str],
    ) -> None:
        """Apply an async mesh-count lookup result if it is still current."""
        if request_id != self._mesh_count_request_id:
            return
        self._mesh_count_cache[cache_key] = result
        self._set_loaded_mesh_count_display(*result)

    def _hydraulic_field_label(self, key: str) -> str:
        """Return dynamic labels for synthetic hydraulic inputs."""
        return FIELD_LABELS.get(key, key)

    def _transport_model(self) -> str:
        raw_value = self.editor_vars.get("TRANSPORT_SETTINGS", {}).get(
            "transportModel",
            tk.StringVar(value="random_walk"),
        ).get()
        parsed_value = _parse_value(raw_value)
        return str(parsed_value).strip().lower() if parsed_value != "" else "random_walk"

    def _run_mode(self) -> str:
        raw_value = self.editor_vars.get("HYDRAULIC_INPUT", {}).get(
            "run_mode",
            tk.StringVar(value="steady"),
        ).get()
        parsed_value = _parse_value(raw_value)
        return str(parsed_value).strip().lower() if parsed_value != "" else "steady"

    def _transport_field_label(self, key: str) -> str:
        """Return dynamic labels for transport controls based on the selected model."""
        transport_model = self._transport_model()
        if transport_model == "langevin":
            override_labels = {
                "useRWx": "Enable stochastic fluctuation in x",
                "useRWy": "Enable stochastic fluctuation in y",
                "useRWz": "Enable stochastic fluctuation in z",
                "betaKh": "Horizontal diffusivity scale beta_Kh",
                "KhMax": "Horizontal diffusivity cap K_h,max (m^2/s)",
                "rAniso": "Horizontal anisotropy ratio r_aniso",
                "alphaKz": "Vertical diffusivity scale alpha_Kz",
                "KzMax": "Vertical diffusivity cap K_z,max (m^2/s)",
                "TL_horizontal": "Horizontal Langevin timescale T_L,h (s)",
                "TL_vertical": "Vertical Langevin timescale T_L,v (s)",
            }
            if key in override_labels:
                return override_labels[key]
        return FIELD_LABELS.get(key, key)

    def _default_release_coordinate_texts(self) -> tuple[str, str]:
        release_defaults = self.blueprints[self.case_type]["sections"].get("PARTICLE_SETTINGS", {})
        return (
            str(release_defaults.get("release_x", "0.0")).strip(),
            str(release_defaults.get("release_y", "0.0")).strip(),
        )

    def _fallback_release_coordinate_texts(self) -> tuple[str, str]:
        last_x, last_y = self.last_manual_release_coords.get(self.case_type, ("", ""))
        default_x, default_y = self._default_release_coordinate_texts()
        return (
            last_x.strip() or default_x,
            last_y.strip() or default_y,
        )

    def _parse_coordinate_series(self, raw_value: Any, field_name: str) -> list[float]:
        if raw_value is None:
            return []
        if isinstance(raw_value, (list, tuple)):
            items = list(raw_value)
        else:
            text = str(raw_value).strip()
            if not text:
                return []
            items = [part.strip() for part in text.split(",")] if "," in text else [text]

        values: list[float] = []
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            try:
                values.append(float(text))
            except ValueError as exc:
                raise ValueError(
                    f"{field_name} must be a number or a comma-separated list of numbers."
                ) from exc
        return values

    def _should_lock_xdmf_field(self, value: Any) -> bool:
        """Lock XDMF display for loaded base cases when the file was found."""
        if self.workflow_mode != "base" or self.base_case_path is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        try:
            return Path(text).expanduser().resolve().exists()
        except OSError:
            return False

    def _particle_shape(self) -> str:
        raw_value = self.editor_vars.get("PARTICLE_SETTINGS", {}).get(
            "shape",
            tk.StringVar(value="sphere"),
        ).get()
        parsed_value = _parse_value(raw_value)
        return str(parsed_value).strip().lower() if parsed_value != "" else "sphere"

    def _set_particle_dimension_label(self, key: str, text: str) -> None:
        widgets = self.section_widgets.get("PARTICLE_SETTINGS", {}).get(key, [])
        if widgets:
            label = widgets[0]
            if isinstance(label, ttk.Label):
                label.configure(text=text)

    def _update_particle_shape_fields(self) -> None:
        if "PARTICLE_SETTINGS" not in self.editor_vars:
            return

        shape = self._particle_shape()
        visible_fields = set(PARTICLE_SHAPE_DIMENSIONS.get(shape, PARTICLE_DIMENSION_FIELDS))
        labels = PARTICLE_DIMENSION_LABELS.get(shape, PARTICLE_DIMENSION_LABELS["prism"])

        for key in PARTICLE_DIMENSION_FIELDS:
            self._set_particle_dimension_label(key, labels.get(key, FIELD_LABELS.get(key, key)))
            self._set_row_visible("PARTICLE_SETTINGS", key, key in visible_fields)

        self._draw_particle_shape_preview(shape)

    def _refresh_transport_labels(self) -> None:
        """Refresh dynamic labels in the transport tab after model changes."""
        widgets = self.section_widgets.get("TRANSPORT_SETTINGS", {})
        for key in (
            "useRWx",
            "useRWy",
            "useRWz",
            "betaKh",
            "KhMax",
            "rAniso",
            "alphaKz",
            "KzMax",
            "TL_horizontal",
            "TL_vertical",
        ):
            row_widgets = widgets.get(key, [])
            if not row_widgets:
                continue
            label_widget = row_widgets[0]
            if isinstance(label_widget, ttk.Label):
                label_widget.configure(text=self._transport_field_label(key))

    def _sync_transport_model_control(self) -> None:
        """Sync the transport-model control state in the transport tab."""
        transport_entry = self.section_entries.get("TRANSPORT_SETTINGS", {}).get("transportModel")
        transport_var = self.editor_vars.get("TRANSPORT_SETTINGS", {}).get("transportModel")
        if not isinstance(transport_entry, ttk.Combobox) or transport_var is None:
            return

        allowed_models = OPTION_VALUES["transportModel"]
        transport_entry.configure(values=allowed_models)

        current_model = self._transport_model()
        if current_model not in allowed_models:
            transport_var.set("random_walk")

        if self.controls_locked_for_run:
            self._set_widget_state(transport_entry, "disabled")
        else:
            self._set_widget_state(transport_entry, "readonly")

    def _sync_uphill_step_control(self) -> None:
        """Disable the max-uphill-step field when uphill stopping is not active."""
        dz_entry = self.section_entries.get("BOUNDARY_SETTINGS", {}).get("dzUpMax")
        uphill_var = self.editor_vars.get("BOUNDARY_SETTINGS", {}).get("uphillPolicy")
        if dz_entry is None or uphill_var is None:
            return

        uphill_policy = str(_parse_value(uphill_var.get()) or "off").strip().lower()
        if self.controls_locked_for_run:
            self._set_widget_state(dz_entry, "disabled")
        elif uphill_policy == "off":
            self._set_widget_state(dz_entry, "disabled")
        else:
            editable = self._is_field_editable("BOUNDARY_SETTINGS", "dzUpMax")
            if isinstance(dz_entry, ttk.Combobox):
                self._set_widget_state(dz_entry, "readonly" if editable else "disabled")
            elif isinstance(dz_entry, ttk.Entry):
                self._set_widget_state(dz_entry, "normal" if editable else "readonly")
            else:
                self._set_widget_state(dz_entry, "normal" if editable else "disabled")

    def _draw_particle_shape_preview(self, shape: str) -> None:
        canvas = self.section_canvases.get("PARTICLE_SETTINGS")
        if canvas is None:
            return

        canvas.delete("all")

        if shape == "sphere":
            canvas.create_oval(65, 25, 165, 125, width=2)
            canvas.create_line(65, 75, 165, 75, dash=(4, 2))
            canvas.create_text(115, 66, text="D")
            return

        if shape == "ellipsoid":
            canvas.create_oval(45, 35, 185, 115, width=2)
            canvas.create_line(45, 75, 185, 75, dash=(4, 2))
            canvas.create_text(80, 66, text="L")
            canvas.create_line(115, 35, 115, 115, dash=(4, 2))
            canvas.create_text(149, 62, text="I")
            canvas.create_line(72, 102, 158, 48, dash=(4, 2))
            canvas.create_text(108, 50, text="S")
            return

        if shape == "cylinder":
            canvas.create_oval(70, 25, 150, 55, width=2)
            canvas.create_line(70, 40, 70, 110, width=2)
            canvas.create_line(150, 40, 150, 110, width=2)
            canvas.create_arc(70, 95, 150, 125, start=180, extent=180, style="arc", width=2)
            canvas.create_arc(70, 95, 150, 125, start=0, extent=180, style="arc", dash=(4, 2), width=2)
            canvas.create_text(157, 75, text="L")
            canvas.create_line(149, 40, 150, 110)
            canvas.create_line(70, 40, 150, 40, dash=(4, 2))
            canvas.create_text(110, 33, text="S")
            return

        if shape == "disk":
            canvas.create_oval(60, 38, 170, 68, width=2)
            canvas.create_line(60, 53, 60, 83, width=2)
            canvas.create_line(170, 53, 170, 83, width=2)
            canvas.create_arc(60, 68, 170, 98, start=180, extent=180, style="arc", width=2)
            canvas.create_arc(60, 68, 170, 98, start=0, extent=180, style="arc", dash=(4, 2), width=2)
            canvas.create_line(60, 53, 170, 53, dash=(4, 2))
            canvas.create_text(115, 46, text="L")
            canvas.create_line(170, 53, 170, 83)
            canvas.create_text(176, 68, text="S")
            return

        # prism
        canvas.create_rectangle(60, 45, 150, 110, width=2)
        canvas.create_line(90, 25, 180, 25, width=2)                 # top solid
        canvas.create_line(180, 25, 180, 90, width=2)                 # right solid
        canvas.create_line(90, 90, 180, 90, width=2, dash=(4, 2))     # bottom dashed
        canvas.create_line(90, 25, 90, 90, width=2, dash=(4, 2))      # left dashed
        canvas.create_line(60, 45, 90, 25, width=2)
        canvas.create_line(150, 45, 180, 25, width=2)
        canvas.create_line(150, 110, 180, 90, width=2)
        canvas.create_line(60, 110, 90, 90, width=2, dash=(4, 2))
        canvas.create_text(105, 125, text="L")
        canvas.create_text(188, 62, text="I")
        canvas.create_text(64, 25, text="S")

    def _is_field_editable(self, section_name: str, key: str) -> bool:
        if section_name == "HYDRAULIC_INPUT" and key in {"__loaded_nodes__", "__loaded_cells__"}:
            return False
        if self.case_type == "synthetic" and section_name == "HYDRAULIC_INPUT" and key == "hydraulicPrimaryVariable":
            return False
        if self.case_type in {"basement", "hecras"} and section_name == "TRANSPORT_SETTINGS" and key == "transportVelocityMode":
            return False
        if self.case_type == "synthetic" and section_name == "HYDRAULIC_INPUT" and key in LOCKED_GEOMETRY_KEYS:
            return False
        if self.case_type in {"synthetic", "basement", "hecras"} and section_name == "HYDRAULIC_INPUT" and key == "hydraulicClosureModel":
            return False
        if section_name == "TRANSPORT_SETTINGS" and key in LOCKED_TRANSPORT_KEYS:
            return False
        if section_name == "HYDRAULIC_INPUT" and key in LOCKED_FLUID_KEYS:
            return False
        if section_name == "PARTICLE_SETTINGS" and key in LOCKED_RELEASE_KEYS:
            return False
        return True

    def _get_option_values(self, key: str, value: Any) -> list[str] | None:
        if self.case_type == "synthetic" and key == "length":
            return SYNTHETIC_LENGTH_OPTIONS
        if self.case_type == "synthetic" and key == "water_depth":
            return SYNTHETIC_WATER_DEPTH_OPTIONS
        if key == "transportVelocityMode":
            if self.case_type == "synthetic":
                values = ["depth_averaged", "loglaw_vertical"]
            elif self.case_type in {"basement", "hecras"}:
                values = ["depth_averaged"]
            else:
                values = OPTION_VALUES.get(key)
            return [_to_display_option_value(key, option) for option in values] if values else None
        if isinstance(value, bool):
            return ["True", "False"]
        values = OPTION_VALUES.get(key)
        if values is None:
            return None
        if value is None and "None" not in values:
            values = ["None", *values]
        return [_to_display_option_value(key, option) for option in values]

    def _set_row_visible(self, section_name: str, key: str, visible: bool) -> None:
        widgets = self.section_widgets.get(section_name, {}).get(key, [])
        for widget in widgets:
            if visible:
                widget.grid()
            else:
                widget.grid_remove()

    def _enforce_particle_evolution_mode(self, changed_key: str | None = None) -> None:
        """Keep biofouling and degradation mutually exclusive in the GUI."""
        evo_vars = self.editor_vars.get("PARTICLE_EVO_SETTINGS", {})
        biofouling_var = evo_vars.get("biofouling")
        degradation_var = evo_vars.get("degradation")
        if biofouling_var is None or degradation_var is None:
            return

        biofouling_on = str(biofouling_var.get()).strip().lower() == "on"
        degradation_on = str(degradation_var.get()).strip().lower() == "on"
        if not (biofouling_on and degradation_on):
            return

        if changed_key == "biofouling":
            degradation_var.set("off")
        elif changed_key == "degradation":
            biofouling_var.set("off")
        else:
            degradation_var.set("off")

    def _current_particle_ws_sign(self) -> int | None:
        """Return the exact sign of the current particle settling velocity."""
        particle_vars = self.editor_vars.get("PARTICLE_SETTINGS", {})
        hydraulic_vars = self.editor_vars.get("HYDRAULIC_INPUT", {})
        try:
            particle = build_particle_properties(
                shape=str(_parse_value(particle_vars.get("shape").get() if particle_vars.get("shape") else "sphere")),
                rho_p=float(_parse_value(particle_vars.get("rho_p").get())),
                L=float(_parse_value(particle_vars.get("L").get())),
                I=_parse_value(particle_vars["I"].get()) if particle_vars.get("I") is not None else None,
                S=_parse_value(particle_vars["S"].get()) if particle_vars.get("S") is not None else None,
                rho_f=float(_parse_value(hydraulic_vars.get("rho_f").get())),
                nu=float(_parse_value(hydraulic_vars.get("nu").get())),
                g=float(_parse_value(hydraulic_vars.get("g").get())),
            )
        except Exception:
            return None
        if particle.ws < 0.0:
            return -1
        if particle.ws > 0.0:
            return 1
        return 0

    def _apply_buoyancy_policy_locks(self) -> tuple[str, str]:
        """Force GUI boundary selectors to the policy allowed by buoyancy."""
        boundary_vars = self.editor_vars.get("BOUNDARY_SETTINGS", {})
        surface_var = boundary_vars.get("surfacePolicy")
        bed_var = boundary_vars.get("bedPolicy")
        raw_surface_policy = surface_var.get() if surface_var is not None else "always_reflect"
        raw_bed_policy = bed_var.get() if bed_var is not None else "always_reflect"
        surface_policy = str(_from_display_option_value("surfacePolicy", _parse_value(raw_surface_policy)) or "always_reflect").strip().lower()
        bed_policy = str(_from_display_option_value("bedPolicy", _parse_value(raw_bed_policy)) or "always_reflect").strip().lower()

        ws_sign = self._current_particle_ws_sign()
        lock_surface = ws_sign is not None and ws_sign == 0
        lock_bed = ws_sign is not None and ws_sign <= 0

        if lock_surface:
            surface_policy = "always_reflect"
            if surface_var is not None and surface_var.get() != "always_reflect":
                surface_var.set("always_reflect")
        if lock_bed:
            bed_policy = "always_reflect"
            if bed_var is not None and bed_var.get() != _to_display_option_value("bedPolicy", "always_reflect"):
                bed_var.set(_to_display_option_value("bedPolicy", "always_reflect"))

        for key, locked in (("surfacePolicy", lock_surface), ("bedPolicy", lock_bed)):
            entry = self.section_entries.get("BOUNDARY_SETTINGS", {}).get(key)
            if entry is None:
                continue
            if self.controls_locked_for_run or locked:
                self._set_widget_state(entry, "disabled")
            elif isinstance(entry, ttk.Combobox):
                self._set_widget_state(entry, "readonly")
            else:
                self._set_widget_state(entry, "normal")

        return surface_policy, bed_policy

    def _update_conditional_fields(self, changed_key: str | None = None) -> None:
        self._enforce_particle_evolution_mode(changed_key)
        run_mode = self._run_mode()
        is_transient = run_mode == "transient"
        transport_model = self._transport_model()
        hydraulic_primary_var = self.editor_vars.get("HYDRAULIC_INPUT", {}).get("hydraulicPrimaryVariable")
        if self.case_type == "synthetic" and hydraulic_primary_var is not None and hydraulic_primary_var.get() != "U":
            hydraulic_primary_var.set("U")
        transport_velocity_var = self.editor_vars.get("TRANSPORT_SETTINGS", {}).get("transportVelocityMode")
        if self.case_type in {"basement", "hecras"} and transport_velocity_var is not None:
            raw_transport_velocity = str(_from_display_option_value("transportVelocityMode", _parse_value(transport_velocity_var.get())) or "").strip().lower()
            if raw_transport_velocity != "depth_averaged":
                transport_velocity_var.set("depth_averaged")
        boundary_vars = self.editor_vars.get("BOUNDARY_SETTINGS", {})
        raw_surface_policy = boundary_vars.get("surfacePolicy").get() if boundary_vars.get("surfacePolicy") is not None else "always_reflect"
        raw_bed_policy = boundary_vars.get("bedPolicy").get() if boundary_vars.get("bedPolicy") is not None else "always_reflect"
        surface_policy = str(_from_display_option_value("surfacePolicy", _parse_value(raw_surface_policy)) or "always_reflect").strip().lower()
        bed_policy = str(_from_display_option_value("bedPolicy", _parse_value(raw_bed_policy)) or "always_reflect").strip().lower()
        surface_policy, bed_policy = self._apply_buoyancy_policy_locks()
        show_surface_drag_coeff = surface_policy in {"deterministic_detachment", "probabilistic_detachment"}
        show_surface_contact_angle = surface_policy in {"deterministic_detachment", "probabilistic_detachment"}
        show_surface_detachment_sigma = surface_policy == "probabilistic_detachment"
        show_tanphi_ratio = bed_policy in {"reflect_if_ustar_gt_crit", "probabilistic_entrainment"}
        show_bed_entrainment_sigma = bed_policy == "probabilistic_entrainment"
        self._sync_transport_model_control()
        transport_model = self._transport_model()

        self._set_row_visible("TRANSPORT_SETTINGS", "tRelease", is_transient)
        if self.case_type in {"basement", "hecras"}:
            for key in (
                "__steady_header__",
                "hydraulicPrimaryVariable",
                "U",
                "z0",
                "__transient_header__",
                "transient_shape",
                "switch_fraction",
                "U0",
                "U1",
                "V",
                "V0",
                "V1",
            ):
                self._set_row_visible("HYDRAULIC_INPUT", key, False)
            self._set_row_visible("HYDRAULIC_INPUT", "__hydraulic_time_header__", is_transient)
            self._set_row_visible("HYDRAULIC_INPUT", "hyd_time_start", is_transient)
            self._set_row_visible("HYDRAULIC_INPUT", "hyd_dt", is_transient)
            self._set_row_visible("HYDRAULIC_INPUT", "hyd_time_end", is_transient)
            self._set_row_visible("HYDRAULIC_INPUT", "hydraulicClosureModel", True)
            self._set_row_visible("HYDRAULIC_INPUT", "__loaded_nodes__", True)
            self._set_row_visible("HYDRAULIC_INPUT", "__loaded_cells__", True)
            self._refresh_loaded_mesh_counts()
            self._refresh_hydraulic_labels()
            location_mode_var = self.editor_vars.get("PARTICLE_SETTINGS", {}).get("location_mode")
            location_mode = location_mode_var.get() if location_mode_var is not None else "coordinates"
            if location_mode == "click":
                self._set_release_coordinate_state(editable=False)
            else:
                self._set_release_coordinate_state(editable=True)
            release_mode_var = self.editor_vars.get("PARTICLE_SETTINGS", {}).get("release_mode")
            release_mode = release_mode_var.get() if release_mode_var is not None else "bulk"
            self._set_row_visible("PARTICLE_SETTINGS", "release_dt", release_mode == "continuous")
            biofouling_var = self.editor_vars.get("PARTICLE_EVO_SETTINGS", {}).get("biofouling")
            biofouling_mode = biofouling_var.get() if biofouling_var is not None else "off"
            evo_enabled = str(biofouling_mode).strip().lower() == "on"
            self._set_row_visible("PARTICLE_EVO_SETTINGS", "BT0", evo_enabled)
            self._set_row_visible("PARTICLE_EVO_SETTINGS", "BR", evo_enabled)
            self._set_row_visible("PARTICLE_EVO_SETTINGS", "rho_biofilm", evo_enabled)
            degradation_var = self.editor_vars.get("PARTICLE_EVO_SETTINGS", {}).get("degradation")
            degradation_mode = degradation_var.get() if degradation_var is not None else "off"
            degradation_enabled = str(degradation_mode).strip().lower() == "on"
            self._set_row_visible("PARTICLE_EVO_SETTINGS", "DR", degradation_enabled)
            self._set_row_visible("TRANSPORT_SETTINGS", "TL_horizontal", transport_model == "langevin")
            self._set_row_visible("TRANSPORT_SETTINGS", "TL_vertical", transport_model == "langevin")
            self._set_row_visible("BOUNDARY_SETTINGS", "surfaceVerticalDragCoeff", show_surface_drag_coeff)
            self._set_row_visible("BOUNDARY_SETTINGS", "surfaceContactAngleDeg", show_surface_contact_angle)
            self._set_row_visible("BOUNDARY_SETTINGS", "surfaceDetachmentSigmaStar", show_surface_detachment_sigma)
            self._set_row_visible("BOUNDARY_SETTINGS", "tanphi_ratio", show_tanphi_ratio)
            self._set_row_visible("BOUNDARY_SETTINGS", "bedEntrainmentSigmaStar", show_bed_entrainment_sigma)
            uphill_var = self.editor_vars.get("BOUNDARY_SETTINGS", {}).get("uphillPolicy")
            uphill_policy = uphill_var.get() if uphill_var is not None else "off"
            self._set_row_visible("BOUNDARY_SETTINGS", "dzUpMax", str(uphill_policy).strip().lower() == "stop")
            self._refresh_transport_labels()
            self._sync_uphill_step_control()
            self._sync_save_click_points_control()
            return

        self._set_row_visible("HYDRAULIC_INPUT", "__steady_header__", self.case_type == "synthetic" and not is_transient)
        self._set_row_visible("HYDRAULIC_INPUT", "U", self.case_type == "synthetic" and not is_transient)
        self._set_row_visible("HYDRAULIC_INPUT", "V", False)
        self._set_row_visible("HYDRAULIC_INPUT", "__transient_header__", self.case_type == "synthetic" and is_transient)
        for key in ("transient_shape", "switch_fraction"):
            self._set_row_visible("HYDRAULIC_INPUT", key, self.case_type == "synthetic" and is_transient)
        self._set_row_visible("HYDRAULIC_INPUT", "U0", self.case_type == "synthetic" and is_transient)
        self._set_row_visible("HYDRAULIC_INPUT", "U1", self.case_type == "synthetic" and is_transient)
        self._set_row_visible("HYDRAULIC_INPUT", "__hydraulic_time_header__", self.case_type == "synthetic" and is_transient)
        self._set_row_visible("HYDRAULIC_INPUT", "hyd_time_start", self.case_type == "synthetic" and is_transient)
        self._set_row_visible("HYDRAULIC_INPUT", "hyd_dt", self.case_type == "synthetic" and is_transient)
        self._set_row_visible("HYDRAULIC_INPUT", "hyd_time_end", self.case_type == "synthetic" and is_transient)
        self._set_row_visible("HYDRAULIC_INPUT", "V0", False)
        self._set_row_visible("HYDRAULIC_INPUT", "V1", False)
        self._set_row_visible("HYDRAULIC_INPUT", "__loaded_nodes__", self.case_type == "synthetic")
        self._set_row_visible("HYDRAULIC_INPUT", "__loaded_cells__", self.case_type == "synthetic")
        if self.case_type == "synthetic":
            self._refresh_loaded_mesh_counts()
        self._refresh_hydraulic_labels()

        location_mode_var = self.editor_vars.get("PARTICLE_SETTINGS", {}).get("location_mode")
        location_mode = location_mode_var.get() if location_mode_var is not None else "coordinates"
        if location_mode == "click":
            self._set_release_coordinate_state(editable=False)
        else:
            self._set_release_coordinate_state(editable=True)
        release_mode_var = self.editor_vars.get("PARTICLE_SETTINGS", {}).get("release_mode")
        release_mode = release_mode_var.get() if release_mode_var is not None else "bulk"
        self._set_row_visible("PARTICLE_SETTINGS", "release_dt", release_mode == "continuous")
        biofouling_var = self.editor_vars.get("PARTICLE_EVO_SETTINGS", {}).get("biofouling")
        biofouling_mode = biofouling_var.get() if biofouling_var is not None else "off"
        evo_enabled = str(biofouling_mode).strip().lower() == "on"
        self._set_row_visible("PARTICLE_EVO_SETTINGS", "BT0", evo_enabled)
        self._set_row_visible("PARTICLE_EVO_SETTINGS", "BR", evo_enabled)
        self._set_row_visible("PARTICLE_EVO_SETTINGS", "rho_biofilm", evo_enabled)
        degradation_var = self.editor_vars.get("PARTICLE_EVO_SETTINGS", {}).get("degradation")
        degradation_mode = degradation_var.get() if degradation_var is not None else "off"
        degradation_enabled = str(degradation_mode).strip().lower() == "on"
        self._set_row_visible("PARTICLE_EVO_SETTINGS", "DR", degradation_enabled)
        self._set_row_visible("TRANSPORT_SETTINGS", "TL_horizontal", transport_model == "langevin")
        self._set_row_visible("TRANSPORT_SETTINGS", "TL_vertical", transport_model == "langevin")
        self._set_row_visible("BOUNDARY_SETTINGS", "surfaceVerticalDragCoeff", show_surface_drag_coeff)
        self._set_row_visible("BOUNDARY_SETTINGS", "surfaceContactAngleDeg", show_surface_contact_angle)
        self._set_row_visible("BOUNDARY_SETTINGS", "surfaceDetachmentSigmaStar", show_surface_detachment_sigma)
        self._set_row_visible("BOUNDARY_SETTINGS", "tanphi_ratio", show_tanphi_ratio)
        self._set_row_visible("BOUNDARY_SETTINGS", "bedEntrainmentSigmaStar", show_bed_entrainment_sigma)
        uphill_var = self.editor_vars.get("BOUNDARY_SETTINGS", {}).get("uphillPolicy")
        uphill_policy = uphill_var.get() if uphill_var is not None else "off"
        self._set_row_visible("BOUNDARY_SETTINGS", "dzUpMax", str(uphill_policy).strip().lower() == "stop")
        self._refresh_transport_labels()
        self._sync_uphill_step_control()
        self._sync_save_click_points_control()

    def _refresh_hydraulic_labels(self) -> None:
        """Refresh dynamic labels in the hydraulics tab after mode changes."""
        widgets = self.section_widgets.get("HYDRAULIC_INPUT", {})
        for key in ("U", "U0", "U1", "V", "V0", "V1"):
            row_widgets = widgets.get(key, [])
            if not row_widgets:
                continue
            label_widget = row_widgets[0]
            if isinstance(label_widget, ttk.Label):
                text = self._hydraulic_field_label(key)
                label_widget.configure(text=text)

        self._update_particle_shape_fields()

    def _set_release_coordinate_state(self, *, editable: bool) -> None:
        release_vars = self.editor_vars.get("PARTICLE_SETTINGS", {})
        release_entries = self.section_entries.get("PARTICLE_SETTINGS", {})
        current_x = release_vars.get("release_x")
        current_y = release_vars.get("release_y")

        for key in ("release_x", "release_y"):
            var = release_vars.get(key)
            entry = release_entries.get(key)
            if var is None or entry is None:
                continue

            if editable:
                self._set_row_visible("PARTICLE_SETTINGS", key, True)
                if var.get() == CLICK_RELEASE_PLACEHOLDER:
                    fallback_x, fallback_y = self._fallback_release_coordinate_texts()
                    var.set(fallback_x if key == "release_x" else fallback_y)
                if isinstance(entry, ttk.Entry):
                    entry.configure(state="normal")
            else:
                if current_x is not None and current_y is not None:
                    if (
                        current_x.get() != CLICK_RELEASE_PLACEHOLDER
                        and current_y.get() != CLICK_RELEASE_PLACEHOLDER
                        and current_x.get().strip()
                        and current_y.get().strip()
                    ):
                        self.last_manual_release_coords[self.case_type] = (current_x.get(), current_y.get())
                var.set(CLICK_RELEASE_PLACEHOLDER)
                self._set_row_visible("PARTICLE_SETTINGS", key, False)
                if isinstance(entry, ttk.Entry):
                    entry.configure(state="readonly")

    def _pick_xdmf_file(self, target_var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title=self._hydraulic_file_dialog_title(),
            filetypes=self._hydraulic_filetypes(),
            initialdir=self._browse_initial_dir(),
        )
        if path:
            self._last_browse_dir = Path(path).expanduser().resolve().parent
            target_var.set(_display_path(path))
            if self.workflow_mode == "new" and self._uses_external_hydraulics():
                xdmf_path = Path(path).expanduser().resolve()
                if self.case_type == "basement":
                    inferred_output_dir = xdmf_path.parent.parent if xdmf_path.parent.name.lower() == "input" else xdmf_path.parent
                else:
                    inferred_output_dir = xdmf_path.parent
                self._set_output_dir(inferred_output_dir, locked=True)

    def _pick_top_level_xdmf_file(self) -> None:
        self._pick_xdmf_file(self.xdmf_var)
        self._sync_selector_state()

    def _choose_output_dir(self) -> None:
        if self.output_dir_locked:
            return
        path = filedialog.askdirectory(
            title="Select HydroLPT output folder",
            initialdir=self._browse_initial_dir(),
        )
        if path:
            resolved = Path(path).expanduser().resolve()
            self._last_browse_dir = resolved
            self.output_dir_var.set(_display_path(resolved))

    def _collect_case_spec(self) -> dict[str, Any]:
        sections: dict[str, dict[str, Any]] = {}
        plots: dict[str, Any] = {}
        xdmf_path = ""
        if self._uses_external_hydraulics():
            if self.workflow_mode == "base":
                xdmf_path = self.loaded_base_xdmf_path.strip()
            else:
                xdmf_path = str(_parse_value(self.xdmf_var.get())).strip().strip("'\"")
        for title, vars_for_section in self.editor_vars.items():
            parsed = {
                key: _parse_value(_from_display_option_value(key, var.get()))
                for key, var in vars_for_section.items()
            }
            if title == "PLOTS":
                plots = self._build_plot_spec(parsed, sections)
                continue
            if title == "PARTICLE_SETTINGS":
                shape = str(parsed.get("shape", "sphere")).lower()
                if shape == "sphere":
                    parsed.pop("I", None)
                    parsed.pop("S", None)
                elif shape == "cylinder":
                    parsed.pop("I", None)
                elif shape == "disk":
                    parsed.pop("I", None)
            if title == "PARTICLE_SETTINGS":
                location_mode = self.editor_vars.get("PARTICLE_SETTINGS", {}).get("location_mode")
                if location_mode is not None and location_mode.get() == "click":
                    parsed.pop("release_x", None)
                    parsed.pop("release_y", None)
                    parsed["manual_centers"] = []
                else:
                    raw_release_x = parsed.pop("release_x", "")
                    raw_release_y = parsed.pop("release_y", "")
                    release_x_values = self._parse_coordinate_series(raw_release_x, "Release x-coordinate")
                    release_y_values = self._parse_coordinate_series(raw_release_y, "Release y-coordinate")

                    if not release_x_values and not release_y_values:
                        fallback_x, fallback_y = self._fallback_release_coordinate_texts()
                        release_x_values = self._parse_coordinate_series(
                            fallback_x,
                            "Release x-coordinate",
                        )
                        release_y_values = self._parse_coordinate_series(
                            fallback_y,
                            "Release y-coordinate",
                        )

                    if not release_x_values or not release_y_values:
                        raise ValueError(
                            "Release x-coordinate and y-coordinate must both be provided."
                        )
                    if len(release_x_values) != len(release_y_values):
                        raise ValueError(
                            "Release x-coordinate and y-coordinate must contain the same number of comma-separated values."
                        )

                    parsed["manual_centers"] = [
                        [release_x, release_y]
                        for release_x, release_y in zip(release_x_values, release_y_values)
                    ]
                    self.last_manual_release_coords[self.case_type] = (
                        ", ".join(f"{value:g}" for value in release_x_values),
                        ", ".join(f"{value:g}" for value in release_y_values),
                    )
            sections[title] = parsed

        particle_settings = sections.get("PARTICLE_SETTINGS", {})
        hydraulic_settings = sections.get("HYDRAULIC_INPUT", {})
        run_mode = str(hydraulic_settings.get("run_mode", "steady")).strip().lower()
        is_transient = run_mode == "transient"

        transport_settings = sections.get("TRANSPORT_SETTINGS", {})
        dt = float(transport_settings.get("dt", 0.0))
        output_dt = float(plots.get("output_dt", dt))
        if dt <= 0.0:
            raise ValueError("Particle timestep dt must be > 0.")
        if output_dt <= 0.0:
            raise ValueError("Output timestep dt_out must be > 0.")
        ratio = output_dt / dt
        output_stride = int(round(ratio))
        if output_dt < dt - 1e-12 or output_stride < 1 or not math.isclose(ratio, output_stride, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                "Output timestep dt_out must be an integer multiple of particle timestep dt "
                f"(dt_out={output_dt:g} s, dt={dt:g} s)."
            )
        plots["output_dt"] = output_stride * dt

        particle = build_particle_properties(
            shape=particle_settings.get("shape", "sphere"),
            rho_p=float(particle_settings.get("rho_p", 1010.0)),
            L=float(particle_settings.get("L")),
            I=particle_settings.get("I"),
            S=particle_settings.get("S"),
            rho_f=float(hydraulic_settings.get("rho_f", 1000.0)),
            nu=float(hydraulic_settings.get("nu", 1e-6)),
            g=float(hydraulic_settings.get("g", 9.81)),
        )
        hmin = float(sections.get("BOUNDARY_SETTINGS", {}).get("hmin", 0.0))
        hmin_max = 2.0 * float(particle.DV)
        if hmin < 0.0:
            raise ValueError("Wet threshold h_min must be >= 0.")
        if hmin > hmin_max:
            raise ValueError(
                "Wet threshold h_min must be <= 2*D_V for the selected particle "
                f"(h_min={hmin:g} m, D_V={particle.DV:g} m, 2*D_V={hmin_max:g} m)."
            )
        boundary_settings = sections.get("BOUNDARY_SETTINGS", {})
        if particle.ws < 0.0:
            boundary_settings["bedPolicy"] = "always_reflect"
            bed_var = self.editor_vars.get("BOUNDARY_SETTINGS", {}).get("bedPolicy")
            if bed_var is not None:
                bed_var.set(_to_display_option_value("bedPolicy", "always_reflect"))
        elif particle.ws == 0.0:
            boundary_settings["bedPolicy"] = "always_reflect"
            boundary_settings["surfacePolicy"] = "always_reflect"
            bed_var = self.editor_vars.get("BOUNDARY_SETTINGS", {}).get("bedPolicy")
            surface_var = self.editor_vars.get("BOUNDARY_SETTINGS", {}).get("surfacePolicy")
            if bed_var is not None:
                bed_var.set(_to_display_option_value("bedPolicy", "always_reflect"))
            if surface_var is not None:
                surface_var.set("always_reflect")
        surface_policy = str(boundary_settings.get("surfacePolicy", "always_reflect")).strip().lower()
        if surface_policy == "probabilistic_detachment":
            sigma_star = float(boundary_settings.get("surfaceDetachmentSigmaStar", 0.2))
            if not math.isfinite(sigma_star) or sigma_star <= 0.0:
                raise ValueError("Surface detachment sigma* must be a finite positive value.")
            boundary_settings["surfaceDetachmentSigmaStar"] = sigma_star
        else:
            boundary_settings.pop("surfaceDetachmentSigmaStar", None)
        if surface_policy in {"deterministic_detachment", "probabilistic_detachment"}:
            contact_angle_deg = float(boundary_settings.get("surfaceContactAngleDeg", 105.0))
            if not math.isfinite(contact_angle_deg) or contact_angle_deg <= 0.0 or contact_angle_deg >= 180.0:
                raise ValueError("Surface contact angle must be a finite value between 0 and 180 degrees.")
            boundary_settings["surfaceContactAngleDeg"] = contact_angle_deg
        else:
            boundary_settings.pop("surfaceContactAngleDeg", None)
        bed_policy = str(boundary_settings.get("bedPolicy", "always_reflect")).strip().lower()
        if bed_policy == "probabilistic_entrainment":
            sigma_star = float(boundary_settings.get("bedEntrainmentSigmaStar", 0.2))
            if not math.isfinite(sigma_star) or sigma_star <= 0.0:
                raise ValueError("Bed entrainment sigma* must be a finite positive value.")
            boundary_settings["bedEntrainmentSigmaStar"] = sigma_star
        else:
            boundary_settings.pop("bedEntrainmentSigmaStar", None)
        if not is_transient:
            transport_settings.pop("tRelease", None)
            for key in ("hyd_time_start", "hyd_dt", "hyd_time_end"):
                hydraulic_settings.pop(key, None)

        release_settings = sections.get("PARTICLE_SETTINGS", {})
        release_mode = str(release_settings.get("release_mode", "bulk")).strip().lower()
        if release_mode != "continuous":
            release_settings.pop("release_dt", None)

        if self.case_type == "synthetic":
            hydraulic = sections.get("HYDRAULIC_INPUT", {})
            length = float(hydraulic["length"])
            domain_width = float(hydraulic["domain_width"])
            hydraulic["nx"] = max(1, int(round(length / FIXED_SYNTHETIC_CELL_SIZE_X)))
            hydraulic["ny"] = max(1, int(round(domain_width / FIXED_SYNTHETIC_CELL_SIZE_Y)))
            hydraulic.setdefault("hydraulicClosureModel", "standard_z0")
            hydraulic["hydraulicPrimaryVariable"] = "U"

            if is_transient:
                hydraulic.pop("U", None)
                hydraulic.pop("ustar", None)
                hydraulic.pop("V", None)
            else:
                for key in ("transient_shape", "switch_fraction", "U0", "U1", "ustar0", "ustar1", "V0", "V1"):
                    hydraulic.pop(key, None)

            hydraulic.pop("ustar", None)
            hydraulic.pop("ustar0", None)
            hydraulic.pop("ustar1", None)
        elif self.case_type in {"basement", "hecras"}:
            hydraulic = sections.setdefault("HYDRAULIC_INPUT", {})
            hydraulic["hydraulicClosureModel"] = "external_adapter"
        duration_limited = False
        if release_settings.get("release_mode", "bulk") == "continuous":
            t_track = float(transport_settings.get("tTrack", 0.0))
            release_dt = float(release_settings.get("release_dt", 0.0))
            n_per_center = int(release_settings.get("n_per_center", 0))

            if release_dt <= 0.0:
                raise ValueError("Inter-release time dT must be > 0 for continuous release.")
            if t_track < 0.0:
                raise ValueError("Tracking duration T_track must be >= 0.")

            max_n_per_center = max(1, int(math.floor(t_track / release_dt)) + 1)
            if n_per_center > max_n_per_center:
                release_settings["n_per_center"] = max_n_per_center
                release_var = self.editor_vars.get("PARTICLE_SETTINGS", {}).get("n_per_center")
                if release_var is not None:
                    release_var.set(str(max_n_per_center))
                n_per_center = max_n_per_center
                duration_limited = True

        if run_mode == "transient":
            hyd_start = float(hydraulic_settings.get("hyd_time_start", 0.0))
            hyd_dt = float(hydraulic_settings.get("hyd_dt", 1.0))
            hyd_end = float(hydraulic_settings.get("hyd_time_end", hyd_start))
            t_release = float(transport_settings.get("tRelease", 0.0))
            t_track = float(transport_settings.get("tTrack", 0.0))

            if hyd_dt <= 0.0:
                raise ValueError("Hydraulic time step dt_h must be > 0 for transient runs.")

            required_end = max(hyd_start + hyd_dt, t_release + t_track)
            if hyd_end < required_end:
                raise ValueError(
                    "Transient hydraulic end time t_end is too small. "
                    f"It must be at least {required_end:.3f} s for the selected release/tracking window."
                )

        clamp_message = self._clamp_release_tracking_count(
            sections,
            duration_limited=duration_limited,
        )
        if clamp_message:
            self._append_log(f"\n{clamp_message}")

        output_dir_raw = self.output_dir_var.get().strip()
        if not output_dir_raw:
            raise ValueError("Please choose a run folder first.")
        output_dir = Path(output_dir_raw).expanduser().resolve()
        schema = case_schema_from_payload(
            case_type=self.case_type,
            sections=sections,
            plots=plots,
            export_data=bool(self.export_data_var.get()),
            xdmf_path=xdmf_path if self._uses_external_hydraulics() else "",
            user_notes=getattr(self.current_schema, "user_notes", ""),
        )
        self.current_schema = schema
        spec = case_schema_to_payload(schema)
        spec["run_name"] = self._current_run_name()
        spec["output_dir"] = str(output_dir)
        spec["preserve_clicked_release_points"] = bool(self.save_click_points_var.get())
        spec["window_anchor"] = {
            "x": self.winfo_rootx(),
            "y": self.winfo_rooty(),
            "width": self.winfo_width(),
            "height": self.winfo_height(),
            "screen_width": self.winfo_screenwidth(),
            "screen_height": self.winfo_screenheight(),
        }
        if self.base_case_path is not None:
            spec["base_script_path"] = str(self.base_case_path)
        return spec

    def _build_plot_spec(self, parsed: dict[str, Any], sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
        plots: dict[str, Any] = {}
        if self.base_case_path is not None:
            try:
                plots.update(load_case_definition(self.base_case_path).get("plots", {}))
            except Exception:
                pass

        for key in ("binary_map", "trajectories"):
            if key in parsed:
                plots[key] = parsed[key]

        for key in (
            "output_dt",
            "binary_map_marker_size",
            "binary_map_mesh_overlay",
            "advection_diffusion",
        ):
            if key in parsed:
                plots[key] = parsed[key]

        return plots

    def _should_show_plots(self, spec: dict[str, Any]) -> bool:
        plots = spec.get("plots", {})
        plot_keys = ("binary_map", "trajectories")
        return any(bool(plots.get(key, False)) for key in plot_keys)

    def _has_active_run(self) -> bool:
        return (self.run_thread is not None and self.run_thread.is_alive()) or (
            self.process is not None and self.process.poll() is None
        )

    def _start_case(self, spec: dict[str, Any]) -> None:
        output_dir = Path(spec["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.log_text.delete("1.0", tk.END)
        self._append_log(f"Running {CASE_LABELS[self.case_type]}\n")
        self._append_log(f"Output folder: {output_dir}\n\n")
        if self.base_case_path is not None:
            self._append_log(f"Base case: {self.base_case_path}\n")
        export_enabled = bool(spec.get("export_data", False))
        self._append_log(f"Data export: {'enabled' if export_enabled else 'disabled'}.\n")
        if "output_dt" in spec.get("plots", {}):
            self._append_log(f"Output timestep dt_out: {float(spec['plots']['output_dt']):g} s.\n")
        self._append_log("HydroLPT.py export enabled.\n\n")
        self.status_var.set("Running")
        self.stop_requested = False
        self._set_run_ui_state(True)
        self._run_case_in_subprocess(spec)

    def _fail_run_validation(self, message: str) -> None:
        self.status_var.set("Failed")
        self._append_log(f"\nRun failed: {message}\n")

    def _fail_save_validation(self, message: str) -> None:
        self.status_var.set("Failed")
        self._append_log(f"\nSave failed: {message}\n")

    def _run_case(self) -> None:
        if self.workflow_mode == "new" and self.case_type == "synthetic" and not self.output_dir_var.get().strip():
            self._fail_run_validation("Please choose a run folder first.")
            return

        if self.workflow_mode == "new" and self._uses_external_hydraulics():
            xdmf_text = str(_parse_value(self.xdmf_var.get())).strip().strip("'\"")
            if not xdmf_text:
                self._fail_run_validation(self._hydraulic_file_prompt())
                return
            try:
                xdmf_path = Path(xdmf_text).expanduser().resolve()
            except OSError:
                xdmf_path = None
            if not self._is_valid_hydraulic_file_path(xdmf_path):
                self._fail_run_validation(self._hydraulic_file_prompt())
                return

        if self.workflow_mode == "base":
            base_case_path = self.base_case_path
            if (
                base_case_path is None
                or not base_case_path.exists()
                or base_case_path.suffix.lower() != ".py"
            ):
                self._fail_run_validation("Please choose a valid HydroLPT.py file first.")
                return

        try:
            spec = self._collect_case_spec()
        except ValueError as exc:
            self._fail_run_validation(str(exc))
            return
        if self._uses_external_hydraulics() and not spec.get("xdmf_path"):
            self._fail_run_validation(self._hydraulic_file_prompt())
            return

        self._close_lingering_plot_process()

        if self._has_active_run():
            self._pending_run_spec = spec
            self._suppress_next_run_exit_message = True
            self._append_log("\nClosing current run and starting fresh...\n")
            if self.process is not None and self.process.poll() is None:
                self.status_var.set("Stopping")
                self.process.terminate()
            else:
                self._suppress_next_run_exit_message = False
                self._pending_run_spec = None
                self.status_var.set("Running")
                self._append_log("Current run cannot be restarted until it finishes.\n")
            return

        self._start_case(spec)

    def _save_case(self) -> None:
        if (self.run_thread is not None and self.run_thread.is_alive()) or (
            self.process is not None and self.process.poll() is None
        ):
            self.status_var.set("Running")
            self._append_log("\nSave failed: Please wait for the current run to finish before saving.\n")
            return

        if self.workflow_mode == "new" and self.case_type == "synthetic" and not self.output_dir_var.get().strip():
            self._fail_save_validation("Please choose a run folder first.")
            return

        if self.workflow_mode == "new" and self._uses_external_hydraulics():
            xdmf_text = str(_parse_value(self.xdmf_var.get())).strip().strip("'\"")
            if not xdmf_text:
                self._fail_save_validation(self._hydraulic_file_prompt())
                return
            try:
                xdmf_path = Path(xdmf_text).expanduser().resolve()
            except OSError:
                xdmf_path = None
            if not self._is_valid_hydraulic_file_path(xdmf_path):
                self._fail_save_validation(self._hydraulic_file_prompt())
                return

        if self.workflow_mode == "base":
            base_case_path = self.base_case_path
            if (
                base_case_path is None
                or not base_case_path.exists()
                or base_case_path.suffix.lower() != ".py"
            ):
                self._fail_save_validation("Please choose a valid HydroLPT.py file first.")
                return

        try:
            spec = self._collect_case_spec()
            if self._uses_external_hydraulics() and not spec.get("xdmf_path"):
                self._fail_save_validation(self._hydraulic_file_prompt())
                return
            normalized_case = normalize_case_spec(spec)
            script_path = save_case_file(spec)
        except ValueError as exc:
            self._fail_save_validation(str(exc))
            return
        except Exception as exc:
            self.status_var.set("Failed")
            self._append_log(f"\nSave failed: {exc}\n")
            return

        self.status_var.set("Saved")
        self._apply_normalized_case_sections(normalized_case)
        self._adopt_saved_case_path(script_path)
        self._append_log(f"\nSaved case script: {script_path}\n")

    def _adopt_saved_case_path(self, script_path: Path) -> None:
        """Reload the newly saved case so the GUI reflects the file on disk."""
        if self.base_case_path is None:
            return

        resolved_script_path = Path(script_path).expanduser().resolve()
        try:
            if resolved_script_path == Path(self.base_case_path).expanduser().resolve():
                return
        except OSError:
            pass
        current_tab_index = self.notebook.index(self.notebook.select()) if self.notebook.tabs() else 0
        self._load_base_case(resolved_script_path)
        tabs = self.notebook.tabs()
        if tabs:
            self.notebook.select(min(current_tab_index, len(tabs) - 1))

    def _run_case_in_subprocess(self, spec: dict[str, Any]) -> None:
        spec["export_case_files"] = True
        spec["show_plots"] = self._should_show_plots(spec)
        fd, temp_path = tempfile.mkstemp(prefix="hydrolpt_gui_spec_", suffix=".json")
        os.close(fd)
        spec_path = Path(temp_path)
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

        cmd = self._subprocess_command(spec_path)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        self._append_log("Starting HydroLPT run process...\n")
        self._run_token_counter += 1
        run_token = self._run_token_counter
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                bufsize=0,
                cwd=self._subprocess_cwd(),
                env=env,
            )
        except Exception as exc:
            spec_path.unlink(missing_ok=True)
            self.status_var.set("Failed")
            self._set_run_ui_state(False)
            self._append_log(f"\nRun failed to start: {exc}\n")
            return
        self.process = process
        self.temp_spec_path = spec_path
        self._active_run_token = run_token
        self.run_thread = threading.Thread(
            target=self._stream_process_output,
            args=(process, spec_path, run_token),
            daemon=True,
        )
        self.run_thread.start()

    def _subprocess_command(self, spec_path: Path) -> list[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, RUN_SPEC_FLAG, str(spec_path)]
        return [sys.executable, "-u", "-m", "gui.gui_runner", str(spec_path)]

    def _subprocess_cwd(self) -> Path:
        if getattr(sys, "frozen", False):
            return Path.cwd()
        return Path(__file__).resolve().parents[1]

    def _stream_process_output(self, process: subprocess.Popen[bytes], spec_path: Path, run_token: int) -> None:
        assert process.stdout is not None
        plots_opened = False
        for chunk in iter(process.stdout.readline, b""):
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            self.after(0, self._append_log, text)
            if (not plots_opened) and ("Opening plots..." in text):
                plots_opened = True
                self.after(0, self._on_plots_opened, run_token, process, spec_path)
        return_code = process.wait()
        self.after(0, self._on_subprocess_complete, return_code, run_token, spec_path, process)

    def _execute_case_thread(self, spec: dict[str, Any]) -> None:
        try:
            script_path, config_json_path, config_txt_path, _run_info = execute_case_spec(
                spec,
                log_callback=self._append_log_threadsafe,
                show_plots=False,
                export_case_files=True,
            )
        except Exception as exc:
            self.after(0, self._on_run_failed, str(exc))
            return

        self.after(0, self._on_run_complete, script_path, config_json_path, config_txt_path)

    def _on_plots_opened(
        self,
        run_token: int,
        process: subprocess.Popen[bytes],
        spec_path: Path,
    ) -> None:
        if self._active_run_token != run_token or self.process is not process:
            return

        self._apply_saved_click_points_from_temp_spec(spec_path)
        self._plots_opened_tokens.add(run_token)
        self.status_var.set("Completed")
        self._append_log(
            "\nSimulation finished. Plot windows remain open, and the run folder may stay locked until those windows are closed.\n"
        )
        self._lingering_plot_process = process
        self.process = None
        self.run_thread = None
        self.temp_spec_path = None
        self._active_run_token = None
        self._set_run_ui_state(False)

    def _on_subprocess_complete(
        self,
        return_code: int,
        run_token: int,
        spec_path: Path,
        process: subprocess.Popen[bytes],
    ) -> None:
        self._apply_saved_click_points_from_temp_spec(spec_path)
        spec_path.unlink(missing_ok=True)

        if self._suppress_next_run_exit_message:
            self._suppress_next_run_exit_message = False
            if self.process is process:
                self.process = None
            self.run_thread = None
            self.temp_spec_path = None
            self._active_run_token = None
            pending_spec = self._pending_run_spec
            self._pending_run_spec = None
            if pending_spec is not None:
                self.after(0, self._start_case, pending_spec)
            else:
                self._set_run_ui_state(False)
            return

        if run_token in self._plots_opened_tokens:
            self._plots_opened_tokens.discard(run_token)
            if self._lingering_plot_process is process and process.poll() is not None:
                self._lingering_plot_process = None
            return

        if self._active_run_token != run_token and self.process is not process:
            return

        if return_code == 0:
            self.status_var.set("Completed")
            output_dir = Path(self.output_dir_var.get()).expanduser().resolve()
            generated_script_path = self.base_case_path if self.base_case_path is not None else output_dir / GUI_CASE_FILENAME
            self._adopt_saved_case_path(generated_script_path)
            self._append_log(f"\nGenerated case script: {generated_script_path}\n")
            self._append_log("\nRun finished successfully.\n")
        else:
            self.status_var.set("Failed")
            self._append_log(f"\nRun process exited with code {return_code}.\n")

        if self.process is process:
            self.process = None
        if self._lingering_plot_process is process:
            self._lingering_plot_process = None
        self.run_thread = None
        self.temp_spec_path = None
        self._active_run_token = None
        self._set_run_ui_state(False)

    def _on_run_complete(
        self,
        script_path: Path | None,
        config_json_path: Path | None,
        config_txt_path: Path | None,
    ) -> None:
        self.status_var.set("Completed")
        if script_path is not None:
            self._adopt_saved_case_path(script_path)
            self._append_log(f"\nGenerated case script: {script_path}\n")
        self._append_log("\nRun finished successfully.\n")
        self.run_thread = None
        self._set_run_ui_state(False)

    def _on_run_failed(self, message: str) -> None:
        self.status_var.set("Failed")
        self._append_log(f"\nRun failed: {message}\n")
        self.run_thread = None
        self._set_run_ui_state(False)

    def _stop_case(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.status_var.set("Stopping")
            return

        if self.run_thread is None or not self.run_thread.is_alive():
            self.status_var.set("Ready")
            return
        self.stop_requested = True
        self.status_var.set("Running")
        self._append_log(
            "\nStop requested: in-process cancellation is not wired yet. Close plot windows if any appear and wait for the current run to finish.\n"
        )


def main() -> None:
    """Launch the HydroLPT desktop GUI."""
    _configure_windows_app_id()
    app = HydroLPTGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

