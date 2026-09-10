"""Shared constants and schema metadata for GUI case management."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT_DIR / "runs"
GUI_CASE_FILENAME = "HydroLPT_v0.9.py"
SECTION_EXPORT_ORDER = [
    "HYDRAULIC_INPUT",
    "PARTICLE_SETTINGS",
    "PARTICLE_EVO_SETTINGS",
    "TRANSPORT_SETTINGS",
    "BOUNDARY_SETTINGS",
]
SECTION_FIELD_ORDER: dict[str, list[str]] = {
    "HYDRAULIC_INPUT": [
        "length", "domain_width", "wetted_width", "water_depth",
        "run_mode", "hydraulicPrimaryVariable", "U", "V",
        "hydraulicClosureModel", "z0", "transient_shape", "switch_fraction",
        "U0", "V0", "U1", "V1",
        "hyd_time_start", "hyd_dt", "hyd_time_end",
        "rho_f", "nu", "g", "nx", "ny",
    ],
    "PARTICLE_SETTINGS": [
        "shape", "L", "I", "S", "rho_p", "release_mode", "release_dt",
        "location_mode", "release_x", "release_y", "manual_centers",
        "n_per_center", "nTrack", "releaseSigma", "zFrac",
    ],
    "PARTICLE_EVO_SETTINGS": [
        "biofouling", "BT0", "BR", "rho_biofilm",
        "degradation", "DR",
    ],
    "TRANSPORT_SETTINGS": [
        "tRelease", "dt", "tTrack", "hydraulicFieldMode", "transportVelocityMode", "transportModel",
        "rng_seed", "useRWx", "useRWy", "useRWz", "betaKh", "KhMax", "rAniso", "alphaKz", "KzMax",
        "TL_horizontal", "TL_vertical",
    ],
    "BOUNDARY_SETTINGS": [
        "surfacePolicy", "surfaceDetachmentSigmaStar", "surfaceVerticalDragCoeff", "surfaceContactAngleDeg",
        "bedPolicy", "bedEntrainmentSigmaStar", "tanphi_ratio", "dryPolicy", "hmin",
        "outsidePolicy", "uphillPolicy", "dzUpMax",
    ],
    "PLOTS": [
        "binary_map", "trajectories", "binary_map_marker_size", "binary_map_mesh_overlay", "advection_diffusion",
    ],
}

SECTION_COMMENTS: dict[str, dict[str, str]] = {
    "HYDRAULIC_INPUT": {
        "run_mode": '"steady" or "transient"',
        "length": "flume length [m]",
        "domain_width": "total domain width [m]",
        "wetted_width": "wetted channel width [m]",
        "water_depth": "flow depth [m]",
        "rho_f": "fluid density [kg/m3]",
        "nu": "fluid kinematic viscosity [m2/s]",
        "g": "gravitational acceleration [m/s2]",
        "nx": "number of cells in x",
        "ny": "number of cells in y",
        "hydraulicPrimaryVariable": '"U"',
        "U": "steady synthetic streamwise velocity [m/s]",
        "V": "steady synthetic transverse velocity [m/s]",
        "z0": "roughness height for standard_z0 closure [m]",
        "transient_shape": '"step" or "linear" transition shape',
        "switch_fraction": "switch time as fraction of total transient duration [-]",
        "U0": "initial streamwise velocity [m/s]",
        "U1": "final streamwise velocity [m/s]",
        "hyd_time_start": "hydraulic simulation start time [s], used for transient runs",
        "hyd_dt": "hydraulic simulation timestep [s], used for transient runs",
        "hyd_time_end": "hydraulic simulation end time [s], used for transient runs",
        "V0": "initial transverse velocity [m/s]",
        "V1": "final transverse velocity [m/s]",
    },
    "PARTICLE_SETTINGS": {
        "location_mode": 'location mode: "coordinates" or "click"',
        "shape": 'particle shape: "sphere", "ellipsoid", "cylinder", "disk", or "prism"',
        "rho_p": "particle density [kg/m3]",
        "L": "particle longest axis [m]",
        "I": "particle intermediate axis [m]",
        "S": "particle shortest axis [m]",
        "manual_centers": "list of [x, y] release centers",
        "release_mode": '"bulk" or "continuous" release mode',
        "n_per_center": "particles released per center",
        "release_dt": "time gap between successive particles for continuous release [s]",
        "nTrack": "number of saved trajectories",
        "releaseSigma": "horizontal cloud spread [m]",
        "zFrac": "initial z as fraction of local depth (0=bed, 1=surface)",
    },
    "PARTICLE_EVO_SETTINGS": {
        "biofouling": '"off" or "on"',
        "BT0": "initial biofilm thickness [m]",
        "BR": "constant biofilm growth rate [m/s]",
        "rho_biofilm": "biofilm density [kg/m3]",
        "degradation": '"off" or "on"',
        "DR": "constant degradation rate [% of initial size per second]",
    },
    "TRANSPORT_SETTINGS": {
        "dt": "particle timestep [s]",
        "tTrack": "particle tracking duration [s]",
        "tRelease": "release time for transient runs [s]",
        "hydraulicFieldMode": '"cellwise" or "local_idw"',
        "transportVelocityMode": '"depth_averaged" or "loglaw_vertical"',
        "transportModel": '"random_walk" or "langevin"',
        "rng_seed": "random seed: None for a fresh seed each run, or an integer for reproducible release/random-walk paths",
        "useRWx": "streamwise random walk on/off",
        "useRWy": "transverse random walk on/off",
        "useRWz": "vertical random walk on/off",
        "betaKh": "horizontal diffusivity factor",
        "KhMax": "cap on horizontal diffusivity [m2/s]",
        "rAniso": "horizontal anisotropy ratio: K_parallel = rAniso K_h, K_perp = K_h/rAniso",
        "alphaKz": "vertical diffusivity factor",
        "KzMax": "cap on vertical diffusivity [m2/s]",
        "TL_horizontal": "horizontal Lagrangian timescale for Langevin mode [s]",
        "TL_vertical": "vertical Lagrangian timescale for Langevin mode [s]",
    },
    "BOUNDARY_SETTINGS": {
        "surfacePolicy": '"always_reflect", "always_stick", "deterministic_detachment", or "probabilistic_detachment"',
        "bedPolicy": '"always_reflect", "always_deposit", "reflect_if_ustar_gt_crit", or "probabilistic_entrainment"',
        "surfaceVerticalDragCoeff": "vertical drag coefficient used for surface detachment",
        "surfaceContactAngleDeg": "air-water-particle contact angle used for surface detachment [deg]",
        "surfaceDetachmentSigmaStar": "dimensionless spread sigma* for probabilistic surface detachment",
        "tanphi_ratio": "friction angle ratio for bed entrainment",
        "bedEntrainmentSigmaStar": "dimensionless spread sigma* for probabilistic bed entrainment",
        "outsidePolicy": 'currently "stop"',
        "dryPolicy": '"reflect", "stop", or "stick_active"',
        "hmin": "minimum wet depth threshold [m]",
        "uphillPolicy": '"off" or "stop"',
        "dzUpMax": "max allowed uphill bed jump [m]",
    },
    "PLOTS": {
        "binary_map": "plot release/end-state map",
        "binary_map_marker_size": "marker size for binary map",
        "binary_map_mesh_overlay": "show original mesh over smoothed binary map",
        "trajectories": "plot tracked particle trajectories",
        "advection_diffusion": "advection-diffusion diagnostic settings",
    },
}
OUTPUT_FIELD_ORDER = [
    "export_data", "output_dt", "binary_map", "trajectories", "binary_map_marker_size", "binary_map_mesh_overlay", "advection_diffusion",
]
OUTPUT_COMMENTS = {
    "export_data": "export CSV/JSON output files",
    "output_dt": "tracked trajectory output timestep [s]; must be a multiple of particle dt",
    "binary_map": "plot release/end-state map",
    "trajectories": "plot tracked particle trajectories",
    "binary_map_marker_size": "marker size for binary map",
    "binary_map_mesh_overlay": "show original mesh over smoothed binary map",
    "advection_diffusion": "advection-diffusion diagnostic settings",
}
SECTION_HEADING_GROUPS: dict[str, list[tuple[str, list[str]]]] = {
    "HYDRAULIC_INPUT": [
        ("Geometry", ["length", "domain_width", "wetted_width", "water_depth"]),
        ("Hydraulic Inputs", ["run_mode", "hydraulicPrimaryVariable", "U", "V", "hydraulicClosureModel", "z0", "transient_shape", "switch_fraction", "U0", "V0", "U1", "V1"]),
        ("Time settings", ["hyd_time_start", "hyd_dt", "hyd_time_end"]),
        ("Fluid properties", ["rho_f", "nu", "g"]),
        ("Mesh properties", ["nx", "ny"]),
    ],
    "PARTICLE_SETTINGS": [
        ("Particle properties", ["shape", "L", "I", "S", "rho_p"]),
        ("Particle release", ["release_mode", "release_dt", "location_mode", "release_x", "release_y", "manual_centers", "n_per_center", "nTrack", "releaseSigma", "zFrac"]),
    ],
    "PARTICLE_EVO_SETTINGS": [
        ("Particle evolution", ["biofouling", "BT0", "BR", "rho_biofilm", "degradation", "DR"]),
    ],
    "TRANSPORT_SETTINGS": [
        ("Time settings", ["tRelease", "dt", "tTrack"]),
        ("Advection and dispersion", ["hydraulicFieldMode", "transportVelocityMode", "transportModel", "rng_seed", "useRWx", "useRWy", "useRWz", "betaKh", "KhMax", "rAniso", "alphaKz", "KzMax", "TL_horizontal", "TL_vertical"]),
    ],
    "BOUNDARY_SETTINGS": [
        ("Boundary interaction framework", ["surfacePolicy", "surfaceDetachmentSigmaStar", "surfaceVerticalDragCoeff", "surfaceContactAngleDeg", "bedPolicy", "bedEntrainmentSigmaStar", "tanphi_ratio", "dryPolicy", "hmin", "outsidePolicy", "uphillPolicy", "dzUpMax"]),
    ],
    "OUTPUT_SETTINGS": [
        ("Export", ["export_data", "output_dt"]),
        ("Plotting", ["binary_map", "trajectories", "binary_map_marker_size", "binary_map_mesh_overlay", "advection_diffusion"]),
    ],
}
