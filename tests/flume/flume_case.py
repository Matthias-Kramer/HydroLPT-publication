from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core._01_hydraulic_input_and_adapter.synthetic import run_synthetic_case

HYDRAULIC_INPUT = {
    # Geometry
    "length": 20.0,  # flume length [m]
    "domain_width": 1.0,  # total domain width [m]
    "wetted_width": 0.6,  # wetted channel width [m]
    "water_depth": 0.5,  # flow depth [m]

    # Hydraulic Inputs
    "run_mode": "steady",  # "steady" or "transient"
    "hydraulicPrimaryVariable": "U",  # "U"
    "U": 0.3,  # steady synthetic streamwise velocity [m/s]
    "V": 0.0,  # steady synthetic transverse velocity [m/s]
    "hydraulicClosureModel": "standard_z0",
    "z0": 0.0005,  # roughness height for standard_z0 closure [m]

    # Fluid properties
    "rho_f": 1000.0,  # fluid density [kg/m3]
    "nu": 1e-06,  # fluid kinematic viscosity [m2/s]
    "g": 9.81,  # gravitational acceleration [m/s2]

    # Mesh properties
    "nx": 400,  # number of cells in x
    "ny": 20  # number of cells in y
}

PARTICLE_SETTINGS = {
    # Particle properties
    "shape": "sphere",  # particle shape: "sphere", "ellipsoid", "cylinder", "disk", or "prism"
    "L": 0.005,  # sphere diameter D [m]
    "rho_p": 990.0,  # particle density [kg/m3]

    # Particle release
    "release_mode": "instantaneous",  # "bulk" or "continuous" release mode
    "location_mode": "coordinates",  # location mode: "coordinates" or "click"
    "manual_centers": [[1.0, 0.5]],  # list of [x, y] release centers
    "n_per_center": 1000,  # particles released per center
    "nTrack": 10,  # number of saved trajectories
    "releaseSigma": 0.0,  # horizontal cloud spread [m]
    "zFrac": 0.5  # initial z as fraction of local depth (0=bed, 1=surface)
}

PARTICLE_EVO_SETTINGS = {
    # Particle evolution
    "biofouling": "off",  # "off" or "on"
    "degradation": "off"  # "off" or "on"
}

TRANSPORT_SETTINGS = {
    # Time settings
    "dt": 0.2,  # particle timestep [s]
    "tTrack": 5.0,  # particle tracking duration [s]

    # Advection and dispersion
    "hydraulicFieldMode": "cellwise",  # "cellwise" or "local_idw"
    "transportVelocityMode": "depth_averaged",  # "depth_averaged" or "loglaw_vertical"
    "transportModel": "random_walk",  # "random_walk" or "langevin"
    "rng_seed": 1,  # random seed: None for a fresh seed each run, or an integer for reproducible release/random-walk paths
    "useRWx": True,  # streamwise random walk on/off
    "useRWy": False,  # transverse random walk on/off
    "useRWz": True,  # vertical random walk on/off
    "betaKh": 0.6,  # horizontal diffusivity factor
    "KhMax": 5.0,  # cap on horizontal diffusivity [m2/s]
    "rAniso": 3.0,  # horizontal anisotropy ratio: K_parallel = rAniso K_h, K_perp = K_h/rAniso
    "alphaKz": 1.0,  # vertical diffusivity factor
    "KzMax": 0.3  # cap on vertical diffusivity [m2/s]
}

BOUNDARY_SETTINGS = {
    # Boundary interaction framework
    "surfacePolicy": "always_reflect",  # "always_reflect", "always_stick", "deterministic_detachment", or "probabilistic_detachment"
    "surfaceVerticalDragCoeff": 2.0,  # vertical drag coefficient used for surface detachment
    "bedPolicy": "always_reflect",  # "always_reflect", "always_deposit", "reflect_if_ustar_gt_crit", or "probabilistic_entrainment"
    "tanphi_ratio": 0.55,  # friction angle ratio for bed entrainment
    "dryPolicy": "tangential",  # "tangential", "stop", or "stick_active"
    "hmin": 0.01,  # minimum wet depth threshold [m]
    "outsidePolicy": "stop",  # currently "stop"
    "uphillPolicy": "off"  # "off" or "stop"
}

OUTPUT_SETTINGS = {
    # Export
    "export_data": True,  # export CSV/JSON output files
    "output_dt": 1.0,  # tracked trajectory output timestep [s]; must be a multiple of particle dt

    # Plotting
    "binary_map": True,  # plot release/end-state map
    "trajectories": True,  # plot tracked particle trajectories
    "binary_map_marker_size": 5.0,  # marker size for binary map
    "binary_map_mesh_overlay": True  # show original mesh over smoothed binary map
}

USER_NOTES = """
Small synthetic flume case used as a runnable publication smoke test.
"""

SETTINGS = {
    **HYDRAULIC_INPUT,
    **PARTICLE_SETTINGS,
    **PARTICLE_EVO_SETTINGS,
    **TRANSPORT_SETTINGS,
    **BOUNDARY_SETTINGS,
    **OUTPUT_SETTINGS,
}


def main() -> None:
    run_synthetic_case(__file__, settings=SETTINGS, show_plots=False)


if __name__ == "__main__":
    main()
