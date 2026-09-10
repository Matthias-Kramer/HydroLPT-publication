from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core._01_hydraulic_input_and_adapter.hecras import run_hecras_case

HYDRAULIC_INPUT = {
    "run_mode": "steady",
    "rho_f": 1000.0,
    "nu": 1e-6,
    "g": 9.81,
    "hydraulicClosureModel": "external_adapter",
}

PARTICLE_SETTINGS = {
    "location_mode": "coordinates",
    "shape": "sphere",
    "rho_p": 1010.0,
    "L": 5e-3,
    "release_x": 2077955.433913742,
    "release_y": 636415.8222505485,
    "manual_centers": [[2077955.433913742, 636415.8222505485]],
    "release_mode": "instantaneous",
    "n_per_center": 1,
    "release_dt": 1.0,
    "nTrack": 1,
    "releaseSigma": 0.0,
    "zFrac": 0.5,
}

PARTICLE_EVO_SETTINGS = {
    "biofouling": "off",
    "degradation": "off",
}

TRANSPORT_SETTINGS = {
    "dt": 1.0,
    "tTrack": 1.0,
    "tRelease": 0.0,
    "hydraulicFieldMode": "cellwise",
    "transportVelocityMode": "depth_averaged",
    "transportModel": "random_walk",
    "rng_seed": 1,
    "useRWx": True,
    "useRWy": True,
    "useRWz": True,
    "betaKh": 0.6,
    "KhMax": 5.0,
    "rAniso": 3.0,
    "alphaKz": 1.0,
    "KzMax": 0.3,
}

BOUNDARY_SETTINGS = {
    "surfacePolicy": "always_reflect",
    "bedPolicy": "always_reflect",
    "surfaceVerticalDragCoeff": 2.0,
    "surfaceContactAngleDeg": 105.0,
    "surfaceDetachmentSigmaStar": 0.2,
    "tanphi_ratio": 0.55,
    "bedEntrainmentSigmaStar": 0.2,
    "outsidePolicy": "stop",
    "dryPolicy": "reflect",
    "hmin": 1e-2,
    "uphillPolicy": "off",
    "dzUpMax": 0.4,
}

OUTPUT_SETTINGS = {
    "export_data": True,
    "output_dt": 300.0,
    "binary_map": False,
    "trajectories": False,
}

PLAN_HDF_PATH = str(Path(__file__).parent / "data" / "HydroFlowOptimization.p01.hdf")

SETTINGS = {
    **HYDRAULIC_INPUT,
    **PARTICLE_SETTINGS,
    **PARTICLE_EVO_SETTINGS,
    **TRANSPORT_SETTINGS,
    **BOUNDARY_SETTINGS,
    **OUTPUT_SETTINGS,
}

USER_NOTES = """
Small Merced River HEC-RAS case used as a runnable publication smoke test.
"""


def main() -> None:
    run_hecras_case(__file__, settings=SETTINGS, plan_hdf_path=PLAN_HDF_PATH, show_plots=False)


if __name__ == "__main__":
    main()
