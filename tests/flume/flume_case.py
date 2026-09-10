from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core._01_hydraulic_input_and_adapter.synthetic import run_synthetic_case

HYDRAULIC_INPUT = {
    "run_mode": "steady",
    "length": 20.0,
    "domain_width": 1.0,
    "wetted_width": 0.6,
    "water_depth": 0.5,
    "rho_f": 1000.0,
    "nu": 1e-6,
    "g": 9.81,
    "hydraulicClosureModel": "standard_z0",
    "hydraulicPrimaryVariable": "U",
    "U": 0.3,
    "V": 0.0,
    "z0": 5e-4,
}

PARTICLE_SETTINGS = {
    "location_mode": "coordinates",
    "shape": "sphere",
    "rho_p": 1010.0,
    "L": 5e-3,
    "release_x": 1.0,
    "release_y": 0.5,
    "manual_centers": [[1.0, 0.5]],
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
    "dt": 0.2,
    "tTrack": 0.4,
    "tRelease": 0.0,
    "hydraulicFieldMode": "cellwise",
    "transportVelocityMode": "depth_averaged",
    "transportModel": "random_walk",
    "rng_seed": 1,
    "useRWx": False,
    "useRWy": False,
    "useRWz": False,
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
    "output_dt": 1.0,
    "binary_map": False,
    "trajectories": False,
}

SETTINGS = {
    **HYDRAULIC_INPUT,
    **PARTICLE_SETTINGS,
    **PARTICLE_EVO_SETTINGS,
    **TRANSPORT_SETTINGS,
    **BOUNDARY_SETTINGS,
    **OUTPUT_SETTINGS,
}

USER_NOTES = """
Small synthetic flume case used as a runnable publication smoke test.
"""


def main() -> None:
    run_synthetic_case(__file__, settings=SETTINGS, show_plots=False)


if __name__ == "__main__":
    main()
