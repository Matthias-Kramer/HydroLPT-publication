from __future__ import annotations

import os
import unittest
from copy import deepcopy
from pathlib import Path

from core._00_case_management import default_case_blueprints
from core.app import execute_case_spec


def test_merced_river_hecras_smoke(tmp_path):
    default_plan_hdf = Path(__file__).parent / "data" / "HydroFlowOptimization.p01.hdf"
    plan_hdf = os.environ.get("HYDROLPT_MERCED_PLAN_HDF", str(default_plan_hdf))

    plan_hdf_path = Path(plan_hdf)
    if not plan_hdf_path.exists():
        raise unittest.SkipTest(f"Merced River HEC-RAS plan HDF not found: {plan_hdf_path}")

    release_x = float(os.environ.get("HYDROLPT_MERCED_RELEASE_X", "2077955.433913742"))
    release_y = float(os.environ.get("HYDROLPT_MERCED_RELEASE_Y", "636415.8222505485"))

    spec = deepcopy(default_case_blueprints()["hecras"])
    spec["run_name"] = "test_merced_river"
    spec["output_dir"] = str(tmp_path)
    spec["xdmf_path"] = str(plan_hdf_path)
    spec["export_data"] = True
    spec["sections"]["HYDRAULIC_INPUT"].update(
        {
            "run_mode": "steady",
        }
    )
    spec["plots"].update(
        {
            "binary_map": False,
            "trajectories": False,
            "output_dt": 300.0,
        }
    )
    spec["sections"]["PARTICLE_SETTINGS"].update(
        {
            "location_mode": "coordinates",
            "release_x": release_x,
            "release_y": release_y,
            "manual_centers": [[release_x, release_y]],
            "release_mode": "instantaneous",
            "n_per_center": 1,
            "nTrack": 1,
            "releaseSigma": 0.0,
        }
    )
    spec["sections"]["TRANSPORT_SETTINGS"].update(
        {
            "dt": 1.0,
            "tTrack": 1.0,
            "tRelease": 0.0,
            "rng_seed": 1,
        }
    )

    _script_path, _config_json, _config_txt, result = execute_case_spec(
        spec,
        show_plots=False,
        block_on_plots=False,
        export_case_files=False,
    )

    assert len(result["selected_release_centers"]) == 1
    assert (tmp_path / "output" / "run_info.json").exists()
