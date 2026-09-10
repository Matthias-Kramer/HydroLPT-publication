from __future__ import annotations

from copy import deepcopy

from core._00_case_management import default_case_blueprints
from core.app import execute_case_spec


def test_synthetic_flume_smoke(tmp_path):
    spec = deepcopy(default_case_blueprints()["synthetic"])
    spec["run_name"] = "test_flume"
    spec["output_dir"] = str(tmp_path)
    spec["export_data"] = True
    spec["plots"].update(
        {
            "binary_map": False,
            "trajectories": False,
            "output_dt": 1.0,
        }
    )
    spec["sections"]["PARTICLE_SETTINGS"].update(
        {
            "location_mode": "coordinates",
            "release_x": 1.0,
            "release_y": 0.5,
            "manual_centers": [[1.0, 0.5]],
            "release_mode": "instantaneous",
            "n_per_center": 1,
            "nTrack": 1,
            "releaseSigma": 0.0,
        }
    )
    spec["sections"]["TRANSPORT_SETTINGS"].update(
        {
            "dt": 0.2,
            "tTrack": 0.4,
            "tRelease": 0.0,
            "rng_seed": 1,
            "useRWx": False,
            "useRWy": False,
            "useRWz": False,
        }
    )

    _script_path, _config_json, _config_txt, result = execute_case_spec(
        spec,
        show_plots=False,
        block_on_plots=False,
        export_case_files=False,
    )

    assert result["selected_release_centers"] == [[1.0, 0.5]]
    assert (tmp_path / "output" / "run_info.json").exists()
    assert (tmp_path / "output" / "particles_final.csv").exists()
