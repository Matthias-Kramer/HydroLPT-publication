"""Path helpers for HydroLPT run directories.

This module resolves the standard hydraulic input files expected by
HydroLPT for a single run folder. These files are produced by an
external hydraulic model and are later consumed by the BASEMENT adapter:

- `results.xdmf`: metadata and dataset topology reference
- `results.h5`: primary hydraulic fields such as water surface elevation
- `results_aux.h5`: auxiliary hydraulic fields such as velocity

The functions here are intentionally pure and filesystem-light so they
are easy to test in isolation.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_INPUT_FILENAMES: dict[str, str] = {
    "xdmf": "results.xdmf",
    "h5": "results.h5",
    "h5aux": "results_aux.h5",
}


def _resolve_input_directory(caller_file: str | Path) -> Path:
    """Return the canonical `input` directory for a run script."""
    return Path(caller_file).resolve().parent / "input"


def get_default_paths(caller_file: str | Path) -> dict[str, Path]:
    """Return the default hydraulic input files for a run script.

    Parameters
    ----------
    caller_file
        Path to the Python run script that defines a HydroLPT case.

    Returns
    -------
    dict[str, Path]
        Mapping of standard HydroLPT hydraulic input keys to absolute
        paths inside the run-local `input` directory.
    """
    input_directory = _resolve_input_directory(caller_file)
    return {
        key: input_directory / filename
        for key, filename in DEFAULT_INPUT_FILENAMES.items()
    }

