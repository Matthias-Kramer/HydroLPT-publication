# HydroLPT

HydroLPT is a Lagrangian particle-tracking tool for hydraulic model outputs, with a desktop GUI for setting up and running particle transport simulations.

This publication repository contains the HydroLPT source code, the GUI entry point, minimal GUI assets, documentation, citation metadata, and two GUI-loadable test cases. Large hydraulic model files beyond the bundled Merced GUI test file, simulation outputs, benchmarks, and full paper figure datasets are excluded from GitHub and should be archived separately in a research data repository.

## Features

- Particle release, property assignment, and transport simulation workflows.
- Synthetic, BASEMENT, and HEC-RAS-oriented hydraulic input adapters.
- Boundary interaction handling for bed, surface, dry-cell, and domain-exit behavior.
- Output diagnostics and plotting utilities.
- Desktop GUI entry point for local use.

## Installation

HydroLPT requires Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

For test dependencies:

```powershell
pip install -e .[test]
```

## Run The GUI

From the repository root:

```powershell
python launch_gui.py
```

Or, after installation:

```powershell
hydrolpt-gui
```

## GUI Test Cases

The GUI test cases are split by case:

```text
tests/flume/flume_case.py      Synthetic flume GUI case
tests/merced/merced_case.py    Merced River HEC-RAS GUI case
```

Open these files from the HydroLPT GUI with the case-file loader, or run them directly:

```powershell
python tests/flume/flume_case.py
python tests/merced/merced_case.py
```

The Merced GUI case uses the bundled HEC-RAS plan HDF at `tests/merced/data/HydroFlowOptimization.p01.hdf` and a wet-cell release point from that file: `2077955.433913742, 636415.8222505485`.

## Repository Layout

```text
core/        HydroLPT model, adapter, solver, boundary, and diagnostic code
gui/         Desktop GUI code
assets/      Minimal GUI icon assets
docs/        Publication and data availability notes
tests/       GUI-loadable flume and Merced cases
```

## Data And Reproducibility

The clean code repository does not include heavyweight hydraulic model data, generated `output/` folders, or large paper figure datasets. See [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md) for the intended data split.

For paper reproduction, archive the required model inputs and generated datasets separately, then link that archive from the GitHub release and manuscript.

## Citation

If you use HydroLPT, please cite the associated paper and software release. Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

HydroLPT is released under the MIT License. See [LICENSE](LICENSE).
