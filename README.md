# HydroLPT v0.9

HydroLPT is a Lagrangian particle-tracking tool for hydraulic model outputs, with a desktop GUI for setting up, running, saving, and visualizing particle transport simulations.

This publication repository contains the HydroLPT source code, the GUI entry point, minimal GUI assets, documentation, citation metadata, and two GUI-loadable test cases. Large hydraulic model files beyond the bundled Merced GUI test file, simulation outputs, benchmarks, and full paper figure datasets are excluded from GitHub and should be archived separately in a research data repository.

## Features

- Particle release, property assignment, and transport simulation workflows.
- Synthetic, BASEMENT, and HEC-RAS hydraulic input modes.
- Coordinate-based and interactive click-based particle release setup.
- Bulk and continuous release schedules.
- Random-walk and Langevin transport options.
- Depth-averaged, release-elevation, and log-law velocity sampling modes.
- Boundary interaction handling for bed, free surface, dry-cell, uphill-bed, and domain-exit behavior.
- Optional particle evolution through biofouling or degradation.
- Output diagnostics, final-state mapping, selected particle trajectories, and runnable case-file export.

## Installation

For Windows users, a packaged installer is available in
[`installer/HydroLPT-Setup.exe`](installer/HydroLPT-Setup.exe). The installer
sets up the HydroLPT desktop GUI in the user-local application folder, does not
require administrator rights, and can create a desktop shortcut.

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

## GUI Overview

The HydroLPT GUI is the recommended way to configure local simulations without editing Python case files by hand. It provides a structured editor for hydraulic input, particle properties, transport settings, boundary interaction choices, particle evolution, and output/plot settings.

At the top of the GUI, choose a simulation type and workflow:

- **Synthetic Flume** builds an idealized steady or transient flume case directly inside HydroLPT.
- **BASEMENT HPC** runs particle tracking on exported BASEMENT hydraulic results from a `results.xdmf` file.
- **HEC-RAS 2D** runs particle tracking on a HEC-RAS plan HDF file.
- **New Simulation** starts from default settings for the selected simulation type.
- **Existing Simulation** loads an existing `HydroLPT.py` case file and populates the editor with its settings.

The main editor is organized into tabs:

- **Hydraulic Input** sets the hydraulic model source, fluid properties, synthetic flume geometry, and transient timing where applicable.
- **Particle Settings** defines particle shape, size, density, release mode, number of particles, tracked trajectories, release spread, and vertical release position.
- **Particle Evolution** enables or disables biofouling and degradation parameters.
- **Transport Settings** controls timestep, tracking duration, hydraulic interpolation mode, velocity sampling, random walk, Langevin turbulence, diffusivity limits, and random seed.
- **Boundary Settings** controls surface, bed, dry-cell, outside-domain, and uphill-bed behavior. Dry-cell motion uses the `tangential` policy name, with alternatives `stop` and `stick_active`.
- **Plots** controls data export, trajectory output timestep, binary/final-state maps, trajectory plots, marker size, and mesh overlay.

## GUI Functionality

The GUI supports both direct execution and case-file generation:

- **Run Case** validates the current settings, writes a runnable `HydroLPT.py` case file, and starts the simulation in a subprocess so progress appears in the GUI log.
- **Save** writes the current configuration as a Python case file without starting a simulation.
- **Export case file** is always enabled. Every GUI run writes a runnable case file for reproducibility.
- **Export data** controls whether particle output tables and diagnostic files are written.
- **Save click coordinates** preserves interactively selected release locations in exported cases.
- **Stop Run** terminates the active subprocess when a simulation needs to be interrupted.

For click-based release, the GUI opens a hydraulic-field map and records valid wet, in-domain release points. For HEC-RAS native meshes, the picker draws the model cell polygons directly so release locations line up with the hydraulic cells.

The GUI also applies boundary-policy safeguards based on particle buoyancy. Rising particles use a locked bed reflection policy, settling particles use a locked surface reflection policy, and neutral particles reflect at both vertical boundaries.

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

The Merced GUI case uses the bundled HEC-RAS plan HDF at `tests/merced/data/HydroFlowOptimization.p01.hdf`.

## Repository Layout

```text
core/        HydroLPT model, adapter, solver, boundary, and diagnostic code
gui/         Desktop GUI code
assets/      Minimal GUI icon assets
docs/        Publication and data availability notes
tests/       GUI-loadable flume and Merced cases
```


## License

HydroLPT is released under the MIT License. See [LICENSE](LICENSE).
