from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Callable

from core._00_case_management import (
    flatten_sections,
    generate_case_files,
    GUI_CASE_FILENAME,
    normalize_case_spec,
    safe_run_name,
)
from core._01_hydraulic_input_and_adapter.basement import run_basement_case
from core._01_hydraulic_input_and_adapter.hecras import run_hecras_case
from core._01_hydraulic_input_and_adapter.synthetic import run_synthetic_case


LogCallback = Callable[[str], None]
STANDARD_OUTPUT_FILENAMES = {
    "trajectories_tracked.csv",
    "particles_final.csv",
    "particle_status.csv",
    "run_info.txt",
    "run_info.json",
}


class _StreamToCallback(io.TextIOBase):
    """Minimal text stream wrapper that forwards writes to a callback."""

    def __init__(self, callback: LogCallback | None) -> None:
        self._callback = callback

    def write(self, text: str) -> int:
        if self._callback is not None and text:
            self._callback(text)
        return len(text)

    def flush(self) -> None:
        return


def default_output_dir() -> Path:
    """Return the default GUI output directory in the user's home area."""
    documents_dir = Path.home() / "Documents"
    if documents_dir.exists():
        return documents_dir / "HydroLPT" / "runs" / "gui_output"
    return Path.home() / "HydroLPT" / "runs" / "gui_output"


def _resolve_output_dir(spec: dict[str, Any]) -> Path:
    """Create and return the resolved output directory for a case spec."""
    output_dir = Path(spec["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _generate_case_files_if_requested(
    spec: dict[str, Any],
    export_case_files: bool,
) -> tuple[Path | None, Path | None, Path | None]:
    """Export runnable case files when requested by the caller."""
    if not export_case_files:
        return None, None, None
    return generate_case_files(spec)


def _snapshot_output_state(run_file: Path) -> tuple[Path, bool, set[str]]:
    """Capture which standard output files existed before a run started."""
    output_dir = run_file.parent / "output"
    existed_before = output_dir.exists()
    existing_files: set[str] = set()
    if existed_before:
        existing_files = {
            path.name for path in output_dir.iterdir()
            if path.is_file() and path.name in STANDARD_OUTPUT_FILENAMES
        }
    return output_dir, existed_before, existing_files


def _clear_previous_run_outputs(run_file: Path) -> Path:
    """Remove prior standard output files without deleting the folder itself."""
    output_dir = run_file.parent / "output"
    if not output_dir.exists():
        return output_dir

    for filename in STANDARD_OUTPUT_FILENAMES:
        file_path = output_dir / filename
        if not file_path.exists():
            continue
        try:
            file_path.unlink()
        except PermissionError:
            print(
                f"Warning: could not remove previous output file because it is in use: {file_path}"
            )
        except OSError as exc:
            print(f"Warning: could not remove previous output file {file_path}: {exc}")
    return output_dir


def _cleanup_failed_run_outputs(run_file: Path, snapshot: tuple[Path, bool, set[str]]) -> None:
    """Remove newly created standard output files after a failed run."""
    output_dir, existed_before, existing_files = snapshot
    for filename in STANDARD_OUTPUT_FILENAMES:
        file_path = output_dir / filename
        if filename not in existing_files and file_path.exists():
            file_path.unlink(missing_ok=True)

    if output_dir.exists() and not existed_before:
        try:
            next(output_dir.iterdir())
        except StopIteration:
            output_dir.rmdir()


def _resolve_execution_inputs(
    spec: dict[str, Any],
) -> tuple[str, dict[str, Any], str | None]:
    """Resolve case type, settings, and optional xdmf path from a spec."""
    merged_case = normalize_case_spec(spec)
    case_type = merged_case["case_type"]
    settings = flatten_sections(merged_case["sections"])
    settings.update(merged_case.get("plots", {}))
    settings["export_data"] = bool(merged_case.get("export_data", False))
    window_anchor = spec.get("window_anchor")
    if window_anchor is not None:
        settings["_window_anchor"] = window_anchor
    xdmf_path = merged_case.get("xdmf_path")
    return case_type, settings, xdmf_path


def _run_case(
    *,
    case_type: str,
    run_file: Path,
    settings: dict[str, Any],
    xdmf_path: str | None,
    show_plots: bool,
    block_on_plots: bool,
) -> list[list[float]]:
    """Dispatch the resolved case to the correct backend runner."""
    if case_type == "synthetic":
        centers = run_synthetic_case(
            run_file,
            settings=settings,
            show_plots=show_plots,
            block_on_plots=block_on_plots,
        )
        return [[float(x), float(y)] for x, y in centers.tolist()]

    if case_type == "basement":
        if not xdmf_path:
            raise ValueError("BASEMENT case requires xdmf_path in the spec.")
        centers = run_basement_case(
            run_file,
            settings=settings,
            xdmf_path=xdmf_path,
            show_plots=show_plots,
            block_on_plots=block_on_plots,
        )
        return [[float(x), float(y)] for x, y in centers.tolist()]

    if case_type == "hecras":
        if not xdmf_path:
            raise ValueError("HEC-RAS case requires a plan_hdf_path in the spec.")
        centers = run_hecras_case(
            run_file,
            settings=settings,
            plan_hdf_path=xdmf_path,
            show_plots=show_plots,
            block_on_plots=block_on_plots,
        )
        return [[float(x), float(y)] for x, y in centers.tolist()]

    raise ValueError(f"Unknown case_type: {case_type}")


def _persist_clicked_release_points(
    spec: dict[str, Any],
    selected_release_centers: list[list[float]],
) -> None:
    """Promote accepted click-picked release points into saved coordinates."""
    if not spec.get("preserve_clicked_release_points"):
        return

    particle_settings = spec.get("sections", {}).get("PARTICLE_SETTINGS", {})
    if particle_settings.get("location_mode") != "click":
        return
    if not selected_release_centers:
        return

    release_settings = spec.setdefault("sections", {}).setdefault("PARTICLE_SETTINGS", {})
    release_settings["manual_centers"] = [
        [float(center[0]), float(center[1])]
        for center in selected_release_centers
    ]
    particle_settings["location_mode"] = "coordinates"


def execute_case_spec(
    spec: dict[str, Any],
    *,
    log_callback: LogCallback | None = None,
    show_plots: bool = True,
    block_on_plots: bool = True,
    export_case_files: bool = False,
) -> tuple[Path | None, Path | None, Path | None, dict[str, Any]]:
    """Execute a GUI case specification and optionally export case files."""
    safe_run_name(spec.get("run_name", "gui_case"))
    output_dir = _resolve_output_dir(spec)
    run_file = output_dir / GUI_CASE_FILENAME

    case_type, settings, xdmf_path = _resolve_execution_inputs(spec)
    _clear_previous_run_outputs(run_file)
    output_snapshot = _snapshot_output_state(run_file)

    try:
        redirect_context = contextlib.nullcontext()
        if log_callback is not None:
            stream = _StreamToCallback(log_callback)
            redirect_context = contextlib.ExitStack()
            redirect_context.enter_context(contextlib.redirect_stdout(stream))
            redirect_context.enter_context(contextlib.redirect_stderr(stream))

        with redirect_context:
            selected_release_centers = _run_case(
                case_type=case_type,
                run_file=run_file,
                settings=settings,
                xdmf_path=xdmf_path,
                show_plots=show_plots,
                block_on_plots=block_on_plots,
            )
    except Exception:
        _cleanup_failed_run_outputs(run_file, output_snapshot)
        raise

    _persist_clicked_release_points(spec, selected_release_centers)

    script_path, config_json_path, config_txt_path = _generate_case_files_if_requested(
        spec,
        export_case_files,
    )
    return script_path, config_json_path, config_txt_path, {
        "selected_release_centers": selected_release_centers,
    }

