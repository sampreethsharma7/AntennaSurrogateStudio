import platform
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from app.core.lhs_sampling import generate_lhs_samples

IS_WINDOWS = platform.system() == "Windows"


class CSTNotAvailableError(RuntimeError):
    pass


def cst_automation_available() -> bool:
    if not IS_WINDOWS:
        return False
    try:
        import win32com.client
    except ImportError:
        return False
    try:
        win32com.client.Dispatch("CSTStudio.Application")
        return True
    except Exception:
        return False


def extract_named_outputs(x_values, y_values, output_columns: list[str], axis_metadata: dict) -> dict[str, float]:
    """Interpolate one raw CST 1D sweep (x_values, y_values, e.g. frequency vs S11 in dB) onto the
    specific named output columns (e.g. "S11_2.40GHz"), using the numeric axis value already parsed
    out of each column name by app.core.schema_manager.infer_output_axis. This avoids guessing a
    single scalar point inside CST's result tree and instead reuses the exact axis values the rest
    of the app already derives from the project's output column names."""
    axis_values = axis_metadata.get("axis_values", [])
    if not x_values or len(axis_values) != len(output_columns):
        return {name: None for name in output_columns}
    order = np.argsort(x_values)
    xs = np.asarray(x_values, dtype=float)[order]
    ys = np.asarray(y_values, dtype=float)[order]
    return {name: float(np.interp(axis_value, xs, ys)) for name, axis_value in zip(output_columns, axis_values)}


class CSTSession:
    """
    Thin wrapper around CST Studio Suite's COM automation interface (CSTStudio.Application ->
    Active3D() -> Project), the same object model exposed to CST's own VBA macros.

    This has NOT been verified against a live CST installation - it was written from CST's
    documented automation object model, not tested end-to-end. Before relying on it: open CST's
    Macro Editor (File > Macros > Macro Editor) on your installed version, record a macro for
    "open project", "set a parameter", "start the solver", and "read a 1D result", and confirm the
    method names below match what the recorder generates for your CST version. Adjust as needed.
    """

    def __init__(self, project_path: Path):
        if not IS_WINDOWS:
            raise CSTNotAvailableError("CST automation requires Windows with CST Studio Suite installed.")
        try:
            import win32com.client
        except ImportError as exc:
            raise CSTNotAvailableError("The 'pywin32' package is required for CST automation on Windows.") from exc
        self._win32com = win32com.client
        self.project_path = Path(project_path)
        self.application = None
        self.project = None

    def open(self) -> None:
        self.application = self._win32com.Dispatch("CSTStudio.Application")
        self.application.OpenFile(str(self.project_path))
        self.project = self.application.Active3D()

    def set_parameters(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.project.StoreParameter(name, value)
        self.project.Rebuild()

    def run_solver(self, solver: str = "Time") -> None:
        if solver == "Time":
            self.project.TimeSolver.Start()
        elif solver == "Frequency":
            self.project.FDSolver.Start()
        else:
            raise ValueError(f"Unsupported solver type: {solver}. Use 'Time' or 'Frequency'.")

    def read_1d_result(self, tree_path: str) -> tuple[list[float], list[float]]:
        result = self.project.Results1D(tree_path)
        n = result.GetN()
        return [result.GetX(i) for i in range(n)], [result.GetY(i) for i in range(n)]

    def close(self, save: bool = False) -> None:
        if self.project is not None:
            if save:
                self.project.Save()
            self.project.Quit()


def run_lhs_batch(
    project_template: Path,
    bounds: dict[str, tuple[float, float]],
    n_samples: int,
    output_columns: list[str],
    axis_metadata: dict,
    result_tree_path: str,
    seed: Optional[int] = None,
    solver: str = "Time",
    progress: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    progress = progress or (lambda message: None)
    if not cst_automation_available():
        raise CSTNotAvailableError("CST automation requires Windows with CST Studio Suite installed and pywin32.")
    progress(f"Generating {n_samples} LHS samples")
    samples = generate_lhs_samples(bounds, n_samples, seed)
    rows = []
    session = CSTSession(project_template)
    session.open()
    try:
        for i, sample in enumerate(samples, start=1):
            progress(f"Running CST sample {i}/{n_samples}")
            session.set_parameters(sample)
            session.run_solver(solver)
            x_values, y_values = session.read_1d_result(result_tree_path)
            row = dict(sample)
            row.update(extract_named_outputs(x_values, y_values, output_columns, axis_metadata))
            rows.append(row)
    finally:
        session.close(save=False)
    progress("CST batch complete")
    return pd.DataFrame(rows)
