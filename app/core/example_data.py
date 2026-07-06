from pathlib import Path
import numpy as np
import pandas as pd

from app.core.project_manager import ProjectManager
from app.core.schema_manager import infer_output_axis, save_schema


def ensure_example_projects(projects_dir: Path) -> None:
    manager = ProjectManager(projects_dir)
    if not (projects_dir / "Rectangular_Patch_Antenna_Example" / "project.json").exists():
        project = manager.create_project("Rectangular Patch Antenna Example", "S-parameter prediction", "Synthetic patch antenna surrogate example.", "Patch", "mm/GHz/dB")
        rng = np.random.default_rng(42)
        n = 180
        length = rng.uniform(24, 34, n)
        width = rng.uniform(30, 44, n)
        feed = rng.uniform(-4, 5, n)
        height = rng.uniform(0.8, 2.4, n)
        freqs = np.round(np.linspace(2.30, 2.60, 21), 2)
        data = {"Patch_Length_mm": length, "Patch_Width_mm": width, "Feed_x_mm": feed, "Substrate_Thickness_mm": height}
        resonance = 2.45 + (30 - length) * 0.012 + (38 - width) * 0.006 + feed * 0.002
        for f in freqs:
            data[f"S11_{f:.2f}GHz"] = -5 - 22 * np.exp(-((f - resonance) ** 2) / 0.0025) + rng.normal(0, 0.35, n)
        data["Gain_2.45GHz"] = 6.0 - 0.025 * (length - 29) ** 2 - 0.015 * (width - 37) ** 2 - 0.08 * abs(feed) + rng.normal(0, 0.12, n)
        _finish_example(project, pd.DataFrame(data))
    if not (projects_dir / "Dipole_Radiation_Pattern_Example" / "project.json").exists():
        project = manager.create_project("Dipole Radiation Pattern Example", "Radiation-pattern prediction", "Synthetic dipole pattern surrogate example.", "Dipole", "mm/GHz/dBi")
        rng = np.random.default_rng(7)
        n = 160
        arm = rng.uniform(38, 62, n)
        radius = rng.uniform(0.5, 2.5, n)
        freq = rng.uniform(1.8, 2.4, n)
        theta = np.arange(-90, 91, 10)
        data = {"Arm_Length_mm": arm, "Radius_mm": radius, "Frequency_GHz": freq}
        scale = 1 + 0.006 * (arm - 50) - 0.04 * (freq - 2.1)
        for t in theta:
            base = 2.15 + 8 * np.sin(np.radians(t + 90)) ** 1.7
            data[f"Gain_theta_{t}deg"] = base * scale + 0.12 * radius + rng.normal(0, 0.18, n)
        _finish_example(project, pd.DataFrame(data))


def _finish_example(project: Path, df: pd.DataFrame) -> None:
    df.to_csv(project / "data" / "imported_dataset.csv", index=False)
    df.to_csv(project / "data" / "prepared_training_data.csv", index=False)
    manifest = ProjectManager(project.parent).load_manifest(project)
    if "Patch_Length_mm" in df.columns:
        inputs = ["Patch_Length_mm", "Patch_Width_mm", "Feed_x_mm", "Substrate_Thickness_mm"]
    else:
        inputs = ["Arm_Length_mm", "Radius_mm", "Frequency_GHz"]
    outputs = [c for c in df.columns if c not in inputs]
    manifest.selected_input_columns = inputs
    manifest.selected_output_columns = outputs
    manifest.input_bounds = {col: {"min": float(df[col].min()), "max": float(df[col].max())} for col in inputs}
    manifest.output_axis_metadata = infer_output_axis(outputs)
    manifest.data_paths = {"prepared_training_data": "data/prepared_training_data.csv", "imported_dataset": "data/imported_dataset.csv"}
    manifest.save(project)
    save_schema(project, {"input_columns": inputs, "output_columns": outputs, "output_axis_metadata": manifest.output_axis_metadata})
