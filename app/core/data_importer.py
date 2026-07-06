from pathlib import Path
import shutil
import pandas as pd


class DataImporter:
    def import_wide_csv(self, csv_path: Path, project_dir: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        destination = project_dir / "data" / "original" / Path(csv_path).name
        shutil.copy2(csv_path, destination)
        canonical = project_dir / "data" / "imported_dataset.csv"
        df.to_csv(canonical, index=False)
        return df

    def import_split_csv(self, inputs_path: Path, outputs_path: Path, sample_id: str, project_dir: Path) -> pd.DataFrame:
        inputs = pd.read_csv(inputs_path)
        outputs = pd.read_csv(outputs_path)
        if sample_id not in inputs.columns or sample_id not in outputs.columns:
            raise ValueError(f"Both files must contain sample ID column '{sample_id}'.")
        df = inputs.merge(outputs, on=sample_id, how="inner", validate="one_to_one")
        shutil.copy2(inputs_path, project_dir / "data" / "original" / Path(inputs_path).name)
        shutil.copy2(outputs_path, project_dir / "data" / "original" / Path(outputs_path).name)
        df.to_csv(project_dir / "data" / "imported_dataset.csv", index=False)
        return df
