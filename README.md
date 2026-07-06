# Antenna Surrogate Studio

Antenna Surrogate Studio is a local desktop GUI for antenna/RF engineers, students, and researchers who want to build, validate, save, and reuse XGBoost surrogate models from parametric simulation or measurement data.

The app is local-first. Core project creation, CSV import, training, validation, prediction, and export do not require cloud APIs, API keys, or an internet connection after dependencies are installed.

## Installation On Windows

1. Unzip or copy this folder.
2. Double-click `setup_windows.bat`.
3. After setup completes, double-click `run_app.bat`.

The setup script creates `.venv` and installs `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `joblib`, and `Pillow`.


## Installation On macOS

1. Install Python 3.10 or newer. The python.org macOS installer is the simplest option because it usually includes Tkinter.
2. Open this folder in Finder.
3. Double-click `setup_mac.sh` or run it from Terminal. If macOS asks which app to use, choose Terminal.
4. After setup completes, double-click `run_app.command`.

Terminal setup is also supported:

```bash
cd ~/Documents/AntennaSurrogateStudio
./setup_mac.sh
./run_app.command
```

The macOS and Windows launchers use the same application code, same `requirements.txt`, and same project folder format. Only the installer/launcher scripts differ.

## Workflow

1. Create or open a project from the Library. Future workflow tabs remain locked until this is done.
2. Import a wide CSV or split inputs/outputs CSV files.
3. Select input design variables and output response columns.
4. Prepare the dataset. Blocking validation errors must be fixed before continuing.
5. Review diagnostics and click Continue to Train.
6. Train an XGBoost multi-output surrogate model.
7. Validate model metrics, then make predictions or review model history.
8. Export predictions or the full project bundle.

## Supported CSV Formats

Wide CSV stores inputs and outputs in one table:

```text
Length,Width,Feed_x,S11_2.40GHz,S11_2.41GHz,Gain_2.45GHz
28.5,36.0,3.0,-12.4,-14.1,5.8
```

Split CSV uses separate inputs and outputs files joined by a shared sample ID column.

## Examples

On startup, the app creates two synthetic example projects when dependencies are available:

- Rectangular Patch Antenna Example
- Dipole Radiation Pattern Example

These examples demonstrate dataset configuration, diagnostics, model training, prediction, and model history without using private research data.

## Project Portability

Every project is a self-contained folder under `projects/<project_name>/` with `project.json`, data, models, analysis, predictions, assistant history, logs, and backups. Model training creates versioned artifacts such as `model_v1.joblib` and `model_v1_metadata.json`; older versions are preserved.

## Assistant

The Assistant page includes a Basic Offline Guide that searches bundled product documentation and answers product-use questions. Local LLM provider hooks are present for future versions, but v1 does not require or configure a model.

Assistant responses are generated from local product documentation and optional local models. Your project data is not sent to cloud services by this application.

## Version 1 Limitations

Version 1 implements XGBoost surrogate modeling only. It does not implement DNN training, inverse design, genetic algorithms, CST/HFSS automation, cloud APIs, antenna optimization, or beamforming-specific inverse-search workflows.
