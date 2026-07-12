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

`setup_mac.sh` also installs `libomp` via Homebrew if it is missing. xgboost requires `libomp` to load on macOS; without it, training and prediction fail with a native library load error. If Homebrew is not installed, install it from https://brew.sh, run `brew install libomp`, then re-run `setup_mac.sh`.

Terminal setup is also supported:

```bash
cd ~/Documents/AntennaSurrogateStudio
./setup_mac.sh
./run_app.command
```

The macOS and Windows launchers use the same application code, same `requirements.txt`, and same project folder format. Only the installer/launcher scripts differ.

## Workflow

1. Create or open a project from the Library. Future workflow tabs remain locked until this is done.
2. Optionally use Generate & Run in CST to produce a dataset via Latin Hypercube Sampling and CST Studio Suite automation (Windows only, see below), or import your own CSV.
3. Import a wide CSV or split inputs/outputs CSV files.
4. Select input design variables and output response columns.
5. Prepare the dataset. Blocking validation errors must be fixed before continuing.
6. Review diagnostics and click Continue to Train.
7. Train an XGBoost multi-output surrogate model.
8. Validate model metrics, then make predictions or review model history.
9. Export predictions or the full project bundle.

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

The Assistant page includes a Basic Offline Guide (keyword search over bundled documentation) that is always available. If [Ollama](https://ollama.com) is installed with the `qwen3:1.7b` model pulled (`ollama pull qwen3:1.7b`), the Assistant automatically switches to that local LLM instead, grounded on the full user manual and knowledge base so it can answer free-form questions rather than only matching keywords. It is instructed to decline antenna-design/RF-engineering questions and only answer product-usage questions. On modest hardware (no GPU, 8GB RAM), responses can take 10-60 seconds; the Assistant shows a "thinking" placeholder while it works and never blocks the rest of the UI.

Assistant responses are generated from local product documentation and a local model. Your project data is not sent to cloud services by this application.

## CST Automation (Experimental)

The "Generate & Run in CST" page can generate Latin Hypercube Sampling (LHS) design combinations and drive CST Studio Suite directly via its COM automation interface to run each combination and pull back S-parameter/gain sweeps, producing a wide CSV ready for the normal Import & Configure Data step. This requires Windows with CST Studio Suite and the `pywin32` package installed; it is not available on macOS/Linux, where the page shows why and the manual CSV import path remains fully supported as a fallback.

This integration was written against CST's documented COM/VBA automation object model but has **not** been verified against a live CST installation. Before relying on it, open CST's own Macro Editor (File > Macros > Macro Editor), record a macro for opening a project, setting a parameter, starting the solver, and reading a 1D result, and confirm the method names in `app/core/cst_automation.py` match your CST version - adjust as needed.

## Version 1 Limitations

Version 1 implements XGBoost surrogate modeling only. It does not implement DNN training, inverse design, genetic algorithms, HFSS automation, cloud APIs, antenna optimization, or beamforming-specific inverse-search workflows. CST automation is experimental and Windows-only (see above).
