# Developer Handoff

## Architecture

`app/ui/main_window.py` contains the Tkinter shell and page wiring. Business logic lives in `app/core/`: project manifests, CSV import, validation, diagnostics, training, evaluation, prediction, model registry, exports, compatibility, and logging.

Projects are self-contained folders under `projects/` with a `project.json` manifest and versioned model artifacts.

## Adding DNN Later

Add a new trainer module beside `app/core/model_trainer.py` and register the model type through the manifest/model registry. Keep the saved metadata contract: input order, output order, package versions, schema version, training settings, metrics, and data fingerprint.

## Adding Local LLM Providers Later

Implement `AssistantBackend` in `app/assistant/base_backend.py`. Keep app context restricted to product-support state. Do not pass raw datasets, arbitrary local files, or models to the assistant automatically.

## Compatibility

`app/core/compatibility.py` checks project schema versions. Migrations should create backups in `backups/` before changing files and should never delete data or model artifacts.

## Current Test Status

Automated tests cover manifest roundtrip, validation findings, schema axis inference, compatibility checks, and model version helper behavior. Full XGBoost training tests require dependencies installed in the target `.venv`.
