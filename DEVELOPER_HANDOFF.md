# Developer Handoff

## Architecture

`app/ui/main_window.py` contains the CustomTkinter shell (dark sidebar navigation, numbered workflow steps) and page wiring. Business logic lives in `app/core/`: project manifests, CSV import, validation, diagnostics, training, evaluation, prediction, model registry, exports, compatibility, logging, LHS sampling, and CST automation.

Projects are self-contained folders under `projects/` with a `project.json` manifest and versioned model artifacts.

Sidebar step indices are hardcoded in `PAGES`/`STEP_NUMBERS`/`STEP_DONE_KEY` in `app/ui/main_window.py` and referenced by position throughout `_allowed_tab_indices`/`_step_lock_reason`. If you add or reorder a page, update all of these together.

## Adding DNN Later

Add a new trainer module beside `app/core/model_trainer.py` and register the model type through the manifest/model registry. Keep the saved metadata contract: input order, output order, package versions, schema version, training settings, metrics, and data fingerprint.

## Local LLM Provider

`app/assistant/local_llm_backend.py` implements `AssistantBackend` against a local Ollama server (`qwen3:1.7b` by default), grounded on the full user manual + knowledge base via `DocumentationSearch.combined_text()`. It is only used when `is_available()` detects Ollama is running with that model pulled; otherwise `OfflineGuideBackend` (keyword search) is used. Calls run in the background thread pool (`main_window.py`'s `ask_assistant`/`_poll_queue`) since generation can take tens of seconds on CPU-only hardware - never call `LocalLLMBackend.answer()` on the main thread. To swap models, change `MODEL_NAME` in `local_llm_backend.py` and `ollama pull` the new tag.

## CST Automation

`app/core/lhs_sampling.py` (pure math, fully tested) generates Latin Hypercube samples. `app/core/cst_automation.py` wraps CST Studio Suite's COM automation interface via `pywin32` - Windows-only, guarded by `cst_automation_available()`. **This has not been verified against a live CST installation**; it was written from CST's documented VBA/COM object model. Before trusting it, record a macro in CST's own Macro Editor for open/set-parameter/solve/read-result and confirm method names match `CSTSession` in that file. `extract_named_outputs()` (pure function, tested) interpolates a raw CST sweep onto the app's existing wide-CSV column naming convention (e.g. `S11_2.40GHz`) via `infer_output_axis`, avoiding any CST-version-specific scalar-result assumptions.

## Compatibility

`app/core/compatibility.py` checks project schema versions. Migrations should create backups in `backups/` before changing files and should never delete data or model artifacts.

## Current Test Status

Automated tests cover manifest roundtrip, validation findings, schema axis inference, compatibility checks, and model version helper behavior. Full XGBoost training tests require dependencies installed in the target `.venv`.
