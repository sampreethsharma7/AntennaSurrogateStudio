# Antenna Surrogate Studio

A premium, local-first desktop workspace for preparing antenna data, building
surrogate-model “books,” and returning to those books for inference later.
The Studio is intended for varied antenna types, geometries, solvers, design
variables, and response axes; templates must not imply one fixed antenna
configuration.

The current tester build includes:

- Create or open portable projects.
- One-click access to the five most recently used projects.
- Project continuity that returns to the last page used in each project.
- SnowBuddy, a project-aware local companion that opens in a dedicated
  right-side workspace column from a top-bar action without losing its
  conversation or covering page controls.
- Standard and lightweight Qwen profiles for different hardware.
- Data preparation from a ready input/output CSV pair or a `#Parameters` sweep
  export, written as matched project-owned `inputs.csv` and `outputs.csv`
  tables.
- A Model Training page whose Auto and Custom selections are mapped into a
  validated backend configuration request.
- Deterministic Auto-tuned or direct Custom Linear Regression training from the
  active registered dataset, with local metrics, test predictions, and
  preserved sequential runs.
- Deterministic Auto-tuned or direct Custom XGBoost training with the same
  registered-data, multi-output, metrics, run-artifact, Results, Model Book,
  and inference flow.
- Reproducible standardized Neural Network regression with Auto Medium/High,
  Custom architecture/training controls, and the same end-to-end workflow.
- An **Ensemble AI Engine** that automatically trains all three individual
  families in Auto High, derives normalized inverse-validation-RMSE weights,
  and saves a reproducible weighted multi-output estimator.
- A visual Training Results page with saved-run recommendations, metric cards,
  prediction/residual/error plots, configuration evidence, validation-backed
  Linear Regression, XGBoost, Neural Network, and Ensemble AI Engine comparison, run provenance, and compact
  access to the saved test-data CSV. Calculated findings are supplied to
  SnowBuddy for conversational explanation instead of occupying another panel.
- A **Create Model Book** workflow that promotes a completed run into a named,
  versioned, integrity-checked project-local Model Book without changing the
  source training artifacts.
- A Model Library page for browsing saved books, opening their metadata, and
  persisting one valid book as the active Model Book.
- A single-sample Inference page that generates the active book's required
  numeric fields and provides a reusable multi-curve Scientific Plot Workbench
  with engineering navigation, markers, comprehensive Plot Settings, curve
  management, and preserved per-curve input snapshots.
- A generic **Inverse Design** page that assigns saved inputs as bounded design
  variables or fixed values, defines one minimize/maximize/target objective and
  optional output constraints, then uses deterministic Differential Evolution
  with the active Model Book as its fast evaluator.
- A workflow sidebar that switches between full labeled navigation and compact
  icon-only navigation with tooltips and accessibility names while preserving
  the current page and project.

## Quick start

The Studio is shared as a source-based tester build. It requires 64-bit Python
3.11, 3.12, or 3.13 and creates exactly one private environment at `.venv`.
You do not need to activate it manually.

### Windows

1. Clone the repository or download and extract the ZIP.
2. Double-click `Start Antenna Surrogate Studio.bat`.
3. Let the first-run setup install the requirements; the Studio then opens.

Later launches reuse the same project-local `.venv`. If the window does not
appear, run `run_studio.bat` to retain the diagnostic message.

### macOS and Linux

Clone/download the repository, open a terminal in its folder, and run:

```bash
bash start_studio.sh
```

Linux also needs Python Tk support, commonly installed as `python3-tk`.

See [INSTALL.md](INSTALL.md) for explicit commands, optional local-AI setup,
expected folders, system caveats, and common fixes.

Projects are saved by default in:

```text
Documents/Antenna Surrogate Studio Library/projects/
```

Override the library location with the `ANTENNA_STUDIO_LIBRARY` environment
variable.

The Studio always opens in Welcome mode instead of silently activating the most
recent project. Creating a project moves directly to Data Prep and switches
SnowBuddy to project-specific Focus mode. Opening a project from the folder
chooser or a recent-project card restores the last page used in that project.
**Return to Welcome** removes the active-project context without closing the
application.

## SnowBuddy local AI

SnowBuddy uses [Ollama](https://ollama.com/download) on Windows, macOS, and
Linux. Ollama runs on the local machine, so project context is not sent to a
paid cloud API.

1. Install and start Ollama.
2. Open **Local model** in the SnowBuddy panel.
3. Choose and download one model:

   - **Standard — `qwen3:8b`**: approximately 5.2 GB. Recommended for machines
     with at least 16 GB system RAM or a capable GPU.
   - **Lightweight — `qwen3:1.7b`**: approximately 1.4 GB. Intended for
     lower-memory, CPU-only, and older systems.

The Studio recommends a profile from detected system memory, but the user
always controls the selection. Model choice is stored in the local Studio
library, while project chat history remains inside each project's
`assistant/` folder. If Ollama or the selected model is unavailable, SnowBuddy
continues with its built-in project guide.

Local overrides:

- `SNOWBUDDY_MODEL`: use another installed Ollama model.
- `OLLAMA_BASE_URL`: change the loopback Ollama endpoint or port from
  `http://127.0.0.1:11434`. SnowBuddy rejects non-loopback addresses.
- `ANTENNA_STUDIO_LIBRARY`: change the project-library location.

## Privacy and development transcripts

Production launches are local-only. SnowBuddy accepts Ollama connections only
on `localhost`, `127.0.0.1`, or `::1`. It has no telemetry or transcript-upload
code. Welcome-session and project chat history stay on the device solely to
restore or audit local conversations.

Development evaluation transcripts are a separate, double-gated feature. They
are enabled only when both of these values are set:

```text
ANTENNA_STUDIO_BUILD_CHANNEL=development
SNOWBUDDY_DEVELOPMENT_LOG=1
```

Developers can use `Start Antenna Surrogate Studio DEV.bat` on Windows or
`bash start_studio_dev.sh` on macOS/Linux. These launchers reuse the same
application environment and write one local JSON object per exchange to:

```text
.development/snowbuddy_sessions.jsonl
```

The development log records the question, response, response source, selected
model, project identity, and visible UI-state summary. The folder is excluded
from version control and nothing is transmitted. Normal production launchers
set neither switch and therefore create no development transcript.

## SnowBuddy identity and GUI awareness

SnowBuddy loads two versioned runtime references from the `snowbuddy/` folder:

- `SNOWBUDDY_CHARACTER.md` defines its identity, voice, mission, grounding
  rules, product boundaries, and local-first behavior.
- `BLIND_GUI_READ.md` describes the current visual hierarchy, pages, controls,
  dialogs, labels, and interface states.

Every question also includes a live snapshot of the visible page and current
Data Prep selections. The live snapshot takes precedence over documented
defaults.

SnowBuddy answers the user's immediate question before recommending another
workflow step. Status and “what is loaded?” questions receive the current
sample count, selected variables, output, source mode, and prepared-data state.
Source-method questions compare only the two controls present in the current
GUI. Historical assistant replies remain visible in local chat for continuity,
but retired controls, stale paths, and known obsolete product claims are excluded
from the local-model prompt. Current GUI state and the versioned contracts always
override earlier assistant answers.
Responses that reintroduce retired controls, the removed no-training
placeholder, or unsupported training behavior are replaced by the grounded
built-in guide.

Navigation answers are also stage-gated. SnowBuddy distinguishes project
creation, raw parsing, saved variable selection, prepared-but-unregistered data,
registered training data, completed runs, saved Model Books, active Model Books,
Inference, and Inverse Design. A local-model answer that skips a required gate
or sends an already-active book back for activation is replaced by the built-in
current-position guide. When the visible UI supplies a validation or availability
error, the fallback repeats that concrete reason and gives the smallest valid
recovery step.

Pure next-step and blocker questions use this trusted project guide immediately,
without waiting for Ollama generation. Qwen remains available for explanations,
result interpretation, comparisons, and open-ended assistance.

SnowBuddy is immediately available in **Welcome mode**, before any project
exists, and remains visible in the right-side companion panel throughout Start,
Data Prep, and Model Training. Every Studio launch creates a fresh Welcome session under the
Studio library’s `assistant/welcome_sessions/` folder. Previous sessions remain
local but are not mixed into a new conversation. Creating or opening a project
automatically switches the panel to **Focus mode** and that project’s isolated
`assistant/chat_history.json`. **Return to Welcome** restores the current
launch’s Welcome session.

Completed Data Prep state is restored when a project reopens. If both
`data/prepared/inputs.csv` and `data/prepared/outputs.csv` still exist, the page
shows **Data ready**, offers **View prepared files**, and labels the generation
action **Regenerate both files**; preparation does not need to be repeated.
Projects made by older Studio versions identify their combined
`training_data.csv` as legacy and offer a one-time **Generate separate files**
upgrade.

After preparation, Data Prep provides one **Validate and register** action.
Validation checks the selected multi-input/multi-output contract, matched sample
counts, row structure, finite numeric values, and optional Sample IDs. Only a
passing dataset is registered. Registration creates an integrity-protected,
content-addressed local snapshot under `data/registered/`, displays its sample,
input, and output counts plus dataset ID, and restores that result when the
project reopens. Repeating the action for identical content reuses the same ID.
Training execution is separate and is never started from Data Prep.

The Source subtask also includes a project-local **LHS sample generator** for
designing simulation inputs before solver outputs exist. It uses SciPy Latin
Hypercube sampling with generic user-named variables, finite minimum/maximum
bounds, a sample count, and an optional reproducible seed. A compact dialog
supports up to 20 variable definitions with five visible at a time, previews
the first generated rows and two-dimensional sampling coverage, and exports a
Studio-compatible `inputs.csv`. Export loads only the Input CSV path and clears
the Output CSV path: the user must run the design rows in CST, HFSS, or another
solver and return with an output table in the same row order and with the same
row count. Generated LHS files do not add a Sample ID column. The Studio does
not drive the solver or generate response values.

Data Prep has two source choices:

- **Input + output files** automatically loads and validates two CSVs as soon
  as both paths are selected. Every input column becomes a feature and every
  output column becomes a target, then project-local copies are prepared
  without asking the user to parse or select variables. The files require
  matching sample-row counts. An optional first `Sample ID` column must appear
  in both files with unique, row-aligned IDs. Supplied IDs are preserved in the
  prepared and registered pair and later identify rows in prediction results;
  they are excluded from model features and targets.
- **#Parameters sweep** exposes **Parse** for a raw text export or folder
  containing `#Parameters = {...}` blocks. Only this raw route asks the user to
  choose parsed variables and an output response. After choosing them, the user
  explicitly selects **Save selection** before preparation is enabled. The
  saved contract is restored when the project reopens; changing a selection
  requires saving it again. Each block must include a quoted table header and
  numeric rows. The first table column is the ordered response coordinate; it
  may be Frequency, Theta, Phi, or another solver-defined coordinate. The
  selected response may be S11, gain, efficiency, or another numeric response.
  The parser preserves the coordinate name and recognized unit instead of
  hard-coding theta. For example, `"Frequency / GHz" "S11"` produces ordered
  output columns such as `S11 at Frequency 1 GHz`.

The exact raw-export flow is **#Parameters sweep → Browse file/Browse folder →
Parse → choose Model inputs and one Pattern output → Save selection → Prepare
input + output → Validate and register**. This route consumes completed solver
results. It is separate from the LHS sample generator, which creates only
unsimulated input settings. Input/output files must have matching row counts and
row order, but their numbers of columns do not need to match.

Data Prep is a fixed, non-scrolling accordion of four connected subtasks:
Source, Variables, Prepare, and Register. Every header has a pending, active,
complete, or error symbol, and only one body is open at a time. Fixed footer
buttons navigate back to Start or forward to Model Training.

## Model Training page

Model Training is a fixed, non-scrolling configuration page. Its model dropdown
contains **Linear Regression**, **XGBoost**, **Neural Network**, and **Ensemble AI Engine**. **Auto** is the default training mode;
it exposes **Auto Search Level** with **Medium** selected by default and **High**
as the more-thorough option. Auto mode hides the entire Advanced
Settings section and keeps its underlying controls disabled.
Both dropdown fields and their expanded option menus use the active Instrument
Lab Light or Dark palette rather than the toolkit's generic menu colors.

Selecting **Custom** immediately hides Auto Search Level, reveals Advanced
Settings, and enables the Linear Regression parameter toggles
`fit_intercept` (on by default) and `positive` (off by default). Switching back
to Auto hides and disables that section again.

Selecting **XGBoost** keeps the same Auto/Custom control. Auto shows the Medium
or High Auto Search Level. Custom reveals numeric controls for `n_estimators`,
`max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.

Selecting **Neural Network** also supports Auto Medium/High. Custom reveals
hidden-layer architecture, activation, learning rate, batch size, and epoch
controls. Inputs are standardized automatically before the MLP regressor.

**Train Model** maps the page values to a validated `ModelTrainingRequest`.
Auto submits `linear_regression`, `auto`, the selected `medium` or `high` search
level, and no custom parameters. Custom submits `linear_regression`, `custom`,
no search level, and both Boolean Linear Regression parameters. A valid request
is passed to `submit_model_training_request`, where the backend revalidates it
as the final authority.
XGBoost Auto submits `xgboost`, `auto`, the selected `medium` or `high` search
level, and no custom parameters. XGBoost Custom submits `xgboost`, `custom`, no
search level, and all five numeric values. The backend contract remains the
final authority for type, completeness, supported-name, and range validation.
Neural Network Auto submits `neural_network`, `auto`, and the selected search
level. Custom submits `neural_network`, `custom`, no search level, and all five
Neural Network values.

Selecting **Ensemble AI Engine** locks the request to Auto High and hides Custom
settings. One click trains Linear Regression, XGBoost, and Neural Network in
Auto High. Each successful component is preserved as its own immutable run;
the final Ensemble run is created last. At least two valid components are
required.

The trainer executes **Linear Regression** against the active integrity-checked
Stage 0 registered dataset. It first creates the fixed 80/20 split with
`random_state=42`. Auto then uses deterministic cross-validation on the training
partition only: Medium evaluates the two non-positive configurations with 3
folds, while High evaluates all four `fit_intercept` / `positive` combinations
with 5 folds. The fold count is reduced safely for smaller training partitions,
but never below 2. The lowest mean validation RMSE wins, with deterministic
ties preferring `positive=False` and then `fit_intercept=True`. The selected
model is fitted on the full training partition and evaluated once on the held-
out test set. Custom bypasses search and applies the two validated Boolean
values selected in Advanced Settings.

The XGBoost path reuses the same deterministic 80/20 split and held-out metrics.
Auto performs deterministic cross-validation only on the training partition.
Medium evaluates 3 bounded configurations with 3 folds; High evaluates 6 with
5 folds. The lowest mean validation RMSE wins, with candidate order providing a
stable tie break. The selected configuration is fitted on the complete training
partition and evaluated once on the untouched test set. Custom applies the five
user-facing values directly. Both paths preserve the fixed objective,
`random_state=42`, `n_jobs=1`, histogram tree method, and other deterministic
runtime defaults, and both support the current multi-output target matrix.

The Neural Network path saves a `StandardScaler` + `MLPRegressor` pipeline.
Medium evaluates 3 bounded configurations with 3 folds; High evaluates 6 with
5 folds. Search uses training-only validation RMSE and stable candidate-order
ties. Custom applies the selected architecture, activation, learning rate,
batch size, and epoch budget. Adam, `random_state=42`, disabled epoch shuffling,
and disabled early stopping keep repeated runs reproducible where practical.

The Ensemble AI Engine collects each valid component's training-only validation
RMSE and assigns normalized inverse-RMSE weights; held-out test metrics are not
used for weighting. It recreates the selected configurations on the shared
training-only folds to measure the ensemble's own validation RMSE, fits no new
component configuration from test evidence, and evaluates the saved weighted
estimator once on the common test partition. A failed component is recorded and
the remaining models continue when at least two remain. The Ensemble is
recommended only when its validation RMSE is strictly lower than the best
individual component.

A successful run displays MAE, RMSE, and R². Every successful click creates a
new sequential folder such as `models/runs/run-0003/` containing the
family-specific `linear_regression_model.joblib`, `xgboost_model.joblib`,
`neural_network_model.joblib`, or `ensemble_ai_engine_model.joblib`, plus
`metrics.json`, `test_predictions.csv`,
`training_config.json`, and `run.json`; Auto runs also contain
`auto_search_results.json`. Earlier run folders are preserved. The project
manifest records the complete run history, latest run, dataset, actual
parameters used, split sizes, metrics, and artifact paths. Auto completion shows
the search level, evaluated count, actual fold count, selected parameters,
validation RMSE, and final test metrics. Custom completion shows the selected
parameters and test metrics. The page displays
**Latest Run: Run 3** and restores that value when the project reopens. Invalid
configuration or dataset state produces a friendly message without a raw
traceback or partial fake result. Ensemble folders additionally contain
`ensemble_results.json` and integrity-preserved component artifacts under
`components/`.
Before a run, the action reads **Train Model** and is enabled. During the
backend call it is disabled and reads **Training…**. It is restored to the
enabled **Train Model** state after either success or failure.
In Custom mode, Advanced Settings sits directly beneath Training Mode; flexible
space is kept below the configuration cards so the action bar remains fixed.

## Training Results page

Every successful real run opens the fixed, non-scrolling **Training Results**
page and becomes its latest result. A compact 44-pixel configuration strip has
one purpose: Auto shows **AUTO BEST** and its selected parameters; Custom shows
**CUSTOM USED** and its applied parameters; XGBoost Auto shows **AUTO BEST**
while XGBoost Custom shows **CUSTOM USED**, both with core estimator settings.
It contains no validation summary,
suggestion, explanation, or action. Auto search evidence, Custom-versus-Auto
comparison, and no-evidence guidance live only in **Configuration**. Four
compact cards show R², RMSE, MAE, and Validation RMSE as
value-first readouts. Hovering or focusing the small **?** control reveals the
plain-language meaning and higher/lower guidance without permanently consuming
plot space. Units are included only when explicit target-unit metadata exists.

An ordered one-panel-at-a-time navigator keeps the page compact without page
scrolling. **Predictions** is the default section and always means the native
response-curve comparison for one selected test design. It now reuses the same
Scientific Plot Workbench as Inference. Actual is a solid teal curve with
circular markers; Predicted is a dashed violet curve with diamond markers, so
even overlapping results remain distinguishable. A test-sample dropdown changes
the displayed design, and a compact strip shows its registered input-parameter
values. The **Test Sample** dropdown sits in the workbench's Curves panel above
the fixed Actual and Predicted curve list, using space that would otherwise be
empty. There is no separate inline coordinate row. Plot title, X/Y labels and
limits, typography, grids, scales, legend, and selected-curve style are edited
only through **Plot Settings**. User-defined limits remain in place when the
test sample changes.
It first tries to infer Frequency, Theta, or Phi
and a uniform range from the output-column names, otherwise it uses 1-based
output-point indices. Applying a range requires finite numeric values, a
positive step, an evenly divisible interval, and exactly the same generated
point count as the selected sample's output columns. Invalid input leaves the
current plot unchanged and shows a clear error. A
compact **Open Test Data CSV** action remains available, but no long sample
table is rendered. **Residuals** also uses the shared workbench, rendering
marker-only residuals with a zero-error reference curve. Both curve views gain
the same zoom, pan, reset, autoscale, hover/crosshair, movable legend,
annotations, curve management, and plot-setting controls as Inference. The
specialized Error Distribution panel remains an absolute-error histogram with
its worst-sample summary. The other panels provide Auto candidate evidence or Custom
recommendation comparison, model-family comparison, and compact run provenance. The deterministic
sample-count, validation-gap, residual, outlier, and configuration-separation
findings are supplied to SnowBuddy with the latest saved-run facts rather than
repeated in a separate **What This Means** panel.

Custom suggestions use only completed Auto results whose full dataset
fingerprint, feature columns, target columns, train/test split, and Linear
Regression family match. Configuration advice uses validation RMSE rather than
test metrics, with a named 1% relative tolerance for negligible differences.
When no compatible Auto evidence exists, the page asks the user to run Auto and
does not invent a suggestion. Rendering is read-only and never retrains.

**Model Comparison** uses the currently displayed completed run as its evidence
anchor. It includes only runs with the exact same registered dataset ID and
fingerprint, feature columns, target columns, test size, and random state. From
each family it chooses the compatible run with the lowest available validation
RMSE; runs without validation evidence remain in the compatibility counts but
cannot become the family representative. The family recommendation then uses
validation RMSE only. A deterministic tie prefers Linear Regression as the
simpler family. Test RMSE, MAE, and R² are displayed beside validation RMSE and
in a compact visual metric comparison, but never select the recommendation.
The comparison can represent all four supported families and recommends among
the validation-backed compatible families; at least two are required.
Ensemble AI Engine can be recommended only when its compatible validation RMSE
is lower than the best individual family; held-out metrics never supply or
override that decision.
Each family card shows its training mode and selected parameters and can open
that immutable run's full Training Results. Opening a comparison run changes the
badge to **SELECTED** and resets the detailed result to **Predictions**;
reopening the project returns to the latest completed run by default.

The footer has two non-redundant actions. **Adjust & Train Again** is secondary
and returns to the still-configured Model Training page without starting a run.
**Create Model Book â†’** is the primary forward action and is enabled only for a
completed result. It asks for a Model Book name and keeps Results visible after
saving. Once that run has a saved book, the primary action changes to **Open
Model Library â†’**. Training remains enabled after completion, each new run
receives its own folder, and earlier run artifacts remain unchanged.

## Model Books

Saving a completed result creates a new sequential folder such as
`books/book-0001/`. It contains a copied `linear_regression_model.joblib` and a
self-contained `model_book.json`; the source `models/runs/run-####/` folder is
not modified. `books/index.json` tracks saved books and the explicitly selected
active book. Saving a book does not silently activate it or replace an existing
active selection. The new book is preselected when opened from Results so the
user can inspect it and choose **Set as Active**. Creating the first or a later
Model Book marks the five-step build workflow complete. Stage-aware Resume opens
Model Library until a book is active, then advances to **Run Inference**.

The Model Book manifest preserves:

- Model Book ID, user-provided name, version, and creation timestamp.
- Model name/type and a SHA-256 checksum for the copied estimator.
- Required input-feature columns, target/output columns, optional Sample ID
  metadata, and a structured ordered output axis with a neutral fallback.
- Exact parameters, training mode, search level, and deterministic split
  settings.
- Dataset ID and fingerprint.
- Available validation metrics and final test metrics.
- Source run ID, run number, and training timestamp.
- For Ensemble books, every component model artifact, exact component
  parameters, validation RMSE, source component run, normalized weight, and the
  inverse-validation-RMSE weighting method.

Names are unique within a project using case-insensitive comparison. An existing
book is never overwritten. A saved book can be loaded by ID or name and its
manifest and model checksum are validated before it is returned.

## Model Library

The project sidebar now opens a fixed, non-scrolling **Model Library** page.
It reads the existing `books/index.json` and shows five selectable books per
page. Each card identifies the book, model type, RMSE, R², input/output counts,
and ACTIVE, SELECTED, SAVED, or INVALID state. The selected-model panel keeps
model type, interface counts, required inputs, and the main performance metrics
prominent. Longer input lists have a **View all inputs** action. Source run,
creation time, training configuration, parameters, dataset fingerprint, and
Model Book version are kept secondary under the collapsible **Model Details**
section.

**Set as Active** persists the selection in both `books/index.json` and
`project.json`, so the same active book is restored when the project reopens.
The active book is identified by a non-actionable **✓ Active Model Book** badge.
An invalid or corrupted book remains visible with a friendly error and cannot
be selected as active; valid books in the same project remain usable. Empty
libraries direct the user back to **Training Results → Create Model Book**.

Model Library only browses and selects saved books; it does not run inference.
Open **Inference** to use the active Linear Regression, XGBoost, Neural Network,
or Ensemble AI Engine Model Book for
one local prediction.

## Basic inference page and backend

The fixed, non-scrolling **Inference** page reloads the active Model Book when
opened and generates one numeric field per saved feature. Interfaces with more
than eight inputs use input pages without discarding entered values. **Predict**
is disabled only during the backend call and is restored after success or
failure. A single output is shown as a prominent named value. Multi-output
predictions use compact output-count/minimum/maximum cards and the reusable
**Scientific Plot Workbench** instead of a long text dump. The workbench provides
major/minor grid lines, engineering tick labels, mouse-wheel and button zoom,
drag pan, reset, autoscale, curve hover/crosshair readouts, a draggable legend,
and click-to-place markers. **Plot Settings** controls the title, labels, limits,
linear/log scales when the visible values permit them, grids, legend visibility
and position, title/X-label/Y-label/X-value/Y-value/legend font sizes, legend
sample-line width, and the selected curve's width, line style, marker style, and
marker size. These controls are divided into **Axes & Grid**, **Text & Legend**,
and **Selected Curve** tabs so the dialog remains usable on laptop-sized screens.
Output axes stay neutral unless saved target names provide clear
frequency/theta/phi meaning; units are never invented.

Choose **Replace current curve** to update the selected curve or **Add to plot**
to overlay a new prediction. The compact curve manager selects, shows/hides,
renames, and deletes curves while preserving each curve's exact input-value
snapshot. The in-memory workbench survives page navigation while the same
project and active Model Book remain selected, but it is not prediction history.
The divider between the plot and curve manager is draggable and preserves the
chosen balance while the workbench remains open.
The active Model Book is reduced to a compact header with a **Model Info** action.
The workflow sidebar can collapse to icons, while SnowBuddy closes to a labeled
top-bar action and opens in a dedicated docked column. Neither action changes
navigation or chat state, and SnowBuddy never covers the plot or page actions.
If a successful prediction detects that the scientific canvas is below its
420-pixel usable-width threshold at a compact window size, SnowBuddy closes
automatically and the plot is redrawn; it remains open when sufficient plot space
is available.
**View Raw Values** opens the complete latest inputs and outputs in
saved order. **Export Prediction** writes a user-selected JSON file containing
Model Book metadata, ordered inputs, structured output-axis metadata, and ordered
predicted outputs, or an engineering-friendly curve CSV containing the saved
axis coordinate, predicted value, and output-variable name. Missing
values, invalid numbers, no active book,
corrupted books, and backend failures appear as inline messages without raw
tracebacks.

`studio.inference.InferenceRequest` accepts one dictionary of numeric feature
values and an optional active-Model-Book ID guard. Pass it to
`submit_inference_request(..., project_path=...)`. The service loads the active
book through the existing integrity-checking Model Book loader, requires the
exact saved inputs, rebuilds their saved order, loads the local joblib artifact,
and returns predictions keyed in saved target order. Current multi-output books
for all four model families are supported, including reproducible weighted
Ensemble predictions.

Missing or extra features, text/Boolean/non-finite values, an inactive requested
book, missing/corrupted metadata or artifacts, checksum failures, and invalid
serialized estimators return a structured `INFERENCE_FAILED` result without a
traceback. This workflow performs no batch or CSV inference and writes no
automatic inference history. The inference backend remains read-only; only the
explicit **Export Prediction** action creates a user-chosen JSON or curve CSV file.

## Inverse design

Open **Inverse Design** from the workflow sidebar or the forward action on the
Inference page. The page always reloads the active, integrity-checked Model Book.
Configuration stays on the left while the scientific result plot remains visible
on the right, so the user can adjust and rerun a search without leaving the
workspace.

In Configure, every saved feature must be assigned exactly one role:

- **Variable** — provide a finite lower and upper bound; the lower value must be
  smaller than the upper value.
- **Fixed** — provide the finite value retained throughout the search.

Choose **Single point** or **Mean over range**, then **Minimize**, **Maximize**, or
**Target value**. Single point accepts one numeric coordinate from the Model
Book's saved output axis. Mean over range accepts inclusive numeric start/end
coordinates and evaluates the arithmetic mean of every saved output within that
ordered range as one scalar objective. This keeps the workflow single-objective
without forcing the user to scroll through or type hundreds of output names.
Up to four optional generic output constraints can apply to one exact saved-axis
coordinate or the mean over an inclusive coordinate range, then require that
scalar to be at least a threshold, at most a threshold, or within value bounds.
Constraints are pass/fail filters; they do not replace the objective. With no
constraints, every successfully predicted design inside the configured input
bounds is eligible. No antenna quantity, unit, or response meaning is invented.

**Run Inverse Design** launches SciPy Differential Evolution on a background
thread with a fixed random seed. The optimizer proposes only the variable input
values; the Model Book predictor restores exact feature order, evaluates the
saved Linear Regression, XGBoost, Neural Network, or Ensemble artifact, and
returns outputs in exact saved target order. Objective and constraint evaluation
stay in `studio.inverse_design`, separate from all surrogate-model logic. This
first release is intentionally single-objective and provides no other optimizer
or Pareto workflow.

Each search reports the best input values, achieved objective value, explicit
constraint status, evaluation/iteration counts, and the complete ordered
surrogate response in the same Scientific Plot Workbench used by Inference. An
unconstrained result is labeled **Optimized**, while a constrained success is
labeled **Constraints Met**. A Target-value search is labeled **Closest Found**
and reports both the achieved value and numeric target gap; it does not imply
that an unattainable target was reached.
**Add to plot** preserves earlier optimized responses as separate curves;
**Replace selected curve** updates only the selected curve. Every curve retains
the corresponding optimized input values and can use the workbench's existing
rename, show/hide, delete, annotation, zoom, and Plot Settings controls.
When a newly added response matches an existing plotted response within the
named 0.01% tolerance, the Studio keeps both curves but warns that the designs
or responses are effectively identical.
The divider between Search Configuration and results and the divider between the
plot and Curves manager are draggable. Their positions remain stable while the
page stays open, allowing the user to choose more form, plot, or curve-list space.
The dividers stop at readable pane minima: Search Configuration keeps 520 pixels
and Curves keeps 220 pixels. Input roles, numeric fields, objective ranges, and
constraint controls stretch with their pane instead of retaining stale fixed
geometry, while explanatory and selected-curve text rewraps to the current width.
On a window too narrow to preserve the Inverse Design minima and a docked
SnowBuddy panel together, SnowBuddy uses its temporary focused panel rather than
compressing the engineering controls.
Legend labels use their available width rather than cutting every curve name at
an arbitrary character count.
Each successful click creates a separate immutable folder:

```text
inverse_design/runs/inverse-####/
├── request.json
├── result.json
├── best_prediction.csv
└── evaluation_trace.csv
```

`inverse_design/index.json` identifies the latest completed search, and
`project.json` stores its run count and latest run ID. Reopening the page restores
the latest result when it belongs to the currently active Model Book. Infeasible
constraint searches report that no design satisfying every output constraint was
found within the configured bounds and search budget. Validation/model failures
also show a friendly error. Neither failure creates a fake completed result.

The shared application type scale and scientific-plot defaults are at least 20%
larger than the preceding release. Layout spacing remains compact so the Studio
keeps its fixed, non-scrolling workflow pages at laptop resolution.

### LHS simulation-input workflow

Select **LHS sample generator** in Data Prep's Source subtask, then define each
solver variable in exact CSV column order:

| Variable | Min | Max |
| --- | ---: | ---: |
| patch_length | 20 | 40 |
| patch_width | 15 | 30 |
| feed_offset | 1 | 8 |

Set the sample count and, when reproducibility matters, a whole-number random
seed. **Generate Samples** uses `scipy.stats.qmc.LatinHypercube`, displays the
first five rows, and plots the first two variables as a neutral coverage check.
Both axes show their variable labels and numeric endpoint values. With one
variable, the preview uses sample index on the second axis. The preview
is only a design-coverage view; it is not a solver result.

**Export inputs.csv** writes the selected location atomically in this format:

```text
patch_length,patch_width,feed_offset
29.6452,27.2342,5.1399
34.2605,20.9859,2.19171
```

Column order matches the variable editor. Values stay within their configured
bounds, and the same settings plus seed reproduce the same design. The exported
table intentionally contains only simulation variables. Keep the exact row
order when assembling the later output CSV; without Sample IDs, Data Prep pairs
inputs and outputs by row position and verifies that their row counts match. A
blank seed intentionally creates a new random design. Generated files default
to the active project's `data/generated/lhs/` folder but may be explicitly
saved elsewhere.

Select **Create templates** to generate `inputs_template.csv`,
`outputs_template.csv`, and a local `README.txt` under the active project's
`data/templates/input_output/` folder. The generated tables use this structure:

| Sample ID | Input Parameter 1 | Input Parameter 2 | Input Parameter 3 | ... |
| --- | ---: | ---: | ---: | ---: |
| Design_001 | Value | Value | Value | ... |
| Design_002 | Value | Value | Value | ... |
| Design_003 | Value | Value | Value | ... |

| Sample ID | Output 1 | Output 2 | Output 3 | ... |
| --- | ---: | ---: | ---: | ---: |
| Design_001 | Value | Value | Value | ... |
| Design_002 | Value | Value | Value | ... |
| Design_003 | Value | Value | Value | ... |

The generated CSVs contain numeric example values in place of `Value`, so the
pair can be validated immediately. Rename, add, or remove parameter and output
columns for the antenna. Keep `Sample ID` first in both files and keep the IDs
unique and in matching row order. The Studio uses the IDs to verify pairing,
preserves them for end-to-end result traceability, and excludes them from the
numeric model features and targets.

Output headers should name the response, coordinate, and unit. For example:

```text
S11 at 1 GHz, S11 at 5 GHz, S11 at 10 GHz
```

or:

```text
Gain at theta -90 deg, Gain at theta 0 deg, Gain at theta 90 deg
```

For responses that vary over both frequency and angle, use one output column
per coordinate pair, such as `Gain at 1 GHz / theta -90 deg`, continuing from
the required minimum to maximum ranges.

Scalar responses such as `Efficiency (%)` or `Resonant frequency (GHz)` are
also valid. User-edited template files are never overwritten. Untouched stock
templates from Studio 0.9.0 or 0.9.1 are safely upgraded to the Sample ID
contract. Template paths are loaded into Data Prep automatically so the user
can inspect the format or replace the examples with simulation data. With both
paths loaded, the paired-CSV flow runs automatically.

The desktop uses the Studio-specific **Instrument Lab** theme with complete
Light and Dark palettes. Light is the default, using pale pearl work surfaces,
white instrument panels, teal controls, and dark high-contrast readouts. The
top application bar’s **Appearance** toggle switches the live interface
between Light and the optional graphite Dark palette without resetting the
current page, project, form values, or SnowBuddy chat. The selection is stored locally in
`studio_settings.json` and restored on the next launch. Both modes retain mono
lab labels, bright state indicators, a restrained violet SnowBuddy accent, and
the 12–32 point readability-focused type scale.

The top application bar also provides **File**, **Edit**, **Help**, and a labeled
**SnowBuddy** open/close action. File includes project creation,
project opening, Return to Welcome, and Exit; Help exposes local-model settings
and About. On Start, Create/Open appear in the Active Workspace hero only when
no project is open. An active project instead shows one stage-aware action for
Data Prep, validation/registration, Model Training, Training Results, or Inference.

`AGENTS.md` makes updating the blind GUI map part of the repository’s
development contract. Unit tests compare its recorded hashes with
`studio/ui.py` and `studio/theme.py`, so a GUI source change fails validation
until the map is reviewed and synchronized.

## Developer checks

```text
python -m unittest discover -s tests -v
```

The core parser, project library, chat persistence, and assistant retrieval
layers use only Python's standard library.
