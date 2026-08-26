# SnowBuddy Blind GUI Read

Contract version: 3.47
Studio version: 0.33.1
UI source SHA-256: af40a487818b452e3f976943b5a4a37f1dc2d2dbee35f96569fb28d2e42cce28
Sample Generator UI source SHA-256: 7c59e5fdd2f1ac1a92fe42cadeb914a3a9e36e6252432dae1d2be2298b39b6fe
Results UI source SHA-256: fe0f2fe92ca4d7273c86c24e1bb9e8b0981adef0ac3390d62f60787c4f7ce7ee
Library UI source SHA-256: 6449a5822e601ae4f609c552e04534b084c39440aaf042c936a2db8f1a02b8b0
Inference UI source SHA-256: 4945d117a680c14532a0a4877aa372c8a28b02c512245aa5666de7d2308f580d
Inverse Design UI source SHA-256: 0e06eec31804af981f8e0c2150d349903ff246bea9b9b90a9409aeae469c7b2c
Scientific Plot UI source SHA-256: 3ecb1880ce404907221a627777edeb1a4285119dd7f45e1df2ed9f9b85c0f001
Theme source SHA-256: c1149c09ec5cd35f71710288f9067c6949e00088b8a8d0470011e38ed07aedb8

This file is SnowBuddy’s visual and interaction map. It describes the interface
without assuming screen vision. The live UI-state snapshot supplied at runtime
adds the user’s current page, values, selections, and status.

## Visual language and global shell

- The desktop window is titled “Antenna Surrogate Studio.”
- The Studio-specific Instrument Lab theme has complete Light and Dark
  palettes. Light is the default: pale pearl work surfaces, white instrument
  panels, cool steel borders, teal controls, dark high-contrast readouts, and
  compact mono lab labels. Dark is the optional graphite laboratory palette
  with luminous cyan controls.
- Window geometry is normalized to physical pixels on DPI-aware Windows. The
  no-scroll workspace is supported at 1366 by 768 at 100%, 125%, and 150%
  scaling by bounding the effective CustomTkinter UI scale to 1.08. The initial
  window is at most 1440 by 900, never larger than the physical monitor, and a
  1366-by-768 monitor starts in compact-sidebar mode. The shared type scale runs
  from 15 through 39 points; body copy is 16 or 17 points, captions are 15
  points, and buttons are 16-point semibold. Plot titles, labels, tick values,
  legends, and legacy page-local text are also at least 20% larger than in the
  preceding release.
- SnowBuddy uses a violet accent. Success is green, warnings are amber, and
  errors are red.
- A fixed top application-menu row spans the window. “File,” “Edit,” and
  “Help” sit on the left; a labeled “SnowBuddy” open/close action and the
  “Appearance” Light/Dark control sit on the right.
- File contains New project, Open project, Return to Welcome when available,
  and Exit. Edit is a reserved placeholder. Help opens SnowBuddy local-model
  settings or the About dialog.
- A light-steel or graphite workflow sidebar sits on the left. Expanded, its
  brand subtitle is “RF SURROGATE LAB,” its navigation label is “LAB WORKFLOW,”
  and its footer reads “LOCAL COMPUTE · PRIVATE.” A clear chevron collapses it
  from 226 pixels to a 76-pixel icon-only rail. The same control expands it.
  Navigation remains active and the chosen state survives page changes for the
  current application session. The Active Project card and footer hide only in
  compact mode; no project or page state is changed. Each compact icon exposes
  its page name on pointer hover and keyboard focus and has the same accessibility
  name.
- SnowBuddy is not allocated a permanent workspace column while closed. The
  labeled “SnowBuddy” action in the top application bar opens a dedicated
  390-pixel right-side column only when that leaves at least 980 design pixels
  for the active page. On narrower windows, SnowBuddy opens as a temporary
  focused assistant view: the current page is hidden, not squeezed or covered,
  and closing SnowBuddy restores the unchanged page and full scientific-plot
  width. Chat state and Welcome/Focus mode are preserved in either placement.
  After successful Inference, the Studio still closes SnowBuddy automatically
  if a programmatic prediction began while the focused view hid the plot.
- The top row’s “Appearance” two-state control offers “Light” and “Dark.”
  Switching recolors the existing interface immediately without changing the
  active page, project,
  form values, or chat. The choice is global for this local Studio library and
  is restored from `studio_settings.json` on the next launch.
- The sidebar contains the AS brand badge, Start, Data Prep, Model Training,
  Training Results, Model Library, Inference, and Inverse Design. Collapsed mode shows those same
  destinations as icons. All project workflow pages are available
  while a project is active. Selecting one without a project returns to Start
  with an Open project message.
- The sidebar’s Active Project card shows the open project name and workflow
  status, or “No project open.” With a project active it also shows “Return to
  Welcome.”
- A fresh application launch stays in Welcome mode with no project silently
  preloaded. Creating a project opens Data Prep immediately. Opening an
  existing project restores that project’s last active page.

## Start page

The page heading is “Your surrogate workspace” with the subtitle “Build trusted
antenna models. Save them as books. Reuse them anytime.”

### Active workspace hero

- A pale blue instrument card in Light mode or blue-black instrument card in
  Dark mode, with a teal/cyan border, shows the active project name, next
  action, progress bar, and completed-step count.
- With no project it says “Start something precise” and “No active project.”
- With no project, the hero offers “+ Create project” and “Open project.”
- With a project active, those empty-state actions disappear and the hero shows
  one stage-aware resume action: Continue Data Prep, Validate & Register Data,
  Continue Model Training, Review Training Results, or Run Inference. New/Open
  remain available from File.

### Recent projects

- An appearance-aware “Recent projects” instrument panel displays up to five
  project icon cards in latest-opened order.
- Each card shows project name, relative last-opened time, and status.
- Clicking a project card opens that project.
- When empty, the card says “Your project shelf is empty” and offers “Create
  project.”
- The five cards share one fixed responsive row. The Start page has no page
  scrollbar.
- A fixed bottom workflow footer mirrors the same stage-aware destination as
  the hero action. It is enabled only while a project is open and may read
  Continue Data Prep, Validate & Register Data, Continue Model Training,
  Review Training Results, Open Model Library, or Run Inference.

## SnowBuddy companion panel

- Header: violet star avatar, “SnowBuddy,” the automatically selected Welcome
  or Focus mode plus model status, and a “Local model” or “Model settings”
  button.
- Middle: scrollable local project-chat history. User messages and assistant
  messages use separate rounded bubbles whose contrast follows Light or Dark.
- Bottom: multiline composer, cyan/teal up-arrow send button, and a mode-specific
  hint. Welcome says “Welcome session · Local history · Ctrl+Enter to send.”
  Focus identifies the active project.
- With no project, chat is enabled in Welcome mode. Each application launch
  starts a fresh local Welcome session. SnowBuddy can explain the workflow,
  help choose Create or Open, and check local-model setup.
- Welcome sessions are stored under
  `Antenna Surrogate Studio Library/assistant/welcome_sessions/`. Older sessions
  remain archived locally but are not mixed into a new session’s context.
- Creating or opening a project automatically switches to Focus mode and that
  project’s separate `assistant/chat_history.json`.
- Historical messages remain visible locally. Before a prompt reaches Ollama,
  known obsolete assistant claims—such as retired source controls, only one model
  being available, Model Library being future work, or changing the fixed split—
  are excluded with their paired question. The live snapshot and current contracts
  remain authoritative; saved history is not rewritten.
- “Return to Welcome” removes the active project and restores the current
  launch’s Welcome session.
- The same SnowBuddy panel remains visible while the user moves between Start,
  Data Prep, and Model Training; navigation does not interrupt or relocate the
  conversation.
- Data Prep has no redundant SnowBuddy navigation button because the companion
  is already present and follows the active state automatically.

## Data Prep page

The page heading is “Data Prep” with the subtitle “Generate simulation inputs,
validate paired CSVs, or convert #Parameters exports into model-ready tables.” SnowBuddy remains visible
in the persistent right-side companion panel.

A vertical connected subtask accordion shows:

1. Source — Choose pair or export.
2. Variables — Define model contract.
3. Prepare — Write input/output tables.
4. Register — Validate and lock dataset.

Each subtask header includes a circular status symbol: hollow for pending,
cyan for active, green check for complete, and red exclamation for an error.
Only one subtask body is expanded at a time. Opening another collapses the
current body and the connected rows move up or down, similar to opening one
file inside a compact editor folder. Data Prep has no page scrollbar.

A fixed bottom footer provides “Back to Start” and an enabled “Next: Model
Training” button. The latter opens the Model Training page without starting
training.

### 1 · Select data source

- Segmented source-mode control:
  - “Input + output files”
  - “#Parameters sweep”
- “LHS sample generator” and the violet “Create templates” action sit beside
  the source-mode control. Both work locally and require neither SnowBuddy nor
  an installed language model.
- “LHS sample generator” opens a resizable, non-scrolling project-local dialog.
  The left panel has user-defined Variable name, Min, and Max rows; Add variable
  and per-row remove actions; five visible rows per page for as many as 20
  variables; Samples; optional Seed; and Generate Samples. The right panel shows
  a neutral sampling-coverage plot and the first five generated rows. The footer
  keeps validation/status copy and Export inputs.csv visible. At least one
  variable is required. Names must be non-empty and unique without using the
  reserved `sample_id` name. Bounds must be finite numbers with Min below Max;
  sample count must be a whole number from 1 through 100,000; and a supplied seed
  must be a whole number from 0 through 4,294,967,295.
- Generation uses SciPy Latin Hypercube sampling. The same settings and seed
  reproduce the same samples. Coverage plots the first two variables, or one
  variable against sample index, without inventing units. Both axes show the
  applicable variable label plus visible numeric minimum and maximum endpoints.
  Editing a setting
  invalidates the existing preview and disables export until samples are
  regenerated.
- Export writes a user-chosen CSV containing only variables in the editor order;
  it does not add a Sample ID column. The default project location is
  `data/generated/lhs/inputs.csv`. It loads only the generated Input CSV path,
  clears any Output CSV path to prevent a stale pairing, collapses later subtasks,
  and explains that solver outputs with the same row count and unchanged row
  order are still required. It does not run CST/HFSS, fabricate outputs,
  prepare/register a dataset, or start training.
- Input + output mode exposes two path rows labeled “Input CSV” and “Output
  CSV,” each with its own Browse button. There is no Parse or Analyze button in
  this mode.
- As soon as both CSV paths are present, the Studio automatically validates the
  pair, adopts every numeric input column as a feature, preserves every numeric
  output column as a target, and prepares project-local copies. The user is not
  asked to parse the pair or select variables.
- Both files require non-empty, unique headers, finite numeric
  parameter/output data, and the same number of sample rows. An optional
  `Sample ID` column must be first in both files; IDs must be non-empty, unique,
  and match in row order. Sample ID validates pairing and is preserved in both
  prepared and registered files for result traceability, while remaining excluded
  from numeric model features and targets.
- Create templates writes `inputs_template.csv`, `outputs_template.csv`, and
  `README.txt` under `data/templates/input_output/`, loads both CSV paths into
  the page, shows a confirmation, and opens the local template folder. Both use a
  first `Sample ID` column with `Design_001` through `Design_003`. Inputs are
  labeled `Input Parameter 1` through `Input Parameter 3`; outputs are labeled
  `Output 1` through `Output 3`. The instructions explain that users can add or
  remove variables and rename response columns for frequency, theta, or another
  antenna-specific axis. User-edited templates are not overwritten; untouched
  older stock templates are upgraded.
- #Parameters mode exposes one source-path field with “Browse file” and “Browse
  folder,” followed by “Parse.” Parse is shown only for this raw-extract mode
  and expects blocks beginning with `#Parameters = {...}`, a quoted response-table
  header, and numeric rows. The first quoted/numeric table column is the ordered
  output coordinate. It may be Frequency, Theta, Phi, or another solver-defined
  coordinate; preparation preserves that header meaning and any recognized unit
  in response-aware output names rather than hard-coding `theta_*`. For example,
  `"Frequency / GHz" "S11"` becomes columns such as
  `S11 at Frequency 1 GHz`.
- #Parameters sweep parses completed solver results. It is not the LHS sample
  generator, which only proposes unsimulated input rows. Matching prepared input
  and output row counts/order are required; their column counts may differ.

### 2 · Define the surrogate contract

- Before raw parsing, this area asks the user to parse a raw extract.
- For an input/output pair, this subtask is completed automatically. It reports
  the adopted input/output counts and does not display feature checkboxes.
- After parsing a #Parameters extract, the left “Model inputs” panel lists
  discovered variables as checkboxes in a wrapping, non-scrolling grid. The
  right “Pattern output” panel provides one output dropdown.
- A discovery summary reports samples, available inputs, and available outputs.
- At least one input is always required. #Parameters mode additionally requires
  exactly one output response.
- The raw-extract contract is not committed merely by changing a checkbox or
  dropdown. **Save selection** writes the selected inputs and output to the
  project manifest, changes to **Selection saved**, enables Prepare, and opens
  subtask 3. Changing either selection disables Prepare until the new contract
  is saved again. A saved contract is restored on project reopen.
- Input/output CSV pairs do not show Save selection because their complete
  feature/target contract is adopted automatically.

### 3 · Prepare the input/output tables

- A status strip describes the current operation or readiness state.
- For an input/output pair, preparation starts automatically after the pair is
  validated; the user is not asked for another action.
- For a parsed #Parameters extract, “Prepare input + output” becomes available
  after the discovered contract is explicitly saved in subtask 2.
- Preparing writes matched `data/prepared/inputs.csv` and
  `data/prepared/outputs.csv` tables plus `data/prepared/schema.json` inside the
  active project. Both tables use identical row ordering: one input row maps to
  the output row at the same position.
- On success, the first three subtask symbols turn green, Register expands, the
  status reports samples and output columns, and an “Input and output files
  ready” confirmation dialog lists both saved paths.
- Reopening a successfully prepared project restores the selections and green
  completion state only when both CSV files still exist. The actions become
  “View prepared files” and “Regenerate both files”; preparation is not
  required again.
- “View prepared files” opens the project’s local `data/prepared` folder.
- If either member of the prepared pair is missing, the page warns that an
  input or output file is missing and offers to regenerate both instead of
  claiming readiness.
- A project with the older combined `training_data.csv` format shows a legacy
  notice and the “Generate separate files” action. Successful regeneration
  replaces that generated legacy table with the new pair.

### 4 · Validate and register dataset

- This accordion subtask becomes available only after both prepared CSV tables
  exist.
- The primary action is “Validate and register.” One click first validates the
  prepared pair and only registers it when every validation check passes.
- Validation checks requested input and output columns, matching non-empty row
  counts, row widths, finite numeric feature and target values, and optional
  Sample ID presence, uniqueness, and alignment.
- The result panel remains visible on the page. Before validation it says no
  result is available; while running it describes the active checks; on success
  it reports sample, input, and output counts plus the dataset ID and abbreviated
  fingerprint; on failure it retains the error explanation for correction.
- Registration creates a local integrity-protected snapshot under
  `data/registered/dataset-<fingerprint>/`, updates the project registry, and
  does not upload data.
- Re-registering identical data reuses the same content-based dataset ID.
  Changed data or a changed column contract creates a new registered dataset.
- “View registered files” opens the active registered dataset folder and is
  enabled after successful registration.
- The card explicitly says training is a separate step and is never started by
  Validate and register. The Model Training page is the separate action that
  can run a supported model family after registration.
- Reopening a registered project restores the validation metrics, dataset ID,
  integrity status, and View registered files action.

## Model Training page

The page heading is “Model Training” with the subtitle “Configure how the
surrogate model will be trained.” A “LOCAL · MODEL TRAINING” badge makes the
page boundary explicit. The page is fixed and has no page scrollbar. SnowBuddy
remains visible in Focus mode on the right.

Three stacked instrument cards provide:

1. Model selection — a future-extensible dropdown containing “Linear
   Regression,” “XGBoost,” “Neural Network,” and “Ensemble AI Engine.”
2. Training mode — an immediate Auto/Custom segmented control. Auto is selected
   by default.
3. Advanced Settings — model-specific controls shown only in Custom mode:
   `fit_intercept` and `positive` for Linear Regression; five numeric XGBoost
   fields; or Neural Network hidden layers, activation, learning rate, batch
   size, and epochs.

The Model selection and Auto Search Level dropdowns use integrated, rounded
Instrument Lab fields rather than bright generic split buttons. Each has the
current palette’s control surface, subtle border and arrow well. Its expanded
menu uses the current Light or Dark surface, matching ink text, and the same
teal-tinted navigation hover used elsewhere in the Studio.

In Auto mode, an “Auto Search Level” panel is visible. Its dropdown contains
“Medium” and “High,” with Medium selected by default. The two descriptions are
“Medium — Faster, lower-compute deterministic search” and “High — Slower, more
thorough deterministic search.” The entire Advanced Settings card is hidden,
and its underlying controls are disabled and cannot be edited.

Selecting Custom immediately hides Auto Search Level, shows a Custom-mode note,
and reveals Advanced Settings. For Linear Regression it enables the
`fit_intercept` and `positive` switches, defaulting to on and off. For XGBoost it
shows one compact five-column row containing `n_estimators`, `max_depth`,
`learning_rate`, `subsample`, and `colsample_bytree`. The displayed defaults are
64, 4, 0.1, 1.0, and 1.0. Small range labels show 1–5000, 1–64, and greater than
0 through 1 for the three fractional fields. Selecting Auto again hides the
entire Advanced Settings card and disables its underlying controls.

For Neural Network Custom, one compact five-column row shows Hidden layers,
Activation, Learning rate, Batch size, and Epochs. Hidden layers accept a
comma-separated architecture such as `64, 32`. Activation offers relu, tanh,
logistic, and identity. Defaults are `64, 32`, relu, 0.001, 8, and 180.
Inputs are standardized automatically; the backend contract remains the final
authority for architecture, activation, numeric types, and ranges.

Selecting XGBoost leaves the Auto/Custom segmented control available. In Auto,
Auto Search Level is visible and Advanced Settings is hidden. Its request maps
to `model_name=xgboost`, `training_mode=auto`, the selected lowercase Medium or
High search level, and no custom parameters. In Custom, the request uses
`training_mode=custom`, no search level, and all five numeric field values.
Backend request validation remains authoritative for completeness, names,
numeric types, and ranges.

Selecting Neural Network uses `model_name=neural_network`. Auto maps the
selected Medium/High level with no Custom parameters. Custom supplies the five
displayed values and no search level.

Selecting Ensemble AI Engine uses `model_name=ensemble_ai_engine` and locks the
page to Auto High. The Auto/Custom control is disabled on Auto, the editable
search-level panel and Advanced Settings are hidden, and a concise note states
that Linear Regression, XGBoost, and Neural Network will be trained in Auto High
and weighted from validation RMSE. The backend contract independently rejects
Ensemble requests that are not Auto High or include Custom parameters.

In Custom mode, Advanced Settings stacks directly beneath the Training mode
card with the same compact seam used between the other workflow cards. Flexible
empty space sits below Advanced Settings, not between the two cards, so the
Train Model action remains fixed near the bottom without creating a visual gap.

The “Train Model” button is enabled with an active project. Selecting it creates
and validates a `ModelTrainingRequest` from the current controls. Auto maps to
`model_name=linear_regression`, `training_mode=auto`, the selected lowercase
search level, and no custom parameters. Custom maps to
`model_name=linear_regression`, `training_mode=custom`, no search level, and the
two Boolean switch values. The backend revalidates the request as the final
authority.

Before training, the button is enabled and reads “Train Model.” While the
backend is training, it is disabled and reads “Training…,” preventing duplicate
submissions. After either success or failure, it is enabled again and returns
to “Train Model.”

The left side of the action bar displays the persisted latest-run readout. It
shows “Latest Run: None” before the first successful run and then uses the exact
format “Latest Run: Run 3.” Reopening the project restores this readout.

Linear Regression executes in Auto Medium, Auto High, or Custom. All modes load
the active integrity-checked registered dataset and create the same deterministic
80/20 split with random state 42. Auto performs cross-validation only on the
training partition. Medium evaluates two non-positive configurations with 3
folds. High evaluates all four Boolean combinations with 5 folds. The fold count
is reduced to the available number of training rows when needed, with at least 2
folds required. Lowest mean validation RMSE wins; ties prefer `positive=false`
and then `fit_intercept=true`. The selected model is fitted on the full training
partition and evaluated once on the untouched test set. Custom skips search and
uses the validated switch values directly.

XGBoost uses the same registered dataset and deterministic 80/20 split. Auto
uses training-only deterministic cross-validation. Medium evaluates 3 bounded
configurations with 3 folds; High evaluates 6 with 5 folds. Each candidate uses
the five displayed XGBoost parameters. Lowest mean validation RMSE wins; equal
scores use stable candidate order. The selected configuration is refitted on
the full training partition and evaluated once on the untouched test partition.
Custom directly uses the five displayed values. Both modes preserve the fixed
objective, `random_state=42`, single-worker execution, histogram tree method,
and other runtime settings, and support the current multi-output target matrix.

Neural Network uses the same deterministic split and multi-output target
matrix. Every candidate is a saved scikit-learn pipeline with StandardScaler
input standardization and MLPRegressor. Medium evaluates 3 bounded
architectures/settings with 3 folds; High evaluates 6 with 5 folds. Cross-
validation uses only the training partition, the lowest mean validation RMSE
wins, and ties use stable candidate order. The selected network is refitted on
the complete training partition and evaluated once on the untouched test set.
Custom applies the displayed architecture, activation, learning rate, batch
size, and epoch budget directly. Fixed reproducibility settings include Adam,
`random_state=42`, no epoch shuffling, and no early stopping.

Ensemble AI Engine creates Auto High runs for all three individual families,
then creates the Ensemble run last. A failed component is recorded and the
remaining components continue when at least two valid models remain. Weights
are normalized inverse validation RMSE, never held-out test error. The selected
component configurations are evaluated together on the shared training-only
folds to obtain the Ensemble validation RMSE, then their full-training models
produce one weighted held-out prediction. The Ensemble is recommended only
when its validation RMSE is lower than the best individual model.

Every successful click creates the next sequential folder, such as
`models/runs/run-0003/`, without overwriting earlier runs. Each folder contains
the family-specific `linear_regression_model.joblib`, `xgboost_model.joblib`,
`neural_network_model.joblib`, or `ensemble_ai_engine_model.joblib`, plus `metrics.json`, `test_predictions.csv`,
`training_config.json`, and `run.json`. Auto folders also contain
`auto_search_results.json` with every configuration, fold RMSE values, failed
candidate records, mean validation RMSE values, the actual fold count, selected
parameters, and final test metrics. The configuration artifact records the
training mode, search level, and exact parameters used. The saved predictions
CSV contains sample ID, target name, actual value, predicted value, and
residual.
Ensemble runs also save `ensemble_results.json` and separate immutable component
artifacts under `components/`, including component parameters, validation RMSE,
source runs, normalized weights, failures, and the Ensemble-versus-individual
recommendation evidence.
Supplied registered sample IDs are preserved; registered datasets without an ID
receive stable `Sample_000001`-style row IDs. The project manifest records
TRAINING_COMPLETED, the dataset ID, parameters used, split sizes, metrics, and
artifact paths. The page retains the last request and structured result.

After a successful Auto run, the dialog title is “Auto Search Completed.” Its
body shows the selected model family, Search Level, Configurations Evaluated,
Cross-Validation Folds, that model's Best Parameters, Validation RMSE, Test MAE,
Test RMSE, and Test R². A successful Custom run retains the “Training Completed”
dialog with its mode, applied parameters, and test metrics. Neither dialog shows
fake metrics.

For XGBoost Auto, the same Auto Search Completed dialog lists all five selected
values. XGBoost Custom shows Training Mode: Custom, all five selected values,
and the real test metrics. Training Results uses “AUTO BEST” for current Auto
search runs and “CUSTOM USED” for Custom. Configuration shows the Medium/High
summary and every evaluated candidate with its validation RMSE, failure state,
and selected marker. Legacy fixed-baseline XGBoost runs remain readable and keep
their historical “FIXED BASELINE” presentation.

Neural Network Auto uses the same Auto Search Completed dialog and shows hidden
layers, activation, learning rate, batch size, and epochs. Neural Network
Custom shows those five applied values and the real test metrics. Results uses
the same AUTO BEST or CUSTOM USED distinction and renders every saved candidate.

Ensemble completion uses “Ensemble Training Completed.” It shows the number of
valid/failed components, normalized weights, Ensemble and best-individual
validation RMSE, the validation-based recommendation, and final test metrics.

After either successful dialog is dismissed, the Studio opens Training Results
for the newly completed run. The result is also reloaded whenever the project is
reopened or Training Results is selected from the sidebar.

If request validation fails, a user-facing “Invalid training configuration”
dialog shows the contract message and the backend is not called. Dataset or
execution failures use a “Training failed” dialog. Neither exposes a raw
traceback.
The fixed footer provides “Back to Data Prep” and an enabled “View Training
Results” action. The latter reads saved run artifacts and never starts training.

## Training Results page

Training Results is a fixed, non-scrolling, artifact-backed page for the latest
completed run by default. Its visualizations never retrain or modify that run;
the footer can copy it into a new Model Book. SnowBuddy retains the shared
floating/drawer behavior. The header shows “Training Results” and a latest-run
badge. Opening an older family run from Model Comparison changes the badge to
SELECTED and opens that immutable run's Predictions detail. Before any completed run it displays
“No completed training run is available yet. Train a model to view performance
and prediction plots.” A failed attempt can display “Training did not
complete. No performance results are available for this run.” Missing or
malformed artifacts produce a friendly saved-artifact error without a traceback
or partial metrics. Loading another project, reopening a project, or completing
a new training run always resets the ordered section navigator to Predictions.
A secondary section selected for one displayed result is never carried into a
different project or run. Manual section changes remain active while the user
continues viewing the same result.

The 44-pixel configuration strip and four metric cards remain above the ordered
section navigator. The strip has only a mode badge and the applied parameters:
AUTO BEST with the selected parameters, or CUSTOM USED with the user's
parameters. XGBoost summarizes trees, depth, and learning rate; Neural Network
summarizes hidden layers, activation, and learning rate. Ensemble AI Engine
shows a compact normalized-weight summary and an ENSEMBLE RECOMMENDED or
ENSEMBLE EVALUATED badge. It has no
validation RMSE, search summary, Auto suggestion,
comparison guidance, evidence/no-evidence message, or action button. Auto and
Custom parameter evidence appears only in Configuration; family-level evidence
appears only in Model Comparison. Auto's Configuration header contains
search level, configuration count, folds, and lowest validation RMSE above the
candidate table. Custom's Configuration panel contains the compatible
side-by-side suggestion or the Run Auto guidance.

The four cards are R², RMSE, MAE, and Validation RMSE. A visible **LATEST
SELECTED RUN METRICS** label names the displayed run and model above them so
these values cannot be mistaken for the separate family recommendation. They are compact
62-pixel, value-first tiles: only the metric name, saved numeric value, and a
small ? help control remain visible. Hovering the ? or moving keyboard focus to
it opens a temporary floating explanation with the plain-language definition
and whether higher or lower is better. Leaving the control closes the help.
This reclaimed vertical space is given to the active plot. A unit appears only
when explicit target-unit metadata exists; headers are never guessed for units.
Negative R² receives a baseline comparison rather than a universal good/bad
label.

Six ordered navigation buttons keep the page free of page scrolling, with one
detail panel visible at a time:

1. Predictions — the default and primary panel. It groups the saved prediction
   rows by test sample and shows one selected design at a time. A dropdown lists
   the test sample IDs. A compact highlighted strip shows the selected design's
   registered input-feature names and values. The reusable Scientific Plot
   Workbench overlays Actual as a
   solid teal line with circular markers and Predicted as a dashed violet line
   with diamond markers. Its title includes the selected sample ID; the legend,
   X/Y axis titles, tick labels, and hover details are deliberately prominent.
   Hover details include coordinate, output name, actual, predicted, and
   residual. **Test Sample** is a full-width dropdown inside the workbench's
   Curves panel, above the Actual and Predicted entries. The former inline
   Sample/Min/Max/Step/Apply row is absent. The
   Y-label defaults to Response value plus the saved target unit when available.
   Plot title, X/Y labels and limits, linear/log scales where supported,
   major/minor grids, title/axis/value/legend font sizes, legend placement and
   line sample width, and selected-curve line/marker settings are all edited in
   Plot Settings, matching Inference and making it the only axis-limit editor.
   User-defined limits persist when the selected test sample changes. The
   workbench also provides zoom, pan,
   reset, autoscale, hover/crosshair, movable legend, annotations, and curve
   show/hide, rename, and delete controls.
   The Studio infers
   Frequency, Theta, or Phi
   and a uniform range from output-column names when possible; otherwise it uses
   editable 1-based output-point indices. Apply rejects missing, nonnumeric,
   non-finite, zero/negative-step, reversed, non-divisible, or wrong-point-count
   ranges with a clear dialog and leaves the current plot unchanged. One output
   column produces two
   markers and explains that multiple columns are required for curves. A
   compact Open Test Data CSV action opens `test_predictions.csv`; no long
   sample table is rendered.
2. Residuals — the same Scientific Plot Workbench, with marker-only values of
   `Actual − Predicted`, a dashed horizontal zero-error reference curve, sample
   and output identity in hover details, and a measured directional-bias note.
3. Error Distribution — an absolute-error histogram plus median error, maximum
   error, and the largest-error sample with actual and predicted values.
4. Configuration — Auto shows every candidate ordered by successful validation
   RMSE, failed candidates and reasons, and a clear Selected marker. Custom shows
   user and suggested parameter/validation/test summaries side by side when a
   compatible Auto run exists.
   Ensemble instead shows each component, validation RMSE, normalized weight,
   source run and status, plus recorded failures and the validation-only
   Ensemble-versus-best-individual decision.
5. Model Comparison — anchors compatibility to the currently displayed run and
   includes only completed runs with the same registered dataset ID and full
   fingerprint, exact feature columns, exact target columns, test size, and
   random state. Four concise family cards show the best validation-backed
   Linear Regression, XGBoost, Neural Network, and Ensemble AI Engine run: mode, selected parameters, validation
   RMSE, test RMSE, MAE, and R². Each card has Open Run N Results for detailed
   Predictions, Residuals, Errors, Configuration, and Run Info. The banner says
   Recommended Model with the selected family only when at least two families
   have valid compatible validation evidence. A compact bar view
   visualizes validation RMSE, test RMSE, MAE, and R² with exact values. Test
   metrics are context only and never choose the recommendation. The section
   title is explicitly **MODEL FAMILY COMPARISON**. Its bars represent relative
   quality rather than raw magnitude: longer is always better, downward arrows
   identify Validation RMSE/Test RMSE/MAE as lower-is-better, and an upward
   arrow identifies R² as higher-is-better. Exact values remain beside the bars.
6. Run Info — run ID, model, mode, search level, parameters, training/test
   samples, full dataset fingerprint, and training timestamp.

There is no separate What This Means panel. The backend's deterministic
sample-count, validation/test-gap, residual, error-concentration, Auto-separation,
Custom-comparison, and model-family recommendation findings are added to
SnowBuddy's latest-run context.
SnowBuddy may explain those saved facts conversationally but must not invent
evidence that is absent from the context.

Custom-to-Auto comparison requires the same full dataset fingerprint, exact
feature and target columns, test size, random state, and Linear Regression
family. The best compatible Auto run is selected by validation RMSE. If the Auto
search evaluated the exact Custom parameter pair, a named 1% relative
validation-RMSE tolerance determines Use Auto, performs similarly, or Custom is
stronger. Test metrics are displayed but never select the recommendation. If no
matching validation score exists, the page shows the suggested Auto parameters
without claiming they outperform Custom.

Model-family comparison chooses the lowest-validation-RMSE compatible run from
each family, regardless of which compatible run is newest. Runs without valid
validation RMSE remain counted but cannot represent a family. At least two
families must have validation evidence before a recommendation appears. If top
scores tie within the fixed deterministic tolerance, the earlier simpler family
in the fixed Linear Regression, XGBoost, Neural Network, Ensemble AI Engine
order is preferred. Ensemble is recommended only when its validation score is
strictly lower than the best individual family.
Different datasets, interfaces, or split configurations
are excluded. Malformed runs are ignored safely. Held-out test RMSE, MAE, and
R² never break a tie or substitute for missing validation evidence.

The fixed footer provides two non-redundant actions. **Adjust & Train Again** is
the secondary left action and returns to the still-configured Model Training
page without starting a run. **Create Model Book →** is the primary right action
and is enabled only when a completed result is loaded. It opens a themed name
prompt and creates a new Model Book without modifying the displayed run. Blank
names, duplicate names, failed/incomplete runs, and missing required artifacts
produce clear dialogs. Results remains visible after success, the footer names
the saved book, and the primary action changes to **Open Model Library →**.
Opening it preselects the saved book for inspection without silently activating
it. Each new successful training run remains immutable and becomes the latest
displayed result. Creating a Model Book marks the build workflow 5 of 5 complete;
Resume opens Model Library until a valid book is explicitly active, then changes
to Run Inference.

## Model Book contract

Model Book saving, Model Library, and single-sample Inference are implemented.
Every successful save creates the next project-local
`books/book-####/` folder. It contains the copied trained model artifact and
`model_book.json`. `books/index.json` records the ordered books and active book.
Saving preserves the previous active-book ID, including `None`; activation is a
separate Model Library action.
No source run artifact is changed or deleted.

The manifest records Model Book version 1.0, book ID and user name, creation
timestamp, estimator name/type and SHA-256, selected feature and target columns,
optional Sample ID metadata, a structured ordered output axis with label,
optional unit, coordinate values, and provenance, exact parameters, training mode and search level,
deterministic split settings, dataset ID and fingerprint, validation metrics
when available, test metrics, and source run identity/timestamp. Saved books can
be integrity-checked and loaded by ID or name. Names are compared
case-insensitively and an existing name is never overwritten.
An Ensemble Model Book additionally copies and verifies every valid component
artifact and records its family, exact parameters, validation RMSE, source run,
normalized weight, and the inverse-validation-RMSE weighting method. The main
Ensemble artifact contains the same components and weights so inference exactly
reproduces the saved weighted prediction.

## Model Library page

Model Library is a fixed, non-scrolling project page. It reloads
`books/index.json` whenever opened. The left Model Books panel displays five
books per page with Previous/Next arrow controls. Newest books appear first.
Each whole saved-model card is selectable and shows the Model Book name, model
type, test RMSE and R², input-to-output counts, and one of ACTIVE, SELECTED,
SAVED, or INVALID. There is no separate Open button. The header shows only the
indexed book count; active status remains on the relevant saved-book card and
selected-book detail rather than being repeated in the header and footer.

Selecting a card fills the right-side Selected Model Book panel. Its compact top
summary shows the Model Book name, active/selected status, model type, input
count, and output count. A single output is named in the prediction subtitle;
multiple outputs show a saved axis label/range only when the Model Book contains
reliable structured coordinates; otherwise they use a neutral count. RMSE, MAE,
R², and available Validation RMSE
appear as prominent metric cards. Required Inputs lists up to six feature names
inline and offers View all inputs when the list is longer.

Source run, creation time, training mode/search level, exact parameters, full
dataset fingerprint, and Model Book version are secondary and collapsed under
Model Details until View is selected. For a non-active valid model, Set as
Active validates the book, updates `books/index.json` and `project.json`, and
changes the visible status immediately. The active book uses a non-actionable
“✓ Active Model Book” badge and does not show the activation button. The selected
active book is restored after reopening the project. Selecting a book never
starts inference. The footer uses the concise state “Ready for inference” for
the active selection rather than repeating its name and active status again.

An empty project displays “No Model Books are saved in this project yet” and
directs the user to Create Model Book from a completed Training Result. If one book
is corrupted, it remains visible as INVALID with a friendly metadata/artifact
error while valid books remain usable. A malformed library index produces one
friendly library-level error without a traceback. The footer returns to
Training Results and shows only the current saved-model selection status.

## Inference page

Inference is a fixed, non-scrolling project page for one local prediction with
the active Model Book. It reloads the Model Library whenever opened, so a new
active selection is reflected immediately. The header shows “Inference.” The
compact model strip names the active book and type, so the former duplicate
green active-book badge is hidden for valid books. A warning badge remains
visible when no active Model Book is available.

The narrower left New Sample panel shows the active interface's input-to-output
counts without repeating the Model Book identity from the result header. It generates one
labeled numeric entry for every saved feature in exact feature order. Up to
eight inputs appear in a two-column grid; larger interfaces use Previous/Next
pages while all entered values remain retained. A two-choice control offers
**Replace current curve** (default) and **Add to plot**. Predict validates that
every value is present, numeric, and finite, then calls the unchanged local
inference backend. The button reads “Predicting…” and is disabled only during
that call, then returns to enabled “Predict” after success or failure.
At compact widths, a successful prediction checks whether the scientific canvas
has at least 420 usable pixels. If not, SnowBuddy closes and the workbench
redraws; the footer states “SnowBuddy closed to show plot.” This does not change
chat history.

The wide right Prediction Result workspace gives most of its area to the
reusable Scientific Plot Workbench. Model identity is compressed into one
header row; **Model Info** reveals the book ID and input/output counts. Compact
cards show the selected curve's output count, minimum, and maximum. A single
saved target remains a one-point named prediction, while multi-output
predictions are ordered curves. The plot consumes the structured output-axis
metadata saved with the active Model Book. Legacy books derive the same
deterministic metadata from target names; otherwise they use neutral ordered
output indices without invented meaning or units.

The plot toolbar offers **Explore**, **Pan**, and **Marker** modes plus zoom in,
zoom out, **Reset**, **Autoscale**, and **Plot Settings**. Mouse-wheel zoom is centered on
the pointer. Pan mode drags the viewport. Explore hover draws a crosshair and
reports curve name, X, and Y. The plot shows engineering-formatted major ticks,
major/minor grids, and a draggable multi-curve legend. X-axis tick density
reduces automatically on narrow canvases so enlarged engineering labels remain
separate; full tick density returns when more plot width is available. Marker mode places a
labeled X/Y marker at a nearby curve point or the clicked plot coordinate.
**Plot Settings** uses three compact tabs—Axes & Grid, Text & Legend, and
Selected Curve—with independent scrollable tab bodies and persistent
Cancel/Apply actions. Every control remains reachable when the dialog is
reduced to its supported 620-by-440 minimum. It edits plot title, X/Y labels, numeric X/Y limits, supported
Linear/Log scales, major/minor-grid visibility, and legend visibility/location.
It separately edits the font sizes of the title, X label, Y label, X tick values,
Y tick values, and legend, plus the legend sample-line width. It also applies
line width, Solid/Dashed/Dotted style, marker
Circle/Square/Diamond/None, and marker size only to the selected curve. Invalid
or reversed limits, nonpositive visible data on Log axes, and out-of-range
line/marker sizes are rejected. Autoscale and Reset fit all visible curves.

The compact curve manager pages long curve lists and provides a visibility
checkbox plus whole-row selection for each curve. Its selected-curve area shows
the immutable inputs used to generate that prediction. **Rename**, **Delete**,
and **Clear markers** manage the in-memory workspace. Replace updates only the
selected curve; Add creates another colored overlay. Ordinary navigation away
and back preserves curves while the project and active Model Book are unchanged.
Changing project/book or returning to Welcome clears them. Curves are not
persisted as automatic prediction history. A draggable vertical divider resizes
the plot and Curves manager with minimum usable widths on both sides. Legend
labels use the available legend width; they are no longer cut at a fixed 20
characters. Full curve names always remain available in the Curves manager.

After a successful prediction, **View Raw Values** opens the complete inputs and
predicted outputs in saved interface order. **Export Prediction** opens the
operating system save dialog for either a JSON file containing Model Book identity,
ordered input name/value records, structured output-axis metadata, output count,
and ordered target/value records, or a curve CSV containing output-axis coordinate,
predicted value, and output-variable name in saved order.
This explicit export does not create an automatic project history. Both actions
are disabled before prediction, during prediction, and after a failed result.

With no active book, the page disables Predict and directs the user to Model
Library. Missing or invalid values, backend failures, corrupted manifests,
missing artifacts, and integrity errors appear as friendly inline messages
without tracebacks. The footer returns to Model Library, offers **Inverse Design**
when an active book is available, and states that this is a single-sample
workflow with no automatic history. Curve CSV is an explicit
single-result export; there is no CSV-input or batch inference.

## Inverse Design page

Inverse Design is a fixed, non-scrolling project page that uses the active Model
Book as a fast evaluator inside a generic, single-objective Differential
Evolution search. It reloads the active book whenever opened. The compact header
shows the Model Book name, model family, and input/output counts without
duplicating its full provenance. With no active or valid book, Run Inverse Design
is disabled and the footer directs the user to select a valid Model Book.

The page uses an Inference-style persistent split workspace: Search Configuration
stays on the left and the scientific result workbench remains visible on the
right. A draggable vertical divider adjusts their widths while enforcing minimum
usable sizes. Search Configuration cannot be narrowed below 520 pixels. Its
feature labels, Variable/Fixed controls, numeric fields, objective-range fields,
and constraint controls resize with the pane; explanatory copy rewraps to the
actual width. Dragged or restored divider positions are clamped before the
result workbench is compressed below its usable size. Inside the result, the
scientific canvas retains at least 320 visible pixels and the Curves manager
cannot be narrowed below 220 pixels; both remain adjustable above those minima,
and the selected-curve input summary rewraps as that divider moves. If the window cannot
fit these engineering-control minima and a docked SnowBuddy panel at the same
time, SnowBuddy opens in its temporary focused presentation instead of squeezing
or clipping the form or plot. Inputs, Objective, and Constraints are mutually exclusive
configuration subtasks, so one compact section is visible at a time and the page
never scrolls. **Inputs** lists saved features in exact feature order, five per
page when necessary. Each row chooses Variable or Fixed. Variable rows enable
finite Lower and Upper fields; Fixed rows enable one finite Value field. The
backend requires at least one variable, lower less than upper, and every saved
feature assigned exactly once.

**Objective** has no long saved-output menu. It shows the saved output-axis label,
coordinate bounds, and point count, then accepts a numeric coordinate for
**Single point** or inclusive numeric start/end coordinates for **Mean over
range**. Mean over range evaluates the arithmetic mean of all ordered saved
outputs inside the range as one scalar objective and requires at least two
points. Minimize, Maximize, or Target value applies to that scalar.
An on-page explanation defines the objective as the one predicted scalar that
Differential Evolution improves: lowest for Minimize, highest for Maximize, or
closest to the requested number for Target value. **Constraints** allows up to
four compact rows. Each chooses Single point or Mean over range with numeric
saved-axis coordinates, then At least, At most, or Within range with the
appropriate threshold or value bounds. The visible copy explicitly identifies
constraints as optional pass/fail limits that filter eligible designs and do not
replace the objective. Without constraints, every predicted design inside the
input bounds is eligible. Range means require at least two saved points. No
antenna-specific quantity, unit, objective, bound, or constraint is invented.

The configuration footer returns to Inference, selects **Add to plot** or
**Replace selected curve**, and runs **Run Inverse Design**. While the background
search is active the action is disabled and reads “Optimizing…”, then becomes
available again after success or failure without leaving the configured screen.
The backend is final authority for the contract and uses only deterministic
SciPy Differential Evolution with seed 42. The Model Book predictor alone loads
the artifact, restores saved feature order, and predicts; the inverse-design
layer alone evaluates objective and constraint values.

The scientific plot receives most of the right side. A compact summary above it
shows achieved objective value, explicit constraint status, evaluation and
iteration counts, and exact best inputs. Unconstrained Minimize/Maximize results
are labeled **OPTIMIZED** with Constraints **Not used**. Constrained successes
are labeled **CONSTRAINTS MET**. Target searches are titled **Closest predicted
design**, labeled **CLOSEST FOUND** when unconstrained, and display Achieved and
Target Gap separately; the UI never claims an unattainable target was reached.
Each result adds a saved-order response curve by default; Replace selected curve
updates only the currently selected curve. Curves preserve their own optimized
input values and remain independently renameable, visible, hideable, and
deletable through the reusable Scientific Plot Workbench. If an added response
matches an existing plotted response within the named 0.01% tolerance, both
curves remain available and an amber warning identifies the matching prior
curve. The configuration stays available for the next search.

Every successful search creates a new `inverse_design/runs/inverse-####` folder
containing `request.json`, `result.json`, `best_prediction.csv`, and
`evaluation_trace.csv`; earlier folders are never overwritten. The project-local
index records the latest run, and reopening restores it only when it belongs to
the current active Model Book. Reopening rehydrates the complete structured
result, not only its labels and curve, so the visible outcome state, latest run,
and SnowBuddy's live project context remain consistent. Legacy Target results
without a saved target gap calculate that gap when restored. Invalid fields,
unavailable/corrupt books, and prediction failures show friendly text without a
traceback. If no candidate satisfying all output constraints is found within the
bounds and search budget, the page shows **NO CONSTRAINT MATCH**, clears stale
summary metrics, preserves older plotted curves, and creates no fake completed
result.

## Dialogs

### Raw Prediction Values

Opened from a successful Inference result. This themed, scrollable dialog shows
the active Model Book name/type followed by the exact inputs and every predicted
target in saved order. It is a read-only inspection view with a Close button.

### Export Prediction

Opened from a successful Inference result through the operating system save
dialog. The default destination is the active project's `inference/` folder and
the default format is JSON; Curve CSV is available in the same file-type chooser.
Cancel creates nothing; a write failure is summarized in a
friendly message without a traceback.

### Plot Settings

Opened from **Plot Settings** in the Scientific Plot Workbench. Three scrollable
tabs divide the controls into Axes & Grid, Text & Legend, and Selected Curve. The first has
title, X/Y labels, limits, scales, and grids; the second has font sizes, legend
visibility/location, and legend sample width; the third names the selected curve
and controls line width/style and marker style/size. Only selected-curve controls are
disabled when no curve is selected. Apply rejects blank titles/labels,
non-numeric or reversed limits, incompatible Log scales, unsupported style
values, and text or line sizes outside their documented safe ranges. Explicit
labels and limits remain in force as curves are added until
**Autoscale** or **Reset** is selected.

### Create project

Opened from the no-project hero, empty recent-project card, or File menu. It
asks for a project name and an optional description, then creates the portable
project structure.

### Open project

Uses the operating system’s folder chooser. A valid project folder contains
`project.json`.

### Input/output templates ready

Shown after Create templates. It lists the local input template, output
template, and instruction-file paths. Dismissing the dialog leaves both CSV
paths loaded into Data Prep and opens their containing folder.

### Create Model Book

Available from the Training Results footer only for a completed run. A themed
input dialog asks for the Model Book name. Success reports the assigned
`book-####` ID and confirms that the source run was not changed. Validation or
artifact failures are summarized without a traceback. Dismissing success keeps
the user on Results and changes the primary footer action to Open Model Library.

### Dataset registered

Shown after Validate and register succeeds. It reports the validated sample
count, content-based dataset ID, and local registered-dataset folder.

### Training completed

Shown after Custom completes real training and all artifacts are saved. It shows
the model, Custom mode, parameters used, and MAE, RMSE, and R². It does not show
fake metrics, charts, interpretation, or model comparison.

### Auto Search Completed

Shown after Medium or High Auto search completes, the best model is evaluated on
the held-out test set, and all artifacts are saved. It reports the search level,
configuration count, actual fold count, best parameters, validation RMSE, and
final test metrics.

### Invalid training configuration

Shown when the current Model Training values cannot produce a valid request. It
contains the clear request-validation message and no traceback. No request is
submitted after this dialog.

### Training failed

Shown when no validated dataset is active, the dataset cannot be used, fewer
than two CV folds are possible, or all Auto configurations fail. It contains a
clear local error and no raw traceback. Failed runs show no fake success metrics.

### SnowBuddy local model

- Title: “Run SnowBuddy locally.”
- Explains that Ollama keeps context and inference on the computer.
- Shows detected system RAM and the recommended profile.
- Standard card: `qwen3:8b`, approximately 5.2 GB.
- Lightweight card: `qwen3:1.7b`, approximately 1.4 GB.
- Runtime strip says whether Ollama is missing, the model needs downloading, or
  the selected model is ready locally.
- Actions: “Get Ollama,” “Download selected,” and “Use selected.”
- An installed model disables the download action and displays “Installed.”

## Important interface states

- No project: a fresh launch creates a new Welcome session and shows Start
  without automatically activating the latest project. Data Prep and Model
  Training redirect to Start with an “Open a project” message. Welcome chat can discuss Create
  project, Open project, recent projects, workflow, and local-model settings
  without fabricating project state.
- Project created: Data Prep opens immediately; workflow step 1 of 5 and the
  next action is loading and preparing antenna data. SnowBuddy switches to
  Focus mode automatically.
- Project reopened: the last active page recorded in `project.json` is restored.
- Return to Welcome: active-project context is cleared, the Start page appears,
  and SnowBuddy returns to the current launch’s Welcome session.
- Input/output pair loaded: all input and output columns are adopted,
  preparation runs automatically, and Source plus Variables complete without
  user choices.
- #Parameters source discovered: variables and one output become selectable.
- A project last saved in the retired filename-sweep mode shows a warning to
  choose an input/output CSV pair or #Parameters export; that retired option is
  not displayed.
- Data prepared: workflow step 2 of 5; the input CSV, output CSV, and schema
  paths are recorded in the project and the completed state returns after
  reopening only when both tables exist. Register is the active Data Prep stage.
- Dataset registered: validation has passed, an integrity-protected local
  snapshot is indexed, all four Data Prep subtask symbols are green, and the
  project status is “Dataset registered.” The supported Model Training run is
  now available.
- Model Training Auto: Linear Regression, Auto, and Medium are selected; Auto
  Search Level is visible and the Advanced Settings card is hidden.
- Model Training Custom: Auto Search Level is hidden and Advanced Settings is
  visible and enabled with `fit_intercept=true` and `positive=false` by default.
- Model Training XGBoost Auto: Auto Search Level is visible with Medium selected
  by default, Advanced Settings is hidden, and Custom remains selectable.
- Model Training XGBoost Custom: Advanced Settings shows the five numeric
  fields with their defaults and ranges; Auto Search Level is hidden.
- Model Training Neural Network Auto: Auto Search Level is visible with Medium
  selected by default and Advanced Settings is hidden.
- Model Training Neural Network Custom: Advanced Settings shows hidden layers,
  activation, learning rate, batch size, and epochs; Auto Search Level is hidden.
- Model Training Ensemble AI Engine: Auto High is fixed, the segmented mode
  control is disabled on Auto, editable Auto Search Level and Advanced Settings
  are hidden, and the component/validation-weighting note is visible.
- Training completed: Linear Regression, XGBoost, or Neural Network Auto Medium/High selected
  parameters by deterministic training-only cross-validation, or Custom trained
  with the selected model-family parameters, from the active registered dataset.
  The Studio saved a new
  sequential run folder without changing older runs, updated the Latest Run
  readout, stored real parameters and metrics, and moved the project workflow to
  `model_trained` at step 3 of 5.
- Ensemble training completed: each successful Auto High component has its own
  immutable run, the Ensemble is the latest run, weights sum to one and come
  only from inverse validation RMSE, failed components are visible, and the
  recommendation compares Ensemble validation RMSE with the best individual.
- Training Results available: the latest completed run is loaded from its
  immutable artifact folder and Predictions is selected as the primary view.
  Earlier run folders remain unchanged. A secondary Results section selected for
  another project or earlier run is not reused.
- Model Book saved from Results: the source run remains unchanged, the saved
  book is not silently activated, and the primary action changes to Open Model
  Library. The library opens with that new book selected. Set as Active remains
  an explicit user decision; Resume targets Model Library until it is made.
- Custom Results without a compatible Auto run: no suggestion is invented; the
  page instructs the user to run Auto for evidence-based guidance.
- Custom Results with a compatible Auto run: comparison is allowed only after
  fingerprint, columns, split, and model-family compatibility pass.
- Training Results artifact error: no partial metrics or plots are displayed;
  the page shows a friendly local error without a traceback.
- Invalid training configuration: the training backend was not called and the page
  shows the validation message without a traceback.
- Inverse Design configured: every required feature is explicitly Variable with
  finite bounds or Fixed with one finite value; the scalar objective is either
  one saved-axis coordinate or the mean over an inclusive saved-axis range; a
  goal is selected; optional generic output constraints are visible.
- Inverse Design completed: the latest immutable inverse run is selected, the
  persistent workspace distinguishes Optimized, Constraints Met, and Closest
  Found, shows the saved-order response curve, and enables Run Inverse Design
  again for another independent search. Target results show the numeric target
  gap; effectively duplicate plotted responses carry an explicit warning.
- Inverse Design infeasible or invalid: no completed result is fabricated or
  saved, the clear reason is shown without a traceback, and Configure remains
  editable.
- Model unavailable: SnowBuddy answers with the built-in project guide and
  points to Local model settings.

## Live snapshot vocabulary

At question time SnowBuddy may receive:

- Visible page: Start, Data Prep, Model Training, Training Results, Model Library,
  Inference, or Inverse Design.
- Appearance mode: Light or Dark.
- Top application menu: File, Edit, Help.
- Active project and workflow status.
- On every page: SnowBuddy companion visibility, chat enabled state, and
  SnowBuddy mode (Welcome or Focus).
- On Start: recent-project count.
- On Data Prep: source mode; input and output CSV paths or the #Parameters
  source path; the LHS sample-generator and Create templates actions; whether a
  generated input CSV is waiting for solver outputs; active accordion subtask;
  load or parse status; discovered
  sample/input/output counts; selected inputs; output contract; whether the raw
  variable contract has been explicitly saved; preparation
  status; whether the separate prepared file pair is ready; visible validation
  status and details; registered dataset ID; and whether Validate and register
  is enabled. The snapshot explicitly says that this action does not trigger
  training. It also reports that page scrolling is absent and identifies the
  fixed bottom navigation controls.
- On Model Training: selected model; Auto or Custom training mode; selected
  search level; whether Auto Search Level is available; whether Advanced
  Settings is enabled; `fit_intercept` and `positive` values; whether a
  validated request is retained; the last structured training-result status;
  whether training is currently in progress; the persisted “Latest Run” value;
  and that Train Model runs Linear Regression with deterministic Auto Medium or
  High search or selected Custom parameters; XGBoost with deterministic Auto
  Medium/High search or five validated Custom values; or Neural Network with
  deterministic standardized Auto search or five validated Custom values, from
  the active dataset; Ensemble AI Engine reports fixed Auto High component
  training and validation-derived weighting.
  It also reports that page scrolling is absent.
- On Training Results: displayed run ID; Auto or Custom mode; active results
  section; selected prediction sample; prediction or residual workbench title,
  X/Y labels, curve state, and Plot Settings; whether the primary action is
  Create Model Book or Open Model Library; full
  dataset fingerprint; whether a comparable Auto recommendation is available;
  Ensemble component weights/failures and validation recommendation when applicable;
  empty/failure/artifact-error state; and that the ordered page has no page
  scrolling.
- On Inference: active Model Book name and ID; exact required numeric inputs;
  saved output count; latest prediction status; and that the page supports one
  unsaved sample with no batch or CSV inference.
- On Inverse Design: active Model Book name and ID; deterministic optimizer;
  variable and fixed input names; objective scope, selected coordinate or range,
  and goal; constraint count and each point/range-mean scope; visible configuration
  subtask; latest result status; Add-to-plot or Replace-selected action; plotted-
  curve count; adjustable configuration/result and plot/Curves dividers; and
  immutable-run behavior.

Live snapshot values override defaults in this document.

## Maintenance rule

Every change to `studio/ui.py`, `studio/results_ui.py`, `studio/library_ui.py`,
`studio/inference_ui.py`, `studio/inverse_design_ui.py`, or `studio/theme.py` must
include a review of this file in the same change. Update affected descriptions,
Studio/contract versions when appropriate, and all GUI SHA-256 values above.
Automated tests reject stale hashes.
