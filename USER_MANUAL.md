# Antenna Surrogate Studio User Manual

This manual explains how to operate Antenna Surrogate Studio from project setup
through prediction and inverse design. It focuses on the controls you use and
the results you see. It does not describe the Studio's internal implementation.

## 1. Launching the Studio

On Windows, double-click **Start Antenna Surrogate Studio.bat**. On macOS or
Linux, open a terminal in the application folder and run:

```bash
bash start_studio.sh
```

The Start page opens in Welcome mode. From here you can:

- **Create Project** for a new study.
- **Open Project** to choose an existing project folder.
- Select one of the five recent-project cards.
- Open SnowBuddy before or after selecting a project.

When an existing project opens, the Studio returns to the last main page used
in that project.

## 2. Understanding the workflow

The left workflow bar follows this order:

1. Start
2. Data Prep
3. Model Training
4. Training Results
5. Model Library
6. Inference
7. Inverse Design

The sidebar can collapse to an icon-only rail. Hover over a compact icon to see
its page name. The sidebar, plot divider, and SnowBuddy panel do not change your
project data.

## 3. Creating or opening a project

### Create a project

1. Select **Create Project**.
2. Enter a clear project name.
3. Add a description if useful.
4. Confirm the project.

The Studio creates the project and opens Data Prep.

### Open a project

Use **Open Project** and choose either the project folder or its `project.json`
file. A valid project contains all data, training runs, Model Books, inference
runs, inverse-design runs, and project-specific SnowBuddy history.

Use **File > Return to Welcome** when you want to close the active project
without closing the Studio.

## 4. Preparing data

Data Prep is divided into connected subtasks. Only one subtask is expanded at a
time, and each header shows its current status.

### Option A: Input and output CSV files

Use this option when you already have separate input and output tables.

Input example:

```text
Sample ID,P2,P3,P4
Design_001,1.2,3.5,0.8
Design_002,1.4,3.1,0.9
```

Output example:

```text
Sample ID,Gain at Theta -90 deg,Gain at Theta 0 deg,Gain at Theta 90 deg
Design_001,-12.5,4.8,-11.9
Design_002,-11.7,5.2,-12.2
```

Rules:

- Input and output files must have the same number of data rows.
- All model input and output cells must contain finite numeric values.
- `Sample ID` is optional. If used, it must appear in both files with unique,
  matching IDs in the same row order.
- Input column names should match the parameter names used in your simulator.
- Output headers should include a coordinate and unit only when they are known.

Load both files. The Studio identifies their input and output columns without a
separate Parse step.

### Option B: Parameter-sweep export

Use this option for supported solver exports containing `#Parameters` blocks.

1. Select **#Parameters sweep**.
2. Select **Browse file** or **Browse folder**.
3. Select **Parse**.
4. Choose the parameters that should become model inputs.
5. Choose the response to use as model output.
6. Select **Save selection**.
7. Select **Prepare input + output**.

The coordinate found in the solver table is preserved in the prepared output
names. This may be frequency, theta, phi, or another solver-defined coordinate.

### Option C: Generate simulation inputs with LHS

Use the LHS Sample Generator when you need a set of parameter combinations to
simulate.

1. Open **LHS sample generator** from the Source subtask.
2. Add one row for each simulation variable.
3. Enter each variable name, minimum, and maximum.
4. Enter the sample count.
5. Optionally enter a whole-number seed for repeatable samples.
6. Select **Generate Samples**.
7. Review the table and coverage preview.
8. Select **Export inputs.csv**.

The generated file contains only the user-defined input columns. Run those rows
in your simulator, then return with an output CSV in the exact same row order.

### Validate and register

After preparation, select **Validate and register**. Training remains unavailable
until the data passes this step.

The result panel shows the registered dataset ID and its sample, input, and
output counts. If validation fails, correct the named issue and run the action
again. Registration does not start model training.

## 5. Training a model

Open **Model Training** after registering the dataset.

### Choose a model

The available choices are:

- Linear Regression
- XGBoost
- Neural Network
- Ensemble AI Engine

### Choose a training mode

- **Auto Medium** is the recommended starting point. It is quicker and uses
  less compute.
- **Auto High** explores more options and can take longer.
- **Custom** exposes the settings supported by the selected model.

The Ensemble AI Engine uses its preset automatic workflow and does not require
custom settings.

Select **Train Model** when the configuration is ready. The button changes to
**Training…** and is temporarily disabled. You can continue when it becomes
available again.

Each click creates a new run. Earlier run folders and results are preserved.

## 6. Reading Training Results

Training Results opens the latest completed run by default.

### Main metrics

- **R²** indicates how much test-data variation is represented by the model.
  Higher is generally better, but interpret it together with the error values.
- **RMSE** emphasizes larger prediction errors. Lower is better.
- **MAE** is the average absolute prediction error. Lower is better.
- **Validation RMSE** is the evidence used when the Studio recommends a
  training configuration or model family. Lower is better.

Units appear only when they are available from the saved output information.

### Prediction and residual views

- **Predictions** overlays Actual and Predicted response curves for the selected
  test sample.
- **Residuals** shows the difference between Actual and Predicted values.
- The Curves area selects the test sample and displays its associated inputs.
- Plot Settings controls labels, limits, grids, legend, text sizes, and curve
  appearance.

### Model comparison

Open the comparison section to compare compatible completed model families. The
recommended model is identified from validation evidence. Runs made from a
different dataset or incompatible setup are not mixed into the recommendation.

### Next actions

- Select **Adjust & Train Again** to return to Model Training.
- Select **Create Model Book** when you want to preserve the completed model for
  future use.

## 7. Saving a Model Book

A Model Book is the reusable form of a completed model.

1. Open a successful Training Results run.
2. Select **Create Model Book**.
3. Enter a unique, descriptive name.
4. Confirm the action.

The source training run remains unchanged. Existing Model Books are not
overwritten when a duplicate name is entered.

## 8. Using Model Library

Model Library lists the Model Books saved in the current project.

Select a Model Book card to inspect:

- Model family
- Required inputs
- Output count
- Main performance metrics
- Training configuration
- Source-run and project details

Select **Set as Active** for the Model Book you want to use. The active selection
is preserved when the project reopens. Inference and Inverse Design always use
the active Model Book.

## 9. Running inference

Inference predicts one new input sample at a time.

1. Confirm the intended Model Book is active.
2. Open **Inference**.
3. Enter a numeric value for every required input.
4. Choose **Replace current curve** or **Add to plot**.
5. Select **Predict**.

The result area shows the exact inputs used, output count, minimum, maximum, and
the ordered prediction curve. Every successful prediction is saved as a
separate project run. Reopening the project restores all valid prediction curves
for the active Model Book.

Use:

- **View Raw Values** to inspect every predicted output in saved order.
- **Export Prediction** to save the selected result as JSON or curve CSV.
- **Open Inverse Design** to continue with the same active Model Book.

When several curves are present, select a curve in the Curves area before using
View Raw Values or Export Prediction.

## 10. Running inverse design

Inverse Design searches for input values that satisfy one design goal.

### Configure inputs

For every required model input, choose:

- **Variable** and enter its lower and upper bounds; or
- **Fixed** and enter the value that must remain unchanged.

At least one input must be variable.

### Configure the objective

The objective is the one output quantity you want the search to improve.

- **Single point** selects one saved output coordinate.
- **Mean over range** uses the average response across the inclusive coordinate
  range you enter.
- **Minimize** searches for the lowest objective value.
- **Maximize** searches for the highest objective value.
- **Target value** searches for the result closest to the requested value.

For example, to reduce the response at theta 0, choose the coordinate `0` and
Minimize. To improve the average response from theta -30 through 30, choose Mean
over range, enter `-30` and `30`, and choose Maximize.

### Add optional constraints

Constraints are additional conditions a design must satisfy. Each constraint
can apply to one coordinate or a mean over a coordinate range.

Available conditions are:

- At least a value
- At most a value
- Within a range

If no candidate satisfies every constraint within the configured bounds and
search budget, the Studio reports **No Constraint Match**. It does not present
an infeasible design as a successful result.

### Run and compare searches

1. Choose **Add to plot** to keep existing curves, or **Replace selected curve**.
2. Select **Run Inverse Design**.
3. Review the achieved objective, constraint status, best inputs, and response
   curve.

Each successful search is saved as a separate run. Reopening the page restores
all valid inverse-design curves for the active Model Book. Select any curve in
the Curves area to see its objective, achieved value, constraints, and input
settings.

## 11. Scientific Plot Workbench

The plot workbench is shared across prediction and engineering result views.

You can:

- Zoom with the mouse wheel or plot controls.
- Pan the visible region.
- Reset or autoscale the axes.
- Show major and minor grids.
- Hover to read the curve name and X/Y values.
- Move or hide the legend.
- Add and clear markers.
- Show, hide, rename, select, or delete curves.
- Edit the title, X/Y labels, limits, linear/log scale, fonts, line width, line
  style, and marker style in **Plot Settings**.

Plot labels remain neutral when output meaning or units were not supplied in the
saved data. You may edit them for presentation without changing prediction
values.

## 12. SnowBuddy

SnowBuddy is available from the top application bar throughout the workflow.

- In Welcome mode, it provides general setup and navigation guidance.
- Inside a project, it uses that project's current workflow state and separate
  local chat history.
- Closing the panel does not erase the conversation.
- Without Ollama, its built-in project guide remains available.
- With Ollama, the selected Qwen model runs locally on your computer.

SnowBuddy can explain the current page, identify the next valid step, discuss
saved results, and help recover from visible validation errors. Project data and
chat are not sent to a paid cloud API.

## 13. Saving, reopening, and moving projects

The Studio saves project state and successful run artifacts inside the project
folder. To continue on another computer:

1. Close the Studio after the current action finishes.
2. Copy the complete project folder.
3. Install Antenna Surrogate Studio on the other computer.
4. Open the copied project folder from the Start page.

The active page, registered data, training runs, Model Books, active Model Book,
saved prediction curves, inverse-design curves, and project SnowBuddy history
travel with the complete project folder.

## 14. Common operating issues

### Training is unavailable

Return to Data Prep and confirm that the dataset shows a successful registered
state.

### Predict is unavailable

Open Model Library and set a valid Model Book as active.

### A project does not open

Choose the complete project folder or its `project.json`. Do not select an
individual model artifact.

### Input or output CSV validation fails

Check row counts, numeric cells, column names, and—when present—matching unique
Sample IDs.

### A prediction or inverse curve is missing after reopen

Confirm that the same Model Book is active. Histories are intentionally filtered
to the active book. A damaged run is skipped with a message while other valid
runs remain available.

### The application appears busy

Wait for the active button to return from **Training…** or **Optimizing…**. Large
datasets and higher-compute modes can take longer on laptops.

## Support

- Author: **Sai Sampreeth Indharapu**
- Email: [sampreethsharma@gmail.com](mailto:sampreethsharma@gmail.com)
- LinkedIn: [Sai Sampreeth Indharapu, Ph.D.](https://www.linkedin.com/in/sai-sampreeth-indharapu-ph-d-a98802110/)
