# Four-Element Patch-Array Phase-Sweep Sample

![Four-element microstrip patch antenna array](PatchAntennaArray.png)

This ready-to-use example helps you complete your first Antenna Surrogate
Studio project with a real parameter-sweep export.

## Dataset at a glance

| Item | Included data |
| --- | --- |
| Antenna | Four-element microstrip patch array simulated in CST Microwave Studio |
| Samples | 1,000 phase configurations, split across two raw export files in `data/` |
| Model inputs | `P2`, `P3`, `P4` |
| Fixed reference | `P1 = 0 deg` |
| Input range | Approximately `-180 deg` to `180 deg` |
| Model output | `Gain,Phi=0.0 []` |
| Output coordinates | Theta `-180 deg` to `180 deg` in `1 deg` steps |
| Outputs per sample | 361 |

Latin Hypercube Sampling was used to distribute the phase combinations across
the input range. The original dataset description is available in
[`dataset_description.txt`](dataset_description.txt).

## Folder layout

```text
four_element_patch_array_phase_sweep/
├── data/                         Raw files selected in Data Prep
│   ├── parasweep_phases_output.txt
│   └── parasweep_phases_batch2_output.txt
├── PatchAntennaArray.png         Reference image
├── dataset_description.txt      Dataset notes
└── README.md                     This guide
```

Keeping raw exports in their own data folder prevents documentation and images
from being mixed with simulation files. The same practice is recommended for
future projects.

## Load the sample in the Studio

1. Launch Antenna Surrogate Studio and select **Create project**. Give the
   project a name and select **Create project** again.
2. The Studio opens **Antenna Design**. Select **Continue to Data Prep**.
3. In **Select data source**, choose **#Parameters sweep**.
4. Select **Browse folder** and choose the sample's
   `four_element_patch_array_phase_sweep/data` folder.
5. Select **Parse**. The Studio should report **1,000 samples** and list about
   24 parameters.
6. Under **Model inputs**, select only `P2`, `P3`, and `P4`.
7. Under **Pattern output**, select `Gain,Phi=0.0 []`.
8. Select **Save selection**, followed by **Prepare input + output**.
9. In **Validate and register**, select **Validate and register**.
10. Continue to **Model Training**.

Select the whole `data` folder rather than one text file. Each raw export
contains 500 samples, and the Studio combines both files into the full
1,000-sample dataset.

The parser also discovers about 20 fixed CST settings alongside the phases. Do
not select those as model inputs for this example. `P1` is the fixed phase
reference, so the three varying phases are the appropriate model inputs.

## Suggested first training run

For a quick first experience:

1. Choose **XGBoost**.
2. Keep **Auto** mode and **Medium** search level.
3. Select **Train Model**. On a typical laptop this takes about half a minute.
4. Review the predicted and actual radiation-pattern curves in
   **Training Results**.
5. Select **Create Model Book**, then in **Model Library** select the book and
   choose **Set as Active**. Try a new phase combination in **Inference**, or
   search for the phases that maximise gain at a chosen angle in
   **Inverse Design**.

What to expect on this sample (approximate, from an Auto / Medium run with
the default 80/20 split):

| Model | Training time | Mean per-output test R² | Pooled R² |
| --- | --- | --- | --- |
| Neural Network | about 40 s | about 0.75 | about 0.89 |
| XGBoost | about 30 s | about 0.66 | about 0.86 |
| Linear Regression | under 1 s | about 0.10 | about 0.63 |

Linear Regression is useful only as a baseline here. The pattern depends on the
phases periodically (−180° and +180° are the same phase), which a linear model
cannot represent, so its predicted curves are visibly poor and inverse design
with it tends to push every phase to the edge of its range.

The pooled R² is always higher than the mean per-output R² on this kind of data,
because it also credits the model for reproducing the overall pattern shape
that every sample shares. Compare models on the per-output figure and the RMSE.

This sample is provided to demonstrate the workflow. It is not intended as a
universal antenna benchmark or a claim about performance on other designs.
