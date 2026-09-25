# Stage 3 Step 6B-1: array electrical-spacing analysis

Implemented in the experimental text-parametric-builder worktree. This change adds one registered read-only tool, `engineering.array_spacing` version `1`, over the Stage 6A analysis runtime. It does not change geometry, recipes, checks, feeds, ports, CST, VTK, GUI, ProjectMemory, or provider-specific behavior. No patch, circular-patch, or dipole sizing formula was added.

## Tool contract

```json
{
  "name": "engineering.array_spacing",
  "effect": "analysis",
  "tool_version": "1",
  "arguments": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "scan_angle_deg": {"type": "number", "minimum": -90, "maximum": 90},
      "principal_axis": {"type": "string", "enum": ["all_active", "row", "column"]}
    }
  }
}
```

Both arguments are optional. With no scan angle, the tool reports spacing only and explicitly withholds a grating-lobe conclusion. A supplied scan angle enables the ideal visible-order calculation. `all_active` evaluates each active canonical principal axis; `row` or `column` selects one. Scan ranges are not in the schema and are rejected rather than approximated.

The tool uses `AntennaDesign.array.rows`, `.columns`, and `.spacing_mm`, plus the canonical `frequency_ghz` parameter. It does not inspect ports, preview coordinates, or inferred element geometry. The current canonical model has one physical spacing shared by row and column replication, so the same spacing is reported independently for each active axis.

## Calculations

The defined constant is:

```text
c = 299792458 m/s = 299.792458 mm·GHz
```

Free-space wavelength and electrical spacing are:

```text
lambda0_mm = 299.792458 / frequency_GHz
spacing_lambda0 = spacing_mm / lambda0_mm
```

For a supplied single principal-plane scan angle, visible integer orders use the declared convention:

```text
sin(theta_m) = sin(theta_0) + m * lambda0 / d
```

Every nonzero integer `m` satisfying `|sin(theta_m)| <= 1` is returned as a structured count and angle measurement. The integer search bounds are derived from the inequality rather than an arbitrary fixed search range.

This is an ideal periodic principal-plane calculation. Results state that they exclude finite-array effects, element-pattern suppression, mutual coupling, feed-network amplitude/phase/fabrication errors, substrate and surface-wave effects, and full-wave behavior.

No unconditional `d > 0.5 lambda0` rule is used. In deterministic tests, 0.5 lambda0 at 45 degrees has no visible nonzero order, while 0.8 lambda0 at 45 degrees has `m=-1` at approximately -32.91 degrees.

## Applicability

- A 1x1 design returns `not_applicable`, with row/column counts and no invented spacing.
- A 1xN design reports only the column axis and marks the row axis not applicable.
- An Mx1 design reports only the row axis and marks the column axis not applicable.
- An MxN design reports both axes.
- Missing, non-finite, or non-positive frequency/active-axis spacing returns `unknown` with a specific applicability explanation.
- Requesting an inactive scan axis returns `not_applicable`.
- Supplying `principal_axis` without a scan angle returns `unknown`; the tool does not invent the missing scan assumption.

## Structured result example

For the controlled 2x3 inset-patch array at 2.45 GHz and 0.6 lambda0 spacing:

```text
status: completed
rows: 2 count
columns: 3 count
frequency: 2.45 GHz
lambda0: 122.3642685714 mm
row_spacing: 73.4185611429 mm
row_spacing_lambda: 0.6 lambda0
column_spacing: 73.4185611429 mm
column_spacing_lambda: 0.6 lambda0
```

With a 45-degree scan applied to all active axes, the result additionally contains:

```text
row_visible_nonzero_order_count: 1
row_visible_order_m_neg_1_angle: -73.6499769699 deg
column_visible_nonzero_order_count: 1
column_visible_order_m_neg_1_angle: -73.6499769699 deg
```

The call never changes the design revision or semantic hash. Cache identity remains tool/version + normalized arguments + semantic design hash. Numerically equivalent JSON arguments such as `45` and `45.0` now normalize to the same analysis ID and cache key. Frequency, spacing, or layout changes alter the semantic design hash and naturally invalidate the cache.

## Live Gemini and Nemotron results

The controlled design was the same 2x3 inset-fed patch array at 2.45 GHz with 0.6 lambda0 spacing. All four live runs finished in two decisions: one accepted analysis call followed by finish. There were no schema repairs, runner rejections, retries, provider errors, or design mutations.

### “Check the electrical spacing of this array.”

- Gemini called `engineering.array_spacing` with `{}`. It received the spacing-only result, including the explicit statement that grating-lobe behavior was not classified without a scan assumption. Its final response reported the numerical values correctly but then introduced a broadside interpretation and said 0.6 lambda0 avoids visible grating lobes. That conclusion is true for an ideal broadside periodic array but exceeded the no-scan result supplied by the tool. No prompt adjustment was added.
- Nemotron called the tool with `scan_angle_deg=0` and `principal_axis=all_active`, introducing an explicit broadside assumption that the user had not supplied. The resulting calculation and its final explanation were internally correct and included the ideal-model limitations. This remains a model planning choice, not deterministic behavior.

### “Can I scan this array to 45 degrees without grating lobes?”

- Gemini called the tool with `scan_angle_deg=45.0`, `principal_axis=all_active`. It correctly reported one visible `m=-1` order at -73.65 degrees on both axes and disclosed the ideal-periodic limitations.
- Nemotron called the same tool with numerically equivalent arguments (`45` instead of `45.0`). It reported the same orders and angles and disclosed the limitations. It also derived the familiar maximum-spacing boundary of about 0.586 lambda0 and suggested reducing spacing; that recommendation came from the model, not the analysis result.

Exact provider trajectories and structured results are retained in ignored local artifacts:

- `.test_runs/array_spacing_live_gemini_spacing.json`
- `.test_runs/array_spacing_live_nemotron_spacing.json`
- `.test_runs/array_spacing_live_gemini_scan45.json`
- `.test_runs/array_spacing_live_nemotron_scan45.json`

The reproduction harness is `.test_runs/array_spacing_live.py` and contains no credentials.

## Validation

Focused analysis/planner/runner/integration suite: **167 passed** in 6.127 seconds. Syntax compilation and whitespace checks passed.

Full repository suite: **729 tests in 95.874 seconds: 725 passed, 2 skipped, 2 failed**. The failures are the existing VTK preview availability failures in `test_antenna_builder_page`: `test_circular_patch_and_dipole_rebuild_dynamic_controls_and_preview` and `test_preview_rotation_zoom_and_laptop_footer_remain_reachable`. This task did not modify VTK, preview, or GUI code.

No commit or merge was performed.
