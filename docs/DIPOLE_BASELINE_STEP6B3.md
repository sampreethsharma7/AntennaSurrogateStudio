# Dipole electrical-length baseline — Stage 3 Step 6B-3

`engineering.dipole_baseline` is a read-only analysis tool with an empty planner argument schema. It reads frequency from the canonical design and obtains dipole dimensions from evaluated physical geometry.

For one canonical array element, the tool selects the two physical objects with semantic role `radiating_arm`. Each arm must be a resolved canonical cylinder. It constructs the cylinder's local centerline endpoints from its resolved axis, center, start, and end values, then applies the canonical rigid transform. It derives:

- each arm length from the transformed endpoint distance;
- feed gap from the minimum distance between endpoints on different arms;
- total conductor length from the sum of both arm centerline lengths;
- total dipole length from both arm lengths plus the feed gap;
- tip-to-tip span from the greatest distance among all four endpoints;
- arm radii and diameters from resolved cylinder geometry.

This extraction is invariant under whole-element translation and rotation and preserves unequal arm lengths. It does not infer physical dimensions from parameter names.

Using `c = 299.792458 mm·GHz`, the analytical references are:

```text
lambda0 = c / f
half-wave total-length reference = lambda0 / 2
quarter-wave arm reference = lambda0 / 4
```

The half-wave and quarter-wave quantities are references only. No empirical shortening-factor correction is applied. The tool does not predict exact resonance, input impedance, bandwidth, efficiency, gain, pattern validity, balun behavior, or full-wave performance.

For the current default 2.45 GHz dipole, deterministic evaluation gives:

```text
lambda0                         122.3642685714 mm
total dipole length              58.1230 mm     = 0.4749998 lambda0
total conductor length           56.8994 mm     = 0.4650001 lambda0
arm 1 / arm 2 length             28.4497 mm     = 0.2325001 lambda0 each
half-wave reference              61.1821342857 mm
difference from half-wave        -3.0591342857 mm (-5.000045%)
feed gap                          1.2236 mm      = 0.0099997 lambda0
conductor radius                  0.4079 mm      = 0.0033335 lambda0
```

Any evaluated Boolean history, multiple source primitives, or persisted dipole geometry/Boolean composition marks applicability as `Approximate modified topology` and makes the simple baseline reference-only. Missing frequency or unresolved/two-arm geometry returns `unknown`. Other antenna families return `not_applicable`.

In controlled live tests, Gemini selected the tool for both requested prompts with no schema repair and preserved the full-wave limitations. In its half-wave response it added that the 5% reduction is typical practical thin-wire shortening; that is additional model interpretation, not a tool measurement. Nemotron completed the first prompt with accurate measurements and limitations. For the second prompt it executed the analysis correctly, then the free OpenRouter endpoint returned no structured follow-up plan; the runner reported a provider error after the valid result. No provider-specific changes were made.
