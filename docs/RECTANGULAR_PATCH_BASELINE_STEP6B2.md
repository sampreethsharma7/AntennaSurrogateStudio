# Rectangular patch analytical baseline — Stage 3 Step 6B-2

`engineering.rectangular_patch_baseline` is a read-only analysis tool. It reads the current canonical frequency, substrate thickness, substrate material relative permittivity, and per-element patch dimensions. Its planner argument schema is intentionally empty.

For frequency `f` in GHz, speed of light `c = 299.792458 mm·GHz`, substrate relative permittivity `epsilon_r`, and substrate height `h` in mm, the tool evaluates:

```text
W = c / (2 f) sqrt(2 / (epsilon_r + 1))

epsilon_eff = (epsilon_r + 1)/2
            + (epsilon_r - 1)/(2 sqrt(1 + 12 h/W))

delta_L = 0.412 h ((epsilon_eff + 0.3)(W/h + 0.264))
                    / ((epsilon_eff - 0.258)(W/h + 0.8))

L = c / (2 f sqrt(epsilon_eff)) - 2 delta_L
```

The structured result includes the inputs, estimated width and length, current canonical width and length, signed current-minus-estimate differences, percent differences, effective permittivity, and fringing extension. Every measurement has an explicit unit.

Applicability is limited to the supported rectangular microstrip-patch family. Arrays are evaluated per element. Canonical composition plus evaluated Boolean provenance marks a radiator with slots, unions, or other composed geometry as a modified topology. The calculation still returns an unmodified rectangular-patch reference, with an explicit reference-only limitation. Missing or ambiguous substrate permittivity returns `unknown`; unsupported antenna families return `not_applicable`.

For the current default 2.45 GHz, epsilon_r 4.4, 1.6 mm substrate design, deterministic evaluation gives:

```text
estimated width     37.2342611829 mm
estimated length    28.8092902619 mm
epsilon_eff          4.08085752155
delta_L              0.738598557308 mm
```

The calculation does not establish resonance, impedance match, gain, bandwidth, efficiency, polarization, or full-wave validity and does not recommend or apply geometry changes.

The controlled live Gemini and OpenRouter Nemotron runs both selected the tool for plain and center-slotted designs without schema repair. Both returned the same deterministic result. Both preserved the modified-topology limitation in the slotted case. Gemini preserved full-wave caveats in both cases. Nemotron's plain-design final prose omitted the structured limitations and called the design very well sized under the conventional model; that is model wording rather than an additional tool finding. No provider-specific behavior was added in response.
