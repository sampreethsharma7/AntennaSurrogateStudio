# Engineering analysis reference

The Studio's antenna builder exposes a set of **read-only analysis tools** and the
**evaluated geometry query** they are built on. None of them changes a design.
They create no engineering findings, call no planner, write no project files, and
return no recommendation to alter geometry.

This document covers the two analytical baselines in full — their equations,
inputs, assumptions and limits — and then the geometry query that supplies their
measurements.

> **None of this is electromagnetic validation.** Every tool here returns
> first-order analytical references and measurements of the geometry you already
> have. Matching a reference does not establish resonance, impedance match, gain,
> bandwidth, efficiency, polarization, or pattern validity. Use a full-wave solver
> for that.

## The registered analysis tools

Each is version `1`, requires an existing design, takes no arguments except where
noted, and is exposed to the natural-language planner.

| Tool | What it returns |
| --- | --- |
| `engineering.design_summary` | The canonical design inventory, unmodified. |
| `engineering.array_spacing` | Canonical row/column spacing in free-space wavelengths, and optionally the visible integer spatial orders for one scan angle in ideal periodic principal planes. Takes `scan_angle_deg` (-90 to 90) and `principal_axis` (`all_active`, `row`, `column`). It models no finite-array, element-pattern, coupling, feed-network, substrate, surface-wave or full-wave effect. |
| `engineering.rectangular_patch_baseline` | Conventional first-order patch dimensions compared with one current radiating element. |
| `engineering.dipole_baseline` | Measured dipole arm centerlines, feed gap and conductor thickness compared with free-space lambda0/2 and lambda0/4. |

Every result carries `tool_name`, `tool_version`, a content-derived `analysis_id`,
the design reference it was computed against, the semantic design hash, a
`status`, the `measurements` (each with an explicit unit), plus `assumptions`,
`applicability`, `limitations` and a `message`.

### Status semantics

- `completed` — the analysis ran against resolved geometry.
- `unknown` — a required input was missing or ambiguous, such as an absent
  substrate permittivity or unresolved geometry. No number is invented.
- `not_applicable` — the design is not in the family the tool analyses.

Applicability is reported separately from status. A radiator carrying slots,
unions or other composed geometry still gets a result, but it is marked as a
modified topology and the baseline becomes **reference-only**.

## Rectangular patch analytical baseline

`engineering.rectangular_patch_baseline` reads the canonical frequency, substrate
thickness, substrate relative permittivity, and the per-element patch dimensions.

For frequency `f` in GHz, `c = 299.792458 mm·GHz`, relative permittivity
`epsilon_r`, and substrate height `h` in mm:

```text
W = c / (2 f) sqrt(2 / (epsilon_r + 1))

epsilon_eff = (epsilon_r + 1)/2
            + (epsilon_r - 1)/(2 sqrt(1 + 12 h/W))

delta_L = 0.412 h ((epsilon_eff + 0.3)(W/h + 0.264))
                    / ((epsilon_eff - 0.258)(W/h + 0.8))

L = c / (2 f sqrt(epsilon_eff)) - 2 delta_L
```

`W` is the conventional width choice that places the patch near a practical
aspect ratio; `epsilon_eff` is the standard microstrip effective permittivity;
`delta_L` is the fringing extension at each radiating edge, which is why the
physical length comes out shorter than a half guided wavelength.

The result reports the inputs, the estimated width and length, the current
canonical width and length, the signed current-minus-estimate differences and
their percent equivalents, `effective_dielectric_constant`, and
`fringing_extension`.

For the default 2.45 GHz, `epsilon_r` 4.4, 1.6 mm substrate design:

```text
estimated_patch_width           37.234261182884374 mm
estimated_patch_length          28.809290261854393 mm
effective_dielectric_constant    4.080857521554887
fringing_extension               0.7385985573076748 mm
```

Arrays are evaluated per element; the result describes one radiating element.

**Limitations as the tool reports them.** Slots, loading, finite-ground effects
and feed perturbations are not modeled. Array mutual coupling is not modeled. The
calculation does not estimate impedance match, gain, bandwidth, efficiency or
polarization. No full-wave simulation is performed, and matching the estimate does
not guarantee resonance at the reference frequency.

## Dipole electrical-length baseline

`engineering.dipole_baseline` reads frequency from the canonical design and takes
its dimensions from **evaluated physical geometry**, not from parameter names.

For one canonical array element it selects the two physical objects with semantic
role `radiating_arm`. Each must be a resolved canonical cylinder. It builds the
cylinder's local centerline endpoints from the resolved axis, center, start and
end values, applies the canonical rigid transform, and derives:

- each arm length, from the transformed endpoint distance;
- feed gap, from the minimum distance between endpoints on different arms;
- total conductor length, from the sum of both arm centerlines;
- total dipole length, from both arm lengths plus the feed gap;
- tip-to-tip span, from the greatest distance among all four endpoints;
- arm radii and diameters, from the resolved cylinder geometry.

This extraction is invariant under whole-element translation and rotation, and it
preserves unequal arm lengths rather than averaging them.

With `c = 299.792458 mm·GHz` the analytical references are:

```text
lambda0                          = c / f
half-wave total-length reference = lambda0 / 2
quarter-wave arm reference       = lambda0 / 4
```

These are **references only**. No empirical thin-wire shortening factor is
applied, so a physical dipole sized for resonance sits a few percent below the
half-wave reference; that offset is expected and is not a defect.

For the default 2.45 GHz dipole:

```text
lambda0                                      122.36426857142857 mm
current_total_dipole_length                   58.123  mm  = 0.4749998 lambda0
current_total_conductor_length                56.8994 mm  = 0.4650001 lambda0
current_arm_1_length / current_arm_2_length   28.4497 mm  = 0.2325001 lambda0 each
half_wave_length_reference                    61.182134285714284 mm
quarter_wave_arm_reference                    30.591067142857142 mm
total_length_difference_from_half_wave        -3.0591342857142862 mm  (-5.000045 %)
current_feed_gap                               1.2236 mm  = 0.0099997 lambda0
current_conductor_radius                       0.4079 mm  = 0.0033335 lambda0
```

Each length is also reported as a lambda0 ratio. Any evaluated Boolean history,
multiple source primitives, or persisted dipole composition marks applicability as
an approximate modified topology and makes the baseline reference-only. A missing
frequency, or geometry that does not resolve to exactly two arms, returns
`unknown`.

**Limitations as the tool reports them.** The free-space half-wave and
quarter-wave quantities are reference baselines, not exact physical resonant
lengths. Conductor diameter, feed gap, end effects, nearby materials, supports and
environment can all change the actual resonance. Input impedance, bandwidth,
efficiency, gain, balun behavior and radiation-pattern validity are not evaluated.

## The evaluated geometry query

The query is the derived, read-only view of a design that the analysis tools and
the preview both read. It is the supported way to ask "what geometry does this
design actually have, after transforms and Booleans?"

```python
from studio.antenna_geometry import evaluate_geometry

query = evaluate_geometry(design)
patch = query.object("element_1_1_patch")
shape = patch.require_planar_shape()
```

`EvaluatedGeometryQueryResult` and its frozen record types live in
`studio/antenna_geometry_query.py`. Schema version 1 contains:

- `design_ref` — the exact canonical design ID and revision;
- `semantic_design_hash` — the agent's semantic state fingerprint;
- `geometry_hash` — an evaluated content and provenance fingerprint;
- `objects`, `elements`, `ports`, `booleans`, `excitation`, `coverage`;
- `tolerances`, `warnings`.

### Object and Boolean provenance

Each object retains its canonical ID and name, primitive, semantic role, material
including conductor/dielectric classification, component, original and resolved
tags, element coordinates where canonical tags provide them, transform, resolved
dimensions, bounds, evaluation status and approximation metadata. Roles and
element mapping follow canonical conventions; no role is inferred from
conversation or from visual geometry.

Three planar shapes are exposed, and the distinction matters:

- `local_planar_shape` — the original primitive before its canonical transform,
  when the XY extrusion evaluator can represent it;
- `source_planar_shape` — the original primitive in world XY, **for provenance
  only**;
- `planar_shape` — the final resolved physical world-XY shape, including holes.

Alongside them: `source_primitive_ids` (retained material contributors — the
target and its unions), `boolean_history` (ordered transitive operation IDs,
including subtractions), `booleans` (operation kind, target and tools, evaluation
status and reason), and `consumed_by` (the operation that removed a tool as an
independent object).

A unioned feed stays addressable by its original shape, role and element, but it
is **not** a second physical conductor. Subtraction tools likewise survive as
provenance records only; cutter IDs are reachable through Boolean records and are
deliberately not listed as retained material contributors. A provenance shape is
not a claim that its original region still contains material after later cuts.

Canonical duplication stores a new primitive plus a transform rather than a
duplication-source link, so the query exposes the new canonical ID and does not
invent a historical relationship that canonical state does not record.

### Planar queries and Z

Shapely 2 shapes are immutable. `require_planar_shape()` rejects consumed,
unresolved, empty and non-XY geometry. A returned shape supports `intersects`,
`intersection(...).area`, `distance`, `contains`, `covers` and
`boundary.distance`, including point-in-hole tests.

These are **projected XY predicates, not physical contact classifications**.
`z_min` and `z_max` give the explicit world-Z extrusion interval for supported
planar objects. For nonplanar objects they are only world-Z bounding extents, with
no claimed world-XY extrusion shape. A caller reasoning about real contact must
inspect coverage, Z and the approximation metadata together.

### Array frames and ports

Element row and column indices are 1-based. Origins derive from the canonical
`ArraySpec` row and column vectors, counts and spacing, centered about the layout
origin; local-to-world conversion translates element coordinates by that origin.
Array vectors describe placement, not rotation of individual element axes — object
transforms stay explicit and separate.

The query never uses ports to locate radiator centers; use `query.elements`.
Ports retain ID, name, type, impedance, canonical ordinal and element membership,
and evaluated positive and negative endpoints. No conductor attachment is inferred
or checked.

### Coverage, tolerances and hashing

Coverage distinguishes `completed`, `partially_evaluated`, `unsupported` and
`failed`, across objects, Boolean operations, layout and ports. A failed
expression never becomes a zero coordinate. Unsupported Boolean dependencies
propagate: an unresolved intermediate is never used to claim a final physical
shape.

`is_physical` is `True` for known surviving solids, `False` for consumed or fully
removed objects, and `None` when an object did not resolve. Read
`physical_objects` together with `coverage` and `unresolved_objects` — never as
evidence that the whole design resolved. Standalone tilted or non-Z primitives
have resolved dimensions, transforms and analytic bounds but only partial
coverage: general 3D CSG is not supported.

The evaluator's tolerances are centralized in `GeometryTolerances`:

| Constant | Value |
| --- | --- |
| `layer_mm` — layer compatibility | `1e-7` mm |
| `rotation_deg` — rotation threshold | `1e-10` degrees |
| `mesh_vertex_decimal_places` | `12` |
| `circle_quad_segments` — planar CSG, per circle quadrant | `16` |
| `nonplanar_mesh_circle_segments` — cylinder meshes | `48` |

Circle polygons report their own primitive chord error. That is **not** an RF
tolerance and not a certified global error bound across later Boolean operations.
Small Z mismatches that the layer tolerance accepts receive explicit partial
coverage rather than passing silently.

The SHA-256 `geometry_hash` covers normalized Shapely WKB, evaluated dimensions
and transforms, material and object/element relationships, Boolean provenance,
evaluated ports, coverage and tolerance policy. It excludes design identity and
revision, and design metadata and audit timestamps, so it is stable across
identity-only changes. It uses no mesh or tessellated triangle list. It is a
reproducible content fingerprint **within the same evaluator and GEOS
environment**, not a cross-version geometric-equivalence guarantee.

### What is pinned by tests

`tests/geometry_query_mesh_baseline.json` holds twelve fingerprints captured from
the evaluator, covering every vertex, face and ordering, color, source ID,
evaluated volume, scene bounds, port, operation and warning for these fixtures:
plain inset patch, circular hole, rectangular hole, edge union, circular patch,
dipole, 2x3 array, translated/duplicated, rotated, separated-Z, unsupported
nonplanar Boolean, and tilted standalone primitive.

`tests/test_antenna_geometry_query.py` checks exact baseline equality plus
read-only access, resolved holes, constituent identity, array origins independent
of ports, evaluated endpoints, explicit unknowns, absence of meshing during
queries, and deterministic hashes. `tests/test_rectangular_patch_analysis.py` and
`tests/test_dipole_baseline_analysis.py` pin the two baselines, including the
worked numbers quoted above.
