# Stage 3 Step 2: evaluated geometry query

This is a derived, read-only interface. It creates no engineering findings, calls
no planner, changes no canonical state, and writes no project files.

## Entry point and evaluation boundary

```python
from studio.antenna_geometry import evaluate_geometry

query = evaluate_geometry(design)
patch = query.object("element_1_1_patch")
shape = patch.require_planar_shape()
```

`EvaluatedGeometryQueryResult` and its frozen record types are defined in
`studio/antenna_geometry_query.py`. Schema version 1 contains:

- `design_ref`: exact canonical design ID and revision;
- `semantic_design_hash`: the existing agent semantic state fingerprint;
- `geometry_hash`: evaluated content/provenance fingerprint;
- `objects`, `elements`, `ports`, `booleans`, `coverage`;
- `tolerances`, `warnings`.

The former `build_geometry_scene` evaluation loop now lives in
`evaluate_geometry`. It reuses `_planar_solid`, canonical parameter resolution,
transforms, layer compatibility, and Shapely union/difference in canonical order.
`build_geometry_scene` calls that query once, then passes final planar shapes to
the unchanged extrusion/triangulation code. Nonplanar standalone primitives retain
their existing mesh helpers, using the query's resolved dimensions. No query
reconstructs polygons from triangles. No second Boolean evaluator was introduced.

## Object and Boolean provenance

Each object retains its canonical ID/name, primitive, semantic role, material
including conductor/dielectric classification, component, original/resolved tags,
element coordinates where canonical tags provide them, transform, dimensions,
bounds, evaluation status, and approximation metadata.

The role and element mapping use the existing agent's canonical conventions.
No role is inferred from conversation or visual geometry.

- `local_planar_shape`: original primitive before its canonical transform, when
  representable by the current XY extrusion evaluator;
- `source_planar_shape`: original primitive in world XY, for provenance only;
- `planar_shape`: final resolved physical world-XY shape, including holes;
- `source_primitive_ids`: retained material contributors (target and unions);
- `boolean_history`: ordered transitive operation IDs, including subtractions;
- `booleans`: operation kind, target/tools, evaluation status and reason;
- `consumed_by`: the operation which removed a tool as an independent object.

For a unioned feed, its original shape, role, and element remain addressable, but
it is not a second physical conductor. Subtraction tools likewise remain provenance
records only. Cutter IDs are available through Boolean records; they are not
listed as retained material contributors. Provenance shapes are not assertions
that those original regions still contain material after subsequent cuts.

Canonical duplication currently stores a new primitive plus transform, not a
duplication-source link. The query exposes that new canonical ID and does not
invent historical relationships absent from canonical state.

## Planar queries and Z

Shapely 2 shapes are immutable. `require_planar_shape()` rejects consumed,
unresolved, empty, or non-XY geometry. A returned shape supports `intersects`,
`intersection(...).area`, `distance`, `contains`, `covers`, and
`boundary.distance`, including point-in-hole tests.

These are projected XY predicates, **not physical contact classifications**.
`z_min`/`z_max` describe the explicit world-Z extrusion interval for supported
planar objects. For nonplanar objects they are only world-Z bounding extents;
there is no claimed world-XY extrusion shape. Future checks must inspect coverage,
Z, and approximation metadata. This step runs no such checks.

## Array frames and ports

Element row/column indices are 1-based. Origins are derived from canonical
`ArraySpec` row/column vectors, counts and spacing, centered about the layout
origin. Local/world conversion translates element coordinates by this origin.
Array vectors describe placement, not rotation of individual element axes.
Object transforms remain explicit and separate.

The query never uses ports to locate radiator centers. The old scene's
`element_centers` convenience field remains port-derived for compatibility;
engineering callers must use `query.elements` instead.

Ports retain ID/name/type, impedance, canonical ordinal/element membership, and
evaluated positive/negative endpoints. `canonical_references` is empty because
current `PortSpec` has no target-reference field. No conductor attachment is
inferred or checked.

## Coverage, tolerances, and hashing

Coverage distinguishes `completed`, `partially_evaluated`, `unsupported`, and
`failed`. It covers objects, Boolean operations, layout and ports. Failed
expressions never become zero coordinates. Unsupported Boolean dependencies are
propagated; an unresolved intermediate is not used to claim a final physical shape.

`is_physical` is true for known surviving solids, false for consumed/fully removed
objects, and null for unresolved objects. `physical_objects` must be used together
with `coverage`/`unresolved_objects`, not as evidence that the whole design resolved.
Standalone tilted/non-Z primitives have resolved dimensions/transforms/analytic
bounds but partial query coverage: general 3D CSG is not supported.

The evaluator's existing constants are centralized, with values unchanged:
layer compatibility 1e-7 mm, rotation threshold 1e-10 degrees, mesh vertex keys
rounded to 12 decimal places, 16 segments per circle quadrant for planar CSG,
and 48 segments for nonplanar cylinder meshes. Circle polygons report their
primitive chord error. This is not an RF tolerance or a certified global error
bound for all later Boolean operations. Small Z mismatches accepted by the
existing layer tolerance receive explicit partial coverage.

The SHA-256 geometry fingerprint includes normalized Shapely WKB, evaluated
dimensions/transforms, material and object/element relationships, Boolean
provenance, evaluated ports, coverage and tolerance policy. It excludes design
identity/revision and design metadata/audit timestamps. No mesh or tessellated
triangle list is used. It is a reproducible content fingerprint within the same
evaluator/GEOS environment, not a cross-version geometric-equivalence guarantee.

## Regression evidence

`tests/geometry_query_mesh_baseline.json` was captured from the original evaluator
before refactoring. Its 12 fingerprints cover every vertex, face and ordering,
color, source ID, evaluated volume, scene bounds, port, operation and warning for
plain inset, circular-hole, rectangular-hole, edge union, circular patch, dipole,
2x3 array, translated/duplicated, rotated, separated-Z, unsupported nonplanar
Boolean and tilted standalone primitive fixtures.

`tests/test_antenna_geometry_query.py` checks exact baseline equality as well as
read-only data, resolved holes, constituent identity, array origins independent
of ports, evaluated endpoints, explicit unknowns, no meshing during queries,
and deterministic hashes. Existing CST/native history, geometry, array and feed
tests remain applicable. No dependency or packaging changes are needed.

Validation in the current Python environment:

- New query tests: 16 passed (including the final unknown-status propagation fix).
- Focused geometry/query/feed/array/CST/report group: 61 run, 59 passed, 2 skipped.
- Full `python -m unittest discover -s tests -v`: 621 run, 617 passed,
  2 skipped, 2 failed.

VTK is unavailable in this interpreter, so the two VTK-specific tests skipped.
The two full-suite failures are the pre-existing builder-page tests
`test_circular_patch_and_dipole_rebuild_dynamic_controls_and_preview` (zero mesh
actors instead of two) and `test_preview_rotation_zoom_and_laptop_footer_remain_reachable`
(`preview.available` false). Actual VTK rendering was not verified in this step;
exact mesh-input compatibility was verified against the pre-refactor fingerprints.
