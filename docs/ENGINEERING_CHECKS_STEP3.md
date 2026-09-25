# Stage 3 Step 3: geometric engineering checks

Implemented in `studio/antenna_engineering_checks.py`. This is a standalone,
read-only inspection API. It is not registered as an agent tool and is not called
by the runner, planner, GUI, canonical validator or project persistence.

## API and report boundary

```python
from studio.antenna_engineering_checks import (
    CONTACT_CHECK, PORT_CHECK, EngineeringCheckConfig, run_engineering_checks,
)

report = run_engineering_checks(
    design,
    checks=[CONTACT_CHECK, PORT_CHECK],
    config=EngineeringCheckConfig(clearance_neighbors=1),
)
payload = report.to_dict()
```

`design` is an `AntennaDesign`. Omitting `checks` selects both installed checks:

- `conductor_contact_clearance`, version `1`;
- `port_attachment_distinctness`, version `1`.

The suite version is `geometry_checks_v1`. The suite evaluates canonical geometry
exactly once and passes the identical immutable query to both check functions.
It returns the existing `EngineeringReport`, with exact design reference,
semantic state hash, observations and independent check coverage. No protocol
schema was changed. Findings carry the evaluated geometry hash, canonical
object/port/element references, numerical measurements with units, role/provenance
assumptions, and method/applicability/limitations. Capability references remain
empty; no repair capability is advertised and no repair instructions are emitted.

For direct inspections of an already evaluated query, the module also exposes
`check_conductor_contact_clearance(query, config)` and
`check_port_attachment_distinctness(query, config)`. Each returns observations
and its coverage record.

`engineering_check_cache_key(query, checks=..., config=...)` fingerprints geometry,
suite/check versions and configuration. No cache is installed. A future cache
must refresh the report's exact design reference when reusing geometry-only
results across revisions.

## Conductor filtering and geometric measurements

Only surviving conductive objects participate as physical solids. Dielectrics
and Boolean-consumed tools are excluded. Unresolved geometry with unknown material
classification is conservatively relevant. Supported objects must expose a
completed world-XY polygon or multipolygon plus a trustworthy extrusion interval.

All supported physical conductor pairs are measured; no bounding-box overlap is
used as a contact conclusion. All credible contacts are retained. Noncontact
output is limited to the nearest `clearance_neighbors` neighbours per conductor
(one by default, configurable from zero to eight), deduplicated by object pair.
Thus clearance output grows at most linearly for a fixed setting, rather than
listing thousands of distant pairs. Real contact warnings are not hidden by this
filter. Mixed semantic roles select constituent attribution inside unions;
element membership controls conservative contact severity.

For resolved shapes A/B and their Z intervals:

```text
XY clearance = distance(A, B)
Z overlap = max(0, min(z_max_A, z_max_B) - max(z_min_A, z_min_B))
Z clearance = max(0, max(z_min_A, z_min_B) - min(z_max_A, z_max_B))
minimum clearance = hypot(XY clearance, Z clearance)
contact within tolerance = minimum clearance <= geometric tolerance
```

Observations report XY intersection area, shared XY boundary length, planar/Z
clearance, Z overlap and contact classification. XY intersection area alone is
never called physical contact. It can be positive while the conductors are
separated in Z. Area is an evaluated polygonal area, not intersection volume or
an RF metric.

Same-element contacts are informational. Cross-element or unassigned distinct
object contacts are warnings, describing geometry without declaring every contact
a defect. Canonical unions establish a structural composition: mixed-role
constituent contacts inside them are informational unless different elements are
involved. No findings are blocking.

## Boolean and constituent provenance

Contact is determined from final survivor geometry. To attribute it, retained
source footprints from the query are intersected with that final geometry.
Consumed feed pieces are not resurrected as independent solids; cutting tools
are not material contributors. A fully cut-away feed region produces no internal
feed-contact finding. The strongest mixed-role overlap is reported for each
contacting physical pair, separately from the whole-pair area.

These clipped footprints describe spatial provenance. They do not reconstruct
chronological material ownership where overlapping source regions have been
removed and refilled by later operations. This distinction remains explicit in
the observations and is not used to prescribe a repair.

## Ports and geometric connected components

Attachment is supported for discrete ports. Other evaluated port types can still
be compared for segment coincidence but receive unknown attachment coverage.
Both endpoints come from the query's evaluated coordinates.

An endpoint attaches when its Euclidean distance to a supported conductive
extrusion is within tolerance. The test combines Shapely point/polygon distance
with distance to the Z interval. Boolean holes remain holes; being inside an
original cutter or primitive bounding box cannot establish attachment. Hole
observations reference the resolved survivor, not the removed cutter.

Each polygon island is initially a separate component. A geometric component
graph joins islands only through credible tolerance-aware 3D contact. This avoids
declaring disconnected islands in one Boolean union connected merely because they
share a canonical object ID. Both terminals reaching the same known component
produce a geometric-suspicion warning. This does not assert electrical calibration,
modal correctness, matching, impedance or RF performance.

Element association uses retained constituent footprints at the actual endpoint,
so a union's root element label does not incorrectly override another constituent's
local membership. Global conductors without element association do not imply an
element mismatch. Current canonical ports have no explicit conductor target;
the checks do not invent an expected object from names.

Two port segments coincide when the smaller of the direct/reversed endpoint-pair
distances is within tolerance. Each pair distance is the maximum of its two
endpoint distances. Warnings retain both port IDs and canonical elements.
Distinct-segment summary groups use deterministic complete-link comparisons:
tolerance proximity is not assumed transitive. Partially overlapping or crossing
segments with different endpoints are not classified as identical segments.

## Tolerances, unknowns, and failure behavior

The existing query tolerance `layer_mm` (currently 1e-7 mm) is reused as the linear
geometric tolerance for contact, attachment and endpoint coincidence. No RF
tolerance is introduced. Areas are measured directly; squared linear tolerance
is used only to reject negligible attribution areas.

Circle chord approximation error is considered separately from floating-point
tolerance. Near misses/tangencies within the summed chord error are unknown
unless there is substantial interior overlap. Endpoint attachment near an
approximated boundary is likewise unknown. The query currently supplies a
survivor-wide approximation bound rather than localized boundary provenance.
Consequently a straight edge on a patch with a circular subtraction can also
receive conservative unknown coverage. This is an acknowledged false-unknown
limitation, not a false clean result or a claimed defect.

Coverage values are:

- `completed`: the applicable supported relationships were inspected; warnings
  may still exist, so this is not a certification of correctness;
- `not_applicable`: no relevant conductors or no ports for that check;
- `unknown`: unsupported/partial conductor geometry, uncertain approximation,
  unresolved endpoints/types/layout association prevents a complete conclusion;
- `failed`: evaluation or check execution failed; partial findings from that
  failed check are discarded. Other independent checks may still complete.

Known positive geometry evidence may coexist with unknown coverage. Unmatched
terminals are not called definitively unattached when unresolved conductor
geometry could explain them. Exception reports expose only exception class names,
not arbitrary exception payloads. No design state is mutated or published.

## Existing default 2x3 array result

Both checks complete on this fully evaluable polygonal design:

| Measurement | Result |
| --- | --- |
| Cross-element physical conductor pairs contacting | 3 |
| Resolved whole-conductor XY overlap, each pair | 107.4279 mm² |
| Upper feed / lower radiator overlap, each pair | 60.4995 mm² |
| Remaining feed/feed contribution, by subtraction | 46.9284 mm² |
| Z overlap at each contact | 0.035 mm |
| Canonical ports | 6 |
| Distinct endpoint-pair segments | 3 |
| Coincident port-pair warnings | 3 |
| Endpoint association warnings | 6 |

The feed/feed remainder above explains why whole-conductor area exceeds the
specific feed/radiator area; the checker reports the total and strongest mixed-role
attribution separately. There are no rules keyed to these object IDs or array
dimensions. Tests also reproduce the behavior on renamed 2x2 and 3x1 layouts.
The plain single inset patch produces informational internal union contact and
two attached terminals with no warnings.

## Tests

`tests/test_antenna_engineering_checks.py` covers intended union contact, the array
issue, alternate layouts/names, separate Z layers, small positive clearance,
tolerance contact, output filtering, an object and a port inside a Boolean hole,
removed union constituents, translated/rotated geometry, overlapping bounds without
actual contact, correct/reversed/near-coincident ports, same connected component,
disconnected union islands, local constituent element association, partial geometry,
curved-boundary uncertainty, invalid port endpoints/types, deterministic IDs and
hashes, single evaluation, report serialization, failure coverage and nonmutation.

Existing query fingerprint tests continue to verify unchanged mesh output. No
recipes, feeds, ports, CST code, VTK code, GUI, memory, providers, tools, planner,
runner, canonical validators or existing query/report contracts were changed.

Validation results in the current Python environment:

- 20 new engineering-check tests passed.
- Focused engineering-check, query, report-protocol and geometry suite: 61 passed.
- Full `python -m unittest discover -s tests -v`: 641 tests in 116.721 seconds;
  637 passed, 2 skipped, 2 failed.

The full-suite failures are the unchanged VTK preview-availability failures:
`test_circular_patch_and_dipole_rebuild_dynamic_controls_and_preview`
(`mesh_actor_count` zero instead of two), and
`test_preview_rotation_zoom_and_laptop_footer_remain_reachable`
(`preview.available` false). VTK is unavailable in this interpreter, and the two
VTK-specific tests skipped. No rendering dependencies or GUI behavior were
changed to address that unrelated environment issue.
