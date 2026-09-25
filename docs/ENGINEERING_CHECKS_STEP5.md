# Stage 3 Step 5: support containment and excitation consistency

Implemented only the two automatic geometric/structural checks and their tests. No recipe, feed, port, tool, planner/provider prompt, GUI, CST, VTK, RF analytical tool, or ProjectMemory change. No automatic repair, commit, or merge.

## Suite integration

`run_engineering_checks()` now runs four independently covered checks over **one** `evaluate_geometry()` result:

| Check ID | Version |
| --- | --- |
| conductor_contact_clearance | 2, unchanged |
| port_attachment_distinctness | 1, unchanged |
| substrate_support_containment | 1 |
| excitation_consistency | 1 |

The suite version is `geometry_checks_v3`, which enters the existing per-turn cache identity. Existing runner integration automatically sends all four checks to every provider, including explicit unknown/failed/not-applicable coverage. No new integration or planner behavior was necessary. The existing terminal disposition validator requires acknowledgement/deferment of the new warning IDs too; all warnings remain advisory. No blocking findings were added.

## Containment algorithm

1. Identify declared substrate/support objects from evaluated semantic role `substrate`, or explicit `support`/`support_region` tags. No dimensions, recipe names, or particular object IDs are special-cased. No declared support yields `not_applicable`, not a claim of successful containment; freestanding geometry is not forced inside a fictitious substrate.
2. Inspect physical conductors with completed planar geometry. Consumed Boolean tools are excluded. A union is inspected as its final resolved physical shape, so retained feed and composed-feature material participates automatically.
3. Associate a support only when element membership is compatible and evaluated Z intervals touch/intersect within tolerance. Prefer an element-local support over a global one. A unique resolved support is required. Multiple candidates, unsupported geometry, unknown layers, or a conductor separated from every declared support produce unknown coverage. There is no nearest-board or bounding-box ownership guess.
4. Subtract the resolved support polygon from the resolved conductor polygon. This includes holes, disconnected parts, translations, and supported rotations. Also subtract the tolerance-buffered support for classification. Exact edge termination and overhang smaller than the evaluator's layer tolerance are not warnings.
5. Respect curved-boundary approximation uncertainty. An apparent excess wholly inside the combined approximation/tolerance band is unknown. Curved support boundaries or retained positive curved contributors close to a containing support edge are also treated conservatively. A remote circular subtraction does not make an otherwise exact straight feed/board boundary unknown merely because the survivor carries a circular approximation marker.
6. Emit `support_overhang` only for excess beyond tolerance and approximation bounds; otherwise emit `support_contained` or `support_containment_unknown`. Overhang findings identify the final conductor, support, and retained source contributors intersecting the excess. Subtraction cutters are never treated as remaining conductive features.

Measurements:

- `outside_support_area_mm2`: area of resolved material outside the unbuffered support.
- `outside_tolerance_area_mm2`: area outside the tolerance-buffered support.
- `maximum_sampled_boundary_overhang_mm`: largest support-distance found on the sampled resolved conductor boundary, including interior rings.
- `overhang_sampling_error_bound_mm`: half the maximum sample interval. Distance is 1-Lipschitz, giving an upper error bound for the unsampled **boundary** maximum.
- `minimum_support_edge_clearance_mm`: supplied when the resolved footprint is fully contained; zero correctly describes edge contact.

The maximum is deliberately labelled sampled rather than exact. Boundary segments are densified to at most the conductor bounding-diagonal length / 256, with a tolerance floor. Bounds set only sampling density, never containment truth. The exact resolved polygon difference is the area/overhang classification basis. Boundary distance does not purport to measure the maximum interior distance of an unsupported island covering a substrate hole; the area difference still detects that case.

The canonical representation currently has no explicit general support-reference field. The membership/Z association is therefore recorded as an assumption, and ambiguous cases stay unknown. This is footprint containment, not mechanical attachment or full 3D dielectric-clearance certification.

## Excitation-consistency model

The supported representation is the existing canonically indexed discrete port. Every port has an element ordinal; no corporate-feed, local-port synthesis, excitation weights, or general feed-network semantics are invented.

- Derive expected elements from the evaluated layout when conductive elements are represented.
- Map canonical ports to those elements and report each element with no associated port as `excitation_missing_element_port`.
- Preserve incomplete/unknown coverage for invalid indices, unsupported port kinds, unresolved endpoints/layout/conductors, or near-boundary attachment uncertainty.
- Group evaluated indexed ports by endpoint-pair coincidence, accepting reversed orientation and using the existing tolerance/distance helper. Complete-link grouping avoids assuming tolerance proximity is transitive.
- For each segment group, collect canonical element associations and resolved conductor memberships at its endpoints. Global unindexed ground is neutral. Retained contributor footprints provide local membership inside unions.
- Emit one `excitation_shared_segment` warning per group associated with multiple elements, with nominal-port count, declared-element count, resolved-contact element count, and total associated-element count. A single port touching multiple elements can trigger this even when no duplicate port pair exists.
- Emit `excitation_collection_summary`: expected elements, elements with an indexed port, canonical/evaluated port counts, physical segment groups, and multi-element segment groups.

Multiple distinct ports on a single element are counted, not automatically declared wrong. The present data does not establish an intended one-port maximum or amplitude/phase strategy. An explicit per-element association gap is a credible warning even though coverage of the check can be completed: completed inspection does not mean a clean design.

This complements the existing port check. That check reports individual terminal attachment and pairwise segment coincidence. The new check reports whether the indexed collection covers represented elements and whether physical segment groups have unambiguous element associations. It does not reissue per-terminal attachment or pairwise duplicate findings. Both share geometric helpers and the same evaluated query; no second CAD model or geometry evaluation is introduced.

No finding claims impedance matching, phase/amplitude correctness, RF isolation, or corporate-feed behavior. No capability references or repair suggestions are fabricated.

## Representative findings

**Normal single inset patch:** both physical conductors (ground and radiator/feed union) are contained, with zero outside area and valid board-edge contact. One expected element, one indexed port, one physical segment group, zero multi-element groups. All four checks have completed coverage and only info findings.

**Problematic 2x3 array:** seven physical conductors are contained; no false containment warning. All six elements have canonically indexed ports, but they occupy only three physical segment groups. Each group is associated with two elements and produces one new excitation warning.

The previous twelve warnings remain unchanged: three cross-element conductor contacts, three pairwise coincident-port warnings, and six port-element association warnings. Three new collection-level warnings give **15 warnings total**. The 60.4995 mm² retained constituent overlap and 107.4279 mm² resolved-conductor overlap are unchanged.

**Varied cases:** a 0.1 mm rectangular protrusion gives 0.2 mm² outside area; exact or 5e-8 mm edge termination is tolerated. A support hole is detected using polygon differences despite bounding-box containment. Composed union overhang identifies the retained feature. Circular/Boolean-modified contained radiators pass the containment check. Six distinct indexed physical ports have no excitation warning; a missing port identifies its expected element. Unsupported/tilted geometry produces unknown coverage.

## Live retest (2026-09-23)

The same baseline and unchanged request were sent through the existing provider clients:

> Turn this antenna into a 2x3 array while keeping the existing element design.

No expected defect was mentioned. Captured initial and post-array reports for the first Gemini/Nemotron runs were identical. Both initially used only parameter.set(array_rows=2) and parameter.set(array_columns=3).

**Gemini (`gemini-3.8-flash`):** execute → finish, zero schema repairs/rejections. It acknowledged all 15 current warning IDs and disclosed inter-element contacts and coincident/shared port segments. The accepted result retains the requested element dimensions. The new evidence appears in its response as shared-segment concerns; this single run does not establish a general change in reasoning quality.

**Nemotron (`nvidia/nemotron-3-ultra-550b-a55b:free`):** first run accepted the array batch, but the next provider response contained no structured plan. Terminal status was `provider_error`; no candidate was publishable and the baseline remained authoritative. One explicit test rerun, with identical client/context and no prompt adjustment, also returned no structured plan, this time before an action. These are provider/no-output failures, not a measured reasoning failure or a missing-tool limitation. Its terminal warning acknowledgement/disclosure behavior could not be evaluated in this run. No unavailable call was observed.

Local ignored evidence:

- `.test_runs/step5_live_gemini.json`
- `.test_runs/step5_live_nemotron_attempt1.json`
- `.test_runs/step5_live_nemotron.json` (one retry)
- `.test_runs/step5_live.py`

## Tests

15 new focused tests cover all requested representative cases, tolerance, support holes, provenance, transforms, stable identities, independent failure coverage, one shared evaluation, no mutation, and inclusion of the new warning IDs in the existing disposition contract. Existing two-check expectations were updated only to account for the expanded four-check suite.

Focused regression result: **135 passed** in 4.580 s. Syntax compilation and whitespace checks passed.

Full command: `python -m unittest discover -s tests -v`.

Result: **684 total — 680 passed, 2 skipped, 2 failures**, in 90.348 s. The same pre-existing VTK-dependent GUI failures remain:

- `test_circular_patch_and_dipole_rebuild_dynamic_controls_and_preview`: mesh actor count 0 instead of 2.
- `test_preview_rotation_zoom_and_laptop_footer_remain_reachable`: preview backend unavailable.

No GUI/dependency repair or test weakening was included. The full log is `.test_runs/step5_full_suite.log`. No commits or merges were made.
