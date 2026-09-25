# Structured engineering disposition at finish

Implemented in the experimental text-parametric-builder worktree. No commit or merge.

## Contract

AgentStep has an additive, optional `engineering_disposition` object on `finish` only. The existing envelope stays schema version 1; old messages without the field still parse, with their terminal validity checked against the current report. ToolPlan and registered tool schemas remain unchanged.

```json
{
  "schema_version": 1,
  "status": "finish",
  "message": "Preserved the requested geometry with the disclosed concerns.",
  "calls": [],
  "engineering_disposition": {
    "semantic_design_hash": "<current EngineeringReport.semantic_design_hash>",
    "acknowledged_observation_ids": ["<current observation ID>"],
    "deferred": [
      {
        "observation_id": "<another current observation ID>",
        "reason": "User requested that this geometry be preserved."
      }
    ]
  }
}
```

If present, the object has exactly these three fields. A hash is null or a lowercase SHA-256 digest; acknowledgements are a list of nonempty strings; deferred entries have exactly observation_id and a nonblank reason of at most 400 characters. The empty disposition is `{ "semantic_design_hash": null, "acknowledged_observation_ids": [], "deferred": [] }`. It may also be omitted when there are no warnings. Execute/clarify/refuse reject this field in the strict local parser.

Finish validation requires every current warning ID exactly once across acknowledged and deferred. Unknown IDs, IDs no longer present, duplicate IDs within or between lists, and omitted warnings are rejected. Info IDs need no acknowledgement. If any ID is supplied or warnings exist, the hash must match the current report. Observation IDs intentionally survive parameter edits; the semantic hash prevents an old acknowledgement from silently covering changed measurements on the same relationship. Revision-only changes with identical semantic state are not treated as stale geometry.

Acknowledgement/deferment is not resolution, certification, or a repair action. Valid dispositions permit finish with warnings. The existing blocking-finding prohibition remains, regardless of disposition. No deterministic inspection of the user-facing prose was added.

## Runner feedback and budgets

Canonical validation and report currency checks still precede terminal acceptance. An invalid disposition produces an audited `finish_rejected` entry and an `AgentStepObservation` with:

```text
outcome = rejected
planned_calls = executed_calls = []
failure_category = engineering_disposition
validation_result:
    status = rejected
    category = engineering_disposition
    rejected_step = finish
    current_semantic_design_hash
    stale_design_hash
    unknown_observation_ids
    duplicate_observation_ids
    missing_warning_observation_ids
```

The execution-observation type now permits a no-call rejected observation specifically for this terminal validation category. Other no-call execution observations remain invalid. Its existing serialization carries the new feedback without new top-level fields.

The next ordinary decision receives this feedback, the unchanged working design, current report, and remaining budgets. The rejected finish consumes a decision iteration, but no action-batch or tool-call allowance because nothing executed. Repeated incomplete finishes exhaust the same normal decision budget; there is no extra terminal repair call or hidden loop. All provisional changes remain unpublished until valid finish. Exhaustion, clarify, or refuse still preserves the baseline. Reports and dispositions remain in the trajectory/final step, with no new ProjectMemory behavior.

Malformed JSON or structurally malformed disposition fields continue through the providers' existing schema-repair handling. Semantic completeness/reference rejection belongs to the runner and uses ordinary iterations.

All providers share the same AgentStep schema and generic instructions. The added instructions specify current-ID accounting, semantic hash binding, optional info acknowledgements, advisory warning semantics, and accurate conversational disclosure. No provider-specific prompt or antenna-specific rule was added.

## Measurement clarification

For physical `conductor_contact` observations, two existing numerical values now have explicit names:

| Previous name | Current name | Meaning in the existing 2x3 case |
| --- | --- | --- |
| `xy_intersection_area_mm2` | `resolved_conductor_xy_overlap_area_mm2` | 107.4279 mm²: complete final conductor-pair XY intersection, including retained union constituents |
| `constituent_overlap_area_mm2` | `retained_constituent_xy_overlap_area_mm2` | 60.4995 mm²: strongest mixed-role retained constituent relationship, clipped to final material |

The message identifies the complete conductor pair and the particular constituent roles/IDs, and explicitly says the constituent quantity is not the complete overlap. Other XY-intersection measurements used by clearance or internal union findings retain their existing context-specific names. No measurement was dropped, and no geometry, calculation, threshold, severity, or observation identity changed.

Because emitted measurement names/messages changed, the conductor check version is now `2`, the port check remains `1`, and the suite is `geometry_checks_v2`. Existing cache keys already incorporate these versions. Historical audit reports remain self-contained and retain their original names/versions.

## Live retest

Date: 2026-09-22. Same fixed canonical design ID, baseline geometry, empty memory, request, tools, and provider-neutral instructions were used for both providers. Captured serialized EngineeringReports were compared and were identical at both iterations.

User request, without hints about expected defects:

> Turn this antenna into a 2x3 array while keeping the existing element design.

Models: `gemini-3.8-flash` and `nvidia/nemotron-3-ultra-550b-a55b:free`, using their unchanged JSON-generation/strict-local-parsing paths.

Both returned the same action sequence:

1. Execute parameter.set(array_rows=2), parameter.set(array_columns=3).
2. Finish with current semantic hash, all 12 warning IDs in acknowledged_observation_ids, and no deferred entries.

Both finished successfully with zero schema repairs, zero execution/finish rejections, and no invented/unavailable calls. The current warnings were three physical conductor contacts, three coincident port pairs, and six port-element association findings. Warnings did not block the accepted array result.

Gemini disclosed overlaps, cross-element attachments, and coincident port pairs in its final message. It described them as inherent to the current layout and requiring full-wave verification; the deterministic checks establish the geometry, not that broader engineering characterization.

Nemotron structurally acknowledged all warnings but its final message still said only:

> The antenna is already configured as a 2×3 array (array_rows=2, array_columns=3) with the existing element design preserved. No changes are needed.

This is the response after its array action. Structured accounting succeeded, while conversational warning disclosure failed. No prose policing or prompt workaround was added. Neither final response quoted the overlap areas, so the live retest does **not** demonstrate accurate numerical distinction; that remains unproven despite the deterministic label/value tests passing. These are single runs, not a broad model ranking.

Ignored local evidence files contain raw structured AgentSteps, reports received, dispositions, audits, and resulting canonical state, with no credentials or hidden reasoning:

- `.test_runs/disposition_live_gemini.json`
- `.test_runs/disposition_live_nemotron.json`
- `.test_runs/disposition_live.py` (controlled reproduction harness)

## Validation

14 new focused tests cover empty/no-warning finish, acknowledgement, deferment/reason validation, missing/unknown/stale/duplicate IDs, optional info accounting, current-hash binding, normal-budget correction/exhaustion, warning publication without prose policing, provider-neutral schemas/instructions, legacy ToolPlan behavior, and unchanged overlap values.

Existing Stage 4 warning fixtures now supply valid dispositions. The GUI test planner fixture also supplies a valid AgentStep finish; production GUI code and GUI assertions are unchanged. The existing legacy ToolPlan fingerprint stays unchanged; AgentStep context/schema/instruction fingerprints are updated for this intentional extension.

Focused result: **121 passed** (2.233 s). Two targeted GUI fixture checks also passed.

Final full run: `python -m unittest discover -s tests -v` — **669 total: 665 passed, 2 skipped, 2 failures**, in 90.101 s. Only the pre-existing VTK availability failures remain:

- `test_circular_patch_and_dipole_rebuild_dynamic_controls_and_preview`: mesh actor count 0 instead of 2.
- `test_preview_rotation_zoom_and_laptop_footer_remain_reachable`: preview backend unavailable.

The full log is `.test_runs/disposition_full_suite.log`. No tests were skipped or weakened to hide the fixture incompatibility; fixtures were updated to emit the new valid contract, and the full suite was rerun. Syntax compilation and whitespace checks passed. No production GUI, memory, geometry, or provider-specific changes were made.
