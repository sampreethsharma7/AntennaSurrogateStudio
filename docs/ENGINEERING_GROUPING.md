# Deterministic planner engineering summary

Implemented in the experimental worktree. No engineering checks, calculations, geometry, tools, recipes, ports, feed generation, CST, VTK, GUI, memory, or provider-specific behavior changed. No commit or merge.

## Projection and identity

`summarize_engineering_report(report, design)` produces an immutable `PlannerEngineeringSummary` without modifying the raw EngineeringReport or reevaluating geometry. Canonical object roles come from the existing semantic-role helper over canonical geometry metadata. Finding prose is never a grouping key.

A group signature includes grouping version, check ID/version, category, severity, relationship type (from the existing structured relationship key), the multiset of canonical object roles, relevant relationship flags, measurement names/units/tolerances, and source method/applicability/limitations. Magnitudes are aggregated rather than used as identity. Object IDs, element positions, message text, revision, and audit metadata do not identify a repeated group.

Only reviewed categories in the existing checks are eligible for repeated grouping. Unknown future categories, or object-bearing findings whose canonical roles cannot be resolved, stay individual. This conservative fallback prevents unfamiliar findings from merging based on similar text. No dimensions or particular object IDs are hard-coded.

Group IDs are `eng-group-v1-` plus SHA-256 of the normalized signature. They are stable across input ordering and revision/audit-only changes. Membership may change as a design evolves; therefore group acknowledgements bind to the complete current raw-report hash, not just the group ID.

Every full group retains:

- check/category/severity and relationship type;
- group ID, occurrence count, and all constituent observation IDs;
- affected object/port/element unions;
- individual relationship keys and their affected sets;
- separately named aggregate measurements;
- one representative raw message, explicitly labelled as describing one occurrence, and source limitations.

No LLM summary call or prose parser is used.

## Measurements

Each measurement keeps its original name and unit, with min, max, tolerance, and common_value. A common value is present only when the range is within that measurement's declared tolerance (exact equality if no tolerance exists); the representative common value is the midpoint when values differ. Min/max remain available even then. Incompatible measurement schemas or units split groups.

In the existing conductor-contact group, `retained_constituent_xy_overlap_area_mm2` remains **60.4995 mm²**, while `resolved_conductor_xy_overlap_area_mm2` remains **107.4279 mm²**. They remain two different structured quantities. Observations describing only one of these meanings have different measurement signatures and cannot be grouped as if they were the same quantity. Co-occurrence of both quantities in an existing raw observation is preserved without duplicating that observation into multiple groups.

## Planner context and finish contract

The provider-neutral exchange now serializes `engineering_summary` when a report is present. The raw report remains internal and in audit, rather than being duplicated inline. The compact projection includes current design reference, semantic hash, raw report hash, summary hash, check coverage, raw finding/warning counts, grouped warning count, and groups. It omits each group's raw constituent-ID list and per-occurrence relationship details from the prompt; those remain in the full summary/audit.

For the controlled array, the complete report has **54 findings, including 15 warnings**. Serialized engineering context decreased from **95,772 to 25,640 characters**, about **73% smaller**. This measures only the engineering field, not the entire planner request or token count. Informational findings and coverage remain represented. A clean single patch has zero warning groups.

There are four warning groups:

| Check | Category | Occurrences |
| --- | --- | ---: |
| conductor_contact_clearance | conductor_contact | 3 |
| port_attachment_distinctness | coincident_port_segments | 3 |
| port_attachment_distinctness | port_element_association | 6 |
| excitation_consistency | excitation_shared_segment | 3 |

To minimize the contract change, the existing acknowledgement fields accept either raw observation IDs or group IDs:

```text
engineering_disposition:
    semantic_design_hash: current summary's semantic_design_hash
    acknowledged_observation_ids: [observation IDs and/or group IDs]
    deferred: [{ observation_id: observation ID or group ID, reason: ... }]
    engineering_report_hash: current summary's engineering_report_hash
```

`engineering_report_hash` is required by deterministic validation when any group reference is used. The runner supplies its trusted current summary to the validator. Group references are expanded to their raw members before the existing completeness/duplicate checks. Unknown groups, stale report hashes, repeated groups, group-plus-member double counting, and incomplete warning coverage are rejected. Deferment uses the same expansion and still requires a reason.

Individual-ID dispositions retain their existing semantic-hash contract. No-report AgentStep context/schema/instructions and legacy ToolPlan paths retain their previous fingerprints. The extra report-hash property and generic grouping instructions are supplied only when engineering evidence exists. The raw-report objects still pass through the internal provider-neutral Python interface; only the serialized model context changes.

Warning severity, blocking behavior, canonical validation, publication, and normal-budget correction remain unchanged. No deterministic prose policing was added. Group acknowledgement is not resolution or RF certification.

## Audit

The existing trajectory keeps all full raw EngineeringReport definitions and their hashes unchanged. Each entry additionally records `before_summary_hash`, `after_summary_hash`, and first-use `summaries` definitions. Each definition stores the full summary with membership and the exact compact planner_context. Later entries reference existing definitions rather than repeating them.

Summary hashes identify the complete deterministic summary; compact context carries that hash. Raw report hashes include exact design revision and all report content, so a stable group ID from a prior report cannot be accepted with a stale acknowledgement. Role-aware summaries are cached alongside the existing per-turn raw-report cache. No persistent cache or new project-memory data was introduced.

The summary uses the same credential redaction as raw report serialization. No request headers, credentials, or hidden reasoning are stored.

## Live retest — 2026-09-23

Unchanged request:

> Turn this antenna into a 2x3 array while keeping the existing element design.

Both models received identical initial and post-array summaries, verified against the audit reconstructions. Both returned the same registered action batch:

```json
[
  {"name": "parameter.set", "arguments": {"key": "array_rows", "value": 2}},
  {"name": "parameter.set", "arguments": {"key": "array_columns", "value": 3}}
]
```

Both then returned `finish`, zero calls, and the same disposition:

```json
{
  "semantic_design_hash": "7ac610d166ba4be2264ed747df3f23f9f337813242e14dc5517833099c5bdffc",
  "acknowledged_observation_ids": [
    "eng-group-v1-2004a980c985673ca0ba7a9fbd21a42ca083feeb65aea2d356c56de38448086d",
    "eng-group-v1-8bef0ba6644deacdbac2890513b513f6d6997d45a8a99a24e6f8454ebe336099",
    "eng-group-v1-959bbcc25c20e68e6fab23fed0218860bfa861f3f99fe7272c52dd36e13e336e",
    "eng-group-v1-dc7da5f06e4442c7f8c00708372c081addc2ae623b0772cf185d6e05337471e3"
  ],
  "deferred": [],
  "engineering_report_hash": "a373d10e7cad12bfafb91557fa17692450aa4b5dde966930e093ee1a50d04a91"
}
```

The four IDs respectively identify coincident_port_segments, port_element_association, excitation_shared_segment, and conductor_contact. Expansion verified complete, exactly-once coverage of all 15 raw warnings. Both runs finished in two decisions, with **zero schema repairs, zero runner rejections, zero retries, and no unavailable calls**. Provider transport/decoding remained unchanged.

**Gemini (`gemini-3.8-flash`) final message:**

> The antenna has been successfully converted into a 2x3 array while preserving the existing inset patch element design. Note that geometric checks flagged warnings: 1) Conductor contact between row 1 patches and row 2 feed lines due to multi-row microstrip routing; 2) Coincident port segments at the board edge; and 3) Associated cross-element excitation attachments and shared port segments. These geometric checks do not certify RF isolation, match, or corporate-feed performance.

This accurately discloses the observed categories and avoids an RF-validity claim. It does not quote overlap quantities, so numerical explanation quality is not established by this run.

**Nemotron (`nvidia/nemotron-3-ultra-550b-a55b:free`) final message:**

> The antenna is already configured as a 2×3 array (array_rows=2, array_columns=3) with 6 elements and 6 ports. The existing element design (inset-fed rectangular patches at 2.45 GHz on FR4) is preserved. No changes are needed.

This refers to the working state after its accepted action. It structurally accounts for the warnings but still omits their conversational disclosure. The response is incomplete as an engineering explanation, despite a valid terminal contract. No prompt workaround was added. Its successful structured response compared with the prior no-output run is an observation, not proof that grouping caused the improvement.

Raw terminal plans, reports, exact planner summaries, and full trajectories are in the ignored local artifacts `.test_runs/grouping_live_gemini.json` and `.test_runs/grouping_live_nemotron.json`; `.test_runs/grouping_live.py` reproduces the test without embedded credentials.

## Validation

15 new tests cover grouping equivalence and semantic separation, measurement names/units/tolerances, preserved memberships and affected sets, stable IDs, no raw-report mutation, clean-design groups, complete and deferred group accounting, duplicate/stale/unknown rejection, normal runner correction, summary audit reconstruction, provider parity, legacy paths, and redaction.

Focused regression result: **150 passed** in 7.091 s. Syntax compilation and whitespace checks passed.

Full repository suite: **699 tests in 100.706 s: 695 passed, 2 skipped, 2 failed**. The failures are the existing VTK preview availability failures in `test_antenna_builder_page`: `test_circular_patch_and_dipole_rebuild_dynamic_controls_and_preview` (zero mesh actors instead of two) and `test_preview_rotation_zoom_and_laptop_footer_remain_reachable` (`preview.available` is false). Rendering and GUI code were not changed in this cleanup. Full output is in `.test_runs/grouping_full_suite.log`.
