# Stage 3 Step 4: engineering observations in the bounded runner

Implemented in the experimental text-parametric-builder worktree. No commit or merge.

## Scope and execution boundary

`AntennaAgentRunner.run()` now obtains the existing `run_engineering_checks(design)` report before its first planner decision, after every accepted batch, and checks currency again before accepting finish. No new checks or RF tools were added. Canonical validation, tool execution, budgets, cycle/no-progress protection, recipes, ports, CST, VTK, GUI, provider transport/decoding, and ProjectMemory schemas are unchanged.

The runner's `_TurnEngineeringReports` holds a per-turn cache keyed by semantic canonical hash, check-suite version, and individual check versions. The existing default check configuration is fixed. This uses the semantic side of the requested semantic/geometry cache identity: finding source records retain evaluated geometry hashes, so no second geometry evaluation is needed to look up the cache. Unchanged states, rejected batches, and finish reuse the report. Identity-only revision changes rebind `working_design_ref` without repeating geometry inspection. The cache expires when the turn ends and its size is limited by the runner's existing action/iteration budgets.

A checker response must match the requested design ID, revision, semantic hash, and suite version. A stale/mismatched response or inspection exception becomes explicit failed coverage about the current design, with no retained findings. Exception payloads are not exposed. Existing check-level partial/unknown coverage is preserved. This does not certify the design as clean.

For `design=None`, the runner creates an empty report with `working_design_ref=null`, the semantic hash of null, no findings, and `not_applicable` coverage for both installed checks. It does not call the geometry evaluator or synthesize an antenna.

## Shared planner context

The AgentLoopPlanner protocol and common `_AgentLoopPlannerMixin.plan_agent_step()` accept `engineering_report`. The existing provider-neutral `build_agent_step_exchange()` serializes it alongside current_design, runtime_capabilities, project_memory, agent_observation, and remaining_budgets. It is not inserted into executor-observation prose. Both normal decoding and the existing schema-repair exchange retain the report. The mixin clears the transient report in `finally`, including on provider failure.

Only generic shared instructions were added:

- Findings inspect the current working design and provide evidence, not repair instructions.
- Info need not be a problem; warnings require consideration but allow finish; blocking prohibits finish.
- Incomplete/unknown/failed coverage and absence of findings do not prove RF correctness.
- Material concerns should be considered/disclosed, without inventing unavailable repair tools.
- The inspection establishes none of resonance, match, gain, efficiency, bandwidth, polarization, or full-wave validity.

AgentStep and ToolPlan output schemas are unchanged. No provider-specific prompt, decoding change, phrase handler, or second LLM call was added.

## Finish and transaction safety

Finish still runs canonical validation. A current cached report suffices; warnings/info/unknown coverage do not trigger deterministic clarification or refusal. A blocking finding would reject finish, but the installed Stage 3 checks generate none. No code interprets whether the model's prose adequately discloses a warning.

Every accepted intermediate batch remains unpublished. Clarify/refuse/budget/provider failure preserves the published baseline. The facade still performs one terminal publication, and normalizes the public revision once. Reports identify internal working revisions; the existing turn audit separately identifies the published revision. Semantic/geometry identity survives that publication normalization.

No reports or measurements are passed to the ProjectMemory reducer. Existing terminal conversation handling remains unchanged: a model's final clarification/refusal can still appear as conversational/open-question context. That is not automatic promotion of candidate measurements into facts about the published geometry. Tests verify that abandoned candidate reports exist in audit while canonical_ref and successful important_changes retain the baseline.

## Audit representation

Every trajectory entry has an additive `engineering_audit` field:

```text
before_report_hash
after_report_hash
before_geometry_hashes
after_geometry_hashes
reports: { report_hash: complete serialized EngineeringReport }
finding_delta: { added: [observation IDs], removed: [...], changed: [...] }
```

The report hash is SHA-256 over sorted compact JSON of the redacted serialized report. A complete report definition is stored only on its first occurrence in a turn; later entries reference that definition. Reconstruct by walking entries and accumulating `reports`. Each definition contains exact design_ref, semantic_design_hash, check versions, findings, and coverage. Geometry hashes come from finding provenance; the list is empty where there is no evaluated finding provenance, rather than fabricating a geometry hash.

Deltas compare findings at entry input/output by their stable IDs and complete content, so changed measurements on the same relationship are visible. Rollback entries reference the abandoned input report and restored baseline report; candidate evidence remains reconstructable. Provider failures, rejected calls, no-progress, cycle, and budget exits also retain the associated report. The unchanged builder audit writer persists these trajectory fields into `design/planner_ab.jsonl`. No hidden reasoning, request headers, or credentials are logged.

## Live acceptance runs

The existing clients and ignored local credentials were used on 2026-09-22, without GUI or project publication. Initial sandbox attempts failed before any model output; normal network execution then succeeded. These transport failures remain separately recorded and are not interpreted as model failures.

Models: Gemini `gemini-3.8-flash`; OpenRouter `nvidia/nemotron-3-ultra-550b-a55b:free`. Both use their unchanged JSON generation plus strict local parsing/validation path. Each begins with the same default canonical single-element geometry (independent design IDs), no composed features, and empty ProjectMemory. The warning-allowed case starts from the same default recipe expanded to 2x3. The original array and clean prompts do not mention the expected defects.

Requests:

1. `Turn this antenna into a 2x3 array while keeping the existing element design.`
2. `Keep the current single-element antenna unchanged and finish this design turn.`
3. `Keep this exact geometry and excitation unchanged. I accept retaining the current geometric warning conditions for this experimental model. Finish this turn and disclose any current concerns without claiming RF validation.`

The accepted array report contains three cross-element conductor contacts, each with 60.4995 mm² feed/radiator constituent overlap (107.4279 mm² total pair overlap), six ports in three distinct physical segment groups, three coincident-port warnings, and six port-element association warnings. Both checks have completed coverage. The normal single element has informational findings and no warnings.

| Case | Gemini | Nemotron |
| --- | --- | --- |
| Convert single element to 2x3 | execute, then finish; explicitly disclosed geometric contact and coincident ports between rows | execute, then finish; omitted all current warnings from the final response |
| Clean single element, unchanged | finish in one decision, no unnecessary clarification/refusal | finish in one decision, no unnecessary clarification/refusal |
| Existing warning array, explicitly allowed unchanged | finish in one decision, disclosed overlaps, coincident ports and cross-element attachments, with no RF claims | finish in one decision, disclosed all three warning categories and denied RF validation; overlap wording was imprecise as detailed below |

In both array runs the sole execute batch was:

```json
[
  {"name": "parameter.set", "arguments": {"key": "array_rows", "value": 2}},
  {"name": "parameter.set", "arguments": {"key": "array_columns", "value": 3}}
]
```

The initial report had no warnings; the next decision received the accepted 2x3 report with 12 warnings. All six successful live cases ended with status `finished`, zero schema repairs, zero execution rejections, and no invented/unavailable calls. No corporate-feed or local-port capability was claimed or used. Geometry stayed unchanged in the clean and warning-allowed cases.

Gemini's ordinary array finish said: "Note that the standard recipe generates microstrip feed lines extending to the board edge, which introduces geometric contact and coincident port segments between row 1 and row 2 elements as indicated by geometric clearance checks."

Nemotron's ordinary array finish said: "The antenna is already configured as a 2×3 array (array_rows=2, array_columns=3) with six inset-fed patch elements and six discrete ports. No changes are needed." This is its message **after** its accepted array action. It did not explicitly certify clean geometry, but it failed to disclose the warnings it had received. This is an observed model-response omission, not absent deterministic evidence, a missing-tool failure, or an API failure. No follow-up prompt rule or deterministic response policing was added.

When explicitly asked to retain/disclose the warning geometry, Nemotron did report the contacts and duplicate segments. It described feed/radiator overlap as approximately 107 mm², conflating the full physical pair intersection (107.4279 mm²) with the mixed-role constituent overlap (60.4995 mm²). Thus it recognized the relevant concern but its numerical attribution was imprecise. The unmodified report contains both measurements under distinct keys. No implementation change was made to influence this response.

These are single controlled observations per case/model, not a general model-quality ranking. Local raw structured AgentSteps, execution observations, report definitions/hashes, final states, and decoding metadata are preserved in the ignored artifacts:

- `.test_runs/step4_live_gemini.jsonl`
- `.test_runs/step4_live_nemotron.jsonl`
- `.test_runs/step4_live.py` (reproduction harness; no credentials embedded)

## Validation

14 new integration tests cover initial/updated reports, cache reuse and revision rebinding, current state identity, warning/info/unknown handling, blocking finish protection, empty designs, clarify/refuse rollback, one terminal publication, absence of measurements in durable memory, audit reconstruction/deltas, failed/stale reports, identical shared provider exchanges, and unchanged budget/no-progress handling. Existing runner tests also retain cycle and repeated-rejection coverage.

Focused command:

```text
python -m unittest tests.test_agent_engineering_integration tests.test_antenna_agent_runner tests.test_agent_loop_protocol tests.test_antenna_engineering_protocol tests.test_agent_runner_builder_integration tests.test_antenna_engineering_checks tests.test_antenna_geometry_query -q
```

Result: **106 passed** in 2.063 s. The existing protocol fingerprint test intentionally updates only the system-instruction hash for the new shared instructions; context-without-report and both output-schema fingerprints remain unchanged.

Full command: `python -m unittest discover -s tests -v`.

Result: **655 total: 651 passed, 2 skipped, 2 failed**, in 90.227 s. The same two pre-existing VTK-dependent GUI failures remain in this interpreter:

- `test_circular_patch_and_dipole_rebuild_dynamic_controls_and_preview`: mesh actor count 0 rather than 2.
- `test_preview_rotation_zoom_and_laptop_footer_remain_reachable`: preview backend unavailable.

The two VTK-specific tests are skipped. No viewer, dependency, or test weakening was included in this task. Windows sandbox temporary-directory permissions required running persistence/full tests with normal process permissions. Syntax compilation and `git diff --check` passed (the latter retains pre-existing batch-file line-ending warnings).
