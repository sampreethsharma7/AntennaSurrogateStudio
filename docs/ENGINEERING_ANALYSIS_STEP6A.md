# Stage 3 Step 6A: read-only engineering analysis tools

Implemented in the experimental text-parametric-builder worktree. This step adds the runtime path for bounded, provider-neutral analysis tools. It does not add antenna RF formulas or change recipes, geometry checks, canonical geometry, feeds, ports, CST, VTK, GUI, ProjectMemory, or provider transports.

## Registered-tool effect model

Every `ToolDefinition` now declares an explicit `effect`:

- `design_action`: the default for every existing registered antenna tool; it may return a changed `AntennaDesign`.
- `analysis`: returns an `EngineeringAnalysisResult` and cannot execute through the mutation path.

The effect is registry metadata and is included in both the callable capability manifest and deterministic tool inventory. It is never inferred from a tool name. Tool definitions also expose a version and whether they require an existing design.

The proof capability is `engineering.design_summary` version `1`. It reports canonical object, port, array-element, Boolean-operation, and composed-feature counts. It contains no RF equation, design decision, recommendation, or repair instruction. Because it requires a design, it is absent from the callable manifest when `design=None`.

## EngineeringAnalysisResult

The immutable schema is:

```text
schema_version: 1
tool_name
tool_version
analysis_id
working_design_ref:
  design_id
  revision
semantic_design_hash
status: completed | not_applicable | unknown | failed
measurements:
  <measurement_name>:
    value
    unit
    tolerance
    uncertainty
assumptions[]
applicability
limitations[]
message
```

`analysis_id` is deterministic over tool name/version, normalized arguments, and semantic design hash. Result parsing is strict. Measurements require finite numeric values and retain units plus optional tolerance/uncertainty. Results pass through the existing credential redaction before entering planner context, cache, or serialized trajectory.

## Runner behavior

An `execute` batch is classified before execution from registered effects. It may contain only design actions or only analysis calls. A mixed batch is rejected atomically as `mixed_tool_effects`.

An accepted analysis batch:

- executes against the current working `AntennaDesign`;
- returns `AgentStepObservation.outcome = accepted_analysis`;
- includes the calls and structured analysis results;
- retains the same object state, revision, design reference, and semantic hash;
- has no deterministic design changes;
- is not classified as `no_progress`;
- does not increment the accepted design-action batch counter;
- never enters `aggregate_calls` or publishable design-change summaries.

Only the immediately preceding analysis results are sent to the next planner iteration through `agent_observation`. Older results remain in the per-turn trajectory. The provider-neutral instructions state that analysis is read-only, bounded, subject to applicability and limitations, and not full-wave validation. They also prohibit mixed-effect batches and discourage repeated identical analysis.

## Budgets and cache

`AgentLoopBudgets` adds `analysis_batches`, defaulting to `3`. Analysis calls also consume decision iterations and total proposed tool calls. Reusing a cached result still consumes an analysis batch, so repetition cannot evade the loop bound. Analysis does not consume `accepted_action_batches`.

The cache is in-memory and scoped to one runner turn. Its key is:

```text
tool name + tool version + normalized arguments hash + semantic design hash
```

An identical request against unchanged canonical semantics reuses its result and records a cache hit in both the observation and trajectory. A semantic design change produces a different key and recomputes the analysis.

## Transaction behavior

Analysis never publishes state. A later successful `finish` follows normal publication semantics. A later clarification, refusal, provider failure, rejected action, cycle, or budget exhaustion returns the original baseline as authoritative. Results produced against an abandoned working candidate remain audit evidence in the trajectory only.

The legacy `LLMToolPlan` schema, one-shot mutation executor, existing tool argument schemas, and provider-specific decoding/transport behavior are unchanged. Directly sending an analysis tool through the one-shot mutation executor is rejected rather than treated as a no-op design action.

## Validation

Focused coverage exercises all requested cases: unchanged design actions; analysis without semantic/revision change; accepted-analysis observation; provider-context serialization; result round-trip; finish after analysis; analysis before/after design actions; mixed-effect rejection; independent analysis budget; cache reuse; cache invalidation after design change; unavailable analysis for `design=None`; rollback on clarification/refusal/provider error; redacted trajectory; and unchanged legacy one-shot behavior.

Final focused regression: **150 tests passed**. Syntax compilation and whitespace checks passed.

Full repository suite: **712 tests in 95.362 seconds: 708 passed, 2 skipped, 2 failed**. Both failures are the pre-existing VTK preview availability failures in `test_antenna_builder_page`: `test_circular_patch_and_dipole_rebuild_dynamic_controls_and_preview` and `test_preview_rotation_zoom_and_laptop_footer_remain_reachable`. This step did not modify VTK, preview, or GUI code.

No commit or merge was performed.
