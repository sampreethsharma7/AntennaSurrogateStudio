# Antenna Planner A/B/C/D Report

Evaluation date: 2026-09-20

## Controlled setup

The comparison used the same solver-neutral inset-patch design with one
persistent 3 mm circular slot centered on the radiating patch. Every backend
received the same system instruction, serialized canonical state, composed
operation history, runtime tool manifest, exact thirteen-branch ToolPlan
schema, executor feedback, repair limit, and deterministic validation and
transaction pipeline.

Requests ran sequentially for each backend. State advanced only after an
accepted update, matching the live conversational workflow. Nemotron's accepted
movement therefore became the state presented to its removal request.

No tools, schemas, phrase handlers, provider-specific planning instructions,
or special execution behavior were added for these requests.

## Decoding modes

- **Local Qwen (`qwen3:8b`)** supplies the unchanged schema to Ollama for
  constrained decoding, then applies strict post-generation parsing and
  validation. Audit value:
  `provider_constrained_decoding_and_post_generation_validation`.
- **Gemini (`gemini-3.8-flash`)** requests JSON output, then applies the same
  strict parser and validators. Gemini had previously rejected the unchanged
  union when used as `responseJsonSchema`. Audit value:
  `post_generation_strict_parsing_and_validation`.
- **Groq GPT-OSS 120B (`openai/gpt-oss-120b`)** first requested strict JSON
  Schema output with the unchanged schema. Groq rejected that strict request,
  so the transport selected the authorized JSON-object fallback and retained
  the full schema in the identical shared context for local strict validation.
  Audit value: `json_object_post_generation_strict_parsing_and_validation`.
  The account's 8,000 TPM limit then rejected the unchanged 14.7k-15.3k-token
  controlled request reservations before model generation. A 1,024-token
  completion cap was present; the request still exceeded the account limit. The schema and
  context were not reduced to fit the account tier.
- **OpenRouter Nemotron 3 Ultra
  (`nvidia/nemotron-3-ultra-550b-a55b:free`)** receives the same JSON instruction
  and unchanged schema in the shared context. Its free endpoint does not
  enforce `response_format`, so its text output goes through strict local
  parsing and validation. Audit value:
  `post_generation_strict_parsing_and_validation_no_provider_schema_enforcement`.
  All three API requests generated model output successfully.

## Raw Nemotron plans and outcomes

### Radius edit

```json
{
  "schema_version": 1,
  "status": "refuse",
  "message": "The slot_radius_mm parameter is a planner-created parameter, not a recipe or modifier parameter. The parameter.set tool only supports recipe/modifier parameters (allowlist: arm_length_mm, array_columns, array_rows, board_margin_mm, conductor_radius_mm, copper_thickness_mm, corner_radius_ratio, element_spacing_lambda, feed_gap_mm, feed_offset_mm, feed_width_mm, frequency_ghz, inset_depth_mm, inset_gap_mm, material, patch_length_mm, patch_radius_mm, patch_width_mm, probe_radius_mm, substrate_thickness_mm). No installed capability can modify planner-created parameters after creation.",
  "calls": []
}
```

The plan passed strict parsing as a refusal. No repair or tool execution ran,
and revision 1 with the centered 3 mm slot remained unchanged. This was a
correct recognition of a missing composed-parameter editing capability.

### Movement edit

Initial plan:

```json
{
  "schema_version": 1,
  "status": "execute",
  "message": "Move the existing circular slot 6 mm to the right (positive x) of the patch center by translating the center_slot_tool.",
  "calls": [
    {
      "name": "geometry.translate",
      "arguments": {"object_id": "center_slot_tool", "offset_mm": [6, 0, 0]}
    }
  ]
}
```

The executor rejected this because translation is limited to geometry created
in the same primitive phase. Nemotron then returned this repair:

```json
{
  "schema_version": 1,
  "status": "execute",
  "message": "Move the existing circular slot 6 mm to the right (positive x) of the patch center by filling the old slot with a copper cylinder and subtracting a new slot at the new position.",
  "calls": [
    {
      "name": "geometry.cylinder",
      "arguments": {
        "object_id": "fill_old_slot_tool",
        "material_id": "copper",
        "axis": "z",
        "tags": ["planner_created", "boolean_tool", "custom_geometry"],
        "dimensions": {"center_1": 0, "center_2": 0, "radius": "slot_radius_mm", "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm"}
      }
    },
    {
      "name": "boolean.union",
      "arguments": {"operation_id": "fill_old_slot_union", "target_id": "element_1_1_patch", "tool_ids": ["fill_old_slot_tool"]}
    },
    {
      "name": "geometry.cylinder",
      "arguments": {
        "object_id": "moved_slot_tool",
        "material_id": "copper",
        "axis": "z",
        "tags": ["planner_created", "boolean_tool", "slot"],
        "dimensions": {"center_1": 6, "center_2": 0, "radius": "slot_radius_mm", "start": "substrate_thickness_mm", "end": "substrate_thickness_mm+copper_thickness_mm"}
      }
    },
    {
      "name": "boolean.subtract",
      "arguments": {"operation_id": "moved_slot_subtract", "target_id": "element_1_1_patch", "tool_ids": ["moved_slot_tool"]}
    }
  ]
}
```

All four repaired calls executed and validation accepted revision 2. The state
retained `slot_radius_mm = 3`, added a second composed-operation group, filled
the center cut, and subtracted the active 3 mm slot at x = 6 mm. Its Boolean
history is `center_slot_subtract`, `fill_old_slot_union`, then
`moved_slot_subtract`. This is a valid geometric workaround using installed
primitives, although it does not provide a clean lifecycle edit of the original
operation group.

### Removal edit

```json
{
  "schema_version": 1,
  "status": "clarify",
  "message": "The design currently has two circular slot operations: a center slot that was subtracted then filled, and a moved slot at (6,0) that is currently subtracted. Which slot should be removed? Do you want to fill the moved slot (making the patch solid), or remove all slot-related geometry and boolean operations?",
  "calls": []
}
```

The clarification passed strict parsing. No repair or tool call ran, and
revision 2 remained unchanged. The response safely recognized the difference
between removing the visible slot geometrically and deleting historical
operation groups, but it did not complete the requested removal.

## Case 1: change slot radius to 5 mm

| Backend | Returned ToolPlan | Validation and repair | Executed actions and final result | Classification |
| --- | --- | --- | --- | --- |
| Local Qwen | `execute`: twelve `parameter.set` calls that rewrote base-recipe values to their current values; omitted `slot_radius_mm` | A first overlong/duplicate plan received the one schema repair. The repaired no-op was schema-valid and accepted. No executor repair ran. | Recipe rebuild and persistent-operation replay ran; revision advanced, but `SlotRadius` remained 3 mm. | Model semantic failure exposed by the missing request-fulfilment check. The requested edit also needs an unavailable composed-parameter edit capability. |
| Gemini | `execute`: `parameter.create(slot_radius_mm=5)` | Executor rejected recreation of an existing composed parameter. One executor-feedback repair returned `refuse`. | No tool action published; original revision and 3 mm slot remained. | Missing-tool limitation correctly recognized after repair. |
| Groq | No ToolPlan; provider rejected generation | Strict schema request was rejected; JSON fallback then received HTTP 413 from the account TPM gate. No schema or executor repair could begin. | No actions executed; original revision and design remained. | Provider/account limit. This is not a GPT-OSS planning result and does not test the missing tool. |
| Nemotron | `refuse` with no calls | Valid refusal; no repair needed. | No actions executed; revision 1 and centered 3 mm slot remained. | Missing-tool limitation correctly recognized immediately. No provider failure. |

## Case 2: move slot 6 mm right

| Backend | Returned ToolPlan | Validation and repair | Executed actions and final result | Classification |
| --- | --- | --- | --- | --- |
| Local Qwen | `execute`: `geometry.translate(center_slot_tool, [6,0,0])` | Executor rejected translation of geometry from an earlier composition phase. The one repair repeated the same invalid call. | No actions published; previous valid state remained. | Missing-tool limitation plus model repair failure. |
| Gemini | `execute`: the same `geometry.translate` call | Executor rejected it. The repair returned a precise `refuse` explaining that persisted Boolean features cannot currently be moved. | No actions published; previous valid state remained. | Missing-tool limitation correctly recognized after repair. |
| Groq | No ToolPlan; provider rejected generation | JSON fallback request received HTTP 413 before model output. No repair ran. | No actions executed; original revision and design remained. | Provider/account limit; no model-quality conclusion. |
| Nemotron | Initially the same invalid `geometry.translate` call | Executor rejected it. One repair composed two cylinders, `boolean.union`, and `boolean.subtract`; validation accepted all four calls. | Center hole was filled and a 3 mm slot was cut at x = 6 mm. Revision advanced 1→2 with two composed groups. | Initial reasoning error recovered into a valid geometric workaround. The missing persisted-feature move tool remains visible in the operation history. No provider failure. |

## Case 3: remove slot

| Backend | Returned ToolPlan | Validation and repair | Executed actions and final result | Classification |
| --- | --- | --- | --- | --- |
| Local Qwen | `execute`: reverse-order `boolean.subtract`, `geometry.cylinder`, and `parameter.create` using existing IDs | Preflight rejected ordering, missing earlier tool references, and reused IDs. The repair repeated the plan and was rejected. | No actions published; previous valid state remained. | Missing operation-group removal tool plus model planning/repair failure. |
| Gemini | `refuse` with no calls | Valid refusal; no repair needed. | No actions executed; previous valid state remained. | Missing-tool limitation correctly recognized immediately. |
| Groq | No ToolPlan; provider rejected generation | JSON fallback request received HTTP 413 before model output. No repair ran. | No actions executed; original revision and design remained. | Provider/account limit; no model-quality conclusion. |
| Nemotron | `clarify` with no calls | Valid clarification; no repair needed. | No actions executed; moved-slot revision 2 remained unchanged. | Safe recognition of missing operation-group deletion semantics, but unnecessary ambiguity from the antenna user's perspective because only one physical slot remained active. No provider failure. |

## Findings

Gemini gave the clearest behavior for the installed tool surface: it recognized
all three missing lifecycle capabilities, twice after deterministic feedback
and once directly. Qwen did not complete any requested edit and produced one
accepted semantic no-op. Nemotron recognized the radius limitation directly
and was the only tested model to produce an accepted physical slot movement,
although it required executor feedback and accumulated compensating Boolean
history. Its removal clarification was safe but did not finish the instruction.
Groq integration is complete, but this key's account tier cannot accept the
unchanged controlled context, so Groq is excluded from planning-quality
comparison.

The same provider-neutral gaps remain:

- edit a parameter owned by a persisted composed-operation group;
- move an existing composed feature;
- remove a composed-operation group;
- verify that a schema-valid plan fulfils every meaningful request clause.

All rejected attempts preserved the prior valid design. No arbitrary code,
file access, solver command, or unregistered tool was generated or executed.

## Reproduction artifacts

The ignored harness is `.test_runs/planner_ab_evaluation.py`. Credential-free
raw audits are `.test_runs/planner_ab_qwen.jsonl`,
`.test_runs/planner_ab_gemini.jsonl`, and
`.test_runs/planner_ab_groq.jsonl`, and
`.test_runs/planner_ab_nemotron.jsonl`. Production projects use the same record
shape at `design/planner_ab.jsonl`.
