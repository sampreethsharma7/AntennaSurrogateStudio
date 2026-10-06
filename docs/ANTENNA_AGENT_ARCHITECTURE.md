# Experimental Antenna Design Agent Architecture

The builder uses a progressive tool architecture. Natural language never writes
CST commands or geometry directly.

```text
User instruction
  -> selected planner transport (Local Ollama, Gemini, Groq GPT-OSS, or OpenRouter Nemotron)
  -> identical system instruction + AntennaDesign + capability manifest + ToolPlan schema
  -> ordered registered-tool plan / clarify / refuse
  -> exact ToolPlan parsing and validation
  -> AntennaDesignAgent validates and executes the registered plan
  -> recipe/modifier actions and/or selected primitive tool calls
  -> solver-neutral AntennaDesign graph
  -> layered validation
  -> preview / parameter table / LHS
  -> CSTAdapter
```

## Boundaries

- `antenna_design.py` defines the canonical graph: parameters, materials,
  primitive geometry, transforms, Boolean relationships, ports, array metadata,
  simulation setup, replayable composed-operation groups, and validation
  records.
- `antenna_tools.py` owns the tool registry and the small geometry and EM tool
  implementations. It discovers capability modules in
  `studio.antenna_capabilities` and supports Python entry points in the
  `antenna_surrogate_studio.antenna_capabilities` group.
- `antenna_recipes.py` composes primitive calls into the three installed
  recipes. Recipes are conveniences rather than geometry backends.
- `antenna_modifiers.py` contains reusable post-recipe compositions. The first
  modifier creates four circular corner cutouts from cylinder and subtraction
  tools while keeping their radius tied to patch width.
- `antenna_llm_planner.py` builds one provider-neutral `PlannerExchange`, then
  transports it through loopback Ollama, the optional Gemini REST API, or the
  optional Groq API, or optional OpenRouter API. All four
  receive the same system instruction, current design, live capability
  manifest, and ToolPlan JSON Schema with one exact argument schema per
  registered planning tool. Qwen also receives it as its provider decoding
  schema. Gemini receives it in the shared planning context and returns JSON
  for the same strict post-generation parser and validation contract. Groq
  first receives the unchanged schema in strict structured-output mode and can
  fall back to JSON-object output plus that same parser if the provider rejects
  the exact strict request.
- `antenna_agent.py` owns the planning-tool registry. It includes five design
  actions plus the small planner-exposed primitive subset registered by
  `antenna_tools.py`. It rejects unknown calls and invalid arguments, compiles
  accepted recipe/modifier calls, executes primitive calls through the same
  deterministic registry, and preserves design ID while incrementing the
  revision.
- `antenna_validation.py` validates parameters, primitive geometry, recipe
  structure, ports, arrays, and simulation setup.
- `cst_antenna_adapter.py` is the only layer that emits CST History List/VBA
  operations. Future HFSS, FEKO, or openEMS adapters should consume
  `AntennaDesign` without changing it.
- `antenna_builder.py` is a compatibility and persistence facade used by the UI.

The LLM can call only this selected primitive surface:

- `parameter.create`;
- `geometry.rectangle_sheet`, `geometry.circle_sheet`, and
  `geometry.cylinder`;
- `geometry.translate` and `geometry.duplicate`;
- `boolean.subtract` and `boolean.union`.

The remaining primitive inventory is context only. Runtime schemas restrict
new object IDs, materials, tags, conductor-layer z bounds, transforms, Boolean
targets, and tool references. The capability manifest supplies semantic roles,
stable object IDs, resolved bounds, and element centers, so the model can
identify a radiating patch without guessing. The model cannot emit code, files,
CST commands, solver operations, or unregistered calls. A clarification or
refusal contains no calls and cannot mutate the design.

The Local Ollama transport requests a 16,384-token context so the current design
and runtime schemas are not truncated. Malformed structured output receives one
schema-repair attempt. If an otherwise valid call sequence fails the registered
executor (for example, a modifier parameter appears before `modifier.apply`),
the model receives that exact error for one repair attempt; the rejected state
is never published. A final deterministic capability guard blocks explicit
solver-start requests and substitution of a known uninstalled antenna family.
That guard consults runtime recipe aliases, so installing a validated future
family makes its name available without changing the planner protocol.

Gemini is a transport choice, not a second agent. It cannot see additional
tools and has no model-specific prompt rules or deterministic phrase handlers.
The Gemini API rejected the unchanged thirteen-branch ToolPlan union when it
was supplied as `responseJsonSchema`, so Gemini uses `application/json` response
mode and the existing strict post-generation parser. This changes only where
schema enforcement occurs; it does not simplify the schema, alter any tool, or
bypass applicability and resulting-design validation. Audit records identify
the decoding mode so comparisons remain explicit.
Its API key is loaded from `GEMINI_API_KEY` or an ignored local `.env` file and
is sent only in the `x-goog-api-key` request header. The builder visibly marks
Gemini as cloud because the current design context and request leave the
computer. Each project writes credential-free comparison records to
`design/planner_ab.jsonl`, including backend, request, returned plan, validation,
repair, and final deterministic tool sequence.

Groq is the third transport choice and defaults to `openai/gpt-oss-120b`. It
has no additional tools or prompt rules. Its key is loaded from `GROQ_API_KEY`
or the same ignored local `.env` and is sent only in the Authorization header.
Audit metadata distinguishes strict provider schema enforcement from the
JSON-object fallback. The fallback leaves the ToolPlan schema in the identical
shared planner context and still requires the unchanged local parser,
validators, executor, repair limit, and transactional publication gate.

OpenRouter Nemotron is the fourth transport and uses the fixed model
`nvidia/nemotron-3-ultra-550b-a55b:free`. That endpoint does not enforce
`response_format`, so no provider schema parameter is sent. The shared system
instruction still requires one JSON object, and the unchanged local parser and
validation pipeline remain mandatory. Its key is loaded primarily from
`OPENROUTER_API_KEY`; `OPEN_ROUTER_API_KEY` is accepted as a local compatibility
alias. Neither spelling is written to project data or audit records.

Primitive execution uses an additional transactional gate:

1. every call exists and matches its registered argument schema;
2. all new IDs and earlier-call references are preflighted;
3. each call passes applicability checks before deterministic execution;
4. subtractive tools must be inside a semantic radiating-patch target;
5. every created geometry object must be consumed by a Boolean operation and
   every created parameter must drive planned geometry;
6. the complete resulting design must pass canonical validation.

The new state is published only after all six checks pass. Recipe/design actions
must precede primitive calls, so a plan cannot rebuild the design halfway
through a composition.

## Persistent composed operations

A primitive phase that passes every gate is captured as an immutable
`ComposedOperationGroup` in the canonical design. The group records its source
topology and ordered registered calls separately from the base recipe. Boolean
targets are stored as semantic selectors such as
`radiating_patch_conductor, element [1,1]`, rather than depending only on the
recipe's current object ID.

Later conversational edits and structured parameter-table edits use this order:

```text
rebuild base recipe
  -> reapply installed modifiers
  -> resolve semantic targets in the rebuilt design
  -> replay every composed-operation group through registered executors
  -> validate the complete design
  -> publish the revision
```

The groups serialize with `AntennaDesign`, so project save/reopen preserves
them. Missing tools, unresolved targets, changed antenna families, duplicate
IDs, out-of-bounds cutters, or any other replay failure reject the whole update;
the prior valid design remains unchanged.

## Installed capabilities

Primitive tools currently include parameter and material creation, material
assignment, box and cylinder geometry, rectangular and circular sheets,
translation, rotation, duplication, union, subtraction, discrete ports, array
configuration, and frequency setup. The current CST adapter accepts the
primitive subset used by the installed recipes and rejects rotated objects.
Only the smaller subset listed above is callable by the LLM.

Live `qwen3:8b` evaluation demonstrates genuine primitive composition without a
slot modifier: a centered circular slot, two symmetric circular slots, and a
centered rectangular slot each produced named size parameters, cutting
geometry, and a Boolean subtraction against `element_1_1_patch`. A live
two-turn check then changed `PatchW` to 40 mm and confirmed the stored slot,
Boolean operation, and CST `Solid.Subtract` survived the rebuild.

Installed recipes are:

- inset-fed rectangular patch;
- probe-fed circular patch;
- center-fed dipole.

The installed `corner_circle_cutouts_v1` modifier applies to the rectangular
patch. It creates four subtractive circles whose centers are exactly the patch
corners. `CornerRadius` is derived as `PatchW * CornerRadiusRatio`; the default
ratio is 0.25 and the ratio can be sent to LHS. The same modifier is reapplied
when base patch parameters or array dimensions change. Added copper lobes are
still unsupported and fail explicitly.

Each recipe can replicate elements into linear or planar arrays when its
spacing rules pass. Each element has an independent port. No array feed network
is synthesized.

## Adding a capability

Add a module under `studio/antenna_capabilities` that exposes:

```python
def register_capabilities(registry):
    registry.register_tool(...)
    registry.register_recipe(...)
    registry.register_modifier(...)
```

An independently packaged extension can expose the same callable through the
`antenna_surrogate_studio.antenna_capabilities` entry-point group. A recipe must
declare its required tool names; registration fails when any are unavailable.
New capabilities need primitive validity tests, parameter/update tests, recipe
structure checks, preview checks, and adapter tests before being treated as
validated.

## Validation ladder

The canonical design records these stages:

1. parameter update correctness;
2. primitive geometry validity;
3. antenna recipe structural validity;
4. simulation setup validity.
5. composed-operation history validity.

The CST adapter then applies its own representation gate. A generated native
project remains an unsolved starting design until an engineer reviews the
materials, excitation, boundaries, mesh, and simulated results.

## Relevant local precedents

The implementation adopts two narrow patterns from adjacent projects without
importing their code:

- BlackMagician keeps LLM proposals behind deterministic schemas and one CST
  write boundary.
- Claude CSTMCP exposes discoverable tools, serializes CST automation, and
  refuses in-place overwrites.

Those projects remain independent. Their larger optimization and MCP toolsets
are not dependencies of Antenna Surrogate Studio.
