# Stage 4A persisted-composition editing

Persisted composed geometry remains an immutable recipe-independent operation group. Stage 4A adds stable identity and two generic design actions without adding feature-specific geometry tools.

## Identity and planner context

Every `ComposedToolCall` has an `operation_id`. New calls receive a deterministic content-derived ID at capture time, and that ID is preserved when arguments change. Saved legacy calls without an ID receive the same deterministic ID when loaded. IDs must be nonempty and unique within a group.

The runtime `composed_features` manifest supplies:

- group ID, source family, target selector, scope, selected elements, and coordinate frame;
- created parameter keys, values, units, expressions, and their operation IDs;
- every persisted operation's ID, registered tool name, editable arguments, and semantic target.

This data is derived from canonical persisted structure. It is not an LLM-generated summary.

## Generic actions

`composition.update_operation` accepts a group ID, operation ID, and recursive partial argument update. It merges the update into the stored arguments and validates the complete result against the original registered primitive schema. Direct numeric edits to a leaf driven by a composition-owned parameter update that existing parameter while retaining the symbolic geometry expression.

`composition.delete` removes one logical operation group. The executor rebuilds recipe geometry and replays all retained groups, which removes the deleted group's instantiated tools and Booleans. Parameters created only by the deleted group disappear through the same rebuild. If retained groups still depend on one, replay fails transactionally rather than guessing ownership or deleting shared data.

## Replay and arrays

Edits change the logical stored operation once. Normal replay instantiates it for single, selected, or all-element scope. Target-local coordinates remain offsets from each resolved semantic target, so later array spacing and layout changes move complete decorated elements. World-coordinate groups retain world behavior.

All edit and delete actions use the existing candidate-state transaction. Schema, target-resolution, expression, Boolean applicability, canonical validation, and engineering checks run before publication. A rejected edit, later clarification/refusal, or provider failure leaves the published baseline unchanged.

No CST, VTK, recipe, feed/excitation, memory, or GUI behavior is changed by Stage 4A.
