"""Builder facade over the tool-using, solver-neutral antenna design agent."""

from __future__ import annotations

import json
import math
import re
import sys
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from studio.antenna_agent import (
    AgentInstructionError,
    AgentPlan,
    AntennaDesignAgent,
    CapabilityUnavailableError,
    create_default_agent,
)
from studio.antenna_design import (
    AntennaDesign,
)
from studio.antenna_geometry import GeometryScene, GeometrySolid, build_geometry_scene
from studio.antenna_agent_runner import run_antenna_agent_loop, semantic_design_hash
from studio.antenna_recipes import MATERIALS, MATERIAL_ALIASES
from studio.antenna_tools import CapabilityError, ToolCall
from studio.antenna_llm_planner import (
    AgentLoopBudgets,
    AgentStep,
    AgentTerminalResult,
    LLMToolPlan,
    PlannedToolCall,
    SemanticMemoryProposal,
)
from studio.antenna_validation import validate_design
from studio.cst_antenna_adapter import CSTAdapter
from studio.project_store import atomic_write_json
from studio.project_memory_normalization import (
    SEMANTIC_NORMALIZATION_VERSION,
    canonical_reference_identity,
    canonical_semantic_identity,
    normalize_semantic_proposal,
)
from studio.sample_generator import LHSVariable


AntennaBuilderError = CapabilityError
UnsupportedTopologyError = CapabilityUnavailableError
InstructionNotRecognizedError = AgentInstructionError
AntennaState = AntennaDesign
ANTENNA_TEMPLATE_ID = "inset_patch_v2"
MAX_ARRAY_ELEMENTS = 64
PLANNER_AUDIT_RELATIVE_PATH = Path("design") / "planner_ab.jsonl"
_PLANNER_AUDIT_LOCK = threading.Lock()


class BuilderTurnCancelled(AntennaBuilderError):
    """The UI cancelled a planning turn before its result was published."""


@dataclass(frozen=True, slots=True)
class StateUpdate:
    state: AntennaDesign | None
    changes: tuple[str, ...]
    plan: AgentPlan | None = None
    executed_tools: tuple[ToolCall, ...] = ()
    message: str | None = None

    @property
    def summary(self) -> str:
        if self.message:
            return self.message
        if not self.changes:
            return "No antenna design change was needed."
        return "Updated " + "; ".join(self.changes) + "."


def user_facing_builder_text(value: Any) -> str:
    """Replace implementation identifiers in transcript copy while retaining audit data."""

    text = str(value or "")
    text = text.replace("corner_circle_cutouts_v1", "circular corner notches")
    text = re.sub(r"\bcomposition_[A-Za-z0-9_]+\b", "composed feature", text)
    text = re.sub(r"\bcall_[0-9a-fA-F]{8,}\b", "geometry operation", text)
    return text


def _agent() -> AntennaDesignAgent:
    return create_default_agent()


def _tool_plan_payload(plan: LLMToolPlan | None) -> dict[str, object] | None:
    if plan is None:
        return None
    return {
        "schema_version": 1,
        "status": plan.status,
        "message": plan.message,
        "calls": [
            {"name": call.name, "arguments": call.arguments}
            for call in plan.calls
        ],
    }


def _append_planner_audit(path: str | Path | None, event: dict[str, object]) -> None:
    """Append one credential-free planner comparison record."""

    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(event, sort_keys=True, default=str) + "\n"
    with _PLANNER_AUDIT_LOCK:
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(serialized)


def apply_text_instruction(
    state: AntennaDesign | None,
    instruction: str,
    *,
    planner,
    audit_log_path: str | Path | None = None,
    project_memory: ProjectMemory | Mapping[str, object] | None = None,
) -> StateUpdate:
    """Request and execute one schema-constrained LLM tool plan."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise AgentInstructionError("Enter an antenna instruction first.")
    agent = _agent()
    context = {
        "instruction": instruction,
        "current_design": state.to_dict() if state is not None else None,
        "capability_manifest": agent.capability_manifest(state),
    }
    if project_memory is not None:
        context["project_memory"] = (
            project_memory.to_planner_dict()
            if isinstance(project_memory, ProjectMemory)
            else dict(project_memory)
        )
    elif state is None:
        context["project_memory"] = ProjectMemory.empty().to_dict()
    event: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "planner_backend": getattr(planner, "backend_id", planner.__class__.__name__),
        "planner_model": getattr(planner, "model", None),
        "user_request": instruction,
        "design_id": state.design_id if state is not None else None,
        "design_revision": state.revision if state is not None else None,
        "returned_plan": None,
        "validation_result": None,
        "repair_attempt": None,
        "final_executed_tool_sequence": [],
    }
    try:
        llm_plan = planner.plan(**context)
    except Exception as exc:
        planner_metadata = dict(getattr(planner, "last_run_metadata", {}))
        event["planner_metadata"] = planner_metadata
        attempts = planner_metadata.get("returned_plan_attempts", ())
        if isinstance(attempts, list) and attempts:
            event["returned_plan"] = attempts[-1]
        event["validation_result"] = {"status": "planner_error", "error": str(exc)}
        _append_planner_audit(audit_log_path, event)
        raise
    event["returned_plan"] = _tool_plan_payload(llm_plan)
    event["planner_metadata"] = dict(getattr(planner, "last_run_metadata", {}))
    try:
        result = agent.execute_llm_plan(state, llm_plan)
    except (AgentInstructionError, CapabilityError) as exc:
        if llm_plan.status != "execute":
            event["validation_result"] = {"status": llm_plan.status, "error": str(exc)}
            _append_planner_audit(audit_log_path, event)
            raise
        repair = getattr(planner, "repair", None)
        if not callable(repair):
            event["validation_result"] = {"status": "rejected", "error": str(exc)}
            _append_planner_audit(audit_log_path, event)
            raise
        repair_record: dict[str, object] = {"trigger_error": str(exc), "plan": None}
        event["repair_attempt"] = repair_record
        try:
            llm_plan = repair(
                **context,
                failed_plan=llm_plan,
                error=exc,
            )
            repair_record["plan"] = _tool_plan_payload(llm_plan)
            repair_record["planner_metadata"] = dict(getattr(planner, "last_run_metadata", {}))
            result = agent.execute_llm_plan(state, llm_plan)
        except Exception as repair_exc:
            repair_record["result"] = {"status": "rejected", "error": str(repair_exc)}
            event["validation_result"] = {"status": "rejected", "error": str(repair_exc)}
            _append_planner_audit(audit_log_path, event)
            raise
        repair_record["result"] = {"status": "accepted"}
    event["validation_result"] = {"status": "accepted"}
    event["final_plan"] = _tool_plan_payload(llm_plan)
    event["final_executed_tool_sequence"] = [
        {"name": call.name, "arguments": call.arguments}
        for call in result.executed_tools
    ]
    _append_planner_audit(audit_log_path, event)
    return StateUpdate(result.design, result.changes, result.plan, result.executed_tools)


def apply_registered_tool_plan(
    state: AntennaDesign | None,
    plan: LLMToolPlan,
) -> StateUpdate:
    """Execute a previously produced plan through the same deterministic gate."""

    result = _agent().execute_llm_plan(state, plan)
    return StateUpdate(result.design, result.changes, result.plan, result.executed_tools)


def recipe_parameter_definitions(state: AntennaDesign):
    return _agent().parameter_definitions_for_design(state)


def apply_structured_state_patch(
    state: AntennaDesign,
    patch: dict[str, object],
) -> StateUpdate:
    """Validate a strict structured UI/import patch before tools execute."""

    if not isinstance(patch, dict) or not patch:
        raise CapabilityError("The structured patch contains no supported parameter changes.")
    recipe_id = str(patch.get("recipe_id", state.recipe_id))
    changes_payload = patch.get("changes", patch)
    if not isinstance(changes_payload, dict):
        raise CapabilityError("The structured patch changes field must be a JSON object.")
    updates = {
        str(key): value
        for key, value in changes_payload.items()
        if key != "recipe_id"
    }
    agent = _agent()
    if recipe_id != state.recipe_id:
        try:
            recipe = agent.registry.recipe(recipe_id)
        except CapabilityError as exc:
            raise CapabilityUnavailableError(
                f"The structured patch requested an unavailable antenna recipe: {recipe_id}."
            ) from exc
        allowed = {item.key for item in recipe.parameter_definitions()}
        unknown = sorted(set(updates) - allowed)
        if unknown:
            raise CapabilityError(
                "The structured patch requested unsupported fields: " + ", ".join(unknown) + "."
            )
        values = recipe.defaults()
        values.update(updates)
        design, calls = recipe.build(
            agent.registry,
            values,
            design_id=state.design_id,
            revision=state.revision + 1,
        )
        plan = AgentPlan(recipe_id, "AI translated family change", tuple(updates.items()), recipe.required_tools)
        labels = {item.key: item.label for item in recipe.parameter_definitions()}
        changes = (f"antenna family to {recipe.display_name}", *(
            f"{labels[key]} to {value}" for key, value in updates.items()
        ))
        return StateUpdate(design, tuple(changes), plan, calls)
    result = agent.update_parameters(state, updates, intent="AI structured parameter edit")
    return StateUpdate(result.design, result.changes, result.plan, result.executed_tools)


def cst_history(state: AntennaDesign) -> str:
    return CSTAdapter().history(state)


def cst_macro(state: AntennaDesign) -> str:
    return CSTAdapter().macro(state)


def save_cst_package(
    macro_path: str | Path,
    state: AntennaDesign,
    *,
    conversation: Iterable[dict[str, str]] = (),
) -> tuple[Path, Path]:
    destination = Path(macro_path).expanduser()
    if destination.suffix.lower() != ".bas":
        destination = destination.with_suffix(".bas")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(cst_macro(state), encoding="utf-8", newline="\n")
    manifest_path = destination.with_name(destination.stem + "_design.json")
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 2,
            "status": "experimental_generated_starting_design",
            "design": state.to_dict(),
            "derived": {
                "topology": state.topology_label,
                "element_count": state.array.element_count,
                "port_count": len(state.ports),
                "validation": [
                    {"stage": record.stage, "passed": record.passed, "messages": list(record.messages)}
                    for record in state.validation
                ],
            },
            "solver_adapter": {"id": "cst", "macro": destination.name, "solver_started": False},
            "conversation": list(conversation),
        },
    )
    return destination.resolve(), manifest_path.resolve()


def create_native_cst_project(destination: str | Path, state: AntennaDesign) -> Path:
    if sys.platform != "win32":
        raise CapabilityError("Native CST project creation is available on Windows only.")
    path = Path(destination).expanduser()
    if path.suffix.lower() != ".cst":
        path = path.with_suffix(".cst")
    if path.exists():
        raise CapabilityError("Choose a new filename; the builder will not overwrite a CST project.")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise CapabilityError(
            "Native CST creation requires the Windows automation dependency. Run setup_windows.bat and try again."
        ) from exc
    application = None
    model = None
    operation_error: Exception | None = None
    release_error: Exception | None = None
    pythoncom.CoInitialize()
    try:
        # Use an isolated automation server so cleanup never closes a CST
        # session that the user opened independently.
        application = win32com.client.DispatchEx("CSTStudio.Application")
        model = application.NewMWS()
        _populate_native_cst_project(model, state)
        model.SaveAs(str(path.resolve()), True)
    except Exception as exc:
        operation_error = exc
    finally:
        if model is not None:
            try:
                model.Quit()
            except Exception as exc:
                release_error = exc
        model = None
        application = None
        import gc
        gc.collect()
        pythoncom.CoUninitialize()
    if operation_error is not None:
        raise CapabilityError(f"CST could not create the native project: {operation_error}") from operation_error
    if release_error is not None:
        raise CapabilityError(
            "CST saved the project but its automation process could not be released; "
            f"the file may remain locked: {release_error}"
        ) from release_error
    if not path.is_file():
        raise CapabilityError("CST returned without saving the requested project.")
    return path.resolve()


def _populate_native_cst_project(model: object, state: AntennaDesign) -> None:
    """Populate an open CST model with parameters and native history entries."""
    adapter = CSTAdapter()
    for name, value in adapter.parameter_values(state):
        model.StoreParameter(name, value)
    for operation in adapter.history_operations(state):
        model.AddToHistory(operation.name, operation.script)


DESIGN_STATE_RELATIVE_PATH = Path("design") / "antenna_state.json"
DESIGN_CONVERSATION_RELATIVE_PATH = Path("design") / "builder_conversation.json"
PROJECT_MEMORY_RELATIVE_PATH = Path("design") / "project_memory.json"
PROJECT_MEMORY_SCHEMA_VERSION = 1
BUILDER_CONVERSATION_SCHEMA_VERSION = 2
RECENT_CONTEXT_LIMIT = 6
SEMANTIC_MEMORY_ACTIONS = frozenset({"upsert", "supersede", "resolve"})
SEMANTIC_MEMORY_KINDS = frozenset({
    "requirement", "constraint", "preference", "goal", "future_intent", "priority",
})


def _valid_semantic_key(value: Any) -> bool:
    """Validate a compact ASCII snake-case key without interpreting its meaning."""

    return (
        isinstance(value, str)
        and 2 <= len(value) <= 64
        and "a" <= value[0] <= "z"
        and all(
            ("a" <= character <= "z")
            or ("0" <= character <= "9")
            or character == "_"
            for character in value
        )
    )


@dataclass(frozen=True, slots=True)
class CanonicalDesignRef:
    """Identifies the canonical design revision summarized by project memory."""

    design_id: str
    revision: int

    def to_dict(self) -> dict[str, object]:
        return {"design_id": self.design_id, "revision": self.revision}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CanonicalDesignRef":
        design_id = str(payload.get("design_id", "")).strip()
        if not design_id:
            raise CapabilityError("The project-memory canonical design reference is malformed.")
        try:
            revision = int(payload["revision"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CapabilityError(
                "The project-memory canonical design revision is malformed."
            ) from exc
        if revision < 0:
            raise CapabilityError("The project-memory canonical design revision is malformed.")
        return cls(design_id=design_id, revision=revision)


@dataclass(frozen=True, slots=True)
class ProjectMemoryItem:
    """One persisted fact; ``key`` is the canonical semantic identity."""

    item_id: str
    key: str
    value: Any
    unit: str | None = None
    status: str = "active"
    source_turn_id: str | None = None
    design_revision: int | None = None
    source: str = "deterministic"
    semantic_kind: str | None = None
    evidence_quote: str | None = None
    supersedes_item_id: str | None = None
    created_design_revision: int | None = None
    updated_design_revision: int | None = None
    proposed_key: str | None = None
    proposed_value: Any = None
    normalization_rule: str | None = None
    normalization_version: str | None = None
    constraint_operator: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "item_id": self.item_id,
            "key": self.key,
            "value": self.value,
            "unit": self.unit,
            "status": self.status,
            "source_turn_id": self.source_turn_id,
            "design_revision": self.design_revision,
            "source": self.source,
            "semantic_kind": self.semantic_kind,
            "evidence_quote": self.evidence_quote,
            "supersedes_item_id": self.supersedes_item_id,
            "created_design_revision": self.created_design_revision,
            "updated_design_revision": self.updated_design_revision,
        }
        if self.proposed_key is not None:
            payload["proposed_key"] = self.proposed_key
            payload["proposed_value"] = self.proposed_value
            payload["normalization_rule"] = self.normalization_rule
            payload["normalization_version"] = self.normalization_version
        if self.constraint_operator is not None:
            payload["constraint_operator"] = self.constraint_operator
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProjectMemoryItem":
        item_id = str(payload.get("item_id", "")).strip()
        key = str(payload.get("key", "")).strip()
        status = str(payload.get("status", "active")).strip()
        if not item_id or not key or not status or "value" not in payload:
            raise CapabilityError("A project-memory item is malformed.")
        raw_revision = payload.get("design_revision")
        try:
            revision = None if raw_revision is None else int(raw_revision)
        except (TypeError, ValueError) as exc:
            raise CapabilityError("A project-memory design revision is malformed.") from exc
        if revision is not None and revision < 0:
            raise CapabilityError("A project-memory design revision is malformed.")
        raw_unit = payload.get("unit")
        raw_turn = payload.get("source_turn_id")
        raw_created_revision = payload.get("created_design_revision")
        raw_updated_revision = payload.get("updated_design_revision")
        try:
            created_revision = (
                None if raw_created_revision is None else int(raw_created_revision)
            )
            updated_revision = (
                None if raw_updated_revision is None else int(raw_updated_revision)
            )
        except (TypeError, ValueError) as exc:
            raise CapabilityError("A project-memory provenance revision is malformed.") from exc
        if any(
            value is not None and value < 0
            for value in (created_revision, updated_revision)
        ):
            raise CapabilityError("A project-memory provenance revision is malformed.")
        return cls(
            item_id=item_id,
            key=key,
            value=payload["value"],
            unit=None if raw_unit is None else str(raw_unit),
            status=status,
            source_turn_id=None if raw_turn is None else str(raw_turn),
            design_revision=revision,
            source=str(payload.get("source", "deterministic")),
            semantic_kind=(
                None if payload.get("semantic_kind") is None
                else str(payload["semantic_kind"])
            ),
            evidence_quote=(
                None if payload.get("evidence_quote") is None
                else str(payload["evidence_quote"])
            ),
            supersedes_item_id=(
                None if payload.get("supersedes_item_id") is None
                else str(payload["supersedes_item_id"])
            ),
            created_design_revision=created_revision,
            updated_design_revision=updated_revision,
            proposed_key=(
                None if payload.get("proposed_key") is None
                else str(payload["proposed_key"])
            ),
            proposed_value=payload.get("proposed_value"),
            normalization_rule=(
                None if payload.get("normalization_rule") is None
                else str(payload["normalization_rule"])
            ),
            normalization_version=(
                None if payload.get("normalization_version") is None
                else str(payload["normalization_version"])
            ),
            constraint_operator=(
                None if payload.get("constraint_operator") is None
                else str(payload["constraint_operator"])
            ),
        )


_PROJECT_MEMORY_COLLECTIONS = (
    "requirements",
    "decisions",
    "assumptions",
    "limitations",
    "open_questions",
    "important_changes",
    "recent_context",
)


@dataclass(frozen=True, slots=True)
class ProjectMemory:
    """Compact project-level engineering context, separate from canonical design."""

    schema_version: int = PROJECT_MEMORY_SCHEMA_VERSION
    canonical_ref: CanonicalDesignRef | None = None
    requirements: tuple[ProjectMemoryItem, ...] = ()
    decisions: tuple[ProjectMemoryItem, ...] = ()
    assumptions: tuple[ProjectMemoryItem, ...] = ()
    limitations: tuple[ProjectMemoryItem, ...] = ()
    open_questions: tuple[ProjectMemoryItem, ...] = ()
    important_changes: tuple[ProjectMemoryItem, ...] = ()
    recent_context: tuple[ProjectMemoryItem, ...] = ()

    @classmethod
    def empty(
        cls,
        canonical_ref: CanonicalDesignRef | None = None,
    ) -> "ProjectMemory":
        return cls(canonical_ref=canonical_ref)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "canonical_ref": self.canonical_ref.to_dict() if self.canonical_ref else None,
        }
        for name in _PROJECT_MEMORY_COLLECTIONS:
            payload[name] = [item.to_dict() for item in getattr(self, name)]
        return payload

    def to_planner_dict(self) -> dict[str, object]:
        """Expose current intent compactly while keeping canonical state separate."""

        payload = self.to_dict()
        active_by_identity: dict[str, dict[str, object]] = {}
        for collection in ("requirements", "decisions"):
            deterministic_items = []
            for item in getattr(self, collection):
                if item.source == "user_semantic":
                    if item.status == "active":
                        normalized = normalize_semantic_proposal(
                            key=item.key,
                            value=item.value,
                            unit=item.unit,
                            semantic_kind=item.semantic_kind or "preference",
                            evidence_quote=item.evidence_quote or "",
                        )
                        semantic_key = normalized.semantic_key
                        active_by_identity[semantic_key] = {
                            "item_id": item.item_id,
                            "semantic_kind": item.semantic_kind,
                            "key": semantic_key,
                            "value": normalized.canonical_value,
                            "unit": normalized.unit,
                            "constraint_operator": (
                                item.constraint_operator
                                or normalized.constraint_operator
                            ),
                            "status": item.status,
                            "source": item.source,
                            "evidence_quote": item.evidence_quote,
                        }
                    continue
                deterministic_items.append(item.to_dict())
            payload[collection] = deterministic_items
        if active_by_identity:
            payload["project_intent"] = list(active_by_identity.values())
            payload["context_separation"] = {
                "canonical_current_state": (
                    "Authoritative physical state is supplied separately in current_design."
                ),
                "project_intent": (
                    "User-grounded requirements, preferences, and future goals; these do not prove physical realization."
                ),
            }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProjectMemory":
        try:
            schema_version = int(payload.get("schema_version", 0))
        except (TypeError, ValueError) as exc:
            raise CapabilityError("The saved project memory is malformed.") from exc
        if schema_version != PROJECT_MEMORY_SCHEMA_VERSION:
            raise CapabilityError("Unsupported project-memory schema version.")

        raw_ref = payload.get("canonical_ref")
        if raw_ref is None:
            canonical_ref = None
        elif isinstance(raw_ref, Mapping):
            canonical_ref = CanonicalDesignRef.from_dict(raw_ref)
        else:
            raise CapabilityError("The project-memory canonical reference is malformed.")

        collections: dict[str, tuple[ProjectMemoryItem, ...]] = {}
        for name in _PROJECT_MEMORY_COLLECTIONS:
            raw_items = payload.get(name, [])
            if not isinstance(raw_items, list):
                raise CapabilityError(f"The project-memory {name} collection is malformed.")
            parsed: list[ProjectMemoryItem] = []
            for item in raw_items:
                if not isinstance(item, Mapping):
                    raise CapabilityError(f"The project-memory {name} collection is malformed.")
                parsed.append(ProjectMemoryItem.from_dict(item))
            collections[name] = tuple(parsed)
        return cls(
            schema_version=schema_version,
            canonical_ref=canonical_ref,
            **collections,
        )


@dataclass(frozen=True, slots=True)
class ProjectConstraintEvaluation:
    """Deterministic comparison of one active memory constraint with canonical state."""

    item_id: str
    key: str
    label: str
    operator: str | None
    limit: Any
    unit: str | None
    status: str
    actual: float | None = None
    actual_unit: str | None = None
    message: str = ""


def _length_in_mm(value: Any, unit: str | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    factor = {"mm": 1.0, "cm": 10.0, "m": 1000.0}.get(unit or "mm")
    return None if factor is None else float(value) * factor


def active_project_constraints(memory: ProjectMemory) -> tuple[ProjectMemoryItem, ...]:
    """Return active, user-grounded constraints without interpreting conversation text."""

    return tuple(
        item
        for collection in (memory.requirements, memory.decisions)
        for item in collection
        if item.source == "user_semantic"
        and item.semantic_kind == "constraint"
        and item.status == "active"
    )


def evaluate_project_constraints(
    memory: ProjectMemory,
    design: AntennaDesign | None,
) -> tuple[ProjectConstraintEvaluation, ...]:
    """Evaluate registered constraint identities against one canonical candidate."""

    results: list[ProjectConstraintEvaluation] = []
    for item in active_project_constraints(memory):
        normalized = normalize_semantic_proposal(
            key=item.key,
            value=item.value,
            unit=item.unit,
            semantic_kind="constraint",
            evidence_quote=item.evidence_quote or "",
        )
        key = normalized.semantic_key
        operator = item.constraint_operator or normalized.constraint_operator
        if design is None:
            results.append(ProjectConstraintEvaluation(
                item.item_id, key, key.replace("_", " ").title(), operator,
                normalized.canonical_value, normalized.unit, "not_applicable",
                message="No canonical antenna design exists yet.",
            ))
            continue
        if key != "board_width_limit":
            results.append(ProjectConstraintEvaluation(
                item.item_id, key, key.replace("_", " ").title(), operator,
                normalized.canonical_value, normalized.unit, "unevaluated",
                message="No deterministic evaluator is registered for this constraint.",
            ))
            continue
        limit_mm = _length_in_mm(normalized.canonical_value, normalized.unit)
        actual_mm = float(design.board_width_mm)
        if limit_mm is None or operator != "max":
            results.append(ProjectConstraintEvaluation(
                item.item_id, key, "Board width", operator,
                normalized.canonical_value, normalized.unit, "unevaluated",
                actual=actual_mm, actual_unit="mm",
                message="The board-width constraint has an unsupported value, unit, or operator.",
            ))
            continue
        violating = actual_mm > limit_mm + 1e-9
        status = "violating" if violating else "satisfied"
        relation = "exceeds" if violating else "is within"
        results.append(ProjectConstraintEvaluation(
            item.item_id, key, "Board width", "max", limit_mm, "mm", status,
            actual=actual_mm, actual_unit="mm",
            message=(
                f"Board width {actual_mm:.6g} mm {relation} the active maximum "
                f"of {limit_mm:.6g} mm."
            ),
        ))
    return tuple(results)


def _constraint_disclosure(
    evaluations: tuple[ProjectConstraintEvaluation, ...],
) -> str | None:
    violations = [item.message for item in evaluations if item.status == "violating"]
    if not violations:
        return None
    return "Constraint violation: " + " ".join(violations)


def _with_disclosure(message: str, disclosure: str | None) -> str:
    if not disclosure:
        return message
    base = message.strip()
    return f"{base}\n\n{disclosure}" if base else disclosure


@dataclass(slots=True)
class BuilderProjectSession:
    """Persisted state for one project's experimental antenna builder."""

    design: AntennaDesign | None = None
    conversation: list[dict[str, object]] = field(default_factory=list)
    memory: ProjectMemory = field(default_factory=ProjectMemory.empty)


@dataclass(frozen=True, slots=True)
class BuilderInteractionOutcome:
    """Structured result consumed by the deterministic project-memory reducer."""

    status: str
    message: str
    instruction: str = ""
    interaction_kind: str = "conversation"
    planner_calls: tuple[PlannedToolCall, ...] = ()
    explicit_parameter_updates: tuple[tuple[str, Any], ...] = ()
    changes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BuilderTurnResult:
    """Published session and state update for one terminal builder outcome."""

    session: BuilderProjectSession
    update: StateUpdate
    turn_id: str
    terminal_result: AgentTerminalResult | None = None


def _memory_item(
    *,
    item_id: str,
    key: str,
    value: Any,
    turn_id: str,
    design: AntennaDesign | None,
    unit: str | None = None,
    status: str = "active",
) -> ProjectMemoryItem:
    return ProjectMemoryItem(
        item_id=item_id,
        key=key,
        value=value,
        unit=unit,
        status=status,
        source_turn_id=turn_id,
        design_revision=design.revision if design is not None else None,
    )


def _upsert_memory_item(
    items: tuple[ProjectMemoryItem, ...],
    item: ProjectMemoryItem,
) -> tuple[ProjectMemoryItem, ...]:
    for index, current in enumerate(items):
        if current.item_id == item.item_id:
            return (*items[:index], item, *items[index + 1:])
    return (*items, item)


def _set_memory_item_status(
    items: tuple[ProjectMemoryItem, ...],
    item_id: str,
    status: str,
) -> tuple[ProjectMemoryItem, ...]:
    return tuple(
        replace(item, status=status) if item.item_id == item_id else item
        for item in items
    )


def _semantic_collection(semantic_kind: str) -> str:
    return (
        "requirements"
        if semantic_kind in {"requirement", "constraint", "goal"}
        else "decisions"
    )


def _semantic_scalar(value: Any) -> Any:
    if isinstance(value, str):
        characters: list[str] = []
        for character in value.casefold():
            if ("a" <= character <= "z") or ("0" <= character <= "9"):
                characters.append(character)
            elif characters and characters[-1] != "_":
                characters.append("_")
        return "".join(characters).strip("_")
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return object()


def _planner_call_scalars(calls: tuple[PlannedToolCall, ...]) -> set[Any]:
    values: set[Any] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, (str, int, float, bool)) and not (
            isinstance(value, float) and not math.isfinite(value)
        ):
            values.add(_semantic_scalar(value))

    for call in calls:
        visit(call.arguments)
    return values


def _value_is_grounded_in_quote(value: Any, quote: str) -> bool:
    """Check lexical grounding without interpreting domain meaning or prose intent."""

    normalized_quote = _semantic_scalar(quote)
    if isinstance(value, str):
        normalized_value = _semantic_scalar(value)
        return bool(normalized_value) and normalized_value in normalized_quote
    if isinstance(value, bool):
        return str(value).casefold() in quote.casefold()
    if isinstance(value, (int, float)):
        numeric_text = "".join(
            character if character.isdigit() or character in ".+-eE" else " "
            for character in quote
        )
        for token in numeric_text.split():
            try:
                if float(token) == float(value):
                    return True
            except ValueError:
                continue
        return False
    return False


def apply_semantic_memory_proposals(
    previous: ProjectMemory,
    proposals: tuple[SemanticMemoryProposal, ...],
    *,
    instruction: str,
    planner_calls: tuple[PlannedToolCall, ...],
    design: AntennaDesign | None,
    turn_id: str,
) -> tuple[ProjectMemory, dict[str, object]]:
    """Validate and apply user-grounded semantic intent without touching design state."""

    collections = {
        "requirements": list(previous.requirements),
        "decisions": list(previous.decisions),
    }
    audit: dict[str, object] = {
        "proposed": [item.to_dict() for item in proposals],
        "normalized": [],
        "applied_ids": [],
        "superseded_ids": [],
        "resolved_ids": [],
        "rejected": [],
    }
    seen_keys: set[str] = set()
    call_scalars = _planner_call_scalars(planner_calls)
    revision = design.revision if design is not None else None

    def reject(index: int, proposal: SemanticMemoryProposal, reason: str) -> None:
        audit["rejected"].append({
            "index": index,
            "proposal": proposal.to_dict(),
            "reason": reason,
        })

    def active_matches(
        key: str,
        reference_id: str | None,
        *,
        value: Any,
        unit: str | None,
        semantic_kind: str,
    ):
        matches = []
        for collection_name, items in collections.items():
            for item_index, item in enumerate(items):
                identity = canonical_semantic_identity(
                    key=item.key,
                    value=item.value,
                    unit=item.unit,
                    semantic_kind=item.semantic_kind,
                    evidence_quote=item.evidence_quote,
                )
                if (
                    item.source == "user_semantic"
                    and item.status == "active"
                    and identity == key
                ):
                    matches.append((collection_name, item_index, item))
        if reference_id is None or not matches:
            return matches
        exact = [match for match in matches if match[2].item_id == reference_id]
        if exact:
            return exact
        reference_identity = canonical_reference_identity(
            reference_id,
            value=value,
            unit=unit,
            semantic_kind=semantic_kind,
        )
        return matches if reference_identity == key else []

    for index, proposal in enumerate(proposals):
        action = proposal.action
        semantic_kind = proposal.semantic_kind
        key = proposal.key
        quote = proposal.evidence_quote
        unit = proposal.unit
        reference_id = proposal.reference_id
        value = proposal.value

        if action not in SEMANTIC_MEMORY_ACTIONS:
            reject(index, proposal, "unsupported_action")
            continue
        if semantic_kind not in SEMANTIC_MEMORY_KINDS:
            reject(index, proposal, "unsupported_semantic_kind")
            continue
        if not _valid_semantic_key(key):
            reject(index, proposal, "invalid_semantic_key")
            continue
        if (
            not isinstance(quote, str)
            or not quote
            or len(quote) > 400
            or quote not in instruction
        ):
            reject(index, proposal, "evidence_quote_not_exact_current_turn_substring")
            continue
        if unit is not None and (not isinstance(unit, str) or not unit or len(unit) > 32):
            reject(index, proposal, "invalid_unit")
            continue
        if reference_id is not None and (
            not isinstance(reference_id, str) or not reference_id or len(reference_id) > 160
        ):
            reject(index, proposal, "invalid_reference_id")
            continue
        if action == "resolve":
            if value is not None:
                reject(index, proposal, "resolve_value_must_be_null")
                continue
        elif isinstance(value, bool):
            reject(index, proposal, "boolean_placeholder_not_allowed")
            continue
        elif (
            value is None
            or isinstance(value, (dict, list, tuple))
            or (isinstance(value, str) and (not value or len(value) > 300))
            or (isinstance(value, float) and not math.isfinite(value))
            or not isinstance(value, (str, int, float))
        ):
            reject(index, proposal, "invalid_value")
            continue
        if action != "resolve" and not _value_is_grounded_in_quote(value, quote):
            reject(index, proposal, "value_not_lexically_grounded_in_evidence_quote")
            continue
        if action != "resolve" and _semantic_scalar(value) in call_scalars:
            reject(index, proposal, "duplicates_current_canonical_action")
            continue

        normalized = normalize_semantic_proposal(
            key=key,
            value=value,
            unit=unit,
            semantic_kind=semantic_kind,
            evidence_quote=quote,
        )
        key = normalized.semantic_key
        value = normalized.canonical_value
        unit = normalized.unit
        audit["normalized"].append({
            "index": index,
            "proposed_key": normalized.proposed_key,
            "proposed_value": normalized.proposed_value,
            "semantic_key": key,
            "canonical_value": value,
            "unit": unit,
            "constraint_operator": normalized.constraint_operator,
            "normalization_rule": normalized.normalization_rule,
            "normalization_version": SEMANTIC_NORMALIZATION_VERSION,
        })
        if key in seen_keys:
            reject(index, proposal, "duplicate_key_in_turn")
            continue
        seen_keys.add(key)
        if (
            key == "feed_network_future_goal"
            and design is not None
            and value == design.excitation.strategy
        ):
            reject(index, proposal, "duplicates_current_canonical_state")
            continue

        matches = active_matches(
            key,
            reference_id,
            value=value,
            unit=unit,
            semantic_kind=semantic_kind,
        )
        if action == "upsert" and matches:
            current = matches[0][2]
            current_normalized = normalize_semantic_proposal(
                key=current.key,
                value=current.value,
                unit=current.unit,
                semantic_kind=current.semantic_kind or semantic_kind,
                evidence_quote=current.evidence_quote or "",
            )
            reason = (
                "duplicate_active_value"
                if current.semantic_kind == semantic_kind
                and current_normalized.canonical_value == value
                and current_normalized.unit == unit
                and (
                    current.constraint_operator
                    or current_normalized.constraint_operator
                ) == normalized.constraint_operator
                else "active_key_requires_supersede"
            )
            reject(index, proposal, reason)
            continue
        if action in {"supersede", "resolve"} and len(matches) != 1:
            reject(index, proposal, "active_reference_not_unique_or_missing")
            continue

        target_id = None
        if matches:
            collection_name, item_index, current = matches[0]
            if action == "resolve" and current.semantic_kind != semantic_kind:
                reject(index, proposal, "semantic_kind_does_not_match_active_item")
                continue
            target_id = current.item_id
            new_status = "superseded" if action == "supersede" else "resolved"
            collections[collection_name][item_index] = replace(
                current,
                status=new_status,
                updated_design_revision=revision,
            )
            audit[f"{new_status}_ids"].append(target_id)

        suffix = "resolution" if action == "resolve" else "value"
        item_id = f"semantic:{key}:{turn_id}:{suffix}"
        item = ProjectMemoryItem(
            item_id=item_id,
            key=key,
            value=(matches[0][2].value if action == "resolve" else value),
            unit=(matches[0][2].unit if action == "resolve" else unit),
            status=("resolved" if action == "resolve" else "active"),
            source_turn_id=turn_id,
            design_revision=revision,
            source="user_semantic",
            semantic_kind=semantic_kind,
            evidence_quote=quote,
            supersedes_item_id=target_id,
            created_design_revision=revision,
            updated_design_revision=revision,
            proposed_key=normalized.proposed_key,
            proposed_value=normalized.proposed_value,
            normalization_rule=normalized.normalization_rule,
            normalization_version=SEMANTIC_NORMALIZATION_VERSION,
            constraint_operator=normalized.constraint_operator,
        )
        collections[_semantic_collection(semantic_kind)].append(item)
        audit["applied_ids"].append(item_id)

    return replace(
        previous,
        requirements=tuple(collections["requirements"]),
        decisions=tuple(collections["decisions"]),
    ), audit


def _parameter_value(design: AntennaDesign, key: str, fallback: Any) -> tuple[Any, str | None]:
    if key == "material":
        return design.material, None
    parameter = design.parameter_map().get(key)
    if parameter is None:
        return fallback, None
    return parameter.value, parameter.unit or None


def _composed_feature_delta(
    before: AntennaDesign | None,
    after: AntennaDesign,
) -> dict[str, object] | None:
    before_groups = {
        group.group_id: group for group in (before.composed_operations if before else ())
    }
    after_groups = {group.group_id: group for group in after.composed_operations}
    added = [
        {
            "group_id": group_id,
            "target_role": group.target_selector.role if group.target_selector else None,
            "scope": group.target_selector.scope if group.target_selector else "single",
            "operations": [call.name for call in group.calls],
        }
        for group_id, group in after_groups.items()
        if group_id not in before_groups
    ]
    removed = sorted(set(before_groups) - set(after_groups))
    updated = [
        {
            "group_id": group_id,
            "target_role": group.target_selector.role if group.target_selector else None,
            "scope": group.target_selector.scope if group.target_selector else "single",
            "operations": [call.name for call in group.calls],
        }
        for group_id, group in after_groups.items()
        if group_id in before_groups and group != before_groups[group_id]
    ]
    if not (added or removed or updated):
        return None
    return {"added": added, "updated": updated, "removed": removed}


class ProjectMemoryReducer:
    """Reduce typed builder outcomes into compact, deterministic engineering memory."""

    def reduce(
        self,
        previous: ProjectMemory,
        previous_design: AntennaDesign | None,
        outcome: BuilderInteractionOutcome,
        resulting_design: AntennaDesign | None,
        turn_id: str,
    ) -> ProjectMemory:
        canonical_ref = previous.canonical_ref
        requirements = previous.requirements
        decisions = previous.decisions
        assumptions = previous.assumptions
        limitations = previous.limitations
        open_questions = previous.open_questions
        important_changes = previous.important_changes

        if outcome.status == "executed":
            if resulting_design is None:
                raise CapabilityError("A successful builder outcome must publish a canonical design.")
            canonical_ref = CanonicalDesignRef(
                resulting_design.design_id,
                resulting_design.revision,
            )
            call_names = [call.name for call in outcome.planner_calls]
            selected_recipe = next(
                (
                    str(call.arguments["recipe_id"])
                    for call in outcome.planner_calls
                    if call.name == "recipe.select" and "recipe_id" in call.arguments
                ),
                None,
            )
            if selected_recipe is not None or (
                previous_design is not None
                and previous_design.recipe_id != resulting_design.recipe_id
            ):
                decisions = _upsert_memory_item(
                    decisions,
                    _memory_item(
                        item_id="decision:antenna_family",
                        key="antenna_family",
                        value={
                            "recipe_id": resulting_design.recipe_id,
                            "family": resulting_design.family,
                        },
                        turn_id=turn_id,
                        design=resulting_design,
                    ),
                )

            explicit_updates = dict(outcome.explicit_parameter_updates)
            for call in outcome.planner_calls:
                if call.name == "parameter.set":
                    explicit_updates[str(call.arguments["key"])] = call.arguments["value"]

            if "frequency_ghz" in explicit_updates:
                value, unit = _parameter_value(
                    resulting_design,
                    "frequency_ghz",
                    explicit_updates["frequency_ghz"],
                )
                requirements = _upsert_memory_item(
                    requirements,
                    _memory_item(
                        item_id="requirement:target_frequency",
                        key="target_frequency",
                        value=value,
                        unit=unit or "GHz",
                        turn_id=turn_id,
                        design=resulting_design,
                    ),
                )
                assumptions = _set_memory_item_status(
                    assumptions, "assumption:default_frequency", "superseded"
                )
            if "material" in explicit_updates:
                value, _unit = _parameter_value(
                    resulting_design,
                    "material",
                    explicit_updates["material"],
                )
                decisions = _upsert_memory_item(
                    decisions,
                    _memory_item(
                        item_id="decision:substrate_material",
                        key="substrate_material",
                        value=value,
                        turn_id=turn_id,
                        design=resulting_design,
                    ),
                )
                assumptions = _set_memory_item_status(
                    assumptions, "assumption:default_material", "superseded"
                )

            uses_recipe_defaults = selected_recipe is not None or "design.reset" in call_names
            if uses_recipe_defaults:
                if "frequency_ghz" not in explicit_updates:
                    requirements = _set_memory_item_status(
                        requirements, "requirement:target_frequency", "superseded"
                    )
                    value, unit = _parameter_value(resulting_design, "frequency_ghz", None)
                    assumptions = _upsert_memory_item(
                        assumptions,
                        _memory_item(
                            item_id="assumption:default_frequency",
                            key="default_frequency",
                            value=value,
                            unit=unit or "GHz",
                            turn_id=turn_id,
                            design=resulting_design,
                        ),
                    )
                if "material" not in explicit_updates:
                    decisions = _set_memory_item_status(
                        decisions, "decision:substrate_material", "superseded"
                    )
                    if resulting_design.material != "No substrate":
                        assumptions = _upsert_memory_item(
                            assumptions,
                            _memory_item(
                                item_id="assumption:default_material",
                                key="default_substrate_material",
                                value=resulting_design.material,
                                turn_id=turn_id,
                                design=resulting_design,
                            ),
                        )
                    else:
                        assumptions = _set_memory_item_status(
                            assumptions, "assumption:default_material", "superseded"
                        )

            change_value: dict[str, object] = {}
            ordinary_parameter_changes = []
            for key, fallback in explicit_updates.items():
                if key in {"frequency_ghz", "material"}:
                    continue
                value, unit = _parameter_value(resulting_design, key, fallback)
                ordinary_parameter_changes.append({"key": key, "value": value, "unit": unit})
            if ordinary_parameter_changes:
                change_value["parameters"] = ordinary_parameter_changes

            created_parameters = [
                {
                    "key": str(call.arguments["key"]),
                    "value": call.arguments["value"],
                    "unit": call.arguments.get("unit") or None,
                }
                for call in outcome.planner_calls
                if call.name == "parameter.create"
            ]
            if created_parameters:
                change_value["created_parameters"] = created_parameters

            if previous_design is None or previous_design.array != resulting_design.array:
                if any(key.startswith("array_") or key == "element_spacing_lambda" for key in explicit_updates):
                    change_value["array"] = {
                        "rows": resulting_design.array.rows,
                        "columns": resulting_design.array.columns,
                        "spacing_mm": resulting_design.array.spacing_mm,
                    }

            composed_delta = _composed_feature_delta(previous_design, resulting_design)
            if composed_delta is not None:
                change_value["composed_features"] = composed_delta

            modifier_calls = [
                {"action": call.name, "modifier_id": call.arguments.get("modifier_id")}
                for call in outcome.planner_calls
                if call.name in {"modifier.apply", "modifier.remove"}
            ]
            if modifier_calls:
                change_value["modifiers"] = modifier_calls
            if "design.reset" in call_names:
                change_value["reset"] = True
            if change_value:
                important_changes = _upsert_memory_item(
                    important_changes,
                    _memory_item(
                        item_id=f"change:{turn_id}",
                        key="design_change",
                        value=change_value,
                        turn_id=turn_id,
                        design=resulting_design,
                    ),
                )
        elif outcome.status == "refusal":
            limitations = _upsert_memory_item(
                limitations,
                _memory_item(
                    item_id=f"limitation:{turn_id}",
                    key="unsupported_capability",
                    value=outcome.message,
                    turn_id=turn_id,
                    design=previous_design,
                ),
            )
        elif outcome.status == "clarification":
            open_questions = _upsert_memory_item(
                open_questions,
                _memory_item(
                    item_id=f"question:{turn_id}",
                    key="planner_clarification",
                    value=outcome.message,
                    turn_id=turn_id,
                    design=previous_design,
                ),
            )

        recent_item = _memory_item(
            item_id=f"context:{turn_id}",
            key="builder_turn",
            value={
                "outcome": outcome.status,
                "request": outcome.instruction,
                "response": outcome.message,
            },
            turn_id=turn_id,
            design=resulting_design if outcome.status == "executed" else previous_design,
        )
        recent_context = (*previous.recent_context, recent_item)[-RECENT_CONTEXT_LIMIT:]
        return ProjectMemory(
            schema_version=previous.schema_version,
            canonical_ref=canonical_ref,
            requirements=requirements,
            decisions=decisions,
            assumptions=assumptions,
            limitations=limitations,
            open_questions=open_questions,
            important_changes=important_changes,
            recent_context=recent_context,
        )


def _load_optional_project_design(root: Path) -> AntennaDesign | None:
    state_path = root / DESIGN_STATE_RELATIVE_PATH
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CapabilityError("The saved antenna design is malformed.")
    if isinstance(payload.get("design"), dict):
        return AntennaDesign.from_dict(payload["design"])
    if isinstance(payload.get("state"), dict):
        return _migrate_legacy_state(payload["state"])
    raise CapabilityError("The saved antenna design is malformed.")


def _load_builder_conversation(root: Path) -> list[dict[str, object]]:
    conversation_path = root / DESIGN_CONVERSATION_RELATIVE_PATH
    if not conversation_path.exists():
        return []
    payload = json.loads(conversation_path.read_text(encoding="utf-8"))
    raw_messages = payload.get("messages", []) if isinstance(payload, dict) else []
    if not isinstance(raw_messages, list):
        return []
    messages: list[dict[str, object]] = []
    for item in raw_messages:
        if (
            isinstance(item, dict)
            and item.get("role") in {"user", "builder"}
            and item.get("content")
        ):
            messages.append({
                **item,
                "role": str(item["role"]),
                "content": str(item["content"]),
            })
    return messages


def save_builder_session(
    project_path: str | Path,
    session: BuilderProjectSession,
) -> tuple[Path | None, Path, Path]:
    """Persist the optional design, raw conversation, and engineering memory."""

    root = Path(project_path)
    state_path = root / DESIGN_STATE_RELATIVE_PATH
    conversation_path = root / DESIGN_CONVERSATION_RELATIVE_PATH
    memory_path = root / PROJECT_MEMORY_RELATIVE_PATH
    if session.design is None:
        state_path.unlink(missing_ok=True)
        saved_state_path: Path | None = None
    else:
        atomic_write_json(
            state_path,
            {"schema_version": 2, "status": "experimental", "design": session.design.to_dict()},
        )
        saved_state_path = state_path
    atomic_write_json(
        conversation_path,
        {
            "schema_version": BUILDER_CONVERSATION_SCHEMA_VERSION,
            "messages": list(session.conversation),
        },
    )
    atomic_write_json(memory_path, session.memory.to_dict())
    return saved_state_path, conversation_path, memory_path


def load_builder_session(project_path: str | Path) -> BuilderProjectSession:
    """Load a project without synthesizing an antenna when no design was saved."""

    root = Path(project_path)
    design = _load_optional_project_design(root)
    conversation = _load_builder_conversation(root)
    memory_path = root / PROJECT_MEMORY_RELATIVE_PATH
    if memory_path.exists():
        payload = json.loads(memory_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise CapabilityError("The saved project memory is malformed.")
        memory = ProjectMemory.from_dict(payload)
    else:
        canonical_ref = (
            CanonicalDesignRef(design.design_id, design.revision)
            if design is not None
            else None
        )
        memory = ProjectMemory.empty(canonical_ref=canonical_ref)
    refused_turn_ids = {
        str(record.get("turn_id"))
        for record in conversation
        if record.get("role") == "builder"
        and record.get("outcome") == "refusal"
        and record.get("turn_id")
    }
    if refused_turn_ids:
        def retire_refused(items: tuple[ProjectMemoryItem, ...]) -> tuple[ProjectMemoryItem, ...]:
            return tuple(
                replace(item, status="requested_unsupported")
                if item.source == "user_semantic"
                and item.status == "active"
                and item.source_turn_id in refused_turn_ids
                else item
                for item in items
            )

        memory = replace(
            memory,
            requirements=retire_refused(memory.requirements),
            decisions=retire_refused(memory.decisions),
        )
    return BuilderProjectSession(
        design=design,
        conversation=conversation,
        memory=memory,
    )


def _conversation_turn_records(
    *,
    turn_id: str,
    instruction: str,
    response: str,
    outcome: str,
    previous_design: AntennaDesign | None,
    resulting_design: AntennaDesign | None,
) -> tuple[dict[str, object], dict[str, object]]:
    created_at = datetime.now(timezone.utc).isoformat()
    return (
        {
            "turn_id": turn_id,
            "role": "user",
            "content": instruction,
            "created_at": created_at,
            "outcome": "request",
            "design_revision": (
                previous_design.revision if previous_design is not None else None
            ),
        },
        {
            "turn_id": turn_id,
            "role": "builder",
            "content": response,
            "created_at": created_at,
            "outcome": outcome,
            "design_revision": (
                resulting_design.revision if resulting_design is not None else None
            ),
        },
    )


def _publish_builder_outcome(
    project_path: str | Path,
    session: BuilderProjectSession,
    outcome: BuilderInteractionOutcome,
    *,
    resulting_design: AntennaDesign | None,
    turn_id: str,
    memory_proposals: tuple[SemanticMemoryProposal, ...] = (),
    planner_calls: tuple[PlannedToolCall, ...] = (),
) -> tuple[BuilderProjectSession, dict[str, object]]:
    """Persist one complete outcome before exposing it as the next live session."""

    previous_design = session.design
    memory = ProjectMemoryReducer().reduce(
        session.memory,
        previous_design,
        outcome,
        resulting_design,
        turn_id,
    )
    memory, semantic_audit = apply_semantic_memory_proposals(
        memory,
        memory_proposals,
        instruction=outcome.instruction,
        planner_calls=planner_calls,
        design=(resulting_design if outcome.status == "executed" else previous_design),
        turn_id=turn_id,
    )
    records = _conversation_turn_records(
        turn_id=turn_id,
        instruction=outcome.instruction,
        response=outcome.message,
        outcome=outcome.status,
        previous_design=previous_design,
        resulting_design=(resulting_design if outcome.status == "executed" else previous_design),
    )
    candidate = BuilderProjectSession(
        design=(resulting_design if outcome.status == "executed" else previous_design),
        conversation=[*session.conversation, *records],
        memory=memory,
    )
    save_builder_session(project_path, candidate)
    session.design = candidate.design
    session.conversation = candidate.conversation
    session.memory = candidate.memory
    return session, semantic_audit


class _LegacyOneShotAgentPlanner:
    """Keep old test/utility planners usable at the new conversational boundary."""

    def __init__(self, planner: Any) -> None:
        self.planner = planner
        self.backend_id = getattr(planner, "backend_id", planner.__class__.__name__)
        self.model = getattr(planner, "model", None)
        self.last_run_metadata: dict[str, Any] = {}
        self._executed = False

    def plan_agent_step(self, **request: Any) -> AgentStep:
        observation = request.get("agent_observation")
        if self._executed and getattr(observation, "outcome", None) == "accepted":
            self.last_run_metadata = dict(getattr(self.planner, "last_run_metadata", {}))
            return AgentStep("finish", "The requested antenna update is complete.")
        self._executed = False
        plan = self.planner.plan(
            instruction=request["instruction"],
            current_design=request["current_design"],
            capability_manifest=request["capability_manifest"],
            project_memory=request.get("project_memory"),
            agent_observation=request.get("agent_observation"),
        )
        self.last_run_metadata = dict(getattr(self.planner, "last_run_metadata", {}))
        if not isinstance(plan, LLMToolPlan):
            return plan
        if plan.status == "execute":
            self._executed = True
        return AgentStep(plan.status, plan.message, plan.calls)


def _agent_planner(planner: Any) -> Any:
    return planner if callable(getattr(planner, "plan_agent_step", None)) else _LegacyOneShotAgentPlanner(planner)


def _design_audit_ref(design: AntennaDesign | None) -> dict[str, object] | None:
    if design is None:
        return None
    return {
        "design_id": design.design_id,
        "revision": design.revision,
        "semantic_hash": semantic_design_hash(design),
    }


def _append_agent_turn_audit(
    path: str | Path | None,
    *,
    turn_id: str,
    instruction: str,
    planner: Any,
    budgets: AgentLoopBudgets,
    baseline_design: AntennaDesign | None,
    terminal: AgentTerminalResult,
    published_design: AntennaDesign | None,
    semantic_memory_audit: Mapping[str, object],
) -> None:
    executed = [
        {"name": call.name, "arguments": dict(call.arguments)}
        for entry in terminal.trajectory
        for call in entry.executed_calls
    ]
    _append_planner_audit(path, {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "turn_id": turn_id,
        "planner_backend": getattr(planner, "backend_id", planner.__class__.__name__),
        "planner_model": getattr(planner, "model", None),
        "user_request": instruction,
        "baseline_design_ref": _design_audit_ref(baseline_design),
        "configured_budgets": budgets.to_dict(),
        "trajectory": [entry.to_dict() for entry in terminal.trajectory],
        "terminal_status": terminal.outcome,
        "terminal_message": terminal.message,
        "published_design_ref": _design_audit_ref(published_design),
        "intermediate_changes_discarded": (
            terminal.outcome != "finished"
            and any(entry.execution_status == "accepted" for entry in terminal.trajectory)
        ),
        "aggregate_executed_tools": executed,
        "semantic_memory": dict(semantic_memory_audit),
    })


def _published_agent_design(
    baseline: AntennaDesign | None,
    terminal: AgentTerminalResult,
) -> AntennaDesign:
    if terminal.final_design is None:
        raise CapabilityError("The completed antenna turn did not produce a design to publish.")
    revision = 0 if baseline is None else baseline.revision + 1
    normalized = replace(terminal.final_design, revision=revision, validation=())
    return validate_design(normalized)


def execute_builder_turn(
    project_path: str | Path,
    session: BuilderProjectSession,
    instruction: str,
    *,
    planner,
    audit_log_path: str | Path | None = None,
    turn_id: str | None = None,
    budgets: AgentLoopBudgets | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> BuilderTurnResult:
    """Run one bounded agent turn, then publish exactly one terminal outcome."""

    current_turn_id = turn_id or f"turn-{uuid.uuid4().hex}"
    limits = budgets or AgentLoopBudgets()
    audit_path = (
        audit_log_path
        if audit_log_path is not None
        else Path(project_path) / PLANNER_AUDIT_RELATIVE_PATH
    )
    baseline = session.design
    loop_planner = _agent_planner(planner)
    if cancel_requested is not None and cancel_requested():
        raise BuilderTurnCancelled("The antenna-planning turn was cancelled.")
    terminal = run_antenna_agent_loop(
        agent=_agent(),
        baseline_design=baseline,
        instruction=instruction,
        project_memory=session.memory,
        planner=loop_planner,
        budgets=limits,
    )
    if cancel_requested is not None and cancel_requested():
        raise BuilderTurnCancelled("The antenna-planning turn was cancelled.")

    if terminal.outcome == "finished":
        if terminal.has_publishable_change:
            published_design = _published_agent_design(baseline, terminal)
            evaluations = evaluate_project_constraints(session.memory, published_design)
            disclosure = _constraint_disclosure(evaluations)
            if disclosure:
                terminal = replace(
                    terminal,
                    message=_with_disclosure(terminal.message, disclosure),
                )
            user_message = user_facing_builder_text(terminal.message)
            executor_summaries = tuple(
                str(change["summary"])
                for change in terminal.aggregate_changes
                if change.get("kind") == "executor_summary" and change.get("summary")
            )
            changes = executor_summaries or ("applied the completed agent plan",)
            outcome_status = "executed"
        else:
            published_design = baseline
            changes = ()
            outcome_status = "completed"
            user_message = user_facing_builder_text(terminal.message)
        executed_tools = tuple(
            ToolCall(call.name, dict(call.arguments)) for call in terminal.aggregate_calls
        )
        update = StateUpdate(
            published_design,
            changes,
            executed_tools=executed_tools,
            message=(None if terminal.has_publishable_change else user_message),
        )
        outcome = BuilderInteractionOutcome(
            status=outcome_status,
            message=user_message,
            instruction=instruction,
            planner_calls=terminal.aggregate_calls,
            changes=changes,
        )
        proposals = (
            terminal.final_step.memory_proposals
            if terminal.final_step is not None else ()
        )
        published, semantic_audit = _publish_builder_outcome(
            project_path,
            session,
            outcome,
            resulting_design=published_design,
            turn_id=current_turn_id,
            memory_proposals=proposals,
            planner_calls=terminal.aggregate_calls,
        )
        _append_agent_turn_audit(
            audit_path,
            turn_id=current_turn_id,
            instruction=instruction,
            planner=loop_planner,
            budgets=limits,
            baseline_design=baseline,
            terminal=terminal,
            published_design=published_design,
            semantic_memory_audit=semantic_audit,
        )
        return BuilderTurnResult(published, update, current_turn_id, terminal)

    outcome_status = {
        "clarify": "clarification",
        "refuse": "refusal",
        "duplicate_rejection": (
            "validation_rejection"
            if isinstance(loop_planner, _LegacyOneShotAgentPlanner)
            else "duplicate_rejection"
        ),
    }.get(terminal.outcome, terminal.outcome)
    user_message = user_facing_builder_text(terminal.message)
    outcome = BuilderInteractionOutcome(
        status=outcome_status,
        message=user_message,
        instruction=instruction,
    )
    proposals = (
        terminal.final_step.memory_proposals
        if terminal.final_step is not None and terminal.outcome == "clarify"
        else ()
    )
    published, semantic_audit = _publish_builder_outcome(
        project_path,
        session,
        outcome,
        resulting_design=baseline,
        turn_id=current_turn_id,
        memory_proposals=proposals,
        planner_calls=(),
    )
    _append_agent_turn_audit(
        audit_path,
        turn_id=current_turn_id,
        instruction=instruction,
        planner=loop_planner,
        budgets=limits,
        baseline_design=baseline,
        terminal=terminal,
        published_design=baseline,
        semantic_memory_audit=semantic_audit,
    )
    if terminal.outcome in {"clarify", "refuse"}:
        update = StateUpdate(
            baseline,
            (),
            message=user_message,
        )
        return BuilderTurnResult(published, update, current_turn_id, terminal)
    raise CapabilityError(terminal.message)


def publish_builder_state_update(
    project_path: str | Path,
    session: BuilderProjectSession,
    update: StateUpdate,
    *,
    description: str,
    interaction_kind: str = "parameter_table",
    turn_id: str | None = None,
) -> BuilderTurnResult:
    """Publish a validated non-conversational state update through the same reducer."""

    current_turn_id = turn_id or f"turn-{uuid.uuid4().hex}"
    outcome = BuilderInteractionOutcome(
        status="executed",
        message=update.summary,
        instruction=description,
        interaction_kind=interaction_kind,
        planner_calls=(update.plan.planner_calls if update.plan is not None else ()),
        explicit_parameter_updates=(
            update.plan.parameter_updates if update.plan is not None else ()
        ),
        changes=update.changes,
    )
    published, _semantic_audit = _publish_builder_outcome(
        project_path,
        session,
        outcome,
        resulting_design=update.state,
        turn_id=current_turn_id,
    )
    return BuilderTurnResult(published, update, current_turn_id)


def save_project_design(
    project_path: str | Path,
    state: AntennaDesign,
    conversation: Iterable[dict[str, object]],
) -> tuple[Path, Path]:
    root = Path(project_path)
    state_path = root / DESIGN_STATE_RELATIVE_PATH
    conversation_path = root / DESIGN_CONVERSATION_RELATIVE_PATH
    atomic_write_json(state_path, {"schema_version": 2, "status": "experimental", "design": state.to_dict()})
    atomic_write_json(
        conversation_path,
        {
            "schema_version": BUILDER_CONVERSATION_SCHEMA_VERSION,
            "messages": list(conversation),
        },
    )
    return state_path, conversation_path


def _migrate_legacy_state(payload: dict[str, object]) -> AntennaDesign:
    values = {key: value for key, value in payload.items() if key not in {"template_id", "epsilon_r", "loss_tangent"}}
    return _agent().create_design("inset_patch", values)


def load_project_design(project_path: str | Path) -> tuple[AntennaDesign, list[dict[str, str]]]:
    root = Path(project_path)
    state = AntennaDesign.starting_design()
    messages: list[dict[str, str]] = []
    state_path = root / DESIGN_STATE_RELATIVE_PATH
    if state_path.exists():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise CapabilityError("The saved antenna design is malformed.")
        if isinstance(payload.get("design"), dict):
            state = AntennaDesign.from_dict(payload["design"])
        elif isinstance(payload.get("state"), dict):
            state = _migrate_legacy_state(payload["state"])
        else:
            raise CapabilityError("The saved antenna design is malformed.")
    conversation_path = root / DESIGN_CONVERSATION_RELATIVE_PATH
    if conversation_path.exists():
        payload = json.loads(conversation_path.read_text(encoding="utf-8"))
        raw_messages = payload.get("messages", []) if isinstance(payload, dict) else []
        if isinstance(raw_messages, list):
            messages = [
                {"role": str(item["role"]), "content": str(item["content"])}
                for item in raw_messages
                if isinstance(item, dict)
                and item.get("role") in {"user", "builder"}
                and item.get("content")
            ]
    return state, messages


def lhs_variables_for_state(
    state: AntennaDesign,
    selected_parameters: Iterable[str],
) -> list[LHSVariable]:
    by_name = {parameter.name: parameter for parameter in state.parameters if parameter.sweepable}
    variables: list[LHSVariable] = []
    for name in selected_parameters:
        parameter = by_name.get(name)
        if parameter is None:
            raise CapabilityError(f"{name} is not a sweepable parameter for this design.")
        variables.append(
            LHSVariable(
                name,
                parameter.value * parameter.sweep_lower_factor,
                parameter.value * parameter.sweep_upper_factor,
            )
        )
    if not variables:
        raise CapabilityError("Select at least one parameter to vary in LHS sampling.")
    return variables
