"""Model-agnostic schema-constrained planners for registered antenna tools."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from studio.antenna_analysis import EngineeringAnalysisResult
from studio.antenna_design import AntennaDesign
from studio.antenna_engineering import EngineeringReport
from studio.antenna_engineering_summary import (
    GROUP_PREFIX, PlannerEngineeringSummary, engineering_report_hash, summarize_engineering_report,
)
from studio.antenna_tools import CapabilityError
from studio.planner_credentials import (
    GEMINI,
    GROQ,
    OPENROUTER,
    read_environment_or_env_file,
    read_stored_credential,
)


PLAN_SCHEMA_VERSION = 1
# Contract v2 removes Boolean semantic-memory placeholder values from the
# AgentStep schema, strict parser, and shared provider-neutral instruction.
# The on-wire AgentStep schema version remains 1 for compatible valid payloads.
AGENT_STEP_CONTRACT_VERSION = 3
AGENT_STEP_SCHEMA_VERSION = 1
AGENT_OBSERVATION_SCHEMA_VERSION = 1
AGENT_TRAJECTORY_SCHEMA_VERSION = 1
AGENT_TERMINAL_RESULT_SCHEMA_VERSION = 1
MAX_PLANNER_CALLS = 24
MAX_MEMORY_PROPOSALS = 6
KNOWN_FAMILY_TERMS = (
    "bowtie", "helix", "horn", "loop", "monopole", "pifa", "reflectarray",
    "spiral", "vivaldi", "yagi",
)
SOLVER_EXECUTION_PATTERN = re.compile(
    r"\b(?:start|run|launch|execute)\b.{0,32}\b(?:cst|solver|simulation)\b"
    r"|\b(?:cst|solver|simulation)\b.{0,32}\b(?:start|run|launch|execute)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PlannedToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMToolPlan:
    status: str
    message: str
    calls: tuple[PlannedToolCall, ...]


@dataclass(frozen=True, slots=True)
class SemanticMemoryProposal:
    """Untrusted model proposal; deterministic publication validates every field."""

    action: Any = None
    semantic_kind: Any = None
    key: Any = None
    value: Any = None
    evidence_quote: Any = None
    unit: Any = None
    reference_id: Any = None

    @classmethod
    def from_unvalidated(cls, payload: Mapping[str, Any]) -> "SemanticMemoryProposal":
        return cls(
            action=payload.get("action"),
            semantic_kind=payload.get("semantic_kind"),
            key=payload.get("key"),
            value=payload.get("value"),
            evidence_quote=payload.get("evidence_quote"),
            unit=payload.get("unit"),
            reference_id=payload.get("reference_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "semantic_kind": self.semantic_kind,
            "key": self.key,
            "value": self.value,
            "unit": self.unit,
            "evidence_quote": self.evidence_quote,
            "reference_id": self.reference_id,
        }


@dataclass(frozen=True, slots=True)
class AgentLoopBudgets:
    """Configurable Stage-2 limits; enforcement belongs to the future runner."""

    decision_iterations: int = 5
    accepted_action_batches: int = 3
    rejected_action_batches: int = 2
    total_proposed_tool_calls: int = 24
    analysis_batches: int = 3

    def __post_init__(self) -> None:
        for name in (
            "decision_iterations",
            "accepted_action_batches",
            "rejected_action_batches",
            "total_proposed_tool_calls",
            "analysis_batches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

    def to_dict(self) -> dict[str, int]:
        return {
            "decision_iterations": self.decision_iterations,
            "accepted_action_batches": self.accepted_action_batches,
            "rejected_action_batches": self.rejected_action_batches,
            "total_proposed_tool_calls": self.total_proposed_tool_calls,
            "analysis_batches": self.analysis_batches,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentLoopBudgets":
        expected = {
            "decision_iterations",
            "accepted_action_batches",
            "rejected_action_batches",
            "total_proposed_tool_calls",
            "analysis_batches",
        }
        legacy = expected - {"analysis_batches"}
        if frozenset(payload) not in {frozenset(expected), frozenset(legacy)}:
            raise ValueError("Agent-loop budgets do not match the required schema.")
        return cls(**{name: payload[name] for name in expected if name in payload})


@dataclass(frozen=True, slots=True)
class DeferredEngineeringFinding:
    observation_id: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id.strip():
            raise ValueError("A deferred finding requires an observation ID.")
        if not isinstance(self.reason, str) or not self.reason.strip() or len(self.reason) > 400:
            raise ValueError("A deferred finding requires a reason of 1 to 400 characters.")


@dataclass(frozen=True, slots=True)
class EngineeringDisposition:
    """Terminal accounting, not a geometry action or an RF certification."""

    semantic_design_hash: str | None = None
    acknowledged_observation_ids: tuple[str, ...] = ()
    deferred: tuple[DeferredEngineeringFinding, ...] = ()
    engineering_report_hash: str | None = None

    def __post_init__(self) -> None:
        if self.engineering_report_hash is not None and (
            not isinstance(self.engineering_report_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.engineering_report_hash) is None
        ):
            raise ValueError("Disposition engineering_report_hash must be a SHA-256 digest or null.")
        if self.semantic_design_hash is not None and (
            not isinstance(self.semantic_design_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.semantic_design_hash) is None
        ):
            raise ValueError("Disposition semantic_design_hash must be a SHA-256 digest or null.")
        if not isinstance(self.acknowledged_observation_ids, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in self.acknowledged_observation_ids
        ):
            raise ValueError("Acknowledged observation IDs must be nonempty strings.")
        if not isinstance(self.deferred, (list, tuple)) or any(
            not isinstance(item, DeferredEngineeringFinding) for item in self.deferred
        ):
            raise ValueError("Invalid deferred engineering findings.")
        object.__setattr__(self, "acknowledged_observation_ids", tuple(self.acknowledged_observation_ids))
        object.__setattr__(self, "deferred", tuple(self.deferred))

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_design_hash": self.semantic_design_hash,
            "acknowledged_observation_ids": list(self.acknowledged_observation_ids),
            "deferred": [{"observation_id": item.observation_id, "reason": item.reason} for item in self.deferred],
            **({"engineering_report_hash": self.engineering_report_hash} if self.engineering_report_hash is not None else {}),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "EngineeringDisposition":
        required = {"semantic_design_hash", "acknowledged_observation_ids", "deferred"}
        if not isinstance(payload, Mapping) or set(payload) not in (required, required | {"engineering_report_hash"}):
            raise ValueError("Engineering disposition does not match the required schema.")
        if not isinstance(payload["acknowledged_observation_ids"], list) or not isinstance(payload["deferred"], list):
            raise ValueError("Disposition acknowledged/deferred fields must be arrays.")
        deferred = []
        for item in payload["deferred"]:
            if not isinstance(item, Mapping) or set(item) != {"observation_id", "reason"}:
                raise ValueError("Deferred finding does not match the required schema.")
            deferred.append(DeferredEngineeringFinding(**item))
        return cls(payload["semantic_design_hash"], tuple(payload["acknowledged_observation_ids"]), tuple(deferred),
                   payload.get("engineering_report_hash"))


def validate_engineering_disposition(
    disposition: EngineeringDisposition | None, report: EngineeringReport,
    summary: PlannerEngineeringSummary | None = None,
) -> dict[str, Any]:
    """Return structured terminal feedback; do not interpret the model's prose."""
    disposition = disposition or EngineeringDisposition()
    current = {item.observation_id for item in report.findings}
    warnings = {item.observation_id for item in report.findings if item.severity == "warning"}
    ids = (*disposition.acknowledged_observation_ids, *(item.observation_id for item in disposition.deferred))
    group_ids = {identity for identity in ids if identity.startswith(GROUP_PREFIX)}
    report_hash = engineering_report_hash(report)
    summary_payload = (summary or summarize_engineering_report(report)).to_dict() if group_ids else None
    groups = {group["group_id"]: group["constituent_observation_ids"] for group in summary_payload["groups"]} if summary_payload else {}
    stale_groups = bool(group_ids) and (
        disposition.engineering_report_hash != report_hash or summary_payload["engineering_report_hash"] != report_hash
    )
    unknown_groups = group_ids - groups.keys()
    expanded = [member for identity in ids for member in (groups.get(identity, ()) if identity in group_ids else (identity,))]
    seen, duplicates = set(), set()
    for identity in expanded:
        if identity in seen:
            duplicates.add(identity)
        seen.add(identity)
    stale = (
        disposition.semantic_design_hash != report.semantic_design_hash
        if ids or warnings or disposition.semantic_design_hash is not None else False
    )
    unknown = seen - current
    missing = warnings - seen
    return {
        "status": "rejected" if stale or stale_groups or unknown_groups or duplicates or unknown or missing else "valid",
        "category": "engineering_disposition",
        "rejected_step": "finish" if stale or stale_groups or unknown_groups or duplicates or unknown or missing else None,
        "current_semantic_design_hash": report.semantic_design_hash,
        "stale_design_hash": stale,
        "unknown_observation_ids": sorted(unknown),
        "duplicate_observation_ids": sorted(duplicates),
        "missing_warning_observation_ids": sorted(missing),
        **({"current_engineering_report_hash": report_hash, "stale_group_report_hash": stale_groups,
            "unknown_group_ids": sorted(unknown_groups)} if group_ids else {}),
    }


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One model-emitted decision in the future bounded agent loop."""

    status: str
    message: str
    calls: tuple[PlannedToolCall, ...] = ()
    schema_version: int = AGENT_STEP_SCHEMA_VERSION
    engineering_disposition: EngineeringDisposition | None = None
    memory_proposals: tuple[SemanticMemoryProposal, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != AGENT_STEP_SCHEMA_VERSION:
            raise ValueError("Unsupported AgentStep schema version.")
        if self.status not in {"execute", "finish", "clarify", "refuse"}:
            raise ValueError("Unknown model-emitted AgentStep status.")
        if not isinstance(self.message, str) or len(self.message) > 1200:
            raise ValueError("AgentStep message is invalid.")
        if self.status == "execute" and not self.calls:
            raise ValueError("An execute AgentStep requires at least one tool call.")
        if self.status != "execute" and self.calls:
            raise ValueError(f"An AgentStep with status {self.status!r} requires zero tool calls.")
        if self.engineering_disposition is not None and (
            self.status != "finish" or not isinstance(self.engineering_disposition, EngineeringDisposition)
        ):
            raise ValueError("Engineering disposition is only available on finish.")
        if len(self.memory_proposals) > MAX_MEMORY_PROPOSALS or any(
            not isinstance(item, SemanticMemoryProposal) for item in self.memory_proposals
        ):
            raise ValueError("AgentStep semantic-memory proposals are invalid.")
        if self.memory_proposals and self.status in {"execute", "refuse"}:
            raise ValueError(
                "Semantic-memory proposals are available only on finish or clarify outcomes."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "message": self.message,
            "calls": [
                {"name": call.name, "arguments": dict(call.arguments)}
                for call in self.calls
            ],
            **({"engineering_disposition": self.engineering_disposition.to_dict()}
               if self.engineering_disposition is not None else {}),
            **({"memory_proposals": [item.to_dict() for item in self.memory_proposals]}
               if self.memory_proposals else {}),
        }


@dataclass(frozen=True, slots=True)
class AgentStepObservation:
    """Deterministic execution feedback supplied to a later agent iteration."""

    outcome: str
    planned_calls: tuple[PlannedToolCall, ...]
    working_design_ref: dict[str, Any] | None
    working_design_hash: str
    remaining_budgets: AgentLoopBudgets
    executed_calls: tuple[PlannedToolCall, ...] = ()
    deterministic_changes: tuple[dict[str, Any], ...] = ()
    validation_result: dict[str, Any] = field(default_factory=dict)
    failure_category: str | None = None
    sanitized_error: str | None = None
    failed_action_fingerprint: str | None = None
    analysis_results: tuple[EngineeringAnalysisResult, ...] = ()
    analysis_cache_hits: int = 0
    schema_version: int = AGENT_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != AGENT_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("Unsupported agent observation schema version.")
        if self.outcome not in {"accepted", "accepted_analysis", "rejected"}:
            raise ValueError("Agent observation outcome must be accepted, accepted_analysis, or rejected.")
        terminal_rejection = (
            self.outcome == "rejected" and self.failure_category == "engineering_disposition"
            and self.validation_result.get("rejected_step") == "finish"
        )
        if not self.planned_calls and not terminal_rejection:
            raise ValueError("An execution observation requires at least one planned call.")
        if not self.working_design_hash:
            raise ValueError("An execution observation requires a working-design hash.")
        if type(self.analysis_cache_hits) is not int or self.analysis_cache_hits < 0:
            raise ValueError("analysis_cache_hits must be a nonnegative integer.")
        if not isinstance(self.analysis_results, (tuple, list)) or any(
            not isinstance(item, EngineeringAnalysisResult) for item in self.analysis_results
        ):
            raise ValueError("analysis_results must contain EngineeringAnalysisResult values.")
        object.__setattr__(self, "analysis_results", tuple(self.analysis_results))
        if self.outcome in {"accepted", "accepted_analysis"}:
            if not self.executed_calls:
                raise ValueError("An accepted observation requires actual executed calls.")
            if self.failure_category or self.sanitized_error or self.failed_action_fingerprint:
                raise ValueError("An accepted observation cannot contain failure details.")
            if self.outcome == "accepted_analysis":
                if not self.analysis_results or self.deterministic_changes:
                    raise ValueError("Accepted analysis requires results and cannot contain design changes.")
            elif self.analysis_results or self.analysis_cache_hits:
                raise ValueError("Accepted design actions cannot contain analysis results.")
        else:
            if not self.failure_category or not self.sanitized_error or not self.failed_action_fingerprint:
                raise ValueError("A rejected observation requires category, sanitized error, and fingerprint.")
            if self.executed_calls or self.deterministic_changes:
                raise ValueError("A rejected transactional observation cannot publish executed changes.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "outcome": self.outcome,
            "planned_calls": [
                {"name": call.name, "arguments": dict(call.arguments)}
                for call in self.planned_calls
            ],
            "executed_calls": [
                {"name": call.name, "arguments": dict(call.arguments)}
                for call in self.executed_calls
            ],
            "deterministic_changes": [dict(item) for item in self.deterministic_changes],
            "validation_result": dict(self.validation_result),
            "working_design_ref": dict(self.working_design_ref) if self.working_design_ref else None,
            "working_design_hash": self.working_design_hash,
            "failure_category": self.failure_category,
            "sanitized_error": self.sanitized_error,
            "failed_action_fingerprint": self.failed_action_fingerprint,
            "analysis_results": [item.to_dict() for item in self.analysis_results],
            "analysis_cache_hits": self.analysis_cache_hits,
            "remaining_budgets": self.remaining_budgets.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentStepObservation":
        expected = {
            "schema_version", "outcome", "planned_calls", "executed_calls",
            "deterministic_changes", "validation_result", "working_design_ref",
            "working_design_hash", "failure_category", "sanitized_error",
            "failed_action_fingerprint", "remaining_budgets", "analysis_results", "analysis_cache_hits",
        }
        legacy = expected - {"analysis_results", "analysis_cache_hits"}
        if frozenset(payload) not in {frozenset(expected), frozenset(legacy)} or payload.get("schema_version") != AGENT_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("Agent observation does not match the required schema.")
        calls = payload["planned_calls"]
        if not isinstance(calls, list):
            raise ValueError("Agent observation planned_calls must be a list.")
        return cls(
            outcome=str(payload["outcome"]),
            planned_calls=tuple(_planned_call_from_mapping(item) for item in calls),
            executed_calls=tuple(_planned_call_from_mapping(item) for item in payload["executed_calls"]),
            deterministic_changes=tuple(dict(item) for item in payload["deterministic_changes"]),
            validation_result=dict(payload["validation_result"]),
            working_design_ref=(
                dict(payload["working_design_ref"])
                if payload["working_design_ref"] is not None else None
            ),
            working_design_hash=str(payload["working_design_hash"]),
            failure_category=(str(payload["failure_category"]) if payload["failure_category"] is not None else None),
            sanitized_error=(str(payload["sanitized_error"]) if payload["sanitized_error"] is not None else None),
            failed_action_fingerprint=(
                str(payload["failed_action_fingerprint"])
                if payload["failed_action_fingerprint"] is not None else None
            ),
            analysis_results=tuple(
                EngineeringAnalysisResult.from_dict(item) for item in payload.get("analysis_results", ())
            ),
            analysis_cache_hits=payload.get("analysis_cache_hits", 0),
            remaining_budgets=AgentLoopBudgets.from_dict(payload["remaining_budgets"]),
        )


_SENSITIVE_METADATA_KEYS = (
    "api_key", "apikey", "authorization", "cookie", "password", "secret", "token", "x-goog-api-key",
)


def _redact_sensitive_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if any(marker.replace("-", "_") in normalized for marker in _SENSITIVE_METADATA_KEYS):
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = _redact_sensitive_metadata(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact_sensitive_metadata(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class AgentTrajectoryEntry:
    """Serializable audit record for one future runner iteration."""

    iteration_index: int
    working_state_before: dict[str, Any] | None
    returned_step: AgentStep | None
    planner_metadata: dict[str, Any]
    execution_status: str
    executed_calls: tuple[PlannedToolCall, ...]
    deterministic_changes: tuple[dict[str, Any], ...]
    sanitized_failure: dict[str, Any] | None
    working_state_after: dict[str, Any] | None
    observation: AgentStepObservation | None
    budget_counters: AgentLoopBudgets
    schema_version: int = AGENT_TRAJECTORY_SCHEMA_VERSION
    engineering_audit: dict[str, Any] | None = None
    analysis_results: tuple[EngineeringAnalysisResult, ...] = ()
    analysis_cache_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "iteration_index": self.iteration_index,
            "working_state_before": dict(self.working_state_before) if self.working_state_before else None,
            "returned_step": self.returned_step.to_dict() if self.returned_step else None,
            "planner_metadata": _redact_sensitive_metadata(self.planner_metadata),
            "execution_status": self.execution_status,
            "executed_calls": [
                {"name": call.name, "arguments": dict(call.arguments)}
                for call in self.executed_calls
            ],
            "deterministic_changes": [dict(item) for item in self.deterministic_changes],
            "sanitized_failure": (
                _redact_sensitive_metadata(self.sanitized_failure)
                if self.sanitized_failure is not None else None
            ),
            "working_state_after": dict(self.working_state_after) if self.working_state_after else None,
            "observation": self.observation.to_dict() if self.observation else None,
            "budget_counters": self.budget_counters.to_dict(),
            "engineering_audit": self.engineering_audit,
            "analysis_results": [item.to_dict() for item in self.analysis_results],
            "analysis_cache_hits": self.analysis_cache_hits,
        }


@dataclass(frozen=True, slots=True)
class AgentTerminalResult:
    """Terminal control result reserved for the future bounded runner."""

    outcome: str
    message: str
    working_design_ref: dict[str, Any] | None
    working_design_hash: str | None
    final_design: AntennaDesign | None = None
    aggregate_calls: tuple[PlannedToolCall, ...] = ()
    aggregate_changes: tuple[dict[str, Any], ...] = ()
    has_publishable_change: bool = False
    trajectory: tuple[AgentTrajectoryEntry, ...] = ()
    final_step: AgentStep | None = None
    schema_version: int = AGENT_TERMINAL_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        allowed = {
            "finish", "finished", "clarify", "refuse", "agent_limit", "provider_error",
            "cycle_detected", "invalid_plan", "duplicate_rejection",
        }
        if self.outcome not in allowed:
            raise ValueError("Unknown agent terminal outcome.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "outcome": self.outcome,
            "message": self.message,
            "working_design_ref": dict(self.working_design_ref) if self.working_design_ref else None,
            "working_design_hash": self.working_design_hash,
            "final_design": self.final_design.to_dict() if self.final_design else None,
            "aggregate_calls": [
                {"name": call.name, "arguments": dict(call.arguments)}
                for call in self.aggregate_calls
            ],
            "aggregate_changes": [dict(item) for item in self.aggregate_changes],
            "has_publishable_change": self.has_publishable_change,
            "final_step": self.final_step.to_dict() if self.final_step else None,
            "trajectory": [item.to_dict() for item in self.trajectory],
        }


class ToolUsePlanner(Protocol):
    """Model-agnostic planning interface consumed by the builder facade."""

    def plan(
        self,
        *,
        instruction: str,
        current_design: Mapping[str, Any] | None,
        capability_manifest: Mapping[str, Any],
        project_memory: Mapping[str, Any] | None = None,
        agent_observation: AgentStepObservation | None = None,
        execution_feedback: Mapping[str, Any] | None = None,
    ) -> LLMToolPlan: ...


class AgentLoopPlanner(Protocol):
    """Future versioned agent-step interface, separate from ToolPlan v1."""

    def plan_agent_step(
        self,
        *,
        instruction: str,
        current_design: Mapping[str, Any] | None,
        capability_manifest: Mapping[str, Any],
        remaining_budgets: AgentLoopBudgets,
        project_memory: Mapping[str, Any] | None = None,
        agent_observation: AgentStepObservation | None = None,
        engineering_report: EngineeringReport | None = None,
    ) -> AgentStep: ...


@dataclass(frozen=True, slots=True)
class PlannerExchange:
    """Provider-neutral prompt, context, and output contract."""

    system_instruction: str
    user_content: str
    schema: dict[str, Any]
    callable_names: tuple[str, ...]


PLANNER_SYSTEM_INSTRUCTION = (
    "You are the planning layer of a constrained antenna design agent. User requests concern antenna-design edits or installed antenna capabilities. "
    "Return exactly one JSON object matching the supplied schema. Inspect the current solver-neutral design and runtime capabilities. "
    "The current_design value may be null, which means the project has no antenna yet. In that case, an executable first-design plan must call recipe.select first, "
    "then may use parameter.set for that selected recipe. Do not assume or invent a default antenna. Do not select a recipe merely to answer a capability question; "
    "answer such a question through clarify or refuse with no calls and leave the project empty. "
    "A recipe or parameter listed in runtime_capabilities is supported. For supported requests, use status execute and emit an ordered sequence using only callable_tools. "
    "Use recipe.select for a new antenna family, design.reset when the user asks to clear/reset the current family, parameter.set once per requested value, "
    "and modifier.apply/remove only when listed. Call modifier.apply before parameter.set for any parameter owned by that modifier. "
    "Recipes and modifiers compile into deterministic primitives. You may also call only the primitive tools explicitly marked kind=primitive. "
    "Persisted composed features are listed in composed_features. Use composition.set_scope to change an existing feature's generic target scope; "
    "use scope all with elements [], scope single with exactly one element, or scope selected with the requested element coordinates. "
    "Each composed feature lists stable group and operation IDs, created parameters, and editable primitive arguments. Use composition.update_operation "
    "with those exact IDs to modify an existing feature. Update its parameter.create operation when a named parameter controls the requested size; "
    "otherwise update the relevant primitive or transform arguments. Use composition.delete to remove exactly one identified composed feature. "
    "If multiple composed features could match the user's reference and the structured feature records do not identify one uniquely, clarify instead of choosing arbitrarily. "
    "Put all design actions before primitive calls. Use a modifier only when its documented geometry exactly matches the request. "
    "For a novel supported slot or Boolean composition, create any requested parameter, create planner_created Boolean-tool geometry, then subtract it. "
    "A requested slot radius, width, or height must be a named parameter created with parameter.create (unless derived from an existing parameter), and geometry expressions must reference it; never bury a requested size as a numeric literal. "
    "Use an existing target_id whose semantic role is radiating_patch_conductor and boolean_target=true. Never invent a target ID or alter feed, ground, substrate, or ports. "
    "For z-axis cylinders, center_1 and center_2 are target-local x/y offsets from the selected radiating element center; start and end are strictly the conductor-layer z bounds required by the tool schema. "
    "For rectangles, x_min/x_max and y_min/y_max are target-local offsets; z_min/z_max are strictly the conductor-layer bounds required by the tool schema. "
    "New geometry must use tags planner_created, boolean_tool, and slot, and must be consumed by boolean.subtract or boolean.union. "
    "All key, object_id, operation_id, target_id, and tool ID values are identifiers with no spaces. parameter.create exports its key as the solver parameter name. New geometry IDs must end in _tool. "
    "Boolean operation IDs must end in _subtract or _union. Give every new object and operation a unique descriptive ID. "
    "For Boolean calls, tool_ids contains each newly created cutting-tool object ID exactly once; never put the target ID in tool_ids. "
    "Excitation intent and current physical realization are separate canonical facts. Use excitation.set_strategy when the user explicitly requests an available excitation strategy, "
    "and inspect the structured excitation summary separately from current_physical_port_ids. Do not claim a strategy or feed network is physically realized unless realization_status is realized. "
    "Do not invent unsupported excitation or feed-network synthesis. If excitation intent is ambiguous and materially affects the requested design, clarify instead of silently choosing a strategy. "
    "Never emit code, CST commands, file operations, or solver operations. If ambiguity changes geometry, return clarify with one specific question and no calls. "
    "If no installed capability can represent the request, return refuse with a concise explanation and no calls. Do not approximate an unsupported topology. "
    "Never substitute the closest installed recipe for an unsupported requested antenna family. Only include changes explicitly requested by the user; do not copy unchanged state values into calls. "
    "Before returning, verify that every explicitly requested supported family, material, frequency, dimension, array count, spacing, and modifier appears in the calls exactly once. "
    "Use status execute whenever calls is non-empty. For clarify or refuse, calls must be an empty array. "
    "Preserve the current design unless the user explicitly requests a family change or reset. "
    "Example: 'Create a circular patch at 5.8 GHz on Rogers RT5880 with probe radius 0.8 mm' executes recipe.select(circular_patch_v1), "
    "parameter.set(frequency_ghz, 5.8), parameter.set(material, Rogers RT5880), and parameter.set(probe_radius_mm, 0.8). "
    "Example: 'Turn it into a 1x4 linear array with 0.6 lambda spacing' executes parameter.set(array_rows, 1), "
    "parameter.set(array_columns, 4), and parameter.set(element_spacing_lambda, 0.6). "
    "If the user also requests an existing composition on every element, call composition.set_scope for its group_id with scope all and elements []. "
    "Example: 'Clear and add circular corner cutouts with radius 1/4 patch width' executes design.reset, "
    "modifier.apply(corner_circle_cutouts_v1), then parameter.set(corner_radius_ratio, 0.25), in that order. "
    "Example: a single 3 mm circular slot at patch center does not use the corner modifier. It creates key slot_radius_mm with value 3, "
    "creates z-axis geometry.cylinder object center_slot_tool with target-local center offsets 0,0, radius slot_radius_mm, and the radiating conductor z bounds, "
    "then calls boolean.subtract with operation_id center_slot_subtract, target_id element_1_1_patch, and tool_ids [center_slot_tool]. "
    "For two 2 mm-radius slots 5 mm left and right of patch center, create tools centered at x=-5,y=0 and x=5,y=0 (or duplicate the first by offset [10,0,0]), "
    "using the conductor z bounds, then subtract both tools in one Boolean call. "
    "For a centered 6 by 1 mm rectangle, create slot_width_mm=6 and slot_height_mm=1, then use x bounds -slot_width_mm/2 and slot_width_mm/2 and "
    "y bounds -slot_height_mm/2 and slot_height_mm/2, using the conductor z bounds, then subtract it. "
    "Example: an unsupported horn antenna returns refuse with calls []."
)


def build_planner_exchange(
    *,
    instruction: str,
    current_design: Mapping[str, Any] | None,
    capability_manifest: Mapping[str, Any],
    project_memory: Mapping[str, Any] | None = None,
    agent_observation: AgentStepObservation | None = None,
    execution_feedback: Mapping[str, Any] | None = None,
) -> PlannerExchange:
    """Build the identical prompt, state, manifest, and schema for every backend."""

    callable_tools = tuple(
        item for item in capability_manifest.get("callable_tools", ())
        if isinstance(item, dict) and item.get("name") and isinstance(item.get("arguments"), dict)
    )
    schema = plan_json_schema(callable_tools)
    user_payload: dict[str, Any] = {
        "instruction": instruction,
        "current_design": current_design,
        "runtime_capabilities": capability_manifest,
        "tool_plan_schema": schema,
        "request_to_plan": instruction,
    }
    if project_memory is not None:
        user_payload["project_memory"] = dict(project_memory)
    if agent_observation is not None:
        user_payload["agent_observation"] = agent_observation.to_dict()
    if execution_feedback:
        user_payload["rejected_previous_plan"] = dict(execution_feedback)
        user_payload["repair_requirement"] = (
            "Correct the rejected plan. Remove invented or unavailable parameters, preserve every explicit supported request, "
            "and fix dependency order. Do not change values the user did not request."
        )
    return PlannerExchange(
        system_instruction=PLANNER_SYSTEM_INSTRUCTION,
        user_content=json.dumps(user_payload),
        schema=schema,
        callable_names=tuple(str(item["name"]) for item in callable_tools),
    )


def build_agent_step_exchange(
    *,
    instruction: str,
    current_design: Mapping[str, Any] | None,
    capability_manifest: Mapping[str, Any],
    remaining_budgets: AgentLoopBudgets,
    project_memory: Mapping[str, Any] | None = None,
    agent_observation: AgentStepObservation | None = None,
    engineering_report: EngineeringReport | None = None,
) -> PlannerExchange:
    """Build the provider-neutral bounded-loop exchange over the same tool schemas."""

    callable_tools = tuple(
        item for item in capability_manifest.get("callable_tools", ())
        if isinstance(item, dict) and item.get("name") and isinstance(item.get("arguments"), dict)
    )
    schema = agent_step_json_schema(callable_tools)
    if engineering_report is not None:
        schema["properties"]["engineering_disposition"]["properties"]["engineering_report_hash"] = {
            "type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"}
    system_instruction = PLANNER_SYSTEM_INSTRUCTION.replace(
        "Use status execute whenever calls is non-empty. For clarify or refuse, calls must be an empty array. ",
        "Return exactly one bounded agent decision. Use status execute whenever calls is non-empty. "
        "Use finish with no calls only when the complete user request is satisfied by the current working state, including after observing accepted prior actions. "
        "Do not finish merely because the current action batch succeeded when requested work remains. "
        "Use clarify or refuse with no calls when appropriate. ",
    )
    system_instruction += (
        " The runtime capability manifest describes tools available for the CURRENT working state; it is not necessarily the complete capability set for the whole turn. "
        "After an accepted execute step, the runtime will provide the updated working state, the execution observation, and a newly generated capability manifest. "
        "If the full request cannot be completed with the current tools but a valid current action can make meaningful progress toward it and may unlock state-dependent capabilities, prefer execute over refuse. "
        "Do not refuse solely because a later step is unavailable in the current manifest. "
        "Use refuse only when the installed system cannot satisfy the request through available state transitions and capabilities, or when no valid progress-making action exists. "
        "Do not invent future tools; rely only on the runtime guarantee that capabilities are regenerated after accepted state changes."
    )
    system_instruction += (
        " Registered tools declare an explicit effect. Analysis tools inspect the current working design and return structured engineering results without modifying it. "
        "Use analysis tools when quantitative information is needed to reason about the request, and inspect accepted_analysis results on the next iteration. "
        "Treat analytical outputs as estimates or measurements within their stated applicability, assumptions, and limitations, not as full-wave validation. "
        "When using a structured analysis result, distinguish its actual measurements, assumptions, and limitations from any additional RF assumption. "
        "Do not state that an analysis established a condition it did not evaluate; call an appropriate available analysis tool with the needed explicit assumption, or state the limitation and ask for clarification. "
        "Do not repeatedly request the same analysis against the same design without a reason. "
        "For now, each execute batch must contain only design_action tools or only analysis tools; never mix effects in one batch."
    )
    system_instruction += (
        " EngineeringReport contains deterministic inspection findings about the CURRENT working design. "
        "Findings are evidence to reason about, not predetermined repair instructions. "
        "info does not imply a problem; warning indicates a credible engineering concern but does not automatically prohibit finish. "
        "blocking prohibits finish, although the current engineering checks introduce no blocking findings. "
        "Coverage may be incomplete, unknown, or failed; absence of findings does not prove RF correctness. "
        "Consider material findings when deciding whether to execute, clarify, refuse, or finish. "
        "You may finish with warnings when the complete request is satisfied, disclosing material concerns accurately. "
        "Do not claim these checks establish resonance, match, gain, efficiency, bandwidth, polarization, or full-wave validity. "
        "Do not invent repair capabilities absent from the current tool manifest."
    )
    system_instruction += (
        " On finish, include engineering_disposition when current warnings exist: copy the EngineeringReport semantic_design_hash "
        "and account for every current warning observation_id exactly once, either in acknowledged_observation_ids or in deferred with a short reason. "
        "Do not reuse a disposition from a different working state or invent observation IDs. Info findings need no acknowledgement. "
        "With no warnings, disposition may be omitted or contain null semantic_design_hash and empty lists. "
        "Do not include engineering_disposition on execute, clarify, or refuse. "
        "Acknowledging or deferring warnings permits finish; it does not certify their resolution or RF correctness. "
        "When finishing with engineering warnings, accurately disclose the material acknowledged/deferred findings in the user-facing message. "
        "A rejected finish returns structured disposition feedback for correction on the next decision within the normal remaining budgets."
    )
    system_instruction += (
        " Terminal finish and clarify decisions may include memory_proposals for durable intent explicitly stated in the CURRENT user instruction. "
        "A refuse decision must not include memory_proposals: an unsupported or rejected request is not an accepted project goal. "
        "Every proposal must quote an exact nonempty substring of that instruction in evidence_quote. "
        "Use the explicitly stated semantic value itself; never substitute true or false for a named value such as RHCP, corporate feed, or manufacturing simplicity. "
        "Use at most six proposals and only for durable requirements, constraints, preferences, goals, future intent, priorities, or explicit decisions likely to matter later. "
        "Do not propose memory for transient execution commands, current physical state already represented by the canonical design, tool history, feature coordinates, or engineering measurements. "
        "Distinguish future goals and preferences from what is physically realized now, and never infer an unstated preference. "
        "Use a concise normalized snake_case key. Use supersede for a changed active value and resolve when the user explicitly withdraws an earlier item. "
        "Do not include memory_proposals on execute or refuse; wait for a finish or clarify decision so the runtime can validate and publish them once."
    )
    user_payload: dict[str, Any] = {
        "instruction": instruction,
        "current_design": current_design,
        "runtime_capabilities": capability_manifest,
        "agent_step_schema": schema,
        "request_to_plan": instruction,
        "remaining_budgets": remaining_budgets.to_dict(),
    }
    if project_memory is not None:
        user_payload["project_memory"] = dict(project_memory)
    if agent_observation is not None:
        user_payload["agent_observation"] = agent_observation.to_dict()
    if engineering_report is not None:
        # Recheck references against this exchange's live manifest.
        installed_ids = {
            str(item[key])
            for section, key in (
                ("callable_tools", "name"), ("deterministic_primitive_tools", "name"),
                ("recipes", "recipe_id"), ("modifiers", "modifier_id"),
            )
            for item in capability_manifest.get(section, ())
            if isinstance(item, Mapping) and item.get(key)
        }
        engineering_report.validate_capabilities(installed_ids)
        expected_ref = (
            {"design_id": current_design.get("design_id"), "revision": current_design.get("revision")}
            if current_design is not None else None
        )
        report_ref = engineering_report.working_design_ref
        if (report_ref.to_dict() if report_ref is not None else None) != expected_ref:
            raise ValueError("Engineering report does not reference the current working design revision.")
        summary = summarize_engineering_report(engineering_report,
            AntennaDesign.from_dict(current_design) if current_design is not None else None)
        user_payload["engineering_summary"] = summary.to_planner_dict()
        system_instruction += (
            " Engineering evidence is supplied as engineering_summary, a deterministic projection of the raw EngineeringReport retained internally. "
            "Groups represent equivalent structured relationships; representative_message describes one occurrence only. "
            "Use occurrence counts, affected sets, and separately named measurement ranges/common values for the whole group. "
            "Never conflate differently named measurements. Coverage and advisory severity semantics are unchanged. "
            "For finish, prefer group_id references in the existing acknowledged_observation_ids or deferred[].observation_id fields. "
            "An individual observation ID remains valid if available. A group reference accounts for all its current constituent observations. "
            "Copy semantic_design_hash and engineering_report_hash from engineering_summary when using groups. "
            "Account for each warning group exactly once; do not acknowledge both a group and any of its constituent observations. "
            "Group accounting does not mean the findings were resolved. Disclose material concerns accurately in the message."
        )
    return PlannerExchange(
        system_instruction=system_instruction,
        user_content=json.dumps(user_payload),
        schema=schema,
        callable_names=tuple(str(item["name"]) for item in callable_tools),
    )


def _load_local_api_key(name: str, *, env_file: str | Path | None = None) -> str:
    """Read one credential from the process environment or ignored local .env.

    Retained as the environment-and-file half of the resolution order; the
    reader itself now lives in `planner_credentials` so the credential store
    can share it without an import cycle.
    """

    return read_environment_or_env_file(name, env_file=env_file)


def _load_planner_api_key(
    name: str,
    provider_id: str,
    *,
    env_file: str | Path | None = None,
) -> str:
    """Resolve one provider's key: environment, then `.env`, then the OS store.

    An explicitly supplied `env_file` scopes the lookup to the environment and
    that one file. Callers pass it to say exactly where the credential should
    come from, and silently reaching past it into the machine's credential
    store would make that request meaningless.
    """

    direct = read_environment_or_env_file(name, env_file=env_file)
    if direct or env_file is not None:
        return direct
    return read_stored_credential(provider_id)


def load_gemini_api_key(*, env_file: str | Path | None = None) -> str:
    """Read Gemini credentials from the environment, .env, then the OS store."""

    return _load_planner_api_key("GEMINI_API_KEY", GEMINI, env_file=env_file)


def load_groq_api_key(*, env_file: str | Path | None = None) -> str:
    """Read Groq credentials from the environment, .env, then the OS store."""

    return _load_planner_api_key("GROQ_API_KEY", GROQ, env_file=env_file)


def load_openrouter_api_key(*, env_file: str | Path | None = None) -> str:
    """Read OpenRouter credentials from the environment, .env, then the OS store."""

    return (
        read_environment_or_env_file("OPENROUTER_API_KEY", env_file=env_file)
        or read_environment_or_env_file("OPEN_ROUTER_API_KEY", env_file=env_file)
        or ("" if env_file is not None else read_stored_credential(OPENROUTER))
    )


def _tool_call_schema_variants(callable_tools: tuple[Mapping[str, Any], ...]) -> list[dict[str, Any]]:
    call_variants = []
    for tool in callable_tools:
        name = str(tool["name"])
        arguments = dict(tool["arguments"])
        call_variants.append({
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "arguments"],
            "properties": {
                "name": {"type": "string", "const": name},
                "arguments": arguments,
            },
        })
    return call_variants


def plan_json_schema(callable_tools: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    call_variants = _tool_call_schema_variants(callable_tools)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "status", "message", "calls"],
        "properties": {
            "schema_version": {"type": "integer", "const": PLAN_SCHEMA_VERSION},
            "status": {"type": "string", "enum": ["execute", "clarify", "refuse"]},
            "message": {"type": "string", "maxLength": 1200},
            "calls": {
                "type": "array",
                "maxItems": MAX_PLANNER_CALLS,
                "items": {"oneOf": call_variants},
            },
        },
    }


def agent_step_json_schema(callable_tools: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    """Return the separate AgentStep v1 envelope over unchanged tool schemas."""

    call_variants = _tool_call_schema_variants(callable_tools)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "status", "message", "calls"],
        "properties": {
            "schema_version": {"type": "integer", "const": AGENT_STEP_SCHEMA_VERSION},
            "status": {"type": "string", "enum": ["execute", "finish", "clarify", "refuse"]},
            "engineering_disposition": {
                "type": "object",
                "additionalProperties": False,
                "required": ["semantic_design_hash", "acknowledged_observation_ids", "deferred"],
                "properties": {
                    "semantic_design_hash": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
                    "acknowledged_observation_ids": {"type": "array", "items": {"type": "string", "minLength": 1}},
                    "deferred": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["observation_id", "reason"],
                        "properties": {
                            "observation_id": {"type": "string", "minLength": 1},
                            "reason": {"type": "string", "minLength": 1, "maxLength": 400},
                        },
                    }},
                },
            },
            "memory_proposals": {
                "type": "array",
                "maxItems": MAX_MEMORY_PROPOSALS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "action", "semantic_kind", "key", "value", "unit",
                        "evidence_quote", "reference_id",
                    ],
                    "properties": {
                        "action": {"type": "string", "enum": ["upsert", "supersede", "resolve"]},
                        "semantic_kind": {
                            "type": "string",
                            "enum": [
                                "requirement", "constraint", "preference", "goal",
                                "future_intent", "priority",
                            ],
                        },
                        "key": {"type": "string", "pattern": "^[a-z][a-z0-9_]{1,63}$"},
                        "value": {"type": ["string", "number", "null"], "maxLength": 300},
                        "unit": {"type": ["string", "null"], "maxLength": 32},
                        "evidence_quote": {"type": "string", "minLength": 1, "maxLength": 400},
                        "reference_id": {"type": ["string", "null"], "maxLength": 160},
                    },
                },
            },
            "message": {"type": "string", "maxLength": 1200},
            "calls": {
                "type": "array",
                "maxItems": MAX_PLANNER_CALLS,
                "items": {"oneOf": call_variants},
            },
        },
    }


def _decode_planner_payload(payload: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload, str):
        cleaned = payload.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()
        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise CapabilityError("The LLM planner returned malformed JSON.") from exc
        if not isinstance(raw, dict):
            raise CapabilityError("The LLM planner response must be a JSON object.")
        return raw
    return dict(payload)


def _planned_call_from_mapping(payload: Mapping[str, Any]) -> PlannedToolCall:
    if not isinstance(payload, Mapping) or set(payload) != {"name", "arguments"}:
        raise ValueError("A planned tool call does not match the required schema.")
    name = payload["name"]
    arguments = payload["arguments"]
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise ValueError("A planned tool call has invalid name or arguments.")
    return PlannedToolCall(name, dict(arguments))


def _parse_registered_calls(
    raw_calls: Any,
    *,
    callable_tool_names: tuple[str, ...],
) -> tuple[PlannedToolCall, ...]:
    if not isinstance(raw_calls, list) or len(raw_calls) > MAX_PLANNER_CALLS:
        raise CapabilityError("The LLM planner returned an invalid tool-call list.")
    allowed = set(callable_tool_names)
    calls: list[PlannedToolCall] = []
    call_targets: set[tuple[str, str]] = set()
    for item in raw_calls:
        try:
            call = _planned_call_from_mapping(item)
        except ValueError as exc:
            raise CapabilityError(str(exc)) from exc
        name, arguments = call.name, call.arguments
        if name not in allowed:
            raise CapabilityError(f"The LLM requested an unregistered planning tool: {name!r}.")
        target = ""
        if name in {"parameter.set", "parameter.create"}:
            target = str(arguments.get("key", ""))
        elif name == "recipe.select":
            target = "recipe"
        elif name == "design.reset":
            target = "design"
        elif name in {"modifier.apply", "modifier.remove"}:
            target = str(arguments.get("modifier_id", ""))
        elif name in {"geometry.rectangle_sheet", "geometry.cylinder", "geometry.circle_sheet"}:
            target = str(arguments.get("object_id", ""))
        elif name == "geometry.translate":
            target = str(arguments.get("object_id", ""))
        elif name == "geometry.duplicate":
            target = str(arguments.get("new_id", ""))
        elif name in {"boolean.subtract", "boolean.union"}:
            target = str(arguments.get("operation_id", ""))
        signature = (name, target)
        if signature in call_targets:
            raise CapabilityError(f"The LLM plan targets {name} {target!r} more than once.")
        call_targets.add(signature)
        calls.append(call)
    return tuple(calls)


def parse_agent_step(
    payload: str | Mapping[str, Any],
    *,
    callable_tool_names: tuple[str, ...],
) -> AgentStep:
    """Strictly parse the new model-emitted AgentStep control envelope."""

    raw = _decode_planner_payload(payload)
    required = {"schema_version", "status", "message", "calls"}
    allowed = required | {"engineering_disposition", "memory_proposals"}
    if not required.issubset(raw) or not set(raw).issubset(allowed):
        raise CapabilityError("The LLM agent response does not match the required AgentStep schema.")
    if raw["schema_version"] != AGENT_STEP_SCHEMA_VERSION:
        raise CapabilityError("The LLM agent returned an unsupported AgentStep schema version.")
    status = raw["status"]
    if status not in {"execute", "finish", "clarify", "refuse"}:
        raise CapabilityError("The LLM agent returned an invalid AgentStep status.")
    if not isinstance(raw["message"], str) or len(raw["message"]) > 1200:
        raise CapabilityError("The LLM agent message is invalid.")
    calls = _parse_registered_calls(raw["calls"], callable_tool_names=callable_tool_names)
    if status == "execute" and not calls:
        raise CapabilityError("An execute AgentStep must contain at least one registered tool call.")
    if status != "execute" and calls:
        raise CapabilityError(f"An AgentStep with status {status!r} must contain zero tool calls.")
    try:
        disposition = (EngineeringDisposition.from_dict(raw["engineering_disposition"])
                       if "engineering_disposition" in raw else None)
        raw_proposals = raw.get("memory_proposals", [])
        if not isinstance(raw_proposals, list) or len(raw_proposals) > MAX_MEMORY_PROPOSALS:
            raise ValueError("AgentStep semantic-memory proposals are invalid.")
        proposals = tuple(
            SemanticMemoryProposal.from_unvalidated(item)
            for item in raw_proposals
            if isinstance(item, Mapping)
        )
        if len(proposals) != len(raw_proposals):
            raise ValueError("AgentStep semantic-memory proposals are invalid.")
        if any(isinstance(item.value, bool) for item in proposals):
            raise ValueError(
                "Semantic-memory proposal values must contain the stated semantic value; "
                "boolean placeholders are not allowed."
            )
        return AgentStep(status=status, message=raw["message"].strip(), calls=calls,
                         engineering_disposition=disposition, memory_proposals=proposals)
    except ValueError as exc:
        raise CapabilityError(str(exc)) from exc


def parse_llm_tool_plan(payload: str | Mapping[str, Any], *, callable_tool_names: tuple[str, ...]) -> LLMToolPlan:
    """Strictly parse a model response before any registered tool executes."""

    raw = _decode_planner_payload(payload)
    if set(raw) != {"schema_version", "status", "message", "calls"}:
        raise CapabilityError("The LLM planner response does not match the required plan schema.")
    if raw["schema_version"] != PLAN_SCHEMA_VERSION:
        raise CapabilityError("The LLM planner returned an unsupported plan schema version.")
    status = raw["status"]
    if status not in {"execute", "clarify", "refuse"}:
        raise CapabilityError("The LLM planner returned an invalid plan status.")
    if not isinstance(raw["message"], str) or len(raw["message"]) > 1200:
        raise CapabilityError("The LLM planner message is invalid.")
    calls = _parse_registered_calls(raw["calls"], callable_tool_names=callable_tool_names)
    if status == "execute" and not calls:
        raise CapabilityError("An executable LLM plan must contain at least one registered tool call.")
    if status != "execute" and calls:
        raise CapabilityError("Clarification and refusal plans cannot contain tool calls.")
    return LLMToolPlan(status, raw["message"].strip(), calls)


def enforce_request_capabilities(
    plan: LLMToolPlan,
    *,
    instruction: str,
    capability_manifest: Mapping[str, Any],
) -> LLMToolPlan:
    """Refuse explicit capability mismatches even when a small model substitutes."""

    if SOLVER_EXECUTION_PATTERN.search(instruction):
        return LLMToolPlan(
            "refuse",
            "Starting CST, a solver, or a simulation is not available from the antenna design agent.",
            (),
        )
    installed_words = " ".join(
        str(value).casefold()
        for recipe in capability_manifest.get("recipes", ())
        if isinstance(recipe, Mapping)
        for value in (
            recipe.get("recipe_id", ""),
            recipe.get("family", ""),
            recipe.get("display_name", ""),
            *(recipe.get("aliases", ()) if isinstance(recipe.get("aliases"), list) else ()),
        )
    )
    requested = instruction.casefold()
    unavailable = next(
        (
            term for term in KNOWN_FAMILY_TERMS
            if re.search(rf"\b{re.escape(term)}\b", requested)
            and not re.search(rf"\b{re.escape(term)}\b", installed_words)
        ),
        None,
    )
    if unavailable:
        return LLMToolPlan(
            "refuse",
            f"The requested {unavailable} antenna family is not installed as a validated recipe.",
            (),
        )
    return plan


def _enforce_agent_step_capabilities(
    step: AgentStep,
    *,
    instruction: str,
    capability_manifest: Mapping[str, Any],
) -> AgentStep:
    """Apply the existing deterministic capability guard to an AgentStep."""

    if step.status == "finish":
        return step
    checked = enforce_request_capabilities(
        LLMToolPlan(step.status, step.message, step.calls),
        instruction=instruction,
        capability_manifest=capability_manifest,
    )
    if checked.status == step.status and checked.message == step.message and checked.calls == step.calls:
        return step
    return AgentStep(
        checked.status,
        checked.message,
        checked.calls,
        engineering_disposition=step.engineering_disposition,
        memory_proposals=step.memory_proposals,
    )


class _AgentLoopPlannerMixin:
    """Expose AgentStep decoding through an existing provider transport."""

    _agent_loop_remaining_budgets: AgentLoopBudgets | None = None
    _agent_loop_engineering_report: EngineeringReport | None = None

    def plan_agent_step(
        self,
        *,
        instruction: str,
        current_design: Mapping[str, Any] | None,
        capability_manifest: Mapping[str, Any],
        remaining_budgets: AgentLoopBudgets,
        project_memory: Mapping[str, Any] | None = None,
        agent_observation: AgentStepObservation | None = None,
        engineering_report: EngineeringReport | None = None,
    ) -> AgentStep:
        self._agent_loop_remaining_budgets = remaining_budgets
        self._agent_loop_engineering_report = engineering_report
        try:
            result = self.plan(
                instruction=instruction,
                current_design=current_design,
                capability_manifest=capability_manifest,
                project_memory=project_memory,
                agent_observation=agent_observation,
            )
        finally:
            self._agent_loop_remaining_budgets = None
            self._agent_loop_engineering_report = None
        if not isinstance(result, AgentStep):
            raise CapabilityError("The provider did not return a valid AgentStep.")
        return result

    def _exchange(
        self,
        *,
        instruction: str,
        current_design: Mapping[str, Any] | None,
        capability_manifest: Mapping[str, Any],
        project_memory: Mapping[str, Any] | None,
        agent_observation: AgentStepObservation | None,
        execution_feedback: Mapping[str, Any] | None,
    ) -> PlannerExchange:
        budgets = self._agent_loop_remaining_budgets
        if budgets is not None:
            return build_agent_step_exchange(
                instruction=instruction,
                current_design=current_design,
                capability_manifest=capability_manifest,
                project_memory=project_memory,
                agent_observation=agent_observation,
                remaining_budgets=budgets,
                engineering_report=self._agent_loop_engineering_report,
            )
        return build_planner_exchange(
            instruction=instruction,
            current_design=current_design,
            capability_manifest=capability_manifest,
            project_memory=project_memory,
            agent_observation=agent_observation,
            execution_feedback=execution_feedback,
        )

    def _parse_response(
        self,
        content: str,
        *,
        exchange: PlannerExchange,
        instruction: str,
        capability_manifest: Mapping[str, Any],
    ) -> LLMToolPlan | AgentStep:
        if self._agent_loop_remaining_budgets is not None:
            return _enforce_agent_step_capabilities(
                parse_agent_step(content, callable_tool_names=exchange.callable_names),
                instruction=instruction,
                capability_manifest=capability_manifest,
            )
        parsed = parse_llm_tool_plan(content, callable_tool_names=exchange.callable_names)
        return enforce_request_capabilities(
            parsed,
            instruction=instruction,
            capability_manifest=capability_manifest,
        )

    def _schema_repair_instruction(self, error: Exception) -> str:
        if self._agent_loop_remaining_budgets is not None:
            statuses = "Use execute with registered calls, or finish/clarify/refuse with an empty calls array."
        else:
            statuses = "Use execute with registered calls, or clarify/refuse with an empty calls array."
        return (
            f"That plan failed deterministic validation: {error} "
            "Return a corrected object matching the supplied schema exactly. "
            f"Do not add fields. {statuses}"
        )


class SchemaConstrainedLLMPlanner(_AgentLoopPlannerMixin):
    """Ask a loopback Ollama model for a strict registered-tool plan."""

    def __init__(self, model: str, *, base_url: str, timeout: int = 60, opener: Callable[..., Any] = urllib.request.urlopen) -> None:
        from studio.assistant import local_ollama_base_url

        self.model = model.strip()
        self.base_url = local_ollama_base_url(base_url)
        self.timeout = timeout
        self._opener = opener
        self.backend_id = "local_qwen"
        self.last_run_metadata: dict[str, Any] = {}

    def plan(
        self,
        *,
        instruction: str,
        current_design: Mapping[str, Any] | None,
        capability_manifest: Mapping[str, Any],
        project_memory: Mapping[str, Any] | None = None,
        agent_observation: AgentStepObservation | None = None,
        execution_feedback: Mapping[str, Any] | None = None,
    ) -> LLMToolPlan:
        if not self.model:
            raise CapabilityError("Choose a local Ollama model before using the antenna design agent.")
        exchange = self._exchange(
            instruction=instruction,
            current_design=current_design,
            capability_manifest=capability_manifest,
            project_memory=project_memory,
            agent_observation=agent_observation,
            execution_feedback=execution_feedback,
        )
        messages = [
            {"role": "system", "content": exchange.system_instruction},
            {"role": "user", "content": exchange.user_content},
        ]
        self.last_run_metadata = {
            "backend": self.backend_id,
            "model": self.model,
            "schema_enforcement": "provider_constrained_decoding_and_post_generation_validation",
            "schema_repairs": 0,
            "executor_repair": execution_feedback is not None,
            "returned_plan_attempts": [],
        }
        last_error: CapabilityError | None = None
        for attempt in range(2):
            request = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps({
                    "model": self.model,
                    "messages": messages,
                    "format": exchange.schema,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.0, "num_ctx": 16384},
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                raise CapabilityError("The local LLM tool planner is unavailable. Start the configured Ollama model and try again.") from exc
            message = body.get("message", {}) if isinstance(body, dict) else {}
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str):
                raise CapabilityError("The local LLM tool planner returned no structured plan.")
            try:
                recorded_content: Any = json.loads(content)
            except json.JSONDecodeError:
                recorded_content = content
            self.last_run_metadata["returned_plan_attempts"].append(recorded_content)
            try:
                return self._parse_response(
                    content,
                    exchange=exchange,
                    instruction=instruction,
                    capability_manifest=capability_manifest,
                )
            except CapabilityError as exc:
                last_error = exc
                if attempt == 0:
                    self.last_run_metadata["schema_repairs"] = 1
                    messages.extend((
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": self._schema_repair_instruction(exc)},
                    ))
        raise CapabilityError(f"The local LLM tool planner returned an invalid plan twice: {last_error}")

    def repair(
        self,
        *,
        instruction: str,
        current_design: Mapping[str, Any] | None,
        capability_manifest: Mapping[str, Any],
        failed_plan: LLMToolPlan,
        error: Exception,
        project_memory: Mapping[str, Any] | None = None,
    ) -> LLMToolPlan:
        return self.plan(
            instruction=instruction,
            current_design=current_design,
            capability_manifest=capability_manifest,
            project_memory=project_memory,
            execution_feedback={
                "error": str(error),
                "plan": {
                    "schema_version": PLAN_SCHEMA_VERSION,
                    "status": failed_plan.status,
                    "message": failed_plan.message,
                    "calls": [
                        {"name": call.name, "arguments": call.arguments}
                        for call in failed_plan.calls
                    ],
                },
            },
        )


class GeminiSchemaConstrainedPlanner(_AgentLoopPlannerMixin):
    """Use Gemini as a cloud transport for the identical planner contract."""

    DEFAULT_MODEL = "gemini-3.8-flash"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        env_file: str | Path | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 90,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        normalized_url = base_url.rstrip("/")
        if normalized_url != self.DEFAULT_BASE_URL:
            raise CapabilityError("Gemini planner requests are restricted to the official Google API endpoint.")
        self.model = (model or os.environ.get("GEMINI_MODEL") or self.DEFAULT_MODEL).strip()
        self.api_key = (api_key or load_gemini_api_key(env_file=env_file)).strip()
        self.base_url = normalized_url
        self.timeout = timeout
        self._opener = opener
        self.backend_id = "gemini_cloud"
        self.last_run_metadata: dict[str, Any] = {}

    @staticmethod
    def _response_text(body: Any) -> str | None:
        candidates = body.get("candidates", ()) if isinstance(body, dict) else ()
        if not isinstance(candidates, list) or not candidates:
            return None
        content = candidates[0].get("content", {}) if isinstance(candidates[0], dict) else {}
        parts = content.get("parts", ()) if isinstance(content, dict) else ()
        if not isinstance(parts, list):
            return None
        text_parts = [part.get("text") for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
        return "".join(text_parts) if text_parts else None

    def plan(
        self,
        *,
        instruction: str,
        current_design: Mapping[str, Any] | None,
        capability_manifest: Mapping[str, Any],
        project_memory: Mapping[str, Any] | None = None,
        agent_observation: AgentStepObservation | None = None,
        execution_feedback: Mapping[str, Any] | None = None,
    ) -> LLMToolPlan:
        if not self.api_key:
            raise CapabilityError(
                "Gemini cloud planner requires GEMINI_API_KEY. Set it in the environment or in the ignored project .env file; Local Ollama remains available."
            )
        if not self.model:
            raise CapabilityError("Choose a Gemini model before using the cloud planner.")
        exchange = self._exchange(
            instruction=instruction,
            current_design=current_design,
            capability_manifest=capability_manifest,
            project_memory=project_memory,
            agent_observation=agent_observation,
            execution_feedback=execution_feedback,
        )
        messages = [{"role": "user", "content": exchange.user_content}]
        self.last_run_metadata = {
            "backend": self.backend_id,
            "model": self.model,
            "schema_enforcement": "post_generation_strict_parsing_and_validation",
            "schema_repairs": 0,
            "executor_repair": execution_feedback is not None,
            "returned_plan_attempts": [],
        }
        last_error: CapabilityError | None = None
        for attempt in range(2):
            contents = [
                {
                    "role": "model" if message["role"] == "assistant" else "user",
                    "parts": [{"text": message["content"]}],
                }
                for message in messages
            ]
            request = urllib.request.Request(
                f"{self.base_url}/v1beta/models/{urllib.parse.quote(self.model, safe='-_.')}:generateContent",
                data=json.dumps({
                    "systemInstruction": {"parts": [{"text": exchange.system_instruction}]},
                    "contents": contents,
                    "generationConfig": {
                        "responseMimeType": "application/json",
                    },
                }).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                method="POST",
            )
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    error_body = json.loads(exc.read().decode("utf-8"))
                    error_payload = error_body.get("error", {}) if isinstance(error_body, dict) else {}
                    if isinstance(error_payload, dict) and isinstance(error_payload.get("message"), str):
                        detail = error_payload["message"].replace(self.api_key, "[redacted]").strip()
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    pass
                suffix = f": {detail}" if detail else ". Check GEMINI_API_KEY and the configured Gemini model."
                raise CapabilityError(
                    f"Gemini cloud planner request failed with HTTP {exc.code}{suffix}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                raise CapabilityError("The Gemini cloud planner is unavailable. Check the network connection and try again.") from exc
            content = self._response_text(body)
            if not isinstance(content, str):
                raise CapabilityError("The Gemini cloud planner returned no structured plan.")
            try:
                recorded_content: Any = json.loads(content)
            except json.JSONDecodeError:
                recorded_content = content
            self.last_run_metadata["returned_plan_attempts"].append(recorded_content)
            try:
                return self._parse_response(
                    content,
                    exchange=exchange,
                    instruction=instruction,
                    capability_manifest=capability_manifest,
                )
            except CapabilityError as exc:
                last_error = exc
                if attempt == 0:
                    self.last_run_metadata["schema_repairs"] = 1
                    messages.extend((
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": self._schema_repair_instruction(exc)},
                    ))
        raise CapabilityError(f"The Gemini cloud planner returned an invalid plan twice: {last_error}")

    def repair(
        self,
        *,
        instruction: str,
        current_design: Mapping[str, Any] | None,
        capability_manifest: Mapping[str, Any],
        failed_plan: LLMToolPlan,
        error: Exception,
        project_memory: Mapping[str, Any] | None = None,
    ) -> LLMToolPlan:
        return self.plan(
            instruction=instruction,
            current_design=current_design,
            capability_manifest=capability_manifest,
            project_memory=project_memory,
            execution_feedback={
                "error": str(error),
                "plan": {
                    "schema_version": PLAN_SCHEMA_VERSION,
                    "status": failed_plan.status,
                    "message": failed_plan.message,
                    "calls": [
                        {"name": call.name, "arguments": call.arguments}
                        for call in failed_plan.calls
                    ],
                },
            },
        )


class GroqSchemaConstrainedPlanner(_AgentLoopPlannerMixin):
    """Use Groq GPT-OSS as a cloud transport for the identical planner contract."""

    DEFAULT_MODEL = "openai/gpt-oss-120b"
    DEFAULT_BASE_URL = "https://api.groq.com"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        env_file: str | Path | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 90,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        normalized_url = base_url.rstrip("/")
        if normalized_url != self.DEFAULT_BASE_URL:
            raise CapabilityError("Groq planner requests are restricted to the official Groq API endpoint.")
        self.model = (model or self.DEFAULT_MODEL).strip()
        self.api_key = (api_key or load_groq_api_key(env_file=env_file)).strip()
        self.base_url = normalized_url
        self.timeout = timeout
        self._opener = opener
        self.backend_id = "groq_gpt_oss_120b_cloud"
        self.last_run_metadata: dict[str, Any] = {}
        self._strict_schema_supported: bool | None = None
        self._strict_schema_rejection_reason = ""

    @staticmethod
    def _response_text(body: Any) -> str | None:
        choices = body.get("choices", ()) if isinstance(body, dict) else ()
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        return content if isinstance(content, str) else None

    def _read_http_error(self, exc: urllib.error.HTTPError) -> str:
        detail = ""
        try:
            error_body = json.loads(exc.read().decode("utf-8"))
            error_payload = error_body.get("error", {}) if isinstance(error_body, dict) else {}
            if isinstance(error_payload, dict) and isinstance(error_payload.get("message"), str):
                detail = error_payload["message"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        sanitized = detail.replace(self.api_key, "[redacted]")
        sanitized = re.sub(r"\borg_[A-Za-z0-9]+\b", "[organization]", sanitized)
        return sanitized.strip()

    @staticmethod
    def _is_exact_schema_rejection(code: int, detail: str) -> bool:
        normalized = detail.casefold()
        if code == 400:
            return any(term in normalized for term in ("json_schema", "response_format", "schema"))
        if code == 413:
            return "request too large" in normalized or "tokens per minute" in normalized
        return False

    def _request_body(
        self,
        *,
        messages: list[dict[str, str]],
        schema: Mapping[str, Any],
        strict_schema: bool,
    ) -> dict[str, Any]:
        response_format: dict[str, Any]
        if strict_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "antenna_tool_plan",
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            response_format = {"type": "json_object"}
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "stream": False,
            "max_completion_tokens": 1024,
            "response_format": response_format,
        }

    def _send(
        self,
        *,
        messages: list[dict[str, str]],
        schema: Mapping[str, Any],
        strict_schema: bool,
    ) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/openai/v1/chat/completions",
            data=json.dumps(self._request_body(
                messages=messages,
                schema=schema,
                strict_schema=strict_schema,
            )).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "AntennaSurrogateStudio/experimental",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = self._read_http_error(exc)
            if strict_schema and self._is_exact_schema_rejection(exc.code, detail):
                reason = (
                    "Provider token limit rejected the unchanged strict-schema request."
                    if exc.code == 413
                    else "Provider rejected the unchanged ToolPlan schema in strict mode."
                )
                raise _GroqExactSchemaRejected(reason) from exc
            suffix = f": {detail}" if detail else ". Check GROQ_API_KEY and the configured Groq model."
            raise CapabilityError(
                f"Groq cloud planner request failed with HTTP {exc.code}{suffix}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise CapabilityError("The Groq cloud planner is unavailable. Check the network connection and try again.") from exc
        content = self._response_text(body)
        if not isinstance(content, str):
            raise CapabilityError("The Groq cloud planner returned no structured plan.")
        return content

    def plan(
        self,
        *,
        instruction: str,
        current_design: Mapping[str, Any] | None,
        capability_manifest: Mapping[str, Any],
        project_memory: Mapping[str, Any] | None = None,
        agent_observation: AgentStepObservation | None = None,
        execution_feedback: Mapping[str, Any] | None = None,
    ) -> LLMToolPlan:
        if not self.api_key:
            raise CapabilityError(
                "Groq cloud planner requires GROQ_API_KEY. Set it in the environment or in the ignored project .env file; Local Ollama remains available."
            )
        if not self.model:
            raise CapabilityError("Choose a Groq model before using the cloud planner.")
        exchange = self._exchange(
            instruction=instruction,
            current_design=current_design,
            capability_manifest=capability_manifest,
            project_memory=project_memory,
            agent_observation=agent_observation,
            execution_feedback=execution_feedback,
        )
        messages = [
            {"role": "system", "content": exchange.system_instruction},
            {"role": "user", "content": exchange.user_content},
        ]
        strict_schema = self._strict_schema_supported is not False
        self.last_run_metadata = {
            "backend": self.backend_id,
            "model": self.model,
            "schema_enforcement": (
                "provider_strict_json_schema_and_post_generation_validation"
                if strict_schema else "json_object_post_generation_strict_parsing_and_validation"
            ),
            "strict_schema_fallback": not strict_schema,
            "schema_repairs": 0,
            "executor_repair": execution_feedback is not None,
            "returned_plan_attempts": [],
        }
        if self._strict_schema_rejection_reason:
            self.last_run_metadata["strict_schema_rejection"] = self._strict_schema_rejection_reason
        last_error: CapabilityError | None = None
        for attempt in range(2):
            try:
                content = self._send(
                    messages=messages,
                    schema=exchange.schema,
                    strict_schema=strict_schema,
                )
                if strict_schema:
                    self._strict_schema_supported = True
            except _GroqExactSchemaRejected as exc:
                self._strict_schema_supported = False
                strict_schema = False
                self._strict_schema_rejection_reason = str(exc) or "The provider rejected the unchanged ToolPlan schema."
                self.last_run_metadata["schema_enforcement"] = "json_object_post_generation_strict_parsing_and_validation"
                self.last_run_metadata["strict_schema_fallback"] = True
                self.last_run_metadata["strict_schema_rejection"] = self._strict_schema_rejection_reason
                content = self._send(
                    messages=messages,
                    schema=exchange.schema,
                    strict_schema=False,
                )
            try:
                recorded_content: Any = json.loads(content)
            except json.JSONDecodeError:
                recorded_content = content
            self.last_run_metadata["returned_plan_attempts"].append(recorded_content)
            try:
                return self._parse_response(
                    content,
                    exchange=exchange,
                    instruction=instruction,
                    capability_manifest=capability_manifest,
                )
            except CapabilityError as exc:
                last_error = exc
                if attempt == 0:
                    self.last_run_metadata["schema_repairs"] = 1
                    messages.extend((
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": self._schema_repair_instruction(exc)},
                    ))
        raise CapabilityError(f"The Groq cloud planner returned an invalid plan twice: {last_error}")

    def repair(
        self,
        *,
        instruction: str,
        current_design: Mapping[str, Any] | None,
        capability_manifest: Mapping[str, Any],
        failed_plan: LLMToolPlan,
        error: Exception,
        project_memory: Mapping[str, Any] | None = None,
    ) -> LLMToolPlan:
        return self.plan(
            instruction=instruction,
            current_design=current_design,
            capability_manifest=capability_manifest,
            project_memory=project_memory,
            execution_feedback={
                "error": str(error),
                "plan": {
                    "schema_version": PLAN_SCHEMA_VERSION,
                    "status": failed_plan.status,
                    "message": failed_plan.message,
                    "calls": [
                        {"name": call.name, "arguments": call.arguments}
                        for call in failed_plan.calls
                    ],
                },
            },
        )


class _GroqExactSchemaRejected(Exception):
    """Internal signal to retry the same exchange in JSON object mode."""


class OpenRouterNemotronPlanner(_AgentLoopPlannerMixin):
    """Use OpenRouter Nemotron as a cloud transport for the shared planner contract."""

    DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
    DEFAULT_BASE_URL = "https://openrouter.ai"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        env_file: str | Path | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 180,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        normalized_url = base_url.rstrip("/")
        if normalized_url != self.DEFAULT_BASE_URL:
            raise CapabilityError("OpenRouter planner requests are restricted to the official OpenRouter API endpoint.")
        self.model = (model or self.DEFAULT_MODEL).strip()
        self.api_key = (api_key or load_openrouter_api_key(env_file=env_file)).strip()
        self.base_url = normalized_url
        self.timeout = timeout
        self._opener = opener
        self.backend_id = "openrouter_nemotron_3_ultra_free_cloud"
        self.last_run_metadata: dict[str, Any] = {}

    @staticmethod
    def _response_text(body: Any) -> str | None:
        choices = body.get("choices", ()) if isinstance(body, dict) else ()
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        return content if isinstance(content, str) else None

    def _read_http_error(self, exc: urllib.error.HTTPError) -> str:
        detail = ""
        try:
            error_body = json.loads(exc.read().decode("utf-8"))
            error_payload = error_body.get("error", {}) if isinstance(error_body, dict) else {}
            if isinstance(error_payload, dict) and isinstance(error_payload.get("message"), str):
                detail = error_payload["message"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        return detail.replace(self.api_key, "[redacted]").strip()

    def plan(
        self,
        *,
        instruction: str,
        current_design: Mapping[str, Any] | None,
        capability_manifest: Mapping[str, Any],
        project_memory: Mapping[str, Any] | None = None,
        agent_observation: AgentStepObservation | None = None,
        execution_feedback: Mapping[str, Any] | None = None,
    ) -> LLMToolPlan:
        if not self.api_key:
            raise CapabilityError(
                "OpenRouter Nemotron planner requires OPENROUTER_API_KEY. Set it in the environment or in the ignored project .env file; Local Ollama remains available."
            )
        if not self.model:
            raise CapabilityError("Choose an OpenRouter model before using the cloud planner.")
        exchange = self._exchange(
            instruction=instruction,
            current_design=current_design,
            capability_manifest=capability_manifest,
            project_memory=project_memory,
            agent_observation=agent_observation,
            execution_feedback=execution_feedback,
        )
        messages = [
            {"role": "system", "content": exchange.system_instruction},
            {"role": "user", "content": exchange.user_content},
        ]
        self.last_run_metadata = {
            "backend": self.backend_id,
            "model": self.model,
            "schema_enforcement": "post_generation_strict_parsing_and_validation_no_provider_schema_enforcement",
            "schema_repairs": 0,
            "executor_repair": execution_feedback is not None,
            "returned_plan_attempts": [],
        }
        last_error: CapabilityError | None = None
        for attempt in range(2):
            request = urllib.request.Request(
                f"{self.base_url}/api/v1/chat/completions",
                data=json.dumps({
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.0,
                    "stream": False,
                }).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "AntennaSurrogateStudio/experimental",
                },
                method="POST",
            )
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = self._read_http_error(exc)
                suffix = f": {detail}" if detail else ". Check OPENROUTER_API_KEY and free-model availability."
                raise CapabilityError(
                    f"OpenRouter Nemotron planner request failed with HTTP {exc.code}{suffix}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                raise CapabilityError(
                    "The OpenRouter Nemotron planner is unavailable. Check the network connection and try again."
                ) from exc
            content = self._response_text(body)
            if not isinstance(content, str):
                raise CapabilityError("The OpenRouter Nemotron planner returned no structured plan.")
            try:
                recorded_content: Any = json.loads(content)
            except json.JSONDecodeError:
                recorded_content = content
            self.last_run_metadata["returned_plan_attempts"].append(recorded_content)
            try:
                return self._parse_response(
                    content,
                    exchange=exchange,
                    instruction=instruction,
                    capability_manifest=capability_manifest,
                )
            except CapabilityError as exc:
                last_error = exc
                if attempt == 0:
                    self.last_run_metadata["schema_repairs"] = 1
                    messages.extend((
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": self._schema_repair_instruction(exc)},
                    ))
        raise CapabilityError(f"The OpenRouter Nemotron planner returned an invalid plan twice: {last_error}")

    def repair(
        self,
        *,
        instruction: str,
        current_design: Mapping[str, Any] | None,
        capability_manifest: Mapping[str, Any],
        failed_plan: LLMToolPlan,
        error: Exception,
        project_memory: Mapping[str, Any] | None = None,
    ) -> LLMToolPlan:
        return self.plan(
            instruction=instruction,
            current_design=current_design,
            capability_manifest=capability_manifest,
            project_memory=project_memory,
            execution_feedback={
                "error": str(error),
                "plan": {
                    "schema_version": PLAN_SCHEMA_VERSION,
                    "status": failed_plan.status,
                    "message": failed_plan.message,
                    "calls": [
                        {"name": call.name, "arguments": call.arguments}
                        for call in failed_plan.calls
                    ],
                },
            },
        )
