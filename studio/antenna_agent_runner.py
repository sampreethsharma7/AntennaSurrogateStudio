"""Bounded, unpublished agent-loop execution over the antenna tool registry."""

from __future__ import annotations

import copy
import re
from dataclasses import replace
from typing import Any, Mapping

from studio.antenna_agent import (
    AgentInstructionError,
    AntennaDesignAgent,
    CapabilityUnavailableError,
)
from studio.antenna_analysis import (
    normalize_analysis_arguments, semantic_design_hash, semantic_design_payload, stable_hash,
)
from studio.antenna_design import AntennaDesign, DesignValidationError
from studio.antenna_engineering import EngineeringCheckCoverage, EngineeringDesignRef, EngineeringReport
from studio.antenna_engineering_checks import CHECK_VERSIONS, SUITE_VERSION, run_engineering_checks
from studio.antenna_engineering_summary import engineering_report_hash, summarize_engineering_report
from studio.antenna_llm_planner import (
    AgentLoopBudgets,
    AgentLoopPlanner,
    AgentStep,
    AgentStepObservation,
    AgentTerminalResult,
    AgentTrajectoryEntry,
    LLMToolPlan,
    PlannedToolCall,
    validate_engineering_disposition,
)
from studio.antenna_tools import CapabilityError, ToolCall
from studio.antenna_validation import validate_design


_SENSITIVE_KEY_MARKERS = (
    "api_key", "apikey", "authorization", "cookie", "password", "secret", "token", "x_goog_api_key",
)


_stable_hash = stable_hash  # Compatibility export used by existing audit tests.


def rejected_action_fingerprint(
    design: AntennaDesign | None,
    calls: tuple[PlannedToolCall, ...],
) -> str:
    """Identify one ordered action batch against one semantic working state."""

    return _stable_hash({
        "working_state_hash": semantic_design_hash(design),
        "calls": [
            {"name": call.name, "arguments": dict(call.arguments)}
            for call in calls
        ],
    })


def _design_ref(design: AntennaDesign | None) -> dict[str, Any] | None:
    if design is None:
        return None
    return {
        "design_id": design.design_id,
        "revision": design.revision,
        "semantic_hash": semantic_design_hash(design),
    }


def _sanitize_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
    text = re.sub(
        r"(?i)(api[_-]?key|token|password|secret)(\s*[:=]\s*)[^\s,;}&]+",
        r"\1\2[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)([?&](?:key|api_key|token)=)[^&\s]+", r"\1[REDACTED]", text)
    return text[:1200]


def _safe_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if any(marker in normalized for marker in _SENSITIVE_KEY_MARKERS):
                safe[str(key)] = "[REDACTED]"
            else:
                safe[str(key)] = _safe_metadata(item)
        return safe
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _sanitize_text(value)


def _planner_metadata(planner: AgentLoopPlanner) -> dict[str, Any]:
    metadata = getattr(planner, "last_run_metadata", {})
    return _safe_metadata(metadata) if isinstance(metadata, Mapping) else {}


def _memory_payload(project_memory: Mapping[str, Any] | Any | None) -> dict[str, Any] | None:
    if project_memory is None:
        return None
    if isinstance(project_memory, Mapping):
        return copy.deepcopy(dict(project_memory))
    planner_converter = getattr(project_memory, "to_planner_dict", None)
    if callable(planner_converter):
        converted = planner_converter()
        if isinstance(converted, Mapping):
            return copy.deepcopy(dict(converted))
    converter = getattr(project_memory, "to_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return copy.deepcopy(dict(converted))
    raise TypeError("project_memory must be a mapping or expose to_dict().")


def _remaining_budgets(
    limits: AgentLoopBudgets,
    *,
    decisions: int,
    accepted: int,
    rejected: int,
    proposed_calls: int,
    analysis_batches: int,
) -> AgentLoopBudgets:
    return AgentLoopBudgets(
        decision_iterations=max(0, limits.decision_iterations - decisions),
        accepted_action_batches=max(0, limits.accepted_action_batches - accepted),
        rejected_action_batches=max(0, limits.rejected_action_batches - rejected),
        total_proposed_tool_calls=max(0, limits.total_proposed_tool_calls - proposed_calls),
        analysis_batches=max(0, limits.analysis_batches - analysis_batches),
    )


def _validation_result(design: AntennaDesign) -> dict[str, Any]:
    return {
        "status": "valid",
        "records": [
            {
                "stage": record.stage,
                "passed": record.passed,
                "messages": list(record.messages),
            }
            for record in design.validation
        ],
    }


def _actual_calls(calls: tuple[ToolCall, ...]) -> tuple[PlannedToolCall, ...]:
    return tuple(PlannedToolCall(call.name, dict(call.arguments)) for call in calls)


def _compact_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return {"count": len(value), "hash": _stable_hash(value)}
    if isinstance(value, Mapping):
        return {"keys": sorted(str(key) for key in value), "hash": _stable_hash(value)}
    return value


def deterministic_change_summary(
    before: AntennaDesign | None,
    after: AntennaDesign,
    executor_changes: tuple[str, ...] = (),
) -> tuple[dict[str, Any], ...]:
    """Describe semantic top-level differences without copying the full design."""

    previous = semantic_design_payload(before)
    current = semantic_design_payload(after)
    changes: list[dict[str, Any]] = []
    if previous is None:
        changes.append({
            "kind": "design_created",
            "recipe_id": after.recipe_id,
            "family": after.family,
        })
    else:
        for field_name in sorted(set(previous) | set(current)):
            old_value = previous.get(field_name)
            new_value = current.get(field_name)
            if old_value != new_value:
                changes.append({
                    "kind": "canonical_field_changed",
                    "field": field_name,
                    "before": _compact_value(old_value),
                    "after": _compact_value(new_value),
                })
    changes.extend(
        {"kind": "executor_summary", "summary": _sanitize_text(message)}
        for message in executor_changes
    )
    return tuple(changes)


def _failure_category(error: Exception) -> str:
    if isinstance(error, CapabilityUnavailableError):
        return "missing_capability"
    if isinstance(error, DesignValidationError):
        return "validation_rejected"
    if isinstance(error, AgentInstructionError):
        return "execution_rejected"
    if isinstance(error, CapabilityError):
        return "capability_rejected"
    return "execution_error"


class _TurnEngineeringReports:
    """Bounded by the turn's states; no persistent cache or canonical mutation."""

    def __init__(self) -> None:
        self.cache: dict[tuple, EngineeringReport] = {}
        self.recorded: set[str] = set()
        self.summaries = {}
        self.recorded_summaries: set[str] = set()

    def current(self, design: AntennaDesign | None) -> EngineeringReport:
        semantic_hash = semantic_design_hash(design)
        ref = EngineeringDesignRef(design.design_id, design.revision) if design else None
        key = (semantic_hash, SUITE_VERSION, tuple(sorted(CHECK_VERSIONS.items())))
        if key not in self.cache:
            if design is None:
                report = EngineeringReport(None, semantic_hash, SUITE_VERSION, coverage=tuple(
                    EngineeringCheckCoverage(name, version, "not_applicable", "No canonical design exists.",
                                             "No geometry or ports to inspect.")
                    for name, version in sorted(CHECK_VERSIONS.items())
                ))
            else:
                try:
                    report = run_engineering_checks(design)
                    if (report.working_design_ref != ref or report.semantic_design_hash != semantic_hash
                            or report.check_suite_version != SUITE_VERSION):
                        raise ValueError("Inspection returned a stale report.")
                except Exception as exc:
                    # A failed inspection is explicitly unknown evidence, never a clean result.
                    # Exception payloads may contain credentials: retain only the exception type.
                    report = EngineeringReport(ref, semantic_hash, SUITE_VERSION, coverage=tuple(
                        EngineeringCheckCoverage(name, version, "failed", "Current canonical design.",
                            f"Engineering inspection unavailable ({type(exc).__name__}); no clean conclusion is available.")
                        for name, version in sorted(CHECK_VERSIONS.items())
                    ))
            self.cache[key] = report
        cached = self.cache[key]
        # Semantic reuse across revisions must never carry the previous revision's reference.
        report = cached if cached.working_design_ref == ref else replace(cached, working_design_ref=ref)
        report_hash = engineering_report_hash(report)
        if report_hash not in self.summaries:
            self.summaries[report_hash] = summarize_engineering_report(report, design)
        return report

    def audit(self, before: EngineeringReport, after: EngineeringReport) -> dict[str, Any]:
        definitions = {}
        hashes = []
        summary_hashes, summaries = [], {}
        for report in (before, after):
            payload = report.to_dict()
            report_hash = _stable_hash(payload)
            hashes.append(report_hash)
            if report_hash not in self.recorded:
                definitions[report_hash] = payload
                self.recorded.add(report_hash)
            summary = self.summaries[report_hash]
            summary_hashes.append(summary.summary_hash)
            if summary.summary_hash not in self.recorded_summaries:
                summaries[summary.summary_hash] = {"summary": summary.to_dict(), "planner_context": summary.to_planner_dict()}
                self.recorded_summaries.add(summary.summary_hash)
        old = {item.observation_id: item.to_dict() for item in before.findings}
        new = {item.observation_id: item.to_dict() for item in after.findings}
        return {
            "before_report_hash": hashes[0],
            "after_report_hash": hashes[1],
            "before_geometry_hashes": sorted({item.source.evaluated_geometry_hash for item in before.findings
                                             if item.source.evaluated_geometry_hash}),
            "after_geometry_hashes": sorted({item.source.evaluated_geometry_hash for item in after.findings
                                            if item.source.evaluated_geometry_hash}),
            "reports": definitions,
            "before_summary_hash": summary_hashes[0],
            "after_summary_hash": summary_hashes[1],
            "summaries": summaries,
            "finding_delta": {
                "added": sorted(new.keys() - old.keys()),
                "removed": sorted(old.keys() - new.keys()),
                "changed": sorted(key for key in new.keys() & old.keys() if new[key] != old[key]),
            },
        }


class AntennaAgentRunner:
    """Run bounded planner/executor iterations without publishing project state."""

    def __init__(self, agent: AntennaDesignAgent) -> None:
        self.agent = agent

    def run(
        self,
        *,
        baseline_design: AntennaDesign | None,
        instruction: str,
        project_memory: Mapping[str, Any] | Any | None,
        planner: AgentLoopPlanner,
        budgets: AgentLoopBudgets | None = None,
    ) -> AgentTerminalResult:
        limits = budgets or AgentLoopBudgets()
        memory = _memory_payload(project_memory)
        baseline_hash = semantic_design_hash(baseline_design)
        working_design = baseline_design
        latest_observation: AgentStepObservation | None = None
        trajectory: list[AgentTrajectoryEntry] = []
        aggregate_calls: list[PlannedToolCall] = []
        aggregate_changes: list[dict[str, Any]] = []
        rejected_fingerprints: set[str] = set()
        accepted_state_hashes = {baseline_hash}
        decision_count = 0
        accepted_count = 0
        rejected_count = 0
        proposed_call_count = 0
        analysis_batch_count = 0
        analysis_cache: dict[tuple[str, str, str, str], Any] = {}
        engineering = _TurnEngineeringReports()
        baseline_report = engineering.current(baseline_design)
        current_report = baseline_report

        def record(entry: AgentTrajectoryEntry) -> None:
            # Rollback entries point back at the baseline; their abandoned input
            # report remains reconstructable audit evidence, not project memory.
            after_report = (
                current_report if entry.working_state_after == _design_ref(working_design)
                else baseline_report
            )
            trajectory.append(replace(entry, engineering_audit=engineering.audit(before_report, after_report)))

        def remaining() -> AgentLoopBudgets:
            return _remaining_budgets(
                limits,
                decisions=decision_count,
                accepted=accepted_count,
                rejected=rejected_count,
                proposed_calls=proposed_call_count,
                analysis_batches=analysis_batch_count,
            )

        def terminal(
            outcome: str,
            message: str,
            *,
            final_step: AgentStep | None = None,
            final_design: AntennaDesign | None = None,
        ) -> AgentTerminalResult:
            successful = outcome == "finished"
            authoritative = final_design if successful else baseline_design
            authoritative_hash = semantic_design_hash(authoritative)
            return AgentTerminalResult(
                outcome=outcome,
                message=_sanitize_text(message),
                working_design_ref=_design_ref(authoritative),
                working_design_hash=authoritative_hash,
                final_design=final_design if successful else None,
                aggregate_calls=tuple(aggregate_calls),
                aggregate_changes=tuple(aggregate_changes),
                has_publishable_change=(successful and authoritative_hash != baseline_hash),
                trajectory=tuple(trajectory),
                final_step=final_step,
            )

        while decision_count < limits.decision_iterations:
            before_design = working_design
            before_hash = semantic_design_hash(before_design)
            before_ref = _design_ref(before_design)
            before_report = current_report = engineering.current(before_design)
            planning_budgets = remaining()
            try:
                step = planner.plan_agent_step(
                    instruction=instruction,
                    current_design=(before_design.to_dict() if before_design is not None else None),
                    capability_manifest=self.agent.capability_manifest(before_design),
                    project_memory=copy.deepcopy(memory),
                    agent_observation=latest_observation,
                    remaining_budgets=planning_budgets,
                    engineering_report=before_report,
                )
            except Exception as exc:
                decision_count += 1
                safe_error = _sanitize_text(exc)
                record(AgentTrajectoryEntry(
                    iteration_index=decision_count,
                    working_state_before=before_ref,
                    returned_step=None,
                    planner_metadata=_planner_metadata(planner),
                    execution_status="provider_error",
                    executed_calls=(),
                    deterministic_changes=(),
                    sanitized_failure={"category": "provider_error", "error": safe_error},
                    working_state_after=before_ref,
                    observation=None,
                    budget_counters=remaining(),
                ))
                return terminal("provider_error", safe_error)

            decision_count += 1
            metadata = _planner_metadata(planner)
            if not isinstance(step, AgentStep):
                safe_error = "The planner did not return a valid AgentStep."
                record(AgentTrajectoryEntry(
                    iteration_index=decision_count,
                    working_state_before=before_ref,
                    returned_step=None,
                    planner_metadata=metadata,
                    execution_status="invalid_plan",
                    executed_calls=(),
                    deterministic_changes=(),
                    sanitized_failure={"category": "invalid_plan", "error": safe_error},
                    working_state_after=before_ref,
                    observation=None,
                    budget_counters=remaining(),
                ))
                return terminal("invalid_plan", safe_error)

            if step.status == "finish":
                try:
                    final_design = validate_design(working_design) if working_design is not None else None
                    current_report = engineering.current(final_design)
                    if any(item.severity == "blocking" for item in current_report.findings):
                        raise CapabilityError("Current engineering findings prohibit finish.")
                except (CapabilityError, DesignValidationError) as exc:
                    safe_error = _sanitize_text(exc)
                    record(AgentTrajectoryEntry(
                        iteration_index=decision_count,
                        working_state_before=before_ref,
                        returned_step=step,
                        planner_metadata=metadata,
                        execution_status="invalid_plan",
                        executed_calls=(),
                        deterministic_changes=(),
                        sanitized_failure={"category": "final_validation", "error": safe_error},
                        working_state_after=before_ref,
                        observation=None,
                        budget_counters=remaining(),
                    ))
                    return terminal("invalid_plan", safe_error, final_step=step)
                disposition_validation = validate_engineering_disposition(step.engineering_disposition, current_report,
                    engineering.summaries[engineering_report_hash(current_report)])
                if disposition_validation["status"] == "rejected":
                    safe_error = "Finish must account for current warning IDs using the current report's semantic design hash. Correct the structured disposition."
                    latest_observation = AgentStepObservation(
                        outcome="rejected", planned_calls=(), validation_result=disposition_validation,
                        working_design_ref=before_ref, working_design_hash=before_hash,
                        failure_category="engineering_disposition", sanitized_error=safe_error,
                        failed_action_fingerprint=_stable_hash({"state": before_hash, "step": step.to_dict()}),
                        remaining_budgets=remaining(),
                    )
                    record(AgentTrajectoryEntry(
                        iteration_index=decision_count, working_state_before=before_ref,
                        returned_step=step, planner_metadata=metadata, execution_status="finish_rejected",
                        executed_calls=(), deterministic_changes=(),
                        sanitized_failure={"category": "engineering_disposition", "error": safe_error},
                        working_state_after=before_ref, observation=latest_observation,
                        budget_counters=remaining(),
                    ))
                    # This decision used no tools or action batch. The normal
                    # decision budget bounds terminal correction; no hidden repair.
                    continue
                final_ref = _design_ref(final_design)
                record(AgentTrajectoryEntry(
                    iteration_index=decision_count,
                    working_state_before=before_ref,
                    returned_step=step,
                    planner_metadata=metadata,
                    execution_status="finished",
                    executed_calls=(),
                    deterministic_changes=(),
                    sanitized_failure=None,
                    working_state_after=final_ref,
                    observation=None,
                    budget_counters=remaining(),
                ))
                return terminal("finished", step.message, final_step=step, final_design=final_design)

            if step.status in {"clarify", "refuse"}:
                record(AgentTrajectoryEntry(
                    iteration_index=decision_count,
                    working_state_before=before_ref,
                    returned_step=step,
                    planner_metadata=metadata,
                    execution_status=step.status,
                    executed_calls=(),
                    deterministic_changes=(),
                    sanitized_failure=None,
                    working_state_after=_design_ref(baseline_design),
                    observation=None,
                    budget_counters=remaining(),
                ))
                return terminal(step.status, step.message, final_step=step)

            proposed_call_count += len(step.calls)
            action_fingerprint = rejected_action_fingerprint(before_design, step.calls)
            try:
                effects = {self.agent.planner_tool_effect(call.name) for call in step.calls}
            except CapabilityError:
                # Preserve existing unknown-tool handling in the deterministic executor.
                effects = {"design_action"}
            if len(effects) > 1:
                rejected_count += 1
                rejected_fingerprints.add(action_fingerprint)
                safe_error = "An execute batch may contain design actions or analysis calls, but not both."
                latest_observation = AgentStepObservation(
                    outcome="rejected", planned_calls=step.calls,
                    validation_result={"status": "rejected", "category": "mixed_tool_effects"},
                    working_design_ref=before_ref, working_design_hash=before_hash,
                    failure_category="mixed_tool_effects", sanitized_error=safe_error,
                    failed_action_fingerprint=action_fingerprint, remaining_budgets=remaining(),
                )
                record(AgentTrajectoryEntry(
                    iteration_index=decision_count, working_state_before=before_ref,
                    returned_step=step, planner_metadata=metadata, execution_status="rejected",
                    executed_calls=(), deterministic_changes=(),
                    sanitized_failure={"category": "mixed_tool_effects", "error": safe_error},
                    working_state_after=before_ref, observation=latest_observation,
                    budget_counters=remaining(),
                ))
                continue
            is_analysis = effects == {"analysis"}
            exhausted_reason: str | None = None
            if proposed_call_count > limits.total_proposed_tool_calls:
                exhausted_reason = "total proposed tool-call budget"
            elif is_analysis and analysis_batch_count >= limits.analysis_batches:
                exhausted_reason = "analysis-batch budget"
            elif not is_analysis and accepted_count >= limits.accepted_action_batches:
                exhausted_reason = "accepted action-batch budget"
            elif rejected_count >= limits.rejected_action_batches:
                exhausted_reason = "rejected action-batch budget"
            if exhausted_reason is not None:
                safe_error = f"The agent exhausted its {exhausted_reason}."
                latest_observation = AgentStepObservation(
                    outcome="rejected",
                    planned_calls=step.calls,
                    validation_result={"status": "not_executed", "category": "agent_limit"},
                    working_design_ref=before_ref,
                    working_design_hash=before_hash,
                    failure_category="agent_limit",
                    sanitized_error=safe_error,
                    failed_action_fingerprint=action_fingerprint,
                    remaining_budgets=remaining(),
                )
                record(AgentTrajectoryEntry(
                    iteration_index=decision_count,
                    working_state_before=before_ref,
                    returned_step=step,
                    planner_metadata=metadata,
                    execution_status="agent_limit",
                    executed_calls=(),
                    deterministic_changes=(),
                    sanitized_failure={"category": "agent_limit", "error": safe_error},
                    working_state_after=before_ref,
                    observation=latest_observation,
                    budget_counters=remaining(),
                ))
                return terminal("agent_limit", safe_error, final_step=step)

            if is_analysis:
                try:
                    results = []
                    pending_cache: dict[tuple[str, str, str, str], Any] = {}
                    cache_hits = 0
                    for call in step.calls:
                        definition = self.agent.registry.tool(call.name)
                        cache_key = (
                            call.name,
                            definition.tool_version,
                            _stable_hash(normalize_analysis_arguments(call.arguments)),
                            before_hash,
                        )
                        if cache_key in analysis_cache:
                            result = analysis_cache[cache_key]
                            cache_hits += 1
                        elif cache_key in pending_cache:
                            result = pending_cache[cache_key]
                            cache_hits += 1
                        else:
                            result = self.agent.execute_analysis_batch(before_design, (call,))[0]
                            pending_cache[cache_key] = result
                        if result.semantic_design_hash != before_hash:
                            raise CapabilityError("Analysis returned a stale working-design hash.")
                        results.append(result)
                    analysis_cache.update(pending_cache)
                except (CapabilityError, DesignValidationError, AgentInstructionError) as exc:
                    rejected_count += 1
                    rejected_fingerprints.add(action_fingerprint)
                    category = _failure_category(exc)
                    safe_error = _sanitize_text(exc)
                    latest_observation = AgentStepObservation(
                        outcome="rejected", planned_calls=step.calls,
                        validation_result={"status": "rejected", "category": category},
                        working_design_ref=before_ref, working_design_hash=before_hash,
                        failure_category=category, sanitized_error=safe_error,
                        failed_action_fingerprint=action_fingerprint, remaining_budgets=remaining(),
                    )
                    record(AgentTrajectoryEntry(
                        iteration_index=decision_count, working_state_before=before_ref,
                        returned_step=step, planner_metadata=metadata, execution_status="rejected",
                        executed_calls=(), deterministic_changes=(),
                        sanitized_failure={"category": category, "error": safe_error},
                        working_state_after=before_ref, observation=latest_observation,
                        budget_counters=remaining(),
                    ))
                    continue
                analysis_batch_count += 1
                actual_calls = tuple(step.calls)
                latest_observation = AgentStepObservation(
                    outcome="accepted_analysis", planned_calls=step.calls,
                    executed_calls=actual_calls, deterministic_changes=(),
                    validation_result={"status": "read_only", "design_unchanged": True},
                    working_design_ref=before_ref, working_design_hash=before_hash,
                    analysis_results=tuple(results), analysis_cache_hits=cache_hits,
                    remaining_budgets=remaining(),
                )
                record(AgentTrajectoryEntry(
                    iteration_index=decision_count, working_state_before=before_ref,
                    returned_step=step, planner_metadata=metadata,
                    execution_status="accepted_analysis", executed_calls=actual_calls,
                    deterministic_changes=(), sanitized_failure=None,
                    working_state_after=before_ref, observation=latest_observation,
                    budget_counters=remaining(), analysis_results=tuple(results),
                    analysis_cache_hits=cache_hits,
                ))
                continue

            if action_fingerprint in rejected_fingerprints:
                # The proposal consumed one decision and its proposed-call allowance. It does
                # not consume another rejected-execution batch because the executor is skipped.
                safe_error = "The identical action batch was already rejected against this working state."
                latest_observation = AgentStepObservation(
                    outcome="rejected",
                    planned_calls=step.calls,
                    validation_result={"status": "not_executed", "category": "duplicate_rejection"},
                    working_design_ref=before_ref,
                    working_design_hash=before_hash,
                    failure_category="duplicate_rejection",
                    sanitized_error=safe_error,
                    failed_action_fingerprint=action_fingerprint,
                    remaining_budgets=remaining(),
                )
                record(AgentTrajectoryEntry(
                    iteration_index=decision_count,
                    working_state_before=before_ref,
                    returned_step=step,
                    planner_metadata=metadata,
                    execution_status="skipped_duplicate",
                    executed_calls=(),
                    deterministic_changes=(),
                    sanitized_failure={"category": "duplicate_rejection", "error": safe_error},
                    working_state_after=before_ref,
                    observation=latest_observation,
                    budget_counters=remaining(),
                ))
                return terminal("duplicate_rejection", safe_error, final_step=step)

            try:
                result = self.agent.execute_llm_plan(
                    before_design,
                    LLMToolPlan("execute", step.message, step.calls),
                )
            except (CapabilityError, DesignValidationError, AgentInstructionError) as exc:
                rejected_count += 1
                rejected_fingerprints.add(action_fingerprint)
                category = _failure_category(exc)
                safe_error = _sanitize_text(exc)
                latest_observation = AgentStepObservation(
                    outcome="rejected",
                    planned_calls=step.calls,
                    validation_result={"status": "rejected", "category": category},
                    working_design_ref=before_ref,
                    working_design_hash=before_hash,
                    failure_category=category,
                    sanitized_error=safe_error,
                    failed_action_fingerprint=action_fingerprint,
                    remaining_budgets=remaining(),
                )
                record(AgentTrajectoryEntry(
                    iteration_index=decision_count,
                    working_state_before=before_ref,
                    returned_step=step,
                    planner_metadata=metadata,
                    execution_status="rejected",
                    executed_calls=(),
                    deterministic_changes=(),
                    sanitized_failure={"category": category, "error": safe_error},
                    working_state_after=before_ref,
                    observation=latest_observation,
                    budget_counters=remaining(),
                ))
                continue

            candidate = result.design
            candidate_hash = semantic_design_hash(candidate)
            actual_calls = _actual_calls(result.executed_tools)
            if candidate_hash == before_hash:
                rejected_count += 1
                rejected_fingerprints.add(action_fingerprint)
                safe_error = "The executed action batch produced no semantic design change."
                latest_observation = AgentStepObservation(
                    outcome="rejected",
                    planned_calls=step.calls,
                    validation_result={"status": "rejected", "category": "no_progress"},
                    working_design_ref=before_ref,
                    working_design_hash=before_hash,
                    failure_category="no_progress",
                    sanitized_error=safe_error,
                    failed_action_fingerprint=action_fingerprint,
                    remaining_budgets=remaining(),
                )
                record(AgentTrajectoryEntry(
                    iteration_index=decision_count,
                    working_state_before=before_ref,
                    returned_step=step,
                    planner_metadata=metadata,
                    execution_status="no_progress",
                    executed_calls=actual_calls,
                    deterministic_changes=(),
                    sanitized_failure={"category": "no_progress", "error": safe_error},
                    working_state_after=before_ref,
                    observation=latest_observation,
                    budget_counters=remaining(),
                ))
                continue

            accepted_count += 1
            changes = deterministic_change_summary(before_design, candidate, result.changes)
            aggregate_calls.extend(step.calls)
            aggregate_changes.extend(changes)
            working_design = candidate
            current_report = engineering.current(candidate)
            latest_observation = AgentStepObservation(
                outcome="accepted",
                planned_calls=step.calls,
                executed_calls=actual_calls,
                deterministic_changes=changes,
                validation_result=_validation_result(candidate),
                working_design_ref=_design_ref(candidate),
                working_design_hash=candidate_hash,
                remaining_budgets=remaining(),
            )
            execution_status = "accepted"
            is_cycle = candidate_hash in accepted_state_hashes
            if is_cycle:
                execution_status = "cycle_detected"
            record(AgentTrajectoryEntry(
                iteration_index=decision_count,
                working_state_before=before_ref,
                returned_step=step,
                planner_metadata=metadata,
                execution_status=execution_status,
                executed_calls=actual_calls,
                deterministic_changes=changes,
                sanitized_failure=(
                    {"category": "cycle_detected", "error": "The action returned to an earlier semantic design state."}
                    if is_cycle else None
                ),
                working_state_after=_design_ref(candidate),
                observation=latest_observation,
                budget_counters=remaining(),
            ))
            if is_cycle:
                return terminal(
                    "cycle_detected",
                    "The agent returned to an earlier semantic design state.",
                    final_step=step,
                )
            accepted_state_hashes.add(candidate_hash)

        return terminal(
            "agent_limit",
            "The agent exhausted its decision-iteration budget before finishing.",
        )


def run_antenna_agent_loop(
    *,
    agent: AntennaDesignAgent,
    baseline_design: AntennaDesign | None,
    instruction: str,
    project_memory: Mapping[str, Any] | Any | None,
    planner: AgentLoopPlanner,
    budgets: AgentLoopBudgets | None = None,
) -> AgentTerminalResult:
    """Functional entry point for the unpublished Stage-2 runner."""

    return AntennaAgentRunner(agent).run(
        baseline_design=baseline_design,
        instruction=instruction,
        project_memory=project_memory,
        planner=planner,
        budgets=budgets,
    )
