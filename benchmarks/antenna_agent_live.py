"""Live Stage-5B adapter for the frozen antenna-agent benchmark.

The adapter invokes the production builder and planner interfaces without adding
planner context or changing agent behavior. Importing this module performs no
network activity. Provider calls occur only through ``run_live_provider``.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping

from benchmarks.antenna_agent_benchmark import (
    DEFAULT_DEFINITION_PATH,
    BenchmarkRunner,
    ReplayExecutor,
    evaluate_case,
    load_benchmark,
    markdown_summary,
)
from benchmarks.contract_provenance import observation_contract_provenance
from studio.antenna_agent import (
    PlannerClarificationRequired,
    PlannerRefusal,
    create_default_agent,
)
from studio.antenna_analysis import semantic_design_hash, stable_hash
from studio.antenna_builder import (
    BuilderProjectSession,
    CanonicalDesignRef,
    ProjectMemory,
    execute_builder_turn,
    save_builder_session,
)
from studio.antenna_design import AntennaDesign, evaluate_scalar, resolved_dimensions
from studio.antenna_engineering_checks import run_engineering_checks
from studio.antenna_engineering_summary import engineering_report_hash
from studio.antenna_llm_planner import (
    AgentLoopBudgets,
    GeminiSchemaConstrainedPlanner,
    LLMToolPlan,
    OpenRouterNemotronPlanner,
    PlannedToolCall,
)
from studio.antenna_tools import CapabilityError


EXPECTED_BENCHMARK_SHA256 = "281efa92743b03c69f2a16ad3ee8d6d01dafd339541e43cbcd0ac675ee86b868"
LIVE_BUDGETS = AgentLoopBudgets()
RETRY_POLICY = {
    "maximum_attempts": 2,
    "delay_seconds": 5,
    "eligible_failure": "provider_error only",
    "context_policy": "retry from the identical pre-turn persisted session and unchanged prompt",
}
PROVIDERS = {
    "gemini": "gemini-3.8-flash",
    "openrouter": "nvidia/nemotron-3-ultra-550b-a55b:free",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_frozen_benchmark(path: str | Path = DEFAULT_DEFINITION_PATH) -> str:
    digest = __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest()
    if digest != EXPECTED_BENCHMARK_SHA256:
        raise RuntimeError(
            f"Frozen benchmark SHA mismatch: expected {EXPECTED_BENCHMARK_SHA256}, got {digest}."
        )
    load_benchmark(path)
    return digest


def _plan(*calls: tuple[str, dict[str, Any]]) -> LLMToolPlan:
    return LLMToolPlan(
        "execute",
        "deterministic benchmark fixture construction",
        tuple(PlannedToolCall(name, arguments) for name, arguments in calls),
    )


def _circle_feature(
    design: AntennaDesign,
    *,
    prefix: str,
    radius: float,
    x: float = 0.0,
    y: float = 0.0,
    operation: str = "subtract",
) -> AntennaDesign:
    agent = create_default_agent()
    return agent.execute_llm_plan(
        design,
        _plan(
            ("parameter.create", {
                "key": f"{prefix}_radius_mm", "label": f"{prefix} radius", "value": radius,
                "unit": "mm", "sweepable": True,
            }),
            ("geometry.cylinder", {
                "object_id": f"{prefix}_tool", "material_id": "copper", "axis": "z",
                "tags": ["planner_created", "boolean_tool", "slot"],
                "dimensions": {
                    "center_1": x, "center_2": y, "radius": f"{prefix}_radius_mm",
                    "start": "substrate_thickness_mm",
                    "end": "substrate_thickness_mm+copper_thickness_mm",
                },
            }),
            (f"boolean.{operation}", {
                "operation_id": f"{prefix}_{operation}",
                "target_id": "element_1_1_patch",
                "tool_ids": [f"{prefix}_tool"],
            }),
        ),
    ).design


def _rectangle_feature(design: AntennaDesign, *, prefix: str, width: float, height: float) -> AntennaDesign:
    agent = create_default_agent()
    return agent.execute_llm_plan(
        design,
        _plan(
            ("parameter.create", {
                "key": f"{prefix}_width_mm", "label": f"{prefix} width", "value": width,
                "unit": "mm", "sweepable": True,
            }),
            ("parameter.create", {
                "key": f"{prefix}_height_mm", "label": f"{prefix} height", "value": height,
                "unit": "mm", "sweepable": True,
            }),
            ("geometry.rectangle_sheet", {
                "object_id": f"{prefix}_tool", "material_id": "copper",
                "tags": ["planner_created", "boolean_tool", "slot"],
                "dimensions": {
                    "x_min": f"-{prefix}_width_mm/2", "x_max": f"{prefix}_width_mm/2",
                    "y_min": f"-{prefix}_height_mm/2", "y_max": f"{prefix}_height_mm/2",
                    "z_min": "substrate_thickness_mm",
                    "z_max": "substrate_thickness_mm+copper_thickness_mm",
                },
            }),
            ("boolean.subtract", {
                "operation_id": f"{prefix}_subtract", "target_id": "element_1_1_patch",
                "tool_ids": [f"{prefix}_tool"],
            }),
        ),
    ).design


def build_fixture(fixture_id: str) -> BuilderProjectSession:
    """Create one isolated, deterministic benchmark baseline without using an LLM."""

    if fixture_id == "empty_project":
        return BuilderProjectSession(None, [], ProjectMemory.empty())
    agent = create_default_agent()
    recipe = "circular_patch" if fixture_id == "circular_patch_clean" else (
        "dipole" if fixture_id == "dipole_clean" else "inset_patch"
    )
    design = agent.create_design(recipe, design_id=f"benchmark_{fixture_id}", revision=0)
    if fixture_id == "inset_patch_center_circle":
        design = _circle_feature(design, prefix="center_slot", radius=3.0)
    elif fixture_id == "inset_patch_two_features":
        design = _circle_feature(design, prefix="left_slot", radius=2.0, x=-5.0)
        design = _rectangle_feature(design, prefix="right_slot", width=4.0, height=1.5)
    elif fixture_id == "inset_patch_parameter_rectangle":
        design = _rectangle_feature(design, prefix="rect_slot", width=6.0, height=1.0)
    elif fixture_id == "inset_patch_2x3_decorated":
        design = _circle_feature(design, prefix="center_slot", radius=3.0)
        design = agent.update_parameters(
            design, {"array_rows": 2, "array_columns": 3}, intent="fixture array"
        ).design
        group_id = design.composed_operations[0].group_id
        design = agent.execute_llm_plan(design, _plan((
            "composition.set_scope", {"group_id": group_id, "scope": "all", "elements": []},
        ))).design
        design = agent.execute_llm_plan(design, _plan((
            "excitation.set_strategy", {"strategy": "independent_ports"},
        ))).design
    elif fixture_id == "independent_2x2":
        design = agent.update_parameters(
            design, {"array_rows": 2, "array_columns": 2}, intent="fixture array"
        ).design
        design = agent.execute_llm_plan(design, _plan((
            "excitation.set_strategy", {"strategy": "independent_ports"},
        ))).design
    elif fixture_id == "warning_patch_overlap":
        design = _circle_feature(
            design, prefix="overhang_union", radius=18.0, x=20.0, operation="union"
        )
    elif fixture_id == "warning_patch_heavy":
        design = _circle_feature(
            design, prefix="overhang_union", radius=18.0, x=20.0, operation="union"
        )
        design = agent.update_parameters(
            design, {"array_rows": 2, "array_columns": 3}, intent="fixture array"
        ).design
        group_id = design.composed_operations[0].group_id
        design = agent.execute_llm_plan(design, _plan((
            "composition.set_scope", {"group_id": group_id, "scope": "all", "elements": []},
        ))).design
        design = agent.execute_llm_plan(design, _plan((
            "excitation.set_strategy", {"strategy": "independent_ports"},
        ))).design
    elif fixture_id not in {"inset_patch_clean", "circular_patch_clean", "dipole_clean"}:
        raise ValueError(f"Unknown benchmark fixture: {fixture_id}")
    memory = ProjectMemory.empty(CanonicalDesignRef(design.design_id, design.revision))
    return BuilderProjectSession(design, [], memory)


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _expression_value(design: AntennaDesign, value: Any) -> float | None:
    try:
        return float(evaluate_scalar(value, design))
    except Exception:
        return None


def canonical_facts(design: AntennaDesign | None) -> dict[str, Any]:
    if design is None:
        return {"design_exists": False}
    groups = design.composed_operations
    calls = [call for group in groups for call in group.calls]
    circle_calls = [call for call in calls if call.name in {"geometry.cylinder", "geometry.circle_sheet"}]
    rectangle_calls = [call for call in calls if call.name == "geometry.rectangle_sheet"]
    subtract_calls = [call for call in calls if call.name == "boolean.subtract"]
    union_calls = [call for call in calls if call.name == "boolean.union"]
    circle_subtractions = 0
    rectangle_subtractions = 0
    for group in groups:
        names = {call.name for call in group.calls}
        multiplier = design.array.element_count if group.target_selector and group.target_selector.scope == "all" else 1
        if "boolean.subtract" in names and names & {"geometry.cylinder", "geometry.circle_sheet"}:
            circle_subtractions += multiplier
        if "boolean.subtract" in names and "geometry.rectangle_sheet" in names:
            rectangle_subtractions += multiplier
    circle_union_count = sum(
        1 for group in groups
        if any(call.name == "boolean.union" for call in group.calls)
        and any(call.name in {"geometry.cylinder", "geometry.circle_sheet"} for call in group.calls)
    )
    centers = []
    radii = []
    for call in circle_calls:
        dimensions = call.arguments().get("dimensions", {})
        centers.append((
            _expression_value(design, dimensions.get("center_1")),
            _expression_value(design, dimensions.get("center_2")),
        ))
        radii.append(_expression_value(design, dimensions.get("radius")))
    rectangle_widths = []
    for call in rectangle_calls:
        dimensions = call.arguments().get("dimensions", {})
        low = _expression_value(design, dimensions.get("x_min"))
        high = _expression_value(design, dimensions.get("x_max"))
        if low is not None and high is not None:
            rectangle_widths.append(high - low)
    scope = groups[0].target_selector.scope if len(groups) == 1 and groups[0].target_selector else None
    target_role = groups[0].target_selector.role if len(groups) == 1 and groups[0].target_selector else None
    probe_inside = True
    if design.family == "circular_patch":
        radiating = [item for item in design.geometry if "radiating_patch" in item.tags]
        if radiating and design.ports:
            dimensions = resolved_dimensions(radiating[0], design)
            radius = float(dimensions.get("radius", 0))
            port = design.ports[0]
            probe_inside = (port.positive[0] ** 2 + port.positive[1] ** 2) ** 0.5 <= radius
    return {
        "design_exists": True,
        "family": design.family,
        "frequency_ghz": design.frequency_ghz,
        "material": design.material,
        "array_rows": design.array.rows,
        "array_columns": design.array.columns,
        "element_spacing_lambda": design.element_spacing_lambda,
        "excitation_strategy": design.excitation.strategy,
        "physical_port_count": len(design.ports),
        "composed_feature_count": len(groups),
        "logical_composed_feature_count": len(groups),
        "circle_subtraction_count": circle_subtractions,
        "physical_circle_subtraction_count": circle_subtractions,
        "rectangular_subtraction_count": rectangle_subtractions,
        "circle_union_count": circle_union_count,
        "feature_target_role": target_role,
        "feature_scope": scope,
        "distinct_feature_offsets": len(set(centers)) == len(centers) and len(centers) > 1,
        "circle_radius_mm": radii[0] if len(radii) == 1 else None,
        "rectangle_width_mm": rectangle_widths[0] if len(rectangle_widths) == 1 else None,
        "feature_local_offset_x_mm": centers[0][0] if len(centers) == 1 else None,
        "boolean_subtraction_count": len(subtract_calls),
        "boolean_union_count": len(union_calls),
        "probe_inside_patch": probe_inside,
    }


def _memory_items(session: BuilderProjectSession) -> list[dict[str, Any]]:
    payload = session.memory.to_dict()
    return [
        item
        for name in (
            "requirements", "decisions", "assumptions", "limitations", "open_questions",
            "important_changes", "recent_context",
        )
        for item in payload.get(name, [])
    ]


def _planner_factory(provider: str, model: str, env_file: Path):
    if provider == "gemini":
        return GeminiSchemaConstrainedPlanner(model=model, env_file=env_file)
    if provider == "openrouter":
        return OpenRouterNemotronPlanner(model=model, env_file=env_file)
    raise ValueError(f"Unsupported live benchmark provider: {provider}")


def _terminal_from_audit(record: Mapping[str, Any] | None) -> tuple[str, str]:
    if not record:
        return "provider_error", "Planner initialization or execution failed before audit capture."
    return str(record.get("terminal_status", "provider_error")), str(record.get("terminal_message", ""))


def _provider_or_schema_failure(record: Mapping[str, Any] | None) -> str:
    """Separate transport/provider absence from exhausted structured-output parsing."""

    if not record:
        return "provider"
    trajectories = record.get("trajectory", [])
    if not isinstance(trajectories, list) or not trajectories:
        return "provider"
    metadata = trajectories[-1].get("planner_metadata", {})
    if not isinstance(metadata, Mapping):
        return "provider"
    attempts = metadata.get("returned_plan_attempts", [])
    repairs = metadata.get("schema_repairs", 0)
    return "schema" if attempts or repairs else "provider"


def _collect_hash_values(value: Any, *, key_hint: str = "") -> set[str]:
    hashes: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            hashes.update(_collect_hash_values(item, key_hint=str(key)))
    elif isinstance(value, list):
        for item in value:
            hashes.update(_collect_hash_values(item, key_hint=key_hint))
    elif (
        "hash" in key_hint.casefold()
        and isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        hashes.add(value)
    return hashes


def _hallucinated_tools(trajectories: list[Mapping[str, Any]]) -> list[str]:
    installed = set(create_default_agent().available_capabilities()["planner_tools"])
    proposed: set[str] = set()
    for entry in trajectories:
        metadata = entry.get("planner_metadata", {})
        for attempt in metadata.get("returned_plan_attempts", []) if isinstance(metadata, Mapping) else []:
            if not isinstance(attempt, Mapping):
                continue
            calls = attempt.get("calls", [])
            if not isinstance(calls, list):
                continue
            for call in calls:
                if isinstance(call, Mapping) and isinstance(call.get("name"), str):
                    proposed.add(call["name"])
    return sorted(proposed - installed)


def _warning_mentions(report, response_text: str) -> list[str]:
    """Record conservative presentation evidence independently from disposition."""

    normalized = response_text.casefold().replace("_", " ").replace("-", " ")
    mentioned = []
    for finding in report.findings if report is not None else ():
        if finding.severity != "warning":
            continue
        category_tokens = [
            token for token in finding.category.casefold().replace("_", " ").split()
            if len(token) >= 5
        ]
        if finding.observation_id.casefold() in normalized or any(
            token in normalized for token in category_tokens
        ):
            mentioned.append(finding.observation_id)
    return mentioned


def _unsupported_claims(case: Mapping[str, Any], response_text: str) -> list[str]:
    """Flag narrow benchmark-defined RF overclaims; this never affects production prose."""

    acceptance = case.get("acceptance", {})
    analysis = acceptance.get("analysis", {}) if isinstance(acceptance, Mapping) else {}
    normalized = " ".join(response_text.casefold().split())
    checks: list[tuple[bool, tuple[str, ...], str]] = [
        (
            bool(analysis.get("forbid_proven_resonance_claim")),
            ("proves resonance", "proven resonance", "will resonate", "is resonant"),
            "Dipole analytical baseline was presented as proven resonance.",
        ),
        (
            bool(analysis.get("forbid_simulated_resonance_claim")),
            ("simulated resonance", "full-wave resonance", "confirmed resonance"),
            "Patch analytical baseline was presented as simulated or confirmed resonance.",
        ),
        (
            bool(analysis.get("forbid_unconditional_grating_lobe_claim")),
            ("no grating lobes", "grating lobes are impossible", "guaranteed grating-lobe free"),
            "Spacing analysis was presented as an unconditional grating-lobe conclusion.",
        ),
    ]
    return [message for enabled, phrases, message in checks if enabled and any(item in normalized for item in phrases)]


@dataclass(slots=True)
class LiveBenchmarkExecutor:
    run_root: Path
    env_file: Path
    retry_delay: Callable[[float], None] = time.sleep

    def execute(self, case: Mapping[str, Any], *, provider: str, model: str) -> Mapping[str, Any]:
        case_root = self.run_root / provider / case["case_id"]
        case_root.mkdir(parents=True, exist_ok=True)
        audit_path = case_root / "trajectory.jsonl"
        session = build_fixture(case["initial_fixture"])
        save_builder_session(case_root, session)
        initial_design_hash = semantic_design_hash(session.design)
        initial_memory_hash = stable_hash(session.memory.to_dict())
        attempt_records: list[dict[str, Any]] = []
        successful_turns = 0
        dependent_turns_skipped = 0
        final_terminal = "provider_error"
        final_message = ""
        provider_failed = False

        for turn_index, instruction in enumerate(case["turns"], start=1):
            turn_completed = False
            for attempt_index in range(1, RETRY_POLICY["maximum_attempts"] + 1):
                attempt_session = copy.deepcopy(session)
                before_count = len(_read_json_lines(audit_path))
                with TemporaryDirectory(prefix=f"{case['case_id']}-", dir=case_root) as temp:
                    attempt_project = Path(temp)
                    save_builder_session(attempt_project, attempt_session)
                    error_text = None
                    try:
                        planner = _planner_factory(provider, model, self.env_file)
                        result = execute_builder_turn(
                            attempt_project,
                            attempt_session,
                            instruction,
                            planner=planner,
                            audit_log_path=audit_path,
                            turn_id=f"{case['case_id']}-turn-{turn_index}-attempt-{attempt_index}",
                            budgets=LIVE_BUDGETS,
                        )
                        attempt_session = result.session
                    except (PlannerClarificationRequired, PlannerRefusal, CapabilityError) as exc:
                        error_text = str(exc)
                    except Exception as exc:
                        error_text = f"{type(exc).__name__}: {exc}"
                    new_records = _read_json_lines(audit_path)[before_count:]
                    audit = new_records[-1] if new_records else None
                    terminal, message = _terminal_from_audit(audit)
                    if not message and error_text:
                        message = error_text[:1200]
                    attempt_records.append({
                        "turn_index": turn_index,
                        "attempt_index": attempt_index,
                        "instruction": instruction,
                        "terminal_outcome": terminal,
                        "message": message,
                        "audit_record": audit,
                    })

                failure_kind = _provider_or_schema_failure(audit) if terminal == "provider_error" else None
                if terminal == "provider_error" and failure_kind == "provider":
                    if attempt_index < RETRY_POLICY["maximum_attempts"]:
                        self.retry_delay(float(RETRY_POLICY["delay_seconds"]))
                        continue
                    provider_failed = True
                    final_terminal = terminal
                    final_message = message
                    break

                if terminal == "provider_error" and failure_kind == "schema":
                    final_terminal = terminal
                    final_message = message
                    break

                session = attempt_session
                save_builder_session(case_root, session)
                final_terminal = terminal
                final_message = message
                turn_completed = terminal == "finished"
                if turn_completed:
                    successful_turns += 1
                break

            if provider_failed or final_terminal == "provider_error":
                dependent_turns_skipped = len(case["turns"]) - turn_index
                break
            if not turn_completed and turn_index < len(case["turns"]):
                dependent_turns_skipped = len(case["turns"]) - turn_index
                break

        report = run_engineering_checks(session.design) if session.design is not None else None
        report_hashes = {engineering_report_hash(report)} if report is not None else set()
        all_audits = [
            item["audit_record"] for item in attempt_records if item.get("audit_record")
        ]
        trajectories = [
            entry
            for audit in all_audits
            for entry in audit.get("trajectory", [])
        ]
        report_hashes.update(_collect_hash_values(trajectories))
        calls = [
            call
            for audit in all_audits
            for call in audit.get("aggregate_executed_tools", [])
        ]
        rejected = [
            entry for entry in trajectories
            if entry.get("execution_status") in {
                "rejected", "finish_rejected", "invalid_plan", "no_progress", "skipped_duplicate",
            }
        ]
        analysis_results = [
            result
            for entry in trajectories
            for result in entry.get("analysis_results", [])
        ]
        analysis_tools = sorted({str(item.get("tool_name")) for item in analysis_results if item.get("tool_name")})
        schema_failures = sum(
            int(entry.get("planner_metadata", {}).get("schema_repairs", 0) or 0)
            for entry in trajectories
        )
        warning_ids = (
            [item.observation_id for item in report.findings if item.severity == "warning"]
            if report is not None else []
        )
        final_step = None
        for entry in reversed(trajectories):
            if entry.get("returned_step"):
                final_step = entry["returned_step"]
                break
        disposition = (final_step or {}).get("engineering_disposition") or {}
        acknowledged = list(disposition.get("acknowledged_observation_ids", []))
        deferred = [item.get("observation_id") for item in disposition.get("deferred", []) if item.get("observation_id")]
        if provider_failed or final_terminal == "provider_error":
            completed = "partial" if successful_turns else "no"
        elif successful_turns == len(case["turns"]):
            completed = "yes"
        elif successful_turns:
            completed = "partial"
        else:
            completed = "no"
        failure_signals = []
        if provider_failed:
            failure_signals.append("provider_api_failure")
        if schema_failures:
            failure_signals.append("schema_structured_output_failure")
        if final_terminal in {"invalid_plan", "duplicate_rejection", "cycle_detected", "agent_limit"}:
            failure_signals.append("planner_reasoning_failure")
        if rejected and "planner_reasoning_failure" not in failure_signals:
            failure_signals.append("planner_reasoning_failure")

        response_text = "\n".join(
            str(item.get("message", "")) for item in attempt_records if item.get("message")
        )
        observation = {
            "case_id": case["case_id"],
            "provider": provider,
            "model": model,
            "initial_semantic_hash": initial_design_hash,
            "final_semantic_hash": semantic_design_hash(session.design),
            "initial_memory_hash": initial_memory_hash,
            "final_memory_hash": stable_hash(session.memory.to_dict()),
            "engineering_report_hashes": sorted(report_hashes),
            "trajectory_reference": str(audit_path.relative_to(self.run_root)),
            "terminal_result": {
                "outcome": final_terminal,
                "message": final_message,
                "dependent_turns_skipped": dependent_turns_skipped,
            },
            "terminal_outcome": final_terminal,
            "task_completed": completed,
            "canonical_facts": canonical_facts(session.design),
            "selected_capabilities": sorted({str(call.get("name")) for call in calls if call.get("name")}),
            "hallucinated_capabilities": _hallucinated_tools(trajectories),
            "engineering_warning_ids": warning_ids,
            "acknowledged_warning_ids": acknowledged,
            "deferred_warning_ids": deferred,
            "warnings_mentioned_in_prose": _warning_mentions(report, response_text),
            "engineering_disposition": disposition,
            "analysis_tools": analysis_tools,
            "analysis_results": analysis_results,
            "unsupported_model_claims": _unsupported_claims(case, response_text),
            "memory_items": _memory_items(session),
            "schema_agent_step_failures": schema_failures,
            "provider_api_failure": provider_failed,
            "execution_rejections": len(rejected),
            "decision_iterations": len(trajectories),
            "design_action_batches": sum(entry.get("execution_status") == "accepted" for entry in trajectories),
            "analysis_batches": sum(entry.get("execution_status") == "accepted_analysis" for entry in trajectories),
            "failure_signals": failure_signals,
            "attempts": attempt_records,
            "trajectory": trajectories,
            "tool_calls": calls,
            "rejected_calls": rejected,
            "final_user_facing_response": final_message,
            "presentation_review_method": "warning ID/category-token evidence in captured user-facing responses",
            "rf_claim_review_method": "frozen case-specific forbidden-claim fragments",
            "contract_provenance": observation_contract_provenance(
                EXPECTED_BENCHMARK_SHA256
            ),
        }
        (case_root / "observation.json").write_text(
            json.dumps(observation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return observation


def provider_summary(run: Mapping[str, Any]) -> str:
    results = run["results"]
    evaluable = [item for item in results if not item["provider_api_failure"]]
    failed = [item for item in evaluable if item["task_completed"] == "no"]
    lines = [
        markdown_summary(run).rstrip(),
        "",
        "## Live evaluation totals",
        "",
        f"- completed: {sum(item['task_completed'] == 'yes' for item in evaluable)}",
        f"- partial: {sum(item['task_completed'] == 'partial' for item in evaluable)}",
        f"- failed: {len(failed)}",
        f"- unevaluable due to provider/API: {len(results) - len(evaluable)}",
        f"- canonical-state correct: {sum(item['canonical_state_correct'] for item in evaluable)} / {len(evaluable)}",
        f"- semantic-memory correct: {sum(item['semantic_memory_correct'] for item in evaluable)} / {len(evaluable)}",
        f"- unsupported capability hallucinations: {sum(item['unsupported_capability_hallucinated'] for item in evaluable)}",
        f"- missed required clarifications: {sum(item['required_clarification_missed'] for item in evaluable)}",
        f"- unnecessary clarifications: {sum(item['unnecessary_clarification'] for item in evaluable)}",
        f"- engineering-disposition failures: {sum(not item['engineering_findings_accounted_for'] for item in evaluable)}",
        f"- presentation omissions: {sum(len(item['presentation_warning_omissions']) for item in evaluable)}",
        f"- unsupported RF claims: {sum(len(item['unsupported_model_claims']) for item in evaluable)}",
        f"- schema/structured-output failures: {sum(item['schema_agent_step_failures'] for item in results)}",
        f"- execution rejections: {sum(item['execution_rejections'] for item in results)}",
        f"- rollback failures: {sum(not item['transaction_rollback_correct'] for item in evaluable)}",
        "",
        "## Failure taxonomy",
        "",
    ]
    counts: dict[str, int] = {}
    for item in results:
        for category in item["failure_classifications"]:
            counts[category] = counts.get(category, 0) + 1
    lines.extend(f"- {category}: {count}" for category, count in sorted(counts.items()))
    lines.append("")
    return "\n".join(lines)


def _result_index(run: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {item["case_id"]: item for item in run["results"]}


def cross_provider_markdown(gemini: Mapping[str, Any], nemotron: Mapping[str, Any]) -> str:
    left = _result_index(gemini)
    right = _result_index(nemotron)
    dimensions = (
        "terminal_outcome", "task_completed", "canonical_state_correct", "semantic_memory_correct",
        "engineering_findings_accounted_for", "analytical_result_grounded_correctly",
        "transaction_rollback_correct",
    )
    lines = [
        "# Cross-provider comparison",
        "",
        "Only cases with evaluable results from both providers are compared.",
        "",
        "| Case | Gemini evidence | Nemotron evidence | Differing dimensions |",
        "|---|---|---|---|",
    ]
    compared = 0
    disagreements = 0
    for case_id in sorted(set(left) & set(right)):
        a, b = left[case_id], right[case_id]
        if a["provider_api_failure"] or b["provider_api_failure"]:
            continue
        compared += 1
        differing = [name for name in dimensions if a[name] != b[name]]
        if differing:
            disagreements += 1
            lines.append(
                f"| {case_id} | outcome={a['terminal_outcome']}, completed={a['task_completed']}, canonical={a['canonical_state_correct']} | "
                f"outcome={b['terminal_outcome']}, completed={b['task_completed']}, canonical={b['canonical_state_correct']} | "
                f"{', '.join(differing)} |"
            )
    if not disagreements:
        lines.append("| — | — | — | No observable disagreements among evaluable paired cases |")
    lines.extend(["", f"Paired evaluable cases: {compared}", f"Cases with observable disagreement: {disagreements}", ""])
    return "\n".join(lines)


def failure_inventory(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    entries = []
    taxonomy_totals: dict[str, int] = {}
    for provider, run in runs.items():
        for result in run["results"]:
            symptoms = []
            if result["provider_api_failure"]:
                symptoms.append("provider/API failure")
            if not result["canonical_state_correct"]:
                symptoms.append("final canonical facts did not meet frozen criteria")
            if not result["semantic_memory_correct"]:
                symptoms.append("semantic memory did not meet frozen criteria")
            if result["unsupported_capability_hallucinated"]:
                symptoms.append("unsupported capability was selected or hallucinated")
            if result["unsupported_model_claims"]:
                symptoms.append("final prose exceeded structured RF evidence")
            for category in result["failure_classifications"]:
                taxonomy_totals[category] = taxonomy_totals.get(category, 0) + 1
            if symptoms or result["failure_classifications"]:
                entries.append({
                    "case": result["case_id"],
                    "provider": provider,
                    "symptom": symptoms,
                    "failure_taxonomy": result["failure_classifications"],
                    "reproducible": None,
                    "likely_architecture_issue": None,
                    "provider_model_specific": None,
                    "possible_future_investigation": "Review preserved trajectory and observable state after Stage 5B; no fix applied.",
                })
    return {"taxonomy_totals": taxonomy_totals, "entries": entries}


def _augment_failure_taxonomy(run: dict[str, Any]) -> None:
    installed = set(create_default_agent().available_capabilities()["planner_tools"])
    for result in run["results"]:
        categories = list(result["failure_classifications"])

        def add(category: str) -> None:
            if category not in categories:
                categories.append(category)

        if result["provider_api_failure"]:
            add("provider_api_failure")
        else:
            unavailable_requirements = set(result["missing_required_capabilities"]) - installed
            if unavailable_requirements:
                add("bad_tool_abstraction")
            elif result["missing_required_capabilities"]:
                add("planner_reasoning_failure")
            if (
                not result["canonical_state_correct"]
                or not result["semantic_memory_correct"]
                or result["unsupported_capability_hallucinated"]
                or result["required_clarification_missed"]
                or result["unnecessary_clarification"]
                or not result["engineering_findings_accounted_for"]
                or not result["analytical_result_grounded_correctly"]
            ):
                add("planner_reasoning_failure")
            if not result["transaction_rollback_correct"]:
                add("transaction_runtime_failure")
        result["failure_classifications"] = categories


def run_live_provider(
    *,
    definition: Mapping[str, Any],
    provider: str,
    model: str,
    run_root: Path,
    env_file: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    executor = LiveBenchmarkExecutor(run_root, env_file)
    observations = []
    for case in definition["cases"]:
        observations.append(executor.execute(case, provider=provider, model=model))
    replay = ReplayExecutor({item["case_id"]: item for item in observations})
    run = BenchmarkRunner(definition, replay).run(provider=provider, model=model)
    _augment_failure_taxonomy(run)
    return run, observations


def _git_output(worktree: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=worktree, check=False, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


def dirty_state_fingerprint(worktree: Path) -> str:
    status = _git_output(worktree, "status", "--porcelain=v1", "--untracked-files=all")
    status_lines = [
        line for line in status.splitlines()
        if not line[3:].replace("\\", "/").startswith("benchmarks/results/")
    ]
    status = "\n".join(status_lines)
    tracked_diff = _git_output(worktree, "diff", "--binary", "HEAD")
    untracked = []
    for line in status.splitlines():
        if line.startswith("?? "):
            path = worktree / line[3:]
            if path.is_file() and path.name != ".env":
                untracked.append({"path": line[3:], "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()})
    return stable_hash({"status": status, "tracked_diff": tracked_diff, "untracked": untracked})


def new_manifest(worktree: Path, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "benchmark_version": "antenna-agent-unseen-v1",
        "benchmark_sha256": verify_frozen_benchmark(),
        "git_revision": _git_output(worktree, "rev-parse", "HEAD"),
        "worktree_branch": _git_output(worktree, "branch", "--show-current"),
        "dirty_state_fingerprint": dirty_state_fingerprint(worktree),
        "providers": [
            {"provider": "gemini", "model": PROVIDERS["gemini"]},
            {"provider": "openrouter", "model": PROVIDERS["openrouter"]},
        ],
        "runner_budgets": LIVE_BUDGETS.to_dict(),
        "retry_policy": dict(RETRY_POLICY),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "started_at_utc": None,
        "ended_at_utc": None,
        "status": "awaiting_explicit_cloud_authorization",
    }


def write_complete_outputs(
    run_root: Path,
    *,
    gemini: Mapping[str, Any],
    nemotron: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    for stem, run in (("gemini", gemini), ("nemotron", nemotron)):
        (run_root / f"{stem}_results.json").write_text(
            json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_root / f"{stem}_summary.md").write_text(provider_summary(run), encoding="utf-8")
    (run_root / "cross_provider_comparison.md").write_text(
        cross_provider_markdown(gemini, nemotron), encoding="utf-8"
    )
    inventory = failure_inventory({"gemini": gemini, "nemotron": nemotron})
    (run_root / "failure_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def configured_credentials(env_file: Path) -> dict[str, bool]:
    values: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return {
        "gemini": bool(os.environ.get("GEMINI_API_KEY") or values.get("GEMINI_API_KEY")),
        "openrouter": bool(os.environ.get("OPEN_ROUTER_API_KEY") or values.get("OPEN_ROUTER_API_KEY") or values.get("OPENROUTER_API_KEY")),
    }
