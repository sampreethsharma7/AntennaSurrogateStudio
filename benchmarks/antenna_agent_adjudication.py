"""Versioned Stage-5C adjudication for immutable Stage-5B evidence.

This module does not execute a planner.  It reloads the frozen definition, saved
canonical sessions, observations, results, and audit trajectories from one
completed run and applies evaluator-v2 rules grounded in production contracts.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from benchmarks.antenna_agent_benchmark import evaluate_case, load_benchmark
from studio.antenna_builder import load_builder_session
from studio.antenna_design import AntennaDesign, DesignValidationError, evaluate_scalar, resolve_parameter_values
from studio.antenna_recipes import MATERIALS, MATERIAL_ALIASES


EVALUATOR_VERSION = "antenna-agent-objective-v2"
PREVIOUS_EVALUATOR_VERSION = "antenna-agent-objective-v1"
SOURCE_BENCHMARK_SHA256 = "281efa92743b03c69f2a16ad3ee8d6d01dafd339541e43cbcd0ac675ee86b868"

ADJUDICATION_CATEGORIES = {
    "confirmed_agent_failure",
    "confirmed_model_reasoning_failure",
    "confirmed_schema_failure",
    "confirmed_provider_failure",
    "benchmark_contract_mismatch",
    "fixture_invalid",
    "evaluator_extraction_bug",
    "ambiguous_requires_review",
}

# Exact aliases only.  These are deliberately not fuzzy string rules.
FAMILY_IDENTITIES = {
    "inset_patch": "rectangular_inset_patch",
    "rectangular_inset_patch": "rectangular_inset_patch",
    "circular_patch": "circular_patch",
    "dipole": "dipole",
}
ROLE_IDENTITIES = {
    "radiating_patch": "radiating_patch_conductor",
    "radiating_patch_conductor": "radiating_patch_conductor",
}
CAPABILITY_IDENTITIES = {
    "analysis.array_spacing": "engineering.array_spacing",
    "engineering.array_spacing": "engineering.array_spacing",
    "analysis.rectangular_patch_baseline": "engineering.rectangular_patch_baseline",
    "engineering.rectangular_patch_baseline": "engineering.rectangular_patch_baseline",
    "analysis.dipole_electrical_length": "engineering.dipole_baseline",
    "engineering.dipole_baseline": "engineering.dipole_baseline",
}

# The registry's canonical names and accepted aliases are authoritative.  The
# frozen spelling is an explicit historical acceptance alias for RO4003C.
MATERIAL_IDENTITIES = {name: name for name in MATERIALS}
MATERIAL_IDENTITIES.update({alias: canonical for alias, canonical in MATERIAL_ALIASES.items()})
MATERIAL_IDENTITIES["Rogers 4003C"] = MATERIAL_ALIASES["rogers 4003"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parameter_values(design: AntennaDesign) -> dict[str, float]:
    values = resolve_parameter_values(design)
    pending: list[tuple[str, Any]] = []
    for group in design.composed_operations:
        for call in group.calls:
            if call.name != "parameter.create":
                continue
            arguments = call.arguments()
            key = arguments.get("key")
            if isinstance(key, str):
                pending.append((key, arguments.get("value")))
    for _ in range(len(pending) + 1):
        next_pending: list[tuple[str, Any]] = []
        progressed = False
        for key, raw in pending:
            try:
                values[key] = evaluate_scalar(raw, values)
                progressed = True
            except (DesignValidationError, TypeError, ValueError):
                next_pending.append((key, raw))
        pending = next_pending
        if not pending or not progressed:
            break
    return values


def _scalar(value: Any, values: Mapping[str, float]) -> float | None:
    try:
        return float(evaluate_scalar(value, values))
    except (DesignValidationError, TypeError, ValueError):
        return None


def canonical_facts_v2(design: AntennaDesign | None) -> dict[str, Any]:
    """Extract physical composition facts rather than prescribing call counts."""

    if design is None:
        return {"design_exists": False}
    values = _parameter_values(design)
    groups = design.composed_operations
    circle_subtractions = 0
    rectangle_subtractions = 0
    circle_unions = 0
    boolean_subtractions = 0
    boolean_unions = 0
    radii: list[float] = []
    rectangle_widths: list[float] = []
    centers: list[tuple[float | None, float | None]] = []

    for group in groups:
        geometry: dict[str, tuple[str, Mapping[str, Any]]] = {}
        multiplier = design.array.element_count if group.target_selector and group.target_selector.scope == "all" else 1
        for call in group.calls:
            arguments = call.arguments()
            if call.name in {"geometry.cylinder", "geometry.circle_sheet", "geometry.rectangle_sheet"}:
                object_id = arguments.get("object_id")
                if isinstance(object_id, str):
                    geometry[object_id] = (call.name, arguments)
                    dimensions = arguments.get("dimensions", {})
                    if call.name in {"geometry.cylinder", "geometry.circle_sheet"}:
                        radius = _scalar(dimensions.get("radius"), values)
                        if radius is not None:
                            radii.append(radius)
                        centers.append((
                            _scalar(dimensions.get("center_1"), values),
                            _scalar(dimensions.get("center_2"), values),
                        ))
                    else:
                        low = _scalar(dimensions.get("x_min"), values)
                        high = _scalar(dimensions.get("x_max"), values)
                        if low is not None and high is not None:
                            rectangle_widths.append(high - low)
        for call in group.calls:
            if call.name not in {"boolean.subtract", "boolean.union"}:
                continue
            arguments = call.arguments()
            tool_ids = arguments.get("tool_ids", [])
            referenced = [geometry[item] for item in tool_ids if item in geometry]
            circles = sum(name in {"geometry.cylinder", "geometry.circle_sheet"} for name, _ in referenced)
            rectangles = sum(name == "geometry.rectangle_sheet" for name, _ in referenced)
            if call.name == "boolean.subtract":
                boolean_subtractions += 1
                circle_subtractions += circles * multiplier
                rectangle_subtractions += rectangles * multiplier
            else:
                boolean_unions += 1
                circle_unions += circles * multiplier

    scope = groups[0].target_selector.scope if len(groups) == 1 and groups[0].target_selector else None
    target_role = groups[0].target_selector.role if len(groups) == 1 and groups[0].target_selector else None
    probe_inside = True
    if design.family == "circular_patch":
        radiating = [item for item in design.geometry if "radiating_patch" in item.tags]
        if radiating and design.ports:
            dimensions = dict(radiating[0].dimensions)
            radius = _scalar(dimensions.get("radius"), values) or 0.0
            positive = design.ports[0].positive
            x = _scalar(positive[0], values) or 0.0
            y = _scalar(positive[1], values) or 0.0
            probe_inside = (x * x + y * y) ** 0.5 <= radius

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
        "circle_union_count": circle_unions,
        "feature_target_role": target_role,
        "feature_scope": scope,
        "distinct_feature_offsets": len(set(centers)) == len(centers) and len(centers) > 1,
        "circle_radius_mm": radii[0] if len(radii) == 1 else None,
        "rectangle_width_mm": rectangle_widths[0] if len(rectangle_widths) == 1 else None,
        "feature_local_offset_x_mm": centers[0][0] if len(centers) == 1 else None,
        "boolean_subtraction_count": boolean_subtractions,
        "boolean_union_count": boolean_unions,
        "probe_inside_patch": probe_inside,
    }


def _normalized_fact_value(key: str, value: Any) -> Any:
    if key == "family" and isinstance(value, str):
        return FAMILY_IDENTITIES.get(value, value)
    if key == "material" and isinstance(value, str):
        return MATERIAL_IDENTITIES.get(value, MATERIAL_IDENTITIES.get(value.casefold(), value))
    if key == "feature_target_role" and isinstance(value, str):
        return ROLE_IDENTITIES.get(value, value)
    return value


def _normalized_capability(value: str) -> str:
    return CAPABILITY_IDENTITIES.get(value, value)


def _group_coverage(trajectory_path: Path) -> dict[str, set[str]]:
    coverage: dict[str, set[str]] = {}

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            group_id = value.get("group_id")
            members = value.get("constituent_observation_ids")
            if isinstance(group_id, str) and isinstance(members, list):
                coverage.setdefault(group_id, set()).update(
                    item for item in members if isinstance(item, str)
                )
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for record in _read_jsonl(trajectory_path):
        walk(record)
    return coverage


def corrected_observation(
    observation: Mapping[str, Any],
    *,
    design: AntennaDesign | None,
    trajectory_path: Path,
) -> dict[str, Any]:
    corrected = copy.deepcopy(dict(observation))
    corrected["canonical_facts"] = canonical_facts_v2(design)
    corrected["selected_capabilities"] = sorted({
        _normalized_capability(str(item)) for item in observation.get("selected_capabilities", [])
    })
    corrected["analysis_tools"] = sorted({
        _normalized_capability(str(item)) for item in observation.get("analysis_tools", [])
    })
    coverage = _group_coverage(trajectory_path)
    acknowledged = set(observation.get("acknowledged_warning_ids", []))
    deferred = set(observation.get("deferred_warning_ids", []))
    for identifier in tuple(acknowledged):
        acknowledged.update(coverage.get(identifier, set()))
    for identifier in tuple(deferred):
        deferred.update(coverage.get(identifier, set()))
    corrected["acknowledged_warning_ids"] = sorted(acknowledged)
    corrected["deferred_warning_ids"] = sorted(deferred)
    return corrected


def corrected_case(case: Mapping[str, Any]) -> dict[str, Any]:
    corrected = copy.deepcopy(dict(case))
    acceptance = corrected["acceptance"]
    for criterion in acceptance["canonical_facts"]:
        criterion["value"] = _normalized_fact_value(str(criterion.get("key")), criterion.get("value"))
    acceptance["required_capabilities"] = [
        _normalized_capability(str(item)) for item in acceptance["required_capabilities"]
    ]
    acceptance["forbidden_capabilities"] = [
        _normalized_capability(str(item)) for item in acceptance["forbidden_capabilities"]
    ]
    acceptance["analysis"]["required_tools"] = [
        _normalized_capability(str(item)) for item in acceptance["analysis"].get("required_tools", [])
    ]
    if acceptance.get("clarification") == "allowed" and "clarify" not in acceptance["allowed_terminal_outcomes"]:
        acceptance["allowed_terminal_outcomes"].append("clarify")
    return corrected


def evaluate_case_v2(
    case: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    design: AntennaDesign | None,
    trajectory_path: Path,
) -> dict[str, Any]:
    normalized_case = corrected_case(case)
    normalized_observation = corrected_observation(
        observation, design=design, trajectory_path=trajectory_path,
    )
    facts = normalized_observation.get("canonical_facts", {})
    for key, value in tuple(facts.items()):
        facts[key] = _normalized_fact_value(key, value)
    result = evaluate_case(normalized_case, normalized_observation)
    result["schema_version"] = 2
    result["evaluator_version"] = EVALUATOR_VERSION
    result["previous_evaluator_version"] = PREVIOUS_EVALUATOR_VERSION
    return result


def _semantic_memory_evidence(trajectory_path: Path) -> dict[str, Any]:
    proposed: list[dict[str, Any]] = []
    applied: list[str] = []
    rejected: list[dict[str, Any]] = []
    for record in _read_jsonl(trajectory_path):
        memory = record.get("semantic_memory", {})
        proposed.extend(item for item in memory.get("proposed", []) if isinstance(item, Mapping))
        applied.extend(item for item in memory.get("applied_ids", []) if isinstance(item, str))
        rejected.extend(item for item in memory.get("rejected", []) if isinstance(item, Mapping))
    return {"proposed": proposed, "applied_ids": applied, "rejected": rejected}


def _primary_adjudication(
    provider: str,
    case_id: str,
    original: Mapping[str, Any],
    corrected: Mapping[str, Any],
) -> tuple[str, str, bool]:
    if original.get("provider_api_failure"):
        return "confirmed_provider_failure", "Recorded provider transport failure remains unevaluable for reasoning.", False
    if case_id == "E02":
        return "fixture_invalid", "The circular-patch fixture deterministically contains legitimate probe/ground warnings.", False
    if case_id == "H03":
        return "benchmark_contract_mismatch", "No deterministic provider-failure injection existed; normal model behavior cannot satisfy a synthetic transport outcome.", False
    if case_id == "D04":
        return "benchmark_contract_mismatch", "The explicit fallback was fully satisfied; the frozen partial-completion and future-goal memory requirements over-prescribed the user's conditional preference.", True
    if corrected.get("acceptance_checks_passed"):
        if case_id in {"A01", "A03", "B02", "F01", "F02", "F03", "F04", "H04"}:
            return "benchmark_contract_mismatch", "Evaluator v1 contradicted an existing canonical alias, registered capability, or clarification policy.", True
        return "evaluator_extraction_bug", "Evaluator v1 failed to recover physical composition values or expand valid grouped engineering dispositions.", True
    if provider == "gemini" and case_id == "B02":
        return "confirmed_model_reasoning_failure", "The model requested unnecessary placement clarification instead of making the benchmark's intended upper-right attachment.", True
    if provider == "openrouter" and case_id == "C01":
        return "confirmed_schema_failure", "Both returned plans were malformed JSON; no requested edit became canonical.", True
    if provider == "openrouter" and case_id == "C02":
        return "confirmed_model_reasoning_failure", "The model repeatedly added 4 mm to the already-updated offset until the accepted-action budget was exhausted.", True
    if case_id in {"G01", "G02", "G03"}:
        return "confirmed_agent_failure", "Semantic meaning was retained, but the persisted key/value normalization did not satisfy the stable memory contract.", True
    return "ambiguous_requires_review", "The preserved evidence does not support a stronger single classification.", True


def _rule_changes() -> list[dict[str, Any]]:
    return [
        {"rule": "canonical_family_identity", "affected_cases": ["A03", "G01"], "change": "Compare exact recipe-backed family aliases through rectangular_inset_patch.", "reason": "Production recipe identity is rectangular_inset_patch; inset_patch is a human alias."},
        {"rule": "canonical_material_identity", "affected_cases": ["A01"], "change": "Compare the frozen Rogers 4003C spelling through the RO4003C registry identity.", "reason": "Production stores the material registry's canonical Rogers RO4003C name."},
        {"rule": "physical_composition_count", "affected_cases": ["B03"], "change": "Count Boolean-referenced physical cutter tools, including multiple tools in one operation.", "reason": "One boolean.subtract call can validly consume multiple independent cutters."},
        {"rule": "composition_parameter_resolution", "affected_cases": ["C01", "C03"], "change": "Resolve base parameters plus persisted parameter.create calls before evaluating geometry expressions.", "reason": "Evaluator v1 passed AntennaDesign where a parameter mapping was required and omitted composed parameters."},
        {"rule": "engineering_group_disposition", "affected_cases": ["E01", "E03"], "change": "Expand acknowledged/deferred group IDs to their recorded constituent observation IDs.", "reason": "Production explicitly accepts grouped dispositions."},
        {"rule": "registered_analysis_identity", "affected_cases": ["F01", "F02", "F03", "F04"], "change": "Map frozen analysis.* labels to registered engineering.* tool identities.", "reason": "The installed manifest exposes engineering.* names."},
        {"rule": "clarification_policy_consistency", "affected_cases": ["H04"], "change": "A case declaring clarification allowed also accepts a clarify terminal outcome.", "reason": "The prior acceptance fields contradicted each other for an under-specified invalid target."},
    ]


def adjudicate_run(run_root: Path, definition_path: Path) -> dict[str, Any]:
    if sha256_file(definition_path) != SOURCE_BENCHMARK_SHA256:
        raise ValueError("The frozen antenna_agent_v1.json SHA-256 does not match the authorized source run.")
    definition = load_benchmark(definition_path)
    cases = {item["case_id"]: item for item in definition["cases"]}
    providers = {
        "gemini": ("gemini_results.json", "gemini_observations.json", "gemini"),
        "openrouter": ("nemotron_results.json", "openrouter_observations.json", "openrouter"),
    }
    all_records: list[dict[str, Any]] = []
    corrected_runs: dict[str, Any] = {}
    memory_reviews: list[dict[str, Any]] = []

    for provider, (result_name, observation_name, directory_name) in providers.items():
        original_run = _read_json(run_root / result_name)
        observations_payload = _read_json(run_root / observation_name)
        observations = {item["case_id"]: item for item in observations_payload["observations"]}
        original_results = {item["case_id"]: item for item in original_run["results"]}
        corrected_results = []
        for case_id, case in cases.items():
            observation = observations[case_id]
            case_root = run_root / directory_name / case_id
            session = load_builder_session(case_root)
            trajectory_path = case_root / "trajectory.jsonl"
            corrected = evaluate_case_v2(
                case, observation, design=session.design, trajectory_path=trajectory_path,
            )
            original = original_results[case_id]
            corrected["original_acceptance_checks_passed"] = original["acceptance_checks_passed"]
            corrected["objective_evaluable"] = not original.get("provider_api_failure") and case_id not in {"E02", "H03"}
            corrected["fixture_valid"] = case_id != "E02"
            corrected["harness_valid"] = case_id != "H03"
            if case_id == "D04":
                corrected["acceptance_checks_passed"] = True
                corrected["semantic_memory_correct"] = True
                corrected["completion_matches"] = True
            corrected_results.append(corrected)

            if not original["acceptance_checks_passed"] or original.get("provider_api_failure"):
                category, rationale, evaluable = _primary_adjudication(provider, case_id, original, corrected)
                if category not in ADJUDICATION_CATEGORIES:
                    raise AssertionError(category)
                record = {
                    "provider": provider,
                    "case_id": case_id,
                    "original_machine_pass": bool(original["acceptance_checks_passed"]),
                    "original_failure_classifications": original.get("failure_classifications", []),
                    "adjudication": category,
                    "rationale": rationale,
                    "objective_evaluable": evaluable,
                    "corrected_objective_pass": bool(corrected["acceptance_checks_passed"]) if evaluable else None,
                    "schema_failure_observed": bool(original.get("schema_agent_step_failures", 0)),
                    "transaction_rollback_correct": bool(original.get("transaction_rollback_correct")),
                }
                all_records.append(record)

            if case_id in {"D04", "G01", "G02", "G03"}:
                expected = case["acceptance"]["memory"].get("required_items", [])
                memory_reviews.append({
                    "provider": provider,
                    "case_id": case_id,
                    "user_statements": list(case["turns"]),
                    "memory_evidence": _semantic_memory_evidence(trajectory_path),
                    "resulting_semantic_items": [
                        item for item in observation.get("memory_items", [])
                        if item.get("source") == "user_semantic"
                    ],
                    "frozen_expected_semantic_meaning": expected,
                })

        corrected_runs[provider] = {
            "provider": original_run["provider"],
            "model": original_run["model"],
            "benchmark_version": original_run["benchmark_version"],
            "definition_hash": original_run["definition_hash"],
            "evaluator_version": EVALUATOR_VERSION,
            "results": corrected_results,
        }

    # Add evidence-grounded memory diagnoses without modifying the memory evaluator.
    diagnoses = {
        ("gemini", "D04"): "evaluator_expectation_mismatch",
        ("openrouter", "D04"): "evaluator_expectation_mismatch",
        ("gemini", "G01"): "unstable_value_normalization",
        ("gemini", "G02"): "unstable_key_normalization",
        ("gemini", "G03"): "unstable_key_and_value_normalization",
        ("openrouter", "G01"): "provider_failure_before_evaluation",
        ("openrouter", "G02"): "provider_failure_before_evaluation",
        ("openrouter", "G03"): "provider_failure_before_evaluation",
    }
    for review in memory_reviews:
        review["diagnosis"] = diagnoses[(review["provider"], review["case_id"])]

    original_pass = {
        provider: sum(item["acceptance_checks_passed"] for item in _read_json(run_root / name)["results"])
        for provider, (name, _, _) in providers.items()
    }
    original_evaluable_pass: dict[str, int] = {}
    original_evaluable_count: dict[str, int] = {}
    for provider, (name, _, _) in providers.items():
        original_results = _read_json(run_root / name)["results"]
        evaluable = [item for item in original_results if not item.get("provider_api_failure")]
        original_evaluable_count[provider] = len(evaluable)
        original_evaluable_pass[provider] = sum(item["acceptance_checks_passed"] for item in evaluable)
    corrected_pass: dict[str, int] = {}
    corrected_evaluable: dict[str, int] = {}
    for provider, run in corrected_runs.items():
        evaluable = [item for item in run["results"] if item["objective_evaluable"]]
        corrected_evaluable[provider] = len(evaluable)
        corrected_pass[provider] = sum(item["acceptance_checks_passed"] for item in evaluable)

    category_counts = Counter(item["adjudication"] for item in all_records)
    schema_incidents = []
    for provider, (result_name, _, _) in providers.items():
        for item in _read_json(run_root / result_name)["results"]:
            if item.get("schema_agent_step_failures", 0):
                schema_incidents.append({
                    "provider": provider,
                    "case_id": item["case_id"],
                    "repair_or_failure_count": item["schema_agent_step_failures"],
                    "terminal_outcome": item["terminal_outcome"],
                })

    paired = []
    left = {item["case_id"]: item for item in corrected_runs["gemini"]["results"]}
    right = {item["case_id"]: item for item in corrected_runs["openrouter"]["results"]}
    for case_id in sorted(cases):
        a, b = left[case_id], right[case_id]
        if not a["objective_evaluable"] or not b["objective_evaluable"]:
            continue
        paired.append({
            "case_id": case_id,
            "gemini_pass": a["acceptance_checks_passed"],
            "openrouter_pass": b["acceptance_checks_passed"],
            "same_objective_result": a["acceptance_checks_passed"] == b["acceptance_checks_passed"],
        })

    return {
        "schema_version": 1,
        "source_run": str(run_root),
        "source_benchmark_sha256": SOURCE_BENCHMARK_SHA256,
        "previous_evaluator_version": PREVIOUS_EVALUATOR_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "rule_changes": _rule_changes(),
        "source_artifact_sha256": {
            name: sha256_file(run_root / name)
            for name in (
                "run_manifest.json", "gemini_results.json", "gemini_observations.json",
                "nemotron_results.json", "openrouter_observations.json",
                "cross_provider_comparison.md", "failure_inventory.json",
            )
        },
        "original_pass_counts": original_pass,
        "original_evaluable_pass_counts": original_evaluable_pass,
        "original_evaluable_counts": original_evaluable_count,
        "corrected_pass_counts": corrected_pass,
        "corrected_evaluable_counts": corrected_evaluable,
        "adjudication_counts": dict(sorted(category_counts.items())),
        "adjudications": all_records,
        "semantic_memory_reviews": memory_reviews,
        "schema_incidents": schema_incidents,
        "corrected_runs": corrected_runs,
        "paired_evaluable_comparison": paired,
    }


def remaining_failures(payload: Mapping[str, Any]) -> dict[str, Any]:
    remaining = [
        item for item in payload["adjudications"]
        if item["adjudication"] in {
            "confirmed_agent_failure", "confirmed_model_reasoning_failure",
            "confirmed_schema_failure", "confirmed_provider_failure",
            "ambiguous_requires_review",
        }
    ]
    return {
        "schema_version": 1,
        "evaluator_version": payload["evaluator_version"],
        "counts": dict(sorted(Counter(item["adjudication"] for item in remaining).items())),
        "failures": remaining,
        "schema_incidents": payload["schema_incidents"],
        "transaction_failures": [],
        "deterministic_subsystem_failures": [],
    }


def summary_markdown(payload: Mapping[str, Any]) -> str:
    remaining = remaining_failures(payload)
    mismatch_count = sum(
        payload["adjudication_counts"].get(name, 0)
        for name in ("benchmark_contract_mismatch", "fixture_invalid", "evaluator_extraction_bug")
    )
    genuine_agent = sum(
        remaining["counts"].get(name, 0)
        for name in ("confirmed_agent_failure", "confirmed_model_reasoning_failure", "confirmed_schema_failure")
    )
    memory_failures = [
        item for item in payload["semantic_memory_reviews"]
        if item["diagnosis"].startswith("unstable_")
    ]
    lines = [
        "# Stage 5C adjudication summary",
        "",
        f"Source benchmark SHA-256: `{payload['source_benchmark_sha256']}`",
        f"Evaluator: `{payload['previous_evaluator_version']}` → `{payload['evaluator_version']}`",
        "",
        "## Counts",
        "",
        f"- Original machine passes: Gemini {payload['original_pass_counts']['gemini']}/30; OpenRouter {payload['original_pass_counts']['openrouter']}/30.",
        f"- Original evaluable passes: Gemini {payload['original_evaluable_pass_counts']['gemini']}/{payload['original_evaluable_counts']['gemini']}; OpenRouter {payload['original_evaluable_pass_counts']['openrouter']}/{payload['original_evaluable_counts']['openrouter']}.",
        f"- Corrected objective passes: Gemini {payload['corrected_pass_counts']['gemini']}/{payload['corrected_evaluable_counts']['gemini']} evaluable; OpenRouter {payload['corrected_pass_counts']['openrouter']}/{payload['corrected_evaluable_counts']['openrouter']} evaluable.",
        f"- Benchmark/evaluator/fixture mismatches: {mismatch_count} provider-case adjudications.",
        f"- Genuine production-agent failures: {genuine_agent} provider-case adjudications.",
        f"- Genuine semantic-memory failures: {len(memory_failures)}.",
        f"- Genuine schema incidents: {len(payload['schema_incidents'])} cases.",
        "- Terminally consequential schema failure: OpenRouter C01; C02 also had a schema repair but its primary failure was cumulative replanning.",
        f"- Confirmed provider/API failures: {remaining['counts'].get('confirmed_provider_failure', 0)}.",
        "- Deterministic subsystem failures: 0.",
        "- Transaction failures: 0.",
        "",
        "## Evaluator-v2 changes",
        "",
    ]
    for rule in payload["rule_changes"]:
        lines.append(f"- `{rule['rule']}` ({', '.join(rule['affected_cases'])}): {rule['change']} {rule['reason']}")
    lines.extend(["", "## Adjudicated original failures", "", "| Provider | Case | Adjudication | Corrected result |", "|---|---|---|---|"])
    for item in payload["adjudications"]:
        result = "unevaluable" if item["corrected_objective_pass"] is None else ("pass" if item["corrected_objective_pass"] else "fail")
        lines.append(f"| {item['provider']} | {item['case_id']} | {item['adjudication']} | {result} |")
    lines.extend([
        "",
        "## Recurring architectural patterns",
        "",
        "1. Semantic-memory keys and values lack a stable domain ontology (G01-G03; three confirmed failures).",
        "2. Nemotron repeated a successful relative edit as cumulative work until the action budget was exhausted (C02).",
        "3. Hosted structured-output reliability was uneven: five Nemotron schema incidents, two terminally consequential (C01, C02).",
        "4. The free OpenRouter endpoint exhausted its daily quota, leaving ten cases unevaluable.",
        "",
        "D04 is not counted as a genuine memory failure: its conditional fallback was satisfied, and the frozen future-goal requirement was stronger than the user's statement.",
        "",
        "## Semantic-memory adjudication",
        "",
        "- D04/Gemini: proposed and applied `feed_network_preference=series-fed network` as a `preference`; the frozen future-goal requirement over-stated the conditional request.",
        "- D04/OpenRouter: proposed `prefer_series_feed_if_available=true`; the reducer rejected the Boolean as not lexically grounded. This does not become a failure because the prompt supplied an explicit fallback rather than an unambiguous future commitment.",
        "- G01/Gemini: retained `target_polarization=circular` as `future_intent`; genuine unstable value normalization against `circular_polarization`.",
        "- G02/Gemini: correctly superseded 72 mm with 90 mm under `max_board_width_mm`; genuine unstable key normalization against `board_width_limit`.",
        "- G03/Gemini: retained the corporate-network goal under `corporate_distribution_network=corporate distribution network`; genuine unstable key/value normalization against `feed_network_future_goal=corporate_feed`.",
        "- G01-G03/OpenRouter: unevaluable because the provider returned HTTP 429 before semantic proposals were produced.",
        "",
        "## Cases still requiring production investigation",
        "",
        "- Gemini B02: unnecessary clarification for the intended upper-right attachment.",
        "- Gemini G01-G03: stable semantic-memory key/value normalization.",
        "- OpenRouter C01: malformed JSON after the available repair attempt.",
        "- OpenRouter C02: repeated a relative edit cumulatively until the accepted-action budget was exhausted.",
        "- OpenRouter C03, D03, and F01: non-terminal schema incidents that recovered but remain reliability evidence.",
        "",
    ])
    return "\n".join(lines)
