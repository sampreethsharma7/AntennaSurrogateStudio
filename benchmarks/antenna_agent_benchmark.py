"""Frozen, provider-neutral benchmark harness for the antenna engineering agent.

This module deliberately depends only on recorded observable results or an injected
executor.  It does not alter or special-case the production planner or agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


BENCHMARK_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
DEFAULT_DEFINITION_PATH = Path(__file__).with_name("antenna_agent_v1.json")
FAILURE_TAXONOMY = (
    "missing_capability",
    "bad_tool_abstraction",
    "insufficient_observation_context",
    "planner_reasoning_failure",
    "schema_structured_output_failure",
    "provider_api_failure",
    "transaction_runtime_failure",
    "ambiguous_benchmark_request",
    "deterministic_subsystem_failure",
)
COMPLETION_VALUES = ("yes", "no", "partial")
CLARIFICATION_POLICIES = ("required", "allowed", "forbidden")
MUTATION_POLICIES = ("required", "forbidden", "optional")


class BenchmarkDefinitionError(ValueError):
    """Raised when frozen definitions or recorded observations are malformed."""


class BenchmarkExecutor(Protocol):
    """Pluggable executor; live provider adapters can be supplied without changing cases."""

    def execute(
        self,
        case: Mapping[str, Any],
        *,
        provider: str,
        model: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class BenchmarkSelection:
    case_id: str | None = None
    category: str | None = None

    def select(self, cases: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
        selected = tuple(
            case for case in cases
            if (self.case_id is None or case["case_id"] == self.case_id)
            and (self.category is None or case["category"] == self.category)
        )
        if self.case_id is not None and not selected:
            raise BenchmarkDefinitionError(f"Unknown benchmark case: {self.case_id}")
        if self.category is not None and not selected:
            raise BenchmarkDefinitionError(f"Unknown benchmark category: {self.category}")
        return selected


class ReplayExecutor:
    """Deterministic executor for mocked or previously recorded provider observations."""

    def __init__(self, observations: Mapping[str, Mapping[str, Any]]) -> None:
        self._observations = dict(observations)

    @classmethod
    def from_file(cls, path: str | Path) -> "ReplayExecutor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = payload.get("observations") if isinstance(payload, Mapping) else None
        if not isinstance(raw, list):
            raise BenchmarkDefinitionError("Replay input must contain an observations array.")
        observations: dict[str, Mapping[str, Any]] = {}
        for item in raw:
            if not isinstance(item, Mapping) or not isinstance(item.get("case_id"), str):
                raise BenchmarkDefinitionError("A replay observation is malformed.")
            if item["case_id"] in observations:
                raise BenchmarkDefinitionError(f"Duplicate replay observation: {item['case_id']}")
            observations[item["case_id"]] = item
        return cls(observations)

    def execute(
        self,
        case: Mapping[str, Any],
        *,
        provider: str,
        model: str,
    ) -> Mapping[str, Any]:
        try:
            observation = dict(self._observations[case["case_id"]])
        except KeyError as exc:
            raise BenchmarkDefinitionError(
                f"Replay input has no observation for {case['case_id']}."
            ) from exc
        recorded_provider = observation.get("provider")
        recorded_model = observation.get("model")
        if recorded_provider not in (None, provider) or recorded_model not in (None, model):
            raise BenchmarkDefinitionError(
                f"Replay observation {case['case_id']} belongs to a different provider/model."
            )
        observation["provider"] = provider
        observation["model"] = model
        return observation


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_benchmark(path: str | Path = DEFAULT_DEFINITION_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_benchmark(payload)
    return payload


def _validate_benchmark(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise BenchmarkDefinitionError("Benchmark definition must be an object.")
    required = {"schema_version", "benchmark_version", "title", "categories", "fixtures", "cases"}
    if set(payload) != required or payload["schema_version"] != BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkDefinitionError("Benchmark definition does not match schema version 1.")
    if not isinstance(payload["benchmark_version"], str) or not payload["benchmark_version"]:
        raise BenchmarkDefinitionError("benchmark_version must be non-empty.")
    if not isinstance(payload["categories"], Mapping) or not payload["categories"]:
        raise BenchmarkDefinitionError("Benchmark categories are missing.")
    if not isinstance(payload["fixtures"], Mapping) or "empty_project" not in payload["fixtures"]:
        raise BenchmarkDefinitionError("Benchmark fixtures are missing.")
    if not isinstance(payload["cases"], list) or not 24 <= len(payload["cases"]) <= 30:
        raise BenchmarkDefinitionError("Frozen benchmark v1 must contain 24 to 30 cases.")
    ids: set[str] = set()
    for case in payload["cases"]:
        _validate_case(case, payload["categories"], payload["fixtures"])
        if case["case_id"] in ids:
            raise BenchmarkDefinitionError(f"Duplicate case ID: {case['case_id']}")
        ids.add(case["case_id"])


def _validate_case(
    case: Any,
    categories: Mapping[str, Any],
    fixtures: Mapping[str, Any],
) -> None:
    required = {"case_id", "category", "title", "initial_fixture", "turns", "acceptance"}
    if not isinstance(case, Mapping) or set(case) != required:
        raise BenchmarkDefinitionError("A benchmark case has unexpected fields.")
    if not isinstance(case["case_id"], str) or not case["case_id"]:
        raise BenchmarkDefinitionError("Case IDs must be non-empty strings.")
    if case["category"] not in categories or case["initial_fixture"] not in fixtures:
        raise BenchmarkDefinitionError(f"Case {case['case_id']} has an unknown category or fixture.")
    turns = case["turns"]
    if not isinstance(turns, list) or not turns or any(not isinstance(item, str) or not item for item in turns):
        raise BenchmarkDefinitionError(f"Case {case['case_id']} requires non-empty user turns.")
    acceptance = case["acceptance"]
    expected = {
        "allowed_terminal_outcomes", "completion", "mutation", "rollback_required",
        "clarification", "canonical_facts", "required_capabilities",
        "forbidden_capabilities", "engineering", "analysis", "memory",
    }
    if not isinstance(acceptance, Mapping) or set(acceptance) != expected:
        raise BenchmarkDefinitionError(f"Case {case['case_id']} acceptance schema is malformed.")
    if (
        not isinstance(acceptance["allowed_terminal_outcomes"], list)
        or not acceptance["allowed_terminal_outcomes"]
        or acceptance["completion"] not in COMPLETION_VALUES
        or acceptance["mutation"] not in MUTATION_POLICIES
        or type(acceptance["rollback_required"]) is not bool
        or acceptance["clarification"] not in CLARIFICATION_POLICIES
    ):
        raise BenchmarkDefinitionError(f"Case {case['case_id']} has invalid outcome policies.")
    for name in ("canonical_facts", "required_capabilities", "forbidden_capabilities"):
        if not isinstance(acceptance[name], list):
            raise BenchmarkDefinitionError(f"Case {case['case_id']} {name} must be an array.")
    for name in ("engineering", "analysis", "memory"):
        if not isinstance(acceptance[name], Mapping):
            raise BenchmarkDefinitionError(f"Case {case['case_id']} {name} must be an object.")


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "eq":
        return actual == expected
    if operator == "ne":
        return actual != expected
    if operator == "exists":
        return actual is not None
    if operator == "absent":
        return actual is None
    if operator == "gte":
        return isinstance(actual, (int, float)) and actual >= expected
    if operator == "lte":
        return isinstance(actual, (int, float)) and actual <= expected
    if operator == "contains":
        return isinstance(actual, (list, tuple, set, str)) and expected in actual
    if operator == "contains_all":
        return isinstance(actual, (list, tuple, set)) and set(expected).issubset(set(actual))
    raise BenchmarkDefinitionError(f"Unknown acceptance operator: {operator}")


def _fact_results(
    criteria: Sequence[Mapping[str, Any]],
    facts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results = []
    for criterion in criteria:
        key = criterion.get("key")
        operator = criterion.get("op", "eq")
        expected = criterion.get("value")
        actual = facts.get(key)
        results.append({
            "key": key,
            "operator": operator,
            "expected": expected,
            "actual": actual,
            "passed": _compare(actual, operator, expected),
        })
    return results


def _memory_results(
    criteria: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for criterion in criteria:
        matches = [
            item for item in items
            if item.get("key") == criterion.get("key")
            and item.get("status") == criterion.get("status", "active")
        ]
        expected_value = criterion.get("value")
        passed = bool(matches) and (
            "value" not in criterion
            or any(item.get("value") == expected_value for item in matches)
        )
        results.append({"criterion": dict(criterion), "matched": matches, "passed": passed})
    return results


def _classify_failures(observation: Mapping[str, Any]) -> list[str]:
    signals = observation.get("failure_signals", [])
    if not isinstance(signals, list):
        raise BenchmarkDefinitionError("failure_signals must be an array.")
    unknown = sorted(set(signals) - set(FAILURE_TAXONOMY))
    if unknown:
        raise BenchmarkDefinitionError(f"Unknown failure classifications: {unknown}")
    failures = list(dict.fromkeys(signals))
    if observation.get("provider_api_failure") and "provider_api_failure" not in failures:
        failures.append("provider_api_failure")
    if observation.get("schema_agent_step_failures", 0) and "schema_structured_output_failure" not in failures:
        failures.append("schema_structured_output_failure")
    return failures


def evaluate_case(case: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate only observable state and structured run records."""

    for name in (
        "initial_semantic_hash", "final_semantic_hash", "initial_memory_hash", "final_memory_hash",
    ):
        if not _is_hash(observation.get(name)):
            raise BenchmarkDefinitionError(f"Observation {case['case_id']} has invalid {name}.")
    report_hashes = observation.get("engineering_report_hashes", [])
    if not isinstance(report_hashes, list) or any(not _is_hash(item) for item in report_hashes):
        raise BenchmarkDefinitionError("engineering_report_hashes must contain SHA-256 digests.")

    acceptance = case["acceptance"]
    terminal = observation.get("terminal_outcome")
    completion = observation.get("task_completed")
    if completion not in COMPLETION_VALUES:
        raise BenchmarkDefinitionError("task_completed must be yes, no, or partial.")
    canonical = _fact_results(
        acceptance["canonical_facts"],
        observation.get("canonical_facts", {}),
    )
    memory = _memory_results(
        acceptance["memory"].get("required_items", []),
        observation.get("memory_items", []),
    )
    selected = set(observation.get("selected_capabilities", []))
    hallucinated = set(observation.get("hallucinated_capabilities", []))
    required_capabilities = set(acceptance["required_capabilities"])
    forbidden_capabilities = set(acceptance["forbidden_capabilities"])

    changed = observation["initial_semantic_hash"] != observation["final_semantic_hash"]
    mutation_correct = (
        acceptance["mutation"] == "optional"
        or (acceptance["mutation"] == "required" and changed)
        or (acceptance["mutation"] == "forbidden" and not changed)
    )
    rollback_correct = (
        not acceptance["rollback_required"]
        or observation["initial_semantic_hash"] == observation["final_semantic_hash"]
    )
    clarification = terminal == "clarify"
    clarification_policy = acceptance["clarification"]
    required_clarification_missed = clarification_policy == "required" and not clarification
    unnecessary_clarification = clarification_policy == "forbidden" and clarification

    engineering = acceptance["engineering"]
    warning_ids = set(observation.get("engineering_warning_ids", []))
    accounted = set(observation.get("acknowledged_warning_ids", [])) | set(
        observation.get("deferred_warning_ids", [])
    )
    required_warning_ids = set(engineering.get("required_warning_ids", []))
    minimum_warning_count = int(engineering.get("minimum_warning_count", 0))
    maximum_warning_count = engineering.get("maximum_warning_count")
    disposition_required = bool(engineering.get("disposition_required", False))
    engineering_correct = (
        required_warning_ids.issubset(warning_ids)
        and len(warning_ids) >= minimum_warning_count
        and (maximum_warning_count is None or len(warning_ids) <= int(maximum_warning_count))
        and (not disposition_required or warning_ids.issubset(accounted))
    )
    presentation_omissions = sorted(warning_ids - set(observation.get("warnings_mentioned_in_prose", [])))

    analysis = acceptance["analysis"]
    required_analysis = set(analysis.get("required_tools", []))
    observed_analysis = set(observation.get("analysis_tools", []))
    unsupported_claims = list(observation.get("unsupported_model_claims", []))
    analysis_correct = required_analysis.issubset(observed_analysis) and not unsupported_claims

    failures = _classify_failures(observation)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "category": case["category"],
        "provider": observation.get("provider"),
        "model": observation.get("model"),
        "initial_fixture": case["initial_fixture"],
        "terminal_outcome": terminal,
        "terminal_outcome_allowed": terminal in acceptance["allowed_terminal_outcomes"],
        "task_completed": completion,
        "completion_expected": acceptance["completion"],
        "completion_matches": completion == acceptance["completion"],
        "canonical_state_correct": all(item["passed"] for item in canonical),
        "canonical_fact_results": canonical,
        "required_tool_capability_selected": required_capabilities.issubset(selected),
        "missing_required_capabilities": sorted(required_capabilities - selected),
        "unsupported_capability_hallucinated": bool(hallucinated | (selected & forbidden_capabilities)),
        "hallucinated_or_forbidden_capabilities": sorted(hallucinated | (selected & forbidden_capabilities)),
        "engineering_findings_accounted_for": engineering_correct,
        "presentation_warning_omissions": presentation_omissions,
        "analytical_result_grounded_correctly": analysis_correct,
        "unsupported_model_claims": unsupported_claims,
        "unnecessary_clarification": unnecessary_clarification,
        "required_clarification_missed": required_clarification_missed,
        "mutation_correct": mutation_correct,
        "transaction_rollback_correct": rollback_correct,
        "semantic_memory_correct": all(item["passed"] for item in memory),
        "memory_fact_results": memory,
        "schema_agent_step_failures": int(observation.get("schema_agent_step_failures", 0)),
        "provider_api_failure": bool(observation.get("provider_api_failure", False)),
        "execution_rejections": int(observation.get("execution_rejections", 0)),
        "decision_iterations": int(observation.get("decision_iterations", 0)),
        "design_action_batches": int(observation.get("design_action_batches", 0)),
        "analysis_batches": int(observation.get("analysis_batches", 0)),
        "failure_classifications": failures,
        "initial_semantic_hash": observation["initial_semantic_hash"],
        "final_semantic_hash": observation["final_semantic_hash"],
        "initial_memory_hash": observation["initial_memory_hash"],
        "final_memory_hash": observation["final_memory_hash"],
        "engineering_report_hashes": report_hashes,
        "trajectory_reference": observation.get("trajectory_reference"),
        "terminal_result": observation.get("terminal_result"),
    }
    result["acceptance_checks_passed"] = all((
        result["terminal_outcome_allowed"],
        result["completion_matches"],
        result["canonical_state_correct"],
        result["required_tool_capability_selected"],
        not result["unsupported_capability_hallucinated"],
        result["engineering_findings_accounted_for"],
        result["analytical_result_grounded_correctly"],
        not result["unnecessary_clarification"],
        not result["required_clarification_missed"],
        result["mutation_correct"],
        result["transaction_rollback_correct"],
        result["semantic_memory_correct"],
    ))
    return result


class BenchmarkRunner:
    def __init__(self, definition: Mapping[str, Any], executor: BenchmarkExecutor) -> None:
        _validate_benchmark(definition)
        self.definition = dict(definition)
        self.executor = executor

    def run(
        self,
        *,
        provider: str,
        model: str,
        selection: BenchmarkSelection = BenchmarkSelection(),
    ) -> dict[str, Any]:
        if not provider or not model:
            raise BenchmarkDefinitionError("Provider and model are required benchmark metadata.")
        cases = selection.select(self.definition["cases"])
        results = []
        for case in cases:
            observation = self.executor.execute(case, provider=provider, model=model)
            results.append(evaluate_case(case, observation))
        dimensions = {
            "cases": len(results),
            "completed_yes": sum(item["task_completed"] == "yes" for item in results),
            "completed_partial": sum(item["task_completed"] == "partial" for item in results),
            "canonical_state_correct": sum(item["canonical_state_correct"] for item in results),
            "semantic_memory_correct": sum(item["semantic_memory_correct"] for item in results),
            "rollback_correct": sum(item["transaction_rollback_correct"] for item in results),
            "provider_api_failures": sum(item["provider_api_failure"] for item in results),
            "schema_agent_step_failures": sum(item["schema_agent_step_failures"] for item in results),
            "execution_rejections": sum(item["execution_rejections"] for item in results),
        }
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "benchmark_version": self.definition["benchmark_version"],
            "definition_hash": _stable_hash(self.definition),
            "provider": provider,
            "model": model,
            "selection": {"case_id": selection.case_id, "category": selection.category},
            "dimension_totals": dimensions,
            "results": results,
        }


def markdown_summary(run: Mapping[str, Any]) -> str:
    lines = [
        f"# Antenna agent benchmark {run['benchmark_version']}",
        "",
        f"Provider/model: `{run['provider']}` / `{run['model']}`",
        "",
        "No single quality score is calculated. Each objective dimension remains separate.",
        "",
        "| Case | Category | Outcome | Completed | Canonical | Memory | Rollback | Failures |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for item in run["results"]:
        failures = ", ".join(item["failure_classifications"]) or "—"
        lines.append(
            f"| {item['case_id']} | {item['category']} | {item['terminal_outcome']} | "
            f"{item['task_completed']} | {'yes' if item['canonical_state_correct'] else 'no'} | "
            f"{'yes' if item['semantic_memory_correct'] else 'no'} | "
            f"{'yes' if item['transaction_rollback_correct'] else 'no'} | {failures} |"
        )
    totals = run["dimension_totals"]
    lines.extend([
        "",
        "## Dimension totals",
        "",
        *[f"- {key.replace('_', ' ')}: {value}" for key, value in totals.items()],
        "",
    ])
    return "\n".join(lines)


def write_run_outputs(
    run: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_target.write_text(markdown_summary(run), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definitions", default=str(DEFAULT_DEFINITION_PATH))
    parser.add_argument("--observations", help="Recorded/mock observation JSON; no provider call is made.")
    parser.add_argument("--provider", required=False)
    parser.add_argument("--model", required=False)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--case")
    scope.add_argument("--category")
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    parser.add_argument("--list", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    definition = load_benchmark(args.definitions)
    if args.list:
        for case in definition["cases"]:
            print(f"{case['case_id']}\t{case['category']}\t{case['title']}")
        return 0
    required = {
        "--observations": args.observations,
        "--provider": args.provider,
        "--model": args.model,
        "--json-out": args.json_out,
        "--markdown-out": args.markdown_out,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit("Missing required arguments: " + ", ".join(missing))
    runner = BenchmarkRunner(definition, ReplayExecutor.from_file(args.observations))
    run = runner.run(
        provider=args.provider,
        model=args.model,
        selection=BenchmarkSelection(args.case, args.category),
    )
    write_run_outputs(run, json_path=args.json_out, markdown_path=args.markdown_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
