"""Frozen Stage-5E validation for semantic ProjectMemory normalization.

Importing and preparing this benchmark performs no network activity. Live
provider calls require the caller to pass ``authorized=True`` explicitly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from benchmarks.antenna_agent_live import (
    LIVE_BUDGETS,
    RETRY_POLICY,
    _planner_factory,
    _provider_or_schema_failure,
    _read_json_lines,
    build_fixture,
    canonical_facts,
    configured_credentials,
)
from benchmarks.contract_provenance import observation_contract_provenance
from studio.antenna_agent import PlannerClarificationRequired, PlannerRefusal
from studio.antenna_builder import ProjectMemory, apply_semantic_memory_proposals, execute_builder_turn
from studio.antenna_llm_planner import SemanticMemoryProposal
from studio.antenna_tools import CapabilityError


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "benchmarks" / "results" / "stage5e_memory"
DEFINITION_PATH = RESULT_ROOT / "frozen_test_definitions.json"
EXPECTED_TEST_SET_SHA256 = "5b396d8f1bf3ff66c7b0204f9cb19b977e625b452230a2d960c35de1472a1862"
RECORDED_STAGE5B_ROOT = (
    ROOT / "benchmarks" / "results" / "v1" / "20260924_stage5b_hosted_v1" / "gemini"
)
PROVIDERS = {
    "gemini": "gemini-3.8-flash",
    "openrouter": "nvidia/nemotron-3-ultra-550b-a55b:free",
}
REGISTERED_KEYS = {
    "target_polarization",
    "board_width_limit",
    "feed_network_future_goal",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_frozen_definition(path: Path = DEFINITION_PATH) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_TEST_SET_SHA256:
        raise RuntimeError(
            f"Stage-5E test-set SHA mismatch: expected {EXPECTED_TEST_SET_SHA256}, got {digest}."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("The Stage-5E frozen test definition is malformed.")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != payload.get("case_count"):
        raise RuntimeError("The Stage-5E frozen case count is inconsistent.")
    if sum(len(case.get("turns", [])) for case in cases) != payload.get("prompt_turn_count"):
        raise RuntimeError("The Stage-5E frozen prompt-turn count is inconsistent.")
    if len({case.get("case_id") for case in cases}) != len(cases):
        raise RuntimeError("The Stage-5E frozen case IDs must be unique.")
    if payload.get("providers") != PROVIDERS:
        raise RuntimeError("The Stage-5E provider/model set is inconsistent.")
    return payload


def _semantic_items(memory_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for collection in ("requirements", "decisions")
        for item in memory_payload.get(collection, [])
        if isinstance(item, Mapping) and item.get("source") == "user_semantic"
    ]


def _active_items(memory_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [item for item in _semantic_items(memory_payload) if item.get("status") == "active"]


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and float(actual) == float(expected)
    return actual == expected


def replay_recorded_stage5b() -> dict[str, Any]:
    """Replay only recorded semantic proposals; no planner/provider is invoked."""

    output: dict[str, Any] = {
        "source": "recorded Stage-5B Gemini trajectories",
        "cloud_calls": 0,
        "cases": {},
    }
    for case_id in ("G01", "G02", "G03"):
        memory = ProjectMemory.empty()
        turns = []
        path = RECORDED_STAGE5B_ROOT / case_id / "trajectory.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            raw_proposals = record.get("semantic_memory", {}).get("proposed", [])
            if not raw_proposals:
                continue
            proposals = tuple(
                SemanticMemoryProposal.from_unvalidated(item) for item in raw_proposals
            )
            memory, audit = apply_semantic_memory_proposals(
                memory,
                proposals,
                instruction=record["user_request"],
                planner_calls=(),
                design=None,
                turn_id=record["turn_id"],
            )
            turns.append({
                "turn_id": record["turn_id"],
                "instruction": record["user_request"],
                "proposed": raw_proposals,
                "normalization_audit": audit,
            })
        payload = memory.to_dict()
        output["cases"][case_id] = {
            "turns": turns,
            "semantic_items": _semantic_items(payload),
            "planner_context": memory.to_planner_dict().get("project_intent", []),
        }
    return output


def _dimension(passed: bool, details: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "details": details}


def _find_active(active: list[dict[str, Any]], expected: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in active
        if item.get("key") == expected.get("key")
        and _same_value(item.get("value"), expected.get("value"))
        and ("unit" not in expected or item.get("unit") == expected.get("unit"))
        and (
            "constraint_operator" not in expected
            or item.get("constraint_operator") == expected.get("constraint_operator")
        )
        and (
            "semantic_kind" not in expected
            or item.get("semantic_kind") in expected.get("semantic_kind", [])
        )
    ]


def evaluate_case(case: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    turns = observation.get("turn_results", [])
    final_memory = observation.get("final_memory", {})
    items = _semantic_items(final_memory)
    active = [item for item in items if item.get("status") == "active"]
    proposals_by_turn = [turn.get("semantic_memory", {}).get("proposed", []) for turn in turns]
    evidence_checks = [
        isinstance(proposal.get("evidence_quote"), str)
        and bool(proposal.get("evidence_quote"))
        and proposal["evidence_quote"] in turn.get("instruction", "")
        for turn, proposals in zip(turns, proposals_by_turn)
        for proposal in proposals
    ]

    policy = expected.get("proposal_policy", "required")
    if policy == "required_each_turn":
        proposal_ok = bool(turns) and all(bool(items_) for items_ in proposals_by_turn)
    elif policy == "semantic_turns_required":
        indexes = expected.get("semantic_turn_indexes", [])
        proposal_ok = all(
            1 <= index <= len(proposals_by_turn) and bool(proposals_by_turn[index - 1])
            for index in indexes
        )
    elif policy == "optional":
        proposal_ok = True
    else:
        proposal_ok = any(proposals_by_turn)

    required = expected.get("required_active", [])
    key_results = [
        {
            "expected": requirement.get("key"),
            "matches": [item for item in active if item.get("key") == requirement.get("key")],
        }
        for requirement in required
    ]
    value_results = [
        {
            "key": requirement.get("key"),
            "expected": requirement.get("value"),
            "matches": [
                item for item in active
                if item.get("key") == requirement.get("key")
                and _same_value(item.get("value"), requirement.get("value"))
            ],
        }
        for requirement in required
    ]
    kind_results = [
        {
            "key": requirement.get("key"),
            "expected": requirement.get("semantic_kind", []),
            "matches": [
                item for item in active
                if item.get("key") == requirement.get("key")
                and (
                    "semantic_kind" not in requirement
                    or item.get("semantic_kind") in requirement.get("semantic_kind", [])
                )
            ],
        }
        for requirement in required
    ]
    unit_operator_results = [
        {
            "key": requirement.get("key"),
            "expected_unit": requirement.get("unit"),
            "expected_constraint_operator": requirement.get("constraint_operator"),
            "matches": [
                item for item in active
                if item.get("key") == requirement.get("key")
                and ("unit" not in requirement or item.get("unit") == requirement.get("unit"))
                and (
                    "constraint_operator" not in requirement
                    or item.get("constraint_operator") == requirement.get("constraint_operator")
                )
            ],
        }
        for requirement in required
    ]
    key_ok = all(len(item["matches"]) == 1 for item in key_results)
    value_ok = all(len(item["matches"]) == 1 for item in value_results)
    kind_ok = all(len(item["matches"]) == 1 for item in kind_results)
    unit_operator_ok = all(len(item["matches"]) == 1 for item in unit_operator_results)
    forbidden_keys = set(expected.get("forbidden_active_keys", []))
    forbidden_values = set(expected.get("forbidden_active_values", []))
    erroneous = [
        item for item in active
        if item.get("key") in forbidden_keys or item.get("value") in forbidden_values
    ]

    counts: dict[str, int] = {}
    for item in active:
        key = str(item.get("key"))
        counts[key] = counts.get(key, 0) + 1
    duplicates = {key: count for key, count in counts.items() if count > 1}

    planner_intent = observation.get("planner_context", {}).get("project_intent", [])
    planner_counts: dict[str, int] = {}
    for item in planner_intent:
        key = str(item.get("key"))
        planner_counts[key] = planner_counts.get(key, 0) + 1
    planner_ok = not any(count > 1 for count in planner_counts.values()) and all(
        any(
            context.get("key") == requirement.get("key")
            and _same_value(context.get("value"), requirement.get("value"))
            for context in planner_intent
        )
        for requirement in expected.get("required_active", [])
    )

    history_expected = expected.get("history")
    history_details: dict[str, Any] = {"not_applicable": True}
    history_ok = True
    if isinstance(history_expected, Mapping):
        history_items = [item for item in items if item.get("key") == history_expected.get("key")]
        status_counts = {
            status: sum(item.get("status") == status for item in history_items)
            for status in ("active", "superseded", "resolved")
        }
        history_ok = (
            status_counts["superseded"] >= history_expected.get("minimum_superseded", 0)
            and status_counts["resolved"] >= history_expected.get("minimum_resolved", 0)
        )
        history_details = {"items": history_items, "status_counts": status_counts}

    turn_expectations = expected.get("turn_expectations", [])
    turn_expectation_results = []
    for index, turn_expected in enumerate(turn_expectations):
        snapshot = turns[index].get("memory", {}) if index < len(turns) else {}
        snapshot_active = _active_items(snapshot)
        if turn_expected.get("status") == "absent":
            passed = not any(item.get("key") == turn_expected.get("key") for item in snapshot_active)
        else:
            passed = len(_find_active(snapshot_active, turn_expected)) == 1
        turn_expectation_results.append({"turn_index": index + 1, "passed": passed, "expected": turn_expected})
    turn_lifecycle_ok = all(item["passed"] for item in turn_expectation_results)

    unknown_expected = expected.get("unknown_preservation")
    unknown_details: Any = {"not_applicable": True}
    unknown_ok = True
    if isinstance(unknown_expected, Mapping):
        unknowns = [item for item in active if item.get("key") not in REGISTERED_KEYS]
        allowed_kinds = set(unknown_expected.get("semantic_kind", []))
        matching = [item for item in unknowns if item.get("semantic_kind") in allowed_kinds]
        unknown_ok = bool(matching) if unknown_expected.get("required") else True
        unknown_details = {"matching": matching, "all_unknown_active": unknowns}

    provenance_missing = [
        item for item in items
        if item.get("proposed_key") is None
        or "proposed_value" not in item
        or not item.get("evidence_quote")
        or not item.get("normalization_rule")
        or not item.get("normalization_version")
    ]

    expected_excitation = expected.get("canonical_excitation")
    excitation_actual = observation.get("final_canonical_facts", {}).get("excitation_strategy")
    state_intent_ok = (
        expected_excitation is None or excitation_actual == expected_excitation
    ) and not any(item.get("value") in forbidden_values for item in active)

    recall_groups = expected.get("recall_token_groups", [])
    final_response = str(observation.get("final_user_facing_response", "")).casefold()
    recall_results = [
        {"tokens": group, "present": all(str(token).casefold() in final_response for token in group)}
        for group in recall_groups
    ]
    recall_ok = all(item["present"] for item in recall_results)

    dimensions = {
        "proposal_emitted": _dimension(proposal_ok, [len(items_) for items_ in proposals_by_turn]),
        "evidence_quote_valid": _dimension(all(evidence_checks), evidence_checks),
        "canonical_semantic_key": _dimension(key_ok, key_results),
        "canonical_value": _dimension(value_ok, value_results),
        "semantic_kind": _dimension(kind_ok, kind_results),
        "unit_constraint_operator": _dimension(unit_operator_ok, unit_operator_results),
        "supersede_resolve_correctness": _dimension(
            history_ok and turn_lifecycle_ok,
            {"history": history_details, "turns": turn_expectation_results},
        ),
        "unknown_concept_preservation": _dimension(unknown_ok, unknown_details),
        "normalization_correctness": _dimension(key_ok and value_ok, {
            "key_results": key_results,
            "value_results": value_results,
        }),
        "over_normalization": _dimension(not erroneous, erroneous),
        "duplicate_active_concepts": _dimension(not duplicates, duplicates),
        "planner_context_representation": _dimension(planner_ok, planner_intent),
        "provenance_preserved": _dimension(not provenance_missing, provenance_missing),
        "current_state_future_intent_separation": _dimension(
            state_intent_ok,
            {"expected_excitation": expected_excitation, "actual_excitation": excitation_actual},
        ),
        "final_conversational_recall": _dimension(recall_ok, recall_results),
    }
    passed = all(item["passed"] for item in dimensions.values())
    return {"passed": passed, "dimensions": dimensions}


def _provider_failure(audit: Mapping[str, Any] | None, error_text: str | None) -> bool:
    if audit and audit.get("terminal_status") != "provider_error":
        return False
    if audit:
        return _provider_or_schema_failure(audit) == "provider"
    return bool(error_text)


def run_live_case(
    case: Mapping[str, Any],
    *,
    provider: str,
    model: str,
    run_root: Path,
    env_file: Path,
) -> dict[str, Any]:
    case_root = run_root / provider / case["case_id"]
    case_root.mkdir(parents=True, exist_ok=True)
    audit_path = case_root / "trajectory.jsonl"
    session = build_fixture(case["initial_fixture"])
    initial_facts = canonical_facts(session.design)
    turn_results = []
    provider_failures = 0

    for turn_index, instruction in enumerate(case["turns"], start=1):
        attempts = []
        completed_session = None
        for attempt_index in range(1, RETRY_POLICY["maximum_attempts"] + 1):
            attempt_session = copy.deepcopy(session)
            before_count = len(_read_json_lines(audit_path))
            error_text = None
            with TemporaryDirectory(prefix=f"{case['case_id']}-", dir=case_root) as directory:
                project_path = Path(directory)
                try:
                    planner = _planner_factory(provider, model, env_file)
                    execute_builder_turn(
                        project_path,
                        attempt_session,
                        instruction,
                        planner=planner,
                        audit_log_path=audit_path,
                        turn_id=f"{case['case_id']}-turn-{turn_index}-attempt-{attempt_index}",
                        budgets=LIVE_BUDGETS,
                    )
                except (PlannerClarificationRequired, PlannerRefusal, CapabilityError) as exc:
                    error_text = str(exc)
                except Exception as exc:  # transport/provider errors must remain observable
                    error_text = f"{type(exc).__name__}: {exc}"
            records = _read_json_lines(audit_path)[before_count:]
            audit = records[-1] if records else None
            failure = _provider_failure(audit, error_text)
            attempts.append({
                "attempt_index": attempt_index,
                "provider_api_failure": failure,
                "error": error_text,
                "audit": audit,
            })
            if failure and attempt_index < RETRY_POLICY["maximum_attempts"]:
                time.sleep(float(RETRY_POLICY["delay_seconds"]))
                continue
            if failure:
                provider_failures += 1
            else:
                completed_session = attempt_session
            break

        if completed_session is not None:
            session = completed_session
        selected = attempts[-1]
        audit = selected.get("audit") or {}
        turn_results.append({
            "turn_index": turn_index,
            "instruction": instruction,
            "terminal_status": audit.get("terminal_status", "provider_error"),
            "terminal_message": audit.get("terminal_message") or selected.get("error") or "",
            "semantic_memory": audit.get("semantic_memory", {}),
            "memory": session.memory.to_dict(),
            "planner_context": session.memory.to_planner_dict(),
            "canonical_facts": canonical_facts(session.design),
            "attempts": attempts,
        })

    observation = {
        "case_id": case["case_id"],
        "category": case["category"],
        "provider": provider,
        "model": model,
        "evaluable": provider_failures == 0,
        "provider_api_failure_count": provider_failures,
        "initial_canonical_facts": initial_facts,
        "final_canonical_facts": canonical_facts(session.design),
        "final_memory": session.memory.to_dict(),
        "planner_context": session.memory.to_planner_dict(),
        "turn_results": turn_results,
        "final_user_facing_response": turn_results[-1]["terminal_message"] if turn_results else "",
        "contract_provenance": observation_contract_provenance(
            EXPECTED_TEST_SET_SHA256
        ),
    }
    observation["evaluation"] = (
        evaluate_case(case, observation)
        if observation["evaluable"]
        else {"passed": False, "dimensions": {}, "unevaluable_reason": "provider_api_failure"}
    )
    _write_json(case_root / "observation.json", observation)
    return observation


def run_live_provider(
    provider: str,
    *,
    authorized: bool,
    run_root: Path = RESULT_ROOT,
    env_file: Path | None = None,
) -> dict[str, Any]:
    """Run one provider only after the caller supplies explicit authorization."""

    if not authorized:
        raise PermissionError("Stage-5E cloud validation requires explicit user authorization.")
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported Stage-5E provider: {provider}")
    definition = load_frozen_definition()
    stem = "gemini" if provider == "gemini" else "nemotron"
    if (run_root / f"{stem}_results.json").exists():
        raise RuntimeError(f"Stage-5E results for {provider} already exist; the frozen run cannot be overwritten.")
    _start_or_verify_live_freeze(run_root, provider)
    env_path = env_file or (ROOT / ".env")
    results = [
        run_live_case(
            case,
            provider=provider,
            model=PROVIDERS[provider],
            run_root=run_root,
            env_file=env_path,
        )
        for case in definition["cases"]
    ]
    payload = {
        "benchmark_id": definition["benchmark_id"],
        "test_set_sha256": EXPECTED_TEST_SET_SHA256,
        "provider": provider,
        "model": PROVIDERS[provider],
        "completed_at_utc": _utc_now(),
        "results": results,
        "counts": provider_counts(results),
    }
    _write_json(run_root / f"{stem}_results.json", payload)
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = list(manifest.get("providers_completed", []))
    if provider not in completed:
        completed.append(provider)
    manifest["providers_completed"] = completed
    manifest["live_status"] = "captured" if len(completed) == len(PROVIDERS) else "in_progress"
    _write_json(manifest_path, manifest)
    return payload


def provider_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evaluable = [item for item in results if item.get("evaluable")]
    dimension_failures: dict[str, int] = {}
    for result in evaluable:
        for name in _dimension_failures(result):
            dimension_failures[name] = dimension_failures.get(name, 0) + 1
    return {
        "total_cases": len(results),
        "evaluable": len(evaluable),
        "passed": sum(bool(item.get("evaluation", {}).get("passed")) for item in evaluable),
        "failed": sum(not bool(item.get("evaluation", {}).get("passed")) for item in evaluable),
        "provider_failure_cases": len(results) - len(evaluable),
        "provider_failure_attempts": sum(int(item.get("provider_api_failure_count", 0)) for item in results),
        "dimension_failure_counts": dimension_failures,
    }


def _source_hashes() -> dict[str, str]:
    return {
        "normalization_source_sha256": hashlib.sha256(
            (ROOT / "studio" / "project_memory_normalization.py").read_bytes()
        ).hexdigest(),
        "builder_source_sha256": hashlib.sha256(
            (ROOT / "studio" / "antenna_builder.py").read_bytes()
        ).hexdigest(),
        "harness_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _start_or_verify_live_freeze(run_root: Path, provider: str) -> None:
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Prepare Stage 5E before starting live validation.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_set_sha256") != EXPECTED_TEST_SET_SHA256:
        raise RuntimeError("The prepared Stage-5E test set does not match the frozen definition.")
    for key, value in _source_hashes().items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Stage-5E freeze violation: {key} changed after preparation.")
    manifest.setdefault("live_started_at_utc", _utc_now())
    manifest["live_status"] = "in_progress"
    started = list(manifest.get("providers_started", []))
    if provider not in started:
        started.append(provider)
    manifest["providers_started"] = started
    _write_json(manifest_path, manifest)


def _dimension_failures(result: Mapping[str, Any]) -> list[str]:
    return [
        name for name, value in result.get("evaluation", {}).get("dimensions", {}).items()
        if not value.get("passed")
    ]


def finalize_outputs(run_root: Path = RESULT_ROOT) -> dict[str, Any]:
    gemini = json.loads((run_root / "gemini_results.json").read_text(encoding="utf-8"))
    nemotron = json.loads((run_root / "nemotron_results.json").read_text(encoding="utf-8"))
    providers = {"gemini": gemini, "nemotron": nemotron}
    all_evaluable = [
        result
        for payload in providers.values()
        for result in payload["results"]
        if result["evaluable"]
    ]
    has_complete_provider = any(
        payload["counts"]["evaluable"] == payload["counts"]["total_cases"]
        and payload["counts"]["passed"] == payload["counts"]["total_cases"]
        for payload in providers.values()
    )
    if any(not result["evaluation"]["passed"] for result in all_evaluable):
        validation_status = "not_validated"
    elif has_complete_provider:
        validation_status = "validated"
    else:
        validation_status = "inconclusive_provider_coverage"
    inventory = {
        "entries": [
            {
                "provider": name,
                "case_id": result["case_id"],
                "provider_api_failure": not result["evaluable"],
                "failed_dimensions": _dimension_failures(result),
            }
            for name, payload in providers.items()
            for result in payload["results"]
            if not result["evaluable"] or not result["evaluation"]["passed"]
        ]
    }
    _write_json(run_root / "failure_inventory.json", inventory)

    gemini_index = {item["case_id"]: item for item in gemini["results"]}
    nemotron_index = {item["case_id"]: item for item in nemotron["results"]}
    comparisons = []
    for case_id in sorted(set(gemini_index) & set(nemotron_index)):
        left, right = gemini_index[case_id], nemotron_index[case_id]
        comparisons.append({
            "case_id": case_id,
            "paired_evaluable": left["evaluable"] and right["evaluable"],
            "gemini_passed": left["evaluation"]["passed"] if left["evaluable"] else None,
            "nemotron_passed": right["evaluation"]["passed"] if right["evaluable"] else None,
            "gemini_failed_dimensions": _dimension_failures(left),
            "nemotron_failed_dimensions": _dimension_failures(right),
        })
    comparison = {
        "test_set_sha256": EXPECTED_TEST_SET_SHA256,
        "validation_status": validation_status,
        "cases": comparisons,
    }
    _write_json(run_root / "cross_provider_comparison.json", comparison)
    comparison_lines = [
        "# Stage 5E cross-provider comparison",
        "",
        "| Case | Paired evaluable | Gemini | Nemotron |",
        "|---|---:|---:|---:|",
    ]
    for item in comparisons:
        comparison_lines.append(
            f"| {item['case_id']} | {item['paired_evaluable']} | "
            f"{item['gemini_passed']} | {item['nemotron_passed']} |"
        )
    comparison_lines.append("")
    (run_root / "cross_provider_comparison.md").write_text(
        "\n".join(comparison_lines), encoding="utf-8"
    )

    lines = [
        "# Stage 5E semantic-memory validation",
        "",
        f"Frozen test-set SHA-256: `{EXPECTED_TEST_SET_SHA256}`",
        "",
        f"Validation status: **{validation_status}**",
        "",
        "| Provider | Evaluable | Passed | Failed | Provider-failure cases |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, payload in providers.items():
        counts = payload["counts"]
        lines.append(
            f"| {name} | {counts['evaluable']} | {counts['passed']} | "
            f"{counts['failed']} | {counts['provider_failure_cases']} |"
        )
    lines.extend(["", "## Objective dimension failures", ""])
    for name, payload in providers.items():
        failures = payload["counts"].get("dimension_failure_counts", {})
        lines.append(f"- {name}: {failures or 'none'}")
    lines.extend(["", "## Failure inventory", ""])
    if inventory["entries"]:
        for item in inventory["entries"]:
            reason = "provider/API" if item["provider_api_failure"] else ", ".join(item["failed_dimensions"])
            lines.append(f"- {item['provider']} {item['case_id']}: {reason}")
    else:
        lines.append("- No failures.")
    lines.append("")
    (run_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["live_status"] = "complete"
    manifest["live_completed_at_utc"] = _utc_now()
    _write_json(manifest_path, manifest)
    return {
        "validation_status": validation_status,
        "providers": {name: payload["counts"] for name, payload in providers.items()},
        "failures": inventory,
    }


def prepare_outputs(run_root: Path = RESULT_ROOT) -> dict[str, Any]:
    definition = load_frozen_definition()
    replay = replay_recorded_stage5b()
    _write_json(run_root / "deterministic_g01_g02_g03_replay.json", replay)
    (run_root / "test_set_sha256.txt").write_text(EXPECTED_TEST_SET_SHA256 + "\n", encoding="utf-8")
    env_file = ROOT / ".env"
    manifest = {
        "benchmark_id": definition["benchmark_id"],
        "test_set_sha256": EXPECTED_TEST_SET_SHA256,
        "case_count": definition["case_count"],
        "prompt_turn_count": definition["prompt_turn_count"],
        "providers": definition["providers"],
        "configured_credentials_present": configured_credentials(env_file),
        **_source_hashes(),
        "prepared_at_utc": _utc_now(),
        "live_status": "awaiting_explicit_cloud_authorization",
        "cloud_calls_during_prepare": 0,
    }
    _write_json(run_root / "run_manifest.json", manifest)
    (run_root / "summary.md").write_text(
        "# Stage 5E semantic-memory validation\n\n"
        f"Frozen test-set SHA-256: `{EXPECTED_TEST_SET_SHA256}`\n\n"
        "Deterministic replay is complete. Live Gemini and Nemotron validation is awaiting explicit cloud authorization.\n",
        encoding="utf-8",
    )
    return {"manifest": manifest, "deterministic_replay": replay}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Prepare frozen outputs without network calls.")
    parser.add_argument("--run-provider", choices=tuple(PROVIDERS))
    parser.add_argument(
        "--authorized-cloud",
        action="store_true",
        help="Required acknowledgment that the user explicitly authorized Stage-5E cloud calls.",
    )
    parser.add_argument("--finalize", action="store_true", help="Compare already captured provider results.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.run_provider:
        run_live_provider(args.run_provider, authorized=args.authorized_cloud)
    elif args.finalize:
        finalize_outputs()
    else:
        prepare_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
