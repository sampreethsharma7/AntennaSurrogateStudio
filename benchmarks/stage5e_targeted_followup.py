"""Recorded replay and failed-case-only live follow-up for Stage 5E.

The frozen Stage-5E definitions and prompts are read without modification.
Cloud execution is opt-in and limited to the provider-specific cases that
failed the original Stage-5E run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from benchmarks.stage5e_memory import (
    EXPECTED_TEST_SET_SHA256,
    PROVIDERS,
    evaluate_case,
    load_frozen_definition,
    provider_counts,
    run_live_case,
)
from studio.antenna_builder import ProjectMemory, apply_semantic_memory_proposals
from studio.antenna_llm_planner import SemanticMemoryProposal


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "benchmarks" / "results" / "stage5e_memory"
RESULT_ROOT = ROOT / "benchmarks" / "results" / "stage5e_memory_targeted_followup"
FAILED_CASES = {
    "gemini": ("P03", "F02", "U02", "M01"),
    "openrouter": ("P01", "P03", "F01", "F02", "U01", "U02", "M01"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _baseline_path(provider: str) -> Path:
    return BASELINE_ROOT / ("gemini_results.json" if provider == "gemini" else "nemotron_results.json")


def _source_hashes() -> dict[str, str]:
    paths = {
        "normalization_source_sha256": ROOT / "studio" / "project_memory_normalization.py",
        "builder_source_sha256": ROOT / "studio" / "antenna_builder.py",
        "planner_source_sha256": ROOT / "studio" / "antenna_llm_planner.py",
        "followup_harness_sha256": Path(__file__),
    }
    return {name: _sha(path) for name, path in paths.items()}


def _semantic_items(memory: ProjectMemory) -> list[dict[str, Any]]:
    return [
        item.to_dict()
        for collection in (memory.requirements, memory.decisions)
        for item in collection
        if item.source == "user_semantic"
    ]


def replay_provider_recorded_outputs(provider: str) -> dict[str, Any]:
    """Replay the original parsed proposals through the updated publication layer."""

    definition = load_frozen_definition()
    cases = {item["case_id"]: item for item in definition["cases"]}
    baseline = json.loads(_baseline_path(provider).read_text(encoding="utf-8"))
    results = []
    boolean_contract_violations = []
    for original in baseline["results"]:
        case = cases[original["case_id"]]
        memory = ProjectMemory.empty()
        replayed_turns = []
        for turn in original["turn_results"]:
            raw_proposals = turn.get("semantic_memory", {}).get("proposed", [])
            proposals = tuple(
                SemanticMemoryProposal.from_unvalidated(item) for item in raw_proposals
            )
            for proposal in proposals:
                if isinstance(proposal.value, bool):
                    boolean_contract_violations.append({
                        "case_id": original["case_id"],
                        "turn_index": turn["turn_index"],
                        "key": proposal.key,
                        "value": proposal.value,
                    })
            source_audit = (turn.get("attempts") or [{}])[-1].get("audit") or {}
            turn_id = source_audit.get("turn_id") or (
                f"{original['case_id']}-turn-{turn['turn_index']}-attempt-1"
            )
            memory, audit = apply_semantic_memory_proposals(
                memory,
                proposals,
                instruction=turn["instruction"],
                planner_calls=(),
                design=None,
                turn_id=turn_id,
            )
            replayed = dict(turn)
            replayed["semantic_memory"] = audit
            replayed["memory"] = memory.to_dict()
            replayed["planner_context"] = memory.to_planner_dict()
            replayed_turns.append(replayed)
        observation = dict(original)
        observation["turn_results"] = replayed_turns
        observation["final_memory"] = memory.to_dict()
        observation["planner_context"] = memory.to_planner_dict()
        observation["recorded_output_replay"] = True
        observation["evaluation"] = evaluate_case(case, observation)
        results.append(observation)
    return {
        "benchmark_id": "stage5e-targeted-followup-recorded-replay-v1",
        "test_set_sha256": EXPECTED_TEST_SET_SHA256,
        "provider": provider,
        "model": PROVIDERS[provider],
        "source_result_sha256": _sha(_baseline_path(provider)),
        "replayed_case_count": len(results),
        "replayed_turn_count": sum(len(item["turn_results"]) for item in results),
        "boolean_contract_violations": boolean_contract_violations,
        "results": results,
        "counts": provider_counts(results),
    }


def prepare(run_root: Path = RESULT_ROOT) -> dict[str, Any]:
    definition = load_frozen_definition()
    run_root.mkdir(parents=True, exist_ok=True)
    replay = {
        provider: replay_provider_recorded_outputs(provider)
        for provider in PROVIDERS
    }
    for provider, payload in replay.items():
        _write_json(run_root / f"recorded_replay_{provider}.json", payload)
    manifest = {
        "benchmark_id": "stage5e-targeted-followup-v1",
        "prepared_at_utc": _utc_now(),
        "status": "prepared",
        "test_set_sha256": EXPECTED_TEST_SET_SHA256,
        "frozen_case_count": definition["case_count"],
        "frozen_prompt_turn_count": definition["prompt_turn_count"],
        "targeted_failed_cases": {key: list(value) for key, value in FAILED_CASES.items()},
        "baseline_result_sha256": {
            provider: _sha(_baseline_path(provider)) for provider in PROVIDERS
        },
        **_source_hashes(),
    }
    _write_json(run_root / "run_manifest.json", manifest)
    return {"manifest": manifest, "replay": replay}


def _verify_freeze(run_root: Path) -> dict[str, Any]:
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("test_set_sha256") != EXPECTED_TEST_SET_SHA256:
        raise RuntimeError("Targeted Stage-5E follow-up test-set hash changed.")
    for name, actual in _source_hashes().items():
        if manifest.get(name) != actual:
            raise RuntimeError(f"Targeted Stage-5E follow-up freeze violation: {name}.")
    for provider in PROVIDERS:
        if manifest["baseline_result_sha256"].get(provider) != _sha(_baseline_path(provider)):
            raise RuntimeError(f"Original Stage-5E {provider} result changed after preparation.")
    return manifest


def run_targeted_provider(
    provider: str,
    *,
    authorized: bool,
    run_root: Path = RESULT_ROOT,
) -> dict[str, Any]:
    if not authorized:
        raise PermissionError("Targeted Stage-5E cloud follow-up requires explicit authorization.")
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    manifest = _verify_freeze(run_root)
    output_path = run_root / f"targeted_{provider}_results.json"
    if output_path.exists():
        raise RuntimeError(f"Targeted results already exist for {provider}.")
    case_index = {item["case_id"]: item for item in load_frozen_definition()["cases"]}
    selected = [case_index[case_id] for case_id in FAILED_CASES[provider]]
    live_root = run_root / "live"
    results = [
        run_live_case(
            case,
            provider=provider,
            model=PROVIDERS[provider],
            run_root=live_root,
            env_file=ROOT / ".env",
        )
        for case in selected
    ]
    payload = {
        "benchmark_id": "stage5e-targeted-followup-v1",
        "test_set_sha256": EXPECTED_TEST_SET_SHA256,
        "provider": provider,
        "model": PROVIDERS[provider],
        "completed_at_utc": _utc_now(),
        "selected_case_ids": list(FAILED_CASES[provider]),
        "results": results,
        "counts": provider_counts(results),
    }
    _write_json(output_path, payload)
    completed = list(manifest.get("providers_completed", []))
    if provider not in completed:
        completed.append(provider)
    manifest["providers_completed"] = completed
    manifest["status"] = "live_complete" if len(completed) == len(PROVIDERS) else "live_in_progress"
    _write_json(run_root / "run_manifest.json", manifest)
    return payload


def _failure_details(result: Mapping[str, Any]) -> dict[str, Any]:
    rejected = [
        rejected
        for turn in result.get("turn_results", [])
        for rejected in turn.get("semantic_memory", {}).get("rejected", [])
    ]
    terminal = [
        {
            "turn_index": turn.get("turn_index"),
            "status": turn.get("terminal_status"),
            "message": turn.get("terminal_message"),
        }
        for turn in result.get("turn_results", [])
        if turn.get("terminal_status") not in {"finished", "finish"}
    ]
    failed_dimensions = [
        name
        for name, value in result.get("evaluation", {}).get("dimensions", {}).items()
        if not value.get("passed")
    ]
    return {
        "case_id": result.get("case_id"),
        "provider_api_failure": not result.get("evaluable", False),
        "failed_dimensions": failed_dimensions,
        "rejected_memory_proposals": rejected,
        "nonfinish_terminal_turns": terminal,
        "classification": (
            "provider_api_failure"
            if not result.get("evaluable", False)
            else "model_proposal_or_recall_failure"
        ),
    }


def finalize(run_root: Path = RESULT_ROOT) -> dict[str, Any]:
    _verify_freeze(run_root)
    providers = {
        provider: json.loads(
            (run_root / f"targeted_{provider}_results.json").read_text(encoding="utf-8")
        )
        for provider in PROVIDERS
    }
    failures = {
        provider: [
            _failure_details(result)
            for result in payload["results"]
            if not result.get("evaluation", {}).get("passed", False)
        ]
        for provider, payload in providers.items()
    }
    comparison = {
        "benchmark_id": "stage5e-targeted-followup-v1",
        "test_set_sha256": EXPECTED_TEST_SET_SHA256,
        "providers": {
            provider: payload["counts"] for provider, payload in providers.items()
        },
        "remaining_failures": failures,
    }
    _write_json(run_root / "targeted_comparison.json", comparison)
    _write_json(run_root / "remaining_failure_inventory.json", failures)
    lines = [
        "# Stage 5E targeted follow-up",
        "",
        f"Frozen test-set SHA-256: `{EXPECTED_TEST_SET_SHA256}`",
        "",
        "The original 11 cases / 17 turns were replayed for each provider from recorded outputs before live calls.",
        "Only provider-specific cases that failed the original run were rerun live.",
        "",
        "| Provider | Targeted cases | Evaluable | Passed | Failed | Provider failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for provider, payload in providers.items():
        counts = payload["counts"]
        lines.append(
            f"| {provider} | {counts['total_cases']} | {counts['evaluable']} | "
            f"{counts['passed']} | {counts['failed']} | {counts['provider_failure_cases']} |"
        )
    lines.extend(["", "## Remaining failures", ""])
    for provider, items in failures.items():
        if not items:
            lines.append(f"- {provider}: none")
        for item in items:
            lines.append(
                f"- {provider} {item['case_id']}: {item['classification']}; "
                f"dimensions={item['failed_dimensions']}"
            )
    lines.append("")
    (run_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = _utc_now()
    _write_json(manifest_path, manifest)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run-provider", choices=tuple(PROVIDERS))
    actions.add_argument("--finalize", action="store_true")
    parser.add_argument("--authorized-cloud", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        result = prepare()
    elif args.run_provider:
        result = run_targeted_provider(
            args.run_provider,
            authorized=args.authorized_cloud,
        )
    else:
        result = finalize()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
