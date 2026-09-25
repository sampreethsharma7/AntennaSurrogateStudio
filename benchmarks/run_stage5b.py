"""Execute the frozen Stage-5B benchmark against both authorized hosted planners.

This command performs live cloud calls. Do not invoke it before explicit user
authorization for the Stage-5B run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.antenna_agent_benchmark import BenchmarkRunner, ReplayExecutor, load_benchmark
from benchmarks.antenna_agent_live import (
    PROVIDERS,
    LiveBenchmarkExecutor,
    _augment_failure_taxonomy,
    build_fixture,
    canonical_facts,
    configured_credentials,
    cross_provider_markdown,
    dirty_state_fingerprint,
    failure_inventory,
    new_manifest,
    provider_summary,
    utc_now,
    verify_frozen_benchmark,
    write_complete_outputs,
)
from studio.antenna_analysis import semantic_design_hash, stable_hash
from studio.antenna_engineering_checks import run_engineering_checks
from studio.antenna_engineering_summary import engineering_report_hash


def _unexpected_observation(case, provider: str, model: str, exc: Exception):
    session = build_fixture(case["initial_fixture"])
    report = run_engineering_checks(session.design) if session.design is not None else None
    design_hash = semantic_design_hash(session.design)
    memory_hash = stable_hash(session.memory.to_dict())
    return {
        "case_id": case["case_id"], "provider": provider, "model": model,
        "initial_semantic_hash": design_hash, "final_semantic_hash": design_hash,
        "initial_memory_hash": memory_hash, "final_memory_hash": memory_hash,
        "engineering_report_hashes": [engineering_report_hash(report)] if report else [],
        "trajectory_reference": None,
        "terminal_result": {
            "outcome": "invalid_plan",
            "message": f"Unexpected deterministic benchmark execution failure: {type(exc).__name__}",
        },
        "terminal_outcome": "invalid_plan", "task_completed": "no",
        "canonical_facts": canonical_facts(session.design),
        "selected_capabilities": [], "hallucinated_capabilities": [],
        "engineering_warning_ids": [], "acknowledged_warning_ids": [],
        "deferred_warning_ids": [], "warnings_mentioned_in_prose": [],
        "analysis_tools": [], "analysis_results": [], "unsupported_model_claims": [],
        "memory_items": [], "schema_agent_step_failures": 0,
        "provider_api_failure": False, "execution_rejections": 0,
        "decision_iterations": 0, "design_action_batches": 0, "analysis_batches": 0,
        "failure_signals": ["deterministic_subsystem_failure"],
        "attempts": [], "trajectory": [], "tool_calls": [], "rejected_calls": [],
        "final_user_facing_response": "",
    }


def _run_provider(definition, executor, provider: str, model: str, run_root: Path):
    observations = []
    checkpoint = run_root / f"{provider}_observations.json"
    for case in definition["cases"]:
        try:
            observation = executor.execute(case, provider=provider, model=model)
        except Exception as exc:
            observation = _unexpected_observation(case, provider, model, exc)
        observations.append(observation)
        checkpoint.write_text(
            json.dumps({"observations": observations}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    run = BenchmarkRunner(
        definition,
        ReplayExecutor({item["case_id"]: item for item in observations}),
    ).run(provider=provider, model=model)
    _augment_failure_taxonomy(run)
    return run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    worktree = Path(__file__).resolve().parents[1]
    env_file = worktree / ".env"
    verify_frozen_benchmark()
    credentials = configured_credentials(env_file)
    missing = [provider for provider, present in credentials.items() if not present]
    if missing:
        raise SystemExit("Missing configured credential for: " + ", ".join(missing))

    run_root = worktree / "benchmarks" / "results" / "v1" / args.run_id
    if run_root.exists():
        raise SystemExit(f"Run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    manifest = new_manifest(worktree, args.run_id)
    manifest["started_at_utc"] = utc_now()
    manifest["status"] = "running_frozen_live_benchmark"
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    frozen_fingerprint = manifest["dirty_state_fingerprint"]

    definition = load_benchmark()
    executor = LiveBenchmarkExecutor(run_root, env_file)
    try:
        gemini = _run_provider(definition, executor, "gemini", PROVIDERS["gemini"], run_root)
        nemotron = _run_provider(
            definition, executor, "openrouter", PROVIDERS["openrouter"], run_root
        )
        manifest["ended_at_utc"] = utc_now()
        manifest["code_fingerprint_after"] = dirty_state_fingerprint(worktree)
        manifest["production_and_benchmark_code_unchanged_during_run"] = (
            manifest["code_fingerprint_after"] == frozen_fingerprint
        )
        manifest["status"] = "complete"
        write_complete_outputs(
            run_root, gemini=gemini, nemotron=nemotron, manifest=manifest
        )
    except BaseException:
        manifest["ended_at_utc"] = utc_now()
        manifest["code_fingerprint_after"] = dirty_state_fingerprint(worktree)
        manifest["status"] = "aborted"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
