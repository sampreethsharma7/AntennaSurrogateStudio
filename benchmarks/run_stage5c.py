"""Generate Stage-5C adjudication artifacts without calling any provider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.antenna_agent_adjudication import adjudicate_run, remaining_failures, summary_markdown


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("benchmarks/results/v1/20260924_stage5b_hosted_v1"),
    )
    parser.add_argument(
        "--definition",
        type=Path,
        default=Path("benchmarks/antenna_agent_v1.json"),
    )
    args = parser.parse_args(argv)
    payload = adjudicate_run(args.run_root, args.definition)
    target = args.run_root / "adjudicated"
    target.mkdir(parents=True, exist_ok=True)
    adjudication = {key: value for key, value in payload.items() if key != "corrected_runs"}
    _write_json(target / "adjudication.json", adjudication)
    _write_json(target / "corrected_objective_results.json", {
        "schema_version": 1,
        "evaluator_version": payload["evaluator_version"],
        "providers": payload["corrected_runs"],
        "paired_evaluable_comparison": payload["paired_evaluable_comparison"],
    })
    _write_json(target / "remaining_agent_failures.json", remaining_failures(payload))
    (target / "adjudication_summary.md").write_text(summary_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
