# Antenna engineering agent benchmark v1

`antenna-agent-unseen-v1` is a frozen, provider-neutral evaluation set. It contains 30 cases across design creation, free-form geometry, editing, arrays and excitation, engineering checks, RF analysis, project memory, and transaction safety.

The benchmark does not prescribe preferred tool sequences. Cases declare observable acceptance criteria: terminal outcomes, completion, canonical facts, mutation and rollback rules, capabilities whose selection is itself under test, engineering disposition, analytical grounding, and semantic-memory state.

The runner consumes normalized observations through an injected executor. The bundled CLI uses recorded or mocked observations and therefore never calls a model provider:

```powershell
python -m benchmarks.antenna_agent_benchmark --list

python -m benchmarks.antenna_agent_benchmark `
  --observations path\to\recorded_observations.json `
  --provider local_ollama `
  --model qwen3:8b `
  --case F03 `
  --json-out .test_runs\benchmark\F03.json `
  --markdown-out .test_runs\benchmark\F03.md
```

Replace `--case F03` with `--category F`, or omit both options for the full suite. A future live adapter can implement `BenchmarkExecutor.execute()` without changing the frozen cases or evaluator.

Recorded observations contain the provider/model, terminal result, normalized canonical facts, selected and hallucinated capabilities, engineering disposition evidence, analysis tools and unsupported claims, memory items, trajectory counters, and reproducibility hashes. Initial and final semantic design hashes determine mutation and rollback. ProjectMemory hashes are recorded separately; a clarification or refusal may legitimately retain noncanonical context while canonical state rolls back.

The result JSON keeps objective dimensions separate. It does not compute an AI-quality score. The Markdown output is a compact table over the same dimensions.

Failure classifications are frozen to:

1. `missing_capability`
2. `bad_tool_abstraction`
3. `insufficient_observation_context`
4. `planner_reasoning_failure`
5. `schema_structured_output_failure`
6. `provider_api_failure`
7. `transaction_runtime_failure`
8. `ambiguous_benchmark_request`
9. `deterministic_subsystem_failure`

Provider/API failure is recorded independently from model or agent failure. Engineering-disposition coverage and warning mentions in final prose are also separate. A missing prose warning is retained as `presentation_warning_omissions` and does not invalidate a correct structured disposition.

Example frozen case:

```json
{
  "case_id": "F04",
  "category": "F",
  "initial_fixture": "dipole_clean",
  "turns": [
    "Compare the modeled dipole length with a free-space half-wave baseline, including the analytical limitations."
  ],
  "acceptance": {
    "allowed_terminal_outcomes": ["finished"],
    "completion": "yes",
    "mutation": "forbidden",
    "rollback_required": false,
    "clarification": "forbidden",
    "canonical_facts": [],
    "required_capabilities": ["analysis.dipole_electrical_length"],
    "forbidden_capabilities": [],
    "engineering": {},
    "analysis": {
      "required_tools": ["analysis.dipole_electrical_length"],
      "forbid_proven_resonance_claim": true
    },
    "memory": {"required_items": []}
  }
}
```

The evaluator expects any claim beyond the structured analysis to be recorded in `unsupported_model_claims`. In particular, analytical baselines do not prove resonance or simulated RF performance, and spacing ratios alone do not prove an unconditional grating-lobe conclusion.
