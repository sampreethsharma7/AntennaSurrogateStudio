# Antenna agent benchmark antenna-agent-unseen-v1

Provider/model: `openrouter` / `nvidia/nemotron-3-ultra-550b-a55b:free`

No single quality score is calculated. Each objective dimension remains separate.

| Case | Category | Outcome | Completed | Canonical | Memory | Rollback | Failures |
|---|---|---|---|---:|---:|---:|---|
| A01 | A | finished | yes | no | yes | yes | planner_reasoning_failure |
| A02 | A | finished | yes | yes | yes | yes | — |
| A03 | A | finished | yes | no | yes | yes | planner_reasoning_failure |
| A04 | A | refuse | no | yes | yes | yes | — |
| B01 | B | finished | yes | yes | yes | yes | — |
| B02 | B | finished | yes | no | yes | yes | planner_reasoning_failure |
| B03 | B | finished | yes | no | yes | yes | planner_reasoning_failure |
| B04 | B | finished | yes | yes | yes | yes | — |
| C01 | C | provider_error | no | no | yes | yes | schema_structured_output_failure, planner_reasoning_failure |
| C02 | C | agent_limit | no | no | yes | yes | schema_structured_output_failure, planner_reasoning_failure |
| C03 | C | finished | yes | no | yes | yes | schema_structured_output_failure, planner_reasoning_failure |
| C04 | C | clarify | no | yes | yes | yes | — |
| D01 | D | finished | yes | yes | yes | yes | — |
| D02 | D | finished | yes | yes | yes | yes | — |
| D03 | D | finished | yes | yes | yes | yes | schema_structured_output_failure |
| D04 | D | finished | yes | yes | no | yes | planner_reasoning_failure |
| E01 | E | finished | yes | yes | yes | yes | planner_reasoning_failure |
| E02 | E | finished | yes | yes | yes | yes | planner_reasoning_failure |
| E03 | E | finished | yes | yes | yes | yes | planner_reasoning_failure |
| F01 | F | finished | yes | yes | yes | yes | schema_structured_output_failure, bad_tool_abstraction, planner_reasoning_failure |
| F02 | F | provider_error | no | yes | yes | yes | provider_api_failure |
| F03 | F | provider_error | no | yes | yes | yes | provider_api_failure |
| F04 | F | provider_error | no | yes | yes | yes | provider_api_failure |
| G01 | G | provider_error | no | no | no | yes | provider_api_failure |
| G02 | G | provider_error | no | yes | no | yes | provider_api_failure |
| G03 | G | provider_error | no | no | no | yes | provider_api_failure |
| H01 | H | provider_error | no | yes | yes | yes | provider_api_failure |
| H02 | H | provider_error | no | yes | yes | yes | provider_api_failure |
| H03 | H | provider_error | no | yes | yes | yes | provider_api_failure |
| H04 | H | provider_error | no | yes | yes | yes | provider_api_failure |

## Dimension totals

- cases: 30
- completed yes: 16
- completed partial: 0
- canonical state correct: 21
- semantic memory correct: 26
- rollback correct: 30
- provider api failures: 10
- schema agent step failures: 5
- execution rejections: 0

## Live evaluation totals

- completed: 16
- partial: 0
- failed: 4
- unevaluable due to provider/API: 10
- canonical-state correct: 13 / 20
- semantic-memory correct: 19 / 20
- unsupported capability hallucinations: 0
- missed required clarifications: 0
- unnecessary clarifications: 0
- engineering-disposition failures: 3
- presentation omissions: 8
- unsupported RF claims: 0
- schema/structured-output failures: 5
- execution rejections: 0
- rollback failures: 0

## Failure taxonomy

- bad_tool_abstraction: 1
- planner_reasoning_failure: 12
- provider_api_failure: 10
- schema_structured_output_failure: 5
