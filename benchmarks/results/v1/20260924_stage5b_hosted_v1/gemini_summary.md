# Antenna agent benchmark antenna-agent-unseen-v1

Provider/model: `gemini` / `gemini-3.8-flash`

No single quality score is calculated. Each objective dimension remains separate.

| Case | Category | Outcome | Completed | Canonical | Memory | Rollback | Failures |
|---|---|---|---|---:|---:|---:|---|
| A01 | A | finished | yes | no | yes | yes | planner_reasoning_failure |
| A02 | A | finished | yes | yes | yes | yes | — |
| A03 | A | finished | yes | no | yes | yes | planner_reasoning_failure |
| A04 | A | refuse | no | yes | yes | yes | — |
| B01 | B | finished | yes | yes | yes | yes | — |
| B02 | B | clarify | no | no | yes | yes | planner_reasoning_failure |
| B03 | B | finished | yes | no | yes | yes | planner_reasoning_failure |
| B04 | B | finished | yes | yes | yes | yes | — |
| C01 | C | finished | yes | no | yes | yes | planner_reasoning_failure |
| C02 | C | finished | yes | yes | yes | yes | — |
| C03 | C | finished | yes | no | yes | yes | planner_reasoning_failure |
| C04 | C | clarify | no | yes | yes | yes | — |
| D01 | D | finished | yes | yes | yes | yes | — |
| D02 | D | finished | yes | yes | yes | yes | — |
| D03 | D | finished | yes | yes | yes | yes | — |
| D04 | D | finished | yes | yes | no | yes | planner_reasoning_failure |
| E01 | E | finished | yes | yes | yes | yes | planner_reasoning_failure |
| E02 | E | finished | yes | yes | yes | yes | planner_reasoning_failure |
| E03 | E | finished | yes | yes | yes | yes | planner_reasoning_failure |
| F01 | F | finished | yes | yes | yes | yes | bad_tool_abstraction, planner_reasoning_failure |
| F02 | F | finished | yes | yes | yes | yes | bad_tool_abstraction, planner_reasoning_failure |
| F03 | F | finished | yes | yes | yes | yes | bad_tool_abstraction, planner_reasoning_failure |
| F04 | F | finished | yes | yes | yes | yes | bad_tool_abstraction, planner_reasoning_failure |
| G01 | G | finished | yes | no | no | yes | planner_reasoning_failure |
| G02 | G | finished | yes | yes | no | yes | planner_reasoning_failure |
| G03 | G | finished | yes | yes | no | yes | planner_reasoning_failure |
| H01 | H | refuse | no | yes | yes | yes | — |
| H02 | H | clarify | no | yes | yes | yes | — |
| H03 | H | clarify | no | yes | yes | yes | planner_reasoning_failure |
| H04 | H | clarify | no | yes | yes | yes | — |

## Dimension totals

- cases: 30
- completed yes: 23
- completed partial: 0
- canonical state correct: 23
- semantic memory correct: 26
- rollback correct: 30
- provider api failures: 0
- schema agent step failures: 0
- execution rejections: 0

## Live evaluation totals

- completed: 23
- partial: 0
- failed: 7
- unevaluable due to provider/API: 0
- canonical-state correct: 23 / 30
- semantic-memory correct: 26 / 30
- unsupported capability hallucinations: 0
- missed required clarifications: 0
- unnecessary clarifications: 2
- engineering-disposition failures: 3
- presentation omissions: 3
- unsupported RF claims: 0
- schema/structured-output failures: 0
- execution rejections: 0
- rollback failures: 0

## Failure taxonomy

- bad_tool_abstraction: 4
- planner_reasoning_failure: 18
