# Cross-provider comparison

Only cases with evaluable results from both providers are compared.

| Case | Gemini evidence | Nemotron evidence | Differing dimensions |
|---|---|---|---|
| B02 | outcome=clarify, completed=no, canonical=False | outcome=finished, completed=yes, canonical=False | terminal_outcome, task_completed |
| C01 | outcome=finished, completed=yes, canonical=False | outcome=provider_error, completed=no, canonical=False | terminal_outcome, task_completed |
| C02 | outcome=finished, completed=yes, canonical=True | outcome=agent_limit, completed=no, canonical=False | terminal_outcome, task_completed, canonical_state_correct |

Paired evaluable cases: 20
Cases with observable disagreement: 3
