# Stage 5E targeted follow-up

Frozen test-set SHA-256: `5b396d8f1bf3ff66c7b0204f9cb19b977e625b452230a2d960c35de1472a1862`

The frozen prompts were unchanged. Recorded outputs for all 11 cases / 17 turns per provider were replayed before any live calls.

## Recorded-output replay

- Gemini: 10/11 passed; only U02 remained failed.
- Nemotron: 4/11 passed; seven recorded Boolean placeholders remained explicit contract failures.

## Failed-case-only live rerun

| Provider | Cases rerun | Evaluable | Passed | Failed | Provider failures |
|---|---:|---:|---:|---:|---:|
| Gemini | 4 | 4 | 2 | 2 | 0 |
| OpenRouter Nemotron | 7 | 7 | 2 | 5 | 0 |

No live proposal used a Boolean semantic value. No over-normalization or provider/API failure was observed.

## Remaining failures

- Gemini F02: new grounded `feed_network` key, outside the approved `series_fed_network` alias.
- Gemini M01: a shortened evidence quote prevented the guarded board alias; a new generic `feed_network` key was outside the approved corporate alias.
- Nemotron P03 and U01: clarification with no memory proposal.
- Nemotron F02 and U02: ungrounded paraphrased values.
- Nemotron M01: new board alias, ungrounded corporate-feed proposal, and resulting recall omission.

No additional normalization aliases were added for these targeted-rerun forms.
