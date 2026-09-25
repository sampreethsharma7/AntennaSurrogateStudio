> **Superseded research evidence.** These observations were recorded under AgentStep contract v1. Contract v2 removed Boolean semantic-memory placeholders from the output schema, strict parser, and shared planner instruction. The provider comparison has not been rerun under v2 and must not be presented as current evidence.

# Stage 5E semantic-memory validation

Frozen test-set SHA-256: `5b396d8f1bf3ff66c7b0204f9cb19b977e625b452230a2d960c35de1472a1862`

Validation status: **not validated**

| Provider | Evaluable | Passed | Failed | Provider/API failures |
|---|---:|---:|---:|---:|
| Gemini `gemini-3.8-flash` | 11 | 7 | 4 | 0 |
| OpenRouter Nemotron `nvidia/nemotron-3-ultra-550b-a55b:free` | 11 | 4 | 7 | 0 |

## Deterministic replay

- G01: `target_polarization = circular_polarization`.
- G02: `board_width_limit = 90 mm`, `constraint_operator = max`; 72 mm is retained as superseded.
- G03: `feed_network_future_goal = corporate_feed`.
- Original proposed keys, values, exact evidence quotes, source turns, and normalization metadata were preserved.

## Objective results

Both providers passed all cases for:

- exact evidence-quote validity;
- supersede/resolve lifecycle behavior;
- protection against over-normalization;
- duplicate-active-concept prevention;
- provenance persistence;
- separation of current canonical state from future intent;
- final conversational recall.

The anti-merge cases passed for both providers. Substrate width was not converted into a board-width maximum, and circular slots were not converted into polarization intent.

## Normalization coverage misses

Gemini produced accepted, grounded representations that remained noncanonical:

- P03: `polarization = LHCP` was not mapped to `target_polarization = lhcp`.
- F02: `series_fed_network = explore later` was not mapped to `feed_network_future_goal = series_feed`.
- M01: `max_board_dimension_mm = 85 mm` and `corporate_feed = later` were retained as generic identities.

Nemotron's M01 also retained `max_board_dimension_mm = 85 mm` as the generic `max_board_dimension` identity.

## Model proposal failures before normalization

- Gemini U02 paraphrased the value as `manufacturing simplicity prioritized over visual appearance`; lexical grounding correctly rejected it.
- Nemotron used Boolean `true` values for P01, P03, F01, F02, U01, and U02. The same validation correctly rejected them.
- Nemotron M01 used `corporate_feed_future = true`; it was rejected before normalization.

These are proposal-contract/grounding failures, not evidence that the normalizer merged concepts incorrectly.

## Unknown concepts

- Gemini preserved U01 under a generic identity and failed U02 at grounding validation.
- Nemotron failed both unknown-concept cases at grounding validation.
- No unknown concept was forced into polarization, board constraint, or feed-network identities.

## Lifecycle and state separation

Both providers correctly completed the 75 mm to 95 mm board-limit supersession and subsequent resolution. Both kept current independent excitation separate from future feed intent. Both produced accurate human-readable recall in M01, although their structured memory identities were incomplete.

## Conclusion

Stage 5D is **not fully validated**. It fixed G01-G03 and showed no over-normalization, lifecycle, duplication, provenance, or state-separation regression. Live testing found additional accepted semantic forms outside the small registry, and it also exposed provider proposal-grounding failures that occur before normalization.
