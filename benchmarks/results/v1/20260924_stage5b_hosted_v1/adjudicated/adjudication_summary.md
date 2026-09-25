# Stage 5C adjudication summary

Source benchmark SHA-256: `281efa92743b03c69f2a16ad3ee8d6d01dafd339541e43cbcd0ac675ee86b868`
Evaluator: `antenna-agent-objective-v1` → `antenna-agent-objective-v2`

## Counts

- Original machine passes: Gemini 11/30; OpenRouter 9/30.
- Original evaluable passes: Gemini 11/30; OpenRouter 8/20.
- Corrected objective passes: Gemini 24/28 evaluable; OpenRouter 17/19 evaluable.
- Benchmark/evaluator/fixture mismatches: 25 provider-case adjudications.
- Genuine production-agent failures: 6 provider-case adjudications.
- Genuine semantic-memory failures: 3.
- Genuine schema incidents: 5 cases.
- Terminally consequential schema failure: OpenRouter C01; C02 also had a schema repair but its primary failure was cumulative replanning.
- Confirmed provider/API failures: 10.
- Deterministic subsystem failures: 0.
- Transaction failures: 0.

## Evaluator-v2 changes

- `canonical_family_identity` (A03, G01): Compare exact recipe-backed family aliases through rectangular_inset_patch. Production recipe identity is rectangular_inset_patch; inset_patch is a human alias.
- `canonical_material_identity` (A01): Compare the frozen Rogers 4003C spelling through the RO4003C registry identity. Production stores the material registry's canonical Rogers RO4003C name.
- `physical_composition_count` (B03): Count Boolean-referenced physical cutter tools, including multiple tools in one operation. One boolean.subtract call can validly consume multiple independent cutters.
- `composition_parameter_resolution` (C01, C03): Resolve base parameters plus persisted parameter.create calls before evaluating geometry expressions. Evaluator v1 passed AntennaDesign where a parameter mapping was required and omitted composed parameters.
- `engineering_group_disposition` (E01, E03): Expand acknowledged/deferred group IDs to their recorded constituent observation IDs. Production explicitly accepts grouped dispositions.
- `registered_analysis_identity` (F01, F02, F03, F04): Map frozen analysis.* labels to registered engineering.* tool identities. The installed manifest exposes engineering.* names.
- `clarification_policy_consistency` (H04): A case declaring clarification allowed also accepts a clarify terminal outcome. The prior acceptance fields contradicted each other for an under-specified invalid target.

## Adjudicated original failures

| Provider | Case | Adjudication | Corrected result |
|---|---|---|---|
| gemini | A01 | benchmark_contract_mismatch | pass |
| gemini | A03 | benchmark_contract_mismatch | pass |
| gemini | B02 | confirmed_model_reasoning_failure | fail |
| gemini | B03 | evaluator_extraction_bug | pass |
| gemini | C01 | evaluator_extraction_bug | pass |
| gemini | C03 | evaluator_extraction_bug | pass |
| gemini | D04 | benchmark_contract_mismatch | pass |
| gemini | E01 | evaluator_extraction_bug | pass |
| gemini | E02 | fixture_invalid | unevaluable |
| gemini | E03 | evaluator_extraction_bug | pass |
| gemini | F01 | benchmark_contract_mismatch | pass |
| gemini | F02 | benchmark_contract_mismatch | pass |
| gemini | F03 | benchmark_contract_mismatch | pass |
| gemini | F04 | benchmark_contract_mismatch | pass |
| gemini | G01 | confirmed_agent_failure | fail |
| gemini | G02 | confirmed_agent_failure | fail |
| gemini | G03 | confirmed_agent_failure | fail |
| gemini | H03 | benchmark_contract_mismatch | unevaluable |
| gemini | H04 | benchmark_contract_mismatch | pass |
| openrouter | A01 | benchmark_contract_mismatch | pass |
| openrouter | A03 | benchmark_contract_mismatch | pass |
| openrouter | B02 | benchmark_contract_mismatch | pass |
| openrouter | B03 | evaluator_extraction_bug | pass |
| openrouter | C01 | confirmed_schema_failure | fail |
| openrouter | C02 | confirmed_model_reasoning_failure | fail |
| openrouter | C03 | evaluator_extraction_bug | pass |
| openrouter | D04 | benchmark_contract_mismatch | pass |
| openrouter | E01 | evaluator_extraction_bug | pass |
| openrouter | E02 | fixture_invalid | unevaluable |
| openrouter | E03 | evaluator_extraction_bug | pass |
| openrouter | F01 | benchmark_contract_mismatch | pass |
| openrouter | F02 | confirmed_provider_failure | unevaluable |
| openrouter | F03 | confirmed_provider_failure | unevaluable |
| openrouter | F04 | confirmed_provider_failure | unevaluable |
| openrouter | G01 | confirmed_provider_failure | unevaluable |
| openrouter | G02 | confirmed_provider_failure | unevaluable |
| openrouter | G03 | confirmed_provider_failure | unevaluable |
| openrouter | H01 | confirmed_provider_failure | unevaluable |
| openrouter | H02 | confirmed_provider_failure | unevaluable |
| openrouter | H03 | confirmed_provider_failure | unevaluable |
| openrouter | H04 | confirmed_provider_failure | unevaluable |

## Recurring architectural patterns

1. Semantic-memory keys and values lack a stable domain ontology (G01-G03; three confirmed failures).
2. Nemotron repeated a successful relative edit as cumulative work until the action budget was exhausted (C02).
3. Hosted structured-output reliability was uneven: five Nemotron schema incidents, two terminally consequential (C01, C02).
4. The free OpenRouter endpoint exhausted its daily quota, leaving ten cases unevaluable.

D04 is not counted as a genuine memory failure: its conditional fallback was satisfied, and the frozen future-goal requirement was stronger than the user's statement.

## Semantic-memory adjudication

- D04/Gemini: proposed and applied `feed_network_preference=series-fed network` as a `preference`; the frozen future-goal requirement over-stated the conditional request.
- D04/OpenRouter: proposed `prefer_series_feed_if_available=true`; the reducer rejected the Boolean as not lexically grounded. This does not become a failure because the prompt supplied an explicit fallback rather than an unambiguous future commitment.
- G01/Gemini: retained `target_polarization=circular` as `future_intent`; genuine unstable value normalization against `circular_polarization`.
- G02/Gemini: correctly superseded 72 mm with 90 mm under `max_board_width_mm`; genuine unstable key normalization against `board_width_limit`.
- G03/Gemini: retained the corporate-network goal under `corporate_distribution_network=corporate distribution network`; genuine unstable key/value normalization against `feed_network_future_goal=corporate_feed`.
- G01-G03/OpenRouter: unevaluable because the provider returned HTTP 429 before semantic proposals were produced.

## Cases still requiring production investigation

- Gemini B02: unnecessary clarification for the intended upper-right attachment.
- Gemini G01-G03: stable semantic-memory key/value normalization.
- OpenRouter C01: malformed JSON after the available repair attempt.
- OpenRouter C02: repeated a relative edit cumulatively until the accepted-action budget was exhausted.
- OpenRouter C03, D03, and F01: non-terminal schema incidents that recovered but remain reliability evidence.
