"""Small deterministic vocabulary for durable antenna project intent.

The LLM proposes meaning and exact evidence.  This module only canonicalizes a
validated proposal's representation; it never extracts intent from user prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SEMANTIC_NORMALIZATION_VERSION = "semantic-memory-normalization-v2"


@dataclass(frozen=True, slots=True)
class NormalizedSemanticProposal:
    semantic_key: str
    canonical_value: Any
    unit: str | None
    constraint_operator: str | None
    normalization_rule: str
    proposed_key: str
    proposed_value: Any


_UNIT_IDENTITIES = {
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "ghz": "GHz",
    "mhz": "MHz",
    "khz": "kHz",
    "hz": "Hz",
    "deg": "deg",
    "degree": "deg",
    "degrees": "deg",
}

_UNIT_SUFFIXES = {
    "mm": "_mm",
    "cm": "_cm",
    "m": "_m",
    "GHz": "_ghz",
    "MHz": "_mhz",
    "kHz": "_khz",
    "Hz": "_hz",
    "deg": "_deg",
}

_POLARIZATION_KEYS = {
    "target_polarization",
    "polarization_target",
    "desired_polarization",
    "polarization",
}
_POLARIZATION_VALUES = {
    "linear": "linear",
    "linear_polarization": "linear",
    "circular": "circular_polarization",
    "circular_polarization": "circular_polarization",
    "rhcp": "rhcp",
    "right_hand_circular": "rhcp",
    "right_hand_circular_polarization": "rhcp",
    "lhcp": "lhcp",
    "left_hand_circular": "lhcp",
    "left_hand_circular_polarization": "lhcp",
}

_BOARD_WIDTH_KEYS = {
    "max_board_width",
    "board_width_max",
    "board_width_limit",
}
_STAGE5E_BOARD_DIMENSION_KEYS = {
    "max_board_dimension",
}

_FUTURE_FEED_KEYS = {
    "future_feed_network",
    "feed_network_goal",
    "feed_network_future_goal",
}
_FEED_VALUES = {
    "corporate": "corporate_feed",
    "corporate_feed": "corporate_feed",
    "corporate_distribution": "corporate_feed",
    "corporate_distribution_network": "corporate_feed",
    "series": "series_feed",
    "series_feed": "series_feed",
    "series_fed": "series_feed",
    "series_fed_network": "series_feed",
    "independent": "independent_ports",
    "independent_ports": "independent_ports",
    "separate_element_ports": "independent_ports",
}


def _snake_case(value: str) -> str:
    characters: list[str] = []
    for character in value.strip().casefold():
        if ("a" <= character <= "z") or ("0" <= character <= "9"):
            characters.append(character)
        elif characters and characters[-1] != "_":
            characters.append("_")
    return "".join(characters).strip("_")


def normalize_engineering_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    stripped = unit.strip()
    return _UNIT_IDENTITIES.get(stripped.casefold(), stripped)


def _generic_key(key: str, unit: str | None) -> tuple[str, bool]:
    normalized = _snake_case(key)
    suffix = _UNIT_SUFFIXES.get(unit or "")
    if suffix and normalized.endswith(suffix) and len(normalized) > len(suffix):
        return normalized[: -len(suffix)], True
    return normalized, normalized != key


def _evidence_supports_board_maximum(evidence_quote: str) -> bool:
    """Recognize only the width/maximum wording demonstrated by Stage 5E.

    The registered key supplies the board-dimension context.  Evidence must
    independently identify the board and an upper-bound relationship.  This
    deliberately excludes neutral dimensions such as ``substrate width = 80``.
    """

    evidence = f"_{_snake_case(evidence_quote)}_"
    board = "_board_" in evidence
    maximum = any(
        cue in evidence
        for cue in (
            "_under_",
            "_maximum_",
            "_max_",
            "_at_most_",
            "_up_to_",
            "_no_wider_than_",
            "_not_exceed_",
        )
    )
    return board and maximum


def normalize_semantic_proposal(
    *,
    key: str,
    value: Any,
    unit: str | None,
    semantic_kind: str,
    evidence_quote: str,
) -> NormalizedSemanticProposal:
    """Canonicalize an already validated, grounded semantic proposal."""

    canonical_unit = normalize_engineering_unit(unit)
    generic_key, structurally_changed = _generic_key(key, canonical_unit)
    normalized_value = _snake_case(value) if isinstance(value, str) else value
    rule = "generic:snake_case"
    if structurally_changed:
        rule = "generic:snake_case_and_redundant_unit_suffix"

    if generic_key in _POLARIZATION_KEYS and (
        value is None or normalized_value in _POLARIZATION_VALUES
    ):
        canonical_value = (
            _POLARIZATION_VALUES.get(normalized_value, value)
            if isinstance(value, str)
            else value
        )
        return NormalizedSemanticProposal(
            "target_polarization",
            canonical_value,
            canonical_unit,
            None,
            "registered:polarization_target",
            key,
            value,
        )

    if (
        generic_key in _BOARD_WIDTH_KEYS
        and semantic_kind == "constraint"
        and (
            value is None
            or (isinstance(value, (int, float)) and not isinstance(value, bool))
        )
    ):
        return NormalizedSemanticProposal(
            "board_width_limit",
            None if value is None else float(value),
            canonical_unit,
            "max",
            "registered:board_width_maximum_constraint",
            key,
            value,
        )

    if (
        generic_key in _STAGE5E_BOARD_DIMENSION_KEYS
        and semantic_kind == "constraint"
        and _evidence_supports_board_maximum(evidence_quote)
        and (
            value is None
            or (isinstance(value, (int, float)) and not isinstance(value, bool))
        )
    ):
        return NormalizedSemanticProposal(
            "board_width_limit",
            None if value is None else float(value),
            canonical_unit,
            "max",
            "registered:stage5e_board_dimension_maximum_constraint",
            key,
            value,
        )

    future_feed_key = generic_key in _FUTURE_FEED_KEYS
    explicit_corporate_future = (
        generic_key == "corporate_distribution_network"
        and semantic_kind in {"goal", "future_intent"}
        and (
            value is None
            or normalized_value in {
                "corporate", "corporate_feed", "corporate_distribution",
                "corporate_distribution_network",
            }
        )
    )
    stage5e_series_future = (
        generic_key == "series_fed_network"
        and semantic_kind == "future_intent"
    )
    stage5e_corporate_future = (
        generic_key == "corporate_feed"
        and semantic_kind == "future_intent"
    )
    if (
        (
            future_feed_key
            and semantic_kind in {"goal", "future_intent"}
            and (value is None or normalized_value in _FEED_VALUES)
        )
        or explicit_corporate_future
        or stage5e_series_future
        or stage5e_corporate_future
    ):
        if stage5e_series_future:
            canonical_value = "series_feed"
        elif stage5e_corporate_future:
            canonical_value = "corporate_feed"
        else:
            canonical_value = (
                _FEED_VALUES.get(normalized_value, value)
                if isinstance(value, str)
                else value
            )
        return NormalizedSemanticProposal(
            "feed_network_future_goal",
            canonical_value,
            canonical_unit,
            None,
            "registered:future_feed_network_goal",
            key,
            value,
        )

    return NormalizedSemanticProposal(
        generic_key,
        value,
        canonical_unit,
        None,
        rule,
        key,
        value,
    )


def canonical_semantic_identity(
    *,
    key: str,
    value: Any,
    unit: str | None,
    semantic_kind: str | None,
    evidence_quote: str | None,
) -> str:
    """Resolve stored legacy/new items through the same deterministic identity."""

    if semantic_kind is None:
        return key
    return normalize_semantic_proposal(
        key=key,
        value=value,
        unit=unit,
        semantic_kind=semantic_kind,
        evidence_quote=evidence_quote or "",
    ).semantic_key


def canonical_reference_identity(
    reference_id: str,
    *,
    value: Any,
    unit: str | None,
    semantic_kind: str,
) -> str | None:
    """Recover the historical proposed key embedded in semantic item IDs."""

    if not reference_id.startswith("semantic:"):
        return None
    parts = reference_id.split(":", 3)
    if len(parts) < 3 or not parts[1]:
        return None
    return normalize_semantic_proposal(
        key=parts[1],
        value=value,
        unit=unit,
        semantic_kind=semantic_kind,
        evidence_quote="",
    ).semantic_key
