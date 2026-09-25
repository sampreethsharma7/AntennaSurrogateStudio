"""Contract provenance attached to live antenna-agent observations."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from studio.antenna_agent import create_default_agent
from studio.antenna_llm_planner import (
    AGENT_STEP_CONTRACT_VERSION,
    AGENT_STEP_SCHEMA_VERSION,
    AgentLoopBudgets,
    build_agent_step_exchange,
    plan_json_schema,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def current_contract_fingerprints() -> dict[str, str]:
    """Fingerprint the same provider-neutral fixture protected by the contract test."""

    manifest = create_default_agent().capability_manifest(None)
    exchange = build_agent_step_exchange(
        instruction="Inspect installed capabilities.",
        current_design=None,
        capability_manifest=manifest,
        remaining_budgets=AgentLoopBudgets(),
    )
    return {
        "agent_user_content_sha256": _sha256_text(exchange.user_content),
        "agent_system_instruction_sha256": _sha256_text(exchange.system_instruction),
        "agent_step_schema_sha256": _sha256_text(
            json.dumps(exchange.schema, sort_keys=True)
        ),
        "toolplan_schema_sha256": _sha256_text(
            json.dumps(
                plan_json_schema(tuple(manifest["callable_tools"])),
                sort_keys=True,
            )
        ),
    }


def observation_contract_provenance(
    benchmark_sha256: str,
    *,
    captured_at_utc: str | None = None,
) -> dict[str, Any]:
    """Return the immutable contract identity required for a live observation."""

    return {
        "agent_step_contract_version": AGENT_STEP_CONTRACT_VERSION,
        "agent_step_schema_version": AGENT_STEP_SCHEMA_VERSION,
        "benchmark_sha256": benchmark_sha256,
        "captured_at_utc": captured_at_utc or datetime.now(timezone.utc).isoformat(),
        "contract_fingerprints": current_contract_fingerprints(),
    }
