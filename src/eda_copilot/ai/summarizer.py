from __future__ import annotations

from typing import Any


def build_llm_evidence_context(evidence_packet: dict[str, Any]) -> dict[str, Any]:
    """Return the only context an optional LLM summary is allowed to read."""

    allowed_keys = [
        "metadata",
        "dataset_overview",
        "response_summary",
        "column_type_summary",
        "missingness_summary",
        "feature_ranking",
        "data_quality_warnings",
        "leakage_warnings",
        "drift_summary",
        "caveats",
    ]
    return {key: evidence_packet.get(key) for key in allowed_keys}


def deterministic_summary_placeholder(evidence_packet: dict[str, Any]) -> dict[str, Any]:
    """Provide a non-LLM summary while preserving the AI layer contract."""

    ranking = evidence_packet.get("feature_ranking", [])
    quality = evidence_packet.get("data_quality_warnings", [])
    leakage = evidence_packet.get("leakage_warnings", [])
    return {
        "status": "ai_summary_not_configured",
        "message": "AI summary is optional and must summarize only the deterministic evidence packet.",
        "top_feature_candidates": [row["column"] for row in ranking[:5]],
        "top_data_quality_concerns": quality[:5],
        "top_leakage_warnings": leakage[:5],
    }
