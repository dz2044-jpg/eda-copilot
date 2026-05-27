from __future__ import annotations

from typing import Any


def build_llm_evidence_context(evidence_packet: dict[str, Any]) -> dict[str, Any]:
    """Return the only context an optional LLM summary is allowed to read."""

    allowed_keys = [
        "metadata",
        "dataset_overview",
        "profile_summary",
        "response_summary",
        "column_type_summary",
        "missingness_summary",
        "feature_ranking",
        "data_quality_warnings",
        "leakage_warnings",
        "drift_summary",
        "quality_checks",
        "comparison_summary",
        "visual_specs",
        "caveats",
    ]
    context = {key: evidence_packet.get(key) for key in allowed_keys}
    if isinstance(context.get("dataset_overview"), dict):
        context["dataset_overview"] = _sanitized_overview(context["dataset_overview"])
    return context


def deterministic_summary_placeholder(evidence_packet: dict[str, Any]) -> dict[str, Any]:
    """Provide a non-LLM summary while preserving the AI layer contract."""

    ranking = evidence_packet.get("feature_ranking", [])
    quality = evidence_packet.get("data_quality_warnings", [])
    leakage = evidence_packet.get("leakage_warnings", [])
    quality_checks = evidence_packet.get("quality_checks", {})
    return {
        "status": "ai_summary_not_configured",
        "message": "AI summary is optional and must summarize only the deterministic evidence packet.",
        "quality_gate_status": quality_checks.get("overall_status"),
        "top_feature_candidates": [row["column"] for row in ranking[:5]],
        "top_data_quality_concerns": quality[:5],
        "top_leakage_warnings": leakage[:5],
    }


def _sanitized_overview(overview: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(overview)
    sanitized.pop("sample_rows", None)
    data_dictionary = []
    for row in overview.get("data_dictionary", []):
        if isinstance(row, dict):
            clean_row = dict(row)
            clean_row.pop("sample_values", None)
            data_dictionary.append(clean_row)
    sanitized["data_dictionary"] = data_dictionary
    sanitized["row_samples_removed_for_ai"] = True
    return sanitized
