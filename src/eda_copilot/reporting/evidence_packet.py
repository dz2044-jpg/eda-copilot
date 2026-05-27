from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from eda_copilot import __version__
from eda_copilot.core.config import EDAConfig
from eda_copilot.utils.serialization import to_jsonable


def build_evidence_packet(
    config: EDAConfig,
    dataset_overview: dict[str, Any],
    column_type_summary: dict[str, Any],
    missingness_summary: dict[str, Any],
    univariate_summary: dict[str, Any],
    response_summary: dict[str, Any],
    bivariate_summary: dict[str, Any],
    feature_ranking: list[dict[str, Any]],
    data_quality_warnings: list[dict[str, Any]],
    leakage_warnings: list[dict[str, Any]],
    drift_summary: dict[str, Any],
    plots: list[dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    caveats: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the deterministic evidence packet consumed by reports and AI summaries."""

    packet = {
        "metadata": {
            "tool_name": "eda_copilot",
            "tool_version": __version__,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "config": config.to_dict(),
        "dataset_overview": dataset_overview,
        "response_summary": response_summary,
        "column_type_summary": column_type_summary,
        "missingness_summary": missingness_summary,
        "univariate_summary": univariate_summary,
        "bivariate_summary": bivariate_summary,
        "feature_ranking": feature_ranking,
        "data_quality_warnings": data_quality_warnings,
        "leakage_warnings": leakage_warnings,
        "drift_summary": drift_summary,
        "plots": plots or [],
        "artifacts": artifacts or [],
        "caveats": caveats or [],
    }
    return to_jsonable(packet)
