from __future__ import annotations

from typing import Any

from eda_copilot.core.config import EDAConfig
from eda_copilot.eda.type_inference import feature_profiles


def build_feature_ranking(
    config: EDAConfig,
    type_summary: dict[str, Any],
    missingness_summary: dict[str, Any],
    response_summary: dict[str, Any],
    data_quality_warnings: list[dict[str, Any]],
    leakage_warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine deterministic diagnostics into a univariate feature ranking table."""

    missing_by_column = {item["column"]: item for item in missingness_summary.get("columns", [])}
    response_by_column = {
        item["column"]: item for item in response_summary.get("feature_relationships", [])
    }
    quality_by_column = _warning_map(data_quality_warnings, "issue_type")
    leakage_by_column = _warning_map(leakage_warnings, "reason")
    quality_details_by_column = _warning_details_by_column(data_quality_warnings)
    leakage_details_by_column = _warning_details_by_column(leakage_warnings)

    rows = []
    for profile in feature_profiles(type_summary, config):
        column = profile["name"]
        response = response_by_column.get(column, {})
        signal_metric, signal_value, signal_score = _signal_from_response(response_summary, response)
        quality_warnings = quality_by_column.get(column, [])
        leakage_warnings_for_column = leakage_by_column.get(column, [])
        quality_penalty = _quality_penalty(
            quality_details_by_column.get(column, []),
            leakage_details_by_column.get(column, []),
        )
        rows.append(
            {
                "column": column,
                "semantic_type": profile["semantic_type"],
                "missing_percentage": missing_by_column.get(column, {}).get("missing_percentage", profile["missing_percentage"]),
                "unique_count": profile["unique_count"],
                "signal_metric": signal_metric,
                "signal_value": signal_value,
                "signal_score": signal_score,
                "signal_direction": response.get("signal_direction"),
                "quality_penalty": quality_penalty,
                "recommended_review_reason": _recommended_review_reason(
                    column=column,
                    signal_metric=signal_metric,
                    quality_warnings=quality_warnings,
                    leakage_warnings=leakage_warnings_for_column,
                ),
                "response_rate_spread": response.get("response_rate_spread"),
                "auc": response.get("auc"),
                "auc_abs": response.get("auc_abs"),
                "chi_square_p_value": response.get("chi_square_p_value"),
                "data_quality_warnings": quality_warnings,
                "leakage_warnings": leakage_warnings_for_column,
            }
        )

    sorted_rows = sorted(
        rows,
        key=lambda item: (
            bool(item["leakage_warnings"]),
            item["signal_score"] if item["signal_score"] is not None else -1.0,
            -float(item["quality_penalty"]),
        ),
        reverse=True,
    )
    for index, row in enumerate(sorted_rows, start=1):
        row["rank"] = index
    return sorted_rows


def _signal_from_response(
    response_summary: dict[str, Any],
    relationship: dict[str, Any],
) -> tuple[str | None, float | None, float | None]:
    if not relationship or not response_summary.get("available"):
        return None, None, None
    problem_type = response_summary.get("problem_type")
    if problem_type == "binary_classification":
        if relationship.get("auc_abs") is not None:
            auc_abs = float(relationship["auc_abs"])
            return "auc_abs", auc_abs, abs(auc_abs - 0.5) * 2.0
        if relationship.get("cramers_v") is not None:
            value = float(relationship["cramers_v"])
            return "cramers_v", value, value
        if relationship.get("response_rate_spread") is not None:
            value = float(relationship["response_rate_spread"])
            return "response_rate_spread", value, value
    if problem_type == "regression":
        if relationship.get("correlation") is not None:
            value = float(relationship["correlation"])
            return "correlation_abs", abs(value), abs(value)
        if relationship.get("target_mean_spread") is not None:
            value = float(relationship["target_mean_spread"])
            return "target_mean_spread", value, value
    return None, None, None


def _warning_map(warnings: list[dict[str, Any]], label_key: str) -> dict[str, list[str]]:
    mapped: dict[str, list[str]] = {}
    for warning in warnings:
        column = warning.get("column")
        if not column:
            continue
        mapped.setdefault(column, []).append(str(warning.get(label_key)))
    return mapped


def _warning_details_by_column(warnings: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    mapped: dict[str, list[dict[str, Any]]] = {}
    for warning in warnings:
        column = warning.get("column")
        if not column:
            continue
        mapped.setdefault(str(column), []).append(warning)
    return mapped


def _quality_penalty(
    quality_warnings: list[dict[str, Any]],
    leakage_warnings: list[dict[str, Any]],
) -> float:
    weights = {"high": 0.5, "medium": 0.25, "low": 0.1}
    score = 0.0
    for warning in [*quality_warnings, *leakage_warnings]:
        score += weights.get(str(warning.get("severity", "low")), 0.1)
    return min(score, 1.0)


def _recommended_review_reason(
    column: str,
    signal_metric: str | None,
    quality_warnings: list[str],
    leakage_warnings: list[str],
) -> str:
    if leakage_warnings:
        return "Review potential leakage before using this feature."
    if quality_warnings:
        return "Review data quality warnings before modeling."
    if signal_metric:
        return f"Review because `{column}` has a measurable univariate response signal."
    return "Review only if the feature is relevant to the project context."
