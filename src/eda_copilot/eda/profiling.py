from __future__ import annotations

from collections import Counter
import re
from typing import Any

import numpy as np
import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.eda.type_inference import profiles_by_name
from eda_copilot.utils.serialization import to_jsonable


def build_profile_summary(
    df: pd.DataFrame,
    config: EDAConfig,
    type_summary: dict[str, Any],
    missingness_summary: dict[str, Any],
    univariate_summary: dict[str, Any],
    bivariate_summary: dict[str, Any],
    data_quality_warnings: list[dict[str, Any]],
    leakage_warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a native profiling summary inspired by one-click profilers.

    The summary intentionally reuses deterministic evidence computed elsewhere
    instead of delegating core EDA behavior to an external profiler.
    """

    alerts = _build_alerts(
        missingness_summary=missingness_summary,
        bivariate_summary=bivariate_summary,
        data_quality_warnings=data_quality_warnings,
        leakage_warnings=leakage_warnings,
    )
    profile_map = profiles_by_name(type_summary)
    return to_jsonable(
        {
            "available": True,
            "profile_depth": config.profile_depth,
            "sample_policy": config.sample_policy,
            "sensitive_report": {
                "sensitive_columns": list(config.sensitive_columns),
                "redacted_columns": _redacted_columns(profile_map, config),
                "row_samples_included": config.sample_policy != "none",
            },
            "alert_summary": _alert_summary(alerts),
            "alerts": alerts,
            "text_summary": _text_summary(df, profile_map, config),
            "datetime_summary": _datetime_summary(df, profile_map, univariate_summary, config),
            "profiling_modes": {
                "minimal": "Overview, types, warnings, missingness, response, and rankings.",
                "standard": "Minimal profile plus text/date summaries, correlations, drift, and visual specs.",
                "deep": "Standard profile plus token samples and larger diagnostic detail where safe.",
            },
        }
    )


def _build_alerts(
    missingness_summary: dict[str, Any],
    bivariate_summary: dict[str, Any],
    data_quality_warnings: list[dict[str, Any]],
    leakage_warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for warning in data_quality_warnings:
        alerts.append(
            {
                "source": "data_quality",
                "category": _category_for_issue(str(warning.get("issue_type", ""))),
                "severity": warning.get("severity", "low"),
                "scope": "column" if warning.get("column") else "dataset",
                "column": warning.get("column"),
                "issue_type": warning.get("issue_type"),
                "evidence": warning.get("evidence"),
                "recommended_action": warning.get("recommended_action"),
            }
        )

    for warning in leakage_warnings:
        alerts.append(
            {
                "source": "leakage",
                "category": "leakage",
                "severity": warning.get("severity", "high"),
                "scope": "column",
                "column": warning.get("column"),
                "issue_type": warning.get("reason"),
                "evidence": warning.get("evidence"),
                "recommended_action": warning.get("recommended_next_step"),
            }
        )

    for pair in bivariate_summary.get("high_correlation_pairs", []):
        alerts.append(
            {
                "source": "bivariate",
                "category": "correlation",
                "severity": "medium",
                "scope": "pair",
                "column": pair.get("left"),
                "related_column": pair.get("right"),
                "issue_type": "high_correlation",
                "evidence": f"Absolute correlation is {abs(float(pair.get('correlation', 0.0))):.3f}.",
                "recommended_action": "Review whether both features are needed before modeling.",
            }
        )

    for column in missingness_summary.get("high_missingness_columns", []):
        alerts.append(
            {
                "source": "missingness",
                "category": "missingness",
                "severity": "medium",
                "scope": "column",
                "column": column.get("column"),
                "issue_type": "high_missingness",
                "evidence": f"{float(column.get('missing_percentage', 0.0)):.1%} of values are missing.",
                "recommended_action": "Assess missingness mechanism before imputation or exclusion.",
            }
        )
    return sorted(alerts, key=lambda row: _severity_sort(str(row.get("severity", "low"))))


def _category_for_issue(issue_type: str) -> str:
    if "missing" in issue_type or "blank" in issue_type:
        return "missingness"
    if "duplicate" in issue_type:
        return "duplicates"
    if "constant" in issue_type or "cardinality" in issue_type:
        return "distribution"
    if "name" in issue_type or "id" in issue_type:
        return "schema"
    return "data_quality"


def _severity_sort(severity: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(severity, 3)


def _alert_summary(alerts: list[dict[str, Any]]) -> dict[str, Any]:
    by_severity = Counter(str(alert.get("severity", "low")) for alert in alerts)
    by_category = Counter(str(alert.get("category", "data_quality")) for alert in alerts)
    return {
        "total": len(alerts),
        "by_severity": dict(sorted(by_severity.items())),
        "by_category": dict(sorted(by_category.items())),
        "top_alerts": alerts[:10],
    }


def _redacted_columns(profile_map: dict[str, dict[str, Any]], config: EDAConfig) -> list[str]:
    columns = []
    for column, profile in profile_map.items():
        roles = set(profile.get("roles", []))
        if (
            column in config.sensitive_columns
            or column in config.id_columns
            or {"sensitive", "id", "id_candidate"} & roles
        ):
            columns.append(column)
    return sorted(columns)


def _text_summary(
    df: pd.DataFrame,
    profile_map: dict[str, dict[str, Any]],
    config: EDAConfig,
) -> list[dict[str, Any]]:
    if config.profile_depth == "minimal":
        return []

    summaries = []
    for column, profile in profile_map.items():
        if profile.get("semantic_type") != "text" or column not in df.columns:
            continue
        values = df[column].dropna().astype(str)
        lengths = values.str.len()
        words = values.str.split().str.len()
        summary: dict[str, Any] = {
            "column": column,
            "count": int(values.count()),
            "mean_length": float(lengths.mean()) if not lengths.empty else None,
            "p95_length": float(lengths.quantile(0.95)) if not lengths.empty else None,
            "mean_word_count": float(words.mean()) if not words.empty else None,
            "blank_string_count": int((values.str.strip() == "").sum()),
        }
        if config.profile_depth == "deep":
            summary["top_terms"] = _top_terms(values)
        summaries.append(summary)
    return summaries


def _top_terms(values: pd.Series) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for value in values.head(1000):
        counter.update(re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", value.lower()))
    return [{"term": term, "count": int(count)} for term, count in counter.most_common(20)]


def _datetime_summary(
    df: pd.DataFrame,
    profile_map: dict[str, dict[str, Any]],
    univariate_summary: dict[str, Any],
    config: EDAConfig,
) -> list[dict[str, Any]]:
    if config.profile_depth == "minimal":
        return []

    base_by_column = {
        row["column"]: row for row in univariate_summary.get("datetime", [])
    }
    summaries = []
    for column, profile in profile_map.items():
        if profile.get("semantic_type") != "datetime" or column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        clean = parsed.dropna().sort_values()
        gaps = clean.diff().dropna()
        base = dict(base_by_column.get(column, {"column": column}))
        base.update(
            {
                "monotonic_increasing": bool(parsed.dropna().is_monotonic_increasing),
                "median_gap_seconds": _timedelta_seconds(gaps.median()) if not gaps.empty else None,
                "largest_gap_seconds": _timedelta_seconds(gaps.max()) if not gaps.empty else None,
                "inferred_frequency": _infer_frequency(gaps),
            }
        )
        summaries.append(base)
    return summaries


def _timedelta_seconds(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value / np.timedelta64(1, "s"))


def _infer_frequency(gaps: pd.Series) -> str | None:
    if gaps.empty:
        return None
    median_seconds = _timedelta_seconds(gaps.median())
    if median_seconds is None or median_seconds <= 0:
        return None
    if median_seconds <= 60:
        return "sub_hourly"
    if median_seconds <= 3600:
        return "hourly"
    if median_seconds <= 86400:
        return "daily"
    if median_seconds <= 604800:
        return "weekly"
    return "irregular_or_monthly_plus"
