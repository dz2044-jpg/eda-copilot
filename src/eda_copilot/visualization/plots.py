from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px


def missingness_bar(missingness_summary: dict[str, Any]):
    """Build a missingness percentage bar chart."""

    frame = pd.DataFrame(missingness_summary.get("columns", []))
    if frame.empty:
        return None
    frame = frame.sort_values("missing_percentage", ascending=False).head(40)
    return px.bar(
        frame,
        x="missing_percentage",
        y="column",
        orientation="h",
        labels={"missing_percentage": "Missing %", "column": "Column"},
        title="Top Missingness by Column",
    )


def target_distribution(response_summary: dict[str, Any]):
    """Build a target distribution chart when response analysis is available."""

    if not response_summary.get("available"):
        return None
    if response_summary.get("problem_type") == "binary_classification":
        counts = response_summary.get("class_counts", {})
        frame = pd.DataFrame(
            [{"class": str(label), "count": count} for label, count in counts.items()]
        )
        return px.bar(frame, x="class", y="count", title="Response Class Counts")
    if response_summary.get("problem_type") == "regression":
        return None
    return None


def feature_ranking_bar(feature_ranking: list[dict[str, Any]]):
    """Build a top-feature signal chart."""

    frame = pd.DataFrame(feature_ranking)
    if frame.empty or "signal_score" not in frame.columns:
        return None
    frame = frame.dropna(subset=["signal_score"]).sort_values("signal_score", ascending=False).head(20)
    if frame.empty:
        return None
    return px.bar(
        frame,
        x="signal_score",
        y="column",
        orientation="h",
        color="semantic_type",
        title="Top Univariate Feature Signals",
    )


def drift_shift_bar(drift_summary: dict[str, Any]):
    """Build a chart for top reference/current distribution shifts."""

    if not drift_summary.get("available"):
        return None
    frame = pd.DataFrame(drift_summary.get("top_shifted_features", []))
    if frame.empty or "value" not in frame.columns:
        return None
    frame["abs_value"] = frame["value"].abs()
    frame = frame.sort_values("abs_value", ascending=False).head(20)
    return px.bar(
        frame,
        x="abs_value",
        y="column",
        orientation="h",
        color="status" if "status" in frame.columns else "metric",
        title="Top Reference/Current Distribution Shifts",
    )


def quality_check_status_bar(quality_checks: dict[str, Any]):
    """Build a quality gate status count chart."""

    summary = quality_checks.get("summary", {})
    if not summary:
        return None
    frame = pd.DataFrame(
        [
            {"status": status, "count": int(summary.get(status, 0))}
            for status in ["pass", "warn", "fail"]
        ]
    )
    if frame["count"].sum() == 0:
        return None
    return px.bar(frame, x="status", y="count", title="Quality Check Status")


def correlation_heatmap(bivariate_summary: dict[str, Any]):
    """Build a numeric correlation heatmap."""

    records = bivariate_summary.get("numeric_correlation_matrix", [])
    if not records:
        return None
    frame = pd.DataFrame(records).set_index("column")
    if frame.empty:
        return None
    return px.imshow(frame.astype(float), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric Correlation Heatmap")
