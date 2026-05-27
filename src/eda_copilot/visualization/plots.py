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


def correlation_heatmap(bivariate_summary: dict[str, Any]):
    """Build a numeric correlation heatmap."""

    records = bivariate_summary.get("numeric_correlation_matrix", [])
    if not records:
        return None
    frame = pd.DataFrame(records).set_index("column")
    if frame.empty:
        return None
    return px.imshow(frame.astype(float), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric Correlation Heatmap")
