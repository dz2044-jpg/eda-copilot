from __future__ import annotations

from typing import Any

from eda_copilot.core.config import EDAConfig


def build_visual_specs(
    config: EDAConfig,
    missingness_summary: dict[str, Any],
    response_summary: dict[str, Any],
    bivariate_summary: dict[str, Any],
    feature_ranking: list[dict[str, Any]],
    drift_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Describe reproducible chart specs that can be rendered or exported."""

    specs = []
    if missingness_summary.get("columns"):
        specs.append(
            _spec(
                spec_id="missingness_bar",
                title="Top Missingness by Column",
                chart_type="bar",
                source_section="missingness_summary.columns",
                fields={"x": "missing_percentage", "y": "column"},
                purpose="Prioritize columns with missingness review needs.",
            )
        )
    if response_summary.get("available"):
        specs.append(
            _spec(
                spec_id="target_distribution",
                title="Response Distribution",
                chart_type="bar",
                source_section="response_summary.class_counts",
                fields={"x": "class", "y": "count"},
                purpose="Check target balance and positive-class setup.",
            )
        )
    if feature_ranking:
        specs.append(
            _spec(
                spec_id="feature_ranking",
                title="Top Univariate Feature Signals",
                chart_type="bar",
                source_section="feature_ranking",
                fields={"x": "signal_score", "y": "column", "color": "semantic_type"},
                purpose="Inspect response-aware candidate feature signals.",
            )
        )
    if bivariate_summary.get("numeric_correlation_matrix"):
        specs.append(
            _spec(
                spec_id="correlation_heatmap",
                title="Numeric Correlation Heatmap",
                chart_type="heatmap",
                source_section="bivariate_summary.numeric_correlation_matrix",
                fields={"x": "numeric columns", "y": "numeric columns", "color": "correlation"},
                purpose="Find redundant numeric predictors.",
            )
        )
    if drift_summary.get("available") and drift_summary.get("top_shifted_features"):
        specs.append(
            _spec(
                spec_id="drift_shift_bar",
                title="Top Reference/Current Distribution Shifts",
                chart_type="bar",
                source_section="drift_summary.top_shifted_features",
                fields={"x": "abs_value", "y": "column", "color": "status"},
                purpose="Review features whose distributions changed across configured groups.",
            )
        )

    for spec in specs:
        spec["profile_depth"] = config.profile_depth
        spec["status"] = "ready"
    return specs


def _spec(
    spec_id: str,
    title: str,
    chart_type: str,
    source_section: str,
    fields: dict[str, str],
    purpose: str,
) -> dict[str, Any]:
    return {
        "spec_id": spec_id,
        "title": title,
        "chart_type": chart_type,
        "source_section": source_section,
        "fields": fields,
        "purpose": purpose,
    }
