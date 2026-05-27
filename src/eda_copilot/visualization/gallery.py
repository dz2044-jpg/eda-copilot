from __future__ import annotations

from typing import Any

from eda_copilot.visualization.plots import (
    correlation_heatmap,
    drift_shift_bar,
    feature_ranking_bar,
    missingness_bar,
    quality_check_status_bar,
    target_distribution,
)


def build_plot_gallery(evidence_packet: dict[str, Any]) -> dict[str, Any]:
    """Build standard Plotly figures from an evidence packet."""

    plot_builders = {
        "missingness_bar": lambda: missingness_bar(evidence_packet["missingness_summary"]),
        "target_distribution": lambda: target_distribution(evidence_packet["response_summary"]),
        "feature_ranking": lambda: feature_ranking_bar(evidence_packet["feature_ranking"]),
        "correlation_heatmap": lambda: correlation_heatmap(evidence_packet["bivariate_summary"]),
        "drift_shift_bar": lambda: drift_shift_bar(evidence_packet["drift_summary"]),
        "quality_check_status": lambda: quality_check_status_bar(evidence_packet["quality_checks"]),
    }
    figures = {}
    for name, builder in plot_builders.items():
        figure = builder()
        if figure is not None:
            figures[name] = figure
    return figures
