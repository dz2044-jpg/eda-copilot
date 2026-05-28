from eda_copilot.core.config import EDAConfig
from eda_copilot.visualization.gallery import build_plot_gallery
from eda_copilot.visualization.specs import build_visual_specs


def test_target_distribution_spec_matches_available_binary_renderer() -> None:
    response_summary = {
        "available": True,
        "problem_type": "binary_classification",
        "class_counts": {0: 2, 1: 3},
    }

    specs = build_visual_specs(EDAConfig(), {}, response_summary, {}, [], {})
    figures = build_plot_gallery(
        {
            "missingness_summary": {},
            "response_summary": response_summary,
            "feature_ranking": [],
            "bivariate_summary": {},
            "drift_summary": {},
            "quality_checks": {},
        }
    )

    assert "target_distribution" in {spec["spec_id"] for spec in specs}
    assert "target_distribution" in figures


def test_target_distribution_spec_is_not_emitted_without_renderer() -> None:
    response_summary = {
        "available": True,
        "problem_type": "regression",
        "target_summary": {"count": 5},
    }

    specs = build_visual_specs(EDAConfig(), {}, response_summary, {}, [], {})
    figures = build_plot_gallery(
        {
            "missingness_summary": {},
            "response_summary": response_summary,
            "feature_ranking": [],
            "bivariate_summary": {},
            "drift_summary": {},
            "quality_checks": {},
        }
    )

    assert "target_distribution" not in {spec["spec_id"] for spec in specs}
    assert "target_distribution" not in figures
