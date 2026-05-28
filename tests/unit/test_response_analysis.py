import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.eda.response_analysis import analyze_response
from eda_copilot.eda.type_inference import infer_column_types


def test_binary_response_analysis_computes_auc_and_rates() -> None:
    df = pd.DataFrame(
        {
            "score": [0.1, 0.2, 0.8, 0.9, 0.15, 0.85],
            "category": ["low", "low", "high", "high", "low", "high"],
            "target": [0, 0, 1, 1, 0, 1],
        }
    )
    config = EDAConfig(response_column="target")
    types = infer_column_types(df, config)

    result = analyze_response(df, config, types)
    relationships = {row["column"]: row for row in result["feature_relationships"]}

    assert result["problem_type"] == "binary_classification"
    assert result["response_rate"] == 0.5
    assert result["class_table"][1]["is_positive_class"] is True
    assert result["warnings"][0]["issue_type"] == "positive_class_auto_selected"
    assert relationships["score"]["auc_abs"] == 1.0
    assert relationships["score"]["signal_direction"] == "positive"
    assert relationships["score"]["monotonic_trend"] is True
    assert relationships["category"]["response_rate_spread"] == 1.0
    assert relationships["category"]["top_risky_categories"][0]["category"] == "high"


def test_binary_response_warnings_include_missing_and_imbalance() -> None:
    df = pd.DataFrame(
        {
            "x": list(range(12)),
            "target": [1] + [0] * 10 + [None],
        }
    )
    config = EDAConfig(response_column="target")
    types = infer_column_types(df, config)

    result = analyze_response(df, config, types)
    issue_types = {warning["issue_type"] for warning in result["warnings"]}

    assert result["imbalance_ratio"] == 10.0
    assert {"missing_response_values", "severe_class_imbalance", "positive_class_auto_selected"}.issubset(issue_types)


def test_regression_response_summary_flags_outliers_and_missing_values() -> None:
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "target": [1.0, 1.0, 1.0, 1.0, 100.0, None],
        }
    )
    config = EDAConfig(response_column="target", problem_type="regression")
    types = infer_column_types(df, config)

    result = analyze_response(df, config, types)
    issue_types = {warning["issue_type"] for warning in result["warnings"]}

    assert result["target_summary"]["missing_count"] == 1
    assert result["target_summary"]["iqr_outlier_count"] == 1
    assert "missing_response_values" in issue_types
    assert "regression_response_outliers" in issue_types
