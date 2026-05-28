import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.core.workflow import run_eda


def test_feature_ranking_prioritizes_strong_binary_signal() -> None:
    df = pd.DataFrame(
        {
            "weak": [1, 2, 1, 2, 1, 2],
            "strong_score": [0.1, 0.2, 0.8, 0.9, 0.15, 0.85],
            "target": [0, 0, 1, 1, 0, 1],
        }
    )

    result = run_eda(df, EDAConfig(response_column="target"))
    ranking = result.evidence_packet["feature_ranking"]

    assert ranking[0]["column"] == "strong_score"
    assert ranking[0]["signal_metric"] == "auc_abs"
    assert ranking[0]["rank"] == 1
    assert ranking[0]["signal_direction"] == "positive"
    assert "Near-perfect univariate AUC" in ranking[0]["leakage_warnings"]
    assert ranking[0]["recommended_review_reason"] == "Review potential leakage before using this feature."


def test_feature_ranking_handles_categorical_binary_signal() -> None:
    df = pd.DataFrame(
        {
            "category": ["low", "low", "high", "high", "low", "high"],
            "target": [0, 0, 1, 1, 0, 1],
        }
    )

    result = run_eda(df, EDAConfig(response_column="target"))
    ranking = result.evidence_packet["feature_ranking"]

    assert ranking[0]["column"] == "category"
    assert ranking[0]["signal_metric"] == "cramers_v"
    assert ranking[0]["response_rate_spread"] == 1.0


def test_feature_ranking_handles_regression_signal() -> None:
    df = pd.DataFrame(
        {
            "weak": [1, 1, 1, 2, 2, 2],
            "strong": [1, 2, 3, 8, 9, 10],
            "target": [2, 4, 6, 16, 18, 20],
        }
    )

    result = run_eda(df, EDAConfig(response_column="target", problem_type="regression"))
    ranking = result.evidence_packet["feature_ranking"]

    assert ranking[0]["column"] == "strong"
    assert ranking[0]["signal_metric"] == "correlation_abs"
    assert ranking[0]["signal_direction"] == "positive"
