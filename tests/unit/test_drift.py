import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.core.workflow import run_eda


def test_drift_uses_normalized_missing_group_masks() -> None:
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 100.0, 110.0],
            "split": [None, None, "test", "test"],
            "target": [0, 0, 1, 1],
        }
    )

    result = run_eda(
        df,
        EDAConfig(
            response_column="target",
            train_test_column="split",
            drift_warning_threshold=0.2,
            drift_fail_threshold=0.5,
        ),
    )

    drift = result.evidence_packet["drift_summary"]
    assert drift["available"] is True
    assert drift["baseline_group"] == "<MISSING>"
    assert drift["comparison_group"] == "test"
    assert drift["top_shifted_features"][0]["column"] == "x"
    assert drift["top_shifted_features"][0]["status"] == "fail"


def test_drift_prefers_train_test_order_and_reports_ignored_groups() -> None:
    df = pd.DataFrame(
        {
            "x": [10.0, 11.0, 1.0, 2.0, 50.0, 51.0],
            "split": ["test", "test", "train", "train", "holdout", "holdout"],
            "target": [1, 1, 0, 0, 1, 0],
        }
    )

    result = run_eda(
        df,
        EDAConfig(response_column="target", train_test_column="split"),
    )

    drift = result.evidence_packet["drift_summary"]
    comparison = result.evidence_packet["comparison_summary"]
    assert drift["baseline_group"] == "train"
    assert drift["comparison_group"] == "test"
    assert drift["ignored_groups"] == ["holdout"]
    assert comparison["reference_group"] == "train"
    assert comparison["current_group"] == "test"
    assert comparison["ignored_groups"] == ["holdout"]
