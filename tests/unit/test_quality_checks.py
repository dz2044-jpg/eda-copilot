import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.core.workflow import run_eda


def test_quality_checks_fail_on_large_drift() -> None:
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 10.0, 11.0],
            "target": [0, 0, 1, 1],
            "split": ["train", "train", "test", "test"],
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
    assert drift["overall_status"] == "fail"
    assert drift["top_shifted_features"][0]["status"] == "fail"
    assert result.evidence_packet["quality_checks"]["overall_status"] == "fail"


def test_quality_checks_pass_missingness_gate_when_clean() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 0, 1, 1]})

    result = run_eda(df, EDAConfig(response_column="target"))
    checks = result.evidence_packet["quality_checks"]["checks"]

    missingness_check = [
        check for check in checks if check["check_id"] == "missingness.high_columns"
    ][0]
    assert missingness_check["status"] == "pass"
