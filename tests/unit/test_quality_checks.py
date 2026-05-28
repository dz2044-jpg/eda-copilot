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


def test_data_quality_warnings_are_structured_and_schema_aware() -> None:
    df = pd.DataFrame(
        {
            "Account ID": ["A1", "A1", "A3", "A4"],
            "account-id": ["B1", "B2", "B3", "B4"],
            "amount_text": ["10.5", "20.0", "30.25", "40.0"],
            "flag_text": ["yes", "no", "yes", "no"],
            "target": [0, 0, 1, 1],
        }
    )

    result = run_eda(
        df,
        EDAConfig(
            response_column="target",
            id_columns=("Account ID",),
        ),
    )
    warnings = result.evidence_packet["data_quality_warnings"]
    by_issue = {warning["issue_type"]: warning for warning in warnings}

    assert by_issue["duplicate_id_values"]["warning_id"] == "duplicate_id_values.account_id"
    assert by_issue["duplicate_id_values"]["scope"] == "column"
    assert by_issue["duplicate_id_values"]["metric_name"] == "duplicate_id_count"
    assert by_issue["normalized_name_collision"]["scope"] == "schema"
    assert by_issue["numeric_parse_candidate"]["column"] == "amount_text"
    assert by_issue["boolean_parse_candidate"]["column"] == "flag_text"
