import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.core.workflow import run_eda


def test_profile_redacts_sensitive_and_id_samples() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "notes": [
                "Customer prefers digital statements and long explanations.",
                "Customer called twice about renewal and billing questions.",
                "Agent entered a detailed note about household income.",
                "Follow up required because documentation was incomplete.",
            ],
            "opened_at": ["2024-01-01", "2024-01-02", "2024-01-10", "2024-01-11"],
            "target": [0, 0, 1, 1],
        }
    )

    result = run_eda(
        df,
        EDAConfig(
            response_column="target",
            id_columns=("customer_id",),
            sensitive_columns=("notes",),
            date_columns=("opened_at",),
            profile_depth="deep",
            sample_policy="redacted",
        ),
    )

    overview = result.evidence_packet["dataset_overview"]
    assert overview["sample_rows"][0]["customer_id"] == "<REDACTED>"
    assert overview["sample_rows"][0]["notes"] == "<REDACTED>"
    dictionary = {row["column"]: row for row in overview["data_dictionary"]}
    assert dictionary["customer_id"]["sample_values"] == ["<REDACTED>"]
    assert dictionary["notes"]["sample_values"] == ["<REDACTED>"]
    assert result.evidence_packet["profile_summary"]["text_summary"]
    assert result.evidence_packet["profile_summary"]["datetime_summary"]


def test_minimal_profile_skips_correlation_matrix() -> None:
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [1, 2, 3, 4],
            "target": [0, 0, 1, 1],
        }
    )

    result = run_eda(df, EDAConfig(response_column="target", profile_depth="minimal"))

    bivariate = result.evidence_packet["bivariate_summary"]
    assert bivariate["numeric_correlation_matrix"] == []
    assert bivariate["skipped"] == "profile_depth=minimal"
    assert result.evidence_packet["profile_summary"]["text_summary"] == []
