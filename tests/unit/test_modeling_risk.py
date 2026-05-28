import json

import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.core.workflow import run_eda


def test_modeling_risk_summary_combines_leakage_cardinality_missingness_and_drift() -> None:
    df = pd.DataFrame(
        {
            "record_key": [f"R{i:03d}" for i in range(24)],
            "leaky_score": [0.0] * 12 + [1.0] * 12,
            "sparse_flag": ["Y"] * 23 + ["N"],
            "missing_signal": [None] * 12 + list(range(12)),
            "split": ["train"] * 12 + ["test"] * 12,
            "target": [0] * 12 + [1] * 12,
        }
    )

    result = run_eda(
        df,
        EDAConfig(
            response_column="target",
            train_test_column="split",
            high_cardinality_threshold=5,
            near_constant_threshold=0.95,
            drift_warning_threshold=0.1,
            drift_fail_threshold=0.5,
        ),
    )

    modeling = result.evidence_packet["modeling_risk_summary"]
    categories = {risk["category"] for risk in modeling["risks"]}

    assert modeling["overall_status"] == "fail"
    assert {"leakage", "identifier", "cardinality", "missingness", "drift", "sparsity"}.issubset(categories)
    assert json.dumps(modeling)


def test_clean_modeling_risk_summary_is_pass_when_no_risks_are_detected() -> None:
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "group": ["a", "b", "a", "b", "a", "b"],
        }
    )

    result = run_eda(df, EDAConfig())

    modeling = result.evidence_packet["modeling_risk_summary"]
    assert modeling["overall_status"] == "pass"
    assert modeling["risks"] == []
