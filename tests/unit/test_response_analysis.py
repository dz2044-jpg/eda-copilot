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
    assert relationships["score"]["auc_abs"] == 1.0
    assert relationships["category"]["response_rate_spread"] == 1.0
