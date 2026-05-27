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
