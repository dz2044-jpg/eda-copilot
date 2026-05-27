import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.eda.missingness import analyze_missingness


def test_missingness_counts_null_blank_and_special_codes() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "numeric": [1, None, 999, 0],
            "category": ["A", "", "UNKNOWN", None],
        }
    )

    result = analyze_missingness(df, EDAConfig(response_column="target"))
    by_column = {row["column"]: row for row in result["columns"]}

    assert by_column["numeric"]["missing_count"] == 1
    assert by_column["numeric"]["zero_count"] == 1
    assert by_column["numeric"]["special_missing_code_counts"]["999"] == 1
    assert by_column["category"]["blank_string_count"] == 1
    assert by_column["category"]["special_missing_code_counts"]["unknown"] == 1
    assert len(result["missingness_by_response"]) == 2
