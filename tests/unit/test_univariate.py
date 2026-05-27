import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.eda.type_inference import infer_column_types
from eda_copilot.eda.univariate import analyze_univariate


def test_univariate_numeric_and_categorical_summaries() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 100], "segment": ["a", "a", "b", "c"]})
    config = EDAConfig()
    types = infer_column_types(df, config)

    result = analyze_univariate(df, config, types)

    numeric = {row["column"]: row for row in result["numeric"]}
    categorical = {row["column"]: row for row in result["categorical"]}
    assert numeric["x"]["median"] == 2.5
    assert numeric["x"]["iqr_outlier_count"] == 1
    assert categorical["segment"]["unique_count"] == 3
