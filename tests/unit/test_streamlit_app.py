import pandas as pd

from eda_copilot.app.streamlit_app import _dataset_key


def test_dataset_key_is_stable_for_same_shape_and_columns() -> None:
    df = pd.DataFrame({"x": [1, 2], "target": [0, 1]})

    assert _dataset_key(df, "demo") == _dataset_key(df, "demo")
