import pandas as pd

from eda_copilot.app.streamlit_app import _dataset_key, _run_label


def test_dataset_key_is_stable_for_same_shape_and_columns() -> None:
    df = pd.DataFrame({"x": [1, 2], "target": [0, 1]})

    assert _dataset_key(df, "demo") == _dataset_key(df, "demo")


def test_run_label_includes_metadata_when_available() -> None:
    label = _run_label(
        {
            "created_at_utc": "2026-01-01T00:00:00Z",
            "dataset_name": "demo",
            "run_id": "abc123",
            "run_dir": "/tmp/run",
        }
    )

    assert "demo" in label
    assert "abc123" in label
