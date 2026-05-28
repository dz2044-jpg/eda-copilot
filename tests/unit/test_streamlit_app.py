import pandas as pd
import streamlit as st

from eda_copilot.app.streamlit_app import _dataset_key, _run_label, _set_loaded_dataset


def test_dataset_key_is_stable_for_same_shape_and_columns() -> None:
    df = pd.DataFrame({"x": [1, 2], "target": [0, 1]})

    assert _dataset_key(df, "demo") == _dataset_key(df, "demo")


def test_dataset_key_changes_when_values_change() -> None:
    left = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
    right = pd.DataFrame({"x": [1, 99], "target": [0, 1]})

    assert _dataset_key(left, "demo") != _dataset_key(right, "demo")


def test_set_loaded_dataset_clears_stale_eda_result_when_data_changes() -> None:
    st.session_state.clear()
    st.session_state["eda_result"] = object()
    st.session_state["eda_config"] = object()

    _set_loaded_dataset(pd.DataFrame({"x": [1, 2]}), "demo")

    assert "eda_result" not in st.session_state
    assert "eda_config" not in st.session_state


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
