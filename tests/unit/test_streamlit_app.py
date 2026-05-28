import sys
import types

import pandas as pd
import streamlit as st

from eda_copilot.app.optional_visual_explorer import render_optional_pygwalker
from eda_copilot.app.streamlit_app import _dataset_key, _result_config_is_stale, _run_label, _set_loaded_dataset
from eda_copilot.core.config import EDAConfig


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


def test_result_config_is_stale_when_sidebar_config_changes() -> None:
    st.session_state.clear()
    original = EDAConfig(response_column="target")
    changed = EDAConfig(response_column="target", profile_depth="deep")
    st.session_state["eda_result"] = object()
    st.session_state["eda_config"] = original

    assert _result_config_is_stale(original) is False
    assert _result_config_is_stale(changed) is True


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


def test_optional_visual_explorer_reports_renderer_failures(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "pygwalker", types.ModuleType("pygwalker"))
    monkeypatch.setattr(st, "checkbox", lambda _label: True)
    errors: list[str] = []
    monkeypatch.setattr(st, "error", errors.append)

    def broken_renderer(_dataset_key: str, _df: pd.DataFrame, _spec_path: str) -> object:
        raise RuntimeError("renderer failed")

    monkeypatch.setattr("eda_copilot.app.optional_visual_explorer._pygwalker_renderer", broken_renderer)

    render_optional_pygwalker(pd.DataFrame({"x": [1, 2]}), "demo")

    assert errors
    assert "renderer failed" in errors[0]
