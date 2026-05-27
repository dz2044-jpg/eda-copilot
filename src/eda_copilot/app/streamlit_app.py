from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from eda_copilot.ai.summarizer import deterministic_summary_placeholder
from eda_copilot.core.config import EDAConfig, EDAValidationError
from eda_copilot.core.workflow import run_eda
from eda_copilot.reporting.export import export_run_artifacts
from eda_copilot.visualization.gallery import build_plot_gallery


def run_app() -> None:
    """Run the Streamlit application."""

    st.set_page_config(page_title="EDA Copilot", layout="wide")
    st.title("EDA Copilot")

    df, dataset_name = _load_dataset_ui()
    if df is None:
        st.info("Upload a CSV or Parquet file, or load the sample dataset.")
        return

    config = _configuration_ui(df, dataset_name)
    if st.button("Run EDA", type="primary"):
        try:
            result = run_eda(df, config)
        except EDAValidationError as exc:
            st.error(str(exc))
            return
        st.session_state["eda_result"] = result
        st.session_state["eda_config"] = config

    result = st.session_state.get("eda_result")
    if not result:
        return

    evidence = result.evidence_packet
    tabs = st.tabs(
        [
            "Overview",
            "Missingness",
            "Univariate",
            "Response",
            "Bivariate",
            "Quality",
            "Ranking",
            "AI Summary",
            "Export",
        ]
    )
    with tabs[0]:
        _overview_tab(evidence)
    with tabs[1]:
        _table_and_plot_tab(evidence, "missingness_summary", "columns", "missingness_bar")
    with tabs[2]:
        _univariate_tab(evidence)
    with tabs[3]:
        _response_tab(evidence)
    with tabs[4]:
        _bivariate_tab(evidence)
    with tabs[5]:
        _render_dataframe(evidence.get("data_quality_warnings", []))
        _render_dataframe(evidence.get("leakage_warnings", []))
    with tabs[6]:
        _render_dataframe(evidence.get("feature_ranking", []))
        _plot_if_available(evidence, "feature_ranking")
    with tabs[7]:
        st.json(deterministic_summary_placeholder(evidence))
    with tabs[8]:
        _export_tab(result.markdown_report, evidence, st.session_state["eda_config"])


def _load_dataset_ui() -> tuple[pd.DataFrame | None, str]:
    uploaded = st.sidebar.file_uploader("Dataset", type=["csv", "parquet"])
    if st.sidebar.button("Load sample dataset"):
        st.session_state["loaded_df"] = _sample_dataset()
        st.session_state["loaded_dataset_name"] = "sample_credit"
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix == ".csv":
            st.session_state["loaded_df"] = pd.read_csv(uploaded)
            st.session_state["loaded_dataset_name"] = Path(uploaded.name).stem
        elif suffix == ".parquet":
            st.session_state["loaded_df"] = pd.read_parquet(uploaded)
            st.session_state["loaded_dataset_name"] = Path(uploaded.name).stem
        else:
            st.sidebar.error("Unsupported file type.")
            return None, "dataset"
    if "loaded_df" not in st.session_state:
        return None, "dataset"
    return st.session_state["loaded_df"], st.session_state.get("loaded_dataset_name", "dataset")


def _configuration_ui(df: pd.DataFrame, dataset_name: str) -> EDAConfig:
    columns = list(df.columns)
    st.sidebar.header("Configuration")
    response_options = ["<none>"] + columns
    default_response = _default_response_column(columns)
    response_column = st.sidebar.selectbox(
        "Response column",
        response_options,
        index=response_options.index(default_response),
    )
    problem_type = st.sidebar.selectbox(
        "Problem type",
        ["auto", "binary_classification", "multiclass_classification", "regression", "unsupervised"],
    )
    id_columns = st.sidebar.multiselect("ID columns", columns)
    date_columns = st.sidebar.multiselect("Date columns", columns)
    train_test_options = ["<none>"] + columns
    train_test_column = st.sidebar.selectbox("Train/test column", train_test_options)
    segment_columns = st.sidebar.multiselect("Segment columns", columns)
    exclude_columns = st.sidebar.multiselect("Exclude columns", columns)

    return EDAConfig(
        dataset_name=dataset_name,
        response_column=None if response_column == "<none>" else response_column,
        problem_type=problem_type,
        id_columns=tuple(id_columns),
        date_columns=tuple(date_columns),
        train_test_column=None if train_test_column == "<none>" else train_test_column,
        segment_columns=tuple(segment_columns),
        exclude_columns=tuple(exclude_columns),
    )


def _default_response_column(columns: list[str]) -> str:
    target_names = {"target", "response", "label", "outcome", "approved"}
    for column in columns:
        if column.lower() in target_names:
            return column
    return "<none>"


def _overview_tab(evidence: dict[str, Any]) -> None:
    overview = evidence["dataset_overview"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{overview['row_count']:,}")
    col2.metric("Columns", f"{overview['column_count']:,}")
    col3.metric("Duplicate rows", f"{overview['duplicate_row_count']:,}")
    col4.metric("Warnings", f"{len(evidence.get('data_quality_warnings', [])):,}")
    st.subheader("Column Types")
    st.json(overview["column_type_summary"])
    st.subheader("Data Dictionary Draft")
    _render_dataframe(overview["data_dictionary"])
    st.subheader("Sample Rows")
    _render_dataframe(overview["sample_rows"])


def _table_and_plot_tab(evidence: dict[str, Any], section: str, table_key: str, plot_key: str) -> None:
    _plot_if_available(evidence, plot_key)
    _render_dataframe(evidence.get(section, {}).get(table_key, []))


def _univariate_tab(evidence: dict[str, Any]) -> None:
    univariate = evidence.get("univariate_summary", {})
    st.subheader("Numeric")
    _render_dataframe(univariate.get("numeric", []))
    st.subheader("Categorical")
    _render_dataframe(univariate.get("categorical", []))
    st.subheader("Datetime")
    _render_dataframe(univariate.get("datetime", []))


def _response_tab(evidence: dict[str, Any]) -> None:
    response = evidence.get("response_summary", {})
    _plot_if_available(evidence, "target_distribution")
    st.json(response)


def _bivariate_tab(evidence: dict[str, Any]) -> None:
    _plot_if_available(evidence, "correlation_heatmap")
    st.subheader("High Correlations")
    _render_dataframe(evidence.get("bivariate_summary", {}).get("high_correlation_pairs", []))
    st.subheader("Drift")
    st.json(evidence.get("drift_summary", {}))


def _plot_if_available(evidence: dict[str, Any], plot_key: str) -> None:
    figures = build_plot_gallery(evidence)
    figure = figures.get(plot_key)
    if figure is not None:
        st.plotly_chart(figure, width="stretch")


def _render_dataframe(records: Any) -> None:
    frame = pd.DataFrame(records)
    if not frame.empty:
        for column in frame.columns:
            if frame[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
                frame[column] = frame[column].map(
                    lambda value: json.dumps(value)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
    st.dataframe(frame, width="stretch")


def _export_tab(markdown_report: str, evidence: dict[str, Any], config: EDAConfig) -> None:
    st.download_button("Download Markdown report", markdown_report, file_name="eda_report.md")
    st.download_button(
        "Download evidence packet",
        json.dumps(evidence, indent=2),
        file_name="evidence_packet.json",
        mime="application/json",
    )
    if st.button("Save artifacts to workspace"):
        run_dir = export_run_artifacts(evidence, markdown_report, config)
        st.success(f"Saved artifacts to {run_dir}")
    st.subheader("Report Preview")
    st.markdown(markdown_report)


def _sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "application_id": [f"A{i:04d}" for i in range(1, 81)],
            "age": [22, 35, 44, 51, 29, 62, 38, 47] * 10,
            "income": [45000, 72000, 88000, 91000, None, 120000, 64000, 77000] * 10,
            "risk_class": ["standard", "preferred", "standard", "substandard", "preferred", "preferred", "standard", "substandard"] * 10,
            "channel": ["agent", "web", "agent", "partner", "web", "agent", "partner", "web"] * 10,
            "final_decision_score": [0.1, 0.9, 0.2, 0.85, 0.7, 0.95, 0.3, 0.8] * 10,
            "approved": [0, 1, 0, 1, 1, 1, 0, 1] * 10,
            "split": ["train"] * 60 + ["test"] * 20,
        }
    )
