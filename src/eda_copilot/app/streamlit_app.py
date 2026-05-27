from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from eda_copilot.ai.query_planner import plan_evidence_question
from eda_copilot.ai.summarizer import deterministic_summary_placeholder
from eda_copilot.core.config import EDAConfig, EDAValidationError
from eda_copilot.core.data_loading import load_tabular_dataset
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
            "Profile",
            "Missingness",
            "Univariate",
            "Response",
            "Bivariate",
            "Quality",
            "Ranking",
            "Visual Explorer",
            "Ask",
            "AI Summary",
            "Export",
        ]
    )
    with tabs[0]:
        _overview_tab(evidence)
    with tabs[1]:
        _profile_tab(evidence)
    with tabs[2]:
        _table_and_plot_tab(evidence, "missingness_summary", "columns", "missingness_bar")
    with tabs[3]:
        _univariate_tab(evidence)
    with tabs[4]:
        _response_tab(evidence)
    with tabs[5]:
        _bivariate_tab(evidence)
    with tabs[6]:
        _quality_tab(evidence)
    with tabs[7]:
        _render_dataframe(evidence.get("feature_ranking", []))
        _plot_if_available(evidence, "feature_ranking")
    with tabs[8]:
        _visual_explorer_tab(evidence, df, dataset_name)
    with tabs[9]:
        _ask_tab(evidence)
    with tabs[10]:
        st.json(deterministic_summary_placeholder(evidence))
    with tabs[11]:
        _export_tab(result.markdown_report, evidence, st.session_state["eda_config"])


def _load_dataset_ui() -> tuple[pd.DataFrame | None, str]:
    uploaded = st.sidebar.file_uploader("Dataset", type=["csv", "parquet"])
    if st.sidebar.button("Load sample dataset"):
        st.session_state["loaded_df"] = _sample_dataset()
        st.session_state["loaded_dataset_name"] = "sample_credit"
    if uploaded is not None:
        try:
            loaded = load_tabular_dataset(uploaded, uploaded.name)
        except EDAValidationError as exc:
            st.sidebar.error(str(exc))
            return None, "dataset"
        st.session_state["loaded_df"] = loaded.dataframe
        st.session_state["loaded_dataset_name"] = loaded.dataset_name
    if "loaded_df" not in st.session_state:
        return None, "dataset"
    df = st.session_state["loaded_df"]
    st.sidebar.caption(f"Loaded {len(df):,} rows x {len(df.columns):,} columns")
    return df, st.session_state.get("loaded_dataset_name", "dataset")


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
    sensitive_columns = st.sidebar.multiselect("Sensitive columns", columns)
    exclude_columns = st.sidebar.multiselect("Exclude columns", columns)
    profile_depth = st.sidebar.selectbox("Profile depth", ["minimal", "standard", "deep"], index=1)
    sample_policy = st.sidebar.selectbox("Sample policy", ["redacted", "preview", "none"])
    drift_warning_threshold = st.sidebar.number_input(
        "Drift warning threshold",
        min_value=0.0,
        max_value=10.0,
        value=0.20,
        step=0.05,
    )
    drift_fail_threshold = st.sidebar.number_input(
        "Drift fail threshold",
        min_value=drift_warning_threshold,
        max_value=10.0,
        value=max(0.50, drift_warning_threshold),
        step=0.05,
    )

    return EDAConfig(
        dataset_name=dataset_name,
        response_column=None if response_column == "<none>" else response_column,
        problem_type=problem_type,
        id_columns=tuple(id_columns),
        date_columns=tuple(date_columns),
        train_test_column=None if train_test_column == "<none>" else train_test_column,
        segment_columns=tuple(segment_columns),
        sensitive_columns=tuple(sensitive_columns),
        exclude_columns=tuple(exclude_columns),
        profile_depth=profile_depth,
        sample_policy=sample_policy,
        drift_warning_threshold=drift_warning_threshold,
        drift_fail_threshold=drift_fail_threshold,
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
    col4.metric("Quality gate", evidence.get("quality_checks", {}).get("overall_status", "unknown"))
    st.subheader("Column Types")
    st.json(overview["column_type_summary"])
    st.subheader("Data Dictionary Draft")
    _render_dataframe(overview["data_dictionary"])
    st.subheader("Sample Rows")
    _render_dataframe(overview["sample_rows"])


def _profile_tab(evidence: dict[str, Any]) -> None:
    profile = evidence.get("profile_summary", {})
    st.subheader("Profile Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Profile depth", str(profile.get("profile_depth", "unknown")))
    col2.metric("Alerts", f"{profile.get('alert_summary', {}).get('total', 0):,}")
    col3.metric("Sample policy", str(profile.get("sample_policy", "unknown")))

    st.subheader("Column Explorer")
    dictionary = evidence.get("dataset_overview", {}).get("data_dictionary", [])
    type_options = ["<all>"] + sorted(
        {
            str(row.get("semantic_type"))
            for row in dictionary
            if row.get("semantic_type") is not None
        }
    )
    search = st.text_input("Column search", value="")
    semantic_type = st.selectbox("Semantic type", type_options)
    filtered = []
    for row in dictionary:
        column_name = str(row.get("column", ""))
        if search and search.lower() not in column_name.lower():
            continue
        if semantic_type != "<all>" and row.get("semantic_type") != semantic_type:
            continue
        filtered.append(row)
    _render_dataframe(filtered)

    st.subheader("Alerts")
    _render_dataframe(profile.get("alerts", []))
    st.subheader("Text Summary")
    _render_dataframe(profile.get("text_summary", []))
    st.subheader("Datetime Summary")
    _render_dataframe(profile.get("datetime_summary", []))


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
    _plot_if_available(evidence, "drift_shift_bar")
    st.subheader("High Correlations")
    _render_dataframe(evidence.get("bivariate_summary", {}).get("high_correlation_pairs", []))
    st.subheader("Drift")
    st.json(evidence.get("drift_summary", {}))
    st.subheader("Reference/Current Comparison")
    st.json(evidence.get("comparison_summary", {}))


def _quality_tab(evidence: dict[str, Any]) -> None:
    _plot_if_available(evidence, "quality_check_status")
    st.subheader("Quality Checks")
    _render_dataframe(evidence.get("quality_checks", {}).get("checks", []))
    st.subheader("Data Quality Warnings")
    _render_dataframe(evidence.get("data_quality_warnings", []))
    st.subheader("Leakage Warnings")
    _render_dataframe(evidence.get("leakage_warnings", []))


def _visual_explorer_tab(evidence: dict[str, Any], df: pd.DataFrame, dataset_name: str) -> None:
    st.subheader("Saved Visual Specs")
    specs = evidence.get("visual_specs", [])
    _render_dataframe(specs)
    figures = build_plot_gallery(evidence)
    if figures:
        selected = st.selectbox("Chart", sorted(figures.keys()))
        st.plotly_chart(figures[selected], width="stretch", key=f"visual_{selected}")
    _render_optional_pygwalker(df, dataset_name)


def _ask_tab(evidence: dict[str, Any]) -> None:
    with st.form("evidence_question_form"):
        question = st.text_input("Evidence question", value="")
        submitted = st.form_submit_button("Plan answer")
    if submitted and question:
        st.json(plan_evidence_question(question, evidence))


def _plot_if_available(evidence: dict[str, Any], plot_key: str) -> None:
    figures = build_plot_gallery(evidence)
    figure = figures.get(plot_key)
    if figure is not None:
        st.plotly_chart(figure, width="stretch", key=f"plot_{plot_key}")


def _render_optional_pygwalker(df: pd.DataFrame, dataset_name: str) -> None:
    try:
        import pygwalker  # noqa: F401
    except ImportError:
        st.info("Optional PyGWalker explorer is not installed.")
        return

    if not st.checkbox("Open PyGWalker explorer"):
        return

    spec_dir = Path("artifacts/runs/visual_specs")
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{_dataset_key(df, dataset_name)}.json"
    renderer = _pygwalker_renderer(_dataset_key(df, dataset_name), df, str(spec_path))
    renderer.explorer()


@st.cache_resource(show_spinner=False)
def _pygwalker_renderer(dataset_key: str, _df: pd.DataFrame, spec_path: str):
    from pygwalker.api.streamlit import StreamlitRenderer

    return StreamlitRenderer(_df, spec=spec_path, spec_io_mode="rw")


def _dataset_key(df: pd.DataFrame, dataset_name: str) -> str:
    columns = "_".join(str(column) for column in df.columns[:20])
    digest = hashlib.sha256(columns.encode("utf-8")).hexdigest()[:12]
    return f"{dataset_name}_{len(df)}x{len(df.columns)}_{digest}"


def _render_dataframe(records: Any) -> None:
    frame = pd.DataFrame(records)
    if not frame.empty:
        for column in frame.columns:
            has_nested_values = frame[column].map(lambda value: isinstance(value, (dict, list, tuple))).any()
            mixed_object_values = (
                frame[column].dtype == "object"
                and frame[column].dropna().map(lambda value: type(value).__name__).nunique() > 1
            )
            if has_nested_values or mixed_object_values:
                frame[column] = frame[column].map(_display_cell)
    st.dataframe(frame, width="stretch")


def _display_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    if value is None:
        return value
    return str(value)


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
