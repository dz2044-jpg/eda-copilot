from __future__ import annotations

from pathlib import Path

import pandas as pd

from eda_copilot.core.config import EDAConfig, validate_config_against_dataframe
from eda_copilot.core.state import EDAResult
from eda_copilot.eda.bivariate import analyze_bivariate
from eda_copilot.eda.comparison import build_comparison_summary
from eda_copilot.eda.data_quality import detect_data_quality_issues
from eda_copilot.eda.drift import analyze_drift
from eda_copilot.eda.feature_ranking import build_feature_ranking
from eda_copilot.eda.leakage import detect_leakage_risks
from eda_copilot.eda.missingness import analyze_missingness
from eda_copilot.eda.modeling_risk import build_modeling_risk_summary
from eda_copilot.eda.overview import build_dataset_overview
from eda_copilot.eda.profiling import build_profile_summary
from eda_copilot.eda.quality_checks import build_quality_checks
from eda_copilot.eda.response_analysis import analyze_response
from eda_copilot.eda.type_inference import infer_column_types
from eda_copilot.eda.univariate import analyze_univariate
from eda_copilot.reporting.evidence_packet import build_evidence_packet
from eda_copilot.reporting.export import export_run_artifacts
from eda_copilot.reporting.report_builder import build_markdown_report
from eda_copilot.visualization.specs import build_visual_specs


def run_eda(
    df: pd.DataFrame,
    config: EDAConfig,
    *,
    export_artifacts: bool = False,
    output_base_dir: Path = Path("artifacts/runs"),
) -> EDAResult:
    """Execute the deterministic EDA workflow.

    Args:
        df: Input dataframe.
        config: EDA run configuration.
        export_artifacts: Whether to save standardized run artifacts.
        output_base_dir: Base folder for run artifacts.

    Returns:
        EDAResult containing the evidence packet, Markdown report, and optional artifact path.
    """

    validate_config_against_dataframe(df, config)
    working_df = df.copy()

    type_summary = infer_column_types(working_df, config)
    overview = build_dataset_overview(working_df, config, type_summary)
    missingness = analyze_missingness(working_df, config)
    univariate = analyze_univariate(working_df, config, type_summary)
    response = analyze_response(working_df, config, type_summary)
    bivariate = (
        _minimal_bivariate_summary()
        if config.profile_depth == "minimal"
        else analyze_bivariate(working_df, config, type_summary)
    )
    data_quality_warnings = detect_data_quality_issues(
        working_df,
        config,
        type_summary,
        missingness,
    )
    leakage_warnings = detect_leakage_risks(config, type_summary, response)
    feature_ranking = build_feature_ranking(
        config,
        type_summary,
        missingness,
        response,
        data_quality_warnings,
        leakage_warnings,
    )
    drift = analyze_drift(working_df, config, type_summary)
    comparison = build_comparison_summary(working_df, config, type_summary)
    modeling_risk = build_modeling_risk_summary(
        config,
        type_summary,
        missingness,
        bivariate,
        feature_ranking,
        data_quality_warnings,
        leakage_warnings,
        drift,
        response,
    )
    profile_summary = build_profile_summary(
        working_df,
        config,
        type_summary,
        missingness,
        univariate,
        bivariate,
        data_quality_warnings,
        leakage_warnings,
    )
    quality_checks = build_quality_checks(
        config,
        overview,
        response,
        missingness,
        bivariate,
        data_quality_warnings,
        leakage_warnings,
        drift,
        modeling_risk,
    )
    visual_specs = build_visual_specs(
        config,
        missingness,
        response,
        bivariate,
        feature_ranking,
        drift,
    )
    caveats = _build_caveats(config, response)

    evidence_packet = build_evidence_packet(
        config=config,
        dataset_overview=overview,
        profile_summary=profile_summary,
        column_type_summary=type_summary,
        missingness_summary=missingness,
        univariate_summary=univariate,
        response_summary=response,
        bivariate_summary=bivariate,
        feature_ranking=feature_ranking,
        data_quality_warnings=data_quality_warnings,
        leakage_warnings=leakage_warnings,
        drift_summary=drift,
        modeling_risk_summary=modeling_risk,
        quality_checks=quality_checks,
        comparison_summary=comparison,
        visual_specs=visual_specs,
        caveats=caveats,
    )
    markdown_report = build_markdown_report(evidence_packet)
    artifact_dir = None
    if export_artifacts:
        artifact_dir = export_run_artifacts(
            evidence_packet=evidence_packet,
            markdown_report=markdown_report,
            config=config,
            output_base_dir=output_base_dir,
        )
    return EDAResult(
        evidence_packet=evidence_packet,
        markdown_report=markdown_report,
        artifact_dir=artifact_dir,
    )


def _build_caveats(config: EDAConfig, response: dict[str, object]) -> list[str]:
    caveats = [
        "All metrics in this report were calculated deterministically from the input dataframe.",
        "Business interpretation requires confirmation of data lineage, timing, and project context.",
    ]
    if config.problem_type == "auto" and response.get("problem_type") == "binary_classification":
        caveats.append("Binary positive class was selected deterministically; confirm it matches the project definition.")
    if not response.get("available"):
        caveats.append("Response-aware sections were skipped because no usable response variable was selected.")
    if config.sample_policy == "redacted":
        caveats.append("Configured ID, sensitive, and ID-like sample values are redacted in the evidence packet.")
    if config.profile_depth == "minimal":
        caveats.append("Minimal profiling skips correlation matrix generation and text/date deep summaries.")
    caveats.append("Modeling risk signals are deterministic review aids and do not approve a model for use.")
    return caveats


def _minimal_bivariate_summary() -> dict[str, object]:
    return {
        "numeric_correlation_matrix": [],
        "high_correlation_pairs": [],
        "possible_duplicate_columns": [],
        "skipped": "profile_depth=minimal",
    }
