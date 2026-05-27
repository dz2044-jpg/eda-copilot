from __future__ import annotations

from pathlib import Path

import pandas as pd

from eda_copilot.core.config import EDAConfig, validate_config_against_dataframe
from eda_copilot.core.state import EDAResult
from eda_copilot.eda.bivariate import analyze_bivariate
from eda_copilot.eda.data_quality import detect_data_quality_issues
from eda_copilot.eda.drift import analyze_drift
from eda_copilot.eda.feature_ranking import build_feature_ranking
from eda_copilot.eda.leakage import detect_leakage_risks
from eda_copilot.eda.missingness import analyze_missingness
from eda_copilot.eda.overview import build_dataset_overview
from eda_copilot.eda.response_analysis import analyze_response
from eda_copilot.eda.type_inference import infer_column_types
from eda_copilot.eda.univariate import analyze_univariate
from eda_copilot.reporting.evidence_packet import build_evidence_packet
from eda_copilot.reporting.export import export_run_artifacts
from eda_copilot.reporting.report_builder import build_markdown_report


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
    bivariate = analyze_bivariate(working_df, config, type_summary)
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
    caveats = _build_caveats(config, response)

    evidence_packet = build_evidence_packet(
        config=config,
        dataset_overview=overview,
        column_type_summary=type_summary,
        missingness_summary=missingness,
        univariate_summary=univariate,
        response_summary=response,
        bivariate_summary=bivariate,
        feature_ranking=feature_ranking,
        data_quality_warnings=data_quality_warnings,
        leakage_warnings=leakage_warnings,
        drift_summary=drift,
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
    return caveats
