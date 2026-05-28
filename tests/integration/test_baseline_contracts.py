from __future__ import annotations

from pathlib import Path

import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.core.workflow import run_eda


REQUIRED_EVIDENCE_PACKET_KEYS = {
    "metadata",
    "config",
    "dataset_overview",
    "profile_summary",
    "response_summary",
    "column_type_summary",
    "missingness_summary",
    "univariate_summary",
    "bivariate_summary",
    "feature_ranking",
    "data_quality_warnings",
    "leakage_warnings",
    "drift_summary",
    "modeling_risk_summary",
    "quality_checks",
    "comparison_summary",
    "visual_specs",
    "plots",
    "artifacts",
    "external_artifacts",
    "caveats",
}

REQUIRED_EXPORTED_ARTIFACTS = {
    "input_manifest.json",
    "config_snapshot.json",
    "evidence_packet.json",
    "comparison_summary.json",
    "visual_specs.json",
    "modeling_risk_summary.json",
    "run_metadata.json",
    "artifact_manifest.json",
    "data_quality_warnings.csv",
    "profile_alerts.csv",
    "quality_checks.csv",
    "feature_ranking.csv",
    "eda_report.md",
    "eda_report.html",
    "audit_log.jsonl",
    "plots",
}


def test_evidence_packet_required_sections_remain_available() -> None:
    result = run_eda(_sample_dataframe(), _sample_config())

    assert REQUIRED_EVIDENCE_PACKET_KEYS.issubset(result.evidence_packet)
    assert result.evidence_packet["metadata"]["tool_name"] == "eda_copilot"
    assert result.evidence_packet["config"]["dataset_name"] == "baseline_contract"
    assert isinstance(result.evidence_packet["visual_specs"], list)


def test_exported_artifact_names_remain_available(tmp_path: Path) -> None:
    result = run_eda(
        _sample_dataframe(),
        _sample_config(),
        export_artifacts=True,
        output_base_dir=tmp_path,
    )

    assert result.artifact_dir is not None
    exported_names = {path.name for path in result.artifact_dir.iterdir()}
    assert REQUIRED_EXPORTED_ARTIFACTS.issubset(exported_names)


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "application_id": ["A1", "A2", "A3", "A4", "A5", "A6"],
            "age": [25, 35, 45, 55, 65, 75],
            "income": [50_000, 72_000, 91_000, 110_000, 98_000, 120_000],
            "target": [0, 0, 1, 1, 0, 1],
            "split": ["train", "train", "train", "test", "test", "test"],
        }
    )


def _sample_config() -> EDAConfig:
    return EDAConfig(
        dataset_name="baseline_contract",
        response_column="target",
        id_columns=("application_id",),
        train_test_column="split",
    )
