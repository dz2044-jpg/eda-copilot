from pathlib import Path
import json

import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.core.workflow import run_eda
from eda_copilot.reporting.validation import validate_artifact_manifest


def test_workflow_builds_evidence_packet_and_exports(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "application_id": ["A1", "A2", "A3", "A4"],
            "age": [25, 35, 45, 55],
            "target": [0, 0, 1, 1],
            "split": ["train", "train", "test", "test"],
        }
    )
    config = EDAConfig(
        dataset_name="unit_test",
        response_column="target",
        id_columns=("application_id",),
        train_test_column="split",
    )

    result = run_eda(df, config, export_artifacts=True, output_base_dir=tmp_path)

    assert result.artifact_dir is not None
    assert (result.artifact_dir / "evidence_packet.json").exists()
    assert (result.artifact_dir / "eda_report.md").exists()
    assert (result.artifact_dir / "quality_checks.csv").exists()
    assert (result.artifact_dir / "profile_alerts.csv").exists()
    assert (result.artifact_dir / "visual_specs.json").exists()
    assert (result.artifact_dir / "run_metadata.json").exists()
    assert (result.artifact_dir / "artifact_manifest.json").exists()
    assert result.evidence_packet["dataset_overview"]["row_count"] == 4
    assert "profile_summary" in result.evidence_packet
    assert "quality_checks" in result.evidence_packet
    assert "modeling_risk_summary" in result.evidence_packet
    assert "comparison_summary" in result.evidence_packet
    assert "visual_specs" in result.evidence_packet

    manifest = json.loads((result.artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert validate_artifact_manifest(manifest, result.artifact_dir) == []
