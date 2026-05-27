from pathlib import Path

import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.core.workflow import run_eda


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
    assert result.evidence_packet["dataset_overview"]["row_count"] == 4
