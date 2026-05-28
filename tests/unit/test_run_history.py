from pathlib import Path

import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.core.workflow import run_eda
from eda_copilot.reporting.export import export_run_artifacts
from eda_copilot.reporting.run_history import compare_evidence_packets, discover_run_history


def test_discover_run_history_reports_incomplete_or_corrupt_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_run"
    run_dir.mkdir()
    (run_dir / "artifact_manifest.json").write_text("{bad json", encoding="utf-8")

    runs = discover_run_history(tmp_path)

    assert runs[0]["status"] == "incomplete"
    assert any("Missing run_metadata.json" in error for error in runs[0]["errors"])
    assert any("Invalid JSON in artifact_manifest.json" in error for error in runs[0]["errors"])


def test_compare_evidence_packets_reports_shape_quality_and_feature_changes() -> None:
    left = run_eda(
        pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 0, 1, 1]}),
        EDAConfig(response_column="target"),
    ).evidence_packet
    right = run_eda(
        pd.DataFrame({"x": [1, 2, 3, 4, 5], "z": [5, 4, 3, 2, 1], "target": [0, 0, 1, 1, 1]}),
        EDAConfig(response_column="target"),
    ).evidence_packet

    comparison = compare_evidence_packets(left, right)

    assert comparison["available"] is True
    assert comparison["row_count_delta"]["delta"] == 1
    assert comparison["column_count_delta"]["delta"] == 1
    assert "top_feature_changes" in comparison


def test_discover_run_history_reads_exported_metadata(tmp_path: Path) -> None:
    result = run_eda(
        pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 0, 1, 1]}),
        EDAConfig(dataset_name="history_test", response_column="target"),
    )
    export_run_artifacts(result.evidence_packet, result.markdown_report, EDAConfig(dataset_name="history_test", response_column="target"), tmp_path)

    runs = discover_run_history(tmp_path)

    assert runs[0]["status"] == "available"
    assert runs[0]["dataset_name"] == "history_test"
    assert runs[0]["artifact_count"] is not None


def test_export_run_artifacts_creates_unique_directories_for_repeated_runs(tmp_path: Path) -> None:
    result = run_eda(
        pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 0, 1, 1]}),
        EDAConfig(dataset_name="collision_test", response_column="target"),
    )
    config = EDAConfig(dataset_name="collision_test", response_column="target")

    first = export_run_artifacts(result.evidence_packet, result.markdown_report, config, tmp_path)
    second = export_run_artifacts(result.evidence_packet, result.markdown_report, config, tmp_path)

    assert first != second
    assert first.exists()
    assert second.exists()
