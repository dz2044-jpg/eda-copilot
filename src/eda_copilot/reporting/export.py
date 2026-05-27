from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.reporting.report_builder import build_basic_html_report
from eda_copilot.utils.serialization import to_jsonable


def export_run_artifacts(
    evidence_packet: dict[str, Any],
    markdown_report: str,
    config: EDAConfig,
    output_base_dir: Path = Path("artifacts/runs"),
) -> Path:
    """Persist standardized run artifacts and return the run directory."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_slug = _slugify(config.dataset_name)
    run_dir = output_base_dir / f"{timestamp}_{dataset_slug}"
    run_dir.mkdir(parents=True, exist_ok=False)

    _write_json(run_dir / "input_manifest.json", _input_manifest(evidence_packet))
    _write_json(run_dir / "config_snapshot.json", config.to_dict())
    _write_json(run_dir / "evidence_packet.json", evidence_packet)
    _write_csv(run_dir / "data_quality_warnings.csv", evidence_packet.get("data_quality_warnings", []))
    _write_csv(run_dir / "feature_ranking.csv", evidence_packet.get("feature_ranking", []))
    (run_dir / "eda_report.md").write_text(markdown_report, encoding="utf-8")
    (run_dir / "eda_report.html").write_text(build_basic_html_report(markdown_report), encoding="utf-8")
    (run_dir / "audit_log.jsonl").write_text(
        json.dumps(
            to_jsonable(
                {
                    "event": "eda_run_exported",
                    "created_at": datetime.now().isoformat(),
                    "dataset_name": config.dataset_name,
                }
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "plots").mkdir(exist_ok=True)
    return run_dir


def _input_manifest(evidence_packet: dict[str, Any]) -> dict[str, Any]:
    overview = evidence_packet.get("dataset_overview", {})
    return {
        "dataset_name": overview.get("dataset_name"),
        "row_count": overview.get("row_count"),
        "column_count": overview.get("column_count"),
        "memory_usage_bytes": overview.get("memory_usage_bytes"),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    pd.DataFrame(records).to_csv(path, index=False)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "dataset"
