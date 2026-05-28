from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eda_copilot.utils.serialization import to_jsonable


def discover_run_history(base_dir: Path = Path("artifacts/runs")) -> list[dict[str, Any]]:
    """Discover exported run folders from metadata and artifact manifests."""

    if not base_dir.exists():
        return []

    runs = []
    for run_dir in sorted((path for path in base_dir.iterdir() if path.is_dir()), reverse=True):
        errors = []
        metadata = _read_json(run_dir / "run_metadata.json", errors)
        manifest = _read_json(run_dir / "artifact_manifest.json", errors)
        status = "available" if metadata and manifest and not errors else "incomplete"
        runs.append(
            {
                "run_dir": str(run_dir),
                "run_id": metadata.get("run_id") if isinstance(metadata, dict) else None,
                "dataset_name": metadata.get("dataset_name") if isinstance(metadata, dict) else None,
                "created_at_utc": metadata.get("created_at_utc") if isinstance(metadata, dict) else None,
                "row_count": metadata.get("row_count") if isinstance(metadata, dict) else None,
                "column_count": metadata.get("column_count") if isinstance(metadata, dict) else None,
                "artifact_count": manifest.get("file_count") if isinstance(manifest, dict) else None,
                "status": status,
                "errors": errors,
            }
        )
    return runs


def load_evidence_packet(run_dir: Path) -> dict[str, Any]:
    """Load an exported evidence packet without reading the source dataset."""

    path = run_dir / "evidence_packet.json"
    return json.loads(path.read_text(encoding="utf-8"))


def compare_run_directories(left_run_dir: Path, right_run_dir: Path) -> dict[str, Any]:
    """Compare two exported runs using saved evidence packets only."""

    return compare_evidence_packets(
        load_evidence_packet(left_run_dir),
        load_evidence_packet(right_run_dir),
        left_metadata=_safe_json(left_run_dir / "run_metadata.json"),
        right_metadata=_safe_json(right_run_dir / "run_metadata.json"),
    )


def compare_evidence_packets(
    left: dict[str, Any],
    right: dict[str, Any],
    left_metadata: dict[str, Any] | None = None,
    right_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an AI-safe deterministic comparison between two evidence packets."""

    left_overview = left.get("dataset_overview", {})
    right_overview = right.get("dataset_overview", {})
    left_quality = left.get("quality_checks", {})
    right_quality = right.get("quality_checks", {})
    left_modeling = left.get("modeling_risk_summary", {})
    right_modeling = right.get("modeling_risk_summary", {})
    left_drift = left.get("drift_summary", {})
    right_drift = right.get("drift_summary", {})

    return to_jsonable(
        {
            "available": True,
            "comparison_type": "evidence_packet_to_evidence_packet",
            "left_run": _run_identity(left, left_metadata),
            "right_run": _run_identity(right, right_metadata),
            "row_count_delta": _numeric_delta(left_overview.get("row_count"), right_overview.get("row_count")),
            "column_count_delta": _numeric_delta(left_overview.get("column_count"), right_overview.get("column_count")),
            "data_quality_warning_count_delta": _numeric_delta(
                len(left.get("data_quality_warnings", [])),
                len(right.get("data_quality_warnings", [])),
            ),
            "leakage_warning_count_delta": _numeric_delta(
                len(left.get("leakage_warnings", [])),
                len(right.get("leakage_warnings", [])),
            ),
            "quality_gate_status_change": _status_change(left_quality.get("overall_status"), right_quality.get("overall_status")),
            "drift_status_change": _status_change(left_drift.get("overall_status"), right_drift.get("overall_status")),
            "modeling_risk_status_change": _status_change(left_modeling.get("overall_status"), right_modeling.get("overall_status")),
            "top_feature_changes": _feature_rank_changes(left.get("feature_ranking", []), right.get("feature_ranking", [])),
            "caveat": "Run comparison uses saved evidence packets only and does not read raw source datasets.",
        }
    )


def _run_identity(evidence_packet: dict[str, Any], metadata: dict[str, Any] | None) -> dict[str, Any]:
    overview = evidence_packet.get("dataset_overview", {})
    packet_metadata = evidence_packet.get("metadata", {})
    return {
        "run_id": metadata.get("run_id") if metadata else None,
        "dataset_name": overview.get("dataset_name") or (metadata or {}).get("dataset_name"),
        "created_at_utc": (metadata or {}).get("created_at_utc") or packet_metadata.get("created_at_utc"),
    }


def _feature_rank_changes(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_ranks = {str(row.get("column")): int(row.get("rank") or index) for index, row in enumerate(left[:20], start=1)}
    right_ranks = {str(row.get("column")): int(row.get("rank") or index) for index, row in enumerate(right[:20], start=1)}
    left_top = set(left_ranks)
    right_top = set(right_ranks)
    common = sorted(left_top & right_top)
    return {
        "new_top_features": sorted(right_top - left_top),
        "removed_top_features": sorted(left_top - right_top),
        "rank_deltas": [
            {
                "column": column,
                "left_rank": left_ranks[column],
                "right_rank": right_ranks[column],
                "rank_delta": right_ranks[column] - left_ranks[column],
            }
            for column in common
            if left_ranks[column] != right_ranks[column]
        ][:20],
    }


def _numeric_delta(left: Any, right: Any) -> dict[str, Any]:
    if left is None or right is None:
        return {"left": left, "right": right, "delta": None}
    return {"left": left, "right": right, "delta": right - left}


def _status_change(left: Any, right: Any) -> dict[str, Any]:
    return {
        "left": left,
        "right": right,
        "changed": left != right,
    }


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"Missing {path.name}.")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON in {path.name}: {exc.msg}.")
        return {}


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
