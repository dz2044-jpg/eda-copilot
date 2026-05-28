from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from eda_copilot import __version__
from eda_copilot.core.config import EDAConfig
from eda_copilot.reporting.report_builder import build_basic_html_report
from eda_copilot.utils.serialization import to_jsonable


def export_run_artifacts(
    evidence_packet: dict[str, Any],
    markdown_report: str,
    config: EDAConfig,
    output_base_dir: Path = Path("artifacts/runs"),
    ai_summary: dict[str, Any] | None = None,
    run_comparison_summary: dict[str, Any] | None = None,
) -> Path:
    """Persist standardized run artifacts and return the run directory."""

    created_at_utc = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_slug = _slugify(config.dataset_name)
    run_dir = _unique_run_dir(output_base_dir, f"{timestamp}_{dataset_slug}")
    run_dir.mkdir(parents=True, exist_ok=False)

    run_metadata = _run_metadata(config, evidence_packet, created_at_utc)
    bundle = build_export_bundle(
        evidence_packet=evidence_packet,
        markdown_report=markdown_report,
        config=config,
        run_metadata=run_metadata,
        ai_summary=ai_summary,
        run_comparison_summary=run_comparison_summary,
    )
    kind_by_path: dict[str, str] = {}
    for item in bundle:
        relative_path = str(item["relative_path"])
        kind = str(item["kind"])
        kind_by_path[relative_path] = kind
        path = run_dir / relative_path
        if kind == "directory":
            path.mkdir(exist_ok=True)
        elif kind == "json":
            _write_json(path, item["payload"])
        elif kind == "csv":
            _write_csv(path, item["payload"])
        elif kind in {"markdown", "html", "jsonl"}:
            path.write_text(str(item["payload"]), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported export artifact kind: {kind}")

    _write_json(run_dir / "artifact_manifest.json", _artifact_manifest(run_dir, kind_by_path, created_at_utc))
    return run_dir


def build_export_bundle(
    evidence_packet: dict[str, Any],
    markdown_report: str,
    config: EDAConfig,
    run_metadata: dict[str, Any],
    ai_summary: dict[str, Any] | None = None,
    run_comparison_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the standardized artifacts for one exported EDA run."""

    bundle: list[dict[str, Any]] = [
        _bundle_item("input_manifest.json", "json", _input_manifest(evidence_packet)),
        _bundle_item("config_snapshot.json", "json", config.to_dict()),
        _bundle_item("run_metadata.json", "json", run_metadata),
        _bundle_item("evidence_packet.json", "json", evidence_packet),
        _bundle_item("comparison_summary.json", "json", evidence_packet.get("comparison_summary", {})),
        _bundle_item("visual_specs.json", "json", {"visual_specs": evidence_packet.get("visual_specs", [])}),
        _bundle_item("modeling_risk_summary.json", "json", evidence_packet.get("modeling_risk_summary", {})),
        _bundle_item("data_quality_warnings.csv", "csv", evidence_packet.get("data_quality_warnings", [])),
        _bundle_item("profile_alerts.csv", "csv", evidence_packet.get("profile_summary", {}).get("alerts", [])),
        _bundle_item("quality_checks.csv", "csv", evidence_packet.get("quality_checks", {}).get("checks", [])),
        _bundle_item("feature_ranking.csv", "csv", evidence_packet.get("feature_ranking", [])),
        _bundle_item("eda_report.md", "markdown", markdown_report),
        _bundle_item("eda_report.html", "html", build_basic_html_report(markdown_report)),
        _bundle_item(
            "audit_log.jsonl",
            "jsonl",
            json.dumps(
                to_jsonable(
                    {
                        "event": "eda_run_exported",
                        "created_at": run_metadata["created_at_utc"],
                        "run_id": run_metadata["run_id"],
                        "dataset_name": config.dataset_name,
                    }
                )
            )
            + "\n",
        ),
        _bundle_item("plots", "directory", None),
    ]
    if ai_summary is not None:
        bundle.append(_bundle_item("ai_summary.json", "json", ai_summary))
    if run_comparison_summary is not None:
        bundle.append(_bundle_item("run_comparison_summary.json", "json", run_comparison_summary))
    return bundle


def _input_manifest(evidence_packet: dict[str, Any]) -> dict[str, Any]:
    overview = evidence_packet.get("dataset_overview", {})
    return {
        "dataset_name": overview.get("dataset_name"),
        "row_count": overview.get("row_count"),
        "column_count": overview.get("column_count"),
        "memory_usage_bytes": overview.get("memory_usage_bytes"),
    }


def _run_metadata(config: EDAConfig, evidence_packet: dict[str, Any], created_at_utc: str) -> dict[str, Any]:
    config_payload = config.to_dict()
    config_hash = _sha256_bytes(json.dumps(to_jsonable(config_payload), sort_keys=True).encode("utf-8"))
    overview = evidence_packet.get("dataset_overview", {})
    run_seed = json.dumps(
        to_jsonable(
            {
                "created_at_utc": created_at_utc,
                "dataset_name": config.dataset_name,
                "config_hash": config_hash,
                "shape": [overview.get("row_count"), overview.get("column_count")],
            }
        ),
        sort_keys=True,
    )
    return {
        "run_id": _sha256_bytes(run_seed.encode("utf-8"))[:16],
        "dataset_name": config.dataset_name,
        "tool_name": "eda_copilot",
        "tool_version": __version__,
        "created_at_utc": created_at_utc,
        "config_hash": config_hash,
        "row_count": overview.get("row_count"),
        "column_count": overview.get("column_count"),
    }


def _artifact_manifest(run_dir: Path, kind_by_path: dict[str, str], created_at_utc: str) -> dict[str, Any]:
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        relative_path = path.relative_to(run_dir).as_posix()
        files.append(
            {
                "file_name": path.name,
                "kind": kind_by_path.get(relative_path, _infer_kind(path)),
                "relative_path": relative_path,
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
                "created_at_utc": created_at_utc,
            }
        )
    return {
        "created_at_utc": created_at_utc,
        "file_count": len(files),
        "files": files,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    pd.DataFrame(records).to_csv(path, index=False)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "dataset"


def _unique_run_dir(output_base_dir: Path, base_name: str) -> Path:
    candidate = output_base_dir / base_name
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        candidate = output_base_dir / f"{base_name}_{index:03d}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not create a unique run directory for {base_name}.")


def _bundle_item(relative_path: str, kind: str, payload: Any) -> dict[str, Any]:
    return {"relative_path": relative_path, "kind": kind, "payload": payload}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _infer_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".md":
        return "markdown"
    if suffix == ".html":
        return "html"
    if suffix == ".jsonl":
        return "jsonl"
    return "file"
