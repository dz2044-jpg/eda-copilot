from __future__ import annotations

from pathlib import Path
from typing import Any


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

EXPECTED_SECTION_TYPES: dict[str, type] = {
    "metadata": dict,
    "config": dict,
    "dataset_overview": dict,
    "profile_summary": dict,
    "response_summary": dict,
    "column_type_summary": dict,
    "missingness_summary": dict,
    "univariate_summary": dict,
    "bivariate_summary": dict,
    "feature_ranking": list,
    "data_quality_warnings": list,
    "leakage_warnings": list,
    "drift_summary": dict,
    "modeling_risk_summary": dict,
    "quality_checks": dict,
    "comparison_summary": dict,
    "visual_specs": list,
    "plots": list,
    "artifacts": list,
    "external_artifacts": list,
    "caveats": list,
}

REQUIRED_MANIFEST_FILE_KEYS = {
    "file_name",
    "kind",
    "relative_path",
    "size_bytes",
    "sha256",
    "created_at_utc",
}


def validate_evidence_packet(evidence_packet: dict[str, Any]) -> list[str]:
    """Return contract validation errors for an evidence packet."""

    errors = []
    missing = sorted(REQUIRED_EVIDENCE_PACKET_KEYS - set(evidence_packet))
    if missing:
        errors.append(f"Evidence packet is missing required sections: {', '.join(missing)}.")

    for key, expected_type in EXPECTED_SECTION_TYPES.items():
        if key in evidence_packet and not isinstance(evidence_packet[key], expected_type):
            errors.append(
                f"Evidence packet section {key!r} must be {expected_type.__name__}, "
                f"got {type(evidence_packet[key]).__name__}."
            )
    return errors


def validate_ai_context_sanitized(context: dict[str, Any]) -> list[str]:
    """Return errors if an AI-facing context contains row-level sample fields."""

    errors = []
    overview = context.get("dataset_overview")
    if not isinstance(overview, dict):
        return ["AI context dataset_overview must be an object."]
    if "sample_rows" in overview:
        errors.append("AI context must not include dataset_overview.sample_rows.")
    for row in overview.get("data_dictionary", []):
        if isinstance(row, dict) and "sample_values" in row:
            column = row.get("column", "<unknown>")
            errors.append(f"AI context data_dictionary for {column} must not include sample_values.")
    return errors


def validate_artifact_manifest(manifest: dict[str, Any], run_dir: Path | None = None) -> list[str]:
    """Return validation errors for an exported artifact manifest."""

    errors = []
    files = manifest.get("files")
    if not isinstance(files, list):
        return ["Artifact manifest must include a files list."]

    for index, item in enumerate(files):
        if not isinstance(item, dict):
            errors.append(f"Artifact manifest files[{index}] must be an object.")
            continue
        missing = sorted(REQUIRED_MANIFEST_FILE_KEYS - set(item))
        if missing:
            errors.append(f"Artifact manifest files[{index}] is missing: {', '.join(missing)}.")
        if run_dir is not None and item.get("relative_path"):
            path = run_dir / str(item["relative_path"])
            if not path.exists():
                errors.append(f"Artifact manifest references missing file: {item['relative_path']}.")
    return errors
