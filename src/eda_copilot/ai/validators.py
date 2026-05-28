from __future__ import annotations

from typing import Any


ALLOWED_SUMMARY_EVIDENCE_SECTIONS = {
    "metadata",
    "dataset_overview",
    "profile_summary",
    "response_summary",
    "column_type_summary",
    "missingness_summary",
    "feature_ranking",
    "data_quality_warnings",
    "leakage_warnings",
    "drift_summary",
    "modeling_risk_summary",
    "quality_checks",
    "comparison_summary",
    "visual_specs",
    "caveats",
}

RAW_ROW_KEYS = {"sample_rows", "sample_values", "raw_rows", "row_records", "top_terms"}


def validate_summary_references(summary: dict[str, Any], evidence_packet: dict[str, Any]) -> list[str]:
    """Validate that an AI summary references known columns only.

    This lightweight guard is intended for future LLM integration. It cannot prove
    factual correctness, but it catches obvious invented column names.
    """

    known_columns = {
        column["name"]
        for column in evidence_packet.get("column_type_summary", {}).get("columns", [])
    }
    referenced = set(summary.get("referenced_columns", []))
    unknown = sorted(referenced - known_columns)
    errors = [f"Unknown referenced column: {column}" for column in unknown]

    referenced_sections = set(summary.get("referenced_evidence_sections", []))
    unknown_sections = sorted(referenced_sections - ALLOWED_SUMMARY_EVIDENCE_SECTIONS)
    errors.extend(f"Unknown or disallowed evidence section: {section}" for section in unknown_sections)
    errors.extend(_raw_row_field_errors(summary))
    return errors


def _raw_row_field_errors(payload: Any, path: str = "summary") -> list[str]:
    errors = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{path}.{key}"
            if key in RAW_ROW_KEYS:
                errors.append(f"AI summary must not include raw row field: {child_path}.")
            errors.extend(_raw_row_field_errors(value, child_path))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            errors.extend(_raw_row_field_errors(item, f"{path}[{index}]"))
    return errors
