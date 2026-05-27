from __future__ import annotations

from typing import Any


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
    return [f"Unknown referenced column: {column}" for column in unknown]
