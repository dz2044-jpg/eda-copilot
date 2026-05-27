from __future__ import annotations

from typing import Any

import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.eda.type_inference import profiles_by_name
from eda_copilot.utils.serialization import records_from_frame, to_jsonable


def build_dataset_overview(
    df: pd.DataFrame,
    config: EDAConfig,
    type_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compute high-level dataset facts for reports and UI."""

    duplicate_rows = int(df.duplicated().sum())
    duplicate_id_counts = {
        column: int(df[column].duplicated().sum())
        for column in config.id_columns
        if column in df.columns
    }
    profile_map = profiles_by_name(type_summary)
    data_dictionary = []
    for column in df.columns:
        profile = profile_map[column]
        data_dictionary.append(
            {
                "column": column,
                "pandas_dtype": profile["pandas_dtype"],
                "semantic_type": profile["semantic_type"],
                "roles": profile["roles"],
                "normalized_name": profile.get("normalized_name"),
                "missing_percentage": profile["missing_percentage"],
                "unique_count": profile["unique_count"],
                "warnings": profile.get("warnings", []),
                "schema_warnings": profile.get("schema_warnings", []),
                "name_warnings": profile.get("name_warnings", []),
                "python_type_counts": profile.get("python_type_counts", {}),
                "numeric_parse_rate": profile.get("numeric_parse_rate"),
                "datetime_parse_rate": profile.get("datetime_parse_rate"),
                "boolean_parse_rate": profile.get("boolean_parse_rate"),
                "sample_values": _sample_values_for_report(profile, config),
            }
        )

    return {
        "dataset_name": config.dataset_name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "duplicate_row_count": duplicate_rows,
        "duplicate_row_percentage": float(duplicate_rows / max(len(df), 1)),
        "duplicate_id_counts": duplicate_id_counts,
        "column_type_summary": type_summary["summary"],
        "sample_policy": config.sample_policy,
        "sample_rows": _sample_rows_for_report(df, config, profile_map),
        "data_dictionary": to_jsonable(data_dictionary),
    }


def _sample_values_for_report(profile: dict[str, Any], config: EDAConfig) -> list[Any]:
    if config.sample_policy == "none":
        return []
    if config.sample_policy == "redacted" and _should_redact(profile, config):
        return ["<REDACTED>"] if profile["non_null_count"] else []
    return profile["sample_values"]


def _sample_rows_for_report(
    df: pd.DataFrame,
    config: EDAConfig,
    profile_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if config.sample_policy == "none":
        return []
    records = records_from_frame(df.head(5))
    if config.sample_policy != "redacted":
        return records

    redacted_columns = {
        column
        for column, profile in profile_map.items()
        if _should_redact(profile, config)
    }
    for record in records:
        for column in redacted_columns:
            if column in record and record[column] is not None:
                record[column] = "<REDACTED>"
    return records


def _should_redact(profile: dict[str, Any], config: EDAConfig) -> bool:
    roles = set(profile.get("roles", []))
    return bool(
        profile["name"] in config.sensitive_columns
        or profile["name"] in config.id_columns
        or {"sensitive", "id", "id_candidate"} & roles
    )
