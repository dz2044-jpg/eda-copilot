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
    data_dictionary = [
        {
            "column": column,
            "pandas_dtype": profile_map[column]["pandas_dtype"],
            "semantic_type": profile_map[column]["semantic_type"],
            "roles": profile_map[column]["roles"],
            "missing_percentage": profile_map[column]["missing_percentage"],
            "unique_count": profile_map[column]["unique_count"],
            "sample_values": profile_map[column]["sample_values"],
        }
        for column in df.columns
    ]

    return {
        "dataset_name": config.dataset_name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "duplicate_row_count": duplicate_rows,
        "duplicate_row_percentage": float(duplicate_rows / max(len(df), 1)),
        "duplicate_id_counts": duplicate_id_counts,
        "column_type_summary": type_summary["summary"],
        "sample_rows": records_from_frame(df.head(5)),
        "data_dictionary": to_jsonable(data_dictionary),
    }
