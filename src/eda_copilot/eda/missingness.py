from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api import types as pdt

from eda_copilot.core.config import EDAConfig
from eda_copilot.eda.response_analysis import encode_binary_response, is_binary_response
from eda_copilot.utils.serialization import records_from_frame, to_jsonable


SPECIAL_MISSING_STRINGS = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "unknown",
    "unk",
    "missing",
    "not available",
}
SPECIAL_MISSING_NUMERIC = {-1, -9, -99, 999, 9999, 99999}


def analyze_missingness(df: pd.DataFrame, config: EDAConfig) -> dict[str, Any]:
    """Calculate column, row, response, and segment missingness diagnostics."""

    table = [_column_missingness(df[column]) for column in df.columns]
    missing_flags = df.isna()
    row_missing_count = missing_flags.sum(axis=1)
    high_missing_rows = int((row_missing_count / max(len(df.columns), 1) >= 0.50).sum())

    response_breakdown = []
    if config.response_column and config.response_column in df.columns:
        target = df[config.response_column]
        if is_binary_response(target):
            response_breakdown = _missingness_by_binary_response(df, config.response_column)

    segment_breakdown: dict[str, list[dict[str, Any]]] = {}
    for segment in config.segment_columns:
        if segment in df.columns:
            segment_breakdown[segment] = _missingness_by_segment(df, segment)

    corr_records: list[dict[str, Any]] = []
    missing_columns = [item["column"] for item in table if item["missing_count"] > 0]
    if 1 < len(missing_columns) <= 50:
        corr = missing_flags[missing_columns].astype(int).corr()
        corr_records = records_from_frame(corr.reset_index(names="column"))

    return {
        "columns": table,
        "high_missingness_columns": [
            item
            for item in table
            if item["missing_percentage"] >= config.high_missingness_threshold
        ],
        "row_missingness": {
            "mean_missing_columns": float(row_missing_count.mean()) if len(df) else 0.0,
            "max_missing_columns": int(row_missing_count.max()) if len(df) else 0,
            "rows_with_at_least_50pct_missing": high_missing_rows,
        },
        "missingness_by_response": response_breakdown,
        "missingness_by_segment": segment_breakdown,
        "missingness_correlation": corr_records,
    }


def _column_missingness(series: pd.Series) -> dict[str, Any]:
    row_count = len(series)
    missing_count = int(series.isna().sum())
    blank_string_count = _blank_string_count(series)
    special_codes = _special_missing_codes(series)
    zero_count = int((series == 0).sum()) if pdt.is_numeric_dtype(series) else None
    return {
        "column": str(series.name),
        "missing_count": missing_count,
        "missing_percentage": float(missing_count / max(row_count, 1)),
        "zero_count": zero_count,
        "blank_string_count": blank_string_count,
        "special_missing_code_counts": special_codes,
    }


def _blank_string_count(series: pd.Series) -> int:
    if pdt.is_numeric_dtype(series):
        return 0
    values = series.dropna().astype(str).str.strip()
    return int((values == "").sum())


def _special_missing_codes(series: pd.Series) -> dict[str, int]:
    non_null = series.dropna()
    if non_null.empty:
        return {}
    if pdt.is_numeric_dtype(series):
        counts = {
            str(code): int((non_null == code).sum())
            for code in SPECIAL_MISSING_NUMERIC
            if int((non_null == code).sum()) > 0
        }
        return dict(sorted(counts.items()))

    normalized = non_null.astype(str).str.strip().str.lower()
    counts = {
        label: int((normalized == label).sum())
        for label in SPECIAL_MISSING_STRINGS
        if int((normalized == label).sum()) > 0
    }
    return dict(sorted(counts.items()))


def _missingness_by_binary_response(df: pd.DataFrame, response_column: str) -> list[dict[str, Any]]:
    encoded, positive_class, _negative_class = encode_binary_response(df[response_column])
    records = []
    for column in df.columns:
        if column == response_column:
            continue
        missing = df[column].isna()
        present = ~missing
        records.append(
            {
                "column": column,
                "positive_class": to_jsonable(positive_class),
                "missing_count": int(missing.sum()),
                "missing_response_rate": _mean_or_none(encoded[missing]),
                "present_response_rate": _mean_or_none(encoded[present]),
                "response_rate_gap": _gap(encoded[missing], encoded[present]),
            }
        )
    return records


def _missingness_by_segment(df: pd.DataFrame, segment: str) -> list[dict[str, Any]]:
    records = []
    top_segments = df[segment].astype("object").where(df[segment].notna(), "<MISSING>")
    counts = top_segments.value_counts(dropna=False).head(20)
    for segment_value in counts.index:
        mask = top_segments == segment_value
        missing_pct = df.loc[mask].isna().mean(numeric_only=False)
        for column, pct in missing_pct.items():
            records.append(
                {
                    "segment_column": segment,
                    "segment_value": to_jsonable(segment_value),
                    "column": column,
                    "missing_percentage": float(pct),
                }
            )
    return records


def _mean_or_none(values: pd.Series) -> float | None:
    clean = values.dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _gap(left: pd.Series, right: pd.Series) -> float | None:
    left_mean = _mean_or_none(left)
    right_mean = _mean_or_none(right)
    if left_mean is None or right_mean is None:
        return None
    return float(left_mean - right_mean)
