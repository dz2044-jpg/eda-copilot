from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api import types as pdt

from eda_copilot.core.config import EDAConfig
from eda_copilot.eda.type_inference import feature_profiles
from eda_copilot.utils.serialization import records_from_frame


def analyze_bivariate(
    df: pd.DataFrame,
    config: EDAConfig,
    type_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compute deterministic feature relationship diagnostics."""

    numeric_columns = [
        profile["name"]
        for profile in feature_profiles(type_summary, config)
        if profile["name"] in df.columns and pdt.is_numeric_dtype(df[profile["name"]])
    ][: config.max_correlation_columns]

    correlation_matrix: list[dict[str, Any]] = []
    high_correlation_pairs: list[dict[str, Any]] = []
    if len(numeric_columns) > 1:
        corr = df[numeric_columns].corr(numeric_only=True)
        correlation_matrix = records_from_frame(corr.reset_index(names="column"))
        high_correlation_pairs = _high_correlation_pairs(corr, config.high_correlation_threshold)

    return {
        "numeric_correlation_matrix": correlation_matrix,
        "high_correlation_pairs": high_correlation_pairs,
        "possible_duplicate_columns": _possible_duplicate_columns(df, numeric_columns),
    }


def _high_correlation_pairs(corr: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    pairs = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                pairs.append({"left": left, "right": right, "correlation": float(value)})
    return sorted(pairs, key=lambda item: abs(item["correlation"]), reverse=True)


def _possible_duplicate_columns(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    duplicates = []
    limited_columns = columns[:100]
    for i, left in enumerate(limited_columns):
        for right in limited_columns[i + 1 :]:
            if df[left].equals(df[right]):
                duplicates.append({"left": left, "right": right, "reason": "Exact value match."})
    return duplicates
