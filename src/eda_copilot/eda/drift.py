from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from eda_copilot.core.config import EDAConfig
from eda_copilot.eda.type_inference import feature_profiles


def analyze_drift(
    df: pd.DataFrame,
    config: EDAConfig,
    type_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compare simple distributions across a train/test or segment column."""

    group_column = config.train_test_column or (config.segment_columns[0] if config.segment_columns else None)
    if not group_column or group_column not in df.columns:
        return {"available": False, "reason": "No train/test or segment column selected."}

    group_counts = df[group_column].astype("object").where(df[group_column].notna(), "<MISSING>").value_counts()
    groups = group_counts.index.tolist()
    if len(groups) < 2:
        return {"available": False, "reason": "Selected comparison column has fewer than two groups."}

    baseline = groups[0]
    comparison = groups[1]
    mask_base = df[group_column] == baseline
    mask_comp = df[group_column] == comparison
    shifted_features = []
    for profile in feature_profiles(type_summary, config):
        column = profile["name"]
        if column == group_column or column not in df.columns:
            continue
        if pdt.is_numeric_dtype(df[column]):
            smd = _standardized_mean_difference(df.loc[mask_base, column], df.loc[mask_comp, column])
            if smd is not None:
                shifted_features.append(
                    {
                        "column": column,
                        "metric": "standardized_mean_difference",
                        "value": smd,
                        "status": _drift_status(smd, config),
                    }
                )
        else:
            psi = _categorical_psi(df.loc[mask_base, column], df.loc[mask_comp, column])
            shifted_features.append(
                {
                    "column": column,
                    "metric": "categorical_psi",
                    "value": psi,
                    "status": _drift_status(psi, config),
                }
            )

    shifted_features = sorted(shifted_features, key=lambda item: abs(item["value"]), reverse=True)
    status = "pass"
    if any(item["status"] == "fail" for item in shifted_features):
        status = "fail"
    elif any(item["status"] == "warn" for item in shifted_features):
        status = "warn"
    return {
        "available": True,
        "overall_status": status,
        "group_column": group_column,
        "baseline_group": baseline,
        "comparison_group": comparison,
        "warning_threshold": config.drift_warning_threshold,
        "fail_threshold": config.drift_fail_threshold,
        "row_count_by_group": {str(key): int(value) for key, value in group_counts.items()},
        "top_shifted_features": shifted_features[:25],
    }


def _drift_status(value: float, config: EDAConfig) -> str:
    abs_value = abs(float(value))
    if abs_value >= config.drift_fail_threshold:
        return "fail"
    if abs_value >= config.drift_warning_threshold:
        return "warn"
    return "pass"


def _standardized_mean_difference(left: pd.Series, right: pd.Series) -> float | None:
    left_numeric = pd.to_numeric(left, errors="coerce").dropna()
    right_numeric = pd.to_numeric(right, errors="coerce").dropna()
    if len(left_numeric) < 2 or len(right_numeric) < 2:
        return None
    pooled = np.sqrt((left_numeric.var() + right_numeric.var()) / 2)
    if pooled == 0 or pd.isna(pooled):
        return None
    return float((left_numeric.mean() - right_numeric.mean()) / pooled)


def _categorical_psi(left: pd.Series, right: pd.Series) -> float:
    left_counts = left.astype("object").where(left.notna(), "<MISSING>").value_counts(normalize=True)
    right_counts = right.astype("object").where(right.notna(), "<MISSING>").value_counts(normalize=True)
    categories = sorted(set(left_counts.index).union(set(right_counts.index)), key=str)
    epsilon = 1e-6
    psi = 0.0
    for category in categories:
        expected = float(left_counts.get(category, 0.0)) + epsilon
        actual = float(right_counts.get(category, 0.0)) + epsilon
        psi += (actual - expected) * np.log(actual / expected)
    return float(psi)
