from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api import types as pdt

from eda_copilot.core.config import EDAConfig
from eda_copilot.eda.type_inference import feature_profiles
from eda_copilot.utils.serialization import to_jsonable


def build_comparison_summary(
    df: pd.DataFrame,
    config: EDAConfig,
    type_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compare reference/current groups from a split or segment column."""

    group_column = config.train_test_column or (config.segment_columns[0] if config.segment_columns else None)
    if not group_column or group_column not in df.columns:
        return {"available": False, "reason": "No train/test or segment column selected."}

    group_values = df[group_column].astype("object").where(df[group_column].notna(), "<MISSING>")
    counts = group_values.value_counts(dropna=False)
    if len(counts) < 2:
        return {"available": False, "reason": "Selected comparison column has fewer than two groups."}

    reference_group, current_group = _reference_current_groups(counts)
    reference_mask = group_values == reference_group
    current_mask = group_values == current_group

    column_changes = []
    for profile in feature_profiles(type_summary, config):
        column = profile["name"]
        if column == group_column or column not in df.columns:
            continue
        column_changes.append(
            _column_change(
                series=df[column],
                reference_mask=reference_mask,
                current_mask=current_mask,
                semantic_type=str(profile["semantic_type"]),
                profile_depth=config.profile_depth,
            )
        )

    return to_jsonable(
        {
            "available": True,
            "comparison_column": group_column,
            "reference_group": reference_group,
            "current_group": current_group,
            "row_count_by_group": {str(key): int(value) for key, value in counts.items()},
            "top_column_changes": sorted(
                column_changes,
                key=lambda row: abs(float(row.get("change_score") or 0.0)),
                reverse=True,
            )[:25],
        }
    )


def _reference_current_groups(counts: pd.Series) -> tuple[Any, Any]:
    normalized = {str(value).lower(): value for value in counts.index}
    if "train" in normalized and "test" in normalized:
        return normalized["train"], normalized["test"]
    if "reference" in normalized and "current" in normalized:
        return normalized["reference"], normalized["current"]
    groups = counts.index.tolist()
    return groups[0], groups[1]


def _column_change(
    series: pd.Series,
    reference_mask: pd.Series,
    current_mask: pd.Series,
    semantic_type: str,
    profile_depth: str,
) -> dict[str, Any]:
    reference = series[reference_mask]
    current = series[current_mask]
    missing_delta = float(current.isna().mean() - reference.isna().mean())
    result: dict[str, Any] = {
        "column": str(series.name),
        "semantic_type": semantic_type,
        "reference_missing_percentage": float(reference.isna().mean()),
        "current_missing_percentage": float(current.isna().mean()),
        "missing_percentage_delta": missing_delta,
    }

    if pdt.is_numeric_dtype(series):
        reference_numeric = pd.to_numeric(reference, errors="coerce")
        current_numeric = pd.to_numeric(current, errors="coerce")
        reference_mean = reference_numeric.mean()
        current_mean = current_numeric.mean()
        mean_delta = current_mean - reference_mean
        result.update(
            {
                "comparison_metric": "mean_delta",
                "reference_mean": float(reference_mean) if pd.notna(reference_mean) else None,
                "current_mean": float(current_mean) if pd.notna(current_mean) else None,
                "mean_delta": float(mean_delta) if pd.notna(mean_delta) else None,
                "change_score": float(mean_delta) if pd.notna(mean_delta) else missing_delta,
            }
        )
        return result

    reference_values = set(reference.dropna().astype(str).unique())
    current_values = set(current.dropna().astype(str).unique())
    new_values = sorted(current_values - reference_values)[:10]
    removed_values = sorted(reference_values - current_values)[:10]
    result.update(
        {
            "comparison_metric": "category_set_delta",
            "reference_unique_count": len(reference_values),
            "current_unique_count": len(current_values),
            "new_current_values": new_values if profile_depth != "minimal" else [],
            "removed_current_values": removed_values if profile_depth == "deep" else [],
            "change_score": max(abs(missing_delta), len(new_values) / max(len(current_values), 1)),
        }
    )
    return result
