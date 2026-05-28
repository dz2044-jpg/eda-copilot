from __future__ import annotations

from typing import Any

import pandas as pd


MISSING_GROUP_LABEL = "<MISSING>"


def normalized_group_values(series: pd.Series) -> pd.Series:
    """Return group labels with missing values represented consistently."""

    return series.astype("object").where(series.notna(), MISSING_GROUP_LABEL)


def reference_current_groups(counts: pd.Series) -> tuple[Any, Any]:
    """Choose deterministic reference/current groups from normalized counts."""

    normalized = {str(value).lower(): value for value in counts.index}
    if "train" in normalized and "test" in normalized:
        return normalized["train"], normalized["test"]
    if "reference" in normalized and "current" in normalized:
        return normalized["reference"], normalized["current"]
    groups = counts.index.tolist()
    return groups[0], groups[1]


def ignored_groups(counts: pd.Series, selected_groups: tuple[Any, Any]) -> list[Any]:
    """Return groups not used in a two-group comparison."""

    selected = set(selected_groups)
    return [group for group in counts.index.tolist() if group not in selected]
