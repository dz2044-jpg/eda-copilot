from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from eda_copilot.core.config import EDAConfig
from eda_copilot.eda.type_inference import profiles_by_name
from eda_copilot.utils.serialization import to_jsonable


NUMERIC_TYPES = {"numeric_continuous", "numeric_discrete", "binary"}


def analyze_univariate(
    df: pd.DataFrame,
    config: EDAConfig,
    type_summary: dict[str, Any],
) -> dict[str, Any]:
    """Calculate per-column univariate summaries."""

    profile_map = profiles_by_name(type_summary)
    numeric: list[dict[str, Any]] = []
    categorical: list[dict[str, Any]] = []
    datetime_columns: list[dict[str, Any]] = []

    for column in df.columns:
        profile = profile_map[column]
        semantic_type = profile["semantic_type"]
        if semantic_type in NUMERIC_TYPES and pdt.is_numeric_dtype(df[column]):
            numeric.append(_numeric_summary(df[column]))
        elif semantic_type == "datetime":
            datetime_columns.append(_datetime_summary(df[column]))
        else:
            categorical.append(_categorical_summary(df[column], config))

    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_columns,
    }


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    quantiles = numeric.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    q1 = quantiles.loc[0.25] if not quantiles.empty else np.nan
    q3 = quantiles.loc[0.75] if not quantiles.empty else np.nan
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    p01 = quantiles.loc[0.01] if not quantiles.empty else np.nan
    p99 = quantiles.loc[0.99] if not quantiles.empty else np.nan
    return to_jsonable(
        {
            "column": str(series.name),
            "count": int(numeric.count()),
            "mean": float(numeric.mean()) if not numeric.empty else None,
            "median": float(numeric.median()) if not numeric.empty else None,
            "std": float(numeric.std()) if numeric.count() > 1 else None,
            "min": float(numeric.min()) if not numeric.empty else None,
            "max": float(numeric.max()) if not numeric.empty else None,
            "percentiles": {
                "p01": p01,
                "p05": quantiles.loc[0.05] if not quantiles.empty else np.nan,
                "p25": q1,
                "p50": quantiles.loc[0.50] if not quantiles.empty else np.nan,
                "p75": q3,
                "p95": quantiles.loc[0.95] if not quantiles.empty else np.nan,
                "p99": p99,
            },
            "skewness": float(numeric.skew()) if numeric.count() > 2 else None,
            "kurtosis": float(numeric.kurtosis()) if numeric.count() > 3 else None,
            "unique_count": int(numeric.nunique(dropna=True)),
            "iqr_outlier_count": int(((numeric < lower) | (numeric > upper)).sum())
            if not numeric.empty
            else 0,
            "p01_p99_outlier_count": int(((numeric < p01) | (numeric > p99)).sum())
            if not numeric.empty
            else 0,
        }
    )


def _categorical_summary(series: pd.Series, config: EDAConfig) -> dict[str, Any]:
    values = series.astype("object").where(series.notna(), "<MISSING>")
    counts = values.value_counts(dropna=False)
    top = counts.head(config.max_categories)
    rare_mask = counts / max(len(series), 1) < config.rare_category_threshold
    rare_category_count = int(rare_mask.sum())
    rare_row_count = int(counts[rare_mask].sum()) if rare_category_count else 0
    return to_jsonable(
        {
            "column": str(series.name),
            "unique_count": int(series.nunique(dropna=True)),
            "top_categories": [
                {
                    "value": value,
                    "count": int(count),
                    "percentage": float(count / max(len(series), 1)),
                }
                for value, count in top.items()
            ],
            "rare_category_count": rare_category_count,
            "rare_category_percentage": float(rare_row_count / max(len(series), 1)),
            "cardinality_warning": bool(series.nunique(dropna=True) > config.high_cardinality_threshold),
        }
    )


def _datetime_summary(series: pd.Series) -> dict[str, Any]:
    parsed = pd.to_datetime(series, errors="coerce")
    clean = parsed.dropna()
    return to_jsonable(
        {
            "column": str(series.name),
            "count": int(clean.count()),
            "min": clean.min() if not clean.empty else None,
            "max": clean.max() if not clean.empty else None,
            "unique_count": int(clean.nunique(dropna=True)),
            "parse_failure_count": int(parsed.isna().sum() - series.isna().sum()),
        }
    )
