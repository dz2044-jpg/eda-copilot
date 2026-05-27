from __future__ import annotations

from collections import Counter
import warnings
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from eda_copilot.core.config import EDAConfig
from eda_copilot.utils.serialization import to_jsonable


SUSPICIOUS_NAME_KEYWORDS = (
    "target",
    "outcome",
    "approved",
    "declined",
    "decision",
    "final",
    "result",
    "score",
    "label",
    "default",
    "claim",
    "lapse",
    "fraud",
    "bad",
    "post_",
    "after_",
)

TRUE_FALSE_STRINGS = {"true", "false", "t", "f", "yes", "no", "y", "n"}


def infer_column_types(df: pd.DataFrame, config: EDAConfig) -> dict[str, Any]:
    """Infer semantic column types and warning tags for a dataframe.

    Args:
        df: Input dataset.
        config: EDA configuration with user overrides.

    Returns:
        A JSON-serializable dictionary containing one profile per column and summary counts.
    """

    profiles = [_profile_column(df[column], df, config) for column in df.columns]
    type_counts = Counter(profile["semantic_type"] for profile in profiles)
    role_counts = Counter(role for profile in profiles for role in profile["roles"])
    return {
        "columns": profiles,
        "summary": {
            "semantic_type_counts": dict(sorted(type_counts.items())),
            "role_counts": dict(sorted(role_counts.items())),
            "suspicious_name_columns": [
                profile["name"] for profile in profiles if profile["suspicious_name"]
            ],
        },
    }


def profiles_by_name(type_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index column profiles by name."""

    return {profile["name"]: profile for profile in type_summary.get("columns", [])}


def feature_profiles(type_summary: dict[str, Any], config: EDAConfig) -> list[dict[str, Any]]:
    """Return profiles that should be treated as candidate predictors."""

    excluded = config.analysis_exclusions()
    return [
        profile
        for profile in type_summary.get("columns", [])
        if profile["name"] not in excluded
    ]


def _profile_column(series: pd.Series, df: pd.DataFrame, config: EDAConfig) -> dict[str, Any]:
    name = str(series.name)
    non_null = series.dropna()
    row_count = len(series)
    non_null_count = int(non_null.shape[0])
    missing_count = int(row_count - non_null_count)
    unique_count = int(non_null.nunique(dropna=True))
    unique_ratio = float(unique_count / max(non_null_count, 1))
    top_frequency_ratio = _top_frequency_ratio(non_null)
    roles = _infer_roles(name, series, df, config, unique_count, unique_ratio)
    semantic_type = _infer_semantic_type(
        name=name,
        series=series,
        non_null=non_null,
        config=config,
        unique_count=unique_count,
        top_frequency_ratio=top_frequency_ratio,
    )
    warnings = _infer_type_warnings(
        name=name,
        semantic_type=semantic_type,
        missing_count=missing_count,
        row_count=row_count,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        top_frequency_ratio=top_frequency_ratio,
        config=config,
    )

    return {
        "name": name,
        "pandas_dtype": str(series.dtype),
        "semantic_type": semantic_type,
        "roles": roles,
        "non_null_count": non_null_count,
        "missing_count": missing_count,
        "missing_percentage": _safe_rate(missing_count, row_count),
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "top_frequency_ratio": top_frequency_ratio,
        "suspicious_name": _has_suspicious_name(name),
        "sample_values": _sample_values(non_null),
        "warnings": warnings,
    }


def _infer_roles(
    name: str,
    series: pd.Series,
    df: pd.DataFrame,
    config: EDAConfig,
    unique_count: int,
    unique_ratio: float,
) -> list[str]:
    roles: list[str] = []
    if name == config.response_column:
        roles.append("response")
    if name in config.id_columns:
        roles.append("id")
    if name in config.date_columns:
        roles.append("date")
    if name == config.train_test_column:
        roles.append("train_test_split")
    if name in config.segment_columns:
        roles.append("segment")
    if name in config.sensitive_columns:
        roles.append("sensitive")
    if name == config.weight_column:
        roles.append("weight")
    if name in config.exclude_columns:
        roles.append("excluded")
    if _looks_id_like(name, series, df, unique_count, unique_ratio):
        roles.append("id_candidate")
    return sorted(set(roles))


def _infer_semantic_type(
    name: str,
    series: pd.Series,
    non_null: pd.Series,
    config: EDAConfig,
    unique_count: int,
    top_frequency_ratio: float,
) -> str:
    if unique_count == 0:
        return "all_missing"
    if unique_count == 1:
        return "constant"
    if top_frequency_ratio >= config.near_constant_threshold:
        return "near_constant"
    if name in config.date_columns or pdt.is_datetime64_any_dtype(series):
        return "datetime"
    if pdt.is_bool_dtype(series) or _looks_boolean(non_null):
        return "boolean"
    if unique_count == 2:
        return "binary"
    if pdt.is_numeric_dtype(series):
        if _looks_discrete_numeric(non_null, unique_count):
            return "numeric_discrete"
        return "numeric_continuous"
    if _looks_datetime_like(non_null):
        return "datetime"
    if _looks_text_like(non_null):
        return "text"
    if unique_count > config.high_cardinality_threshold:
        return "categorical_high_cardinality"
    return "categorical_low_cardinality"


def _infer_type_warnings(
    name: str,
    semantic_type: str,
    missing_count: int,
    row_count: int,
    unique_count: int,
    unique_ratio: float,
    top_frequency_ratio: float,
    config: EDAConfig,
) -> list[str]:
    warnings: list[str] = []
    if missing_count == row_count:
        warnings.append("all_missing")
    elif _safe_rate(missing_count, row_count) >= config.high_missingness_threshold:
        warnings.append("high_missingness")
    if semantic_type in {"constant", "near_constant"}:
        warnings.append(semantic_type)
    if semantic_type == "categorical_high_cardinality":
        warnings.append("high_cardinality")
    if unique_ratio >= 0.95 and row_count >= 20:
        warnings.append("mostly_unique")
    if top_frequency_ratio >= config.near_constant_threshold:
        warnings.append("dominant_single_value")
    if _has_suspicious_name(name):
        warnings.append("suspicious_name")
    if unique_count > config.high_cardinality_threshold * 5:
        warnings.append("very_high_cardinality")
    return sorted(set(warnings))


def _looks_boolean(non_null: pd.Series) -> bool:
    values = {str(value).strip().lower() for value in non_null.unique()}
    return 0 < len(values) <= 2 and values.issubset(TRUE_FALSE_STRINGS | {"0", "1"})


def _looks_discrete_numeric(non_null: pd.Series, unique_count: int) -> bool:
    if unique_count <= 20 and pdt.is_integer_dtype(non_null):
        return True
    numeric = pd.to_numeric(non_null, errors="coerce").dropna()
    if numeric.empty:
        return False
    all_integer_values = bool(np.all(np.isclose(numeric % 1, 0)))
    return all_integer_values and unique_count <= 20


def _looks_datetime_like(non_null: pd.Series) -> bool:
    if non_null.empty or pdt.is_numeric_dtype(non_null):
        return False
    sample = non_null.astype(str).head(200)
    has_digit_rate = float(sample.str.contains(r"\d", regex=True).mean())
    has_date_hint_rate = float(sample.str.contains(r"[-/:T ]", regex=True).mean())
    if has_digit_rate < 0.85 or has_date_hint_rate < 0.50:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
    parse_rate = float(parsed.notna().mean())
    return parse_rate >= 0.85 and sample.nunique(dropna=True) > 2


def _looks_text_like(non_null: pd.Series) -> bool:
    if non_null.empty:
        return False
    sample = non_null.astype(str).head(500)
    avg_length = float(sample.str.len().mean())
    avg_words = float(sample.str.split().str.len().mean())
    return avg_length >= 40 or avg_words >= 6


def _looks_id_like(
    name: str,
    series: pd.Series,
    df: pd.DataFrame,
    unique_count: int,
    unique_ratio: float,
) -> bool:
    lowered = name.lower()
    name_hint = lowered.endswith("_id") or lowered in {"id", "uuid", "guid", "key"} or "identifier" in lowered
    mostly_unique = unique_ratio >= 0.95 and len(df) >= 20
    return bool(name_hint or (mostly_unique and unique_count >= max(20, int(len(series) * 0.8))))


def _has_suspicious_name(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in SUSPICIOUS_NAME_KEYWORDS)


def _top_frequency_ratio(non_null: pd.Series) -> float:
    if non_null.empty:
        return 0.0
    counts = non_null.value_counts(dropna=True)
    if counts.empty:
        return 0.0
    return float(counts.iloc[0] / len(non_null))


def _sample_values(non_null: pd.Series) -> list[Any]:
    values = non_null.drop_duplicates().head(5).tolist()
    return to_jsonable(values)


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
