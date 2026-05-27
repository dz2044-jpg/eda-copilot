from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from scipy import stats
from sklearn.metrics import roc_auc_score

from eda_copilot.core.config import EDAConfig, ProblemType
from eda_copilot.eda.type_inference import feature_profiles
from eda_copilot.utils.serialization import to_jsonable


POSITIVE_LABELS = {"1", "true", "t", "yes", "y", "positive", "event", "bad", "claim", "lapse"}


def analyze_response(
    df: pd.DataFrame,
    config: EDAConfig,
    type_summary: dict[str, Any],
) -> dict[str, Any]:
    """Analyze the response variable and deterministic feature relationships."""

    problem_type = infer_problem_type(df, config)
    if problem_type == "unsupervised" or not config.response_column:
        return {
            "problem_type": "unsupervised",
            "available": False,
            "reason": "No response variable was selected.",
        }

    target = df[config.response_column]
    if problem_type == "binary_classification":
        return _binary_response_analysis(df, config, type_summary, target)
    if problem_type == "regression":
        return _regression_response_analysis(df, config, type_summary, target)
    return _multiclass_response_analysis(target, problem_type)


def infer_problem_type(df: pd.DataFrame, config: EDAConfig) -> ProblemType:
    """Infer problem type from configuration and response values."""

    if config.problem_type != "auto":
        return config.problem_type
    if not config.response_column or config.response_column not in df.columns:
        return "unsupervised"

    target = df[config.response_column].dropna()
    unique_count = int(target.nunique(dropna=True))
    if unique_count == 0:
        return "unsupervised"
    if unique_count == 2:
        return "binary_classification"
    if pdt.is_numeric_dtype(target) and unique_count > 10:
        return "regression"
    return "multiclass_classification"


def is_binary_response(series: pd.Series) -> bool:
    """Return whether a series has exactly two non-null response classes."""

    return int(series.dropna().nunique(dropna=True)) == 2


def encode_binary_response(series: pd.Series) -> tuple[pd.Series, Any, Any]:
    """Encode a two-class response as 0/1 with a deterministic positive class.

    The positive class is selected using common positive labels when available.
    Otherwise, values are sorted by their string representation and the last value is used.
    """

    clean_values = series.dropna().drop_duplicates().tolist()
    if len(clean_values) != 2:
        raise ValueError("Binary response encoding requires exactly two non-null values.")

    positive_class = None
    for value in clean_values:
        if str(value).strip().lower() in POSITIVE_LABELS:
            positive_class = value
            break
    if positive_class is None:
        positive_class = sorted(clean_values, key=lambda item: str(item))[-1]
    negative_class = [value for value in clean_values if value != positive_class][0]
    encoded = series.map({negative_class: 0, positive_class: 1}).astype("float")
    return encoded, positive_class, negative_class


def _binary_response_analysis(
    df: pd.DataFrame,
    config: EDAConfig,
    type_summary: dict[str, Any],
    target: pd.Series,
) -> dict[str, Any]:
    encoded, positive_class, negative_class = encode_binary_response(target)
    class_counts = target.value_counts(dropna=False).to_dict()
    non_missing_encoded = encoded.dropna()
    minority_count = int(min(non_missing_encoded.sum(), len(non_missing_encoded) - non_missing_encoded.sum()))
    majority_count = int(max(non_missing_encoded.sum(), len(non_missing_encoded) - non_missing_encoded.sum()))
    relationships = []
    for profile in feature_profiles(type_summary, config):
        column = profile["name"]
        if column not in df.columns:
            continue
        if profile["semantic_type"] in {"numeric_continuous", "numeric_discrete", "binary"} and pdt.is_numeric_dtype(df[column]):
            relationships.append(_numeric_binary_relationship(df[column], encoded))
        else:
            relationships.append(_categorical_binary_relationship(df[column], encoded, config))

    return to_jsonable(
        {
            "problem_type": "binary_classification",
            "available": True,
            "response_column": config.response_column,
            "positive_class": positive_class,
            "negative_class": negative_class,
            "class_counts": class_counts,
            "response_rate": float(non_missing_encoded.mean()) if not non_missing_encoded.empty else None,
            "imbalance_ratio": float(majority_count / max(minority_count, 1)) if minority_count else None,
            "feature_relationships": relationships,
        }
    )


def _numeric_binary_relationship(series: pd.Series, encoded_target: pd.Series) -> dict[str, Any]:
    frame = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "y": encoded_target}).dropna()
    result: dict[str, Any] = {
        "column": str(series.name),
        "feature_type": "numeric",
        "n": int(len(frame)),
        "auc": None,
        "auc_abs": None,
        "spearman_correlation": None,
        "response_rate_spread": None,
        "bin_response_rates": [],
        "t_test_p_value": None,
        "mann_whitney_p_value": None,
        "calculation_notes": [],
    }
    if len(frame) < 3 or frame["x"].nunique(dropna=True) < 2 or frame["y"].nunique(dropna=True) < 2:
        return result

    try:
        auc = float(roc_auc_score(frame["y"], frame["x"]))
        result["auc"] = auc
        result["auc_abs"] = max(auc, 1.0 - auc)
    except ValueError as exc:
        result["calculation_notes"].append(f"AUC unavailable: {exc}")

    try:
        result["spearman_correlation"] = float(frame[["x", "y"]].corr(method="spearman").iloc[0, 1])
    except (ValueError, IndexError) as exc:
        result["calculation_notes"].append(f"Spearman correlation unavailable: {exc}")

    result["bin_response_rates"] = _numeric_bin_response_rates(frame)
    rates = [item["response_rate"] for item in result["bin_response_rates"] if item["response_rate"] is not None]
    if rates:
        result["response_rate_spread"] = float(max(rates) - min(rates))

    group0 = frame.loc[frame["y"] == 0, "x"]
    group1 = frame.loc[frame["y"] == 1, "x"]
    if len(group0) > 1 and len(group1) > 1:
        result["t_test_p_value"] = _safe_pvalue(stats.ttest_ind(group0, group1, equal_var=False, nan_policy="omit").pvalue)
        try:
            result["mann_whitney_p_value"] = _safe_pvalue(stats.mannwhitneyu(group0, group1, alternative="two-sided").pvalue)
        except ValueError:
            result["mann_whitney_p_value"] = None
    return to_jsonable(result)


def _categorical_binary_relationship(
    series: pd.Series,
    encoded_target: pd.Series,
    config: EDAConfig,
) -> dict[str, Any]:
    frame = pd.DataFrame({"x": series.astype("object"), "y": encoded_target}).dropna()
    result: dict[str, Any] = {
        "column": str(series.name),
        "feature_type": "categorical",
        "n": int(len(frame)),
        "unique_count": int(frame["x"].nunique(dropna=True)),
        "response_rate_spread": None,
        "chi_square_p_value": None,
        "cramers_v": None,
        "category_response_rates": [],
    }
    if frame.empty or frame["x"].nunique(dropna=True) < 1 or frame["y"].nunique(dropna=True) < 2:
        return result

    grouped = (
        frame.groupby("x", dropna=False)["y"]
        .agg(["count", "mean"])
        .sort_values(["mean", "count"], ascending=[False, False])
        .head(config.max_categories)
    )
    result["category_response_rates"] = [
        {"category": category, "count": int(row["count"]), "response_rate": float(row["mean"])}
        for category, row in grouped.iterrows()
    ]
    rates = [item["response_rate"] for item in result["category_response_rates"]]
    if rates:
        result["response_rate_spread"] = float(max(rates) - min(rates))

    contingency = pd.crosstab(frame["x"], frame["y"])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, pvalue, _dof, _expected = stats.chi2_contingency(contingency)
        result["chi_square_p_value"] = _safe_pvalue(pvalue)
        n = contingency.to_numpy().sum()
        min_dim = min(contingency.shape) - 1
        result["cramers_v"] = float(np.sqrt(chi2 / (n * min_dim))) if n and min_dim else None
    return to_jsonable(result)


def _regression_response_analysis(
    df: pd.DataFrame,
    config: EDAConfig,
    type_summary: dict[str, Any],
    target: pd.Series,
) -> dict[str, Any]:
    numeric_target = pd.to_numeric(target, errors="coerce")
    relationships = []
    for profile in feature_profiles(type_summary, config):
        column = profile["name"]
        if profile["semantic_type"] in {"numeric_continuous", "numeric_discrete", "binary"} and pdt.is_numeric_dtype(df[column]):
            frame = pd.DataFrame({"x": pd.to_numeric(df[column], errors="coerce"), "y": numeric_target}).dropna()
            corr = float(frame[["x", "y"]].corr().iloc[0, 1]) if len(frame) > 2 and frame["x"].nunique() > 1 else None
            relationships.append({"column": column, "feature_type": "numeric", "correlation": corr, "n": int(len(frame))})
        else:
            frame = pd.DataFrame({"x": df[column].astype("object"), "y": numeric_target}).dropna()
            grouped = frame.groupby("x")["y"].mean().sort_values(ascending=False).head(config.max_categories)
            spread = float(grouped.max() - grouped.min()) if len(grouped) > 1 else None
            relationships.append(
                {
                    "column": column,
                    "feature_type": "categorical",
                    "target_mean_spread": spread,
                    "category_target_means": [
                        {"category": category, "target_mean": float(value)}
                        for category, value in grouped.items()
                    ],
                }
            )
    return to_jsonable(
        {
            "problem_type": "regression",
            "available": True,
            "response_column": config.response_column,
            "target_summary": {
                "count": int(numeric_target.count()),
                "mean": float(numeric_target.mean()) if numeric_target.count() else None,
                "median": float(numeric_target.median()) if numeric_target.count() else None,
                "std": float(numeric_target.std()) if numeric_target.count() > 1 else None,
                "min": float(numeric_target.min()) if numeric_target.count() else None,
                "max": float(numeric_target.max()) if numeric_target.count() else None,
            },
            "feature_relationships": relationships,
        }
    )


def _multiclass_response_analysis(target: pd.Series, problem_type: ProblemType) -> dict[str, Any]:
    counts = target.value_counts(dropna=False)
    return to_jsonable(
        {
            "problem_type": problem_type,
            "available": True,
            "response_column": str(target.name),
            "class_counts": counts.to_dict(),
            "class_percentages": (counts / max(len(target), 1)).to_dict(),
            "feature_relationships": [],
            "caveat": "MVP includes summaries for multiclass targets but not feature-level tests yet.",
        }
    )


def _numeric_bin_response_rates(frame: pd.DataFrame) -> list[dict[str, Any]]:
    unique_count = frame["x"].nunique(dropna=True)
    bins = min(10, int(unique_count))
    if bins < 2:
        return []
    try:
        binned = pd.qcut(frame["x"], q=bins, duplicates="drop")
    except ValueError:
        return []
    grouped = frame.assign(bin=binned).groupby("bin", observed=False)["y"].agg(["count", "mean"])
    return [
        {"bin": str(bin_label), "count": int(row["count"]), "response_rate": float(row["mean"])}
        for bin_label, row in grouped.iterrows()
    ]


def _safe_pvalue(value: float | np.floating[Any]) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
