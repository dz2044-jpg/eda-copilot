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
SEVERE_IMBALANCE_RATIO = 10.0
REGRESSION_SKEW_WARNING_THRESHOLD = 2.0


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
    warnings = _base_response_warnings(target, config.response_column)
    if not is_binary_response(target):
        warnings.append(
            _response_warning(
                issue_type="unusable_binary_response",
                severity="high",
                evidence="Binary response analysis requires exactly two non-null classes.",
                recommended_action="Select a response with two observed classes or change the configured problem type.",
                metric_name="non_null_unique_count",
                metric_value=int(target.dropna().nunique(dropna=True)),
                threshold=2,
            )
        )
        return to_jsonable(
            {
                "problem_type": "binary_classification",
                "available": False,
                "response_column": config.response_column,
                "reason": "Binary response requires exactly two non-null classes.",
                "warnings": warnings,
                "feature_relationships": [],
            }
        )

    encoded, positive_class, negative_class = encode_binary_response(target)
    class_counts = target.value_counts(dropna=False).to_dict()
    non_missing_encoded = encoded.dropna()
    minority_count = int(min(non_missing_encoded.sum(), len(non_missing_encoded) - non_missing_encoded.sum()))
    majority_count = int(max(non_missing_encoded.sum(), len(non_missing_encoded) - non_missing_encoded.sum()))
    imbalance_ratio = float(majority_count / max(minority_count, 1)) if minority_count else None
    warnings.append(
        _response_warning(
            issue_type="positive_class_auto_selected",
            severity="low",
            evidence=f"Positive class was selected deterministically as {positive_class!r}.",
            recommended_action="Confirm that the selected positive class matches the project definition.",
            metric_name="positive_class",
            metric_value=positive_class,
            threshold="confirm with project owner",
        )
    )
    if imbalance_ratio is not None and imbalance_ratio >= SEVERE_IMBALANCE_RATIO:
        warnings.append(
            _response_warning(
                issue_type="severe_class_imbalance",
                severity="high",
                evidence=f"Majority/minority class ratio is {imbalance_ratio:.2f}.",
                recommended_action="Use stratified validation and imbalance-aware metrics before modeling.",
                metric_name="imbalance_ratio",
                metric_value=imbalance_ratio,
                threshold=SEVERE_IMBALANCE_RATIO,
            )
        )

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
            "class_table": _binary_class_table(target, encoded, positive_class, negative_class),
            "response_rate": float(non_missing_encoded.mean()) if not non_missing_encoded.empty else None,
            "imbalance_ratio": imbalance_ratio,
            "warnings": warnings,
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
        "bottom_bin_response_rate": None,
        "top_bin_response_rate": None,
        "monotonic_trend": False,
        "trend_direction": None,
        "signal_direction": None,
        "missing_response_rate": None,
        "present_response_rate": None,
        "missing_present_response_rate_gap": None,
        "t_test_p_value": None,
        "mann_whitney_p_value": None,
        "calculation_notes": [],
    }
    missing = series.isna()
    present = ~missing
    result["missing_response_rate"] = _mean_or_none(encoded_target[missing])
    result["present_response_rate"] = _mean_or_none(encoded_target[present])
    result["missing_present_response_rate_gap"] = _gap(encoded_target[missing], encoded_target[present])
    if len(frame) < 3 or frame["x"].nunique(dropna=True) < 2 or frame["y"].nunique(dropna=True) < 2:
        return result

    try:
        auc = float(roc_auc_score(frame["y"], frame["x"]))
        result["auc"] = auc
        result["auc_abs"] = max(auc, 1.0 - auc)
    except ValueError as exc:
        result["calculation_notes"].append(f"AUC unavailable: {exc}")

    try:
        spearman = float(frame[["x", "y"]].corr(method="spearman").iloc[0, 1])
        result["spearman_correlation"] = spearman
        result["signal_direction"] = _direction(spearman)
    except (ValueError, IndexError) as exc:
        result["calculation_notes"].append(f"Spearman correlation unavailable: {exc}")

    result["bin_response_rates"] = _numeric_bin_response_rates(frame)
    rates = [item["response_rate"] for item in result["bin_response_rates"] if item["response_rate"] is not None]
    if rates:
        result["response_rate_spread"] = float(max(rates) - min(rates))
        result["bottom_bin_response_rate"] = rates[0]
        result["top_bin_response_rate"] = rates[-1]
        result["monotonic_trend"] = _is_monotonic(rates)
        if _is_non_decreasing(rates):
            result["trend_direction"] = "increasing"
        elif _is_non_increasing(rates):
            result["trend_direction"] = "decreasing"

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
        "top_risky_categories": [],
        "rare_category_summary": _rare_category_summary(frame["x"], config) if not frame.empty else {},
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
    result["top_risky_categories"] = result["category_response_rates"][:5]
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
    target_summary = _regression_target_summary(numeric_target, target)
    warnings = _base_response_warnings(target, config.response_column)
    warnings.extend(_regression_response_warnings(target_summary))
    relationships = []
    for profile in feature_profiles(type_summary, config):
        column = profile["name"]
        if profile["semantic_type"] in {"numeric_continuous", "numeric_discrete", "binary"} and pdt.is_numeric_dtype(df[column]):
            frame = pd.DataFrame({"x": pd.to_numeric(df[column], errors="coerce"), "y": numeric_target}).dropna()
            pearson = float(frame[["x", "y"]].corr(method="pearson").iloc[0, 1]) if len(frame) > 2 and frame["x"].nunique() > 1 else None
            spearman = float(frame[["x", "y"]].corr(method="spearman").iloc[0, 1]) if len(frame) > 2 and frame["x"].nunique() > 1 else None
            relationships.append(
                {
                    "column": column,
                    "feature_type": "numeric",
                    "correlation": pearson,
                    "pearson_correlation": pearson,
                    "spearman_correlation": spearman,
                    "signal_direction": _direction(spearman if spearman is not None else pearson),
                    "n": int(len(frame)),
                }
            )
        else:
            frame = pd.DataFrame({"x": df[column].astype("object"), "y": numeric_target}).dropna()
            grouped_all = frame.groupby("x")["y"].agg(["count", "mean"]).sort_values(["mean", "count"], ascending=[False, False])
            grouped = grouped_all.head(config.max_categories)
            means = grouped["mean"]
            spread = float(means.max() - means.min()) if len(means) > 1 else None
            relationships.append(
                {
                    "column": column,
                    "feature_type": "categorical",
                    "n": int(len(frame)),
                    "unique_count": int(frame["x"].nunique(dropna=True)),
                    "target_mean_spread": spread,
                    "category_target_means": [
                        {
                            "category": category,
                            "count": int(row["count"]),
                            "target_mean": float(row["mean"]),
                        }
                        for category, row in grouped.iterrows()
                    ],
                    "top_categories_by_target_mean": [
                        {
                            "category": category,
                            "count": int(row["count"]),
                            "target_mean": float(row["mean"]),
                        }
                        for category, row in grouped.head(5).iterrows()
                    ],
                    "rare_category_summary": _rare_category_summary(frame["x"], config) if not frame.empty else {},
                }
            )
    return to_jsonable(
        {
            "problem_type": "regression",
            "available": True,
            "response_column": config.response_column,
            "target_summary": target_summary,
            "warnings": warnings,
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
            "warnings": _base_response_warnings(target, str(target.name)),
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


def _binary_class_table(
    target: pd.Series,
    encoded: pd.Series,
    positive_class: Any,
    negative_class: Any,
) -> list[dict[str, Any]]:
    non_missing = target.dropna()
    counts = non_missing.value_counts(dropna=True)
    total = int(len(non_missing))
    rows = []
    for value in sorted(counts.index.tolist(), key=lambda item: str(item)):
        rows.append(
            {
                "class": value,
                "count": int(counts[value]),
                "percentage": float(counts[value] / max(total, 1)),
                "encoded_value": int(encoded[target == value].dropna().iloc[0]),
                "is_positive_class": bool(value == positive_class),
                "is_negative_class": bool(value == negative_class),
            }
        )
    return rows


def _base_response_warnings(target: pd.Series, response_column: str | None) -> list[dict[str, Any]]:
    missing_count = int(target.isna().sum())
    if not missing_count:
        return []
    return [
        _response_warning(
            issue_type="missing_response_values",
            severity="high",
            evidence=f"{missing_count} response values are missing.",
            recommended_action="Resolve or exclude records with missing response before supervised modeling.",
            metric_name="missing_count",
            metric_value=missing_count,
            threshold=0,
            response_column=response_column,
        )
    ]


def _regression_target_summary(numeric_target: pd.Series, original_target: pd.Series) -> dict[str, Any]:
    clean = numeric_target.dropna()
    quantiles = clean.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]) if not clean.empty else pd.Series(dtype=float)
    q1 = quantiles.loc[0.25] if not quantiles.empty else np.nan
    q3 = quantiles.loc[0.75] if not quantiles.empty else np.nan
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return to_jsonable(
        {
            "count": int(clean.count()),
            "missing_count": int(original_target.isna().sum() + (numeric_target.isna() & original_target.notna()).sum()),
            "missing_percentage": float(numeric_target.isna().mean()) if len(numeric_target) else 0.0,
            "mean": float(clean.mean()) if not clean.empty else None,
            "median": float(clean.median()) if not clean.empty else None,
            "std": float(clean.std()) if clean.count() > 1 else None,
            "variance": float(clean.var()) if clean.count() > 1 else None,
            "min": float(clean.min()) if not clean.empty else None,
            "max": float(clean.max()) if not clean.empty else None,
            "percentiles": {
                "p01": quantiles.loc[0.01] if not quantiles.empty else None,
                "p05": quantiles.loc[0.05] if not quantiles.empty else None,
                "p25": q1 if not quantiles.empty else None,
                "p50": quantiles.loc[0.50] if not quantiles.empty else None,
                "p75": q3 if not quantiles.empty else None,
                "p95": quantiles.loc[0.95] if not quantiles.empty else None,
                "p99": quantiles.loc[0.99] if not quantiles.empty else None,
            },
            "skewness": float(clean.skew()) if clean.count() > 2 else None,
            "iqr_outlier_count": int(((clean < lower) | (clean > upper)).sum()) if not clean.empty else 0,
            "zero_variance": bool(clean.nunique(dropna=True) <= 1),
        }
    )


def _regression_response_warnings(target_summary: dict[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if target_summary.get("zero_variance"):
        warnings.append(
            _response_warning(
                issue_type="zero_variance_regression_response",
                severity="high",
                evidence="Regression response has zero or near-zero observed variance.",
                recommended_action="Select a response with meaningful variation before modeling.",
                metric_name="unique_count",
                metric_value=1,
                threshold="> 1",
            )
        )
    outliers = int(target_summary.get("iqr_outlier_count") or 0)
    if outliers:
        warnings.append(
            _response_warning(
                issue_type="regression_response_outliers",
                severity="medium",
                evidence=f"{outliers} response values are outside the 1.5*IQR bounds.",
                recommended_action="Review response outliers for data quality issues or modeling treatment.",
                metric_name="iqr_outlier_count",
                metric_value=outliers,
                threshold=0,
            )
        )
    skewness = target_summary.get("skewness")
    if skewness is not None and abs(float(skewness)) >= REGRESSION_SKEW_WARNING_THRESHOLD:
        warnings.append(
            _response_warning(
                issue_type="skewed_regression_response",
                severity="medium",
                evidence=f"Regression response skewness is {float(skewness):.3f}.",
                recommended_action="Consider transformation, robust metrics, or segmentation before modeling.",
                metric_name="skewness_abs",
                metric_value=abs(float(skewness)),
                threshold=REGRESSION_SKEW_WARNING_THRESHOLD,
            )
        )
    return warnings


def _response_warning(
    issue_type: str,
    severity: str,
    evidence: str,
    recommended_action: str,
    metric_name: str,
    metric_value: Any,
    threshold: Any,
    response_column: str | None = None,
) -> dict[str, Any]:
    return {
        "warning_id": f"response.{issue_type}",
        "column": response_column,
        "scope": "response",
        "issue_type": issue_type,
        "severity": severity,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "threshold": threshold,
        "evidence": evidence,
        "recommended_action": recommended_action,
    }


def _direction(value: float | None) -> str | None:
    if value is None or pd.isna(value) or abs(float(value)) < 1e-12:
        return None
    return "positive" if float(value) > 0 else "negative"


def _is_non_decreasing(values: list[float]) -> bool:
    return all(left <= right for left, right in zip(values, values[1:]))


def _is_non_increasing(values: list[float]) -> bool:
    return all(left >= right for left, right in zip(values, values[1:]))


def _is_monotonic(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    return _is_non_decreasing(values) or _is_non_increasing(values)


def _rare_category_summary(series: pd.Series, config: EDAConfig) -> dict[str, Any]:
    counts = series.astype("object").where(series.notna(), "<MISSING>").value_counts(dropna=False)
    rare = counts[counts / max(len(series), 1) < config.rare_category_threshold]
    return {
        "rare_category_count": int(len(rare)),
        "rare_row_count": int(rare.sum()) if len(rare) else 0,
        "rare_row_percentage": float(rare.sum() / max(len(series), 1)) if len(rare) else 0.0,
        "threshold": config.rare_category_threshold,
    }
