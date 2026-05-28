from __future__ import annotations

import re
from typing import Any

import pandas as pd

from eda_copilot.core.config import EDAConfig


def detect_data_quality_issues(
    df: pd.DataFrame,
    config: EDAConfig,
    type_summary: dict[str, Any],
    missingness_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create a standardized table of deterministic data quality warnings."""

    warnings: list[dict[str, Any]] = []
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        warnings.append(
            _warning(
                column=None,
                issue_type="duplicate_rows",
                severity="medium",
                evidence=f"{duplicate_rows} duplicated rows detected.",
                recommended_action="Confirm whether duplicate records are expected before modeling.",
                scope="dataset",
                metric_name="duplicate_row_count",
                metric_value=duplicate_rows,
                threshold=0,
            )
        )

    for id_column in config.id_columns:
        if id_column in df.columns:
            duplicate_ids = int(df[id_column].duplicated().sum())
            if duplicate_ids:
                warnings.append(
                    _warning(
                        column=id_column,
                    issue_type="duplicate_id_values",
                    severity="high",
                    evidence=f"{duplicate_ids} duplicated ID values detected.",
                    recommended_action="Check entity grain and remove or reconcile duplicate IDs.",
                    scope="column",
                    metric_name="duplicate_id_count",
                    metric_value=duplicate_ids,
                    threshold=0,
                )
            )

    warnings.extend(_schema_summary_warnings(type_summary))

    missing_by_column = {
        item["column"]: item for item in missingness_summary.get("columns", [])
    }
    for profile in type_summary.get("columns", []):
        column = profile["name"]
        semantic_type = profile["semantic_type"]
        missing = missing_by_column.get(column, {})
        missing_pct = float(profile["missing_percentage"])
        if semantic_type == "all_missing":
            warnings.append(
                _warning(
                    column=column,
                    issue_type="all_missing",
                    severity="high",
                    evidence="Column has no observed values.",
                    recommended_action="Exclude or fix upstream extraction.",
                    metric_name="non_null_count",
                    metric_value=0,
                    threshold="> 0",
                )
            )
        elif missing_pct >= config.high_missingness_threshold:
            warnings.append(
                _warning(
                    column,
                    "high_missingness",
                    "medium",
                    f"{missing_pct:.1%} of values are missing.",
                    "Assess missingness mechanism and consider imputation or exclusion.",
                    metric_name="missing_percentage",
                    metric_value=missing_pct,
                    threshold=config.high_missingness_threshold,
                )
            )
        if semantic_type == "constant":
            warnings.append(
                _warning(
                    column,
                    "constant_column",
                    "medium",
                    "Column has one non-null value.",
                    "Exclude from modeling.",
                    metric_name="unique_count",
                    metric_value=profile.get("unique_count"),
                    threshold="> 1",
                )
            )
        if semantic_type == "near_constant":
            warnings.append(
                _warning(
                    column,
                    "near_constant_column",
                    "low",
                    f"Top value ratio is {profile['top_frequency_ratio']:.1%}.",
                    "Verify whether the rare values are meaningful before modeling.",
                    metric_name="top_frequency_ratio",
                    metric_value=profile.get("top_frequency_ratio"),
                    threshold=config.near_constant_threshold,
                )
            )
        if semantic_type == "categorical_high_cardinality":
            warnings.append(
                _warning(
                    column,
                    "high_cardinality",
                    "medium",
                    f"{profile['unique_count']} unique values detected.",
                    "Consider grouping, target encoding with validation, or excluding if ID-like.",
                    metric_name="unique_count",
                    metric_value=profile.get("unique_count"),
                    threshold=config.high_cardinality_threshold,
                )
            )
        if "id_candidate" in profile.get("roles", []) and column not in config.id_columns:
            warnings.append(
                _warning(
                    column,
                    "id_like_candidate",
                    "low",
                    "Column name or uniqueness pattern looks ID-like.",
                    "Confirm whether this should be excluded as an identifier.",
                    metric_name="unique_ratio",
                    metric_value=profile.get("unique_ratio"),
                    threshold="confirm role",
                )
            )
        if profile.get("suspicious_name") and column != config.response_column:
            warnings.append(
                _warning(
                    column,
                    "suspicious_column_name",
                    "medium",
                    "Column name contains outcome, decision, score, or post-event wording.",
                    "Check timing and business meaning for leakage risk.",
                    metric_name="suspicious_name",
                    metric_value=True,
                    threshold=False,
                )
            )
        blank_count = int(missing.get("blank_string_count") or 0)
        if blank_count:
            warnings.append(
                _warning(
                    column,
                    "blank_strings",
                    "low",
                    f"{blank_count} blank string values detected.",
                    "Normalize blank strings to missing values in preprocessing.",
                    metric_name="blank_string_count",
                    metric_value=blank_count,
                    threshold=0,
                )
            )
        special_codes = missing.get("special_missing_code_counts") or {}
        if special_codes:
            warnings.append(
                _warning(
                    column,
                    "special_missing_codes",
                    "low",
                    f"Special missing-like codes detected: {special_codes}.",
                    "Confirm whether these codes should be converted to missing values.",
                    metric_name="special_missing_code_count",
                    metric_value=int(sum(int(value) for value in special_codes.values())),
                    threshold=0,
                )
            )
        mixed_types = _mixed_python_type_count(df[column])
        if mixed_types > 1 and df[column].dtype == "object":
            warnings.append(
                _warning(
                    column,
                    "mixed_python_types",
                    "low",
                    f"{mixed_types} Python value types detected in object column.",
                    "Normalize values to a single semantic type before modeling.",
                    metric_name="python_type_count",
                    metric_value=mixed_types,
                    threshold=1,
                )
            )
    return warnings


def _schema_summary_warnings(type_summary: dict[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    summary = type_summary.get("summary", {})
    collisions = summary.get("normalized_name_collisions", [])
    if collisions:
        collision_names = [
            f"{item.get('normalized_name')}: {', '.join(map(str, item.get('columns', [])))}"
            for item in collisions[:5]
        ]
        warnings.append(
            _warning(
                column=None,
                issue_type="normalized_name_collision",
                severity="medium",
                evidence=f"Normalized column name collisions detected: {'; '.join(collision_names)}.",
                recommended_action="Rename colliding columns before downstream modeling or export automation.",
                scope="schema",
                metric_name="normalized_name_collision_count",
                metric_value=len(collisions),
                threshold=0,
            )
        )

    profiles = {
        str(profile.get("name")): profile for profile in type_summary.get("columns", [])
    }
    parse_candidates = summary.get("parse_candidate_columns", {})
    thresholds = {
        "numeric": 0.95,
        "datetime": 0.85,
        "boolean": 1.0,
    }
    for kind, columns in parse_candidates.items():
        for column in columns:
            profile = profiles.get(str(column), {})
            metric_name = f"{kind}_parse_rate"
            metric_value = profile.get(metric_name)
            warnings.append(
                _warning(
                    column=str(column),
                    issue_type=f"{kind}_parse_candidate",
                    severity="low",
                    evidence=f"Column values parse as {kind} at rate {float(metric_value or 0.0):.1%}.",
                    recommended_action=f"Confirm whether the column should be converted to {kind} before analysis.",
                    metric_name=metric_name,
                    metric_value=metric_value,
                    threshold=thresholds.get(str(kind)),
                )
            )
    return warnings


def _mixed_python_type_count(series: pd.Series) -> int:
    return int(series.dropna().map(lambda value: type(value).__name__).nunique(dropna=True))


def _warning(
    column: str | None,
    issue_type: str,
    severity: str,
    evidence: str,
    recommended_action: str,
    scope: str | None = None,
    metric_name: str | None = None,
    metric_value: Any | None = None,
    threshold: Any | None = None,
) -> dict[str, Any]:
    warning_scope = scope or ("column" if column else "dataset")
    return {
        "warning_id": f"{issue_type}.{_stable_token(column or warning_scope)}",
        "column": column,
        "scope": warning_scope,
        "issue_type": issue_type,
        "severity": severity,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "threshold": threshold,
        "evidence": evidence,
        "recommended_action": recommended_action,
    }


def _stable_token(value: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")
    return token or "dataset"
