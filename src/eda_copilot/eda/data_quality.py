from __future__ import annotations

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
                    )
                )

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
                _warning(column, "all_missing", "high", "Column has no observed values.", "Exclude or fix upstream extraction.")
            )
        elif missing_pct >= config.high_missingness_threshold:
            warnings.append(
                _warning(
                    column,
                    "high_missingness",
                    "medium",
                    f"{missing_pct:.1%} of values are missing.",
                    "Assess missingness mechanism and consider imputation or exclusion.",
                )
            )
        if semantic_type == "constant":
            warnings.append(
                _warning(column, "constant_column", "medium", "Column has one non-null value.", "Exclude from modeling.")
            )
        if semantic_type == "near_constant":
            warnings.append(
                _warning(
                    column,
                    "near_constant_column",
                    "low",
                    f"Top value ratio is {profile['top_frequency_ratio']:.1%}.",
                    "Verify whether the rare values are meaningful before modeling.",
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
) -> dict[str, Any]:
    return {
        "column": column,
        "issue_type": issue_type,
        "severity": severity,
        "evidence": evidence,
        "recommended_action": recommended_action,
    }
