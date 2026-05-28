from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
from numbers import Integral, Real
from typing import Any, Literal, get_args

import pandas as pd


ProblemType = Literal[
    "auto",
    "binary_classification",
    "multiclass_classification",
    "regression",
    "unsupervised",
]
ProfileDepth = Literal["minimal", "standard", "deep"]
SamplePolicy = Literal["redacted", "preview", "none"]

VALID_PROBLEM_TYPES = set(get_args(ProblemType))
VALID_PROFILE_DEPTHS = set(get_args(ProfileDepth))
VALID_SAMPLE_POLICIES = set(get_args(SamplePolicy))


class EDAValidationError(ValueError):
    """Raised when user-supplied data or configuration is invalid."""


@dataclass(frozen=True)
class EDAConfig:
    """Configuration for one deterministic EDA run.

    Attributes:
        dataset_name: Human-readable dataset label used in reports and artifacts.
        response_column: Optional response or target column.
        problem_type: Problem framing. Use `auto` to infer from the response.
        id_columns: Columns treated as entity or record identifiers.
        date_columns: Columns the user wants treated as dates.
        train_test_column: Optional split indicator for drift analysis.
        segment_columns: Optional segment columns for grouped diagnostics.
        exclude_columns: Columns excluded from feature-level analysis.
        sensitive_columns: Optional protected columns for future fairness analysis.
        weight_column: Optional observation weight column.
        profile_depth: Controls how much profiling detail is generated.
        sample_policy: Controls whether row previews are redacted, raw, or omitted.
        drift_warning_threshold: Absolute drift metric value that emits a warning check.
        drift_fail_threshold: Absolute drift metric value that fails a quality gate.
    """

    dataset_name: str = "dataset"
    response_column: str | None = None
    problem_type: ProblemType = "auto"
    id_columns: tuple[str, ...] = field(default_factory=tuple)
    date_columns: tuple[str, ...] = field(default_factory=tuple)
    train_test_column: str | None = None
    segment_columns: tuple[str, ...] = field(default_factory=tuple)
    exclude_columns: tuple[str, ...] = field(default_factory=tuple)
    sensitive_columns: tuple[str, ...] = field(default_factory=tuple)
    weight_column: str | None = None
    profile_depth: ProfileDepth = "standard"
    sample_policy: SamplePolicy = "redacted"
    max_categories: int = 20
    rare_category_threshold: float = 0.01
    high_cardinality_threshold: int = 50
    high_missingness_threshold: float = 0.40
    near_constant_threshold: float = 0.98
    high_correlation_threshold: float = 0.80
    high_auc_leakage_threshold: float = 0.98
    max_correlation_columns: int = 50
    drift_warning_threshold: float = 0.20
    drift_fail_threshold: float = 0.50

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly configuration dictionary."""

        return asdict(self)

    def analysis_exclusions(self) -> set[str]:
        """Return columns excluded from predictor-level calculations.

        These columns can still appear in dataset-level profiling, univariate
        summaries, and grouped diagnostics. They are excluded from feature-like
        calculations such as response relationships, bivariate correlations,
        feature ranking, and modeling-risk predictor review.
        """

        excluded = set(self.exclude_columns)
        excluded.update(self.id_columns)
        excluded.update(self.date_columns)
        excluded.update(self.segment_columns)
        excluded.update(self.sensitive_columns)
        if self.response_column:
            excluded.add(self.response_column)
        if self.weight_column:
            excluded.add(self.weight_column)
        if self.train_test_column:
            excluded.add(self.train_test_column)
        return excluded


def validate_config_against_dataframe(df: pd.DataFrame, config: EDAConfig) -> None:
    """Validate that the input dataset and selected configuration are usable.

    Args:
        df: Uploaded or pre-loaded dataset.
        config: User-selected EDA configuration.

    Raises:
        EDAValidationError: If the dataset is empty or configured columns are absent.
    """

    if len(df.columns) == 0:
        raise EDAValidationError("The dataset has no columns.")
    if len(df) == 0:
        raise EDAValidationError("The dataset is empty. Upload a file with at least one row.")
    _validate_dataframe_column_labels(df)
    _validate_config_values(config)
    _validate_configured_column_lists(config)

    configured_columns: dict[str, str | tuple[str, ...] | None] = {
        "response_column": config.response_column,
        "id_columns": config.id_columns,
        "date_columns": config.date_columns,
        "train_test_column": config.train_test_column,
        "segment_columns": config.segment_columns,
        "exclude_columns": config.exclude_columns,
        "sensitive_columns": config.sensitive_columns,
        "weight_column": config.weight_column,
    }
    missing: list[str] = []
    for label, value in configured_columns.items():
        if value is None:
            continue
        values = (value,) if isinstance(value, str) else value
        for column in values:
            if column not in df.columns:
                missing.append(f"{label}={column}")

    if missing:
        joined = ", ".join(missing)
        raise EDAValidationError(f"Configured columns are missing from the dataset: {joined}.")


def _validate_dataframe_column_labels(df: pd.DataFrame) -> None:
    duplicate_columns = [str(column) for column in df.columns[df.columns.duplicated()].unique()]
    if duplicate_columns:
        joined = ", ".join(duplicate_columns)
        raise EDAValidationError(f"Dataset column names must be unique. Duplicates: {joined}.")

    non_string_columns = [repr(column) for column in df.columns if not isinstance(column, str)]
    if non_string_columns:
        joined = ", ".join(non_string_columns)
        raise EDAValidationError(f"Dataset column names must be strings. Invalid columns: {joined}.")

    blank_columns = [repr(column) for column in df.columns if not column.strip()]
    if blank_columns:
        joined = ", ".join(blank_columns)
        raise EDAValidationError(f"Dataset column names must not be blank. Invalid columns: {joined}.")


def _validate_config_values(config: EDAConfig) -> None:
    _validate_choice("problem_type", config.problem_type, VALID_PROBLEM_TYPES)
    _validate_choice("profile_depth", config.profile_depth, VALID_PROFILE_DEPTHS)
    _validate_choice("sample_policy", config.sample_policy, VALID_SAMPLE_POLICIES)

    for label in ("max_categories", "high_cardinality_threshold", "max_correlation_columns"):
        _validate_positive_integer(label, getattr(config, label))

    for label in (
        "rare_category_threshold",
        "high_missingness_threshold",
        "near_constant_threshold",
        "high_correlation_threshold",
        "high_auc_leakage_threshold",
    ):
        _validate_probability(label, getattr(config, label))

    warning = _validate_finite_number("drift_warning_threshold", config.drift_warning_threshold)
    fail = _validate_finite_number("drift_fail_threshold", config.drift_fail_threshold)
    if warning < 0 or fail < 0:
        raise EDAValidationError("Drift thresholds must be non-negative.")
    if warning > fail:
        raise EDAValidationError("drift_warning_threshold must be less than or equal to drift_fail_threshold.")


def _validate_choice(label: str, value: Any, allowed: set[str]) -> None:
    if value not in allowed:
        joined = ", ".join(sorted(allowed))
        raise EDAValidationError(f"{label} must be one of: {joined}.")


def _validate_positive_integer(label: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise EDAValidationError(f"{label} must be a positive integer.")
    if int(value) < 1:
        raise EDAValidationError(f"{label} must be a positive integer.")


def _validate_probability(label: str, value: Any) -> None:
    number = _validate_finite_number(label, value)
    if number < 0 or number > 1:
        raise EDAValidationError(f"{label} must be between 0 and 1.")


def _validate_finite_number(label: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EDAValidationError(f"{label} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise EDAValidationError(f"{label} must be a finite number.")
    return number


def _validate_configured_column_lists(config: EDAConfig) -> None:
    configured_column_groups = {
        "id_columns": config.id_columns,
        "date_columns": config.date_columns,
        "segment_columns": config.segment_columns,
        "exclude_columns": config.exclude_columns,
        "sensitive_columns": config.sensitive_columns,
    }
    duplicate_configured_columns = []
    for label, values in configured_column_groups.items():
        duplicates = sorted(item for item, count in Counter(values).items() if count > 1)
        duplicate_configured_columns.extend(f"{label}={column}" for column in duplicates)

    if duplicate_configured_columns:
        joined = ", ".join(duplicate_configured_columns)
        raise EDAValidationError(f"Configured column lists contain duplicate entries: {joined}.")

    primary_role_groups = {
        "response_column": (config.response_column,) if config.response_column else (),
        "id_columns": config.id_columns,
        "date_columns": config.date_columns,
        "train_test_column": (config.train_test_column,) if config.train_test_column else (),
        "segment_columns": config.segment_columns,
        "sensitive_columns": config.sensitive_columns,
        "weight_column": (config.weight_column,) if config.weight_column else (),
    }
    roles_by_column: dict[str, list[str]] = {}
    for label, values in primary_role_groups.items():
        for column in values:
            roles_by_column.setdefault(column, []).append(label)

    cross_role_duplicates = [
        f"{column} ({', '.join(labels)})"
        for column, labels in sorted(roles_by_column.items())
        if len(labels) > 1
    ]
    if cross_role_duplicates:
        joined = ", ".join(cross_role_duplicates)
        raise EDAValidationError(f"Configured columns cannot be assigned to multiple primary roles: {joined}.")

    primary_columns = set(roles_by_column)
    excluded_role_columns = sorted(set(config.exclude_columns) & primary_columns)
    if excluded_role_columns:
        joined = ", ".join(excluded_role_columns)
        raise EDAValidationError(
            "exclude_columns should only contain extra predictor columns; "
            f"remove role columns from exclude_columns: {joined}."
        )
