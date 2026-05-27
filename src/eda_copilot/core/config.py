from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd


ProblemType = Literal[
    "auto",
    "binary_classification",
    "multiclass_classification",
    "regression",
    "unsupervised",
]


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
    max_categories: int = 20
    rare_category_threshold: float = 0.01
    high_cardinality_threshold: int = 50
    high_missingness_threshold: float = 0.40
    near_constant_threshold: float = 0.98
    high_correlation_threshold: float = 0.80
    high_auc_leakage_threshold: float = 0.98
    max_correlation_columns: int = 50

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly configuration dictionary."""

        return asdict(self)

    def analysis_exclusions(self) -> set[str]:
        """Return columns excluded from predictor-level calculations."""

        excluded = set(self.exclude_columns)
        if self.response_column:
            excluded.add(self.response_column)
        if self.weight_column:
            excluded.add(self.weight_column)
        return excluded


def validate_config_against_dataframe(df: pd.DataFrame, config: EDAConfig) -> None:
    """Validate that the input dataset and selected configuration are usable.

    Args:
        df: Uploaded or pre-loaded dataset.
        config: User-selected EDA configuration.

    Raises:
        EDAValidationError: If the dataset is empty or configured columns are absent.
    """

    if df.empty:
        raise EDAValidationError("The dataset is empty. Upload a file with at least one row.")
    if len(df.columns) == 0:
        raise EDAValidationError("The dataset has no columns.")

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
