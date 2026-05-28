from __future__ import annotations

import pandas as pd
import pytest

from eda_copilot.core.config import EDAConfig, EDAValidationError, validate_config_against_dataframe


def test_validate_config_rejects_duplicate_dataframe_columns() -> None:
    df = pd.DataFrame([[1, 2]], columns=["age", "age"])

    with pytest.raises(EDAValidationError, match="column names must be unique"):
        validate_config_against_dataframe(df, EDAConfig())


def test_validate_config_rejects_non_string_dataframe_columns() -> None:
    df = pd.DataFrame([[1, 2]], columns=["age", 10])

    with pytest.raises(EDAValidationError, match="column names must be strings"):
        validate_config_against_dataframe(df, EDAConfig())


def test_validate_config_rejects_blank_dataframe_columns() -> None:
    df = pd.DataFrame([[1, 2]], columns=["age", " "])

    with pytest.raises(EDAValidationError, match="column names must not be blank"):
        validate_config_against_dataframe(df, EDAConfig())


def test_validate_config_rejects_duplicate_configured_columns() -> None:
    df = pd.DataFrame({"application_id": ["A1"], "target": [1]})
    config = EDAConfig(id_columns=("application_id", "application_id"))

    with pytest.raises(EDAValidationError, match="duplicate entries"):
        validate_config_against_dataframe(df, config)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("problem_type", "typo", "problem_type must be one of"),
        ("profile_depth", "wide", "profile_depth must be one of"),
        ("sample_policy", "raw", "sample_policy must be one of"),
    ],
)
def test_validate_config_rejects_invalid_enum_values(field_name: str, value: str, message: str) -> None:
    df = pd.DataFrame({"x": [1], "target": [0]})
    config = EDAConfig(**{field_name: value})  # type: ignore[arg-type]

    with pytest.raises(EDAValidationError, match=message):
        validate_config_against_dataframe(df, config)


@pytest.mark.parametrize(
    "field_name",
    ["max_categories", "high_cardinality_threshold", "max_correlation_columns"],
)
def test_validate_config_rejects_non_positive_integer_knobs(field_name: str) -> None:
    df = pd.DataFrame({"x": [1], "target": [0]})
    config = EDAConfig(**{field_name: 0})

    with pytest.raises(EDAValidationError, match=f"{field_name} must be a positive integer"):
        validate_config_against_dataframe(df, config)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("rare_category_threshold", -0.01),
        ("high_missingness_threshold", 1.01),
        ("near_constant_threshold", -0.01),
        ("high_correlation_threshold", 1.01),
        ("high_auc_leakage_threshold", 1.01),
    ],
)
def test_validate_config_rejects_probability_thresholds_outside_unit_interval(field_name: str, value: float) -> None:
    df = pd.DataFrame({"x": [1], "target": [0]})
    config = EDAConfig(**{field_name: value})

    with pytest.raises(EDAValidationError, match=f"{field_name} must be between 0 and 1"):
        validate_config_against_dataframe(df, config)


def test_validate_config_rejects_invalid_drift_thresholds() -> None:
    df = pd.DataFrame({"x": [1], "target": [0]})

    with pytest.raises(EDAValidationError, match="Drift thresholds must be non-negative"):
        validate_config_against_dataframe(df, EDAConfig(drift_warning_threshold=-0.1))

    with pytest.raises(EDAValidationError, match="drift_warning_threshold must be less than or equal"):
        validate_config_against_dataframe(
            df,
            EDAConfig(drift_warning_threshold=0.6, drift_fail_threshold=0.5),
        )


def test_validate_config_rejects_cross_role_duplicate_columns() -> None:
    df = pd.DataFrame({"application_id": ["A1"], "target": [1]})
    config = EDAConfig(response_column="target", id_columns=("target",))

    with pytest.raises(EDAValidationError, match="multiple primary roles"):
        validate_config_against_dataframe(df, config)


def test_validate_config_rejects_excluded_role_columns() -> None:
    df = pd.DataFrame({"application_id": ["A1"], "target": [1]})
    config = EDAConfig(response_column="target", exclude_columns=("target",))

    with pytest.raises(EDAValidationError, match="exclude_columns should only contain extra predictor columns"):
        validate_config_against_dataframe(df, config)


def test_analysis_exclusions_include_non_predictor_roles() -> None:
    config = EDAConfig(
        response_column="target",
        id_columns=("application_id",),
        date_columns=("as_of_date",),
        train_test_column="split",
        segment_columns=("region",),
        sensitive_columns=("gender",),
        weight_column="weight",
        exclude_columns=("manual_exclude",),
    )

    assert config.analysis_exclusions() == {
        "application_id",
        "as_of_date",
        "gender",
        "manual_exclude",
        "region",
        "split",
        "target",
        "weight",
    }
