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
