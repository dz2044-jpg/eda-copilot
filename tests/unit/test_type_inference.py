import json

import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.eda.type_inference import infer_column_types, profiles_by_name


def test_infer_column_types_flags_id_and_suspicious_name() -> None:
    df = pd.DataFrame(
        {
            "application_id": [f"A{i:03d}" for i in range(30)],
            "approved": [0, 1] * 15,
            "age": list(range(30)),
            "constant": ["x"] * 30,
        }
    )
    summary = infer_column_types(df, EDAConfig(response_column="approved"))
    profiles = profiles_by_name(summary)

    assert "id_candidate" in profiles["application_id"]["roles"]
    assert profiles["approved"]["suspicious_name"] is True
    assert profiles["constant"]["semantic_type"] == "constant"
    assert profiles["age"]["semantic_type"] == "numeric_continuous"


def test_infer_column_types_reports_parse_candidates_without_coercion() -> None:
    df = pd.DataFrame(
        {
            "amount_text": ["10.5", "20.0", "31.25", "42.0"],
            "event_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "approved_flag": ["yes", "no", "yes", "no"],
        }
    )

    summary = infer_column_types(df, EDAConfig())
    profiles = profiles_by_name(summary)

    assert profiles["amount_text"]["pandas_dtype"] in {"object", "str", "string"}
    assert profiles["amount_text"]["semantic_type"] == "categorical_low_cardinality"
    assert profiles["amount_text"]["numeric_parse_rate"] == 1.0
    assert "numeric_parse_candidate" in profiles["amount_text"]["schema_warnings"]
    assert profiles["event_date"]["semantic_type"] == "datetime"
    assert profiles["event_date"]["datetime_parse_rate"] == 1.0
    assert profiles["approved_flag"]["semantic_type"] == "boolean"
    assert profiles["approved_flag"]["boolean_parse_rate"] == 1.0
    assert summary["summary"]["parse_candidate_columns"] == {
        "numeric": ["amount_text"],
        "datetime": ["event_date"],
        "boolean": ["approved_flag"],
    }
    json.dumps(summary)


def test_infer_column_types_reports_column_name_schema_metadata() -> None:
    df = pd.DataFrame(
        {
            " First Name ": ["A", "B"],
            "first_name": ["C", "D"],
            "Unnamed: 0": [1, 2],
        }
    )

    summary = infer_column_types(df, EDAConfig())
    profiles = profiles_by_name(summary)

    assert profiles[" First Name "]["normalized_name"] == "first_name"
    assert "column_name_whitespace" in profiles[" First Name "]["name_warnings"]
    assert "unnamed_column" in profiles["Unnamed: 0"]["name_warnings"]
    assert summary["summary"]["normalized_name_collisions"] == [
        {"normalized_name": "first_name", "columns": [" First Name ", "first_name"]}
    ]
