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
