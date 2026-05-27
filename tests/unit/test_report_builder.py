import pandas as pd

from eda_copilot.core.config import EDAConfig
from eda_copilot.core.workflow import run_eda


def test_report_builder_includes_required_sections() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 0, 1, 1]})

    result = run_eda(df, EDAConfig(response_column="target"))
    report = result.markdown_report

    assert "# EDA Report:" in report
    assert "## Data Quality Findings" in report
    assert "## Feature Ranking" in report
    assert "## Leakage Warnings" in report
