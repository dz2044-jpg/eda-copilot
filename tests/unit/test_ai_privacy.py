import pandas as pd

from eda_copilot.ai.query_planner import plan_evidence_question
from eda_copilot.ai.summarizer import build_llm_evidence_context
from eda_copilot.core.config import EDAConfig
from eda_copilot.core.workflow import run_eda


def test_llm_context_removes_row_samples_and_sample_values() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "income": [10, 20, 30, 40],
            "target": [0, 0, 1, 1],
        }
    )
    result = run_eda(
        df,
        EDAConfig(response_column="target", id_columns=("customer_id",)),
    )

    context = build_llm_evidence_context(result.evidence_packet)
    overview = context["dataset_overview"]

    assert "sample_rows" not in overview
    assert all("sample_values" not in row for row in overview["data_dictionary"])
    assert overview["row_samples_removed_for_ai"] is True


def test_guarded_question_planner_blocks_raw_row_requests() -> None:
    result = run_eda(
        pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 0, 1, 1]}),
        EDAConfig(response_column="target"),
    )

    answer = plan_evidence_question("show raw rows for x", result.evidence_packet)

    assert answer["status"] == "blocked_raw_data"
    assert answer["reads_raw_dataset"] is False
