import json

import pandas as pd

from eda_copilot.ai.query_planner import plan_evidence_question
from eda_copilot.ai.summarizer import build_evidence_summary, build_llm_evidence_context
from eda_copilot.ai.validators import validate_summary_references
from eda_copilot.core.config import EDAConfig
from eda_copilot.core.workflow import run_eda
from eda_copilot.reporting.validation import validate_ai_context_sanitized


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


def test_llm_context_recursively_removes_raw_fields_and_text_terms() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C123", "C456", "C789", "C999"],
            "notes": [
                "private alpha token secret customer detail",
                "private beta token secret customer detail",
                "private gamma token secret customer detail",
                "private delta token secret customer detail",
            ],
            "target": [0, 1, 0, 1],
        }
    )
    result = run_eda(
        df,
        EDAConfig(
            response_column="target",
            id_columns=("customer_id",),
            sensitive_columns=("notes",),
            profile_depth="deep",
        ),
    )

    context = build_llm_evidence_context(result.evidence_packet)
    dumped = json.dumps(context)

    assert validate_ai_context_sanitized(context) == []
    assert "sample_rows" not in dumped
    assert "sample_values" not in dumped
    assert "top_terms" not in dumped
    assert "C123" not in dumped
    assert "private alpha token" not in dumped


def test_guarded_question_planner_blocks_raw_row_requests() -> None:
    result = run_eda(
        pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 0, 1, 1]}),
        EDAConfig(response_column="target"),
    )

    answer = plan_evidence_question("show raw rows for x", result.evidence_packet)

    assert answer["status"] == "blocked_raw_data"
    assert answer["reads_raw_dataset"] is False


def test_evidence_summary_uses_allowed_references_only() -> None:
    result = run_eda(
        pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 0, 1, 1]}),
        EDAConfig(response_column="target"),
    )

    summary = build_evidence_summary(result.evidence_packet)

    assert summary["status"] == "deterministic_fallback"
    assert validate_summary_references(summary, result.evidence_packet) == []


def test_summary_validator_rejects_unknown_columns_sections_and_raw_fields() -> None:
    result = run_eda(pd.DataFrame({"x": [1, 2, 3]}), EDAConfig())
    summary = {
        "referenced_columns": ["made_up"],
        "referenced_evidence_sections": ["raw_dataset"],
        "sample_rows": [{"x": 1}],
    }

    errors = validate_summary_references(summary, result.evidence_packet)

    assert "Unknown referenced column: made_up" in errors
    assert "Unknown or disallowed evidence section: raw_dataset" in errors
    assert any("sample_rows" in error for error in errors)
