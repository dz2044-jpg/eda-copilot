import json

import pandas as pd

from eda_copilot.ai.summarizer import build_llm_evidence_context
from eda_copilot.core.config import EDAConfig
from eda_copilot.core.workflow import run_eda
from eda_copilot.reporting.validation import (
    validate_ai_context_sanitized,
    validate_artifact_manifest,
    validate_evidence_packet,
)


def test_validate_evidence_packet_accepts_current_contract() -> None:
    result = run_eda(pd.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4]}), EDAConfig())

    assert validate_evidence_packet(result.evidence_packet) == []


def test_validate_evidence_packet_reports_missing_sections() -> None:
    errors = validate_evidence_packet({"metadata": {}})

    assert any("missing required sections" in error for error in errors)


def test_validate_ai_context_sanitized_rejects_row_samples() -> None:
    context = {
        "dataset_overview": {
            "sample_rows": [{"x": 1}],
            "data_dictionary": [{"column": "x", "sample_values": [1]}],
        }
    }

    errors = validate_ai_context_sanitized(context)

    assert "dataset_overview.sample_rows" in errors[0]
    assert any("sample_values" in error for error in errors)


def test_validate_artifact_manifest_checks_required_fields(tmp_path) -> None:
    manifest = {"files": [{"file_name": "evidence_packet.json"}]}

    errors = validate_artifact_manifest(manifest, tmp_path)

    assert any("relative_path" in error for error in errors)


def test_llm_context_from_workflow_is_sanitized() -> None:
    result = run_eda(
        pd.DataFrame({"id": ["A", "B"], "x": [1, 2]}),
        EDAConfig(id_columns=("id",)),
    )

    context = build_llm_evidence_context(result.evidence_packet)

    assert validate_ai_context_sanitized(context) == []
    assert json.dumps(context)
