from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = PROJECT_ROOT / "skills" / "tabular-eda-skill"

REQUIRED_SKILL_FILES = {
    "SKILL.md",
    "actions/run_full_eda.md",
    "actions/review_evidence_packet.md",
    "schemas/run_full_eda_input.schema.json",
    "schemas/run_full_eda_output.schema.json",
    "schemas/evidence_packet_contract.schema.json",
    "examples/basic_binary_eda.md",
    "references/current_contracts.md",
    "fallback.md",
    "validation_checklist.md",
}


def test_tabular_eda_skill_required_docs_exist() -> None:
    missing = [path for path in REQUIRED_SKILL_FILES if not (SKILL_DIR / path).exists()]

    assert missing == []


def test_tabular_eda_skill_docs_capture_core_guardrails() -> None:
    skill_text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    references_text = (SKILL_DIR / "references" / "current_contracts.md").read_text(encoding="utf-8")
    fallback_text = (SKILL_DIR / "fallback.md").read_text(encoding="utf-8")
    combined_text = "\n".join([skill_text, references_text, fallback_text])

    for required_term in ["evidence-only", "raw rows", "EDAConfig", "run_eda"]:
        assert required_term in combined_text


def test_tabular_eda_skill_schema_files_parse_as_json() -> None:
    schema_paths = sorted((SKILL_DIR / "schemas").glob("*.schema.json"))

    assert schema_paths
    for schema_path in schema_paths:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert payload["title"]
