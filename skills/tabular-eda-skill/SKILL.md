---
name: tabular-eda-skill
description: Use for deterministic-first exploratory data analysis of tabular datasets with EDA Copilot, including configuring EDAConfig, running or inspecting run_eda outputs, reviewing evidence packets and artifacts, and drafting evidence-only explanations without raw rows or sensitive row-level values.
---

# Tabular EDA Skill

Use this skill when the task involves repeatable exploratory data analysis for a
tabular dataset, especially when the user needs data quality review, response
analysis, feature relationship triage, leakage checks, drift checks, exported
artifacts, or evidence-grounded reporting.

## Core Rules

- Treat `eda_copilot.core.workflow.run_eda` as the deterministic source of
  truth.
- Map user options to `eda_copilot.core.config.EDAConfig`; do not invent
  configuration fields.
- Keep Streamlit as an interface only. Do not place deterministic calculations
  in UI code.
- Any AI-facing explanation must use evidence-only context. Do not inspect or
  send raw rows, row samples, ID values, sensitive row-level values, or private
  data to AI interpretation.
- If evidence is missing, say what is missing and use the fallback behavior
  rather than inventing metrics, columns, or conclusions.

## Workflow

1. Confirm the dataset is tabular and the requested analysis can be expressed
   through `EDAConfig` and existing EDA Copilot outputs.
2. Run or inspect the deterministic workflow described in
   `actions/run_full_eda.md`.
3. Review evidence packet sections and exported artifacts using
   `references/current_contracts.md`.
4. For explanation or drafting tasks, use `actions/review_evidence_packet.md`
   and keep every claim tied to packet fields.
5. Apply `fallback.md` when input data, configuration, evidence, or optional AI
   support is unavailable.
6. Before handoff, use `validation_checklist.md`.

## Bundled References

- `actions/run_full_eda.md`: action contract for deterministic full EDA runs.
- `actions/review_evidence_packet.md`: action contract for evidence-only review
  and explanation.
- `schemas/*.schema.json`: static documentation-level contracts for action
  inputs, outputs, and required evidence packet keys.
- `examples/basic_binary_eda.md`: minimal example request and expected handling.
- `references/current_contracts.md`: current `EDAConfig`, evidence packet, and
  artifact baseline.
- `fallback.md`: required fallback behavior.
- `validation_checklist.md`: completion checklist.
