# Action: Review Evidence Packet

## Purpose

Review, explain, or draft from an existing EDA Copilot evidence packet while
preserving the evidence-only AI boundary.

## Inputs

- A JSON-serializable evidence packet produced by `run_eda`.
- Optional user focus, such as data quality, response readiness, leakage,
  drift, feature ranking, visual specs, or report drafting.

## Allowed Evidence

Use packet sections documented in `../references/current_contracts.md`, such as
`dataset_overview`, `profile_summary`, `quality_checks`, `response_summary`,
`feature_ranking`, `leakage_warnings`, `drift_summary`,
`modeling_risk_summary`, `comparison_summary`, `visual_specs`, and `caveats`.

For AI-facing work, use sanitized evidence context only. In code, this means the
allowed context should follow `build_llm_evidence_context` behavior: no raw
rows, no row samples, and no data-dictionary sample values.

## Outputs

- Observed facts grounded in packet fields.
- Calculated metrics copied or summarized from deterministic evidence.
- Risks and caveats clearly labeled as interpretation.
- Modeling risk signals framed as deterministic review aids, not model approval.
- Missing evidence or unknowns called out explicitly.
- Next-step recommendations framed as review actions, not final decisions.

## Guardrails

- Do not invent columns, metrics, tests, business explanations, or model
  conclusions.
- Do not request or expose raw rows to answer an explanation request.
- Do not treat feature ranking as causality or model validation.
- If the packet is incomplete, use `../fallback.md`.
