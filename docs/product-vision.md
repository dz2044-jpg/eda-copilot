# Product Vision

EDA Copilot should become a skill-first analytical product for repeatable,
auditable tabular EDA. The product direction is to keep deterministic Python
analysis as the source of truth while allowing optional AI interpretation to
summarize approved evidence packets.

## Positioning

EDA Copilot is not a general chat-over-data tool. It is a deterministic EDA
workflow that can later be exposed through skill actions, application screens,
and report-generation flows. The product should help users inspect data quality,
response readiness, feature relationships, leakage risk, drift, and reporting
artifacts before modeling decisions are made.

## Trust Model

- Python computes metrics, warnings, ranks, quality gates, visual specs, and
  export artifacts.
- Evidence packets are the reviewable contract between deterministic analysis,
  reporting, and optional AI interpretation.
- AI can explain, summarize, draft, or triage from evidence-only context.
- Humans remain responsible for business interpretation, modeling decisions,
  remediation choices, and final report approval.

## Near-Term Direction

- Protect the current deterministic baseline before expanding behavior.
- Add a `tabular-eda-skill` skeleton that documents action contracts and
  evidence-only guardrails.
- Strengthen deterministic modules in small follow-up PRs rather than creating
  a duplicate tool stack.
- Harden evidence packet schemas, artifact manifests, and audit trails before
  adding richer AI summaries.

## Product Guardrails

- Keep Streamlit as an interface over reusable workflow logic.
- Keep deterministic logic separate from prompts and generated text.
- Keep AI context sanitized and evidence-only.
- Do not expose raw rows or sensitive row-level values to AI interpretation.
- Favor inspectable artifacts over opaque generated conclusions.
