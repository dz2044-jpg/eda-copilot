# Skill Direction

The `tabular-eda-skill` should wrap the existing EDA Copilot workflow instead
of replacing it. Its purpose is to give agents a clear operating procedure,
action contracts, guardrails, fallback behavior, and validation checklist for
tabular EDA work.

## Skill Scope

The first skill version is documentation-only. It defines how a future agent
should call or orchestrate the current workflow, but it does not add runtime
skill execution, new calculations, or new public Python APIs.

The skill should align with:

- `EDAConfig` for input configuration.
- `run_eda` for deterministic orchestration.
- The evidence packet for downstream reporting and optional AI interpretation.
- Exported artifacts for review, audit, and handoff.

## Expected Actions

- Run a full deterministic EDA workflow over a validated tabular dataframe.
- Inspect evidence packet sections and artifact outputs.
- Explain deterministic warnings, quality checks, and caveats using
  evidence-only context.
- Draft future reports from evidence packets without inventing metrics.

## Evidence-Only AI Rule

Any AI-facing action must read sanitized evidence packet content only. It must
not inspect raw rows, row samples, ID values, sensitive row-level values, or
private data. If required evidence is missing, the skill should report the gap
and use a deterministic fallback rather than inventing findings.

## Future Expansion

Future PRs may add runtime action handlers, formal schemas, richer AI summary
actions, model-readiness scoring, run history, and domain adapters. Those
changes should extend the current deterministic foundation and include focused
tests.
