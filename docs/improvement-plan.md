# EDA Copilot Improvement Plan

This document is the working implementation plan for turning EDA Copilot into a
skill-first analytical product. Use it to make small, reviewable changes one
phase or PR at a time.

## Current Baseline

Status: In progress

- EDA Copilot already follows a deterministic-first architecture:
  - `eda_copilot.eda` owns profiling, type inference, missingness, response
    analysis, bivariate analysis, feature ranking, leakage checks, drift, and
    quality checks.
  - `eda_copilot.core.workflow.run_eda` orchestrates the deterministic workflow.
  - `eda_copilot.reporting` builds evidence packets, reports, and exported
    artifacts.
  - `eda_copilot.ai` is optional and must read sanitized evidence, not raw data.
  - `eda_copilot.app` provides the Streamlit interface and should stay thin.
- Current test command:

  ```bash
  uv run pytest
  ```

- Current trust boundary:
  - Python calculates metrics, warnings, artifacts, and validation outputs.
  - AI interprets and drafts from approved evidence packets only.
  - Humans make final analytical, modeling, and business judgments.

## Guiding Principles

- Preserve the existing `src/eda_copilot/eda` architecture unless a future PR has
  a clear migration plan and test coverage.
- Keep Streamlit as an interface over reusable workflow logic.
- Keep deterministic business logic separate from AI interpretation and prompts.
- Keep AI evidence-only; do not expose raw rows or sensitive row-level values.
- Prefer small, focused, reviewable PRs over broad rewrites.
- Add tests when behavior changes.
- Do not remove the current workflow while adding skill-first capabilities.
- Document assumptions, limitations, and acceptance gates as features are added.

## Phase Checklist

- [x] Phase 0: Freeze current baseline
  - Status: Completed
  - Document the current app purpose, deterministic modules, evidence packet
    fields, artifact outputs, AI privacy behavior, and known limitations.
  - Add baseline tests for top-level evidence packet keys and exported artifact
    names.
  - Acceptance gate: current workflow behavior is unchanged and `uv run pytest`
    passes.

- [x] Phase 1: Add `tabular-eda-skill` skeleton
  - Status: Completed
  - Create `skills/tabular-eda-skill/` with `SKILL.md`, action docs, schemas,
    examples, references, fallback behavior, and a validation checklist.
  - Align skill actions with existing `EDAConfig`, `run_eda`, evidence packet,
    report, and artifact concepts.
  - Acceptance gate: skill docs define trigger rules, guardrails, action
    contracts, fallback behavior, and evidence-only AI rules.

- [ ] Phase 2: Strengthen deterministic EDA foundation
  - Status: In progress
  - Extend the existing `eda_copilot.eda` modules rather than creating a
    duplicate deterministic tools package.
  - Improve loading, schema inspection, profiling, response analysis, feature
    relationships, missingness, leakage, drift, and modeling-risk checks as
    scoped by future PRs.
  - Acceptance gate: deterministic outputs are JSON-serializable, stable across
    repeated runs, and covered by focused tests.

- [ ] Phase 3: Harden evidence packet and schemas
  - Status: Not started
  - Add or formalize schemas for the EDA evidence packet, action inputs, action
    outputs, and artifact manifests.
  - Ensure packet fields map back to deterministic outputs and exclude raw rows
    from AI context.
  - Acceptance gate: evidence packet validation passes, sensitive columns remain
    excluded from AI context, and every quantitative AI claim can cite packet
    fields.

- [ ] Phase 4: Standardize artifacts and audit trail
  - Status: Not started
  - Standardize each run folder around input manifest, config snapshot,
    validation report, evidence packet, deterministic reports, AI summaries,
    final report, visual specs, and audit log.
  - Save file hashes and run metadata where available.
  - Acceptance gate: every full run creates a unique reviewable artifact folder.

- [ ] Phase 5: Add first evidence-grounded AI summary action
  - Status: Not started
  - Implement one high-quality AI-facing action for `run_full_eda` summary
    generation.
  - Output should separate observed facts, calculated metrics, inferred risks,
    recommendations, unknowns, limitations, and human review notes.
  - Acceptance gate: the summary uses evidence packet fields only and falls back
    clearly when evidence is missing or invalid.

- [ ] Phase 6: Add focused AI actions
  - Status: Not started
  - Add specialized actions for data quality explanation, response analysis,
    feature relationship interpretation, modeling risk detection, and EDA report
    drafting.
  - Acceptance gate: each action has a narrow input contract, output contract,
    fallback path, and evidence-use tests or examples.

- [ ] Phase 7: Improve Streamlit as a skill interface
  - Status: Not started
  - Refine the UI around upload/connect data, response selection, run EDA, review
    deterministic outputs, inspect AI interpretation, and export artifacts.
  - Keep prompts and calculation logic outside the UI layer.
  - Acceptance gate: Streamlit can run the full workflow while deterministic
    outputs remain inspectable before AI summaries.

- [ ] Phase 8: Add model-readiness scoring
  - Status: Not started
  - Add deterministic model-readiness dimensions such as response readiness,
    missingness risk, leakage risk, cardinality risk, drift risk, class
    imbalance, data type consistency, and documentation completeness.
  - Acceptance gate: readiness status is explainable, evidence-backed, and does
    not claim final model approval.

- [ ] Phase 9: Add run history and comparison
  - Status: Not started
  - Support comparing runs such as before/after cleaning, train/test, old/new
    extracts, or current/prior periods.
  - Acceptance gate: comparisons are deterministic and can produce an
    AI-safe comparison evidence packet.

- [ ] Phase 10: Add domain adapters
  - Status: Not started
  - Keep the general tabular EDA skill independent while adding domain-specific
    adapters later.
  - Candidate adapters: insurance underwriting EDA, actuarial experience study,
    model performance, fairness testing, and report drafting.
  - Acceptance gate: domain rules extend the shared foundation without mixing
    domain-specific assumptions into generic EDA behavior.

- [ ] Phase 11: Add Databricks connector later
  - Status: Not started
  - Add Databricks SQL/table access only after the core skill workflow is stable.
  - Acceptance gate: credentials are read securely, row limits are explicit, and
    connector outputs flow through the same deterministic workflow.

- [ ] Phase 12: Add AI evaluation framework
  - Status: Not started
  - Create synthetic evaluation cases for clean data, missingness, leakage, high
    cardinality, drift, imbalance, and insurance-like underwriting data.
  - Score AI outputs for factuality, evidence use, completeness, clarity,
    actionability, caveats, and hallucination resistance.
  - Acceptance gate: AI summaries do not invent metrics and consistently cite or
    reflect deterministic evidence.

## Immediate PR Sequence

- [x] PR 1: Product repositioning and baseline protection
  - Status: Completed
  - Add current-state, product-vision, and skill-direction docs.
  - Add baseline protection tests for evidence packet shape and artifact names.
  - No runtime behavior changes.

- [x] PR 2: `tabular-eda-skill` skeleton
  - Status: Completed
  - Add the skill package, action docs, schemas, examples, fallback behavior, and
    validation checklist.
  - No new EDA calculations.

- [x] PR 3: Dataset loading and schema inspection improvements
  - Status: Completed
  - Strengthen existing loading and type/schema inspection behavior.

- [ ] PR 4: Data quality profiling improvements
  - Status: Not started
  - Extend structured quality warnings and tests.

- [ ] PR 5: Response variable analysis improvements
  - Status: Not started
  - Improve binary/regression response summaries and warnings.

- [ ] PR 6: Feature relationship analysis improvements
  - Status: Not started
  - Rank and summarize feature-response relationships more deeply.

- [ ] PR 7: Modeling risk detection
  - Status: Not started
  - Expand leakage, drift, sparsity, cardinality, and suspicious predictor checks.

- [ ] PR 8: Evidence packet and artifact manifest hardening
  - Status: Not started
  - Add schema validation and stronger artifact manifest conventions.

- [ ] PR 9: First AI summary action
  - Status: Not started
  - Add evidence-grounded full EDA summary generation.

- [ ] PR 10: Streamlit skill workflow
  - Status: Not started
  - Expose the skill-first workflow through a thin Streamlit interface.

- [ ] PR 11: Report export package
  - Status: Not started
  - Add reusable report and artifact export bundle.

- [ ] PR 12: Run history and comparison
  - Status: Not started
  - Add deterministic run comparison and optional comparison summaries.

- [ ] PR 13: Insurance underwriting EDA adapter
  - Status: Not started
  - Add domain-specific underwriting risk and leakage checks using synthetic
    data tests.

## Acceptance Gates For Every Phase

- `uv run pytest` passes.
- Behavior changes are explicitly scoped and documented.
- Evidence packet and AI context remain sanitized.
- No raw rows are sent to AI interpretation.
- Existing workflow remains supported.
- New deterministic logic has focused tests.
- New AI-facing behavior has fallback behavior and evidence-use guardrails.
- Artifacts remain reviewable by humans.

## Work Tracking Notes

- Update the checkbox and `Status:` line when starting or completing a phase.
- Keep each PR scoped to one phase or one clearly related slice of a phase.
- Prefer adding docs and tests before risky refactors.
- Record follow-up ideas in this document instead of expanding a PR mid-flight.
