# Current State

EDA Copilot is a deterministic-first exploratory data analysis application for
tabular data. Its current product surface is a reusable Python workflow with a
thin Streamlit interface and optional AI-facing helpers that operate only on
sanitized evidence.

## Runtime Architecture

- `eda_copilot.core.workflow.run_eda` is the main deterministic orchestration
  entrypoint.
- `eda_copilot.core.config.EDAConfig` defines user-controlled run settings such
  as response column, problem type, ID columns, date columns, split column,
  profile depth, sample policy, and quality thresholds.
- `eda_copilot.core.data_loading` provides reusable CSV and Parquet loading with
  friendly validation errors and lightweight load metadata.
- `eda_copilot.eda` owns deterministic profiling, type inference, missingness,
  response analysis, bivariate analysis, feature ranking, leakage checks, drift,
  comparison, and quality checks.
- `eda_copilot.reporting` builds the evidence packet, Markdown report, HTML
  report, and optional run artifacts.
- `eda_copilot.visualization` builds reusable visual specifications.
- `eda_copilot.ai` is optional and must read sanitized evidence packet content,
  not the raw dataframe.
- `eda_copilot.app` provides Streamlit controls and display surfaces over the
  reusable workflow.

## Evidence Packet Contract

The current evidence packet is a JSON-serializable dictionary with these
top-level sections:

- `metadata`
- `config`
- `dataset_overview`
- `profile_summary`
- `response_summary`
- `column_type_summary`
- `missingness_summary`
- `univariate_summary`
- `bivariate_summary`
- `feature_ranking`
- `data_quality_warnings`
- `leakage_warnings`
- `drift_summary`
- `modeling_risk_summary`
- `quality_checks`
- `comparison_summary`
- `visual_specs`
- `plots`
- `artifacts`
- `external_artifacts`
- `caveats`

Dynamic values such as timestamps are implementation details and should not be
used as stable test fixtures.

`column_type_summary` includes additive schema inspection metadata such as
normalized column names, name warnings, Python type counts, parse-rate hints, and
parse candidate column lists. These fields are diagnostic only; PR 3 does not
coerce dataframe values or change source dtypes.

## Exported Artifacts

When `run_eda(..., export_artifacts=True)` is used, the workflow creates a
timestamped run folder containing:

- `input_manifest.json`
- `config_snapshot.json`
- `evidence_packet.json`
- `comparison_summary.json`
- `visual_specs.json`
- `modeling_risk_summary.json`
- `run_metadata.json`
- `artifact_manifest.json`
- `data_quality_warnings.csv`
- `profile_alerts.csv`
- `quality_checks.csv`
- `feature_ranking.csv`
- `eda_report.md`
- `eda_report.html`
- `audit_log.jsonl`
- `plots/`

Artifact names are part of the current baseline. The run folder name is dynamic
and should not be treated as a stable contract.

## AI Privacy Boundary

AI-facing features must consume evidence-only context. Raw rows, row samples,
configured ID values, sensitive row-level values, and data-dictionary sample
values must not be sent to optional AI interpretation layers. The helper
`build_llm_evidence_context` recursively removes row samples, sample values,
raw row fields, and deep text terms from the evidence context.

## Known Limitations

- The current workflow supports local CSV and Parquet loading through a reusable
  local file loader and the Streamlit upload UI.
- The baseline does not include runtime skill execution, Databricks connectors,
  formal model-readiness approval scoring, or domain adapters.
- Exported run folders now include metadata and manifests that support
  evidence-only run history and comparison.
- Multiclass response summaries exist, but feature-level multiclass tests are
  still limited.
- AI summary generation is optional and not required for deterministic reports.
