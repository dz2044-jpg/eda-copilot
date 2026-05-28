# Current Contracts

## EDAConfig Fields

The current run configuration is `eda_copilot.core.config.EDAConfig`. Use the
existing dataclass fields rather than inventing new skill-specific options:

- `dataset_name`
- `response_column`
- `problem_type`
- `id_columns`
- `date_columns`
- `train_test_column`
- `segment_columns`
- `exclude_columns`
- `sensitive_columns`
- `weight_column`
- `profile_depth`
- `sample_policy`
- `max_categories`
- `rare_category_threshold`
- `high_cardinality_threshold`
- `high_missingness_threshold`
- `near_constant_threshold`
- `high_correlation_threshold`
- `high_auc_leakage_threshold`
- `max_correlation_columns`
- `drift_warning_threshold`
- `drift_fail_threshold`

## Workflow Entrypoint

Use `eda_copilot.core.workflow.run_eda(df, config, export_artifacts=False,
output_base_dir=...)` for deterministic orchestration. The workflow returns an
`EDAResult` with `evidence_packet`, `markdown_report`, and optional
`artifact_dir`.

## Evidence Packet Sections

The current packet includes:

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

## Exported Artifact Names

When export is enabled, expect these names in the run folder:

- `input_manifest.json`
- `config_snapshot.json`
- `run_metadata.json`
- `evidence_packet.json`
- `comparison_summary.json`
- `visual_specs.json`
- `modeling_risk_summary.json`
- `artifact_manifest.json`
- `data_quality_warnings.csv`
- `profile_alerts.csv`
- `quality_checks.csv`
- `feature_ranking.csv`
- `eda_report.md`
- `eda_report.html`
- `audit_log.jsonl`
- `plots/`

## AI Context Boundary

For AI-facing review, use sanitized evidence only. Raw rows, row samples,
configured ID values, sensitive row-level values, and data-dictionary sample
values must stay out of AI context. Sanitization is recursive and also removes
raw row fields and deep text `top_terms`.
