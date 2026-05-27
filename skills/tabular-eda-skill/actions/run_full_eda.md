# Action: Run Full EDA

## Purpose

Run the existing deterministic EDA Copilot workflow for a validated tabular
dataset and return reviewable evidence, reports, and optional artifacts.

## Inputs

- A pandas-compatible dataframe supplied by the host application or caller.
- An `EDAConfig`-compatible configuration:
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
  - supported threshold fields
- Optional export settings:
  - `export_artifacts`
  - `output_base_dir`

See `../schemas/run_full_eda_input.schema.json` for the documentation-level
contract. The schema is descriptive and does not replace runtime validation.

## Deterministic Procedure

1. Validate the dataframe and configuration with the existing EDA Copilot
   validation path.
2. Call `eda_copilot.core.workflow.run_eda`.
3. Preserve the returned `evidence_packet`, `markdown_report`, and optional
   `artifact_dir`.
4. Do not add new EDA calculations in this action skeleton.

## Outputs

- `evidence_packet`: JSON-serializable deterministic evidence.
- `markdown_report`: deterministic Markdown report.
- `artifact_dir`: optional path to the exported run folder.
- Exported artifact names when artifacts are enabled.

See `../schemas/run_full_eda_output.schema.json` for the documentation-level
output contract.

## Guardrails

- Do not bypass `EDAConfig`.
- Do not mutate the input dataframe in place.
- Do not send raw rows to AI interpretation.
- Treat warnings and feature rankings as review signals, not model approval or
  causal claims.
