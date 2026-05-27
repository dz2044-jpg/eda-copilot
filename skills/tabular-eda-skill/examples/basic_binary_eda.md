# Example: Basic Binary EDA

## User Request

Run EDA on `applications.csv`, use `target` as the response, treat
`application_id` as an ID column, and export artifacts.

## Expected Handling

- Load the tabular data through the host application or existing repo workflow.
- Build `EDAConfig` with:
  - `dataset_name="applications"`
  - `response_column="target"`
  - `id_columns=("application_id",)`
  - default `problem_type="auto"`
  - default `sample_policy="redacted"`
- Call `run_eda(df, config, export_artifacts=True)`.
- Return the evidence packet, Markdown report, and artifact directory.
- Explain warnings and quality checks from evidence-only context.

## Expected Caveats

- Confirm the deterministic positive class matches the project definition.
- Treat feature ranking as investigation priority, not causality.
- Do not expose raw rows, row samples, IDs, or sensitive row-level values in AI
  summaries.
