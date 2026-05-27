# EDA Copilot

EDA Copilot is a deterministic, response-aware exploratory data analysis tool for repeatable data science workflows. It computes statistics, warnings, rankings, and reports in Python first; any AI summary layer is expected to summarize only the generated evidence packet.

## MVP scope

- CSV and Parquet loading through a reusable loader and Streamlit
- Optional sample dataset
- Response variable selection
- Automatic column type inference
- Additive schema inspection metadata and parse-candidate hints
- Dataset overview
- Missingness analysis
- Univariate analysis
- Binary response analysis
- Basic bivariate correlations
- Data quality and leakage warnings
- Feature ranking
- Native profile summary with alert taxonomy and redacted sample policy
- Quality checks with pass/warn/fail status
- Reference/current comparison and drift gates
- Saved visual specs and optional visual explorer integration
- Guarded evidence-only question planning
- Markdown report and JSON evidence export

## Run locally

```bash
uv sync --extra dev
uv run streamlit run app.py
```

Run tests:

```bash
uv run pytest
```

## Design guardrails

- Deterministic modules calculate all metrics and warnings.
- LLM features must read only `evidence_packet.json`, not the raw dataset.
- UI code stays thin and delegates calculations to `eda_copilot.core.workflow`.
- Reports separate facts, metrics, inferred interpretation, caveats, and next steps.
- Row samples are removed from AI context, and configured ID/sensitive samples are redacted by default.

Optional integrations can be enabled later with:

```bash
uv add --optional visual pygwalker
uv add --optional profiling fg-data-profiling
uv add --optional monitoring evidently
```
