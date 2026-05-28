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
- Deterministic modeling risk summary
- Reference/current comparison and drift gates
- Run metadata, artifact manifests, and evidence-only run comparison
- Saved visual specs and optional visual explorer integration
- Guarded evidence-only question planning
- Evidence-only AI summary fallback
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

Optional integrations are auto-detected by the UI when installed locally. They
are not part of the base install; add them intentionally when you are ready to
manage the extra dependency surface:

```bash
uv add pygwalker
```
