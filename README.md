# EDA Copilot

EDA Copilot is a deterministic, response-aware exploratory data analysis tool for repeatable data science workflows. It computes statistics, warnings, rankings, and reports in Python first; any AI summary layer is expected to summarize only the generated evidence packet.

## MVP scope

- CSV and Parquet loading through Streamlit
- Optional sample dataset
- Response variable selection
- Automatic column type inference
- Dataset overview
- Missingness analysis
- Univariate analysis
- Binary response analysis
- Basic bivariate correlations
- Data quality and leakage warnings
- Feature ranking
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
