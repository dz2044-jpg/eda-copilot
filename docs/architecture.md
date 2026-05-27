# EDA Copilot Architecture

## Flow

```text
Dataset
  -> configuration validation
  -> column type inference
  -> deterministic EDA modules
  -> evidence packet
  -> report builder
  -> optional artifact export
  -> optional AI summary over evidence only
```

## Layering

- `eda_copilot.eda`: deterministic statistics, tests, warnings, and rankings.
- `eda_copilot.reporting`: evidence packet, Markdown/HTML report, artifact export.
- `eda_copilot.app`: Streamlit UI only; no business statistics should live here.
- `eda_copilot.ai`: optional summary contracts and validators. This layer must not read raw datasets by default.

## MVP modules

- Overview
- Type inference
- Missingness
- Univariate
- Response analysis
- Bivariate correlations
- Data quality warnings
- Leakage warnings
- Feature ranking
- Drift starter module
