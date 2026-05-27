# EDA Copilot Architecture

## Flow

```text
Dataset
  -> configuration validation
  -> column type inference
  -> deterministic EDA modules
  -> evidence packet
  -> native profile, quality checks, comparison, visual specs
  -> report builder
  -> optional artifact export
  -> optional AI summary over evidence only
```

## Layering

- `eda_copilot.eda`: deterministic statistics, tests, warnings, and rankings.
- `eda_copilot.reporting`: evidence packet, Markdown/HTML report, artifact export.
- `eda_copilot.app`: Streamlit UI only; no business statistics should live here.
- `eda_copilot.ai`: optional summary contracts and validators. This layer must not read raw datasets by default.
- `eda_copilot.visualization`: Plotly chart builders and saved visual spec metadata.

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
- Native profile summary
- Quality checks
- Reference/current comparison
- Guarded evidence question planner

## Evidence Extensions

The evidence packet now includes `profile_summary`, `quality_checks`, `comparison_summary`, `visual_specs`, and `external_artifacts`.
These are additive and keep the original deterministic sections intact.

AI-facing context is sanitized by `eda_copilot.ai.summarizer.build_llm_evidence_context`: row samples and data-dictionary sample values are removed before any optional LLM layer sees the packet.
