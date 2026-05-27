# User Guide

## Basic workflow

1. Start the app with `uv run streamlit run app.py`.
2. Upload a CSV or Parquet file, or load the sample dataset.
3. Select a response column when available.
4. Choose the problem type or keep `auto`.
5. Mark ID, date, split, segment, and excluded columns as needed.
6. Run EDA.
7. Review warnings, feature ranking, and the Markdown report.
8. Export the evidence packet for auditability or future AI summarization.

## Interpreting results

- Treat data quality warnings as a review queue.
- Treat leakage warnings as timing and lineage checks.
- Confirm automatically inferred response type and positive class.
- Use feature ranking to prioritize investigation, not as a replacement for model validation.
