# User Guide

## Basic workflow

1. Start the app with `uv run streamlit run app.py`.
2. Upload a CSV or Parquet file, or load the sample dataset.
3. Select a response column when available.
4. Choose the problem type or keep `auto`.
5. Mark ID, date, split, segment, sensitive, and excluded columns as needed.
6. Run EDA.
7. Review profile alerts, quality checks, warnings, feature ranking, and the Markdown report.
8. Export the evidence packet for auditability or future AI summarization.

## Interpreting results

- Treat data quality warnings as a review queue.
- Treat leakage warnings as timing and lineage checks.
- Confirm automatically inferred response type and positive class.
- Use feature ranking to prioritize investigation, not as a replacement for model validation.
- Treat failed quality checks as gates that require remediation or documented exceptions.
- Use the Ask tab for evidence-only questions; it does not inspect raw row-level data.
- Use Visual Explorer specs for reproducible charts. PyGWalker can be added as an optional visual integration when needed.
