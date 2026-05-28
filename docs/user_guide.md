# User Guide

## Basic workflow

1. Start the app with `uv run streamlit run app.py`.
2. Upload a CSV or Parquet file, or load the sample dataset.
3. Select a response column when available.
4. Choose the problem type or keep `auto`.
5. Mark ID, date, split, segment, sensitive, and excluded columns as needed.
6. Run EDA.
7. Review profile alerts, response warnings, quality checks, modeling risks,
   feature ranking, and the Markdown report.
8. Export the evidence packet and artifact manifest for auditability or future
   evidence-only summarization.

## Interpreting results

- Treat data quality warnings as a review queue.
- Treat leakage warnings as timing and lineage checks.
- Confirm automatically inferred response type and positive class.
- Use feature ranking to prioritize investigation, not as a replacement for model validation.
- Treat failed quality checks as gates that require remediation or documented exceptions.
- Treat modeling risk signals as a review queue, not as final model approval.
- Use the Ask tab for evidence-only questions; it does not inspect raw row-level data.
- Use the AI Summary tab for a deterministic evidence-only summary fallback.
- Use Run History to compare saved evidence packets without re-reading raw data.
- Use Visual Explorer specs for reproducible charts. PyGWalker can be added as an optional visual integration when needed.
