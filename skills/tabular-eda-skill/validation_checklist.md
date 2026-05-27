# Validation Checklist

Use this checklist before completing a tabular EDA skill task.

- The request maps to existing `EDAConfig` fields and `run_eda` behavior.
- Deterministic outputs are generated or inspected before any interpretation.
- Evidence packet sections used in the answer are named explicitly.
- Exported artifact paths or names are reported when artifacts are enabled.
- AI-facing content is evidence-only.
- Raw rows, row samples, IDs, sensitive row-level values, and private data are
  not exposed.
- Missing evidence and limitations are stated rather than filled in.
- Feature ranking is framed as investigation priority, not causality.
- Quality checks and leakage warnings are framed as review gates, not final
  modeling decisions.
