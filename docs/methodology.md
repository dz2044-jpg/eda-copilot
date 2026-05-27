# Methodology

EDA Copilot follows deterministic-first analysis.

## Column typing

Columns are assigned semantic types using pandas dtype, observed cardinality, missingness, top-value dominance, parseable dates, text length, ID-like uniqueness, and suspicious name patterns.

## Binary response analysis

For binary targets, the workflow encodes a deterministic positive class. Common positive labels such as `1`, `true`, `yes`, `event`, `bad`, `claim`, and `lapse` take precedence. Otherwise, the positive class is the last class after string sorting. The report flags this as a caveat so the user can confirm it.

## Feature ranking

Feature ranking uses univariate deterministic evidence:

- Numeric binary-classification features use absolute AUC.
- Categorical binary-classification features use Cramer's V or response-rate spread.
- Regression features use absolute correlation or target-mean spread.

The ranking is not a model and should not be interpreted as causal.

## Leakage

Leakage checks are warnings, not proof. They combine suspicious names and near-perfect univariate separation. Timing and lineage must be reviewed with project context.
