# Methodology

EDA Copilot follows deterministic-first analysis.

## Column typing

Columns are assigned semantic types using pandas dtype, observed cardinality, missingness, top-value dominance, parseable dates, text length, ID-like uniqueness, and suspicious name patterns.

## Schema inspection

The type inference step also reports schema diagnostics without modifying the
source dataframe. These diagnostics include normalized column names, name
warnings, Python value type counts, text-column parse rates, normalized-name
collisions, and candidate columns that appear parseable as numeric, datetime, or
boolean values. Parse candidates are review signals only; downstream EDA keeps
the loaded dataframe dtypes unchanged unless a later preprocessing step converts
them explicitly.

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

## Native profiling

The native profile summary consolidates deterministic warnings into an alert taxonomy covering missingness, duplicates, distribution issues, schema concerns, correlations, and leakage. `profile_depth` controls detail:

- `minimal`: skips correlation matrix generation and text/date deep summaries.
- `standard`: includes standard alerts, text summaries, datetime summaries, correlations, drift, and visual specs.
- `deep`: adds deeper text token summaries and larger diagnostic detail where safe.

## Sensitive samples

`sample_policy="redacted"` is the default. Configured ID columns, sensitive columns, and ID-like candidate columns have sample values redacted in the evidence packet. AI context removes row samples and data-dictionary sample values entirely.

## Quality checks

Quality checks convert deterministic evidence into pass/warn/fail rows. High-severity data quality and leakage warnings fail. Drift checks use configurable warning/fail thresholds over absolute drift metrics.
