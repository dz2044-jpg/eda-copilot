# Fallback Behavior

Use these fallback paths when the requested action cannot be completed from the
available deterministic evidence.

## Missing Or Invalid Data

- Report that the dataset is unavailable, empty, not tabular, or cannot be
  validated.
- Do not synthesize a dataframe or fabricate row counts, columns, warnings, or
  metrics.
- Ask for a valid dataset only if no local or caller-provided dataset exists.

## Missing Or Invalid Configuration

- Use `EDAConfig` defaults when the user has not specified optional settings.
- Stop and report the missing field when a required configured column is absent
  from the dataframe.
- Do not invent response, ID, split, sensitive, or excluded columns.

## Missing Evidence

- State which evidence packet section is missing.
- Limit the answer to available packet fields.
- Do not infer quantitative metrics that are not present in deterministic
  evidence.

## Optional AI Unavailable

- Return deterministic summaries and review queues from the evidence packet.
- Explain that AI summaries are optional and must be evidence-only.
- Do not inspect raw rows as a substitute for missing AI context.
