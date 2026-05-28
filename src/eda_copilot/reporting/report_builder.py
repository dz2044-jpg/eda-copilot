from __future__ import annotations

from html import escape
from typing import Any


def build_markdown_report(evidence_packet: dict[str, Any]) -> str:
    """Build a standardized Markdown EDA report from deterministic evidence."""

    overview = evidence_packet["dataset_overview"]
    response = evidence_packet["response_summary"]
    quality = evidence_packet.get("data_quality_warnings", [])
    leakage = evidence_packet.get("leakage_warnings", [])
    ranking = evidence_packet.get("feature_ranking", [])
    missing = evidence_packet.get("missingness_summary", {})
    bivariate = evidence_packet.get("bivariate_summary", {})
    profile = evidence_packet.get("profile_summary", {})
    quality_checks = evidence_packet.get("quality_checks", {})
    comparison = evidence_packet.get("comparison_summary", {})
    modeling_risk = evidence_packet.get("modeling_risk_summary", {})
    caveats = evidence_packet.get("caveats", [])

    lines = [
        f"# EDA Report: {overview['dataset_name']}",
        "",
        "## Executive Summary",
        "",
        f"- Dataset contains {overview['row_count']:,} rows and {overview['column_count']:,} columns.",
        f"- Duplicate rows detected: {overview['duplicate_row_count']:,}.",
        f"- Data quality warnings: {len(quality):,}.",
        f"- Leakage warnings: {len(leakage):,}.",
        f"- Quality gate status: {quality_checks.get('overall_status', 'unavailable')}.",
        f"- Modeling risk status: {modeling_risk.get('overall_status', 'unavailable')}.",
        _response_summary_line(response),
        "",
        "## Run Metadata",
        "",
        *_run_metadata_lines(evidence_packet),
        "",
        "## Dataset Overview",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Rows | {overview['row_count']:,} |",
        f"| Columns | {overview['column_count']:,} |",
        f"| Memory usage bytes | {overview['memory_usage_bytes']:,} |",
        f"| Duplicate rows | {overview['duplicate_row_count']:,} |",
        "",
        "## Native Profile Summary",
        "",
        *_profile_lines(profile),
        "",
        "## Quality Checks",
        "",
        *_quality_check_lines(quality_checks),
        "",
        "## Response Variable Summary",
        "",
        *_response_section(response),
        "",
        "## Data Quality Findings",
        "",
        *_warning_table(quality, ["severity", "column", "issue_type", "evidence", "recommended_action"]),
        "",
        "## Missingness Analysis",
        "",
        *_top_missingness_lines(missing),
        "",
        "## Feature Ranking",
        "",
        *_feature_ranking_table(ranking[:20]),
        "",
        "## Leakage Warnings",
        "",
        *_warning_table(leakage, ["severity", "column", "reason", "evidence", "recommended_next_step"]),
        "",
        "## Modeling Risk Summary",
        "",
        *_modeling_risk_lines(modeling_risk),
        "",
        "## Bivariate Analysis",
        "",
        *_bivariate_lines(bivariate),
        "",
        "## Reference/Current Comparison",
        "",
        *_comparison_lines(comparison),
        "",
        "## Recommended Next Steps",
        "",
        *_next_steps(quality, leakage, response, quality_checks),
        "",
        "## Artifact Manifest",
        "",
        *_artifact_manifest_lines(evidence_packet),
        "",
        "## Appendix: Caveats",
        "",
        *([f"- {caveat}" for caveat in caveats] if caveats else ["- No additional caveats were generated."]),
    ]
    return "\n".join(lines).strip() + "\n"


def build_basic_html_report(markdown_report: str) -> str:
    """Return a dependency-free HTML wrapper for the Markdown report."""

    escaped = escape(markdown_report)
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>EDA Report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;line-height:1.5;}"
        "pre{white-space:pre-wrap;background:#f7f7f7;padding:24px;border-radius:8px;}</style>"
        "</head><body><pre>"
        f"{escaped}"
        "</pre></body></html>"
    )


def _response_summary_line(response: dict[str, Any]) -> str:
    if not response.get("available"):
        return "- Response-aware analysis was not run because no response variable was available."
    if response.get("problem_type") == "binary_classification":
        rate = response.get("response_rate")
        return f"- Binary response rate for positive class `{response.get('positive_class')}`: {rate:.2%}." if rate is not None else "- Binary response was detected."
    return f"- Response analysis problem type: {response.get('problem_type')}."


def _response_section(response: dict[str, Any]) -> list[str]:
    if not response.get("available"):
        return [f"- {response.get('reason', 'Response analysis unavailable.')}"]
    if response.get("problem_type") == "binary_classification":
        lines = [
            f"- Response column: `{response.get('response_column')}`",
            f"- Positive class: `{response.get('positive_class')}`",
            f"- Negative class: `{response.get('negative_class')}`",
        ]
        if response.get("response_rate") is not None:
            lines.append(f"- Response rate: {response['response_rate']:.2%}")
        if response.get("imbalance_ratio") is not None:
            lines.append(f"- Imbalance ratio: {response['imbalance_ratio']:.2f}")
        return lines + _response_warning_lines(response)
    if response.get("problem_type") == "regression":
        summary = response.get("target_summary", {})
        lines = [
            f"- Response column: `{response.get('response_column')}`",
            f"- Mean: {summary.get('mean')}",
            f"- Median: {summary.get('median')}",
            f"- Min / max: {summary.get('min')} / {summary.get('max')}",
        ]
        return lines + _response_warning_lines(response)
    return [
        f"- Response column: `{response.get('response_column')}`",
        f"- Problem type: {response.get('problem_type')}",
        "- MVP does not yet compute multiclass feature-level tests.",
    ] + _response_warning_lines(response)


def _warning_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["- No warnings generated."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows[:25]:
        lines.append("| " + " | ".join(_cell(row.get(column)) for column in columns) + " |")
    return lines


def _profile_lines(profile: dict[str, Any]) -> list[str]:
    if not profile:
        return ["- Native profile summary is unavailable."]
    alert_summary = profile.get("alert_summary", {})
    sensitive_report = profile.get("sensitive_report", {})
    return [
        f"- Profile depth: `{profile.get('profile_depth')}`.",
        f"- Alert count: {alert_summary.get('total', 0)}.",
        f"- Alert categories: {_cell(alert_summary.get('by_category', {}))}.",
        f"- Redacted columns: {_cell(sensitive_report.get('redacted_columns', []))}.",
        f"- Text summaries: {len(profile.get('text_summary', []))}.",
        f"- Datetime summaries: {len(profile.get('datetime_summary', []))}.",
    ]


def _run_metadata_lines(evidence_packet: dict[str, Any]) -> list[str]:
    metadata = evidence_packet.get("metadata", {})
    config = evidence_packet.get("config", {})
    return [
        f"- Tool: `{metadata.get('tool_name', 'eda_copilot')}` {metadata.get('tool_version', '')}.",
        f"- Created at UTC: `{metadata.get('created_at_utc', 'unknown')}`.",
        f"- Dataset name: `{config.get('dataset_name', 'dataset')}`.",
    ]


def _quality_check_lines(quality_checks: dict[str, Any]) -> list[str]:
    checks = quality_checks.get("checks", [])
    if not checks:
        return ["- No quality checks were generated."]
    lines = [
        "| Status | Severity | Check | Metric | Value | Threshold |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    priority = {"fail": 0, "warn": 1, "pass": 2}
    sorted_checks = sorted(checks, key=lambda row: priority.get(str(row.get("status")), 3))[:25]
    for check in sorted_checks:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(check.get("status")),
                    _cell(check.get("severity")),
                    _cell(check.get("name")),
                    _cell(check.get("metric_name")),
                    _cell(check.get("metric_value")),
                    _cell(check.get("threshold")),
                ]
            )
            + " |"
        )
    return lines


def _top_missingness_lines(missing: dict[str, Any]) -> list[str]:
    columns = sorted(
        missing.get("columns", []),
        key=lambda item: item.get("missing_percentage", 0.0),
        reverse=True,
    )[:20]
    if not columns:
        return ["- No missingness results available."]
    lines = ["| Column | Missing % | Missing count |", "| --- | ---: | ---: |"]
    for item in columns:
        lines.append(
            f"| {_code_cell(item['column'])} | {item['missing_percentage']:.2%} | {item['missing_count']:,} |"
        )
    return lines


def _feature_ranking_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- Feature ranking is unavailable."]
    lines = [
        "| Rank | Column | Type | Missing % | Signal metric | Signal value | Direction | Quality penalty | Review reason |",
        "| ---: | --- | --- | ---: | --- | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        signal_value = row.get("signal_value")
        signal_text = f"{signal_value:.4f}" if isinstance(signal_value, (int, float)) else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("rank", "")),
                    _code_cell(row["column"]),
                    _cell(row.get("semantic_type")),
                    f"{row.get('missing_percentage', 0.0):.2%}",
                    _cell(row.get("signal_metric")),
                    signal_text,
                    _cell(row.get("signal_direction")),
                    f"{float(row.get('quality_penalty') or 0.0):.2f}",
                    _cell(row.get("recommended_review_reason")),
                ]
            )
            + " |"
        )
    return lines


def _response_warning_lines(response: dict[str, Any]) -> list[str]:
    warnings = response.get("warnings", [])
    if not warnings:
        return ["- Response warnings: none."]
    lines = ["", "| Severity | Issue | Evidence |", "| --- | --- | --- |"]
    for warning in warnings[:10]:
        lines.append(
            f"| {_cell(warning.get('severity'))} | {_cell(warning.get('issue_type'))} | {_cell(warning.get('evidence'))} |"
        )
    return lines


def _modeling_risk_lines(modeling_risk: dict[str, Any]) -> list[str]:
    if not modeling_risk:
        return ["- Modeling risk summary is unavailable."]
    summary = modeling_risk.get("summary", {})
    lines = [
        f"- Overall status: `{modeling_risk.get('overall_status', 'unknown')}`.",
        f"- Risk count: {summary.get('total', 0)}.",
        f"- Risk categories: {_cell(summary.get('by_category', {}))}.",
    ]
    risks = modeling_risk.get("risks", [])
    if not risks:
        return lines + ["- No deterministic modeling risk signals were generated."]
    lines += [
        "| Severity | Category | Column | Evidence | Recommended action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for risk in risks[:15]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(risk.get("severity")),
                    _cell(risk.get("category")),
                    _cell(risk.get("column")),
                    _cell(risk.get("evidence")),
                    _cell(risk.get("recommended_action")),
                ]
            )
            + " |"
        )
    return lines


def _artifact_manifest_lines(evidence_packet: dict[str, Any]) -> list[str]:
    artifacts = evidence_packet.get("artifacts", [])
    external = evidence_packet.get("external_artifacts", [])
    if not artifacts and not external:
        return ["- `artifact_manifest.json` is generated when artifacts are saved to a run folder."]
    return [
        f"- In-packet artifacts: {len(artifacts)}.",
        f"- External artifacts: {len(external)}.",
    ]


def _bivariate_lines(bivariate: dict[str, Any]) -> list[str]:
    pairs = bivariate.get("high_correlation_pairs", [])
    if not pairs:
        return ["- No high numeric correlations were detected at the configured threshold."]
    lines = ["| Left | Right | Correlation |", "| --- | --- | ---: |"]
    for pair in pairs[:20]:
        lines.append(f"| {_code_cell(pair['left'])} | {_code_cell(pair['right'])} | {pair['correlation']:.4f} |")
    return lines


def _comparison_lines(comparison: dict[str, Any]) -> list[str]:
    if not comparison.get("available"):
        return [f"- {comparison.get('reason', 'Reference/current comparison is unavailable.')}"]
    changes = comparison.get("top_column_changes", [])
    lines = [
        f"- Comparison column: `{comparison.get('comparison_column')}`.",
        f"- Reference/current groups: `{comparison.get('reference_group')}` / `{comparison.get('current_group')}`.",
    ]
    ignored_groups = comparison.get("ignored_groups") or []
    lines.append(f"- Ignored groups: {_cell(ignored_groups) if ignored_groups else 'none'}.")
    if not changes:
        return lines + ["- No column-level comparison rows were generated."]
    lines += [
        "| Column | Metric | Missing delta | Change score |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in changes[:15]:
        lines.append(
            f"| {_code_cell(row.get('column'))} | {_cell(row.get('comparison_metric'))} | "
            f"{float(row.get('missing_percentage_delta', 0.0)):.2%} | "
            f"{float(row.get('change_score') or 0.0):.4f} |"
        )
    return lines


def _next_steps(
    quality: list[dict[str, Any]],
    leakage: list[dict[str, Any]],
    response: dict[str, Any],
    quality_checks: dict[str, Any],
) -> list[str]:
    steps = []
    if quality_checks.get("overall_status") == "fail":
        steps.append("- Resolve failed quality checks or document accepted exceptions before downstream modeling.")
    if leakage:
        steps.append("- Review high-severity leakage candidates before modeling.")
    if quality:
        steps.append("- Resolve high and medium data quality warnings or document why they are acceptable.")
    if response.get("problem_type") == "binary_classification":
        steps.append("- Validate the positive class definition with the project owner.")
    steps.append("- Review the top-ranked features with business timing and data availability constraints.")
    return steps


def _cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def _code_cell(value: Any) -> str:
    text = _cell(value)
    if not text:
        return ""
    delimiter = "`"
    while delimiter in text:
        delimiter += "`"
    return f"{delimiter}{text}{delimiter}"
