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
        _response_summary_line(response),
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
        "## Bivariate Analysis",
        "",
        *_bivariate_lines(bivariate),
        "",
        "## Recommended Next Steps",
        "",
        *_next_steps(quality, leakage, response),
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
        return lines
    if response.get("problem_type") == "regression":
        summary = response.get("target_summary", {})
        return [
            f"- Response column: `{response.get('response_column')}`",
            f"- Mean: {summary.get('mean')}",
            f"- Median: {summary.get('median')}",
            f"- Min / max: {summary.get('min')} / {summary.get('max')}",
        ]
    return [
        f"- Response column: `{response.get('response_column')}`",
        f"- Problem type: {response.get('problem_type')}",
        "- MVP does not yet compute multiclass feature-level tests.",
    ]


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
            f"| `{item['column']}` | {item['missing_percentage']:.2%} | {item['missing_count']:,} |"
        )
    return lines


def _feature_ranking_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- Feature ranking is unavailable."]
    lines = [
        "| Column | Type | Missing % | Signal metric | Signal value | Quality warnings | Leakage warnings |",
        "| --- | --- | ---: | --- | ---: | --- | --- |",
    ]
    for row in rows:
        signal_value = row.get("signal_value")
        signal_text = f"{signal_value:.4f}" if isinstance(signal_value, (int, float)) else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['column']}`",
                    _cell(row.get("semantic_type")),
                    f"{row.get('missing_percentage', 0.0):.2%}",
                    _cell(row.get("signal_metric")),
                    signal_text,
                    _cell(", ".join(row.get("data_quality_warnings", []))),
                    _cell(", ".join(row.get("leakage_warnings", []))),
                ]
            )
            + " |"
        )
    return lines


def _bivariate_lines(bivariate: dict[str, Any]) -> list[str]:
    pairs = bivariate.get("high_correlation_pairs", [])
    if not pairs:
        return ["- No high numeric correlations were detected at the configured threshold."]
    lines = ["| Left | Right | Correlation |", "| --- | --- | ---: |"]
    for pair in pairs[:20]:
        lines.append(f"| `{pair['left']}` | `{pair['right']}` | {pair['correlation']:.4f} |")
    return lines


def _next_steps(
    quality: list[dict[str, Any]],
    leakage: list[dict[str, Any]],
    response: dict[str, Any],
) -> list[str]:
    steps = []
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
