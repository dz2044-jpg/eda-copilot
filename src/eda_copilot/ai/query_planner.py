from __future__ import annotations

from typing import Any

from eda_copilot.ai.summarizer import build_llm_evidence_context


RAW_DATA_KEYWORDS = {
    "raw row",
    "raw rows",
    "show rows",
    "show records",
    "list records",
    "record level",
    "row level",
    "sample row",
    "sample rows",
}


def plan_evidence_question(question: str, evidence_packet: dict[str, Any]) -> dict[str, Any]:
    """Plan a guarded answer using approved deterministic evidence only.

    This function borrows the conversational shape of dataframe chat tools
    without executing generated code or exposing raw records to an LLM.
    """

    clean_question = question.strip()
    context = build_llm_evidence_context(evidence_packet)
    lowered = clean_question.lower()
    if not clean_question:
        return _response("empty_question", [], "Ask a question about the generated EDA evidence.")
    if any(keyword in lowered for keyword in RAW_DATA_KEYWORDS):
        return _response(
            status="blocked_raw_data",
            sources=["metadata", "dataset_overview"],
            answer="This assistant does not answer from raw row-level data. Use the deterministic tables and exports for row inspection.",
            suggested_actions=[
                "Review the redacted sample in the Overview tab.",
                "Use exported evidence rather than raw data for AI summaries.",
            ],
        )
    if "leak" in lowered:
        return _response(
            status="planned",
            sources=["leakage_warnings", "feature_ranking"],
            answer=_top_records_answer("leakage warning", context.get("leakage_warnings", []), "column"),
        )
    if "missing" in lowered or "null" in lowered:
        missing = context.get("missingness_summary", {}).get("columns", [])
        top = sorted(missing, key=lambda row: row.get("missing_percentage", 0.0), reverse=True)[:5]
        return _response(
            status="planned",
            sources=["missingness_summary"],
            answer=_format_missingness_answer(top),
        )
    if "drift" in lowered or "shift" in lowered:
        drift = context.get("drift_summary", {})
        return _response(
            status="planned",
            sources=["drift_summary", "comparison_summary"],
            answer=_format_drift_answer(drift),
        )
    if "quality" in lowered or "check" in lowered or "gate" in lowered:
        quality_checks = context.get("quality_checks", {})
        return _response(
            status="planned",
            sources=["quality_checks", "data_quality_warnings"],
            answer=_format_quality_answer(quality_checks),
        )
    if "modeling risk" in lowered or "model readiness" in lowered or "readiness" in lowered:
        modeling = context.get("modeling_risk_summary", {})
        return _response(
            status="planned",
            sources=["modeling_risk_summary"],
            answer=_format_modeling_risk_answer(modeling),
        )
    if "feature" in lowered or "rank" in lowered or "important" in lowered:
        return _response(
            status="planned",
            sources=["feature_ranking"],
            answer=_top_records_answer("feature", context.get("feature_ranking", []), "column"),
        )
    return _response(
        status="planned",
        sources=[
            "dataset_overview",
            "profile_summary",
            "quality_checks",
            "feature_ranking",
            "caveats",
        ],
        answer=_format_summary_answer(context),
    )


def _response(
    status: str,
    sources: list[str],
    answer: str,
    suggested_actions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reads_raw_dataset": False,
        "allowed_context": "evidence_packet_only",
        "evidence_sources": sources,
        "answer": answer,
        "suggested_actions": suggested_actions or [],
    }


def _top_records_answer(label: str, rows: list[dict[str, Any]], column_key: str) -> str:
    if not rows:
        return f"No {label}s were available in the deterministic evidence."
    labels = [str(row.get(column_key, "<unknown>")) for row in rows[:5]]
    return f"Top {label}s from deterministic evidence: {', '.join(labels)}."


def _format_missingness_answer(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No missingness rows were available in the deterministic evidence."
    parts = [
        f"{row.get('column')} ({float(row.get('missing_percentage', 0.0)):.1%})"
        for row in rows
    ]
    return f"Highest missingness columns: {', '.join(parts)}."


def _format_drift_answer(drift: dict[str, Any]) -> str:
    if not drift.get("available"):
        return str(drift.get("reason", "Drift evidence is not available."))
    shifted = drift.get("top_shifted_features", [])[:5]
    if not shifted:
        return "Drift was configured, but no shifted features were reported."
    parts = [
        f"{row.get('column')}={float(row.get('value', 0.0)):.3f}"
        for row in shifted
    ]
    return f"Top drift signals across {drift.get('baseline_group')} vs {drift.get('comparison_group')}: {', '.join(parts)}."


def _format_quality_answer(quality_checks: dict[str, Any]) -> str:
    if not quality_checks:
        return "Quality check evidence is not available."
    summary = quality_checks.get("summary", {})
    return (
        f"Quality gate status is {quality_checks.get('overall_status')}: "
        f"{summary.get('pass', 0)} pass, {summary.get('warn', 0)} warn, {summary.get('fail', 0)} fail."
    )


def _format_modeling_risk_answer(modeling: dict[str, Any]) -> str:
    if not modeling:
        return "Modeling risk evidence is not available."
    summary = modeling.get("summary", {})
    risks = modeling.get("risks", [])[:5]
    risk_columns = [str(row.get("column")) for row in risks if row.get("column")]
    detail = f" Top risk columns: {', '.join(risk_columns)}." if risk_columns else ""
    return (
        f"Modeling risk status is {modeling.get('overall_status')}; "
        f"{summary.get('total', 0)} deterministic risk signals were generated."
        f"{detail}"
    )


def _format_summary_answer(context: dict[str, Any]) -> str:
    overview = context.get("dataset_overview", {})
    quality = context.get("quality_checks", {})
    return (
        f"Dataset has {overview.get('row_count', 0)} rows and {overview.get('column_count', 0)} columns. "
        f"Quality gate status: {quality.get('overall_status', 'unavailable')}. "
        "Use the listed evidence sources for a deterministic answer."
    )
