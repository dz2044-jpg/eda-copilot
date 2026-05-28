from __future__ import annotations

from typing import Any


ALLOWED_LLM_EVIDENCE_KEYS = [
    "metadata",
    "dataset_overview",
    "profile_summary",
    "response_summary",
    "column_type_summary",
    "missingness_summary",
    "feature_ranking",
    "data_quality_warnings",
    "leakage_warnings",
    "drift_summary",
    "modeling_risk_summary",
    "quality_checks",
    "comparison_summary",
    "visual_specs",
    "caveats",
]


def build_llm_evidence_context(evidence_packet: dict[str, Any]) -> dict[str, Any]:
    """Return the only context an optional LLM summary is allowed to read."""

    context = {key: evidence_packet.get(key) for key in ALLOWED_LLM_EVIDENCE_KEYS}
    if isinstance(context.get("dataset_overview"), dict):
        context["dataset_overview"] = _sanitized_overview(context["dataset_overview"])
    return context


def build_evidence_summary(evidence_packet: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic, evidence-only summary using the AI summary contract."""

    context = build_llm_evidence_context(evidence_packet)
    overview = context.get("dataset_overview") or {}
    response = context.get("response_summary") or {}
    quality_checks = context.get("quality_checks") or {}
    modeling = context.get("modeling_risk_summary") or {}
    quality = context.get("data_quality_warnings") or []
    leakage = context.get("leakage_warnings") or []
    ranking = context.get("feature_ranking") or []
    drift = context.get("drift_summary") or {}
    referenced_sections = {
        "dataset_overview",
        "response_summary",
        "quality_checks",
        "data_quality_warnings",
        "leakage_warnings",
        "feature_ranking",
        "modeling_risk_summary",
        "caveats",
    }
    if drift.get("available"):
        referenced_sections.add("drift_summary")

    observed_facts = [
        _summary_item(
            "dataset_shape",
            f"Dataset contains {overview.get('row_count', 0)} rows and {overview.get('column_count', 0)} columns.",
            "dataset_overview",
        ),
        _summary_item(
            "quality_gate_status",
            f"Quality gate status is {quality_checks.get('overall_status', 'unavailable')}.",
            "quality_checks",
        ),
    ]
    if response.get("available"):
        observed_facts.append(
            _summary_item(
                "response_problem_type",
                f"Response analysis problem type is {response.get('problem_type')}.",
                "response_summary",
                column=response.get("response_column"),
            )
        )

    calculated_metrics = [
        _summary_item(
            "data_quality_warning_count",
            f"{len(quality)} data quality warnings were generated.",
            "data_quality_warnings",
        ),
        _summary_item(
            "leakage_warning_count",
            f"{len(leakage)} leakage warnings were generated.",
            "leakage_warnings",
        ),
        _summary_item(
            "modeling_risk_count",
            f"{modeling.get('summary', {}).get('total', 0)} modeling risk signals were generated.",
            "modeling_risk_summary",
        ),
    ]
    if response.get("problem_type") == "binary_classification" and response.get("response_rate") is not None:
        calculated_metrics.append(
            _summary_item(
                "binary_response_rate",
                f"Positive-class response rate is {float(response['response_rate']):.2%}.",
                "response_summary",
                column=response.get("response_column"),
            )
        )

    inferred_risks = [
        _summary_item(
            f"risk_{index}",
            str(risk.get("evidence", "")),
            "modeling_risk_summary",
            column=risk.get("column"),
        )
        for index, risk in enumerate((modeling.get("risks") or [])[:5], start=1)
    ]
    recommendations = _recommendations(quality, leakage, modeling, response, ranking)
    unknowns = _unknowns(response, drift)
    limitations = [
        _summary_item("evidence_only", caveat, "caveats")
        for caveat in (context.get("caveats") or [])[:5]
    ] or [
        _summary_item(
            "evidence_only",
            "Summary is limited to deterministic evidence and does not inspect raw rows.",
            "caveats",
        )
    ]
    human_review_notes = [
        _summary_item(
            "human_review_required",
            "Confirm data lineage, target definition, feature timing, and accepted quality exceptions before modeling.",
            "quality_checks",
        )
    ]

    referenced_columns = sorted(
        {
            str(item.get("column"))
            for collection in [quality[:5], leakage[:5], ranking[:5], (modeling.get("risks") or [])[:5]]
            for item in collection
            if item.get("column")
        }
    )
    return {
        "status": "deterministic_fallback",
        "allowed_context": "evidence_packet_only",
        "observed_facts": observed_facts,
        "calculated_metrics": calculated_metrics,
        "inferred_risks": inferred_risks,
        "recommendations": recommendations,
        "unknowns": unknowns,
        "limitations": limitations,
        "human_review_notes": human_review_notes,
        "referenced_evidence_sections": sorted(referenced_sections),
        "referenced_columns": referenced_columns,
    }


def deterministic_summary_placeholder(evidence_packet: dict[str, Any]) -> dict[str, Any]:
    """Provide a non-LLM summary while preserving the AI layer contract."""

    summary = build_evidence_summary(evidence_packet)
    summary["message"] = "AI summary is optional; this deterministic fallback summarizes only approved evidence."
    return summary


def _sanitized_overview(overview: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(overview)
    sanitized.pop("sample_rows", None)
    data_dictionary = []
    for row in overview.get("data_dictionary", []):
        if isinstance(row, dict):
            clean_row = dict(row)
            clean_row.pop("sample_values", None)
            data_dictionary.append(clean_row)
    sanitized["data_dictionary"] = data_dictionary
    sanitized["row_samples_removed_for_ai"] = True
    return sanitized


def _summary_item(
    item_id: str,
    text: str,
    evidence_section: str,
    column: Any | None = None,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "text": text,
        "evidence_section": evidence_section,
        "column": column,
    }


def _recommendations(
    quality: list[dict[str, Any]],
    leakage: list[dict[str, Any]],
    modeling: dict[str, Any],
    response: dict[str, Any],
    ranking: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recommendations = []
    if leakage:
        recommendations.append(
            _summary_item(
                "review_leakage",
                "Review leakage warnings before modeling or interpreting feature importance.",
                "leakage_warnings",
            )
        )
    if quality:
        recommendations.append(
            _summary_item(
                "resolve_quality",
                "Resolve high and medium data quality warnings or document accepted exceptions.",
                "data_quality_warnings",
            )
        )
    if modeling.get("overall_status") in {"warn", "fail"}:
        recommendations.append(
            _summary_item(
                "review_modeling_risks",
                "Review deterministic modeling risk signals before downstream modeling.",
                "modeling_risk_summary",
            )
        )
    if response.get("problem_type") == "binary_classification":
        recommendations.append(
            _summary_item(
                "confirm_positive_class",
                "Confirm that the selected positive class matches the project definition.",
                "response_summary",
                column=response.get("response_column"),
            )
        )
    if ranking:
        recommendations.append(
            _summary_item(
                "review_top_features",
                "Review top-ranked features with business timing and data availability constraints.",
                "feature_ranking",
            )
        )
    return recommendations


def _unknowns(response: dict[str, Any], drift: dict[str, Any]) -> list[dict[str, Any]]:
    unknowns = []
    if not response.get("available"):
        unknowns.append(
            _summary_item(
                "response_unavailable",
                str(response.get("reason", "Response-aware analysis was unavailable.")),
                "response_summary",
            )
        )
    if not drift.get("available"):
        unknowns.append(
            _summary_item(
                "drift_unavailable",
                str(drift.get("reason", "Drift comparison was unavailable.")),
                "drift_summary",
            )
        )
    return unknowns
