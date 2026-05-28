from __future__ import annotations

from collections import Counter
from typing import Any

from eda_copilot.core.config import EDAConfig
from eda_copilot.utils.serialization import to_jsonable


MISSINGNESS_RESPONSE_GAP_THRESHOLD = 0.20


def build_modeling_risk_summary(
    config: EDAConfig,
    type_summary: dict[str, Any],
    missingness_summary: dict[str, Any],
    bivariate_summary: dict[str, Any],
    feature_ranking: list[dict[str, Any]],
    data_quality_warnings: list[dict[str, Any]],
    leakage_warnings: list[dict[str, Any]],
    drift_summary: dict[str, Any],
    response_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic modeling-readiness risk signals from EDA evidence.

    The output is a review queue, not a modeling approval decision. It combines
    existing deterministic evidence into a compact, AI-safe section.
    """

    risks: list[dict[str, Any]] = []
    risks.extend(_leakage_risks(leakage_warnings))
    risks.extend(_data_quality_modeling_risks(data_quality_warnings))
    risks.extend(_type_profile_risks(type_summary, config))
    risks.extend(_missingness_response_risks(missingness_summary))
    risks.extend(_bivariate_risks(bivariate_summary, config))
    risks.extend(_drift_risks(drift_summary))
    risks.extend(_response_risks(response_summary))
    risks.extend(_ranking_review_risks(feature_ranking))

    risks = _deduplicate_risks(risks)
    severity_counts = Counter(str(risk.get("severity", "low")) for risk in risks)
    category_counts = Counter(str(risk.get("category", "modeling")) for risk in risks)
    overall_status = "fail" if severity_counts.get("high", 0) else "warn" if risks else "pass"
    return to_jsonable(
        {
            "available": True,
            "overall_status": overall_status,
            "summary": {
                "total": len(risks),
                "by_severity": dict(sorted(severity_counts.items())),
                "by_category": dict(sorted(category_counts.items())),
            },
            "risks": sorted(risks, key=lambda row: _severity_sort(str(row.get("severity", "low")))),
            "caveat": "Modeling risk signals are deterministic review aids and do not approve a model for use.",
        }
    )


def _leakage_risks(warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _risk(
            risk_id=f"leakage.{warning.get('column', 'unknown')}.{_token(warning.get('reason'))}",
            category="leakage",
            severity=str(warning.get("severity", "high")),
            column=warning.get("column"),
            evidence=str(warning.get("evidence", "")),
            recommended_action=str(warning.get("recommended_next_step", "Review feature timing.")),
            source="leakage_warnings",
            metric_name=str(warning.get("reason", "leakage_warning")),
            metric_value=warning.get("column"),
            threshold="no high-severity leakage warnings",
        )
        for warning in warnings
    ]


def _data_quality_modeling_risks(warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    category_by_issue = {
        "high_cardinality": "cardinality",
        "id_like_candidate": "identifier",
        "high_missingness": "missingness",
        "all_missing": "missingness",
        "near_constant_column": "sparsity",
        "constant_column": "sparsity",
        "normalized_name_collision": "schema",
        "mixed_python_types": "schema",
        "numeric_parse_candidate": "schema",
        "datetime_parse_candidate": "schema",
        "boolean_parse_candidate": "schema",
    }
    risks = []
    for warning in warnings:
        issue_type = str(warning.get("issue_type", ""))
        category = category_by_issue.get(issue_type)
        if category is None:
            continue
        risks.append(
            _risk(
                risk_id=f"{category}.{warning.get('warning_id', issue_type)}",
                category=category,
                severity=str(warning.get("severity", "low")),
                column=warning.get("column"),
                evidence=str(warning.get("evidence", "")),
                recommended_action=str(warning.get("recommended_action", "Review before modeling.")),
                source="data_quality_warnings",
                metric_name=warning.get("metric_name"),
                metric_value=warning.get("metric_value"),
                threshold=warning.get("threshold"),
            )
        )
    return risks


def _type_profile_risks(type_summary: dict[str, Any], config: EDAConfig) -> list[dict[str, Any]]:
    risks = []
    for profile in type_summary.get("columns", []):
        column = str(profile.get("name"))
        if column in config.analysis_exclusions():
            continue
        if profile.get("semantic_type") == "categorical_high_cardinality":
            risks.append(
                _risk(
                    risk_id=f"cardinality.{_token(column)}",
                    category="cardinality",
                    severity="medium",
                    column=column,
                    evidence=f"{profile.get('unique_count')} unique values detected.",
                    recommended_action="Review encoding strategy and leakage risk for high-cardinality predictors.",
                    source="column_type_summary",
                    metric_name="unique_count",
                    metric_value=profile.get("unique_count"),
                    threshold=config.high_cardinality_threshold,
                )
            )
        if profile.get("semantic_type") in {"constant", "near_constant"}:
            risks.append(
                _risk(
                    risk_id=f"sparsity.{_token(column)}",
                    category="sparsity",
                    severity="medium" if profile.get("semantic_type") == "constant" else "low",
                    column=column,
                    evidence=f"Column semantic type is {profile.get('semantic_type')}.",
                    recommended_action="Exclude or justify low-variance predictors before modeling.",
                    source="column_type_summary",
                    metric_name="top_frequency_ratio",
                    metric_value=profile.get("top_frequency_ratio"),
                    threshold=config.near_constant_threshold,
                )
            )
    return risks


def _missingness_response_risks(missingness_summary: dict[str, Any]) -> list[dict[str, Any]]:
    risks = []
    for row in missingness_summary.get("missingness_by_response", []):
        gap = row.get("response_rate_gap")
        if gap is None or abs(float(gap)) < MISSINGNESS_RESPONSE_GAP_THRESHOLD:
            continue
        risks.append(
            _risk(
                risk_id=f"missingness_response.{_token(row.get('column'))}",
                category="missingness",
                severity="medium",
                column=row.get("column"),
                evidence=f"Missing-vs-present response-rate gap is {float(gap):.3f}.",
                recommended_action="Assess whether missingness is informative, biased, or caused by post-response data capture.",
                source="missingness_summary.missingness_by_response",
                metric_name="response_rate_gap",
                metric_value=gap,
                threshold=MISSINGNESS_RESPONSE_GAP_THRESHOLD,
            )
        )
    return risks


def _bivariate_risks(bivariate_summary: dict[str, Any], config: EDAConfig) -> list[dict[str, Any]]:
    risks = []
    for pair in bivariate_summary.get("high_correlation_pairs", []):
        risks.append(
            _risk(
                risk_id=f"correlation.{_token(pair.get('left'))}.{_token(pair.get('right'))}",
                category="correlation",
                severity="medium",
                column=pair.get("left"),
                related_column=pair.get("right"),
                evidence=f"Absolute correlation is {abs(float(pair.get('correlation', 0.0))):.3f}.",
                recommended_action="Review redundancy, stability, and model interpretability before using both predictors.",
                source="bivariate_summary.high_correlation_pairs",
                metric_name="abs_correlation",
                metric_value=abs(float(pair.get("correlation", 0.0))),
                threshold=config.high_correlation_threshold,
            )
        )
    for pair in bivariate_summary.get("possible_duplicate_columns", []):
        risks.append(
            _risk(
                risk_id=f"duplicate_column.{_token(pair.get('left'))}.{_token(pair.get('right'))}",
                category="duplicate_column",
                severity="medium",
                column=pair.get("left"),
                related_column=pair.get("right"),
                evidence=str(pair.get("reason", "Exact value match.")),
                recommended_action="Remove or justify duplicate predictors before modeling.",
                source="bivariate_summary.possible_duplicate_columns",
                metric_name="duplicate_column_pair",
                metric_value=True,
                threshold=False,
            )
        )
    return risks


def _drift_risks(drift_summary: dict[str, Any]) -> list[dict[str, Any]]:
    if not drift_summary.get("available"):
        return []
    risks = []
    for row in drift_summary.get("top_shifted_features", []):
        status = str(row.get("status", "pass"))
        if status == "pass":
            continue
        risks.append(
            _risk(
                risk_id=f"drift.{_token(row.get('column'))}.{_token(row.get('metric'))}",
                category="drift",
                severity="high" if status == "fail" else "medium",
                column=row.get("column"),
                evidence=f"{row.get('metric')} is {float(row.get('value', 0.0)):.3f}.",
                recommended_action="Compare reference/current data lineage and segment mix before modeling.",
                source="drift_summary.top_shifted_features",
                metric_name=row.get("metric"),
                metric_value=abs(float(row.get("value", 0.0))),
                threshold=drift_summary.get("fail_threshold") if status == "fail" else drift_summary.get("warning_threshold"),
            )
        )
    return risks


def _response_risks(response_summary: dict[str, Any]) -> list[dict[str, Any]]:
    risks = []
    for warning in response_summary.get("warnings", []):
        if warning.get("issue_type") == "positive_class_auto_selected":
            continue
        severity = str(warning.get("severity", "low"))
        risks.append(
            _risk(
                risk_id=f"response.{_token(warning.get('issue_type'))}",
                category="response",
                severity=severity,
                column=warning.get("column"),
                evidence=str(warning.get("evidence", "")),
                recommended_action=str(warning.get("recommended_action", "Review response configuration.")),
                source="response_summary.warnings",
                metric_name=warning.get("metric_name"),
                metric_value=warning.get("metric_value"),
                threshold=warning.get("threshold"),
            )
        )
    return risks


def _ranking_review_risks(feature_ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
    risks = []
    for row in feature_ranking[:10]:
        penalty = float(row.get("quality_penalty") or 0.0)
        signal_score = row.get("signal_score")
        if penalty < 0.5 or signal_score is None:
            continue
        risks.append(
            _risk(
                risk_id=f"ranked_feature_quality.{_token(row.get('column'))}",
                category="ranked_feature_quality",
                severity="medium",
                column=row.get("column"),
                evidence=f"Top-ranked feature has quality penalty {penalty:.2f}.",
                recommended_action="Resolve feature quality concerns before relying on univariate signal.",
                source="feature_ranking",
                metric_name="quality_penalty",
                metric_value=penalty,
                threshold=0.5,
            )
        )
    return risks


def _risk(
    risk_id: str,
    category: str,
    severity: str,
    column: Any,
    evidence: str,
    recommended_action: str,
    source: str,
    metric_name: Any,
    metric_value: Any,
    threshold: Any,
    related_column: Any | None = None,
) -> dict[str, Any]:
    return {
        "risk_id": risk_id,
        "category": category,
        "severity": severity,
        "column": column,
        "related_column": related_column,
        "evidence": evidence,
        "recommended_action": recommended_action,
        "source": source,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "threshold": threshold,
    }


def _deduplicate_risks(risks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for risk in risks:
        risk_id = str(risk.get("risk_id"))
        existing = deduped.get(risk_id)
        if existing is None or _severity_sort(str(risk.get("severity", "low"))) < _severity_sort(str(existing.get("severity", "low"))):
            deduped[risk_id] = risk
    return list(deduped.values())


def _severity_sort(severity: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(severity, 3)


def _token(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return "".join(character if character.isalnum() else "_" for character in text).strip("_") or "unknown"
