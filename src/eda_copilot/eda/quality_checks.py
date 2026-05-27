from __future__ import annotations

from collections import Counter
from typing import Any

from eda_copilot.core.config import EDAConfig
from eda_copilot.utils.serialization import to_jsonable


def build_quality_checks(
    config: EDAConfig,
    dataset_overview: dict[str, Any],
    missingness_summary: dict[str, Any],
    bivariate_summary: dict[str, Any],
    data_quality_warnings: list[dict[str, Any]],
    leakage_warnings: list[dict[str, Any]],
    drift_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic pass/fail checks for audit and CI-style review."""

    checks: list[dict[str, Any]] = [
        _check(
            check_id="dataset.non_empty",
            name="Dataset contains rows and columns",
            status="pass",
            severity="high",
            metric_name="shape",
            metric_value=f"{dataset_overview.get('row_count', 0)}x{dataset_overview.get('column_count', 0)}",
            threshold="rows > 0 and columns > 0",
            evidence="Input validation passed before profiling.",
        )
    ]
    checks.extend(_warning_checks(data_quality_warnings))
    checks.extend(_leakage_checks(leakage_warnings))
    checks.extend(_missingness_checks(config, missingness_summary))
    checks.extend(_correlation_checks(config, bivariate_summary))
    checks.extend(_drift_checks(config, drift_summary))

    counts = Counter(check["status"] for check in checks)
    overall_status = "fail" if counts.get("fail", 0) else "warn" if counts.get("warn", 0) else "pass"
    return to_jsonable(
        {
            "overall_status": overall_status,
            "summary": {
                "pass": int(counts.get("pass", 0)),
                "warn": int(counts.get("warn", 0)),
                "fail": int(counts.get("fail", 0)),
                "total": len(checks),
            },
            "checks": checks,
        }
    )


def _warning_checks(warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for warning in warnings:
        severity = str(warning.get("severity", "low"))
        status = "fail" if severity == "high" else "warn"
        checks.append(
            _check(
                check_id=f"quality.{warning.get('issue_type', 'warning')}.{warning.get('column') or 'dataset'}",
                name=f"Data quality: {warning.get('issue_type')}",
                status=status,
                severity=severity,
                metric_name=str(warning.get("issue_type")),
                metric_value=warning.get("column") or "dataset",
                threshold="no high-severity warnings",
                evidence=str(warning.get("evidence", "")),
                recommended_action=warning.get("recommended_action"),
            )
        )
    return checks


def _leakage_checks(warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for warning in warnings:
        severity = str(warning.get("severity", "high"))
        checks.append(
            _check(
                check_id=f"leakage.{warning.get('column', 'unknown')}",
                name="Potential leakage candidate",
                status="fail" if severity == "high" else "warn",
                severity=severity,
                metric_name=str(warning.get("reason", "leakage_warning")),
                metric_value=warning.get("column"),
                threshold="no high-severity leakage warnings",
                evidence=str(warning.get("evidence", "")),
                recommended_action=warning.get("recommended_next_step"),
            )
        )
    return checks


def _missingness_checks(config: EDAConfig, missingness_summary: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for row in missingness_summary.get("columns", []):
        value = float(row.get("missing_percentage", 0.0))
        if value < config.high_missingness_threshold:
            continue
        checks.append(
            _check(
                check_id=f"missingness.high.{row.get('column')}",
                name="Column missingness below threshold",
                status="fail" if value >= 0.95 else "warn",
                severity="high" if value >= 0.95 else "medium",
                metric_name="missing_percentage",
                metric_value=value,
                threshold=config.high_missingness_threshold,
                evidence=f"{row.get('column')} has {value:.1%} missing values.",
                recommended_action="Assess missingness mechanism and decide whether to impute, exclude, or fix upstream.",
            )
        )
    if not checks:
        checks.append(
            _check(
                check_id="missingness.high_columns",
                name="No high-missingness columns",
                status="pass",
                severity="medium",
                metric_name="missing_percentage",
                metric_value=0,
                threshold=config.high_missingness_threshold,
                evidence="No column exceeded the configured high-missingness threshold.",
            )
        )
    return checks


def _correlation_checks(config: EDAConfig, bivariate_summary: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = bivariate_summary.get("high_correlation_pairs", [])
    if not pairs:
        return [
            _check(
                check_id="correlation.high_pairs",
                name="No high-correlation feature pairs",
                status="pass",
                severity="medium",
                metric_name="abs_correlation",
                metric_value=0,
                threshold=config.high_correlation_threshold,
                evidence="No feature pairs exceeded the configured correlation threshold.",
            )
        ]
    return [
        _check(
            check_id=f"correlation.high.{pair.get('left')}.{pair.get('right')}",
            name="High-correlation feature pair",
            status="warn",
            severity="medium",
            metric_name="abs_correlation",
            metric_value=abs(float(pair.get("correlation", 0.0))),
            threshold=config.high_correlation_threshold,
            evidence=f"{pair.get('left')} and {pair.get('right')} correlation is {float(pair.get('correlation', 0.0)):.3f}.",
            recommended_action="Review redundancy before modeling.",
        )
        for pair in pairs
    ]


def _drift_checks(config: EDAConfig, drift_summary: dict[str, Any]) -> list[dict[str, Any]]:
    if not drift_summary.get("available"):
        return [
            _check(
                check_id="drift.available",
                name="Reference/current drift comparison configured",
                status="warn",
                severity="low",
                metric_name="comparison_groups",
                metric_value=0,
                threshold="train_test_column or segment_columns configured",
                evidence=str(drift_summary.get("reason", "Drift comparison was not available.")),
            )
        ]

    checks = []
    shifted = drift_summary.get("top_shifted_features", [])
    for row in shifted:
        value = abs(float(row.get("value", 0.0)))
        if value >= config.drift_fail_threshold:
            status = "fail"
            severity = "high"
        elif value >= config.drift_warning_threshold:
            status = "warn"
            severity = "medium"
        else:
            continue
        checks.append(
            _check(
                check_id=f"drift.{row.get('metric')}.{row.get('column')}",
                name="Distribution drift within threshold",
                status=status,
                severity=severity,
                metric_name=str(row.get("metric")),
                metric_value=value,
                threshold=config.drift_fail_threshold if status == "fail" else config.drift_warning_threshold,
                evidence=f"{row.get('column')} drift metric is {value:.3f}.",
                recommended_action="Compare reference/current data lineage and segment mix before modeling.",
            )
        )
    if not checks:
        checks.append(
            _check(
                check_id="drift.thresholds",
                name="No drift metrics exceeded thresholds",
                status="pass",
                severity="medium",
                metric_name="max_abs_drift",
                metric_value=max((abs(float(row.get("value", 0.0))) for row in shifted), default=0.0),
                threshold=config.drift_warning_threshold,
                evidence="No compared feature exceeded configured drift thresholds.",
            )
        )
    return checks


def _check(
    check_id: str,
    name: str,
    status: str,
    severity: str,
    metric_name: str,
    metric_value: Any,
    threshold: Any,
    evidence: str,
    recommended_action: str | None = None,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "name": name,
        "status": status,
        "severity": severity,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "threshold": threshold,
        "evidence": evidence,
        "recommended_action": recommended_action,
    }
