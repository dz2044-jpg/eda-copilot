from __future__ import annotations

from typing import Any

from eda_copilot.core.config import EDAConfig


def detect_leakage_risks(
    config: EDAConfig,
    type_summary: dict[str, Any],
    response_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Detect deterministic leakage risk signals from names and univariate strength."""

    warnings: list[dict[str, Any]] = []
    response_column = config.response_column
    for profile in type_summary.get("columns", []):
        column = profile["name"]
        if column == response_column:
            continue
        if profile.get("suspicious_name"):
            warnings.append(
                _leakage_warning(
                    column=column,
                    reason="Suspicious column name",
                    severity="medium",
                    evidence="Name contains target, outcome, decision, final, result, score, or post-event wording.",
                    recommended_next_step="Confirm feature creation timing relative to the response event.",
                )
            )

    if response_summary.get("problem_type") == "binary_classification":
        for relationship in response_summary.get("feature_relationships", []):
            column = relationship["column"]
            auc_abs = relationship.get("auc_abs")
            spread = relationship.get("response_rate_spread")
            if auc_abs is not None and auc_abs >= config.high_auc_leakage_threshold:
                warnings.append(
                    _leakage_warning(
                        column=column,
                        reason="Near-perfect univariate AUC",
                        severity="high",
                        evidence=f"Absolute AUC is {auc_abs:.3f}.",
                        recommended_next_step="Verify this predictor is available before the prediction time.",
                    )
                )
            elif spread is not None and spread >= 0.95:
                warnings.append(
                    _leakage_warning(
                        column=column,
                        reason="Near-perfect response separation",
                        severity="high",
                        evidence=f"Response-rate spread is {spread:.3f}.",
                        recommended_next_step="Check whether categories encode an outcome or business decision.",
                    )
                )
    return warnings


def _leakage_warning(
    column: str,
    reason: str,
    severity: str,
    evidence: str,
    recommended_next_step: str,
) -> dict[str, Any]:
    return {
        "column": column,
        "reason": reason,
        "severity": severity,
        "evidence": evidence,
        "recommended_next_step": recommended_next_step,
    }
