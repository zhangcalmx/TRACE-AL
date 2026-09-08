"""Deterministic YAML-driven risk-rule engine with explicit Boolean logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import project_root
from .schemas import RISK_ORDER, RiskAssessment, RiskLevel, RiskSignal

PROJECT_ROOT = project_root()
MINIMUM_LOW_RISK_COMPLETENESS = 0.8
RULE_SIGNAL_WEIGHTS = {
    RiskLevel.HIGH: 2,
    RiskLevel.MEDIUM: 1,
    RiskLevel.LOW: 0,
    RiskLevel.UNKNOWN: 0,
}
PRIMARY_ALERT_MINIMUM_WEIGHTED_SCORE = 4


@dataclass(frozen=True)
class SeverityOverride:
    severity: RiskLevel
    condition: dict[str, Any]


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    severity: RiskLevel
    risk_type: str
    signal_class: str
    condition: dict[str, Any]
    severity_overrides: tuple[SeverityOverride, ...]
    rationale: str
    recommendation: str
    evidence_key: str


def _validate_condition(node: Any, *, location: str) -> None:
    """Validate the small, auditable condition DSL used in ``rules.yaml``."""
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError(f"{location} must contain exactly one condition operator")
    operator, value = next(iter(node.items()))
    if operator in {"patient", "surgery"}:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{location}.{operator} must be a non-empty flag")
        return
    if operator not in {"all", "any"}:
        raise ValueError(f"{location} uses unsupported operator: {operator}")
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}.{operator} must be a non-empty list")
    for index, child in enumerate(value):
        _validate_condition(child, location=f"{location}.{operator}[{index}]")


def _evaluate_condition(
    node: dict[str, Any], active_a: set[str], active_b: set[str]
) -> tuple[bool, dict[str, list[str]]]:
    operator, value = next(iter(node.items()))
    if operator in {"patient", "surgery"}:
        active = active_a if operator == "patient" else active_b
        matched = value in active
        return matched, {
            "entity_a": [value] if matched and operator == "patient" else [],
            "entity_b": [value] if matched and operator == "surgery" else [],
        }

    child_results = [_evaluate_condition(child, active_a, active_b) for child in value]
    matched = (
        all(result[0] for result in child_results) if operator == "all" else any(result[0] for result in child_results)
    )
    if not matched:
        return False, {"entity_a": [], "entity_b": []}
    return True, {
        "entity_a": sorted({flag for ok, terms in child_results if ok for flag in terms["entity_a"]}),
        "entity_b": sorted({flag for ok, terms in child_results if ok for flag in terms["entity_b"]}),
    }


class RiskRuleEngine:
    """Evaluate active feature flags using explicit ``all``/``any`` conditions.

    A case with no triggered rule is classified as LOW only when at least 80%
    of the predefined assessment domains are complete. Otherwise it is UNKNOWN.
    """

    def __init__(self, rules_path: Path | None = None) -> None:
        self.rules_path = rules_path or PROJECT_ROOT / "configs" / "rules.yaml"
        self.rules, self.patient_factor_rules, self.policy_version = self._load()
        self._evidence_keys = {rule.rule_id: rule.evidence_key for rule in [*self.rules, *self.patient_factor_rules]}

    @staticmethod
    def _definition(raw: dict[str, Any], *, default_id: str, location: str) -> RuleDefinition:
        rule_id = str(raw.get("id", default_id)).strip()
        if not rule_id:
            raise ValueError(f"{location} has no id")
        try:
            severity = RiskLevel(str(raw.get("severity", "unknown")).lower())
        except ValueError as exc:
            raise ValueError(f"Rule {rule_id} has invalid severity") from exc
        condition = raw.get("when")
        _validate_condition(condition, location=f"{location}.when")
        signal_class = str(raw.get("signal_class", "risk_marker"))
        if signal_class not in {"safety_guardrail", "risk_marker", "context_review"}:
            raise ValueError(f"Rule {rule_id} has invalid signal_class")
        overrides: list[SeverityOverride] = []
        for index, override in enumerate(raw.get("severity_overrides") or [], start=1):
            try:
                override_severity = RiskLevel(str(override.get("severity", "")).lower())
            except (AttributeError, ValueError) as exc:
                raise ValueError(f"Rule {rule_id} severity_overrides[{index}] has invalid severity") from exc
            override_condition = override.get("when")
            _validate_condition(
                override_condition,
                location=f"{location}.severity_overrides[{index}].when",
            )
            if RISK_ORDER[override_severity] <= RISK_ORDER[severity]:
                raise ValueError(f"Rule {rule_id} severity override must be higher than its base severity")
            overrides.append(SeverityOverride(severity=override_severity, condition=override_condition))
        return RuleDefinition(
            rule_id=rule_id,
            severity=severity,
            risk_type=str(raw.get("risk_type", "patient_factor_composite")),
            signal_class=signal_class,
            condition=condition,
            severity_overrides=tuple(overrides),
            rationale=str(raw.get("rationale", "")).strip(),
            recommendation=str(raw.get("recommendation", "")).strip(),
            evidence_key=str(raw.get("evidence_key", rule_id.removeprefix("patient_factor."))),
        )

    def _load(self) -> tuple[list[RuleDefinition], list[RuleDefinition], str]:
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {self.rules_path}")
        payload = yaml.safe_load(self.rules_path.read_text(encoding="utf-8")) or {}
        definitions = [
            self._definition(raw, default_id="", location=f"rules[{index}]")
            for index, raw in enumerate(payload.get("rules", []), start=1)
        ]
        if not definitions:
            raise ValueError("No rules found in rules.yaml")

        patient_definitions: list[RuleDefinition] = []
        for factor_id, raw in (payload.get("patient_factor_rules") or {}).items():
            normalized = dict(raw)
            normalized["id"] = f"patient_factor.{factor_id}"
            normalized.setdefault("risk_type", "patient_factor_composite")
            normalized.setdefault(
                "recommendation",
                "建议由外科、麻醉、营养等多学科团队复核复合风险并制定缓解措施。",
            )
            patient_definitions.append(
                self._definition(
                    normalized,
                    default_id=f"patient_factor.{factor_id}",
                    location=f"patient_factor_rules.{factor_id}",
                )
            )
        return definitions, patient_definitions, str(payload.get("policy_label", "")).strip()

    def evidence_key(self, rule_id: str) -> str:
        return self._evidence_keys.get(rule_id, rule_id.removeprefix("patient_factor."))

    def evaluate(
        self,
        entity_a_flags: list[str] | tuple[str, ...],
        entity_b_flags: list[str] | tuple[str, ...],
        patient_factors: list[str] | tuple[str, ...] = (),
        *,
        input_completeness: float | None = None,
    ) -> RiskAssessment:
        active_a = set(entity_a_flags) | set(patient_factors)
        active_b = set(entity_b_flags)
        completeness = max(0.0, min(1.0, input_completeness or 0.0))
        signals: list[RiskSignal] = []

        for rule in [*self.rules, *self.patient_factor_rules]:
            matched, matched_terms = _evaluate_condition(rule.condition, active_a, active_b)
            if matched:
                effective_severity = rule.severity
                for override in rule.severity_overrides:
                    override_matched, override_terms = _evaluate_condition(override.condition, active_a, active_b)
                    if not override_matched:
                        continue
                    if RISK_ORDER[override.severity] > RISK_ORDER[effective_severity]:
                        effective_severity = override.severity
                    matched_terms = {
                        "entity_a": sorted(set(matched_terms["entity_a"]) | set(override_terms["entity_a"])),
                        "entity_b": sorted(set(matched_terms["entity_b"]) | set(override_terms["entity_b"])),
                    }
                signals.append(
                    RiskSignal(
                        rule_id=rule.rule_id,
                        risk_level=effective_severity,
                        risk_type=rule.risk_type,
                        signal_class=rule.signal_class,
                        rationale=rule.rationale,
                        recommendation=rule.recommendation,
                        matched_terms=matched_terms,
                    )
                )

        signals.sort(key=lambda item: RISK_ORDER[item.risk_level], reverse=True)
        if signals:
            risk_level = max((signal.risk_level for signal in signals), key=RISK_ORDER.__getitem__)
        else:
            risk_level = RiskLevel.LOW if completeness >= MINIMUM_LOW_RISK_COMPLETENESS else RiskLevel.UNKNOWN

        coverage_status = (
            "complete"
            if completeness >= MINIMUM_LOW_RISK_COMPLETENESS
            else "partial"
            if completeness > 0
            else "insufficient"
        )
        risk_types = list(dict.fromkeys(signal.risk_type for signal in signals))
        safety_signals = [signal for signal in signals if signal.signal_class == "safety_guardrail"]
        risk_score = sum(RULE_SIGNAL_WEIGHTS[signal.risk_level] for signal in signals)
        return RiskAssessment(
            risk_level=risk_level,
            risk_types=risk_types,
            signals=signals,
            policy_version=self.policy_version,
            primary_alert=risk_score >= PRIMARY_ALERT_MINIMUM_WEIGHTED_SCORE,
            risk_score=risk_score,
            primary_alert_threshold=PRIMARY_ALERT_MINIMUM_WEIGHTED_SCORE,
            safety_guardrail_present=bool(safety_signals),
            requires_immediate_action=any(signal.risk_level == RiskLevel.HIGH for signal in safety_signals),
            coverage_status=coverage_status,
            input_completeness=completeness,
            confidence=coverage_status,
            confidence_score=completeness,
        )
