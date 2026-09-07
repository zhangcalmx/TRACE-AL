"""Frozen deterministic rule-policy 1.4 and active local research engine."""

from __future__ import annotations

from pathlib import Path

from .candidate_rule_policy_v1_4 import CandidateRulePolicyV14
from .paths import project_root
from .risk_rules import RiskRuleEngine
from .schemas import DerivedCaseFeatures, RiskAssessment

PROJECT_ROOT = project_root()
DEFAULT_FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs" / "rule_policy_v1_4.yaml"


class FrozenRulePolicyV14(CandidateRulePolicyV14):
    """Apply the investigator-authorized, hash-frozen v1.4 aggregation policy."""

    DEFAULT_CONFIG_PATH = DEFAULT_FROZEN_CONFIG_PATH
    EXPECTED_POLICY_VERSION = "1.4"
    EXPECTED_POLICY_STATUS = "frozen_research_validation_policy"
    PRIMARY_ALERT_INTERPRETATION = "达到冻结规则1.4累计主预警阈值。"


class FrozenRiskRuleEngineV14:
    """Expose v1.4 as the system rule engine while preserving v1.1 signals.

    ``configs/rules.yaml`` remains the evidence-bound Boolean trigger layer.
    The frozen v1.4 policy replaces only cumulative scoring, interactions, and
    the main-alert threshold. Immediate-action safety behavior remains
    independent of the cumulative score.
    """

    def __init__(
        self,
        rules_path: Path | None = None,
        policy_path: Path | None = None,
    ) -> None:
        self.base_engine = RiskRuleEngine(rules_path)
        self.rules_path = self.base_engine.rules_path
        self.policy = FrozenRulePolicyV14(policy_path, self.rules_path)
        self.policy_version = self.policy.policy_version

    def evidence_key(self, rule_id: str) -> str:
        return self.base_engine.evidence_key(rule_id)

    def evaluate(
        self,
        entity_a_flags: list[str] | tuple[str, ...],
        entity_b_flags: list[str] | tuple[str, ...],
        patient_factors: list[str] | tuple[str, ...] = (),
        *,
        input_completeness: float | None = None,
    ) -> RiskAssessment:
        patient_flags = list(dict.fromkeys([*entity_a_flags, *patient_factors]))
        surgery_flags = list(dict.fromkeys(entity_b_flags))
        base = self.base_engine.evaluate(
            patient_flags,
            surgery_flags,
            input_completeness=input_completeness,
        )
        applied = self.policy.evaluate(
            base,
            DerivedCaseFeatures(
                patient_flags=patient_flags,
                surgery_flags=surgery_flags,
                input_completeness=base.input_completeness,
            ),
        )
        return base.model_copy(
            update={
                "policy_version": applied.policy_version,
                "primary_alert": applied.primary_alert,
                "risk_score": applied.risk_score,
                "primary_alert_threshold": applied.primary_alert_threshold,
                "requires_immediate_action": applied.requires_immediate_action,
            }
        )
