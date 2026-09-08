"""Active local research rule engine and its aggregation policy."""

from __future__ import annotations

from pathlib import Path

from .candidate_rule_policy import CandidateRulePolicy
from .paths import project_root
from .risk_rules import RiskRuleEngine
from .schemas import DerivedCaseFeatures, RiskAssessment

PROJECT_ROOT = project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "rule_policy.yaml"


class RulePolicy(CandidateRulePolicy):
    """Apply the investigator-authorized aggregation policy."""

    DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_PATH
    EXPECTED_POLICY_STATUS = "research_validation_policy"
    PRIMARY_ALERT_INTERPRETATION = "达到规则累计主预警阈值。"


class RiskRulePolicyEngine:
    """Expose the rule policy as the system rule engine while running the
    Boolean trigger layer.

    ``configs/rules.yaml`` remains the evidence-bound Boolean trigger layer.
    The policy layer performs cumulative scoring, interactions, and the
    main-alert threshold. Immediate-action safety behavior remains independent
    of the cumulative score.
    """

    def __init__(
        self,
        rules_path: Path | None = None,
        policy_path: Path | None = None,
    ) -> None:
        self.base_engine = RiskRuleEngine(rules_path)
        self.rules_path = self.base_engine.rules_path
        self.policy = RulePolicy(policy_path, self.rules_path)
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
