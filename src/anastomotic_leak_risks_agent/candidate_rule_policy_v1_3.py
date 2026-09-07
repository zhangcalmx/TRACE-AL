"""Exploratory domain-capped and interaction-aware rule policy v1.3."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import project_root
from .schemas import RiskAssessment

PROJECT_ROOT = project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "rule_policy_v1_3_candidate.yaml"
DEFAULT_BASE_RULES_PATH = PROJECT_ROOT / "configs" / "rules.yaml"


@dataclass(frozen=True)
class InteractionDefinition:
    interaction_id: str
    all_rules: tuple[str, ...]
    bonus: int


@dataclass(frozen=True)
class DomainDefinition:
    domain_id: str
    aggregation: str
    cap: int
    rules: tuple[str, ...]


@dataclass(frozen=True)
class CandidateV13Assessment:
    policy_version: str
    base_policy_version: str
    risk_score: int
    primary_alert: bool
    primary_alert_threshold: int
    requires_immediate_action: bool
    risk_rule_ids: tuple[str, ...]
    zero_weight_context_rule_ids: tuple[str, ...]
    triggered_interactions: tuple[str, ...]
    score_breakdown: dict[str, int]
    interpretation: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CandidateRulePolicyV13:
    """Rescore v1.1 signals without changing the signed rule definitions."""

    def __init__(
        self,
        config_path: Path | None = None,
        base_rules_path: Path | None = None,
    ) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.base_rules_path = base_rules_path or DEFAULT_BASE_RULES_PATH
        payload = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Candidate policy config must be a mapping")
        self._load(payload)

    def _load(self, payload: dict[str, Any]) -> None:
        self.policy_version = str(payload.get("policy_version", "")).strip()
        self.policy_status = str(payload.get("policy_status", "")).strip()
        self.base_policy_version = str(payload.get("base_policy_version", "")).strip()
        self.base_rules_sha256 = str(payload.get("base_rules_sha256", "")).strip().lower()
        self.lr15_enabled = payload.get("lr15_enabled")
        self.external_api_required = payload.get("external_api_required")
        self.primary_alert_minimum_score = self._nonnegative_int(
            payload.get("primary_alert_minimum_score"),
            "primary_alert_minimum_score",
            positive=True,
        )

        if self.policy_version != "1.3-candidate":
            raise ValueError("Candidate policy_version must be 1.3-candidate")
        if self.policy_status != "exploratory_not_for_clinical_deployment":
            raise ValueError("Candidate policy must remain exploratory")
        if self.base_policy_version != "1.1":
            raise ValueError("Candidate policy must be layered on signed v1.1")
        if self.lr15_enabled is not False:
            raise ValueError("LR15 must remain disabled")
        if self.external_api_required is not False:
            raise ValueError("Candidate policy must not require an external API")
        if _sha256(self.base_rules_path) != self.base_rules_sha256:
            raise ValueError("Signed base rules hash does not match candidate config")

        raw_weights = payload.get("rule_weights")
        if not isinstance(raw_weights, dict) or not raw_weights:
            raise TypeError("rule_weights must be a non-empty mapping")
        self.rule_weights = {
            str(rule_id): self._nonnegative_int(weight, f"rule_weights.{rule_id}")
            for rule_id, weight in raw_weights.items()
        }

        raw_domains = payload.get("domains")
        if not isinstance(raw_domains, dict):
            raise TypeError("domains must be a mapping")
        domains: list[DomainDefinition] = []
        assigned_rules: set[str] = set()
        for domain_id, raw in raw_domains.items():
            if not isinstance(raw, dict):
                raise TypeError(f"domains.{domain_id} must be a mapping")
            aggregation = str(raw.get("aggregation", ""))
            if aggregation not in {"max", "sum"}:
                raise ValueError(f"domains.{domain_id}.aggregation must be max or sum")
            rules = tuple(str(item) for item in raw.get("rules") or [])
            if not rules:
                raise ValueError(f"domains.{domain_id}.rules must not be empty")
            unknown = set(rules) - self.rule_weights.keys()
            duplicate = assigned_rules.intersection(rules)
            if unknown:
                raise ValueError(f"domains.{domain_id} has unknown rules: {sorted(unknown)}")
            if duplicate:
                raise ValueError(f"Rules assigned to more than one domain: {sorted(duplicate)}")
            assigned_rules.update(rules)
            domains.append(
                DomainDefinition(
                    domain_id=str(domain_id),
                    aggregation=aggregation,
                    cap=self._nonnegative_int(raw.get("cap"), f"domains.{domain_id}.cap"),
                    rules=rules,
                )
            )
        self.domains = tuple(domains)
        self.domain_rule_ids = frozenset(assigned_rules)

        raw_interactions = payload.get("interactions")
        if not isinstance(raw_interactions, list):
            raise TypeError("interactions must be a list")
        interactions: list[InteractionDefinition] = []
        interaction_ids: set[str] = set()
        for index, raw in enumerate(raw_interactions):
            if not isinstance(raw, dict):
                raise TypeError(f"interactions[{index}] must be a mapping")
            interaction_id = str(raw.get("id", "")).strip()
            rules = tuple(str(item) for item in raw.get("all_rules") or [])
            if not interaction_id or interaction_id in interaction_ids:
                raise ValueError("Interaction ids must be non-empty and unique")
            if len(rules) < 2 or set(rules) - self.rule_weights.keys():
                raise ValueError(f"Interaction {interaction_id} must reference at least two known rules")
            interaction_ids.add(interaction_id)
            interactions.append(
                InteractionDefinition(
                    interaction_id=interaction_id,
                    all_rules=rules,
                    bonus=self._nonnegative_int(raw.get("bonus"), f"interactions.{interaction_id}.bonus"),
                )
            )
        self.interactions = tuple(interactions)

        safety = payload.get("safety")
        if not isinstance(safety, dict):
            raise TypeError("safety must be a mapping")
        if safety.get("immediate_action_independent_of_score") is not True:
            raise ValueError("Immediate-action safety behavior cannot be disabled")
        completeness = safety.get("minimum_input_completeness_for_low_output")
        if not isinstance(completeness, (int, float)) or not 0 <= float(completeness) <= 1:
            raise ValueError("minimum completeness must be between 0 and 1")
        self.minimum_input_completeness_for_low_output = float(completeness)

    @staticmethod
    def _nonnegative_int(value: Any, name: str, *, positive: bool = False) -> int:
        minimum = 1 if positive else 0
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{name} must be a {qualifier} integer")
        return value

    def evaluate(self, assessment: RiskAssessment) -> CandidateV13Assessment:
        """Return the v1.3 candidate score and alert while retaining safety state."""
        if assessment.policy_version != self.base_policy_version:
            raise ValueError(
                f"Expected base policy {self.base_policy_version}, got {assessment.policy_version}"
            )
        active = {signal.rule_id for signal in assessment.signals}
        breakdown: dict[str, int] = {}

        for rule_id, weight in self.rule_weights.items():
            if rule_id in self.domain_rule_ids or rule_id not in active:
                continue
            breakdown[f"rule:{rule_id}"] = weight

        for domain in self.domains:
            triggered_weights = [self.rule_weights[rule_id] for rule_id in domain.rules if rule_id in active]
            if not triggered_weights:
                continue
            raw_score = max(triggered_weights) if domain.aggregation == "max" else sum(triggered_weights)
            breakdown[f"domain:{domain.domain_id}"] = min(raw_score, domain.cap)

        triggered_interactions: list[str] = []
        for interaction in self.interactions:
            if not all(rule_id in active for rule_id in interaction.all_rules):
                continue
            triggered_interactions.append(interaction.interaction_id)
            breakdown[f"interaction:{interaction.interaction_id}"] = interaction.bonus

        score = sum(breakdown.values())
        primary_alert = score >= self.primary_alert_minimum_score
        zero_weight = tuple(sorted(rule_id for rule_id in active if self.rule_weights.get(rule_id, 0) == 0))
        if assessment.requires_immediate_action:
            interpretation = "独立安全护栏触发；无论候选累计分数多少均需立即处置。"
        elif primary_alert:
            interpretation = "达到规则1.3候选累计预警阈值。"
        elif assessment.input_completeness < self.minimum_input_completeness_for_low_output:
            interpretation = "未达到预警阈值，但输入不完整，禁止解释为低风险。"
        else:
            interpretation = "未达到候选预警阈值；不等同于零风险。"
        return CandidateV13Assessment(
            policy_version=self.policy_version,
            base_policy_version=self.base_policy_version,
            risk_score=score,
            primary_alert=primary_alert,
            primary_alert_threshold=self.primary_alert_minimum_score,
            requires_immediate_action=assessment.requires_immediate_action,
            risk_rule_ids=tuple(sorted(active)),
            zero_weight_context_rule_ids=zero_weight,
            triggered_interactions=tuple(triggered_interactions),
            score_breakdown=dict(sorted(breakdown.items())),
            interpretation=interpretation,
        )
