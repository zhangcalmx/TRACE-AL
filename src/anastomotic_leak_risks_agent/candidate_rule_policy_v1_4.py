"""Executable posthoc candidate rule policy.

This candidate changes only score aggregation and closes the corrected-versus-
persistent technical-integrity loop. It is not a deployable clinical model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import project_root
from .schemas import DerivedCaseFeatures, RiskAssessment

PROJECT_ROOT = project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "rule_policy_candidate.yaml"
DEFAULT_BASE_RULES_PATH = PROJECT_ROOT / "configs" / "rules.yaml"


@dataclass(frozen=True)
class DerivedGroup:
    group_id: str
    any_rules: tuple[str, ...]
    weight: int


@dataclass(frozen=True)
class Interaction:
    interaction_id: str
    all_rules: tuple[str, ...]
    bonus: int


@dataclass(frozen=True)
class CandidateV14Assessment:
    policy_version: str
    base_policy_version: str
    risk_score: int
    primary_alert: bool
    operational_alert: bool
    primary_alert_threshold: int
    requires_immediate_action: bool
    technical_integrity_status: str
    review_only_rule_ids: tuple[str, ...]
    triggered_interactions: tuple[str, ...]
    graded_context: dict[str, str]
    score_breakdown: dict[str, int]
    interpretation: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CandidateRulePolicyV14:
    """Apply the frozen candidate score to a signed assessment."""

    DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_PATH
    EXPECTED_POLICY_VERSION = "1.0-candidate"
    EXPECTED_POLICY_STATUS = "posthoc_exploratory_not_for_clinical_deployment"
    PRIMARY_ALERT_INTERPRETATION = "达到规则1.0后验候选累计预警阈值。"

    def __init__(
        self,
        config_path: Path | None = None,
        base_rules_path: Path | None = None,
    ) -> None:
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
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
        if self.policy_version != self.EXPECTED_POLICY_VERSION:
            raise ValueError(f"Policy version must be {self.EXPECTED_POLICY_VERSION}")
        if self.policy_status != self.EXPECTED_POLICY_STATUS:
            raise ValueError(f"Policy status must be {self.EXPECTED_POLICY_STATUS}")
        if self.base_policy_version:
            raise ValueError("Candidate policy must define no base policy version")
        if self.lr15_enabled is not False:
            raise ValueError("LR15 must remain disabled")
        if self.external_api_required is not False:
            raise ValueError("Candidate policy must not require an external API")
        if _sha256(self.base_rules_path) != self.base_rules_sha256:
            raise ValueError("Signed base rules hash does not match candidate config")

        raw_weights = payload.get("base_rule_weights")
        if not isinstance(raw_weights, dict) or not raw_weights:
            raise TypeError("base_rule_weights must be a non-empty mapping")
        self.base_rule_weights = {
            str(rule_id): self._nonnegative_int(weight, f"base_rule_weights.{rule_id}")
            for rule_id, weight in raw_weights.items()
        }

        raw_groups = payload.get("derived_groups")
        if not isinstance(raw_groups, dict):
            raise TypeError("derived_groups must be a mapping")
        groups: list[DerivedGroup] = []
        for group_id, raw in raw_groups.items():
            if not isinstance(raw, dict):
                raise TypeError(f"derived_groups.{group_id} must be a mapping")
            any_rules = tuple(str(item) for item in raw.get("any_rules") or [])
            if not any_rules:
                raise ValueError(f"derived_groups.{group_id}.any_rules must not be empty")
            groups.append(
                DerivedGroup(
                    group_id=str(group_id),
                    any_rules=any_rules,
                    weight=self._nonnegative_int(raw.get("weight"), f"derived_groups.{group_id}.weight"),
                )
            )
        self.derived_groups = tuple(groups)

        raw_interactions = payload.get("interactions")
        if not isinstance(raw_interactions, list):
            raise TypeError("interactions must be a list")
        interactions: list[Interaction] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_interactions):
            if not isinstance(raw, dict):
                raise TypeError(f"interactions[{index}] must be a mapping")
            interaction_id = str(raw.get("id", "")).strip()
            all_rules = tuple(str(item) for item in raw.get("all_rules") or [])
            if not interaction_id or interaction_id in seen or len(all_rules) < 2:
                raise ValueError("Interaction ids must be unique and reference at least two rules")
            seen.add(interaction_id)
            interactions.append(
                Interaction(
                    interaction_id=interaction_id,
                    all_rules=all_rules,
                    bonus=self._nonnegative_int(raw.get("bonus"), f"interactions.{interaction_id}.bonus"),
                )
            )
        self.interactions = tuple(interactions)

        loop = payload.get("technical_integrity_closed_loop")
        if not isinstance(loop, dict):
            raise TypeError("technical_integrity_closed_loop must be a mapping")
        self.technical_rule_id = str(loop.get("shared_rule_id", "")).strip()
        self.resolved_review_only_flags = frozenset(
            str(item) for item in loop.get("resolved_review_only_flags") or []
        )
        self.unresolved_immediate_action_flags = frozenset(
            str(item) for item in loop.get("unresolved_immediate_action_flags") or []
        )
        if self.technical_rule_id not in self.base_rule_weights:
            raise ValueError("technical shared_rule_id must exist in base_rule_weights")
        if not self.resolved_review_only_flags or not self.unresolved_immediate_action_flags:
            raise ValueError("technical closed-loop flag groups must not be empty")
        if loop.get("resolved_review_only_score") != 0:
            raise ValueError("Resolved technical defects must remain review-only with zero points")
        if loop.get("unresolved_score_from_base_rule_weights") is not True:
            raise ValueError("Unresolved technical defects must retain the configured rule score")

        safety = payload.get("safety")
        if not isinstance(safety, dict) or safety.get("immediate_action_independent_of_score") is not True:
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

    @staticmethod
    def _graded_context(patient_flags: set[str], surgery_flags: set[str]) -> dict[str, str]:
        if "surgery_duration_over_5h" in surgery_flags:
            duration = "over_5h"
        elif "surgery_duration_over_4h" in surgery_flags:
            duration = "over_4h_to_5h"
        elif "surgery_duration_over_3h" in surgery_flags:
            duration = "over_3h_to_4h"
        else:
            duration = "no_threshold_flag_or_unknown"

        if "blood_loss_over_1000ml" in surgery_flags:
            blood_loss = "over_1000ml"
        elif "blood_loss_over_500ml" in surgery_flags:
            blood_loss = "over_500ml_to_1000ml"
        elif "blood_loss_over_300ml" in surgery_flags:
            blood_loss = "over_300ml_to_500ml"
        else:
            blood_loss = "no_threshold_flag_or_unknown"
        if "blood_transfusion_intraop" in surgery_flags:
            blood_loss += "+transfusion"

        if surgery_flags & {
            "ultra_low_anastomosis",
            "coloanal_anastomosis",
            "anastomosis_at_or_below_5cm",
        }:
            anatomy = "ultra_low_or_at_or_below_5cm"
        elif "lower_rectal_cancer" in patient_flags or "anastomosis_6_10cm" in surgery_flags:
            anatomy = "lower_rectal_or_6_to_10cm"
        elif "rectal_cancer" in patient_flags or "colorectal_anastomosis" in surgery_flags:
            anatomy = "rectal_or_colorectal"
        else:
            anatomy = "colon_or_unknown"
        return {
            "operative_duration": duration,
            "blood_loss_and_transfusion": blood_loss,
            "anatomy": anatomy,
        }

    def evaluate(
        self,
        assessment: RiskAssessment,
        derived: DerivedCaseFeatures,
    ) -> CandidateV14Assessment:
        """Return the posthoc v1.0 candidate score and independent safety state."""
        active = {signal.rule_id for signal in assessment.signals}
        patient_flags = set(derived.patient_flags)
        surgery_flags = set(derived.surgery_flags)
        unresolved = bool(surgery_flags & self.unresolved_immediate_action_flags)
        resolved = bool(surgery_flags & self.resolved_review_only_flags) and not unresolved
        breakdown: dict[str, int] = {}
        review_only: list[str] = []

        for rule_id, weight in self.base_rule_weights.items():
            if rule_id not in active:
                continue
            if rule_id == self.technical_rule_id and resolved:
                review_only.append(rule_id)
                continue
            breakdown[f"rule:{rule_id}"] = weight

        for group in self.derived_groups:
            if active.intersection(group.any_rules):
                breakdown[f"group:{group.group_id}"] = group.weight

        triggered_interactions: list[str] = []
        for interaction in self.interactions:
            if not set(interaction.all_rules).issubset(active):
                continue
            if self.technical_rule_id in interaction.all_rules and resolved:
                continue
            triggered_interactions.append(interaction.interaction_id)
            breakdown[f"interaction:{interaction.interaction_id}"] = interaction.bonus

        score = sum(breakdown.values())
        primary_alert = score >= self.primary_alert_minimum_score
        requires_immediate_action = assessment.requires_immediate_action
        operational_alert = primary_alert or requires_immediate_action
        if unresolved:
            technical_status = "unresolved_immediate_action"
        elif resolved:
            technical_status = "corrected_retest_review_only"
        else:
            technical_status = "not_triggered_or_not_assessed"

        if requires_immediate_action:
            interpretation = "独立安全护栏触发；无论候选累计分数多少均需立即处置。"
        elif primary_alert:
            interpretation = self.PRIMARY_ALERT_INTERPRETATION
        elif assessment.input_completeness < self.minimum_input_completeness_for_low_output:
            interpretation = "未达到候选阈值，但输入不完整，禁止解释为低风险。"
        elif resolved:
            interpretation = "技术异常已纠正并复测；保留复核提示但不重复计入候选分数。"
        else:
            interpretation = "未达到候选预警阈值；不等同于零风险。"
        return CandidateV14Assessment(
            policy_version=self.policy_version,
            base_policy_version=self.base_policy_version,
            risk_score=score,
            primary_alert=primary_alert,
            operational_alert=operational_alert,
            primary_alert_threshold=self.primary_alert_minimum_score,
            requires_immediate_action=requires_immediate_action,
            technical_integrity_status=technical_status,
            review_only_rule_ids=tuple(sorted(review_only)),
            triggered_interactions=tuple(triggered_interactions),
            graded_context=self._graded_context(patient_flags, surgery_flags),
            score_breakdown=dict(sorted(breakdown.items())),
            interpretation=interpretation,
        )
