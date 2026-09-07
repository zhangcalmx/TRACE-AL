"""Exploratory dual-operating-point policy layered on signed rule policy v1.1.

This module does not change ``RiskRuleEngine`` or ``configs/rules.yaml``.  It
separates a sensitive screening result from a more specific high-confidence
alert and routes the interval between them to review.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .paths import project_root
from .schemas import RiskAssessment

PROJECT_ROOT = project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "rule_policy_v1_2_candidate.yaml"
DEFAULT_BASE_RULES_PATH = PROJECT_ROOT / "configs" / "rules.yaml"


class CandidateDisposition(str, Enum):
    """Operational disposition; it is not a predicted probability."""

    IMMEDIATE_ACTION = "immediate_action"
    HIGH_CONFIDENCE_ALERT = "high_confidence_alert"
    REVIEW_POSITIVE = "review_positive"
    SCREEN_NEGATIVE = "screen_negative"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class CandidatePolicyAssessment:
    policy_version: str
    base_policy_version: str
    disposition: CandidateDisposition
    risk_score: int
    sensitivity_screen_positive: bool
    high_confidence_alert: bool
    review_required: bool
    immediate_action: bool
    input_completeness: float
    interpretation: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CandidateRulePolicy:
    """Apply the exploratory v1.2 workflow without mutating signed v1.1."""

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

        if self.policy_version != "1.2-candidate":
            raise ValueError("Candidate policy_version must be 1.2-candidate")
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

        points = payload.get("operating_points")
        if not isinstance(points, dict):
            raise TypeError("operating_points must be a mapping")
        self.sensitivity_screen_minimum_score = self._positive_int(
            points.get("sensitivity_screen_minimum_score"),
            "sensitivity_screen_minimum_score",
        )
        self.high_confidence_alert_minimum_score = self._positive_int(
            points.get("high_confidence_alert_minimum_score"),
            "high_confidence_alert_minimum_score",
        )
        if self.sensitivity_screen_minimum_score >= self.high_confidence_alert_minimum_score:
            raise ValueError("Sensitivity screen threshold must be below high-confidence threshold")
        completeness = points.get("minimum_input_completeness_for_screen_negative")
        if not isinstance(completeness, (int, float)) or not 0 <= float(completeness) <= 1:
            raise ValueError("minimum input completeness must be between 0 and 1")
        self.minimum_input_completeness_for_screen_negative = float(completeness)

        safety = payload.get("safety")
        if not isinstance(safety, dict):
            raise TypeError("safety must be a mapping")
        if safety.get("immediate_action_overrides_score") is not True:
            raise ValueError("Immediate-action safety override cannot be disabled")
        if safety.get("non_actionable_guardrail_requires_review") is not True:
            raise ValueError("Non-actionable safety guardrails must require review")

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def evaluate(self, assessment: RiskAssessment) -> CandidatePolicyAssessment:
        """Return a dual-threshold disposition from a signed v1.1 assessment."""
        if assessment.policy_version != self.base_policy_version:
            raise ValueError(
                f"Expected base policy {self.base_policy_version}, got {assessment.policy_version}"
            )

        score = assessment.risk_score
        statistical_screen_positive = score >= self.sensitivity_screen_minimum_score
        statistical_high_confidence = score >= self.high_confidence_alert_minimum_score

        if assessment.requires_immediate_action:
            disposition = CandidateDisposition.IMMEDIATE_ACTION
            interpretation = "独立安全护栏触发：不受累计分数限制，需立即处理。"
        elif statistical_high_confidence:
            disposition = CandidateDisposition.HIGH_CONFIDENCE_ALERT
            interpretation = "达到高特异度工作点：高置信预警。"
        elif statistical_screen_positive or assessment.safety_guardrail_present:
            disposition = CandidateDisposition.REVIEW_POSITIVE
            interpretation = "敏感筛查阳性但未达到高置信阈值：进入临床复核。"
        elif assessment.input_completeness < self.minimum_input_completeness_for_screen_negative:
            disposition = CandidateDisposition.INSUFFICIENT_DATA
            interpretation = "输入完整度不足：禁止自动判为筛查阴性。"
        else:
            disposition = CandidateDisposition.SCREEN_NEGATIVE
            interpretation = "未达到敏感筛查阈值；仍不等同于零风险。"

        review_required = disposition in {
            CandidateDisposition.REVIEW_POSITIVE,
            CandidateDisposition.INSUFFICIENT_DATA,
        }
        return CandidatePolicyAssessment(
            policy_version=self.policy_version,
            base_policy_version=self.base_policy_version,
            disposition=disposition,
            risk_score=score,
            sensitivity_screen_positive=statistical_screen_positive,
            high_confidence_alert=statistical_high_confidence,
            review_required=review_required,
            immediate_action=assessment.requires_immediate_action,
            input_completeness=assessment.input_completeness,
            interpretation=interpretation,
        )
