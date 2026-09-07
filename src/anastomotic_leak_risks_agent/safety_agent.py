"""End-to-end deterministic safety-agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .case_features import derive_case_features
from .evidence_store import EvidenceStore
from .llm_client import LLMClient
from .naive_rag import NaiveRAG
from .normalize import find_entity_a, find_entity_b
from .report import render_report
from .risk_rules import RiskRuleEngine
from .schemas import ClinicalCaseInput, SafetyReport


@dataclass
class SafetyAgentConfig:
    use_lightrag: bool = False
    use_llm_arbiter: bool = False
    tiered_arbiter: bool = True
    top_k_evidence: int = 5
    max_context_chars: int = 6000


class SafetyAgent:
    """Combine rules, evidence retrieval, and deterministic report rendering.

    LLM output may add a grounded explanatory note, but it never changes the
    deterministic rule-derived risk level.
    """

    def __init__(
        self,
        *,
        evidence_store: EvidenceStore | None = None,
        lightrag: Any | None = None,
        llm_client: LLMClient | None = None,
        config: SafetyAgentConfig | None = None,
        rule_engine: RiskRuleEngine | None = None,
    ) -> None:
        self.evidence_store = evidence_store or EvidenceStore()
        self.naive_rag = NaiveRAG(self.evidence_store)
        self.lightrag = lightrag
        self.llm_client = llm_client or LLMClient(provider="stub")
        self.config = config or SafetyAgentConfig()
        self.rule_engine = rule_engine or RiskRuleEngine()

    def evaluate(
        self,
        entity_a: str,
        entity_b: str,
        patient_factors: list[str] | None = None,
        *,
        case_id: str | None = None,
    ) -> SafetyReport:
        """Evaluate legacy registry rows for regression/demo compatibility only."""
        factors = list(patient_factors or [])
        row_a = find_entity_a(entity_a)
        row_b = find_entity_b(entity_b)
        return self._evaluate_flags(
            case_id=case_id or f"{row_a.entity_a_id}__{row_b.entity_b_id}",
            entity_a_label=row_a.primary_name,
            entity_b_label=row_b.primary_name,
            patient_flags=list(dict.fromkeys([*row_a.flags, *factors])),
            surgery_flags=row_b.flags,
            input_completeness=0.0,
            extra_metadata={
                "input_mode": "legacy_registry_demo",
                "entity_a_id": row_a.entity_a_id,
                "entity_b_id": row_b.entity_b_id,
                "warnings": ["特征字典行选择仅供回归测试，不等于患者实际取值。"],
            },
        )

    def evaluate_case(self, case: ClinicalCaseInput) -> SafetyReport:
        """Evaluate a de-identified structured patient and operative profile."""
        derived = derive_case_features(case)
        return self._evaluate_flags(
            case_id=case.case_id,
            entity_a_label="结构化患者画像",
            entity_b_label="结构化手术—吻合口画像",
            patient_flags=derived.patient_flags,
            surgery_flags=derived.surgery_flags,
            input_completeness=derived.input_completeness,
            extra_metadata={
                "input_mode": "structured_case",
                "case_input": case.model_dump(mode="json"),
                "flag_trace": derived.trace,
                "warnings": derived.warnings,
                "assessment_domains": derived.assessment_domains,
                "missing_domains": derived.missing_domains,
            },
        )

    def _evaluate_flags(
        self,
        *,
        case_id: str,
        entity_a_label: str,
        entity_b_label: str,
        patient_flags: list[str],
        surgery_flags: list[str],
        input_completeness: float,
        extra_metadata: dict[str, Any],
    ) -> SafetyReport:
        assessment = self.rule_engine.evaluate(
            patient_flags,
            surgery_flags,
            input_completeness=input_completeness,
        )

        rule_keys = [self.rule_engine.evidence_key(signal.rule_id) for signal in assessment.signals]
        evidence_rows = self.naive_rag.retrieve(
            rule_ids=rule_keys,
            entity_a_flags=patient_flags,
            entity_b_flags=surgery_flags,
            top_k=self.config.top_k_evidence,
        )
        evidence = [item.to_schema() for item in evidence_rows]

        lightrag_context = ""
        if self.config.use_lightrag and self.lightrag is not None and getattr(self.lightrag, "available", True):
            try:
                if hasattr(self.lightrag, "query_case"):
                    result = self.lightrag.query_case(
                        patient_flags=patient_flags,
                        surgery_flags=surgery_flags,
                        triggered_rule_ids=rule_keys,
                        mode="hybrid",
                    )
                else:
                    question = (
                        "结直肠癌吻合口漏证据检索；"
                        f"规则：{', '.join(rule_keys) or '无'}；"
                        f"患者风险标签：{', '.join(patient_flags) or '无'}；"
                        f"手术风险标签：{', '.join(surgery_flags) or '无'}"
                    )
                    result = self.lightrag.query(question, mode="hybrid")
                lightrag_context = str(result)[: self.config.max_context_chars]
            except Exception:
                lightrag_context = ""

        structured_context = "\n".join(f"[{item.evidence_level}] {item.title}: {item.text}" for item in evidence)[
            : self.config.max_context_chars
        ]
        llm_note = ""
        if self.config.use_llm_arbiter and not getattr(self.llm_client, "degraded", True):
            prompt = (
                "请只根据以下规则信号和证据，用中文写一段不超过 250 字的临床复核提示。"
                "不得改变风险等级；证据不足时必须明确说明。\n\n"
                f"风险等级：{assessment.risk_level.value}\n"
                f"规则：{', '.join(signal.rule_id for signal in assessment.signals) or '无'}\n"
                f"证据：\n{structured_context}\n{lightrag_context}"
            )
            response = self.llm_client.complete(
                prompt,
                system="你是医学决策支持助手，必须忠实于给定规则与证据。",
            )
            if not response.degraded:
                llm_note = response.text.strip()

        report_text = render_report(
            case_id=case_id,
            entity_a=entity_a_label,
            entity_b=entity_b_label,
            patient_factors=patient_flags,
            surgery_factors=surgery_flags,
            assessment=assessment,
            evidence=evidence,
            llm_note=llm_note,
        )
        context_char_count = len(structured_context) + len(lightrag_context)
        metadata = {
            "patient_flags": patient_flags,
            "surgery_flags": surgery_flags,
            "n_signals": len(assessment.signals),
            "n_evidence": len(evidence),
            "input_completeness": assessment.input_completeness,
            "coverage_status": assessment.coverage_status,
            "rule_policy_version": assessment.policy_version,
            "primary_alert": assessment.primary_alert,
            "risk_score": assessment.risk_score,
            "primary_alert_threshold": assessment.primary_alert_threshold,
            "safety_guardrail_present": assessment.safety_guardrail_present,
            "requires_immediate_action": assessment.requires_immediate_action,
            "llm_role": "explanation_only" if llm_note else "disabled",
            "llm_label": getattr(self.llm_client, "model", "stub"),
            "lightrag_used": bool(lightrag_context),
            "synthetic_or_clinical": "unspecified",
            **extra_metadata,
        }
        return SafetyReport(
            case_id=case_id,
            entity_a=entity_a_label,
            entity_b=entity_b_label,
            patient_factors=patient_flags,
            risk_assessment=assessment,
            evidence=evidence,
            report_text=report_text,
            context_char_count=context_char_count,
            metadata=metadata,
        )
