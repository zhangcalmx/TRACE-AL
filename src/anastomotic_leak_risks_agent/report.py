"""Deterministic, traceable report rendering."""

from __future__ import annotations

from .schemas import EvidenceChunk, RiskAssessment


def render_report(
    *,
    case_id: str,
    entity_a: str,
    entity_b: str,
    patient_factors: list[str],
    surgery_factors: list[str],
    assessment: RiskAssessment,
    evidence: list[EvidenceChunk],
    llm_note: str = "",
) -> str:
    level_labels = {
        "high": "高优先级规则输出",
        "medium": "需要临床复核",
        "low": "未触发已知高/中风险规则",
        "unknown": "信息不足，风险未知",
    }
    lines = [
        "# 吻合口漏风险决策支持报告",
        "",
        f"- 病例标识：`{case_id}`",
        f"- 患者输入：`{entity_a}`",
        f"- 手术输入：`{entity_b}`",
        f"- 派生的患者标签：{', '.join(patient_factors) if patient_factors else '无'}",
        f"- 派生的手术标签：{', '.join(surgery_factors) if surgery_factors else '无'}",
        f"- 规则优先级：**{level_labels[assessment.risk_level.value]}**（不是个体漏概率）",
        (
            f"- 累积风险主警报：**{'是' if assessment.primary_alert else '否'}**"
            f"（加权分 {assessment.risk_score}/{assessment.primary_alert_threshold}）"
        ),
        f"- 未解决术中安全警报：**{'是' if assessment.requires_immediate_action else '否'}**",
        (
            f"- 输入完整度：**{assessment.input_completeness:.0%}**"
            f"（{assessment.coverage_status}；非风险概率、非模型置信度）"
        ),
        "",
        "## 触发规则",
    ]
    if not assessment.signals:
        if assessment.risk_level.value == "unknown":
            lines.append("输入信息不足，无法安全地给出低风险判定。请补全关键变量后重新评估。")
        else:
            lines.append("在当前规则范围和已评估变量中未触发高/中风险规则；这不等于排除吻合口漏。")
    for index, signal in enumerate(assessment.signals, start=1):
        lines.extend(
            [
                (f"### {index}. {signal.rule_id} [{signal.risk_level.value.upper()} · {signal.signal_class}]"),
                signal.rationale,
                f"建议：{signal.recommendation}",
            ]
        )

    lines.extend(["", "## 证据链"])
    if not evidence:
        lines.append("未检索到可结构化引用的证据片段，需人工复核。")
    for index, item in enumerate(evidence, start=1):
        source = item.source or item.source_type
        locator = item.source_locator or item.pmid or item.doi or item.url
        suffix = f" | {locator}" if locator else " | 缺少可核验定位符"
        lines.append(f"{index}. **[{item.evidence_level or '?'}] {item.title}** — {source}{suffix}\n   {item.text}")

    if llm_note:
        lines.extend(["", "## LLM 辅助说明（不改变规则分级）", llm_note])

    lines.extend(
        [
            "",
            "## 使用限制",
            "本系统是未完成临床验证的研究原型。报告不是个体风险概率，不能替代外科医师判断，不得用于自动诊断或自动决定造口/再手术。",
        ]
    )
    return "\n".join(lines)
