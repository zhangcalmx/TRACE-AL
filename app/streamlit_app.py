"""结直肠癌根治术吻合口漏研究原型的 Streamlit 界面和共享 UI 函数。

The Chat and Screening pages read runtime state from project files
(.executor/progress.json, configs/rules.yaml, data/processed/lightrag_current.json),
so this template is project-agnostic — no per-project hardcoding.

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The entrypoint script always lives in the project checkout, even when the
# package itself is installed into site-packages (Streamlit Cloud). Publish
# the true root so package modules resolve data/configs from the checkout.
# The literal must match ROOT_ENV_VAR in anastomotic_leak_risks_agent/paths.py;
# it is set before any package import because modules resolve roots eagerly.
os.environ.setdefault("ANASTOMOTIC_LEAK_RISKS_ROOT", str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass


def _bridge_streamlit_cloud_secrets() -> None:
    """Mirror Streamlit Community Cloud dashboard secrets into os.environ.

    On Streamlit Cloud, secrets entered in the app dashboard are exposed via
    ``st.secrets`` only, while the LLM clients and the access gate read
    ``os.environ``. Existing variables (e.g. from a local .env) always win.
    """
    try:
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


_bridge_streamlit_cloud_secrets()

from anastomotic_leak_risks_agent.chat_router import (
    ChatIntent,
    build_conversation_context,
    build_retrieval_query,
    classify_chat_intent,
    direct_response_for,
    no_evidence_response,
)
from anastomotic_leak_risks_agent.evidence_store import EvidenceStore
from anastomotic_leak_risks_agent.lightrag_adapter import LightRAGAdapter
from anastomotic_leak_risks_agent.llm_client import DualLLMClient, LLMClient, LLMConfig
from anastomotic_leak_risks_agent.naive_rag import NaiveRAG
from anastomotic_leak_risks_agent.normalize import (
    clear_cache,
    load_entity_a_registry,
    load_entity_b_registry,
)
from anastomotic_leak_risks_agent.safety_agent import SafetyAgent, SafetyAgentConfig
from anastomotic_leak_risks_agent.schemas import (
    ClinicalCaseInput,
    PatientProfileInput,
    SurgeryProfileInput,
)

LANGUAGE_ZH = "中文"
LANGUAGE_EN = "English"
LANGUAGE_STATE_KEY = "ui_language"
LANGUAGE_WIDGET_KEY = "ui_language_selector"


def current_language() -> str:
    """Return the per-session UI language, defaulting safely to Chinese."""
    return st.session_state.get(LANGUAGE_STATE_KEY, LANGUAGE_ZH)


def tr(chinese: str, english: str) -> str:
    """Select one UI string without translating clinical source content."""
    return english if current_language() == LANGUAGE_EN else chinese


def _sync_language_from_widget() -> None:
    """Copy widget-owned state into ordinary state shared by all page scripts."""
    st.session_state[LANGUAGE_STATE_KEY] = st.session_state.get(
        LANGUAGE_WIDGET_KEY,
        LANGUAGE_ZH,
    )


def render_language_control() -> None:
    """Render one global language switch shared by every page."""
    language = st.session_state.setdefault(LANGUAGE_STATE_KEY, LANGUAGE_ZH)
    st.session_state.setdefault(LANGUAGE_WIDGET_KEY, language)
    with st.sidebar:
        selected_language = st.segmented_control(
            tr("界面语言 / Language", "Language"),
            [LANGUAGE_ZH, LANGUAGE_EN],
            key=LANGUAGE_WIDGET_KEY,
            required=True,
            width="stretch",
            persist_state="session",
            on_change=_sync_language_from_widget,
        )
    # This also covers programmatic state changes in tests and restored sessions.
    if selected_language in {LANGUAGE_ZH, LANGUAGE_EN}:
        st.session_state[LANGUAGE_STATE_KEY] = selected_language


def localized_freshness_detail(status: str, chinese_detail: str) -> str:
    if current_language() != LANGUAGE_EN:
        return chinese_detail
    return {
        "fresh": "The index matches the current rules and evidence corpus.",
        "stale_rules_changed": "The rule set has changed. Re-run phases 3 and 4.",
        "stale_evidence_changed": "Evidence text has changed. Rebuild the LightRAG index.",
        "metadata_changed": "Evidence metadata changed. Refresh the EvidenceStore sidecar.",
        "stale_not_built": "No strictly accepted LightRAG index is available.",
        "unknown": "The index status could not be determined.",
    }.get(status, "The index status could not be determined.")


DOMAIN_LABELS = {
    "tumor_site": ("肿瘤部位", "tumor site"),
    "metabolic": ("代谢状态", "metabolic status"),
    "nutrition": ("营养状态", "nutrition"),
    "performance": ("全身与体能状态", "performance status"),
    "inflammation_nutrition": ("炎症与营养指数", "inflammation and nutrition indices"),
    "anastomosis": ("吻合口信息", "anastomosis details"),
    "technical_integrity": ("技术完整性", "technical integrity"),
    "hemodynamics": ("血流动力学", "hemodynamics"),
    "operative_burden": ("手术负荷", "operative burden"),
    "icg_perfusion": ("ICG 灌注", "ICG perfusion"),
}

WARNING_TRANSLATIONS = {
    "已录入 HbA1c，但糖尿病状态不是1型/2型；未自动生成糖尿病控制不佳标签。": (
        "HbA1c was entered, but diabetes status is not type 1 or type 2; "
        "the poor glycemic control flag was not derived."
    ),
    "ICG 灌注不足，但后续处置未知；不自动触发‘未矫正灌注不良’规则。": (
        "ICG perfusion was inadequate, but the corrective action is unknown; "
        "the uncorrected poor-perfusion rule was not triggered."
    ),
    "已录入 ICG 灌注结果，但 ICG 使用状态不是“是”；未生成 ICG 风险标签。": (
        "An ICG perfusion result was entered, but ICG use was not confirmed; no ICG risk flag was derived."
    ),
}

def localized_warning(message: str) -> str:
    if current_language() == LANGUAGE_EN:
        return WARNING_TRANSLATIONS.get(message, f"Source-language warning: {message}")
    return message


RISK_PRESENTATION = {
    "high": {
        "label": ("高优先级规则", "High-priority rule output"),
        "color": "red",
        "icon": ":material/emergency_home:",
        "recommendation": (
            "存在高优先级规则；请结合下方累积风险主警报与术中安全警报分别处置。",
            "A high-priority rule is present; interpret cumulative-risk and intraoperative-safety alerts separately below.",
        ),
    },
    "medium": {
        "label": ("复核级规则", "Review-level rule output"),
        "color": "orange",
        "icon": ":material/warning:",
        "recommendation": (
            "这是复核提示，不单独构成主警报；建议结合其他因素评估。",
            "This is a review prompt, not a primary alert; interpret it with other factors.",
        ),
    },
    "low": {
        "label": ("未触发高/复核级规则", "No high/review-level rule triggered"),
        "color": "green",
        "icon": ":material/check_circle:",
        "recommendation": (
            "当前规则库未提示显著风险，但不能排除未建模因素。",
            "No major rule-based signal was found, but unmodelled risks remain possible.",
        ),
    },
    "unknown": {
        "label": ("暂无法判断", "Unable to determine"),
        "color": "gray",
        "icon": ":material/help:",
        "recommendation": (
            "信息或证据不足，建议补充数据并由临床团队复核。",
            "Information or evidence is insufficient. Add data and obtain clinical review.",
        ),
    },
}

EVIDENCE_LEVEL_LABELS = {
    "A": ("A级证据", "Level A evidence", "blue"),
    "B": ("B级证据", "Level B evidence", "violet"),
    "C": ("C级证据", "Level C evidence", "primary"),
    "D": ("D级证据", "Level D evidence", "orange"),
    "E": ("E级证据", "Level E evidence", "gray"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
def set_page_config() -> None:
    st.set_page_config(
        page_title=tr("吻合口漏风险决策支持", "Anastomotic leak decision support"),
        page_icon=":material/health_and_safety:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def require_access_code() -> None:
    """Optional shared access code for hosted demos.

    Set the APP_ACCESS_CODE environment variable to gate every entry point;
    when it is unset (local development) the gate is disabled entirely.
    """
    expected = os.environ.get("APP_ACCESS_CODE", "").strip()
    if not expected or st.session_state.get("access_code_ok"):
        return
    st.title(tr("吻合口漏风险决策支持", "Anastomotic leak decision support"))
    code = st.text_input(
        tr("请输入访问码", "Enter the access code"),
        type="password",
        help=tr("本页面仅向受邀人员开放。", "Access is restricted to invited users."),
    )
    if not code:
        st.stop()
    if code != expected:
        st.error(tr("访问码不正确。", "Incorrect access code."), icon=":material/lock:")
        st.stop()
    st.session_state["access_code_ok"] = True


# ─────────────────────────────────────────────────────────────────────────────
# Project file readers (Overview tab) — all defensive, never crash
# ─────────────────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict | None:
    try:
        if path.exists():
            # PowerShell and some Windows editors may persist project status
            # files with a UTF-8 BOM.  ``utf-8-sig`` accepts both BOM and
            # regular UTF-8, preventing a healthy index from being shown as
            # stale merely because progress.json could not be decoded.
            return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return None


def read_progress() -> dict:
    return (_load_json(PROJECT_ROOT / ".executor" / "progress.json") or {}).get("phases", {})


def read_rules_yaml_sha() -> str:
    rules = PROJECT_ROOT / "configs" / "rules.yaml"
    if not rules.exists():
        return ""
    return hashlib.sha256(rules.read_bytes()).hexdigest()


def _resolve_pointer_path(raw: str) -> Path | None:
    """Resolve a path stored in the LightRAG pointer against the project root.

    Current pointer files store project-relative paths; legacy files hold
    absolute paths, which may point at a previous copy of the project after
    the folder was moved or synced to another drive.
    """
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.exists():
        return candidate
    parts = Path(raw).parts
    for i in range(len(parts) - 1):
        if parts[i] == "data" and parts[i + 1] == "processed":
            rerooted = PROJECT_ROOT.joinpath(*parts[i:])
            if rerooted.exists():
                return rerooted
    return None


def accepted_index_path() -> Path | None:
    """Return only an index that passed probes and was atomically promoted."""
    pointer = _load_json(PROJECT_ROOT / "data" / "processed" / "lightrag_current.json")
    if not pointer:
        return None
    candidate = _resolve_pointer_path(str(pointer.get("working_dir", "")))
    manifest_path = _resolve_pointer_path(str(pointer.get("acceptance_manifest", "")))
    manifest = _load_json(manifest_path) if manifest_path else None
    if not candidate or not candidate.is_dir() or not manifest or manifest.get("status") != "accepted":
        return None
    return candidate


def _compute_index_freshness() -> tuple[str, str]:
    """Return (status_label, detail).

    status_label ∈ {"fresh", "stale_rules_changed", "stale_evidence_changed",
    "metadata_changed", "stale_not_built", "unknown"}.
    Compares: phase3.rules_yaml_sha256 vs phase4.index_built_against_rules_sha256
    vs current configs/rules.yaml sha.
    """
    progress = read_progress()
    p3 = progress.get("phase3", {}) or {}
    p4 = progress.get("phase4", {}) or {}
    verified_sha = p3.get("rules_yaml_sha256")
    index_sha = p4.get("index_built_against_rules_sha256")
    indexed_content_sha = p4.get("index_built_against_evidence_content_sha256")
    indexed_metadata_sha = p4.get("evidence_metadata_sha256")
    current_sha = read_rules_yaml_sha()
    try:
        store = EvidenceStore()
        evidence_hashes = store.corpus_fingerprints()
        evidence_count = store.count()
    except Exception:
        evidence_hashes = {}
        evidence_count = -1
    pointer = _load_json(PROJECT_ROOT / "data" / "processed" / "lightrag_current.json") or {}
    if accepted_index_path() is None:
        return ("stale_not_built", "尚无通过严格门禁和三项探针验收的 LightRAG 索引")
    index_sha = pointer.get("rules_yaml_sha256") or index_sha
    pointer_fingerprints = pointer.get("evidence_fingerprints", {}) or {}
    indexed_content_sha = pointer_fingerprints.get("content_sha256") or indexed_content_sha
    indexed_metadata_sha = pointer_fingerprints.get("metadata_sha256") or indexed_metadata_sha
    if not verified_sha:
        return ("stale_rules_changed", "当前规则尚未通过严格引用核验")
    if verified_sha and current_sha and verified_sha != current_sha:
        return ("stale_rules_changed", "rules.yaml 自上次核验后已改动，须重跑 phase3 + phase4")
    if index_sha != verified_sha:
        return ("stale_rules_changed", "索引构建于不同版本的 rules.yaml，须重建")
    current_content_sha = evidence_hashes.get("content_sha256")
    if indexed_content_sha and current_content_sha and indexed_content_sha != current_content_sha:
        detail = (
            "证据正文已变化，须重建 LightRAG 图索引"
            f"（索引记录 {indexed_content_sha[:8]}…，当前计算 {current_content_sha[:8]}…，"
            f"本地读取证据 {evidence_count} 条）"
        )
        return ("stale_evidence_changed", detail)
    current_metadata_sha = evidence_hashes.get("metadata_sha256")
    if indexed_metadata_sha and current_metadata_sha and indexed_metadata_sha != current_metadata_sha:
        return ("metadata_changed", "证据元数据已变化；须刷新 EvidenceStore 旁路，无需重建图索引")
    return ("fresh", "索引与当前规则及证据正文一致；证据元数据旁路也已同步")


@st.cache_data(ttl=15, show_spinner=False)
def index_freshness() -> tuple[str, str]:
    """Freshness status, recomputed at most every 15 s instead of on every rerun."""
    return _compute_index_freshness()


# ─────────────────────────────────────────────────────────────────────────────
# Components
# ─────────────────────────────────────────────────────────────────────────────
def render_risk_hero(
    risk_level: str,
    confidence: float,
    n_signals: int,
    n_evidence: int,
    elapsed: float,
    llm_label: str,
    *,
    primary_alert: bool,
    risk_score: int,
    primary_alert_threshold: int,
    safety_guardrail_present: bool,
    requires_immediate_action: bool,
) -> None:
    presentation = RISK_PRESENTATION.get(risk_level, RISK_PRESENTATION["unknown"])
    risk_label = tr(*presentation["label"])
    recommendation = tr(*presentation["recommendation"])
    with st.container(border=True, gap="small"):
        st.header(risk_label)
        st.write(recommendation)
        with st.container(horizontal=True):
            st.badge(
                tr(
                    (
                        f"累积风险主警报：是（{risk_score}/{primary_alert_threshold}）"
                        if primary_alert
                        else f"累积风险主警报：否（{risk_score}/{primary_alert_threshold}）"
                    ),
                    (
                        f"Cumulative-risk alert: yes ({risk_score}/{primary_alert_threshold})"
                        if primary_alert
                        else f"Cumulative-risk alert: no ({risk_score}/{primary_alert_threshold})"
                    ),
                ),
                color="red" if primary_alert else "gray",
                icon=":material/notifications_active:" if primary_alert else ":material/notifications_off:",
            )
            st.badge(
                tr(
                    (
                        "未解决术中安全警报：是"
                        if requires_immediate_action
                        else "术中安全护栏：已触发复核"
                        if safety_guardrail_present
                        else "术中安全护栏：未触发"
                    ),
                    (
                        "Unresolved intraoperative safety alert: yes"
                        if requires_immediate_action
                        else "Intraoperative safety guardrail: review"
                        if safety_guardrail_present
                        else "Intraoperative safety guardrail: not triggered"
                    ),
                ),
                color="red" if requires_immediate_action else "orange" if safety_guardrail_present else "gray",
                icon=":material/emergency_home:" if requires_immediate_action else ":material/health_and_safety:",
            )
        cols = st.columns(4)
        cols[0].metric(
            tr("输入完整度", "Input completeness"),
            f"{confidence:.0%}",
            border=True,
            help=tr(
                "这是字段覆盖程度，不是吻合口漏发生概率。",
                "This is field coverage, not the probability of an anastomotic leak.",
            ),
        )
        cols[1].metric(tr("触发规则", "Triggered rules"), n_signals, border=True)
        cols[2].metric(tr("证据片段", "Evidence excerpts"), n_evidence, border=True)
        cols[3].metric(tr("处理耗时", "Processing time"), f"{elapsed:.1f} s", border=True)
        st.caption(
            tr(
                f"解释模型：{llm_label} · 显示的是规则优先级，不是校准后的吻合口漏概率",
                f"Explanation model: {llm_label} · This is rule priority, not a calibrated leak probability",
            )
        )


def render_evidence_card(level: str, title: str, source: str, text: str) -> None:
    label_zh, label_en, badge_color = EVIDENCE_LEVEL_LABELS.get(level, ("未分级证据", "Unclassified evidence", "gray"))
    display_text = text[:400] + ("..." if len(text) > 400 else "")
    with st.container(border=True, gap="xsmall"):
        st.badge(tr(label_zh, label_en), color=badge_color, icon=":material/library_books:")
        st.markdown(f"**{title}**")
        st.caption(source or tr("来源信息未提供", "Source information unavailable"))
        st.write(display_text)
        if len(text) > 400:
            with st.expander(tr("查看全文", "View full text"), icon=":material/article:"):
                st.write(text)


def render_rule_chip(rule_id: str, severity: str, mechanism: str) -> None:
    presentation = RISK_PRESENTATION.get(severity, RISK_PRESENTATION["unknown"])
    suffix = f" · {mechanism}" if mechanism else ""
    st.badge(
        f"{tr(*presentation['label'])} · {rule_id}{suffix}",
        color=presentation["color"],
        icon=presentation["icon"],
    )


def render_status_pill(status: str) -> None:
    presentation = {
        "fresh": ("索引可用", "Index available", "green", ":material/check_circle:"),
        "stale_rules_changed": ("规则已变化", "Rules changed", "orange", ":material/sync_problem:"),
        "stale_evidence_changed": ("证据已变化", "Evidence changed", "red", ":material/error:"),
        "metadata_changed": ("元数据待刷新", "Metadata refresh needed", "orange", ":material/update:"),
        "stale_not_built": ("索引未构建", "Index not built", "red", ":material/database_off:"),
        "unknown": ("状态未知", "Status unknown", "gray", ":material/help:"),
    }.get(status, (status, status, "gray", ":material/help:"))
    st.badge(tr(presentation[0], presentation[1]), color=presentation[2], icon=presentation[3])


def render_localized_report_summary(report) -> None:
    """Render an English-safe deterministic summary or the source Chinese report."""
    if current_language() != LANGUAGE_EN:
        st.markdown(report.report_text)
        return

    risk_level = report.risk_assessment.risk_level.value
    risk_label = tr(*RISK_PRESENTATION.get(risk_level, RISK_PRESENTATION["unknown"])["label"])
    signals = report.risk_assessment.signals
    st.markdown(
        f"**Final rule-based priority:** {risk_label}  \n"
        f"**Primary alert:** {'Yes' if report.risk_assessment.primary_alert else 'No'}  \n"
        f"**Unresolved safety alert:** {'Yes' if report.risk_assessment.requires_immediate_action else 'No'}  \n"
        f"**Input completeness:** {report.risk_assessment.input_completeness:.0%}  \n"
        f"**Triggered rules:** {len(signals)}  \n"
        f"**Retrieved evidence excerpts:** {len(report.evidence)}"
    )
    if signals:
        st.markdown("**Triggered rule IDs:** " + ", ".join(f"`{signal.rule_id}`" for signal in signals))
    st.caption(
        "This deterministic English summary does not translate or reinterpret clinical source text. "
        "Open Rule details and Evidence chain to inspect the original, auditable content."
    )


def render_footer(report_metadata: dict) -> None:
    st.caption(
        tr(
            "规则引擎 + LightRAG + LLM 辅助解释 · "
            f"病例 {report_metadata.get('case_id', '-')} · "
            f"模型 {report_metadata.get('llm_label', '-')} · "
            f"上下文 {report_metadata.get('context_chars', 0)} 字符",
            "Rule engine + LightRAG + LLM-assisted explanation · "
            f"Case {report_metadata.get('case_id', '-')} · "
            f"Model {report_metadata.get('llm_label', '-')} · "
            f"Context {report_metadata.get('context_chars', 0)} characters",
        )
    )
    st.caption(
        tr(
            "本输出仅供临床决策支持；最终判断由临床医生作出，不得用于自动诊断。",
            "For clinical decision support only. Final judgment remains with the clinical team; do not use for automated diagnosis.",
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cached agent (Screening tab)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="加载系统资源 / Loading system resources…")
def get_agent(use_real_llm: bool, use_lightrag: bool, top_k: int) -> SafetyAgent:
    clear_cache()
    llm = DualLLMClient.from_env() if use_real_llm else LLMClient(LLMConfig(provider="stub"))

    adapter = None
    if use_lightrag and os.environ.get("UI_DISABLE_LIGHTRAG"):
        use_lightrag = False
    if use_lightrag:
        index_path = accepted_index_path()
        freshness, detail = index_freshness()
        if index_path is None or freshness != "fresh":
            st.warning(
                tr(
                    f"LightRAG 未启用：{detail}。当前仅使用本地 EvidenceStore。",
                    "LightRAG is disabled: "
                    f"{localized_freshness_detail(freshness, detail)} "
                    "The local EvidenceStore is being used.",
                )
            )
        else:
            try:
                adapter = LightRAGAdapter(working_dir=str(index_path))
            except Exception as e:
                st.warning(
                    tr(
                        f"LightRAG 初始化失败：{e}；仅使用 EvidenceStore。",
                        f"LightRAG initialization failed: {e}. Using EvidenceStore only.",
                    )
                )

    return SafetyAgent(
        evidence_store=EvidenceStore(),
        lightrag=adapter,
        llm_client=llm,
        config=SafetyAgentConfig(
            use_lightrag=(adapter is not None and adapter.available),
            use_llm_arbiter=use_real_llm,
            tiered_arbiter=True,
            top_k_evidence=top_k,
        ),
    )


@st.cache_resource(show_spinner="加载 LightRAG 索引 / Loading LightRAG index…")
def get_lightrag_adapter():
    """Cached LightRAG adapter for the Chat tab (RAG-augmented Q&A)."""
    if os.environ.get("UI_DISABLE_LIGHTRAG"):
        return None
    try:
        index_dir = accepted_index_path()
        if index_dir is None or index_freshness()[0] != "fresh":
            return None
        return LightRAGAdapter(working_dir=str(index_dir))
    except Exception as e:
        st.warning(tr(f"LightRAG 初始化失败：{e}", f"LightRAG initialization failed: {e}"))
        return None


@st.cache_resource(show_spinner="加载解释模型 / Loading explanation model…")
def get_chat_llm():
    """DeepSeek chat client for the Chat tab. Reads key from .env via llm_client."""
    try:
        return LLMClient(provider="deepseek", temperature=0.2, max_tokens=1024)
    except Exception as e:
        st.warning(tr(f"DeepSeek 客户端初始化失败：{e}", f"DeepSeek client initialization failed: {e}"))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — RAG 问答 / Chat
# ─────────────────────────────────────────────────────────────────────────────
def chat_tab() -> None:
    language_code = "en" if current_language() == LANGUAGE_EN else "zh"
    st.header(tr("证据问答", "Evidence Q&A"))
    st.caption(
        tr(
            "适合查询风险因素、规则依据和证据来源。普通寒暄会直接响应；医学结论必须有项目证据支持。",
            "Ask about risk factors, rule rationales, and evidence sources. "
            "Medical conclusions require support from the project evidence base.",
        )
    )

    fresh_status, _ = index_freshness()
    index_available = accepted_index_path() is not None and fresh_status == "fresh"
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        render_status_pill(fresh_status)
        if index_available:
            st.badge(tr("图检索优先", "Graph retrieval first"), color="blue", icon=":material/account_tree:")
        else:
            st.badge(tr("本地检索回退", "Local retrieval fallback"), color="orange", icon=":material/find_in_page:")
        st.caption(tr("无证据时停止生成医学结论", "No evidence means no generated medical conclusion"))

    if not index_available:
        st.warning(
            tr(
                "LightRAG 索引不可用，当前使用本地 EvidenceStore 关键词检索。",
                "The LightRAG index is unavailable. Local EvidenceStore keyword retrieval is active.",
            ),
            icon=":material/info:",
        )

    if fresh_status in {"stale_rules_changed", "stale_evidence_changed"}:
        st.warning(
            tr(
                "LightRAG 索引已过期，回答可能基于旧规则或旧证据正文。建议重跑 phase4。",
                "The LightRAG index is stale and may reflect older rules or evidence text. Re-run phase 4.",
            ),
            icon=":material/warning:",
        )
    elif fresh_status == "metadata_changed":
        st.info(
            tr(
                "证据元数据已变化，但正文索引仍有效；EvidenceStore 正在使用最新元数据。",
                "Evidence metadata changed, but the text index remains valid. EvidenceStore is using current metadata.",
            ),
            icon=":material/update:",
        )

    with st.container(horizontal=True, vertical_alignment="center"):
        show_evidence = st.toggle(
            tr("显示检索证据原文", "Show retrieved source text"),
            value=True,
            key="chat_show_ev",
            persist_state="session",
        )
        st.space("stretch")
        if st.button(tr("清空对话", "Clear conversation"), icon=":material/delete_sweep:"):
            st.session_state["confirm_clear_chat"] = True

    if st.session_state.get("confirm_clear_chat"):
        with st.container(border=True):
            st.warning(
                tr(
                    "将删除本次会话中的全部问答记录，此操作不可撤销。",
                    "This will delete all messages from this session and cannot be undone.",
                ),
                icon=":material/warning:",
            )
            confirm_cols = st.columns(2)
            with confirm_cols[0]:
                if st.button(
                    tr("确认清空", "Confirm clear"),
                    type="primary",
                    key="confirm_clear_chat_yes",
                    width="stretch",
                ):
                    st.session_state["chat_messages"] = []
                    st.session_state["confirm_clear_chat"] = False
                    st.rerun()
            with confirm_cols[1]:
                if st.button(tr("取消", "Cancel"), key="confirm_clear_chat_no", width="stretch"):
                    st.session_state["confirm_clear_chat"] = False
                    st.rerun()

    adapter = get_lightrag_adapter() if index_available else None
    llm = get_chat_llm()
    if adapter is None:
        st.info(
            tr(
                "当前问答使用可审计的本地证据检索，不调用图索引。",
                "This conversation is using auditable local evidence retrieval without the graph index.",
            ),
            icon=":material/search:",
        )

    if llm is None or getattr(llm, "degraded", False):
        st.warning(
            tr(
                "解释模型不可用，将返回抽取式证据摘要，不生成模型推断。",
                "The explanation model is unavailable. Only extractive evidence summaries will be returned.",
            ),
            icon=":material/cloud_off:",
        )

    # init history
    st.session_state.setdefault("chat_messages", [])

    # render history
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and show_evidence and msg.get("evidence"):
                with st.expander(
                    tr(
                        f"检索证据（{len(msg['evidence'])} 段）",
                        f"Retrieved evidence ({len(msg['evidence'])} excerpts)",
                    ),
                    icon=":material/library_books:",
                ):
                    for i, ev in enumerate(msg["evidence"], 1):
                        st.caption(tr(f"片段 {i}", f"Excerpt {i}"))
                        st.write(ev)

    st.space("small")
    st.caption(
        tr(
            "本输出仅供临床决策支持；最终判断由临床医生作出，不得用于自动诊断。",
            "For clinical decision support only. Final judgment remains with the clinical team; do not use for automated diagnosis.",
        )
    )

    # input
    question = st.chat_input(
        tr(
            "输入问题，例如：低位直肠吻合为什么增加吻合口漏风险？",
            "Ask a question, for example: Why does a low rectal anastomosis increase leak risk?",
        ),
        submit_mode="disable",
    )
    if not question:
        return

    prior_messages = list(st.session_state["chat_messages"])
    intent = classify_chat_intent(question, prior_messages)
    st.session_state["chat_messages"].append({"role": "user", "content": question, "intent": intent.value})
    with st.chat_message("user"):
        st.write(question)

    evidence_chunks: list[str] = []
    answer = ""

    # Conversational and navigation intents do not belong in the medical retriever.
    with st.chat_message("assistant"):
        if intent is not ChatIntent.MEDICAL_QUERY:
            answer = direct_response_for(intent, language=language_code)
        else:
            retrieval_query = build_retrieval_query(
                question,
                prior_messages,
                language=language_code,
            )
            retrieved_text = ""

            # 1) Retrieve evidence only for medical knowledge questions.
            with st.spinner(tr("检索项目证据库…", "Searching the project evidence base…")):
                try:
                    if adapter is not None:
                        retrieved_text = adapter.query(retrieval_query, mode="hybrid") or ""
                        evidence_chunks = [c.strip() for c in retrieved_text.split("\n\n") if c.strip()][:8]
                    else:
                        local_hits = NaiveRAG(EvidenceStore()).query(retrieval_query, top_k=8)
                        evidence_chunks = [
                            f"[{hit.level}] {hit.title}\n{hit.text}\n" + tr("来源：", "Source: ") + hit.source_label
                            for hit in local_hits
                        ]
                        retrieved_text = "\n\n".join(evidence_chunks)
                except Exception as e:
                    st.warning(
                        tr(
                            f"LightRAG 检索失败，已回退本地证据库：{e}",
                            f"LightRAG retrieval failed; local evidence fallback is active: {e}",
                        )
                    )
                    local_hits = NaiveRAG(EvidenceStore()).query(retrieval_query, top_k=8)
                    evidence_chunks = [
                        f"[{hit.level}] {hit.title}\n{hit.text}\n" + tr("来源：", "Source: ") + hit.source_label
                        for hit in local_hits
                    ]
                    retrieved_text = "\n\n".join(evidence_chunks)

            if not retrieved_text.strip():
                answer = no_evidence_response(language=language_code)
            elif llm is None or getattr(llm, "degraded", False):
                bullets = [
                    f"- [{tr('证据', 'Evidence')} {i}] {chunk.splitlines()[0]}"
                    for i, chunk in enumerate(evidence_chunks[:5], 1)
                ]
                answer = tr(
                    "已检索到相关证据。由于 LLM 不可用，以下仅列出证据标题，不进行综合推断：\n",
                    "Relevant evidence was retrieved. Because the LLM is unavailable, only evidence titles are listed without synthesis:\n",
                ) + "\n".join(bullets)
            else:
                # 2) DeepSeek answer grounded in evidence and recent dialogue context.
                with st.spinner(tr("基于证据生成回答…", "Generating an evidence-grounded answer…")):
                    if language_code == "en":
                        system = (
                            "You are a friendly, professional medical Q&A assistant. The question has been classified as a medical knowledge query. "
                            "All medical facts and conclusions must be supported by the supplied evidence excerpts; do not fill gaps from outside knowledge. "
                            "Conversation history is only for resolving follow-ups and is not medical evidence. If the evidence is insufficient, say so explicitly. "
                            "Answer clearly in English and append [Evidence N] to every medical conclusion."
                        )
                    else:
                        system = (
                            "你是友善、专业的医学问答助手。当前问题已经被判定为医学知识问题。"
                            "医学事实和结论只能基于提供的证据片段，不得使用证据之外的知识补全。"
                            "历史对话只用于理解追问和指代，不能作为医学证据。"
                            "如果证据不足以回答，明确说「证据不足」，不要编造。"
                            "回答使用中文、自然清晰，并在每条医学结论后标注 [证据N]。"
                        )
                    history_context = build_conversation_context(
                        prior_messages,
                        language=language_code,
                    )
                    prompt = tr(
                        f"【最近对话】\n{history_context}\n\n"
                        f"【证据片段】\n{retrieved_text[:6000]}\n\n"
                        f"【当前问题】\n{question}",
                        f"[Recent conversation]\n{history_context}\n\n"
                        f"[Evidence excerpts]\n{retrieved_text[:6000]}\n\n"
                        f"[Current question]\n{question}",
                    )
                    try:
                        resp = llm.complete(prompt=prompt, system=system)
                        answer = (
                            resp.text
                            if not resp.degraded
                            else tr(
                                "LLM 调用降级，未生成结论；请查看下方检索证据。",
                                "The LLM call degraded, so no conclusion was generated. Review the retrieved evidence below.",
                            )
                        )
                    except Exception as e:
                        answer = tr(f"模型调用失败：{e}", f"Model call failed: {e}")

        st.write(answer)
        if show_evidence and evidence_chunks:
            with st.expander(
                tr(
                    f"检索证据（{len(evidence_chunks)} 段）",
                    f"Retrieved evidence ({len(evidence_chunks)} excerpts)",
                ),
                icon=":material/library_books:",
            ):
                for i, ev in enumerate(evidence_chunks, 1):
                    st.caption(tr(f"片段 {i}", f"Excerpt {i}"))
                    st.write(ev)

    st.session_state["chat_messages"].append({"role": "assistant", "content": answer, "evidence": evidence_chunks})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — 风险评估 / Screening (structured case → report)
# ─────────────────────────────────────────────────────────────────────────────
def _choice(label: str, values: dict[str, object], *, key: str):
    # Options carry the stable internal values and labels are applied through
    # format_func, so switching the UI language re-labels the current selection
    # instead of silently resetting every selectbox to the first option.
    options = list(values.values())
    label_for_value = {value: text for text, value in values.items()}
    return st.selectbox(
        label,
        options,
        format_func=lambda value: label_for_value.get(value, str(value)),
        key=key,
    )


def _tri_state(label: str, *, key: str) -> bool | None:
    return _choice(
        label,
        {
            tr("未知 / 未评估", "Unknown / not assessed"): None,
            tr("否", "No"): False,
            tr("是", "Yes"): True,
        },
        key=key,
    )


def _optional_number(
    label: str,
    *,
    key: str,
    min_value: float,
    max_value: float,
    step: float,
    integer: bool = False,
):
    value = st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=None,
        step=step,
        placeholder=tr("未录入", "Not entered"),
        key=key,
    )
    if value is None:
        return None
    return int(value) if integer else float(value)


def screening_tab(use_real_llm: bool, use_lightrag: bool, top_k: int) -> None:
    unknown = tr("未知 / 未评估", "Unknown / not assessed")
    st.header(tr("风险评估", "Risk assessment"))
    st.caption(
        tr(
            "录入患者与术中真实取值，系统自动派生多项风险标签。未知值不会被当作正常值，"
            "病例数据不会写入 LightRAG 索引。",
            "Enter known patient and intraoperative values. The system derives multiple risk flags; "
            "unknown values are never treated as normal, and case data is not written to the LightRAG index.",
        )
    )
    if current_language() == LANGUAGE_EN:
        st.markdown(
            ":blue-badge[:material/verified_user: De-identified input] "
            ":gray-badge[:material/do_not_disturb_on: Unknown is not normal]"
        )
    else:
        st.markdown(
            ":blue-badge[:material/verified_user: 去标识化输入] "
            ":gray-badge[:material/do_not_disturb_on: 未知不等于正常]"
        )

    with st.form("structured_case_form", clear_on_submit=False, border=False):
        case_box = st.container(border=True)
        case_id = case_box.text_input(
            tr("病例标识（请使用去标识化研究编号）", "Case ID (use a de-identified research code)"),
            value="streamlit_case",
            max_chars=100,
            key="case_id",
            help=tr(
                "请勿录入姓名、住院号、身份证号或联系电话。",
                "Do not enter a name, hospital ID, national ID, or phone number.",
            ),
        )
        patient_tab, surgery_tab = st.tabs(
            [
                tr(":material/person: 患者画像", ":material/person: Patient profile"),
                tr(":material/surgical: 手术与吻合口", ":material/surgical: Surgery and anastomosis"),
            ]
        )

        with patient_tab:
            patient_basics = st.container(border=True)
            patient_basics.markdown(tr("**基本资料与全身状态**", "**Demographics and functional status**"))
            c1, c2, c3 = patient_basics.columns(3)
            with c1:
                age_years = _optional_number(
                    tr("年龄（岁）", "Age (years)"),
                    key="patient_age",
                    min_value=18,
                    max_value=120,
                    step=1,
                    integer=True,
                )
                asa_grade = _choice(
                    tr("ASA 分级", "ASA class"),
                    {unknown: None, "I": "I", "II": "II", "III": "III", "IV": "IV", "V": "V"},
                    key="patient_asa",
                )
            with c2:
                sex = _choice(
                    tr("性别", "Sex"),
                    {unknown: None, tr("男", "Male"): "male", tr("女", "Female"): "female"},
                    key="patient_sex",
                )
                ecog_status = _choice(
                    tr("ECOG 体能状态", "ECOG performance status"),
                    {unknown: None, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
                    key="patient_ecog",
                )
            with c3:
                bmi = _optional_number(
                    tr("BMI（kg/m²）", "BMI (kg/m²)"),
                    key="patient_bmi",
                    min_value=10.0,
                    max_value=80.0,
                    step=0.1,
                )
                cancer_site = _choice(
                    tr("肿瘤部位", "Tumor site"),
                    {
                        unknown: None,
                        tr("结肠", "Colon"): "colon",
                        tr("直肠", "Rectum"): "rectum",
                        tr("低位直肠", "Lower rectum"): "lower_rectum",
                    },
                    key="patient_cancer_site",
                )

            patient_comorbidity = st.container(border=True)
            patient_comorbidity.markdown(tr("**合并症与急诊相关因素**", "**Comorbidity and urgent-disease factors**"))
            c1, c2, c3, c4 = patient_comorbidity.columns(4)
            with c1:
                diabetes_status = _choice(
                    tr("糖尿病状态", "Diabetes status"),
                    {
                        unknown: None,
                        tr("无", "No diabetes"): "no",
                        tr("1型", "Type 1"): "type1",
                        tr("2型", "Type 2"): "type2",
                    },
                    key="patient_diabetes",
                )
            with c2:
                hba1c_percent = _optional_number(
                    tr("HbA1c（%）", "HbA1c (%)"),
                    key="patient_hba1c",
                    min_value=3.0,
                    max_value=20.0,
                    step=0.1,
                )
            with c3:
                bowel_obstruction = _tri_state(tr("肠梗阻", "Bowel obstruction"), key="patient_obstruction")
            with c4:
                bowel_perforation = _tri_state(tr("肠穿孔", "Bowel perforation"), key="patient_perforation")

            patient_nutrition = st.container(border=True)
            patient_nutrition.markdown(tr("**营养与炎症**", "**Nutrition and inflammation**"))
            c1, c2, c3 = patient_nutrition.columns(3)
            with c1:
                albumin_g_l = _optional_number(
                    tr("白蛋白（g/L）", "Albumin (g/L)"),
                    key="patient_albumin",
                    min_value=10.0,
                    max_value=70.0,
                    step=0.1,
                )
                prealbumin_mg_l = _optional_number(
                    tr("前白蛋白（mg/L）", "Prealbumin (mg/L)"),
                    key="patient_prealbumin",
                    min_value=20.0,
                    max_value=700.0,
                    step=1.0,
                )
            with c2:
                nlr = _optional_number("NLR", key="patient_nlr", min_value=0.0, max_value=100.0, step=0.1)
                pni = _optional_number("PNI", key="patient_pni", min_value=0.0, max_value=100.0, step=0.1)
            with c3:
                nrs2002_score = _choice(
                    tr("NRS-2002 评分", "NRS-2002 score"),
                    {unknown: None, **{str(i): i for i in range(8)}},
                    key="patient_nrs2002",
                )

        with surgery_tab:
            surgery_complexity = st.container(border=True)
            surgery_complexity.markdown(tr("**手术方式与复杂度**", "**Procedure and operative complexity**"))
            c1, c2, c3 = surgery_complexity.columns(3)
            with c1:
                urgency = _choice(
                    tr("手术紧急程度", "Urgency"),
                    {unknown: None, tr("择期", "Elective"): "elective", tr("急诊", "Emergency"): "emergency"},
                    key="surgery_urgency",
                )
                multivisceral_resection = _tri_state(
                    tr("联合脏器切除", "Multivisceral resection"), key="surgery_multivisceral"
                )
            with c2:
                duration_minutes = _optional_number(
                    tr("手术时长（分钟）", "Operative duration (minutes)"),
                    key="surgery_duration",
                    min_value=15,
                    max_value=1440,
                    step=5,
                    integer=True,
                )
                blood_loss_ml = _optional_number(
                    tr("术中出血量（ml）", "Estimated blood loss (mL)"),
                    key="surgery_blood_loss",
                    min_value=0.0,
                    max_value=20000.0,
                    step=10.0,
                )
            with c3:
                transfusion_units = _optional_number(
                    tr("术中输血（U）", "Intraoperative transfusion (units)"),
                    key="surgery_transfusion",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                )
                peritoneal_contamination = _choice(
                    tr("腹腔污染", "Peritoneal contamination"),
                    {
                        unknown: None,
                        tr("无", "None"): "none",
                        tr("轻/中度", "Mild/moderate"): "mild",
                        tr("严重", "Severe"): "severe",
                    },
                    key="surgery_contamination",
                )

            anastomosis_details = st.container(border=True)
            anastomosis_details.markdown(tr("**吻合口特征与完整性**", "**Anastomotic characteristics and integrity**"))
            c1, c2, c3 = anastomosis_details.columns(3)
            with c1:
                anastomosis_site = _choice(
                    tr("吻合部位", "Anastomotic site"),
                    {
                        unknown: None,
                        tr("未行吻合", "No anastomosis"): "none",
                        tr("回结肠", "Ileocolic"): "ileocolic",
                        tr("结结肠", "Colocolic"): "colocolic",
                        tr("结直肠", "Colorectal"): "colorectal",
                        tr("超低位结直肠", "Ultra-low colorectal"): "ultra_low",
                        tr("结肠肛管", "Coloanal"): "coloanal",
                    },
                    key="surgery_anastomosis_site",
                )
                anastomosis_height_cm = _optional_number(
                    tr("吻合口距肛缘（cm）", "Anastomotic height from anal verge (cm)"),
                    key="surgery_anastomosis_height",
                    min_value=0.0,
                    max_value=30.0,
                    step=0.1,
                )
            with c2:
                doughnut_integrity = _choice(
                    tr("吻合圈完整性", "Stapler doughnut integrity"),
                    {
                        unknown: None,
                        tr("未评估", "Not assessed"): "not_assessed",
                        tr("完整", "Complete"): "complete",
                        tr("不完整", "Incomplete"): "incomplete",
                    },
                    key="surgery_doughnut",
                )
                air_leak_test = _choice(
                    tr("充气试验", "Intraoperative air-leak test"),
                    {
                        unknown: None,
                        tr("未进行", "Not performed"): "not_performed",
                        tr("阴性", "Negative"): "negative",
                        tr("阳性，修补后阴性", "Positive, negative after repair"): "positive_repaired",
                        tr("持续阳性", "Persistently positive"): "positive_persistent",
                    },
                    key="surgery_air_leak",
                )
            with c3:
                anastomotic_defect = _tri_state(
                    tr("术中发现吻合口缺损", "Intraoperative anastomotic defect"), key="surgery_defect"
                )

            perfusion_details = st.container(border=True)
            perfusion_details.markdown(tr("**灌注与术中生理**", "**Perfusion and intraoperative physiology**"))
            c1, c2, c3 = perfusion_details.columns(3)
            with c1:
                lowest_map_mmhg = _optional_number(
                    tr("最低 MAP（mmHg）", "Lowest MAP (mmHg)"),
                    key="surgery_map",
                    min_value=20.0,
                    max_value=160.0,
                    step=1.0,
                )
                map_below_65_minutes = _optional_number(
                    tr("MAP <65 持续时间（分钟）", "Minutes with MAP <65"),
                    key="surgery_map_duration",
                    min_value=0,
                    max_value=1440,
                    step=1,
                    integer=True,
                )
            with c2:
                icg_used = _tri_state(tr("是否使用 ICG", "ICG used"), key="surgery_icg_used")
                icg_perfusion_result = _choice(
                    tr("ICG 灌注结果", "ICG perfusion result"),
                    {
                        unknown: None,
                        tr("充分", "Adequate"): "adequate",
                        tr("临界", "Borderline"): "borderline",
                        tr("不足", "Inadequate"): "inadequate",
                    },
                    key="surgery_icg_result",
                )
            with c3:
                icg_action_after_inadequate_perfusion = _choice(
                    tr("ICG 灌注不足后处置", "Action after inadequate ICG perfusion"),
                    {
                        unknown: None,
                        tr("不适用", "Not applicable"): "not_applicable",
                        tr("改变切缘", "Transection line changed"): "transection_line_changed",
                        tr("重建吻合", "Anastomosis reconstructed"): "anastomosis_reconstructed",
                        tr("未采取矫正措施", "No corrective action"): "no_corrective_action",
                        tr("处置不详", "Action unknown"): "unknown",
                    },
                    key="surgery_icg_action",
                )

        submitted = st.form_submit_button(
            tr("开始风险评估", "Run risk assessment"),
            type="primary",
            icon=":material/clinical_notes:",
            width="stretch",
        )

    patient_values = {
        "age_years": age_years,
        "sex": sex,
        "bmi": bmi,
        "asa_grade": asa_grade,
        "ecog_status": ecog_status,
        "diabetes_status": diabetes_status,
        "hba1c_percent": hba1c_percent,
        "albumin_g_l": albumin_g_l,
        "prealbumin_mg_l": prealbumin_mg_l,
        "nlr": nlr,
        "pni": pni,
        "nrs2002_score": nrs2002_score,
        "cancer_site": cancer_site,
        "bowel_obstruction": bowel_obstruction,
        "bowel_perforation": bowel_perforation,
    }
    surgery_values = {
        "urgency": urgency,
        "multivisceral_resection": multivisceral_resection,
        "duration_minutes": duration_minutes,
        "blood_loss_ml": blood_loss_ml,
        "transfusion_units": transfusion_units,
        "anastomosis_site": anastomosis_site,
        "anastomosis_height_cm": anastomosis_height_cm,
        "doughnut_integrity": doughnut_integrity,
        "air_leak_test": air_leak_test,
        "anastomotic_defect": anastomotic_defect,
        "lowest_map_mmhg": lowest_map_mmhg,
        "map_below_65_minutes": map_below_65_minutes,
        "icg_used": icg_used,
        "icg_perfusion_result": icg_perfusion_result,
        "icg_action_after_inadequate_perfusion": icg_action_after_inadequate_perfusion,
        "peritoneal_contamination": peritoneal_contamination,
    }
    inputs_fingerprint = json.dumps(
        {"case_id": case_id, "patient": patient_values, "surgery": surgery_values},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )

    if submitted:
        try:
            case = ClinicalCaseInput(
                case_id=case_id.strip() or "streamlit_case",
                patient=PatientProfileInput(**patient_values),
                surgery=SurgeryProfileInput(**surgery_values),
            )
            with st.spinner(
                tr("运行规则、证据检索与报告流水线中…", "Running rules, evidence retrieval, and report generation…")
            ):
                t0 = time.time()
                report = get_agent(use_real_llm, use_lightrag, top_k).evaluate_case(case)
                elapsed = time.time() - t0
            st.session_state["screening_run"] = st.session_state.get("screening_run", 0) + 1
            st.session_state["screening_result"] = {
                "report": report,
                "elapsed": elapsed,
                "inputs": inputs_fingerprint,
                "settings": (use_real_llm, use_lightrag, top_k),
            }
        except Exception as exc:
            st.error(tr("评估失败：", "Assessment failed: ") + f"{type(exc).__name__}: {exc}")
            return

    result = st.session_state.get("screening_result")
    if not result:
        st.info(
            tr(
                "请录入已知字段后点击“开始风险评估”。可以保留未知项，系统不会将其当作正常值。",
                "Enter the known fields and select Run risk assessment. Unknown items may remain blank and will not be treated as normal.",
            ),
            icon=":material/info:",
        )
        return
    if isinstance(result, tuple):  # tolerate sessions started before the dict format
        result = {
            "report": result[0],
            "elapsed": result[1],
            "inputs": inputs_fingerprint,
            "settings": (use_real_llm, use_lightrag, top_k),
        }
        st.session_state["screening_result"] = result
    report, elapsed = result["report"], result["elapsed"]

    if not submitted:
        stale_notes: list[str] = []
        if result["inputs"] != inputs_fingerprint:
            stale_notes.append(tr("表单内容在上次评估后已修改", "Form inputs changed since the last run"))
        if tuple(result["settings"]) != (use_real_llm, use_lightrag, top_k):
            stale_notes.append(tr("评估设置已更改", "Assessment settings changed"))
        if stale_notes:
            st.info(
                tr(
                    f"{'；'.join(stale_notes)}。当前显示上次提交的结果，重新点击“开始风险评估”以更新。",
                    f"{'; '.join(stale_notes)}. Showing the previous result; select Run risk assessment to update.",
                ),
                icon=":material/edit_note:",
            )

    warnings = report.metadata.get("warnings") or []
    for warning in warnings:
        st.warning(localized_warning(warning), icon=":material/warning:")

    missing_domains = report.metadata.get("missing_domains") or []
    if missing_domains:
        localized_domains = [tr(*DOMAIN_LABELS.get(domain, (domain, domain))) for domain in missing_domains]
        st.warning(
            tr(
                f"输入完整度 {report.risk_assessment.input_completeness:.0%}；"
                f"尚缺少：{'、'.join(localized_domains)}。"
                "若未触发规则，系统将输出暂无法判断，而不是低风险。",
                f"Input completeness is {report.risk_assessment.input_completeness:.0%}. "
                f"Missing domains: {', '.join(localized_domains)}. "
                "If no rule is triggered, the system returns Unable to determine rather than Low risk.",
            ),
            icon=":material/data_alert:",
        )

    risk = report.risk_assessment.risk_level.value
    confidence = float(report.risk_assessment.confidence_score or 0)
    n_signals = len(report.risk_assessment.signals)
    n_evidence = len(report.evidence)
    llm_label = report.metadata.get("llm_label", "-")

    st.subheader(tr("规则优先级结果", "Rule-priority result"))
    render_risk_hero(
        risk,
        confidence,
        n_signals,
        n_evidence,
        elapsed,
        llm_label,
        primary_alert=report.risk_assessment.primary_alert,
        risk_score=report.risk_assessment.risk_score,
        primary_alert_threshold=report.risk_assessment.primary_alert_threshold,
        safety_guardrail_present=report.risk_assessment.safety_guardrail_present,
        requires_immediate_action=report.risk_assessment.requires_immediate_action,
    )

    with st.container(border=True):
        st.markdown(tr("**系统派生标签**", "**System-derived flags**"))
        flag_cols = st.columns(2)
        with flag_cols[0]:
            st.caption(tr("患者标签", "Patient flags"))
            patient_flags = report.metadata.get("patient_flags") or []
            st.write(" · ".join(f"`{flag}`" for flag in patient_flags) or tr("无", "None"))
        with flag_cols[1]:
            st.caption(tr("手术与吻合口标签", "Surgical and anastomotic flags"))
            surgery_flags = report.metadata.get("surgery_flags") or []
            st.write(" · ".join(f"`{flag}`" for flag in surgery_flags) or tr("无", "None"))
        with st.expander(tr("查看标签派生依据", "View flag derivation"), icon=":material/account_tree:"):
            trace = report.metadata.get("flag_trace") or {}
            if trace:
                if current_language() == LANGUAGE_EN:
                    st.caption("Derivation statements below are preserved in their source language.")
                for flag, reason in trace.items():
                    st.write(f"- `{flag}`：{reason}")
            else:
                st.write(tr("未派生风险标签。", "No risk flags were derived."))

    tab_summary, tab_rules, tab_evidence, tab_technical = st.tabs(
        [
            tr(":material/summarize: 报告摘要", ":material/summarize: Report summary"),
            tr(":material/rule: 规则详情", ":material/rule: Rule details"),
            tr(":material/library_books: 证据链", ":material/library_books: Evidence chain"),
            tr(":material/settings: 技术细节", ":material/settings: Technical details"),
        ]
    )

    with tab_summary:
        with st.container(border=True):
            st.subheader(tr("结构化报告", "Structured report"))
            render_localized_report_summary(report)

        if report.risk_assessment.signals:
            st.subheader(tr("主要风险信号", "Main risk signals"))
            sorted_signals = sorted(
                report.risk_assessment.signals,
                key=lambda s: {"high": 0, "medium": 1, "low": 2, "unknown": 3}.get(s.risk_level.value, 4),
            )
            for sig in sorted_signals[:3]:  # top 3 only in summary
                sev = sig.risk_level.value
                render_rule_chip(sig.rule_id, sev, sig.risk_type.split("_")[-1] if sig.risk_type else "")

    with tab_rules:
        if not report.risk_assessment.signals:
            st.info(tr("未触发任何规则。", "No rule was triggered."), icon=":material/check_circle:")
        else:
            if current_language() == LANGUAGE_EN:
                st.caption(
                    "Clinical rationales and recommendations are displayed in their audited source language to avoid unreviewed translation."
                )
            sorted_signals = sorted(
                report.risk_assessment.signals,
                key=lambda s: {"high": 0, "medium": 1, "low": 2, "unknown": 3}.get(s.risk_level.value, 4),
            )
            for sig in sorted_signals:
                sev = sig.risk_level.value
                with st.container(border=True):
                    render_rule_chip(sig.rule_id, sev, sig.risk_type)
                    st.write(tr("**依据（原文）：** ", "**Rationale (source text):** ") + sig.rationale)
                    if sig.recommendation:
                        st.write(tr("**建议（原文）：** ", "**Recommendation (source text):** ") + sig.recommendation)
                    if sig.matched_terms:
                        with st.expander(tr("查看匹配标签", "View matched flags"), icon=":material/label:"):
                            for key, values in sig.matched_terms.items():
                                if values:
                                    st.write(f"- **{key}**：{', '.join(values)}")

    with tab_evidence:
        if not report.evidence:
            st.info(tr("未检索到证据。", "No evidence was retrieved."), icon=":material/library_books:")
        else:
            levels_present = sorted({ev.evidence_level for ev in report.evidence})
            selected_levels = st.pills(
                tr("筛选证据等级", "Filter evidence level"),
                options=["A", "B", "C", "D", "E"],
                default=levels_present,
                selection_mode="multi",
                help=tr(
                    "A=监管或权威来源，B=系统评价，C=原创研究，D=体外或动物研究，E=机制推断。",
                    "A=regulatory or authoritative source, B=systematic review, C=original study, D=in-vitro or animal study, E=mechanistic inference.",
                ),
                # Key the filter to the assessment run so a previous report's
                # selection can never hide levels present in the current one.
                key=f"screen_ev_filter_{st.session_state.get('screening_run', 0)}",
                width="stretch",
            )
            for ev in report.evidence:
                if ev.evidence_level not in selected_levels:
                    continue
                render_evidence_card(ev.evidence_level, ev.title[:80], ev.source, ev.text)

    with tab_technical:
        st.subheader(tr("运行元数据", "Run metadata"))
        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.metric(tr("病例标识", "Case ID"), report.case_id, border=True)
            st.metric(tr("输入模式", "Input mode"), report.metadata.get("input_mode"), border=True)
        with meta_cols[1]:
            st.metric(tr("LLM 角色", "LLM role"), report.metadata.get("llm_role"), border=True)
            st.metric(tr("LLM 模型", "LLM model"), report.metadata.get("llm_label"), border=True)
            st.metric(tr("上下文字符数", "Context characters"), report.context_char_count, border=True)
        with meta_cols[2]:
            st.metric(tr("规则信号数", "Rule signals"), report.metadata.get("n_signals"), border=True)
            st.metric(tr("证据片段数", "Evidence excerpts"), report.metadata.get("n_evidence"), border=True)
            st.metric(tr("耗时", "Elapsed time"), f"{elapsed:.2f} s", border=True)

        with st.expander(tr("完整 metadata JSON", "Full metadata JSON"), icon=":material/data_object:"):
            st.json(report.metadata)
        with st.expander(tr("去标识化病例输入", "De-identified case input"), icon=":material/patient_list:"):
            st.json(report.metadata.get("case_input") or {})

    st.space("small")
    with st.container(horizontal=True):
        st.download_button(
            tr("下载报告（Markdown）", "Download report (Markdown)"),
            data=report.report_text,
            file_name=f"{report.case_id}_risk_report.md",
            mime="text/markdown",
            icon=":material/download:",
        )
        st.download_button(
            tr("下载运行元数据（JSON）", "Download run metadata (JSON)"),
            data=json.dumps(report.metadata, ensure_ascii=False, indent=2, default=str),
            file_name=f"{report.case_id}_metadata.json",
            mime="application/json",
            icon=":material/data_object:",
        )
    render_footer(
        {
            "case_id": report.case_id,
            "llm_label": llm_label,
            "context_chars": report.context_char_count,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main — 2 top-level pages
# ─────────────────────────────────────────────────────────────────────────────
def render_app_header() -> None:
    st.title(tr("吻合口漏风险决策支持", "Anastomotic leak decision support"))
    st.markdown(
        tr(
            ":blue-badge[:material/account_tree: 规则与证据可追溯]",
            ":blue-badge[:material/account_tree: Traceable rules and evidence]",
        )
    )
    st.caption(
        tr(
            "结直肠癌根治术 · 确定性规则分级 · LightRAG 证据检索 · LLM 辅助解释",
            "Curative colorectal cancer surgery · Deterministic rule grading · "
            "LightRAG evidence retrieval · LLM-assisted explanation",
        )
    )


def screening_sidebar() -> tuple[bool, bool, int]:
    """Render page-local screening controls and return their values."""
    with st.sidebar:
        st.subheader(tr("评估设置", "Assessment settings"))
        st.caption(
            tr(
                "这些选项只影响证据检索和解释，不改变规则判定逻辑。",
                "These options affect evidence retrieval and explanation only; "
                "they do not change deterministic rule grading.",
            )
        )
        use_real_llm = st.toggle(
            tr("启用 LLM 辅助解释", "Enable LLM-assisted explanation"),
            value=True,
            help=tr(
                "LLM 仅生成基于证据的辅助说明，不改变规则分级。",
                "The LLM generates evidence-grounded explanations only and does not alter rule grading.",
            ),
        )
        use_lightrag = st.toggle(
            tr("启用 LightRAG 检索", "Enable LightRAG retrieval"),
            value=True,
            help=tr(
                "仅当当前索引通过严格验收时建议开启。",
                "Enable only after the current index passes strict acceptance checks.",
            ),
        )
        top_k = st.slider(
            tr("证据片段数量", "Number of evidence excerpts"),
            min_value=1,
            max_value=15,
            value=5,
            help=tr(
                "数值越大，解释可参考的证据越多，但上下文也更长。",
                "Higher values provide more evidence but increase context length.",
            ),
        )

        with st.expander(tr("系统状态", "System status"), icon=":material/monitoring:"):
            fresh_status, fresh_detail = index_freshness()
            render_status_pill(fresh_status)
            st.caption(localized_freshness_detail(fresh_status, fresh_detail))
            try:
                a_reg = load_entity_a_registry()
                b_reg = load_entity_b_registry()
                st.metric(tr("患者特征项", "Patient features"), len(a_reg), border=True)
                st.metric(
                    tr("手术与吻合口特征项", "Surgical and anastomotic features"),
                    len(b_reg),
                    border=True,
                )
            except Exception as exc:
                st.error(tr("注册表加载失败：", "Registry loading failed: ") + str(exc))

        if st.button(
            tr("清空本地缓存", "Clear local cache"),
            icon=":material/refresh:",
            width="stretch",
        ):
            clear_cache()
            st.cache_resource.clear()
            st.toast(tr("缓存已清空", "Cache cleared"), icon=":material/check_circle:")
    return use_real_llm, use_lightrag, top_k


def chat_page() -> None:
    """Render evidence Q&A in the entrypoint's shared session context."""
    render_app_header()
    chat_tab()


def screening_page() -> None:
    """Render screening in the entrypoint's shared session context."""
    render_app_header()
    use_real_llm, use_lightrag, top_k = screening_sidebar()
    screening_tab(use_real_llm, use_lightrag, top_k)


def main() -> None:
    """Multipage entry point with one shared language/session context."""
    set_page_config()
    require_access_code()
    render_language_control()
    navigation = st.navigation(
        [
            st.Page(
                screening_page,
                title=tr("风险评估", "Risk assessment"),
                icon=":material/clinical_notes:",
                url_path="screening",
                default=True,
            ),
            st.Page(
                chat_page,
                title=tr("证据问答", "Evidence Q&A"),
                icon=":material/forum:",
                url_path="chat",
            ),
        ],
        position="top",
    )
    navigation.run()


if __name__ == "__main__":
    main()
