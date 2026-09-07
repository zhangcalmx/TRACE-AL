"""Deterministic intent routing for the Streamlit chat experience.

Conversational messages should not be sent to a medical evidence retriever.
Keeping this first routing step deterministic also prevents an LLM from making
unsupported medical claims when no evidence has been retrieved.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import Enum


class ChatIntent(str, Enum):
    """Supported top-level chat intents."""

    GREETING = "greeting"
    HELP = "help"
    THANKS = "thanks"
    FAREWELL = "farewell"
    CASE_ASSESSMENT = "case_assessment"
    MEDICAL_QUERY = "medical_query"
    OUT_OF_SCOPE = "out_of_scope"


_GREETING_RE = re.compile(
    r"^(?:你好|您好|嗨|哈[喽啰]|hello|hi|hey|早上好|上午好|中午好|下午好|晚上好|在吗)"
    r"(?:呀|啊|哦|哈|！|!|。|\?|？|～|~)*$",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(
    r"^(?:谢谢|感谢|多谢|辛苦了|thank\s*you|thanks)(?:你|您|啦|了|！|!|。|～|~)*$",
    re.IGNORECASE,
)
_FAREWELL_RE = re.compile(
    r"^(?:再见|拜拜|回头见|bye|goodbye)(?:啦|了|！|!|。|～|~)*$",
    re.IGNORECASE,
)
_FOLLOW_UP_RE = re.compile(
    r"^(?:为什么|怎么回事|怎么办|什么意思|依据是什么|有证据吗|请展开|详细说说|"
    r"继续|还有吗|然后呢|那呢|这个呢|它呢|why|how|what does that mean|"
    r"what is the evidence|any evidence|please explain|tell me more|continue|"
    r"what about this|what about it)(?:\?|？|！|!|。)*$",
    re.IGNORECASE,
)

_HELP_TERMS = (
    "你能做什么",
    "你会什么",
    "怎么使用",
    "如何使用",
    "使用帮助",
    "使用说明",
    "怎么提问",
    "你是谁",
    "介绍一下自己",
    "what can you do",
    "how do i use",
    "how to use",
    "help",
    "who are you",
)

_CASE_TERMS = (
    "帮我评估",
    "进行评估",
    "风险评估",
    "评估这个患者",
    "评估这位患者",
    "患者情况",
    "病例情况",
    "这个患者",
    "这位患者",
    "这个病人",
    "这位病人",
    "assess this patient",
    "assess the patient",
    "risk assessment",
    "patient case",
    "patient profile",
)

_MEDICAL_TERMS = (
    "吻合口漏",
    "吻合口瘘",
    "吻合漏",
    "结直肠",
    "结肠",
    "直肠",
    "吻合口",
    "低位吻合",
    "手术",
    "造口",
    "icg",
    "吲哚菁绿",
    "荧光",
    "灌注",
    "糖尿病",
    "白蛋白",
    "营养不良",
    "bmi",
    "体重",
    "肥胖",
    "年龄",
    "吸烟",
    "饮酒",
    "放疗",
    "化疗",
    "新辅助",
    "急诊",
    "失血",
    "输血",
    "风险因素",
    "危险因素",
    "并发症",
    "anastom",
    "leak",
    "colorectal",
    "rectal",
    "colon",
    "surgery",
    "stoma",
    "perfusion",
    "diabetes",
    "albumin",
    "nutrition",
    "obesity",
    "risk factor",
    "complication",
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _last_user_message(history: Sequence[Mapping[str, object]] | None) -> str:
    for message in reversed(history or ()):
        if message.get("role") == "user":
            return _clean(str(message.get("content", "")))
    return ""


def _looks_like_case_description(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in _CASE_TERMS):
        return True
    has_patient_subject = any(
        term in lowered for term in ("患者", "病人", "病例", "patient", "case")
    )
    has_structured_value = bool(
        re.search(r"\d+(?:\.\d+)?\s*(?:岁|kg|公斤|cm|厘米)", text, re.IGNORECASE)
        or re.search(r"\d+(?:\.\d+)?\s*(?:years?\s*old|y/o)", text, re.IGNORECASE)
        or re.search(
            r"(?:bmi|albumin|hba1c|白蛋白|血红蛋白)\s*[：:=]?\s*\d+",
            text,
            re.IGNORECASE,
        )
    )
    return has_patient_subject and has_structured_value


def _is_medical(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _MEDICAL_TERMS)


def classify_chat_intent(
    message: str,
    history: Sequence[Mapping[str, object]] | None = None,
) -> ChatIntent:
    """Classify a chat message before any evidence retrieval is attempted."""

    text = _clean(message)
    if not text:
        return ChatIntent.OUT_OF_SCOPE

    if _GREETING_RE.fullmatch(text):
        return ChatIntent.GREETING
    if _THANKS_RE.fullmatch(text):
        return ChatIntent.THANKS
    if _FAREWELL_RE.fullmatch(text):
        return ChatIntent.FAREWELL
    if any(term in text.lower() for term in _HELP_TERMS):
        return ChatIntent.HELP
    if _looks_like_case_description(text):
        return ChatIntent.CASE_ASSESSMENT
    if _is_medical(text):
        return ChatIntent.MEDICAL_QUERY

    previous_user_message = _last_user_message(history)
    if _FOLLOW_UP_RE.fullmatch(text) and _is_medical(previous_user_message):
        return ChatIntent.MEDICAL_QUERY

    return ChatIntent.OUT_OF_SCOPE


def direct_response_for(intent: ChatIntent, *, language: str = "zh") -> str:
    """Return a safe non-RAG response for conversational intents."""

    if language.lower().startswith("en"):
        responses_en = {
            ChatIntent.GREETING: (
                "Hello! I’m the anastomotic-leak risk decision-support assistant for "
                "colorectal cancer surgery. I can explain risk factors and their evidence, "
                "or guide you through a structured patient and operative assessment.\n\n"
                "You can ask:\n\n"
                "- Why does a low rectal anastomosis increase leak risk?\n"
                "- What is the evidence linking diabetes to anastomotic leak?\n"
                "- How do I start a patient risk assessment?"
            ),
            ChatIntent.HELP: (
                "I provide three types of support:\n\n"
                "- **Evidence-based Q&A:** risk factors, mechanisms, and supporting sources.\n"
                "- **Structured risk assessment:** enter patient and operative values on the Risk assessment page.\n"
                "- **Result interpretation:** review triggered rules, evidence, and items requiring clinical review.\n\n"
                "Medical statements are constrained to the project evidence base and do not replace clinical judgment."
            ),
            ChatIntent.THANKS: (
                "You’re welcome. You can continue with a risk-factor question, ask for evidence, "
                "or open Risk assessment to enter a case."
            ),
            ChatIntent.FAREWELL: (
                "Goodbye. Return whenever you need to review anastomotic-leak evidence or assess a case."
            ),
            ChatIntent.CASE_ASSESSMENT: (
                "I can help. To avoid missing important variables, open Risk assessment and enter "
                "age, BMI, comorbidities, nutritional markers, anastomotic height, operative details, "
                "and perfusion findings. The system can derive multiple patient and surgical factors together."
            ),
            ChatIntent.OUT_OF_SCOPE: (
                "I’m a focused medical decision-support assistant for anastomotic-leak risk after "
                "colorectal cancer surgery. Ask about a risk factor and its evidence, or type “help” for guidance."
            ),
        }
        return responses_en.get(intent, "")

    responses = {
        ChatIntent.GREETING: (
            "您好！我是结直肠癌根治术吻合口漏风险决策助手。"
            "我可以查询危险因素及其证据，也可以结合患者和手术信息进行结构化风险评估。"
            "您可以问我：\n\n"
            "- 低位直肠吻合为什么会增加吻合口漏风险？\n"
            "- 糖尿病与吻合口漏有什么关系？\n"
            "- 如何开始患者风险评估？"
        ),
        ChatIntent.HELP: (
            "我主要提供三类帮助：\n\n"
            "- **循证问答**：解释吻合口漏危险因素、机制和证据。\n"
            "- **病例风险评估**：请切换到“🔍 风险评估”，填写患者与手术的具体数值和选项。\n"
            "- **结果解释**：说明触发的规则、证据来源和需要人工复核的项目。\n\n"
            "医学结论只基于项目证据库生成，不能替代临床医生判断。"
        ),
        ChatIntent.THANKS: "不客气。如果您愿意，可以继续询问危险因素、证据依据，或进入“风险评估”填写病例信息。",
        ChatIntent.FAREWELL: "再见！需要查询吻合口漏证据或评估病例风险时，随时回来。",
        ChatIntent.CASE_ASSESSMENT: (
            "可以帮助评估。为避免遗漏关键变量，请切换到上方的“🔍 风险评估”页面，"
            "填写年龄、BMI、合并症、营养指标，以及吻合口高度、术式、灌注评估等信息。"
            "系统会同时识别多个患者因素和手术因素，并给出可追溯的规则与证据。"
        ),
        ChatIntent.OUT_OF_SCOPE: (
            "我是专注于结直肠癌根治术吻合口漏风险的医学决策助手。"
            "您可以询问某个危险因素及其证据，或输入“你能做什么”查看使用方法。"
        ),
    }
    return responses.get(intent, "")


def build_retrieval_query(
    message: str,
    history: Sequence[Mapping[str, object]] | None = None,
    *,
    language: str = "zh",
) -> str:
    """Resolve short follow-ups against the most recent user question."""

    text = _clean(message)
    previous_user_message = _last_user_message(history)
    if previous_user_message and _FOLLOW_UP_RE.fullmatch(text):
        if language.lower().startswith("en"):
            return f"{previous_user_message}\nFollow-up: {text}"
        return f"{previous_user_message}\n追问：{text}"
    return text


def build_conversation_context(
    history: Sequence[Mapping[str, object]] | None,
    *,
    max_messages: int = 6,
    max_chars: int = 2400,
    language: str = "zh",
) -> str:
    """Format recent text-only turns for follow-up understanding."""

    rendered: list[str] = []
    for message in list(history or ())[-max_messages:]:
        role = message.get("role")
        content = _clean(str(message.get("content", "")))
        if role not in {"user", "assistant"} or not content:
            continue
        if language.lower().startswith("en"):
            label = "User" if role == "user" else "Assistant"
            rendered.append(f"{label}: {content}")
        else:
            label = "用户" if role == "user" else "助手"
            rendered.append(f"{label}：{content}")
    context = "\n".join(rendered)
    if context:
        return context[-max_chars:]
    return "(No conversation history)" if language.lower().startswith("en") else "（无历史对话）"


def no_evidence_response(*, language: str = "zh") -> str:
    """A helpful, bounded response when a medical query has no evidence hit."""

    if language.lower().startswith("en"):
        return (
            "I understand this is a medical question, but the project evidence base did not "
            "retrieve sufficient support for an answer, so I will not generate a medical conclusion. "
            "Try a more specific factor or procedure, such as low rectal anastomosis, diabetes, "
            "or ICG perfusion assessment; you can also enter a complete case on Risk assessment."
        )

    return (
        "我理解这是一个医学问题，但当前项目证据库没有检索到足以支持回答的相关证据，"
        "因此暂不生成医学结论。您可以补充更具体的因素或术式，例如“低位直肠吻合”、"
        "“糖尿病”或“ICG 灌注评估”；也可以进入“风险评估”填写完整病例信息。"
    )
