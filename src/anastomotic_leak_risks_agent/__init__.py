"""结直肠癌根治术吻合口漏智能风险决策系统 — 手术并发症-吻合口漏风险预测与决策支持 screening agent."""
__version__ = "0.1.0"

__all__ = ["SafetyAgent", "SafetyAgentConfig", "__version__"]


def __getattr__(name: str):
    """Load orchestration classes lazily to avoid import cycles on hot reload."""
    if name in {"SafetyAgent", "SafetyAgentConfig"}:
        from .safety_agent import SafetyAgent, SafetyAgentConfig

        return {"SafetyAgent": SafetyAgent, "SafetyAgentConfig": SafetyAgentConfig}[name]
    raise AttributeError(name)
