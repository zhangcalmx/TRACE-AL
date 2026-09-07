"""Provider-neutral OpenAI-compatible chat client.

The default UI path keeps the historical DeepSeek/CSU behaviour.  Validation
code can select an exact provider, base URL, and model so a failed endpoint is
never silently replaced by a different model.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class ChatResponse:
    text: str
    degraded: bool = False


@dataclass
class LLMConfig:
    provider: str = "auto"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout_seconds: float = 60.0
    seed: int | None = None


class LLMClient:
    def __init__(
        self,
        provider: str | LLMConfig = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        if isinstance(provider, LLMConfig):
            config = provider
            temperature = config.temperature if temperature is None else temperature
            max_tokens = config.max_tokens if max_tokens is None else max_tokens
        else:
            config = LLMConfig(provider=provider)

        requested_provider = config.provider.strip().lower()
        self.provider = requested_provider
        self.degraded = False
        self.temperature = float(
            temperature if temperature is not None else os.getenv("LLM_CHAT_TEMPERATURE", "0.0")
        )
        self.max_tokens = int(max_tokens or os.getenv("LLM_CHAT_MAX_TOKENS", "4096"))
        self.timeout_seconds = float(config.timeout_seconds)
        self.seed = config.seed

        if requested_provider == "stub":
            self.degraded = True
            self.api_key = ""; self.base_url = ""; self.model = "stub"
            return

        if requested_provider == "auto":
            if os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_CHAT_API_KEY"):
                requested_provider = "deepseek"
            elif os.getenv("CSU_API_KEY"):
                requested_provider = "csu"
            elif os.getenv("ARK_API_KEY") and os.getenv("ARK_CHAT_MODEL"):
                requested_provider = "ark"
            else:
                requested_provider = "stub"

        if requested_provider == "deepseek":
            self.api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_CHAT_API_KEY") or ""
            raw_base = config.base_url or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("LLM_CHAT_BASE_URL") or "https://api.deepseek.com/v1"
            self.base_url = raw_base.rstrip("/")
            if self.base_url == "https://api.deepseek.com":
                self.base_url += "/v1"
            self.model = config.model or os.getenv("DEEPSEEK_CHAT_MODEL") or os.getenv("LLM_CHAT_MODEL") or "deepseek-v4-flash"
        elif requested_provider == "csu":
            self.api_key = config.api_key or os.getenv("CSU_API_KEY") or ""
            self.base_url = (config.base_url or os.getenv("CSU_BASE_URL") or "https://api.chat.csu.edu.cn/v1").rstrip("/")
            self.model = config.model or os.getenv("CSU_CHAT_MODEL") or "DeepSeek-V4-Flash"
        elif requested_provider in {"ark", "ark_coding"}:
            self.api_key = config.api_key or os.getenv("ARK_API_KEY") or ""
            default_base = (
                "https://ark.cn-beijing.volces.com/api/coding/v3"
                if requested_provider == "ark_coding"
                else "https://ark.cn-beijing.volces.com/api/v3"
            )
            self.base_url = (
                config.base_url
                or os.getenv("ARK_CHAT_BASE_URL")
                or os.getenv("ARK_BASE_URL")
                or default_base
            ).rstrip("/")
            self.model = config.model or os.getenv("ARK_CHAT_MODEL") or ""
        elif requested_provider == "openai_compatible":
            self.api_key = config.api_key or os.getenv("LLM_CHAT_API_KEY") or ""
            self.base_url = (config.base_url or os.getenv("LLM_CHAT_BASE_URL") or "").rstrip("/")
            self.model = config.model or os.getenv("LLM_CHAT_MODEL") or ""
        elif requested_provider == "stub":
            self.api_key = ""; self.base_url = ""; self.model = "stub"
        else:
            raise ValueError(f"Unsupported LLM provider: {requested_provider}")

        self.provider = requested_provider
        if not self.api_key or not self.base_url or not self.model or requested_provider == "stub":
            self.degraded = True

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        history_messages: list[dict] | None = None,
    ) -> ChatResponse:
        if self.degraded:
            return ChatResponse(text=f"[stub] no LLM key; prompt={len(prompt)} chars", degraded=True)
        messages = []
        if system: messages.append({"role": "system", "content": system})
        for message in history_messages or []:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})
        try:
            return ChatResponse(text=self._chat(messages), degraded=False)
        except Exception as e:
            return ChatResponse(text=f"[llm error] {e}", degraded=True)

    def _chat(self, messages: list[dict]) -> str:
        request: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            request["seed"] = self.seed
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )
            resp = client.chat.completions.create(**request)
            return resp.choices[0].message.content or ""
        except ImportError:
            pass
        body = json.dumps(request).encode()
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(url, data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


class DualLLMClient:
    """Compatibility facade for the original two-provider UI contract.

    The project currently uses a single configured OpenAI-compatible endpoint.
    The facade preserves the public API without pretending that two independent
    model judgements were performed.
    """

    def __init__(self, primary: LLMClient | None = None) -> None:
        self.primary = primary or LLMClient(provider="auto")

    @classmethod
    def from_env(cls) -> DualLLMClient:
        return cls(LLMClient(provider="auto"))

    @property
    def degraded(self) -> bool:
        return self.primary.degraded

    @property
    def model(self) -> str:
        return self.primary.model

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        history_messages: list[dict] | None = None,
    ) -> ChatResponse:
        return self.primary.complete(
            prompt,
            system=system,
            history_messages=history_messages,
        )
