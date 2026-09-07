"""Embedding client (火山方舟 Ark: doubao-embedding-vision-251215).
每次只接受 1 条输入, 并发通过 ThreadPoolExecutor。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0xF900, 0xFAFF))
def _has_cjk(text: str) -> bool:
    return any(any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES) for ch in text)


class ArkEmbedding:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 endpoint_id: str | None = None, embedding_dim: int | None = None,
                 concurrency: int | None = None, timeout: float = 30.0):
        self.api_key = api_key or os.getenv("ARK_API_KEY") or ""
        self.base_url = (base_url or os.getenv("ARK_EMBED_BASE_URL")
                         or "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
        self.endpoint_id = endpoint_id or os.getenv("ARK_EMBED_ENDPOINT_ID") or ""
        self.model_field = self.endpoint_id or "doubao-embedding-vision-251215"
        self.embedding_dim = int(embedding_dim or os.getenv("ARK_EMBED_DIM") or 2048)
        self.concurrency = int(concurrency or os.getenv("ARK_EMBED_CONCURRENCY") or 8)
        self.timeout = timeout
        self._url = f"{self.base_url}/embeddings/multimodal"
        self._dim_verified = False
        if not self.api_key:
            raise RuntimeError("ArkEmbedding: no ARK_API_KEY in .env")

    def _post(self, text: str, *, cn_prefix: bool = False) -> list[float]:
        payload = ("[text] " + text) if cn_prefix else text
        body = {"model": self.model_field, "input": [{"type": "text", "text": payload}], "encoding_format": "float"}
        req_body = json.dumps(body).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(3):
            try:
                req = urllib.request.Request(self._url, data=req_body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                emb = data.get("data")
                if isinstance(emb, dict) and "embedding" in emb:
                    return list(emb["embedding"])
                if isinstance(emb, list) and emb and "embedding" in emb[0]:
                    return list(emb[0]["embedding"])
                raise RuntimeError(f"unexpected: {str(data)[:200]}")
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503):
                    if attempt < 2:
                        time.sleep(0.5 * (2**attempt)); continue
                    if _has_cjk(text) and not cn_prefix:
                        return self._post(text, cn_prefix=True)
                snippet = e.read().decode("utf-8", errors="ignore")[:200] if attempt == 2 else ""
                raise RuntimeError(f"Ark HTTP {e.code}: {snippet}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < 2:
                    time.sleep(0.5 * (2**attempt)); continue
                raise RuntimeError(f"Ark network: {e}")
        raise RuntimeError(f"Ark failed: {text[:50]}")

    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        if isinstance(texts, str): texts = [texts]
        if not texts: return []
        n = min(self.concurrency, len(texts))
        with ThreadPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(self._post, texts))
        if not self._dim_verified:
            got = len(results[0])
            if got != self.embedding_dim:
                raise RuntimeError(f"Ark dim mismatch: got {got}, expected {self.embedding_dim}")
            self._dim_verified = True
        return results

    async def aembed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return self.embed(texts, **kwargs)
