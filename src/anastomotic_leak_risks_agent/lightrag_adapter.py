"""LightRAG adapter: 火山 Ark chat + Ark embedding."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

try:
    from .embedding_client import ArkEmbedding
    from .llm_client import LLMClient
except ImportError:
    from embedding_client import ArkEmbedding
    from llm_client import LLMClient


class LightRAGAdapter:
    """Wrapper around lightrag.LightRAG with Ark chat + Ark embedding."""

    # Query settings validated for the 2026-08-25 acceptance index
    # (acceptance_manifest.json stats: mix mode, top_k 40, chunk_top_k 20,
    # cosine threshold 0.20, rerank disabled). Values are pinned so library
    # default changes cannot silently drift from the manuscript configuration.
    DEFAULT_QUERY_MODE = "mix"
    DEFAULT_QUERY_TOP_K = 40
    DEFAULT_QUERY_CHUNK_TOP_K = 20
    DEFAULT_QUERY_COSINE_THRESHOLD = 0.2

    def __init__(
        self,
        working_dir: str | None = None,
        chat_client: LLMClient | None = None,
        embed_client: ArkEmbedding | None = None,
        language: str = "Chinese",
    ) -> None:
        try:
            from lightrag import LightRAG, QueryParam
            try:
                from lightrag.base import EmbeddingFunc
            except ImportError:
                from lightrag.utils import EmbeddingFunc
        except ImportError as e:
            raise ImportError("LightRAGAdapter requires lightrag-hku.") from e

        self._QueryParam = QueryParam
        self.working_dir = working_dir or os.getenv("LIGHTRAG_WORKING_DIR") or self._current_index_dir()
        os.makedirs(self.working_dir, exist_ok=True)

        self.chat = chat_client or LLMClient(
            provider="auto",
            max_tokens=int(os.getenv("LIGHTRAG_LLM_MAX_TOKENS", "8192")),
        )
        self.embed = embed_client or ArkEmbedding()

        async def llm_model_func(
            prompt: str, system_prompt: str | None = None,
            history_messages: list[dict] | None = None,
            keyword_extraction: bool = False, **kwargs: Any,
        ) -> str:
            sys_p = system_prompt
            if keyword_extraction and not sys_p:
                sys_p = "Extract up to 10 high-level keywords. Return comma-separated only."
            resp = self.chat.complete(
                prompt=prompt,
                system=sys_p,
                history_messages=history_messages,
            )
            return resp.text

        async def _embed_async(texts: list[str]):
            import numpy as np
            vecs = self.embed.embed(texts)
            return np.asarray(vecs, dtype=np.float32)

        embedding_func = EmbeddingFunc(
            embedding_dim=self.embed.embedding_dim,
            max_token_size=8192,
            func=_embed_async,
        )

        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=llm_model_func,
            llm_model_name=self.chat.model,
            embedding_func=embedding_func,
            # Medical evidence indexing favors source fidelity over speculative
            # recall.  One grounded extraction pass is sufficient because raw
            # chunks are also vector indexed; disabling the optional gleaning
            # pass removes a second hallucination surface and halves build cost.
            entity_extract_max_gleaning=0,
            addon_params={"language": language},
        )
        self._initialized = False
        self.available = True

    @staticmethod
    def _current_index_dir() -> str:
        pointer = Path("data/processed/lightrag_current.json")
        if pointer.exists():
            try:
                candidate = str(json.loads(pointer.read_text(encoding="utf-8"))["working_dir"])
                if Path(candidate).is_dir():
                    return candidate
            except (KeyError, OSError, ValueError, TypeError):
                pass
        return "data/processed/lightrag_index"

    def _start_loop_thread(self):
        import queue
        import threading
        self._call_q: queue.Queue = queue.Queue()
        self._result_q: queue.Queue = queue.Queue()

        def _loop_runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            while True:
                coro = self._call_q.get()
                if coro is None:
                    break
                try:
                    self._result_q.put(("ok", loop.run_until_complete(coro)))
                except Exception as e:
                    self._result_q.put(("err", e))
            loop.close()
        self._loop_thread = threading.Thread(target=_loop_runner, daemon=True)
        self._loop_thread.start()

    def _run(self, coro):
        if not hasattr(self, "_loop_thread"):
            self._start_loop_thread()
        self._call_q.put(coro)
        status, payload = self._result_q.get()
        if status == "err":
            raise payload
        return payload

    def _ensure_init(self) -> None:
        if not self._initialized:
            self._run(self.rag.initialize_storages())
            self._initialized = True

    def insert_chunks(self, chunks: list[str], batch_size: int = 50) -> None:
        self._ensure_init()
        for i in range(0, len(chunks), batch_size):
            self._run(self.rag.ainsert(chunks[i:i+batch_size]))

    def query(
        self,
        question: str,
        mode: str = DEFAULT_QUERY_MODE,
        *,
        only_need_context: bool = False,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
    ) -> str:
        self._ensure_init()
        query_options: dict[str, Any] = {
            "mode": mode,
            "top_k": self.DEFAULT_QUERY_TOP_K if top_k is None else top_k,
            "chunk_top_k": self.DEFAULT_QUERY_CHUNK_TOP_K if chunk_top_k is None else chunk_top_k,
            "enable_rerank": False,
            "only_need_context": only_need_context,
        }
        # Older lightrag builds have no query-level cosine cut-off; pass it
        # only where the installed version supports it.
        if "cosine_threshold" in {f.name for f in fields(self._QueryParam)}:
            query_options["cosine_threshold"] = self.DEFAULT_QUERY_COSINE_THRESHOLD
        param = self._QueryParam(**query_options)
        return self._run(self.rag.aquery(question, param=param))

    def query_case(
        self,
        *,
        patient_flags: list[str],
        surgery_flags: list[str],
        triggered_rule_ids: list[str],
        mode: str = "hybrid",
    ) -> str:
        """Retrieve evidence using de-identified derived features only."""
        question = (
            "请从结直肠癌吻合口漏知识库检索与下列规则和风险特征直接相关的证据。"
            "优先返回研究结论、适用人群、阈值和来源，不推断患者身份或临床概率。\n"
            f"触发规则：{', '.join(triggered_rule_ids) or '无'}\n"
            f"患者风险标签：{', '.join(patient_flags) or '无'}\n"
            f"手术风险标签：{', '.join(surgery_flags) or '无'}"
        )
        return self.query(question, mode=mode)

    def stats(self) -> dict:
        disk_bytes = 0
        if os.path.isdir(self.working_dir):
            for root, _dirs, files in os.walk(self.working_dir):
                for f in files:
                    disk_bytes += os.path.getsize(os.path.join(root, f))
        ents = rels = 0
        try:
            kg = self._run(self.rag.get_knowledge_graph())
            if hasattr(kg, "number_of_nodes"):
                ents = kg.number_of_nodes()
                rels = kg.number_of_edges()
            else:
                ents = len(getattr(kg, "nodes", []) or [])
                rels = len(getattr(kg, "edges", []) or [])
        except Exception:
            try:
                storage = getattr(self.rag, "graph_storage", None)
                g = getattr(storage, "_graph", None) or getattr(storage, "graph", None)
                if g is not None:
                    ents = g.number_of_nodes()
                    rels = g.number_of_edges()
            except Exception:
                pass
        if ents == 0 and rels == 0:
            graphml = os.path.join(self.working_dir, "graph_chunk_entity_relation.graphml")
            if os.path.isfile(graphml):
                try:
                    from xml.etree import ElementTree

                    root = ElementTree.parse(graphml).getroot()
                    namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
                    ents = len(root.findall(".//g:node", namespace))
                    rels = len(root.findall(".//g:edge", namespace))
                except Exception:
                    pass
        return {
            "working_dir": self.working_dir,
            "entities_extracted": ents,
            "relations_extracted": rels,
            "index_disk_kb": disk_bytes // 1024,
            "llm_provider": f"ark:{self.chat.model}",
            "embedding_provider": f"ark:{self.embed.model_field}",
            "embedding_dim": self.embed.embedding_dim,
        }
