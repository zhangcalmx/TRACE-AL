"""Deterministic evidence retrieval used as an offline-safe RAG fallback."""
from __future__ import annotations

import re

from .evidence_store import Evidence, EvidenceStore

LEVEL_WEIGHT = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


class NaiveRAG:
    def __init__(self, evidence_store: EvidenceStore | None = None) -> None:
        self.store = evidence_store or EvidenceStore()

    @staticmethod
    def _deduplicate(items: list[Evidence]) -> list[Evidence]:
        seen: set[str] = set()
        result: list[Evidence] = []
        for item in items:
            if item.chunk_id not in seen:
                seen.add(item.chunk_id)
                result.append(item)
        return result

    def retrieve(
        self,
        *,
        rule_ids: list[str] | tuple[str, ...] = (),
        entity_a_flags: list[str] | tuple[str, ...] = (),
        entity_b_flags: list[str] | tuple[str, ...] = (),
        top_k: int = 5,
    ) -> list[Evidence]:
        active_rules = {rule_id for rule_id in rule_ids if rule_id}
        active_a = set(entity_a_flags)
        active_b = set(entity_b_flags)
        scored: list[tuple[float, Evidence]] = []
        for item in self.store.all():
            score = 0.0
            if item.rule_id in active_rules:
                score += 100
            elif item.rule_id == "all":
                score += 5
            matched_a = active_a & set(item.entity_a_flags)
            matched_b = active_b & set(item.entity_b_flags)
            score += 12 * len(matched_a) + 12 * len(matched_b)
            if score <= 0:
                continue
            score += LEVEL_WEIGHT.get(item.level, 0)
            if item.severity == "high":
                score += 1
            scored.append((score, item))
        scored.sort(key=lambda pair: (pair[0], pair[1].chunk_id), reverse=True)
        return [item for _, item in scored[: max(0, top_k)]]

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        lowered = query.casefold()
        words = set(re.findall(r"[a-z0-9_]{2,}", lowered))
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
        words.update(chinese[i : i + 2] for i in range(max(0, len(chinese) - 1)))
        return {term for term in words if term}

    def query(self, query: str, top_k: int = 6) -> list[Evidence]:
        terms = self._query_terms(query)
        scored: list[tuple[float, Evidence]] = []
        for item in self.store.all():
            haystack = f"{item.title} {item.text} {' '.join(item.entity_a_flags)} {' '.join(item.entity_b_flags)}".casefold()
            overlap = sum(1 for term in terms if term in haystack)
            if overlap:
                scored.append((overlap * 10 + LEVEL_WEIGHT.get(item.level, 0), item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[: max(0, top_k)]]
