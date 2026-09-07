"""EvidenceStore — reads evidence_chunks.jsonl for Screening tab."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import project_root
from .schemas import EvidenceChunk

PROJECT_ROOT = project_root()


@dataclass
class Evidence:
    chunk_id: str
    rule_id: str
    legacy_rule_id: str
    title: str
    text: str
    source_type: str
    level: str
    pmids: list[str]
    guidelines: list[str]
    textbooks: list[str]
    entity_a_flags: list[str]
    entity_b_flags: list[str]
    severity: str
    source_url: str
    source_locator: str
    verification_status: str
    source_ids: list[str]

    @property
    def source_label(self) -> str:
        parts = [*(f"PMID:{pmid}" for pmid in self.pmids), *self.guidelines, *self.textbooks]
        return "; ".join(parts) or self.source_locator or self.source_type

    def to_schema(self) -> EvidenceChunk:
        """Convert the compact seed representation into the public report schema."""
        return EvidenceChunk(
            evidence_id=self.chunk_id,
            title=self.title,
            source=self.source_label,
            source_type=self.source_type,
            evidence_level=self.level,
            evidence_relation=self.rule_id,
            source_locator=self.source_locator or self.source_label,
            citation_hint=self.source_locator or self.source_label,
            pmid=self.pmids[0] if self.pmids else "",
            verification_status=self.verification_status,
            entities_a=self.entity_a_flags,
            entities_b=self.entity_b_flags,
            risk_types=[self.rule_id] if self.rule_id else [],
            text=self.text,
            url=self.source_url,
            weight={"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}.get(self.level, 1),
        )


class EvidenceStore:
    """Loads and filters evidence_chunks.jsonl."""

    def __init__(self, path: Path | None = None):
        self._path = path or PROJECT_ROOT / "data" / "seed" / "evidence_chunks.jsonl"
        self._chunks: list[Evidence] = []
        self._loaded = False
        self._rule_id_map = self._load_rule_id_map()

    @staticmethod
    def _load_rule_id_map() -> dict[str, str]:
        rules_path = PROJECT_ROOT / "configs" / "rules.yaml"
        if not rules_path.exists():
            return {}
        payload = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
        return {
            str(index): str(rule.get("id", index))
            for index, rule in enumerate(payload.get("rules", []), start=1)
        }

    def _load(self) -> None:
        if self._loaded:
            return
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                legacy_rule_id = str(rec.get("rule_id", ""))
                self._chunks.append(Evidence(
                    chunk_id=rec.get("chunk_id", ""),
                    rule_id=self._rule_id_map.get(legacy_rule_id, legacy_rule_id),
                    legacy_rule_id=legacy_rule_id,
                    title=rec.get("title", ""),
                    text=rec.get("content", rec.get("text", "")),
                    source_type=rec.get("source_type", ""),
                    level=rec.get("level", "C"),
                    pmids=rec.get("pmids", []),
                    guidelines=rec.get("guidelines", []),
                    textbooks=rec.get("textbooks", []),
                    entity_a_flags=rec.get("entity_a_flags", []),
                    entity_b_flags=rec.get("entity_b_flags", []),
                    severity=rec.get("severity", "medium"),
                    source_url=rec.get("source_url", ""),
                    source_locator=rec.get("source_locator", ""),
                    verification_status=rec.get("verification_status", ""),
                    source_ids=rec.get("source_ids", []),
                ))
        self._loaded = True

    def all(self) -> list[Evidence]:
        self._load()
        return list(self._chunks)

    def count(self) -> int:
        self._load()
        return len(self._chunks)

    def filter_by_rule(self, rule_id: str) -> list[Evidence]:
        self._load()
        return [c for c in self._chunks if c.rule_id == rule_id]

    def general(self) -> list[Evidence]:
        self._load()
        return [c for c in self._chunks if c.rule_id == "all"]

    def filter_by_flags(self, a_flags: list[str] | None = None,
                        b_flags: list[str] | None = None) -> list[Evidence]:
        self._load()
        results = list(self._chunks)
        if a_flags:
            results = [c for c in results if any(f in c.entity_a_flags for f in a_flags)]
        if b_flags:
            results = [c for c in results if any(f in c.entity_b_flags for f in b_flags)]
        return results

    def filter_by_level(self, levels: list[str] | None = None) -> list[Evidence]:
        self._load()
        if not levels:
            return list(self._chunks)
        return [c for c in self._chunks if c.level in levels]

    def corpus_fingerprints(self) -> dict[str, str]:
        """Return separate content and metadata hashes for index freshness.

        LightRAG currently indexes only evidence text. Metadata-only changes
        therefore refresh the deterministic sidecar without forcing a costly
        graph rebuild, while content changes invalidate the graph index.
        """
        self._load()
        content_rows = [
            {"chunk_id": row.chunk_id, "title": row.title, "text": row.text}
            for row in sorted(self._chunks, key=lambda item: item.chunk_id)
        ]
        metadata_rows = [
            {
                "chunk_id": row.chunk_id,
                "rule_id": row.rule_id,
                "source_type": row.source_type,
                "level": row.level,
                "pmids": row.pmids,
                "guidelines": row.guidelines,
                "textbooks": row.textbooks,
                "entity_a_flags": row.entity_a_flags,
                "entity_b_flags": row.entity_b_flags,
                "severity": row.severity,
                "source_url": row.source_url,
                "source_locator": row.source_locator,
                "verification_status": row.verification_status,
                "source_ids": row.source_ids,
            }
            for row in sorted(self._chunks, key=lambda item: item.chunk_id)
        ]

        def digest(value: Any) -> str:
            raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            return hashlib.sha256(raw).hexdigest()

        return {
            "content_sha256": digest(content_rows),
            "metadata_sha256": digest(metadata_rows),
        }
