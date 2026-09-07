"""normalize — load registries from CSV."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any

from .paths import project_root

PROJECT_ROOT = project_root()

_cache: dict[str, Any] = {}


@dataclass
class EntityARow:
    entity_a_id: str
    primary_name: str
    aliases: str
    category: str
    data_type: str
    unit: str
    allowed_values: str
    flags: list[str] = field(default_factory=list)
    risk_direction: str = ""
    evidence_sources: str = ""


@dataclass
class EntityBRow:
    entity_b_id: str
    primary_name: str
    aliases: str
    category: str
    data_type: str
    unit: str
    allowed_values: str
    flags: list[str] = field(default_factory=list)
    risk_direction: str = ""
    evidence_sources: str = ""

    @property
    def generic_name(self) -> str:
        """Backward-compatible name used by the Streamlit scaffold."""
        return self.primary_name


def _split_aliases(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace(",", ";").split(";") if item.strip()]


def _find_row(query: str, rows: dict[str, Any], *, id_attr: str) -> Any:
    normalized = query.strip().casefold()
    if not normalized:
        raise ValueError("Entity query must not be empty")
    if query in rows:
        return rows[query]
    for row in rows.values():
        candidates = [getattr(row, id_attr), row.primary_name, *_split_aliases(row.aliases)]
        if normalized in {candidate.casefold() for candidate in candidates if candidate}:
            return row
    raise KeyError(f"Unknown entity: {query}")


def clear_cache() -> None:
    _cache.clear()


def load_entity_a_registry() -> dict[str, EntityARow]:
    key = "entity_a"
    if key in _cache:
        return _cache[key]
    path = PROJECT_ROOT / "data" / "seed" / "entities_a.csv"
    result: dict[str, EntityARow] = {}
    if not path.exists():
        _cache[key] = result
        return result
    import io
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # CSV有逗号在flags列被引号包裹，用csv模块正确处理
    reader = csv.DictReader(io.StringIO(raw))
    for row in reader:
        flags_raw = row.get("flags", "")
        flags = [f.strip() for f in flags_raw.split(",") if f.strip().strip('"')] if flags_raw else []
        result[row["patient-host-profile_id"]] = EntityARow(
            entity_a_id=row["patient-host-profile_id"],
            primary_name=row.get("primary_name", ""),
            aliases=row.get("aliases", ""),
            category=row.get("category", ""),
            data_type=row.get("data_type", ""),
            unit=row.get("unit", ""),
            allowed_values=row.get("allowed_values", ""),
            flags=flags,
            risk_direction=row.get("risk_direction", ""),
            evidence_sources=row.get("evidence_sources", ""),
        )
    _cache[key] = result
    return result


def load_entity_b_registry() -> dict[str, EntityBRow]:
    key = "entity_b"
    if key in _cache:
        return _cache[key]
    path = PROJECT_ROOT / "data" / "seed" / "entities_b.csv"
    result: dict[str, EntityBRow] = {}
    if not path.exists():
        _cache[key] = result
        return result
    import io
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    reader = csv.DictReader(io.StringIO(raw))
    for row in reader:
        flags_raw = row.get("flags", "")
        flags = [f.strip() for f in flags_raw.split(",") if f.strip().strip('"')] if flags_raw else []
        result[row["surgery-anastomosis-profile_id"]] = EntityBRow(
            entity_b_id=row["surgery-anastomosis-profile_id"],
            primary_name=row.get("primary_name", ""),
            aliases=row.get("aliases", ""),
            category=row.get("category", ""),
            data_type=row.get("data_type", ""),
            unit=row.get("unit", ""),
            allowed_values=row.get("allowed_values", ""),
            flags=flags,
            risk_direction=row.get("risk_direction", ""),
            evidence_sources=row.get("evidence_sources", ""),
        )
    _cache[key] = result
    return result


def find_entity_a(query: str) -> EntityARow:
    """Resolve an Entity A record by ID, primary name, or alias."""
    return _find_row(query, load_entity_a_registry(), id_attr="entity_a_id")


def find_entity_b(query: str) -> EntityBRow:
    """Resolve an Entity B record by ID, primary name, or alias."""
    return _find_row(query, load_entity_b_registry(), id_attr="entity_b_id")
