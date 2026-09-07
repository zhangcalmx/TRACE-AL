"""Canonical, traceable evidence records used by the v2 evidence corpus."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

VerificationStatus = Literal[
    "identifier_present",
    "metadata_verified",
    "claim_verified",
    "full_text_verified",
    "needs_locator",
    "quarantined",
]


class EvidenceSourceRecord(BaseModel):
    source_id: str
    title: str = ""
    source_type: str
    pmid: str = ""
    doi: str = ""
    url: str = ""
    locator: str = ""
    isbn: str = ""
    edition: str = ""
    authors_or_editors: list[str] = Field(default_factory=list)
    publisher: str = ""
    publication_year: str = ""
    chapter: str = ""
    pages: str = ""
    verification_status: VerificationStatus
    identity_sources: list[str] = Field(default_factory=list)
    retraction_status: str = ""
    checked_on: str = ""
    notes: str = ""


class BookCatalogRecord(BaseModel):
    """Bibliographic metadata for a book that may support evidence claims."""

    book_id: str
    title: str
    edition: str
    authors_or_editors: list[str] = Field(default_factory=list)
    publisher: str
    publication_year: str
    isbn: str
    doi: str = ""
    reference_url: str
    metadata_status: Literal["verified", "partial", "conflict"]
    notes: str = ""


class BookEvidenceCandidateRecord(BaseModel):
    """A page-level book claim; only verified rows may enter the active corpus."""

    candidate_id: str
    legacy_chunk_id: str = ""
    book_id: str
    rule_id: str
    claim_summary: str
    chapter: str = ""
    pages: str = ""
    locator_detail: str = ""
    source_excerpt_summary: str = ""
    evidence_level: str = "C"
    language: str = "zh"
    severity: str = "medium"
    entity_a_flags: list[str] = Field(default_factory=list)
    entity_b_flags: list[str] = Field(default_factory=list)
    verified_against_source: bool = False
    reviewer: str = ""
    reviewed_at: str = ""
    rights_basis: Literal[
        "",
        "brief_scholarly_summary",
        "user_owned_copy",
        "licensed_access",
        "open_access",
    ] = ""
    status: Literal["pending_locator", "ready_for_review", "verified"]


class EvidenceClaimRecord(BaseModel):
    claim_id: str
    rule_id: str
    statement: str
    source_ids: list[str] = Field(min_length=1)
    evidence_level: str
    verification_status: VerificationStatus
    locator: str = ""
    support_excerpt: str = ""
    review_basis: str = ""
    directness: str = ""
    reviewer: str = ""
    reviewed_on: str = ""


class EvidenceChunkRecord(BaseModel):
    chunk_id: str
    claim_id: str
    rule_id: str
    title: str
    text: str
    source_ids: list[str] = Field(min_length=1)
    source_type: str
    evidence_level: str
    language: str = "zh"
    severity: str = "medium"
    entity_a_flags: list[str] = Field(default_factory=list)
    entity_b_flags: list[str] = Field(default_factory=list)
    verification_status: VerificationStatus
    locator: str = ""
    directness: str = ""
