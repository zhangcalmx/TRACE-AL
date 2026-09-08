from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.UNKNOWN: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
}


class EntityA(BaseModel):
    """Entity A registry record (e.g., Food / Herb / Gene / Microbiota).

    Rename `EntityA` to the domain-specific name (Food, Herb, Variant, Microbe, Symptom, Supplement).
    Extend fields as needed but keep `flags` (list of mechanism tags) — the rule engine depends on it.
    """

    entity_a_id: str
    primary_name: str = ""
    aliases: list[str] = Field(default_factory=list)
    category: str = ""
    active_mechanisms: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class EntityB(BaseModel):
    """Entity B registry record (e.g., Drug)."""

    entity_b_id: str
    generic_name: str
    aliases: list[str] = Field(default_factory=list)
    drug_class: str = ""
    flags: list[str] = Field(default_factory=list)
    metabolic_pathways: list[str] = Field(default_factory=list)
    transporter_substrates: list[str] = Field(default_factory=list)
    narrow_therapeutic_index: bool = False
    label_source: str = ""


class PatientFactors(BaseModel):
    factors: list[str] = Field(default_factory=list)


class InteractionCase(BaseModel):
    case_id: str
    entity_a: str
    entity_b: str
    patient_factors: list[str] = Field(default_factory=list)
    gold_risk_level: RiskLevel | None = None
    risk_type: list[str] = Field(default_factory=list)
    notes: str = ""


class PatientProfileInput(BaseModel):
    """De-identified patient values used to derive rule flags.

    ``None`` always means unknown/not assessed and must never be interpreted as
    normal. These are case values, not rows from the feature registry.
    """

    age_years: int | None = Field(default=None, ge=18, le=120)
    sex: Literal["male", "female"] | None = None
    bmi: float | None = Field(default=None, ge=10, le=80)
    asa_grade: Literal["I", "II", "III", "IV", "V"] | None = None
    ecog_status: int | None = Field(default=None, ge=0, le=4)
    diabetes_status: Literal["no", "type1", "type2"] | None = None
    hba1c_percent: float | None = Field(default=None, ge=3, le=20)
    albumin_g_l: float | None = Field(default=None, ge=10, le=70)
    prealbumin_mg_l: float | None = Field(default=None, ge=20, le=700)
    nlr: float | None = Field(default=None, ge=0, le=100)
    pni: float | None = Field(default=None, ge=0, le=100)
    nrs2002_score: int | None = Field(default=None, ge=0, le=7)
    cancer_site: Literal["colon", "rectum", "lower_rectum"] | None = None
    bowel_obstruction: bool | None = None
    bowel_perforation: bool | None = None
    # Extension collection fields (collect_only: no trigger flags, no standalone alert).
    preop_hemoglobin_g_l: float | None = Field(default=None, ge=40, le=200)
    smoking_status: Literal["never", "former", "current"] | None = None
    smoking_pack_years: float | None = Field(default=None, ge=0, le=300)
    clinical_frailty_scale: int | None = Field(default=None, ge=1, le=9)
    preop_crp_mg_l: float | None = Field(default=None, ge=0, le=500)
    coronary_artery_disease: bool | None = None
    chronic_lung_disease: Literal["no", "copd", "asthma", "other"] | None = None
    renal_dysfunction: Literal["no", "ckd_stage3", "ckd_stage4", "ckd_stage5_dialysis"] | None = None
    tumor_distance_from_anal_verge_cm: float | None = Field(default=None, ge=0, le=40)
    clinical_t_stage: Literal["cTis", "cT1", "cT2", "cT3", "cT4a", "cT4b"] | None = None
    neoadjuvant_treatment_type: (
        Literal["none", "radiotherapy_only", "chemotherapy_only", "chemoradiotherapy", "total_neoadjuvant_therapy", "other"]
        | None
    ) = None
    previous_abdominal_or_pelvic_surgery: Literal["no", "abdominal", "pelvic", "both"] | None = None
    preop_sepsis_status: Literal["none", "sepsis", "septic_shock"] | None = None
    ct_visceral_fat_area_cm2: float | None = Field(default=None, ge=0, le=1000)


class SurgeryProfileInput(BaseModel):
    """De-identified operative/anastomotic values used to derive rule flags."""

    urgency: Literal["elective", "emergency"] | None = None
    multivisceral_resection: bool | None = None
    duration_minutes: int | None = Field(default=None, ge=15, le=1440)
    blood_loss_ml: float | None = Field(default=None, ge=0, le=20000)
    transfusion_units: float | None = Field(default=None, ge=0, le=100)
    anastomosis_site: Literal["none", "ileocolic", "colocolic", "colorectal", "ultra_low", "coloanal"] | None = None
    anastomosis_height_cm: float | None = Field(default=None, ge=0, le=30)
    doughnut_integrity: Literal["not_assessed", "complete", "incomplete"] | None = None
    air_leak_test: Literal["not_performed", "negative", "positive_repaired", "positive_persistent"] | None = None
    anastomotic_defect: bool | None = None
    lowest_map_mmhg: float | None = Field(default=None, ge=20, le=160)
    map_below_65_minutes: int | None = Field(default=None, ge=0, le=1440)
    icg_used: bool | None = None
    icg_perfusion_result: Literal["adequate", "borderline", "inadequate"] | None = None
    icg_action_after_inadequate_perfusion: (
        Literal[
            "not_applicable",
            "transection_line_changed",
            "anastomosis_reconstructed",
            "no_corrective_action",
            "unknown",
        ]
        | None
    ) = None
    peritoneal_contamination: Literal["none", "mild", "severe"] | None = None
    # Extension collection fields (collect_only: no trigger flags, no standalone alert).
    planned_surgical_approach: Literal["open", "laparoscopic", "robotic", "transanal", "hybrid"] | None = None
    conversion_status: Literal["none", "to_open", "to_other_mis"] | None = None
    procedure_name: (
        Literal[
            "right_hemicolectomy",
            "transverse_colectomy",
            "left_hemicolectomy",
            "sigmoidectomy",
            "anterior_resection",
            "low_anterior_resection",
            "abdominoperineal_resection",
            "total_colectomy",
            "proctocolectomy",
            "other",
        ]
        | None
    ) = None
    rectal_transection_stapler_firing_count: int | None = Field(default=None, ge=0, le=10)
    anastomotic_tension: Literal["not_assessed", "none", "mild", "moderate", "severe"] | None = None
    primary_anastomosis_performed: bool | None = None
    diverting_stoma_type: Literal["none", "loop_ileostomy", "loop_colostomy", "other"] | None = None
    vasopressor_used: bool | None = None
    norepinephrine_equivalent_max_ug_kg_min: float | None = Field(default=None, ge=0, le=10)
    lowest_core_temperature_c: float | None = Field(default=None, ge=30, le=42)
    net_fluid_balance_ml: float | None = Field(default=None, ge=-10000, le=30000)
    mechanical_bowel_prep_status: Literal["none", "completed", "incomplete", "contraindicated", "not_applicable"] | None = None
    oral_antibiotic_bowel_prep_status: Literal["none", "completed", "incomplete", "contraindicated", "not_applicable"] | None = None
    iv_antibiotic_prophylaxis_given: bool | None = None
    iv_antibiotic_prophylaxis_minutes_before_incision: int | None = Field(default=None, ge=-240, le=240)
    iv_antibiotic_redosing_status: Literal["not_indicated", "appropriate", "missed_or_late"] | None = None
    glucose_near_anastomosis_mmol_l: float | None = Field(default=None, ge=1, le=40)
    intraoperative_technical_adverse_event: Literal["none", "stapler_misfire", "tissue_tear", "ischemia", "unplanned_resection", "other"] | None = None
    technical_adverse_event_corrected: Literal["yes", "no", "not_applicable"] | None = None
    lead_surgeon_independent_colorectal_years: int | None = Field(default=None, ge=0, le=60)
    lead_surgeon_rectal_resections_12m: int | None = Field(default=None, ge=0, le=300)
    lead_surgeon_cumulative_robotic_rectal_cases: int | None = Field(default=None, ge=0, le=1000)


class ClinicalCaseInput(BaseModel):
    """A complete de-identified case submitted to the rule pipeline."""

    case_id: str = Field(default="case", min_length=1, max_length=100)
    patient: PatientProfileInput = Field(default_factory=PatientProfileInput)
    surgery: SurgeryProfileInput = Field(default_factory=SurgeryProfileInput)


class DerivedCaseFeatures(BaseModel):
    patient_flags: list[str] = Field(default_factory=list)
    surgery_flags: list[str] = Field(default_factory=list)
    trace: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    input_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    assessment_domains: dict[str, bool] = Field(default_factory=dict)
    missing_domains: list[str] = Field(default_factory=list)


class EvidenceChunk(BaseModel):
    evidence_id: str
    title: str
    source: str
    source_type: str
    evidence_level: str = ""
    evidence_relation: str = ""
    source_locator: str = ""
    citation_hint: str = ""
    pmid: str = ""
    doi: str = ""
    verification_status: str = ""
    source_quality_rank: str = ""
    entities_a: list[str] = Field(default_factory=list)
    entities_b: list[str] = Field(default_factory=list)
    risk_types: list[str] = Field(default_factory=list)
    text: str
    url: str = ""
    weight: int = 1


class RiskSignal(BaseModel):
    rule_id: str
    risk_level: RiskLevel
    risk_type: str
    signal_class: Literal["safety_guardrail", "risk_marker", "context_review"] = "risk_marker"
    rationale: str
    recommendation: str
    matched_terms: dict[str, list[str]] = Field(default_factory=dict)


class RiskAssessment(BaseModel):
    risk_level: RiskLevel
    risk_types: list[str] = Field(default_factory=list)
    signals: list[RiskSignal] = Field(default_factory=list)
    policy_version: str = "legacy"
    primary_alert: bool = False
    risk_score: int = Field(default=0, ge=0)
    primary_alert_threshold: int = Field(default=4, ge=1)
    safety_guardrail_present: bool = False
    requires_immediate_action: bool = False
    coverage_status: Literal["complete", "partial", "insufficient"] = "insufficient"
    input_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    # Deprecated compatibility aliases. They represent input/rule coverage,
    # never a predicted probability or calibrated clinical confidence.
    confidence: str = "insufficient"
    confidence_score: float = 0.0


class SafetyReport(BaseModel):
    case_id: str
    entity_a: str
    entity_b: str
    patient_factors: list[str] = Field(default_factory=list)
    risk_assessment: RiskAssessment
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    report_text: str
    context_char_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
