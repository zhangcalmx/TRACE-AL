"""Convert typed patient and surgery values into auditable rule flags."""

from __future__ import annotations

from .schemas import ClinicalCaseInput, DerivedCaseFeatures


def derive_case_features(case: ClinicalCaseInput) -> DerivedCaseFeatures:
    patient_flags: set[str] = set()
    surgery_flags: set[str] = set()
    trace: dict[str, str] = {}
    warnings: list[str] = []

    def patient(flag: str, reason: str) -> None:
        patient_flags.add(flag)
        trace[flag] = reason

    def surgery(flag: str, reason: str) -> None:
        surgery_flags.add(flag)
        trace[flag] = reason

    p = case.patient
    if p.age_years is not None:
        if p.age_years >= 65:
            patient("age_over_65", f"年龄 {p.age_years} 岁 ≥ 65")
        if p.age_years >= 75:
            patient("age_over_75", f"年龄 {p.age_years} 岁 ≥ 75")
    if p.sex == "male":
        patient("male_gender", "性别为男性")
    if p.bmi is not None:
        if p.bmi < 18.5:
            patient("bmi_under_18_5", f"BMI {p.bmi:g} < 18.5")
        if p.bmi >= 25:
            patient("bmi_over_25", f"BMI {p.bmi:g} ≥ 25")
        if p.bmi >= 30:
            patient("bmi_over_30", f"BMI {p.bmi:g} ≥ 30")
    if p.asa_grade in {"III", "IV", "V"}:
        patient("asa_ge_III", f"ASA 分级为 {p.asa_grade}")
    if p.ecog_status is not None and p.ecog_status >= 2:
        patient("ecog_ge_2", f"ECOG {p.ecog_status} ≥ 2")
    if p.diabetes_status in {"type1", "type2"}:
        patient("diabetes", f"糖尿病类型为 {p.diabetes_status}")
        if p.hba1c_percent is not None and p.hba1c_percent > 7:
            patient("diabetes_poor_control", f"HbA1c {p.hba1c_percent:g}% > 7%")
    elif p.hba1c_percent is not None:
        warnings.append("已录入 HbA1c，但糖尿病状态不是1型/2型；未自动生成糖尿病控制不佳标签。")
    if p.albumin_g_l is not None:
        if p.albumin_g_l < 35:
            patient("albumin_low_35", f"白蛋白 {p.albumin_g_l:g} g/L < 35")
        if p.albumin_g_l < 30:
            patient("albumin_low_30", f"白蛋白 {p.albumin_g_l:g} g/L < 30")
            patient("severe_malnutrition", "白蛋白 < 30 g/L，满足严重营养不良代理条件")
    if p.prealbumin_mg_l is not None and p.prealbumin_mg_l < 200:
        patient("prealbumin_low_200", f"前白蛋白 {p.prealbumin_mg_l:g} mg/L < 200")
    if p.nlr is not None and p.nlr > 5:
        patient("nlr_elevated_over_5", f"NLR {p.nlr:g} > 5")
    if p.pni is not None:
        if p.pni < 45:
            patient("pni_low_45", f"PNI {p.pni:g} < 45")
        if p.pni < 40:
            patient("pni_low_40", f"PNI {p.pni:g} < 40")
            patient("severe_malnutrition", "PNI < 40，满足严重营养不良代理条件")
    if p.nrs2002_score is not None:
        if p.nrs2002_score >= 3:
            patient("malnutrition_risk", f"NRS-2002 {p.nrs2002_score} ≥ 3")
        if p.nrs2002_score >= 5:
            patient("severe_malnutrition", f"NRS-2002 {p.nrs2002_score} ≥ 5")
    if p.cancer_site == "rectum":
        patient("rectal_cancer", "肿瘤部位为直肠")
    elif p.cancer_site == "lower_rectum":
        patient("rectal_cancer", "肿瘤部位为低位直肠")
        patient("lower_rectal_cancer", "肿瘤部位为低位直肠")
    if p.bowel_obstruction is True:
        patient("bowel_obstruction", "存在肠梗阻")
    if p.bowel_perforation is True:
        patient("bowel_perforation", "存在肠穿孔")

    s = case.surgery
    if s.urgency == "emergency":
        surgery("emergency_surgery", "急诊手术")
    if s.multivisceral_resection is True:
        surgery("multivisceral_resection", "实施联合脏器切除")
    if s.duration_minutes is not None:
        if s.duration_minutes > 180:
            surgery("surgery_duration_over_3h", f"手术时长 {s.duration_minutes} 分钟 > 180")
        if s.duration_minutes > 240:
            surgery("surgery_duration_over_4h", f"手术时长 {s.duration_minutes} 分钟 > 240")
        if s.duration_minutes > 300:
            surgery("surgery_duration_over_5h", f"手术时长 {s.duration_minutes} 分钟 > 300")
    if s.blood_loss_ml is not None:
        if s.blood_loss_ml > 300:
            surgery("blood_loss_over_300ml", f"术中出血 {s.blood_loss_ml:g} ml > 300")
        if s.blood_loss_ml > 500:
            surgery("blood_loss_over_500ml", f"术中出血 {s.blood_loss_ml:g} ml > 500")
        if s.blood_loss_ml > 1000:
            surgery("blood_loss_over_1000ml", f"术中出血 {s.blood_loss_ml:g} ml > 1000")
    if s.transfusion_units is not None and s.transfusion_units > 0:
        surgery("blood_transfusion_intraop", f"术中输血 {s.transfusion_units:g} U")

    if s.anastomosis_site in {"colorectal", "ultra_low", "coloanal"}:
        surgery("colorectal_anastomosis", f"吻合部位为 {s.anastomosis_site}")
    if s.anastomosis_site == "ultra_low":
        surgery("ultra_low_anastomosis", "超低位吻合")
    if s.anastomosis_site == "coloanal":
        surgery("coloanal_anastomosis", "结肠肛管吻合")
    if s.anastomosis_height_cm is not None:
        if s.anastomosis_height_cm <= 5:
            surgery(
                "anastomosis_at_or_below_5cm",
                f"吻合口距肛缘 {s.anastomosis_height_cm:g} cm ≤ 5",
            )
        if s.anastomosis_height_cm < 6:
            surgery("anastomosis_under_6cm", f"吻合口距肛缘 {s.anastomosis_height_cm:g} cm < 6")
        elif s.anastomosis_height_cm <= 10:
            surgery("anastomosis_6_10cm", f"吻合口距肛缘 {s.anastomosis_height_cm:g} cm，处于 6–10 cm")
    if s.doughnut_integrity == "incomplete":
        surgery("doughnut_incomplete", "吻合圈不完整")
    if s.air_leak_test in {"positive_repaired", "positive_persistent"}:
        surgery("air_leak_test_positive", f"充气试验结果为 {s.air_leak_test}")
        surgery("air_leak_needed_repair", "充气试验阳性，需要修补或重新吻合")
        if s.air_leak_test == "positive_repaired":
            surgery("air_leak_test_repaired", "充气试验阳性，处理后复测已转阴")
        else:
            surgery("air_leak_test_persistent", "充气试验处理后仍持续阳性")
    if s.anastomotic_defect is True:
        surgery("anastomotic_defect", "术中发现吻合口缺损")
    if s.lowest_map_mmhg is not None:
        if s.lowest_map_mmhg < 65:
            surgery("map_below_65", f"最低 MAP {s.lowest_map_mmhg:g} mmHg < 65")
        if s.lowest_map_mmhg < 55:
            surgery("map_below_55", f"最低 MAP {s.lowest_map_mmhg:g} mmHg < 55")
    if (
        s.lowest_map_mmhg is not None
        and s.lowest_map_mmhg < 65
        and s.map_below_65_minutes is not None
        and s.map_below_65_minutes > 30
    ):
        surgery("hypotension_over_30min", f"MAP < 65 mmHg 持续 {s.map_below_65_minutes} 分钟 > 30")
    if s.icg_used is True:
        if s.icg_perfusion_result == "inadequate":
            surgery("icg_perfusion_inadequate", "ICG 显示灌注不足")
            if s.icg_action_after_inadequate_perfusion == "no_corrective_action":
                surgery("icg_no_corrective_action", "ICG 灌注不足后未记录切缘调整或重建吻合")
            elif s.icg_action_after_inadequate_perfusion in {None, "unknown"}:
                warnings.append("ICG 灌注不足，但后续处置未知；不自动触发‘未矫正灌注不良’规则。")
        elif s.icg_perfusion_result == "borderline":
            surgery("icg_perfusion_borderline", "ICG 显示边界灌注")
    elif s.icg_perfusion_result is not None:
        warnings.append("已录入 ICG 灌注结果，但 ICG 使用状态不是“是”；未生成 ICG 风险标签。")
    if s.peritoneal_contamination == "severe":
        surgery("peritoneal_contamination_severe", "存在严重腹腔污染")

    assessment_domains = {
        "tumor_site": p.cancer_site is not None,
        "metabolic": p.diabetes_status is not None and (p.diabetes_status == "no" or p.hba1c_percent is not None),
        "nutrition": any(value is not None for value in (p.albumin_g_l, p.prealbumin_mg_l, p.nrs2002_score)),
        "performance": p.asa_grade is not None and p.ecog_status is not None,
        "inflammation_nutrition": p.nlr is not None and p.pni is not None,
        "anastomosis": s.anastomosis_site is not None
        and (s.anastomosis_site == "none" or s.anastomosis_height_cm is not None),
        "technical_integrity": any(
            value is not None for value in (s.doughnut_integrity, s.air_leak_test, s.anastomotic_defect)
        ),
        "hemodynamics": s.lowest_map_mmhg is not None and s.map_below_65_minutes is not None,
        "operative_burden": all(
            value is not None for value in (s.duration_minutes, s.blood_loss_ml, s.transfusion_units)
        ),
        "icg_perfusion": s.icg_used is not None
        and (
            s.icg_used is False
            or (
                s.icg_perfusion_result is not None
                and (
                    s.icg_perfusion_result != "inadequate"
                    or s.icg_action_after_inadequate_perfusion not in {None, "unknown"}
                )
            )
        ),
    }
    input_completeness = sum(assessment_domains.values()) / len(assessment_domains)
    missing_domains = [name for name, complete in assessment_domains.items() if not complete]

    return DerivedCaseFeatures(
        patient_flags=sorted(patient_flags),
        surgery_flags=sorted(surgery_flags),
        trace=trace,
        warnings=warnings,
        input_completeness=input_completeness,
        assessment_domains=assessment_domains,
        missing_domains=missing_domains,
    )
