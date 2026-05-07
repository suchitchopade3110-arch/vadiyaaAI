"""
VaidyaAI — Clinical PDF Report Generator
Uses reportlab to produce clinical-grade downloadable reports.
"""
import io
from datetime import timezone, datetime
UTC = timezone.utc

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.disclaimer import MEDICAL_DISCLAIMER


PRIMARY = colors.HexColor("#1B5E20")
SECONDARY = colors.HexColor("#388E3C")
WARN = colors.HexColor("#E65100")
DANGER = colors.HexColor("#B71C1C")
LIGHT_BG = colors.HexColor("#F1F8E9")
BORDER = colors.HexColor("#C8E6C9")


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "VTitle",
        fontSize=22,
        fontName="Helvetica-Bold",
        textColor=PRIMARY,
        alignment=TA_CENTER,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "VSubtitle",
        fontSize=10,
        fontName="Helvetica",
        textColor=SECONDARY,
        alignment=TA_CENTER,
        spaceAfter=16,
    ))
    styles.add(ParagraphStyle(
        "VSectionHead",
        fontSize=13,
        fontName="Helvetica-Bold",
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "VBody",
        fontSize=9,
        fontName="Helvetica",
        textColor=colors.HexColor("#212121"),
        spaceAfter=4,
        leading=14,
    ))
    styles.add(ParagraphStyle(
        "VDisclaimer",
        fontSize=7.5,
        fontName="Helvetica-Oblique",
        textColor=WARN,
        alignment=TA_CENTER,
        spaceAfter=4,
    ))
    return styles


def _flag_color(flag: str):
    normalized = (flag or "").upper()
    if normalized in ("HIGH", "LOW", "CRITICAL"):
        return DANGER
    if normalized in ("BORDERLINE", "WATCH"):
        return WARN
    return PRIMARY


def generate_report_pdf(report_data: dict) -> bytes:
    """
    Accepts unified VaidyaAI report_data dict and returns PDF bytes.

    Expected keys are optional except report_type:
    report_type, patient_id, risk_score, risk_label, lab_values, anomalies,
    shap_factors, explanation, citations, confidence_score, uncertainty_flag,
    generated_at.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    styles = _styles()
    story = []

    story.append(Paragraph("VaidyaAI", styles["VTitle"]))
    story.append(Paragraph("Medical Intelligence Platform - Clinical Report", styles["VSubtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY))
    story.append(Spacer(1, 8))

    generated = report_data.get(
        "generated_at",
        datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )
    meta_rows = [
        ["Report Type", report_data.get("report_type", "lab").upper()],
        ["Patient ID", report_data.get("patient_id", "Anonymous")],
        ["Generated At", generated],
        ["Confidence", f"{report_data.get('confidence_score', 0):.1f}%"],
    ]
    meta_table = Table(meta_rows, colWidths=[45 * mm, 120 * mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
        ("TEXTCOLOR", (0, 0), (0, -1), PRIMARY),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    risk = report_data.get("risk_score", 0)
    label = report_data.get("risk_label", "Unknown")
    risk_color = DANGER if risk >= 70 else (WARN if risk >= 35 else PRIMARY)
    story.append(Paragraph("Risk Assessment", styles["VSectionHead"]))
    risk_table = Table(
        [[f"Risk Score: {risk:.0f} / 100", f"Category: {label.replace('_', ' ').title()}"]],
        colWidths=[82 * mm, 83 * mm],
    )
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("TEXTCOLOR", (0, 0), (-1, -1), risk_color),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 10))

    lab_values = report_data.get("lab_values", [])
    if lab_values:
        story.append(Paragraph("Lab Values", styles["VSectionHead"]))
        rows = [["Test", "Result", "Reference", "Flag"]]
        for lab_value in lab_values:
            flag = lab_value.get("flag", "NORMAL")
            flag_color = _flag_color(flag)
            rows.append([
                lab_value.get("field", ""),
                f"{lab_value.get('value', '')} {lab_value.get('unit', '')}".strip(),
                lab_value.get("reference", "-"),
                Paragraph(
                    f'<font color="#{flag_color.hexval()[2:]}"><b>{flag}</b></font>',
                    styles["VBody"],
                ),
            ])
        lab_table = Table(rows, colWidths=[55 * mm, 40 * mm, 50 * mm, 20 * mm])
        lab_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(lab_table)
        story.append(Spacer(1, 10))

    anomalies = report_data.get("anomalies", [])
    if anomalies:
        story.append(Paragraph("Anomalies Detected", styles["VSectionHead"]))
        for anomaly in anomalies:
            text = (
                f"<b>{anomaly.get('field', anomaly.get('parameter', ''))}</b>: "
                f"{anomaly.get('value', '')} {anomaly.get('unit', '')} "
                f"(ref: {anomaly.get('reference', anomaly.get('reference_range', '-'))}) - "
                f"{anomaly.get('explanation', anomaly.get('severity', ''))}"
            )
            story.append(Paragraph(text, styles["VBody"]))
        story.append(Spacer(1, 8))

    shap = report_data.get("shap_factors", {})
    if shap:
        story.append(Paragraph("SHAP Feature Importance", styles["VSectionHead"]))
        shap_rows = [["Feature", "Importance"]]
        for feature, value in sorted(shap.items(), key=lambda item: -abs(float(item[1]))):
            shap_rows.append([str(feature), f"{float(value):.3f}"])
        shap_table = Table(shap_rows, colWidths=[100 * mm, 65 * mm])
        shap_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), SECONDARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(shap_table)
        story.append(Spacer(1, 10))

    explanation = report_data.get("explanation", "")
    if explanation:
        story.append(Paragraph("AI Clinical Interpretation", styles["VSectionHead"]))
        story.append(Paragraph(str(explanation), styles["VBody"]))
        story.append(Spacer(1, 8))

    citations = report_data.get("citations", [])
    if citations:
        story.append(Paragraph("Evidence Sources", styles["VSectionHead"]))
        for index, citation in enumerate(citations, 1):
            source = citation if isinstance(citation, str) else citation.get("text", str(citation))
            story.append(Paragraph(f"[{index}] {source}", styles["VBody"]))
        story.append(Spacer(1, 8))

    if report_data.get("uncertainty_flag"):
        story.append(Paragraph(
            "LOW CONFIDENCE: Fewer than 2 supporting sources retrieved. "
            "Results may be unreliable. Clinical review strongly recommended.",
            ParagraphStyle(
                "Warn",
                parent=styles["VBody"],
                textColor=WARN,
                fontName="Helvetica-Bold",
            ),
        ))
        story.append(Spacer(1, 6))

    story.append(HRFlowable(width="100%", thickness=0.8, color=WARN))
    story.append(Spacer(1, 4))
    story.append(Paragraph(MEDICAL_DISCLAIMER, styles["VDisclaimer"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()

