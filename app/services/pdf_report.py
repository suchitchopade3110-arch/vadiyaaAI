"""VaidyaAI branded clinical PDF report generator."""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from app.core.disclaimer import MEDICAL_DISCLAIMER


DARK = colors.HexColor("#0D1F1A")
PRIMARY = colors.HexColor("#1D9E75")
PRIMARY_DK = colors.HexColor("#155F47")
AMBER = colors.HexColor("#EF9F27")
DANGER = colors.HexColor("#B2182B")
LIGHT_BG = colors.HexColor("#F0FAF5")
SUBTLE = colors.HexColor("#E8F5EF")
BORDER = colors.HexColor("#C3E8D6")
GREY = colors.HexColor("#6B7280")
WHITE = colors.white
BLACK = colors.HexColor("#111827")

W, H = A4
UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[2]
LOGO_PATH = ROOT / "ui" / "assets" / "vaidyaa-logo.jpeg"


def _hex(color: colors.Color) -> str:
    return f"#{color.hexval()[2:]}"


def _logo_reader() -> ImageReader | None:
    if not LOGO_PATH.exists():
        return None
    try:
        return ImageReader(str(LOGO_PATH))
    except Exception:
        return None


def _watermark_logo(canv, _doc):
    logo = _logo_reader()
    if logo is None:
        return
    canv.saveState()
    size = 110 * mm
    canv.setFillAlpha(0.04)
    canv.drawImage(
        logo,
        (W - size) / 2,
        (H - size) / 2,
        width=size,
        height=size,
        preserveAspectRatio=True,
        mask="auto",
    )
    canv.restoreState()


def _page_template(canv, doc):
    _watermark_logo(canv, doc)
    logo = _logo_reader()
    canv.saveState()
    canv.setFillColor(DARK)
    canv.rect(0, H - 18 * mm, W, 18 * mm, fill=1, stroke=0)
    if logo is not None:
        canv.drawImage(
            logo,
            14 * mm,
            H - 15 * mm,
            width=12 * mm,
            height=12 * mm,
            preserveAspectRatio=True,
            mask="auto",
        )
    canv.setFillColor(WHITE)
    canv.setFont("Helvetica-Bold", 11)
    canv.drawString(28 * mm, H - 9 * mm, "VAIDYAAI")
    canv.setFont("Helvetica", 8)
    canv.setFillColor(colors.HexColor("#9CA3AF"))
    canv.drawString(28 * mm, H - 14 * mm, "Medical Intelligence Platform")
    now = datetime.now(UTC).strftime("%d %b %Y  %H:%M UTC")
    canv.setFont("Helvetica", 7.5)
    canv.drawRightString(W - 14 * mm, H - 9 * mm, now)
    canv.drawRightString(W - 14 * mm, H - 14 * mm, f"Page {doc.page}")

    canv.setFillColor(DARK)
    canv.rect(0, 0, W, 10 * mm, fill=1, stroke=0)
    canv.setFont("Helvetica", 6.5)
    canv.setFillColor(colors.HexColor("#9CA3AF"))
    canv.drawCentredString(
        W / 2,
        3.5 * mm,
        "AI-ASSISTED ANALYSIS ONLY - NOT A MEDICAL DIAGNOSIS - Consult a qualified healthcare professional",
    )
    canv.restoreState()


def _styles():
    styles = getSampleStyleSheet()

    def add(name: str, **kwargs):
        styles.add(ParagraphStyle(name, **kwargs))

    add("SectionTitle", fontName="Helvetica-Bold", fontSize=9, textColor=PRIMARY_DK, spaceBefore=12, spaceAfter=4, leading=12)
    add("Body", fontName="Helvetica", fontSize=8.5, textColor=BLACK, spaceAfter=3, leading=13)
    add("Small", fontName="Helvetica", fontSize=7.5, textColor=GREY, spaceAfter=2, leading=11)
    add("Disclaimer", fontName="Helvetica-Oblique", fontSize=7, textColor=AMBER, alignment=TA_CENTER, leading=10)
    add("PatientName", fontName="Helvetica-Bold", fontSize=18, textColor=DARK, spaceAfter=2, leading=22)
    return styles


def _severity_color(severity: str):
    normalized = str(severity or "").upper()
    if normalized in {"HIGH", "CRITICAL", "ABNORMAL", "ABNORMAL_HIGH", "ABNORMAL_LOW"}:
        return DANGER
    if normalized in {"MODERATE", "BORDERLINE", "WATCH", "MEDIUM"}:
        return AMBER
    return PRIMARY


def _flag_color(flag: str):
    normalized = str(flag or "").upper()
    if any(token in normalized for token in ("ABNORMAL", "HIGH", "CRITICAL", "LOW")):
        return DANGER
    return PRIMARY


def _to_percent(value: Any) -> float:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return min(round(number * 100, 1) if 0 <= number <= 1 else round(number, 1), 100.0)


def _format_date(value: Any) -> str:
    if not value:
        return datetime.now(UTC).strftime("%d %b %Y  %H:%M UTC")
    if isinstance(value, datetime):
        return value.astimezone(UTC).strftime("%d %b %Y  %H:%M UTC")
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC).strftime("%d %b %Y  %H:%M UTC")
    except Exception:
        return text


def _image_buffer_from_report(report_data: dict) -> Any:
    gradcam_b64 = report_data.get("gradcam_base64") or report_data.get("gradcam_b64")
    if gradcam_b64:
        try:
            return io.BytesIO(base64.b64decode(str(gradcam_b64)))
        except Exception:
            return None
    gradcam = report_data.get("gradcam") or {}
    path = report_data.get("gradcam_path") or gradcam.get("heatmap_path") or gradcam.get("heatmap_url")
    if not path:
        return None
    candidate = Path(str(path))
    if not candidate.is_absolute():
        candidate = ROOT / str(path).lstrip("/")
    return str(candidate) if candidate.exists() else None


def _source_key(source: dict) -> str:
    return str(source.get("title") or source.get("source") or source.get("url") or source.get("text", ""))[:80].lower()


def _dedupe_sources(sources: list[Any]) -> list[dict]:
    unique: list[dict] = []
    seen_titles: set[str] = set()
    seen_ids: set[str] = set()
    for item in sources or []:
        source = {"text": item} if isinstance(item, str) else dict(item) if isinstance(item, dict) else {}
        haystack = " ".join(str(source.get(key, "")) for key in ("source", "title", "text", "id")).lower()
        if "keyword_fallback" in haystack:
            continue
        row_id = str(source.get("id") or source.get("chunk_id") or "")
        title_key = _source_key(source)
        if row_id and row_id in seen_ids:
            continue
        if title_key and title_key in seen_titles:
            continue
        if row_id:
            seen_ids.add(row_id)
        if title_key:
            seen_titles.add(title_key)
        unique.append(source)
    return unique


def _confidence_bar(confidence: float):
    bar_width = 165 * mm
    fill_width = max(bar_width * confidence / 100, 1)
    color = PRIMARY if confidence >= 70 else AMBER if confidence >= 40 else DANGER
    table = Table([["", ""]], colWidths=[fill_width, max(bar_width - fill_width, 1)], rowHeights=[5])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), color),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E5E7EB")),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def _report_type(report_data: dict) -> tuple[str, bool]:
    raw_type = str(report_data.get("report_type") or report_data.get("pipeline_type") or "lab").upper()
    is_image = raw_type in {"IMAGE", "IMAGE ANALYSIS", "XRAY", "X-RAY", "CT", "MRI"} or bool(
        report_data.get("image_classification") or report_data.get("classification")
    )
    return raw_type, is_image


def generate_report_pdf(report_data: dict) -> bytes:
    """Accepts unified VaidyaAI report data and returns branded PDF bytes."""
    buf = io.BytesIO()
    left_margin = right_margin = 16 * mm
    top_margin = 22 * mm
    bottom_margin = 14 * mm
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id="main",
                frames=[Frame(left_margin, bottom_margin, W - left_margin - right_margin, H - top_margin - bottom_margin, id="body")],
                onPage=_page_template,
            )
        ]
    )

    styles = _styles()
    story = []
    report_type, is_image = _report_type(report_data)

    badge_color = DARK if is_image else PRIMARY
    header = Table(
        [
            [
                Paragraph('<font color="#FFFFFF"><b>VAIDYAAI REPORT</b></font>', styles["Small"]),
                Paragraph(
                    f'<font color="#FFFFFF"><b>{report_type}</b></font>',
                    ParagraphStyle("Badge", fontName="Helvetica-Bold", fontSize=9, alignment=TA_RIGHT, textColor=WHITE),
                ),
            ]
        ],
        colWidths=[120 * mm, 45 * mm],
    )
    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), badge_color),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (0, 0), 8),
                ("RIGHTPADDING", (-1, 0), (-1, 0), 8),
            ]
        )
    )
    story.append(header)
    story.append(Spacer(1, 5))
    patient = report_data.get("patient_name") or report_data.get("patient_id") or "Patient"
    story.append(Paragraph(str(patient), styles["PatientName"]))
    story.append(Paragraph(f'<font color="#6B7280">REPORT&nbsp;&nbsp;·&nbsp;&nbsp;{_format_date(report_data.get("generated_at"))}</font>', styles["Small"]))
    story.append(Spacer(1, 6))

    confidence = _to_percent(report_data.get("confidence_score") or report_data.get("confidence") or report_data.get("risk_score"))
    conf_color = PRIMARY if confidence >= 70 else AMBER if confidence >= 40 else DANGER
    story.append(Paragraph('<font color="#6B7280">AI confidence</font>', styles["Small"]))
    story.append(Paragraph(f'<b><font color="{_hex(conf_color)}">{confidence:.1f}%</font></b>', ParagraphStyle("Conf", fontName="Helvetica-Bold", fontSize=11, leading=14)))
    story.append(_confidence_bar(confidence))
    story.append(Spacer(1, 10))

    classification = report_data.get("classification") or report_data.get("image_classification") or {}
    findings = classification.get("findings") or report_data.get("findings") or classification.get("all_findings") or []
    if is_image and (findings or classification.get("label") or classification.get("top_class")):
        story.append(Paragraph("KEY FINDINGS", styles["SectionTitle"]))
        if not findings:
            findings = [
                {
                    "label": classification.get("label") or classification.get("top_class", ""),
                    "severity": classification.get("severity", "MODERATE"),
                    "detection_confidence": classification.get("detection_confidence") or classification.get("confidence") or confidence,
                    "description": classification.get("primary_finding") or "",
                }
            ]
        for finding in findings[:6]:
            label = finding.get("label") or finding.get("field") or ""
            severity = str(finding.get("severity") or "MODERATE").upper()
            detection = _to_percent(finding.get("detection_confidence") or finding.get("classification_prob") or finding.get("probability") or finding.get("confidence"))
            note = finding.get("description") or finding.get("clinical_meaning") or ""
            color = _severity_color(severity)
            card = Table(
                [
                    [
                        Paragraph(f'<font color="#111827"><b>{label}</b></font>', styles["Body"]),
                        Paragraph(f'<font color="{_hex(color)}">{severity}</font>', ParagraphStyle("Severity", fontName="Helvetica-Bold", fontSize=8, alignment=TA_RIGHT, leading=12)),
                    ],
                    [Paragraph(f"{detection:.0f}%  {str(note)[:120]}".strip(), styles["Small"]), Paragraph("", styles["Small"])],
                ],
                colWidths=[100 * mm, 65 * mm],
            )
            card.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                        ("LINEBEFORE", (0, 0), (0, -1), 2, color),
                        ("LINEBELOW", (0, 1), (1, 1), 0.3, BORDER),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            story.append(card)
            story.append(Spacer(1, 3))
        story.append(Spacer(1, 4))

    gradcam = _image_buffer_from_report(report_data)
    if is_image and gradcam:
        story.append(Paragraph("GRADCAM ACTIVATION MAP", styles["SectionTitle"]))
        try:
            image = Image(gradcam, width=120 * mm, height=90 * mm)
            image.hAlign = "LEFT"
            story.append(image)
            story.append(
                Paragraph(
                    "Heatmap shows regions of highest model attention. Red/orange = strong activation. Overlay is illustrative; clinical verification required.",
                    styles["Small"],
                )
            )
            story.append(Spacer(1, 8))
        except Exception as exc:
            story.append(Paragraph(f"GradCAM image unavailable: {exc}", styles["Small"]))

    risk = _to_percent(report_data.get("risk_score"))
    if not is_image and risk > 0:
        story.append(Paragraph("RISK ASSESSMENT", styles["SectionTitle"]))
        risk_color = DANGER if risk >= 70 else AMBER if risk >= 35 else PRIMARY
        table = Table(
            [
                [
                    Paragraph(f'<font color="{_hex(risk_color)}"><b>Risk Score: {risk:.0f} / 100</b></font>', styles["Body"]),
                    Paragraph(
                        f'<font color="{_hex(risk_color)}"><b>Category: {str(report_data.get("risk_label") or "").replace("_", " ").title()}</b></font>',
                        ParagraphStyle("RiskCat", fontName="Helvetica-Bold", fontSize=9, alignment=TA_RIGHT, leading=13),
                    ),
                ]
            ],
            colWidths=[82 * mm, 83 * mm],
        )
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG), ("GRID", (0, 0), (-1, -1), 0.4, BORDER), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
        story.append(table)
        story.append(Spacer(1, 8))

    lab_values = report_data.get("lab_values") or []
    if lab_values:
        story.append(Paragraph("LAB VALUES", styles["SectionTitle"]))
        rows = [
            [
                Paragraph('<font color="#FFFFFF"><b>Test</b></font>', styles["Small"]),
                Paragraph('<font color="#FFFFFF"><b>Result</b></font>', styles["Small"]),
                Paragraph('<font color="#FFFFFF"><b>Reference</b></font>', styles["Small"]),
                Paragraph('<font color="#FFFFFF"><b>Flag</b></font>', styles["Small"]),
            ]
        ]
        for value in lab_values:
            flag = value.get("flag", "NORMAL")
            name = value.get("field") or value.get("test") or value.get("name", "")
            result = value.get("value") or value.get("result", "")
            unit = value.get("unit", "")
            rows.append(
                [
                    Paragraph(str(name)[:40], styles["Small"]),
                    Paragraph(f"{result} {unit}".strip()[:30], styles["Small"]),
                    Paragraph(str(value.get("reference", "--"))[:30], styles["Small"]),
                    Paragraph(f'<font color="{_hex(_flag_color(flag))}"><b>{flag}</b></font>', styles["Small"]),
                ]
            )
        table = Table(rows, colWidths=[55 * mm, 38 * mm, 48 * mm, 24 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_DK),
                    ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

    anomalies = report_data.get("anomalies") or []
    if anomalies:
        story.append(Paragraph("ANOMALIES DETECTED", styles["SectionTitle"]))
        for anomaly in anomalies:
            field = anomaly.get("field") or anomaly.get("test") or anomaly.get("parameter", "")
            severity = anomaly.get("severity") or anomaly.get("flag") or ""
            color = _severity_color(severity)
            text = (
                f'<b><font color="{_hex(color)}">{field}</font></b>: '
                f'{anomaly.get("value", "")} {anomaly.get("unit", "")} '
                f'(ref: {anomaly.get("reference", anomaly.get("reference_range", "--"))}) - '
                f'{anomaly.get("explanation") or anomaly.get("clinical_meaning") or severity}'
            )
            story.append(Paragraph(text, styles["Body"]))
        story.append(Spacer(1, 8))

    shap = report_data.get("shap_factors") or report_data.get("shap_values") or {}
    if shap:
        story.append(Paragraph("SHAP FEATURE IMPORTANCE", styles["SectionTitle"]))
        rows = [[Paragraph('<font color="#FFFFFF"><b>Feature</b></font>', styles["Small"]), Paragraph('<font color="#FFFFFF"><b>Importance</b></font>', styles["Small"])]]
        for feature, value in sorted(shap.items(), key=lambda item: -abs(float(item[1]))):
            rows.append([Paragraph(str(feature), styles["Small"]), Paragraph(f"{float(value):.3f}", styles["Small"])])
        table = Table(rows, colWidths=[100 * mm, 65 * mm])
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), PRIMARY), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]), ("GRID", (0, 0), (-1, -1), 0.3, BORDER), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 5)]))
        story.append(table)
        story.append(Spacer(1, 8))

    explanation = report_data.get("explanation") or report_data.get("plain_language_summary") or ""
    if explanation:
        story.append(Paragraph("AI CLINICAL INTERPRETATION", styles["SectionTitle"]))
        box = Table([[Paragraph(str(explanation)[:1800], styles["Body"])]], colWidths=[165 * mm])
        box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SUBTLE), ("BOX", (0, 0), (-1, -1), 0.8, PRIMARY), ("LINEBEFORE", (0, 0), (0, -1), 3, PRIMARY), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10)]))
        story.append(box)
        story.append(Spacer(1, 8))

    sources = _dedupe_sources(report_data.get("sources") or report_data.get("citations") or report_data.get("radiology_evidence") or [])
    if sources:
        story.append(Paragraph("EVIDENCE SOURCES", styles["SectionTitle"]))
        rows = [
            [
                Paragraph('<font color="#FFFFFF"><b>#</b></font>', styles["Small"]),
                Paragraph('<font color="#FFFFFF"><b>Source</b></font>', styles["Small"]),
                Paragraph('<font color="#FFFFFF"><b>Score</b></font>', styles["Small"]),
            ]
        ]
        for index, source in enumerate(sources[:8], 1):
            title = source.get("title") or source.get("source") or source.get("text", "")
            url = source.get("url", "")
            snippet = source.get("text", "")[:120] if source.get("title") else ""
            cell = [Paragraph(f"<b>{str(title)[:90]}</b>", styles["Small"])]
            if url:
                cell.append(Paragraph(str(url)[:120], styles["Small"]))
            if snippet:
                cell.append(Paragraph(snippet, styles["Small"]))
            score = source.get("score")
            rows.append([Paragraph(f"[{index}]", styles["Small"]), cell, Paragraph(f"{float(score):.2f}" if score is not None else "-", styles["Small"])])
        table = Table(rows, colWidths=[10 * mm, 138 * mm, 17 * mm])
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), PRIMARY_DK), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]), ("GRID", (0, 0), (-1, -1), 0.3, BORDER), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 5)]))
        story.append(table)
        story.append(Spacer(1, 8))

    if report_data.get("uncertainty_flag") and len(sources) < 2:
        warning = Table(
            [[Paragraph("<b>LOW CONFIDENCE</b>: Fewer than 2 supporting sources retrieved. Clinical review strongly recommended.", ParagraphStyle("Warn", fontName="Helvetica-Bold", fontSize=8, textColor=AMBER, leading=12))]],
            colWidths=[165 * mm],
        )
        warning.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3C7")), ("BOX", (0, 0), (-1, -1), 0.8, AMBER), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
        story.append(warning)
        story.append(Spacer(1, 6))

    story.append(HRFlowable(width="100%", thickness=0.6, color=BORDER))
    story.append(Spacer(1, 4))
    story.append(Paragraph(MEDICAL_DISCLAIMER, styles["Disclaimer"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()
