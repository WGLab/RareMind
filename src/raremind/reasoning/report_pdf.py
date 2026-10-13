"""Professional PDF rendering for Stage 9A/9B clinical review reports."""
from __future__ import annotations

import re
from html import escape
from pathlib import Path
from typing import Any


def _safe(value: Any) -> str:
    text = str(value or "")
    replacements = {
        "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u2192": "->", "\u2713": "+", "\u2717": "x", "\u26a0": "Warning:",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _inline_markup(value: Any) -> str:
    """Convert the small Markdown subset emitted by report prompts to ReportLab XML."""
    text = escape(_safe(value))
    # ReportLab Paragraph supports <b>; escaping first keeps model output safe.
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def _paragraphs(text: Any, styles: dict[str, Any]) -> list[Any]:
    from reportlab.platypus import KeepTogether, Paragraph, Spacer

    elements: list[Any] = []
    for block in re.split(r"\n\s*\n", _safe(text).strip()):
        if not block.strip():
            continue
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        for line in lines:
            markdown_heading = re.match(r"^#{1,6}\s+(.+)$", line)
            if markdown_heading:
                elements.append(Paragraph(_inline_markup(markdown_heading.group(1)), styles["minor_heading"]))
                continue
            heading = re.match(r"^([A-Z][A-Z0-9 /&()-]{3,}):\s*(.*)$", line)
            if heading:
                if heading.group(2):
                    elements.append(KeepTogether([
                        Paragraph(_inline_markup(heading.group(1)), styles["minor_heading"]),
                        Paragraph(_inline_markup(heading.group(2)), styles["body"]),
                    ]))
                else:
                    elements.append(Paragraph(_inline_markup(heading.group(1)), styles["minor_heading"]))
            elif re.match(r"^[-*]\s+", line):
                content = re.sub(r"^[-*]\s+", "", line)
                elements.append(Paragraph(f"&bull;&nbsp;{_inline_markup(content)}", styles["body"]))
            else:
                elements.append(Paragraph(_inline_markup(line), styles["body"]))
        elements.append(Spacer(1, 5))
    return elements


def render_stage9_pdf(
    report: dict[str, Any],
    output_path: str | Path,
    *,
    case_id: str,
    report_level: str,
) -> Path:
    """Render and validate one static Stage 9 clinical-review PDF."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            BaseDocTemplate, Frame, KeepTogether, LongTable, PageTemplate,
            Paragraph, Spacer, Table, TableStyle,
        )
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError as exc:
        raise ImportError("Stage 9 PDF generation requires reportlab: pip install reportlab pypdf") from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    navy, teal, ink = colors.HexColor("#16324F"), colors.HexColor("#168AAD"), colors.HexColor("#1F2937")
    pale, line, muted = colors.HexColor("#EAF4F7"), colors.HexColor("#C8D7DE"), colors.HexColor("#52616B")
    regular_font, bold_font = "Helvetica", "Helvetica-Bold"
    font_pairs = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for regular_path, bold_path in font_pairs:
        if regular_path.is_file() and bold_path.is_file():
            pdfmetrics.registerFont(TTFont("RareMindSans", str(regular_path)))
            pdfmetrics.registerFont(TTFont("RareMindSans-Bold", str(bold_path)))
            regular_font, bold_font = "RareMindSans", "RareMindSans-Bold"
            break
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontName=bold_font, fontSize=20, leading=24, textColor=navy, alignment=TA_LEFT, spaceAfter=7),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName=regular_font, fontSize=10.5, leading=14, textColor=muted, spaceAfter=14),
        "section": ParagraphStyle("Section", parent=base["Heading2"], fontName=bold_font, fontSize=13, leading=16, textColor=navy, spaceBefore=12, spaceAfter=7),
        "candidate": ParagraphStyle("Candidate", parent=base["Heading3"], fontName=bold_font, fontSize=11.5, leading=14, textColor=teal, spaceBefore=8, spaceAfter=5),
        "minor_heading": ParagraphStyle("Minor", parent=base["Normal"], fontName=bold_font, fontSize=9.2, leading=12, textColor=navy, spaceBefore=5, spaceAfter=2),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName=regular_font, fontSize=9, leading=12.5, textColor=ink, spaceAfter=3),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName=regular_font, fontSize=7.5, leading=9.5, textColor=muted),
        "disclaimer": ParagraphStyle("Disclaimer", parent=base["BodyText"], fontName=bold_font, fontSize=8.2, leading=11, textColor=navy),
    }

    class NumberedCanvas(Canvas):
        def save(self) -> None:
            pages = []
            for page in self._saved_page_states:
                pages.append(page)
            total = len(pages)
            for state in pages:
                self.__dict__.update(state)
                self.setFont("Helvetica", 7.5)
                self.setFillColor(muted)
                self.drawString(0.65 * inch, 0.38 * inch, "RareMind clinical review - research use only")
                self.drawRightString(7.85 * inch, 0.38 * inch, f"Page {self._pageNumber} of {total}")
                Canvas.showPage(self)
            Canvas.save(self)

        def showPage(self) -> None:
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            Canvas.__init__(self, *args, **kwargs)
            self._saved_page_states: list[dict[str, Any]] = []

    level_title = (
        "RareMind Stage 9A Group-level Clinical Review"
        if report_level == "a"
        else "RareMind Stage 9B Subtype-level Clinical Review"
    )
    doc = BaseDocTemplate(
        str(output), pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch,
        topMargin=0.62 * inch, bottomMargin=0.62 * inch,
        title=f"{level_title} - {case_id}",
        author="RareMind",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="review", frames=[frame])])

    case_signal = report.get("case_signal", {}) or {}
    signal_interpretation = case_signal.get("interpretation", {}) or {}
    predicted = _safe(case_signal.get("predicted_class", "not available")).replace("_", " ")
    category = _safe(case_signal.get("signal_category", "not available"))
    score = case_signal.get("confidence_score")
    score_text = f"{float(score):.1f}/100" if score is not None else "not available"
    signal_summary = _safe(signal_interpretation.get("summary", "No Stage 3.5 summary available."))
    top_notice = (
        "Decision-support output for professional review. Rankings and confidence values are relative "
        "pipeline signals, not diagnostic probabilities. Verify all findings against the source record."
        f"<br/><br/><b>Stage 3.5 case signal:</b> {escape(predicted)} | {escape(category)} | "
        f"{escape(score_text)} (uncalibrated)<br/>{_inline_markup(signal_summary)}"
    )
    level_name = "Group-level differential" if report_level == "a" else "Subtype-level differential"
    story: list[Any] = [
        Paragraph(level_title, styles["title"]),
        Paragraph(f"{level_name} | Case {escape(_safe(case_id))}", styles["subtitle"]),
        Table(
            [[Paragraph(
                top_notice,
                styles["disclaimer"],
            )]],
            colWidths=[doc.width], style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), pale), ("BOX", (0, 0), (-1, -1), 0.7, teal),
                ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]),
        ),
        Spacer(1, 9), Paragraph("Patient summary", styles["section"]),
        *_paragraphs(report.get("patient_summary", "Not available."), styles),
    ]
    family = report.get("family_history_assessment")
    if family:
        story.extend([Paragraph("Family history assessment", styles["section"]), *_paragraphs(family, styles)])
    story.append(Paragraph("Stage 3.5 case-signal assessment", styles["section"]))
    story.extend(_paragraphs(signal_summary, styles))
    reasons = signal_interpretation.get("reasons", []) or []
    if reasons:
        story.append(Paragraph("Reasons", styles["minor_heading"]))
        for reason in reasons:
            story.append(Paragraph(f"&bull;&nbsp;{_inline_markup(reason)}", styles["body"]))
    caveat = signal_interpretation.get("caveat")
    if caveat:
        story.append(Paragraph(f"<b>Caveat:</b> {_inline_markup(caveat)}", styles["body"]))

    cards = report.get("top_candidates", []) or []
    assessments = report.get("clinical_assessments", []) or []
    assessment_by_id = {str(item.get("disease_id")): item for item in assessments}
    story.append(Paragraph(f"Top {len(cards)} {'disease groups' if report_level == 'a' else 'exact subtypes'}", styles["section"]))
    for card in cards:
        candidate_id = str(card.get("disease_id", ""))
        rank = card.get("final_rank", "")
        heading = f"{rank}. {_safe(card.get('disease_name', candidate_id))} ({candidate_id})"
        block: list[Any] = [Paragraph(_inline_markup(heading), styles["candidate"])]
        if report_level == "a" and card.get("representative_subtypes"):
            rows = [[Paragraph("Representative subtype", styles["small"]), Paragraph("Exact rank", styles["small"])]]
            for subtype in card["representative_subtypes"]:
                rows.append([
                    Paragraph(_inline_markup(subtype.get("disease_name", "")), styles["small"]),
                    Paragraph(_inline_markup(subtype.get("subtype_rank", "-")), styles["small"]),
                ])
            block.append(Table(rows, colWidths=[doc.width * 0.82, doc.width * 0.18], style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), pale), ("GRID", (0, 0), (-1, -1), 0.35, line),
                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])))
        assessment = assessment_by_id.get(candidate_id, {}).get("text") or card.get("clinical_assessment", "")
        block.extend(_paragraphs(assessment, styles))
        story.append(KeepTogether(block[:2]) if len(block) >= 2 else block[0])
        story.extend(block[2:] if len(block) >= 2 else [])

    story.extend([
        Paragraph("Differential synthesis", styles["section"]),
        *_paragraphs(report.get("differential_synthesis", "Not available."), styles),
        Paragraph("Recommended next steps", styles["section"]),
        *_paragraphs(report.get("recommended_next_steps", "Not available."), styles),
        Paragraph("Pipeline transparency", styles["section"]),
    ])
    header = ["Rank", "Candidate", "MONDO", "Score"]
    rows = [[Paragraph(value, styles["small"]) for value in header]]
    for card in cards:
        rows.append([
            Paragraph(_inline_markup(card.get("final_rank", "")), styles["small"]),
            Paragraph(_inline_markup(card.get("disease_name", "")), styles["small"]),
            Paragraph(_inline_markup(card.get("disease_id", "")), styles["small"]),
            Paragraph(f"{float(card.get('final_score', 0.0)):.3f}", styles["small"]),
        ])
    story.append(LongTable(rows, repeatRows=1, colWidths=[0.42 * inch, 4.35 * inch, 1.05 * inch, 0.62 * inch], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, line), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])))
    doc.build(story, canvasmaker=NumberedCanvas)

    if not output.is_file() or output.stat().st_size < 1000:
        raise RuntimeError(f"Stage 9 PDF was not created correctly: {output}")
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(output))
        if not reader.pages:
            raise RuntimeError(f"Stage 9 PDF has no pages: {output}")
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:2])
        if str(case_id) not in text:
            raise RuntimeError(f"Stage 9 PDF validation could not find case ID {case_id}")
    except ImportError:
        pass
    return output
