"""DataForge AI — analytics PDF report.

Print-first colour system: light paper, dark ink, teal brand accents.
(The UI is dark; a PDF is a document people print and share, so the palette
follows print conventions rather than the browser chrome.)
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Print palette ─────────────────────────────────────────────────────────────
# High-contrast ink on paper. Brand teal only as an accent — never body text.
PAPER       = colors.HexColor("#ffffff")
PAPER_SOFT  = colors.HexColor("#f8fafc")       # alt row / card wash
INK         = colors.HexColor("#0f172a")       # primary text
INK_MUTED   = colors.HexColor("#475569")       # secondary text
INK_FAINT   = colors.HexColor("#94a3b8")       # captions / footer
RULE        = colors.HexColor("#e2e8f0")       # hairlines
RULE_STRONG = colors.HexColor("#cbd5e1")

BRAND       = colors.HexColor("#0d9488")       # teal-600 — headers, accents
BRAND_DEEP  = colors.HexColor("#0f766e")       # teal-700 — cover band
BRAND_SOFT  = colors.HexColor("#ccfbf1")       # teal-100 — chip / highlight wash
BRAND_MID   = colors.HexColor("#14b8a6")       # teal-500

PURPLE      = colors.HexColor("#7c3aed")
EMERALD     = colors.HexColor("#059669")
AMBER       = colors.HexColor("#d97706")
RED         = colors.HexColor("#dc2626")
SKY         = colors.HexColor("#0284c8")

# Back-compat aliases used by core.dl1.report
WHITE       = colors.white
BLUE        = BRAND
DARK_BG     = BRAND_DEEP
SLATE_400   = INK_MUTED
SLATE_700   = INK
PAGE_BG     = PAPER

W, H = A4


def _styles():
    getSampleStyleSheet()  # registers base fonts
    return {
        "title": ParagraphStyle(
            "df_title", fontSize=22, leading=28, textColor=colors.white,
            fontName="Helvetica-Bold", spaceAfter=2, alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "df_subtitle", fontSize=10, leading=14, textColor=colors.HexColor("#99f6e4"),
            fontName="Helvetica",
        ),
        "h2": ParagraphStyle(
            "df_h2", fontSize=12, leading=16, textColor=BRAND_DEEP,
            fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "df_body", fontSize=9, leading=13, textColor=INK_MUTED,
            fontName="Helvetica",
        ),
        "label": ParagraphStyle(
            "df_label", fontSize=8, leading=11, textColor=INK_FAINT,
            fontName="Helvetica", spaceAfter=1,
        ),
        "value": ParagraphStyle(
            "df_value", fontSize=10, leading=13, textColor=INK,
            fontName="Helvetica-Bold",
        ),
        "insight_title": ParagraphStyle(
            "df_it", fontSize=10, leading=13, textColor=INK,
            fontName="Helvetica-Bold",
        ),
        "insight_text": ParagraphStyle(
            "df_ix", fontSize=9, leading=13, textColor=INK_MUTED,
            fontName="Helvetica",
        ),
        "chip": ParagraphStyle(
            "df_chip", fontSize=8, textColor=BRAND_DEEP,
            fontName="Helvetica-Bold",
        ),
        "meta_k": ParagraphStyle(
            "df_mk", fontSize=7.5, leading=10, textColor=INK_FAINT,
            fontName="Helvetica",
        ),
        "meta_v": ParagraphStyle(
            "df_mv", fontSize=9, leading=12, textColor=INK,
            fontName="Helvetica-Bold",
        ),
        "footer": ParagraphStyle(
            "df_footer", fontSize=7, leading=9, textColor=INK_FAINT,
            fontName="Helvetica", alignment=TA_CENTER,
        ),
        "ok": ParagraphStyle(
            "df_ok", fontSize=9, leading=12, textColor=EMERALD,
            fontName="Helvetica-Bold",
        ),
    }


def _table_style(header_color=BRAND):
    """Clean print table: solid brand header, alternating soft rows, dark ink."""
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [PAPER, PAPER_SOFT]),
        ("TEXTCOLOR",     (0, 1), (-1, -1), INK),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.4, RULE),
        ("BOX",           (0, 0), (-1, -1), 0.8, RULE_STRONG),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
    ])


def _draw_page(canvas, doc):
    """Full white page + teal cover band on first page + page number footer."""
    canvas.saveState()
    # Paper fill — without this, "transparent" cells show as whatever the
    # viewer default is and dark-theme leftovers look broken.
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)

    if doc.page == 1:
        # Brand cover band
        canvas.setFillColor(BRAND_DEEP)
        canvas.rect(0, H - 48 * mm, W, 48 * mm, fill=1, stroke=0)
        # Accent hairline under the band
        canvas.setFillColor(BRAND_MID)
        canvas.rect(0, H - 48 * mm, W, 1.8, fill=1, stroke=0)
        # Soft wash stripe for depth
        canvas.setFillColor(colors.HexColor("#115e59"))
        canvas.rect(0, H - 8 * mm, W, 8 * mm, fill=1, stroke=0)
    else:
        # Thin top rule on continuation pages
        canvas.setStrokeColor(BRAND)
        canvas.setLineWidth(1.2)
        canvas.line(18 * mm, H - 10 * mm, W - 18 * mm, H - 10 * mm)
        canvas.setFillColor(BRAND_DEEP)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(18 * mm, H - 8 * mm, "DataForge AI")
        canvas.setFillColor(INK_FAINT)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(W - 18 * mm, H - 8 * mm, "Analytics Report")

    # Footer
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 12 * mm, W - 18 * mm, 12 * mm)
    canvas.setFillColor(INK_FAINT)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(18 * mm, 7 * mm, "DataForge AI · Confidential")
    canvas.drawRightString(W - 18 * mm, 7 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _kpi_row(items, usable_w, st):
    """Four equal meta cards across the top of the report."""
    n = len(items)
    col_w = usable_w / n
    cells = []
    for label, value in items:
        inner = Table(
            [[Paragraph(label, st["meta_k"])],
             [Paragraph(str(value), st["meta_v"])]],
            colWidths=[col_w - 4 * mm],
        )
        inner.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), PAPER_SOFT),
            ("BOX",           (0, 0), (-1, -1), 0.6, RULE),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        cells.append(inner)
    row = Table([cells], colWidths=[col_w] * n)
    row.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    return row


def _fmt(v, digits=2):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try:
            if abs(v) >= 1000:
                return f"{v:,.1f}"
            return f"{v:.{digits}f}"
        except Exception:
            return str(v)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return str(v)


def generate_pdf_report(
    filename: str,
    schema: list,
    eda: dict,
    insights: list,
    df: pd.DataFrame,
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=18 * mm,
    )
    st = _styles()
    els = []
    usable_w = W - 36 * mm

    # ── Cover (sits inside the teal band drawn by _draw_page) ─────────────────
    els.append(Spacer(1, 6 * mm))
    els.append(Paragraph("DataForge AI", st["title"]))
    els.append(Paragraph("Automated Analytics Report", st["subtitle"]))
    els.append(Spacer(1, 14 * mm))  # clear the band

    # KPI strip
    els.append(_kpi_row([
        ("DATASET", filename),
        ("GENERATED", datetime.now().strftime("%Y-%m-%d  %H:%M")),
        ("RECORDS", f"{eda.get('rows', 0):,}"),
        ("FEATURES", str(eda.get("columns", 0))),
    ], usable_w, st))
    els.append(Spacer(1, 8 * mm))

    # ── Schema ────────────────────────────────────────────────────────────────
    els.append(Paragraph("Schema Detection", st["h2"]))
    els.append(HRFlowable(width=usable_w, thickness=0.8, color=BRAND,
                          spaceAfter=4, spaceBefore=0))

    type_color = {
        "numeric": SKY, "categorical": PURPLE,
        "boolean": AMBER, "datetime": EMERALD,
    }
    schema_data = [["Column", "Type", "Unique", "Missing"]]
    for col in schema:
        schema_data.append([
            str(col.get("name", "")),
            str(col.get("type", "")).capitalize(),
            str(col.get("unique", "")),
            str(col.get("missing", 0)),
        ])
    col_ws = [usable_w * x for x in (0.46, 0.20, 0.17, 0.17)]
    tbl = Table(schema_data, colWidths=col_ws, repeatRows=1)
    style = _table_style(BRAND)
    # Colour the type cell by kind (row 1..)
    for i, col in enumerate(schema, start=1):
        t = str(col.get("type", "")).lower()
        if t in type_color:
            style.add("TEXTCOLOR", (1, i), (1, i), type_color[t])
            style.add("FONTNAME",  (1, i), (1, i), "Helvetica-Bold")
        missing = int(col.get("missing", 0) or 0)
        if missing > 0:
            style.add("TEXTCOLOR", (3, i), (3, i), RED)
            style.add("FONTNAME",  (3, i), (3, i), "Helvetica-Bold")
    tbl.setStyle(style)
    els.append(tbl)
    els.append(Spacer(1, 7 * mm))

    # ── Statistical Summary ───────────────────────────────────────────────────
    summary = eda.get("summary", {})
    if summary:
        els.append(Paragraph("Statistical Summary", st["h2"]))
        els.append(HRFlowable(width=usable_w, thickness=0.8, color=PURPLE,
                              spaceAfter=4))
        stat_data = [["Feature", "Count", "Mean", "Std", "Min", "Median", "Max"]]
        for col_name, s in list(summary.items())[:18]:
            median = s.get("50%", s.get("median"))
            stat_data.append([
                str(col_name)[:22],
                _fmt(s.get("count"), 0),
                _fmt(s.get("mean")),
                _fmt(s.get("std")),
                _fmt(s.get("min")),
                _fmt(median),
                _fmt(s.get("max")),
            ])
        col_ws2 = [usable_w * x for x in (0.24, 0.10, 0.13, 0.13, 0.13, 0.14, 0.13)]
        tbl2 = Table(stat_data, colWidths=col_ws2, repeatRows=1)
        tbl2.setStyle(_table_style(PURPLE))
        els.append(tbl2)
        els.append(Spacer(1, 7 * mm))

    # ── Data Quality ──────────────────────────────────────────────────────────
    missing = eda.get("missing", {}) or {}
    rows_total = max(int(eda.get("rows", 1) or 1), 1)
    has_missing = any(int(v or 0) > 0 for v in missing.values())

    els.append(Paragraph("Data Quality", st["h2"]))
    els.append(HRFlowable(width=usable_w, thickness=0.8, color=AMBER,
                          spaceAfter=4))
    if has_missing:
        mv_data = [["Column", "Missing", "% Missing", "Severity"]]
        for col_name, cnt in missing.items():
            cnt = int(cnt or 0)
            if cnt <= 0:
                continue
            pct = cnt / rows_total * 100
            if pct >= 30:
                sev = "High"
            elif pct >= 10:
                sev = "Medium"
            else:
                sev = "Low"
            mv_data.append([str(col_name), f"{cnt:,}", f"{pct:.1f}%", sev])
        if len(mv_data) > 1:
            mv_tbl = Table(
                mv_data,
                colWidths=[usable_w * x for x in (0.40, 0.18, 0.20, 0.22)],
                repeatRows=1,
            )
            mv_style = _table_style(AMBER)
            for i, row in enumerate(mv_data[1:], start=1):
                sev = row[3]
                col = RED if sev == "High" else (AMBER if sev == "Medium" else EMERALD)
                mv_style.add("TEXTCOLOR", (3, i), (3, i), col)
                mv_style.add("FONTNAME",  (3, i), (3, i), "Helvetica-Bold")
            mv_tbl.setStyle(mv_style)
            els.append(mv_tbl)
        else:
            els.append(Paragraph("No missing values detected. Dataset is complete.", st["ok"]))
    else:
        els.append(Paragraph("No missing values detected. Dataset is complete.", st["ok"]))
    els.append(Spacer(1, 7 * mm))

    # ── Sample Data ───────────────────────────────────────────────────────────
    sample = df.head(8)
    cols_to_show = list(sample.columns)[:7]
    if cols_to_show:
        sample = sample[cols_to_show]
        els.append(Paragraph("Sample Data (first 8 rows)", st["h2"]))
        els.append(HRFlowable(width=usable_w, thickness=0.8, color=SKY,
                              spaceAfter=4))
        sample_data = [[str(c)[:18] for c in cols_to_show]]
        for _, row in sample.iterrows():
            sample_data.append([str(row[c])[:22] for c in cols_to_show])
        col_w = usable_w / len(cols_to_show)
        s_tbl = Table(sample_data, colWidths=[col_w] * len(cols_to_show), repeatRows=1)
        s_tbl.setStyle(_table_style(SKY))
        els.append(s_tbl)
        els.append(Spacer(1, 7 * mm))

    # ── Insights ──────────────────────────────────────────────────────────────
    if insights:
        els.append(Paragraph("AI-Generated Insights", st["h2"]))
        els.append(HRFlowable(width=usable_w, thickness=0.8, color=BRAND,
                              spaceAfter=4))
        insight_colors = {
            "success": EMERALD,
            "warning": AMBER,
            "insight": BRAND,
            "info":    SKY,
        }
        for ins in insights:
            c = insight_colors.get(ins.get("type", "info"), BRAND)
            # Left-accent card
            body = Table(
                [[Paragraph(ins.get("title", ""), st["insight_title"])],
                 [Paragraph(ins.get("text", ""), st["insight_text"])]],
                colWidths=[usable_w - 4 * mm],
            )
            body.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), PAPER_SOFT),
                ("BOX",           (0, 0), (-1, -1), 0.5, RULE),
                ("LINEBEFORE",    (0, 0), (0, -1), 3, c),
                ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                ("TOPPADDING",    (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            els.append(KeepTogether([body, Spacer(1, 3 * mm)]))

    # ── Closing ───────────────────────────────────────────────────────────────
    els.append(Spacer(1, 4 * mm))
    els.append(HRFlowable(width=usable_w, thickness=0.6, color=RULE_STRONG))
    els.append(Spacer(1, 2 * mm))
    els.append(Paragraph(
        f"Generated by DataForge AI · {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        st["footer"],
    ))

    doc.build(els, onFirstPage=_draw_page, onLaterPages=_draw_page)
    return buf.getvalue()
