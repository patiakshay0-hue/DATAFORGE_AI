import io
import pandas as pd
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Brand colours ──────────────────────────────────────────────────────────────
DARK_BG   = colors.HexColor('#0d1523')
BLUE      = colors.HexColor('#0ea5e9')
PURPLE    = colors.HexColor('#6366f1')
EMERALD   = colors.HexColor('#10b981')
AMBER     = colors.HexColor('#f59e0b')
RED       = colors.HexColor('#ef4444')
SLATE_700 = colors.HexColor('#334155')
SLATE_400 = colors.HexColor('#94a3b8')
WHITE     = colors.white
PAGE_BG   = colors.HexColor('#080d18')

W, H = A4


def _styles():
    base = getSampleStyleSheet()
    return {
        'title': ParagraphStyle('title', fontSize=22, leading=28,
                                textColor=WHITE, fontName='Helvetica-Bold', spaceAfter=4),
        'subtitle': ParagraphStyle('subtitle', fontSize=10, leading=14,
                                   textColor=SLATE_400, fontName='Helvetica'),
        'h2': ParagraphStyle('h2', fontSize=13, leading=18,
                             textColor=BLUE, fontName='Helvetica-Bold',
                             spaceBefore=14, spaceAfter=6),
        'body': ParagraphStyle('body', fontSize=9, leading=13,
                               textColor=SLATE_400, fontName='Helvetica'),
        'label': ParagraphStyle('label', fontSize=8, leading=12,
                                textColor=SLATE_400, fontName='Helvetica',
                                spaceAfter=2),
        'value': ParagraphStyle('value', fontSize=10, leading=14,
                                textColor=WHITE, fontName='Helvetica-Bold'),
        'insight_title': ParagraphStyle('it', fontSize=10, leading=13,
                                        textColor=WHITE, fontName='Helvetica-Bold'),
        'insight_text': ParagraphStyle('ix', fontSize=9, leading=13,
                                       textColor=SLATE_400, fontName='Helvetica'),
        'chip': ParagraphStyle('chip', fontSize=8, textColor=BLUE,
                               fontName='Helvetica-Bold'),
    }


def _table_style(header_color=BLUE):
    return TableStyle([
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), header_color),
        ('TEXTCOLOR',  (0, 0), (-1, 0), WHITE),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, 0), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#0d1a2e'), colors.HexColor('#0a1628')]),
        ('TEXTCOLOR',  (0, 1), (-1, -1), SLATE_400),
        ('FONTNAME',   (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',   (0, 1), (-1, -1), 8),
        ('GRID',       (0, 0), (-1, -1), 0.4, colors.HexColor('#1e293b')),
        ('ALIGN',      (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ])


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
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm, bottomMargin=16*mm,
    )

    st  = _styles()
    els = []
    usable_w = W - 36*mm

    # ── Cover header ─────────────────────────────────────────────────────────
    def _draw_header(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(DARK_BG)
        canvas.rect(0, H - 52*mm, W, 52*mm, fill=1, stroke=0)
        canvas.setFillColor(BLUE)
        canvas.rect(0, H - 52*mm, W, 1.5, fill=1, stroke=0)
        canvas.restoreState()

    # Header banner content
    els.append(Spacer(1, 8*mm))
    els.append(Paragraph('DataForge AI', st['title']))
    els.append(Paragraph('Automated Analytics Report', st['subtitle']))
    els.append(Spacer(1, 2*mm))
    els.append(HRFlowable(width=usable_w, thickness=1, color=BLUE))
    els.append(Spacer(1, 4*mm))

    # Meta row
    meta = [
        ['Dataset', filename],
        ['Generated', datetime.now().strftime('%Y-%m-%d  %H:%M')],
        ['Records', f"{eda.get('rows', 0):,}"],
        ['Features', str(eda.get('columns', 0))],
    ]
    meta_tbl = Table([[Paragraph(k, st['label']), Paragraph(v, st['value'])]
                      for k, v in meta],
                     colWidths=[35*mm, usable_w - 35*mm])
    meta_tbl.setStyle(TableStyle([
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',   (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
    ]))
    els.append(meta_tbl)
    els.append(Spacer(1, 8*mm))

    # ── Schema ────────────────────────────────────────────────────────────────
    els.append(Paragraph('Schema Detection', st['h2']))

    schema_data = [['Column', 'Type', 'Unique', 'Missing']]
    for col in schema:
        schema_data.append([
            col.get('name', ''),
            col.get('type', '').capitalize(),
            str(col.get('unique', '')),
            str(col.get('missing', 0)),
        ])

    col_ws = [usable_w * x for x in [0.45, 0.20, 0.18, 0.17]]
    tbl = Table(schema_data, colWidths=col_ws, repeatRows=1)
    tbl.setStyle(_table_style())
    els.append(tbl)
    els.append(Spacer(1, 8*mm))

    # ── Statistical Summary ───────────────────────────────────────────────────
    summary = eda.get('summary', {})
    if summary:
        els.append(Paragraph('Statistical Summary', st['h2']))
        stat_data = [['Feature', 'Count', 'Mean', 'Std Dev', 'Min', 'Median', 'Max']]
        for col_name, s in list(summary.items())[:15]:
            stat_data.append([
                col_name,
                str(s.get('count', '—')),
                f"{s['mean']:.2f}"  if isinstance(s.get('mean'),  (int, float)) else '—',
                f"{s['std']:.2f}"   if isinstance(s.get('std'),   (int, float)) else '—',
                f"{s['min']:.2f}"   if isinstance(s.get('min'),   (int, float)) else '—',
                f"{s.get('50%', s.get('median', '—')):.2f}"
                    if isinstance(s.get('50%', s.get('median')), (int, float)) else '—',
                f"{s['max']:.2f}"   if isinstance(s.get('max'),   (int, float)) else '—',
            ])
        col_ws2 = [usable_w * x for x in [0.26, 0.10, 0.13, 0.13, 0.12, 0.13, 0.13]]
        tbl2 = Table(stat_data, colWidths=col_ws2, repeatRows=1)
        tbl2.setStyle(_table_style(header_color=PURPLE))
        els.append(tbl2)
        els.append(Spacer(1, 8*mm))

    # ── Missing Values ────────────────────────────────────────────────────────
    missing = eda.get('missing', {})
    rows_total = eda.get('rows', 1)
    has_missing = any(v > 0 for v in missing.values())

    els.append(Paragraph('Data Quality', st['h2']))
    if has_missing:
        mv_data = [['Column', 'Missing Count', '% Missing']]
        for col_name, cnt in missing.items():
            pct = f"{(cnt / rows_total * 100):.1f}%" if rows_total else '—'
            mv_data.append([col_name, str(cnt), pct])
        mv_tbl = Table(mv_data, colWidths=[usable_w * x for x in [0.5, 0.25, 0.25]],
                       repeatRows=1)
        mv_tbl.setStyle(_table_style(header_color=AMBER))
        els.append(mv_tbl)
    else:
        els.append(Paragraph(
            'No missing values detected. Dataset is complete.',
            ParagraphStyle('ok', fontSize=9, textColor=EMERALD, fontName='Helvetica-Bold')
        ))
    els.append(Spacer(1, 8*mm))

    # ── Sample Data ───────────────────────────────────────────────────────────
    sample = df.head(8)
    cols_to_show = list(sample.columns)[:7]
    sample = sample[cols_to_show]

    els.append(Paragraph('Sample Data (first 8 rows)', st['h2']))
    sample_data = [cols_to_show] + [
        [str(row[c])[:20] for c in cols_to_show]
        for _, row in sample.iterrows()
    ]
    col_w = usable_w / len(cols_to_show)
    s_tbl = Table(sample_data, colWidths=[col_w] * len(cols_to_show), repeatRows=1)
    s_tbl.setStyle(_table_style(header_color=colors.HexColor('#0f4c81')))
    els.append(s_tbl)
    els.append(Spacer(1, 8*mm))

    # ── Insights ──────────────────────────────────────────────────────────────
    if insights:
        els.append(Paragraph('AI-Generated Insights', st['h2']))
        INSIGHT_COLORS = {
            'success': EMERALD,
            'warning': AMBER,
            'insight': BLUE,
            'info':    PURPLE,
        }
        for ins in insights:
            c = INSIGHT_COLORS.get(ins.get('type', 'info'), BLUE)
            block = KeepTogether([
                HRFlowable(width=usable_w, thickness=2, color=c, spaceAfter=4),
                Paragraph(ins.get('title', ''), st['insight_title']),
                Paragraph(ins.get('text',  ''), st['insight_text']),
                Spacer(1, 5*mm),
            ])
            els.append(block)

    # ── Footer note ───────────────────────────────────────────────────────────
    els.append(HRFlowable(width=usable_w, thickness=0.5, color=SLATE_700))
    els.append(Spacer(1, 3*mm))
    els.append(Paragraph(
        f'Generated by DataForge AI · {datetime.now().strftime("%Y-%m-%d")}',
        ParagraphStyle('footer', fontSize=7, textColor=SLATE_700,
                       fontName='Helvetica', alignment=TA_CENTER)
    ))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(els, onFirstPage=_draw_header, onLaterPages=_draw_header)
    return buf.getvalue()
