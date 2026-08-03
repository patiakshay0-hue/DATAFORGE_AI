"""Deep Learning 1.0 PDF report.

Reuses the brand styling and table helpers from `core.exporter` so the output looks
like the rest of the product, and adds the sections this module needs: model
configuration, the pattern the user selected, and the feature recommendation.

Charts are drawn as native reportlab vectors rather than embedded images — no extra
dependency, and the report stays reproducible server-side.
"""

from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    KeepTogether, Flowable,
)

from core.exporter import (
    _styles, _table_style, BLUE, PURPLE, EMERALD, AMBER, SLATE_400, SLATE_700,
    WHITE, DARK_BG, W, H,
)

VIOLET = colors.HexColor("#8b5cf6")


class LossCurve(Flowable):
    """Training/validation reconstruction loss, drawn directly on the canvas."""

    def __init__(self, history, width, height=45 * mm):
        super().__init__()
        self.history = history or []
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        if len(self.history) < 2:
            return

        train = [h.get("train_loss", 0) or 0 for h in self.history]
        val = [h.get("val_loss", 0) or 0 for h in self.history]
        lo = min(min(train), min(val))
        hi = max(max(train), max(val))
        span = (hi - lo) or 1.0

        pad = 8 * mm
        plot_w, plot_h = self.width - pad * 1.5, self.height - pad

        c.setStrokeColor(colors.HexColor("#1e293b"))
        c.setLineWidth(0.4)
        for i in range(5):                       # horizontal gridlines
            y = pad + plot_h * i / 4
            c.line(pad, y, pad + plot_w, y)

        def series(values, colour):
            c.setStrokeColor(colour)
            c.setLineWidth(1.2)
            path = c.beginPath()
            for i, v in enumerate(values):
                x = pad + plot_w * i / max(1, len(values) - 1)
                y = pad + plot_h * (v - lo) / span
                path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
            c.drawPath(path)

        series(train, BLUE)
        series(val, AMBER)

        c.setFillColor(SLATE_400)
        c.setFont("Helvetica", 6)
        c.drawString(pad, 2 * mm, "epoch 1")
        c.drawRightString(pad + plot_w, 2 * mm, f"epoch {len(self.history)}")
        c.setFillColor(BLUE)
        c.drawString(pad, self.height - 3 * mm, "— train")
        c.setFillColor(AMBER)
        c.drawString(pad + 16 * mm, self.height - 3 * mm, "— validation")


class ArchitectureDiagram(Flowable):
    """Encoder → bottleneck → decoder, as stacked bars whose width tracks layer size."""

    def __init__(self, layers, width, height=None):
        super().__init__()
        self.layers = layers or []
        self.width = width
        self.height = height or (len(self.layers) * 7 * mm + 4 * mm)

    def draw(self):
        c = self.canv
        if not self.layers:
            return
        row_h = 6 * mm
        for i, label in enumerate(self.layers):
            y = self.height - (i + 1) * row_h
            # The bottleneck is the point of the whole architecture — highlight it.
            is_latent = "Latent" in label
            bar_w = self.width * (0.32 if is_latent else 0.62)
            c.setFillColor(VIOLET if is_latent else colors.HexColor("#1e293b"))
            c.roundRect((self.width - bar_w) / 2, y, bar_w, row_h - 1.6 * mm, 1.5, fill=1, stroke=0)
            c.setFillColor(WHITE if is_latent else SLATE_400)
            c.setFont("Helvetica-Bold" if is_latent else "Helvetica", 7)
            c.drawCentredString(self.width / 2, y + 1.6 * mm, str(label))


def _header(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, H - 52 * mm, W, 52 * mm, fill=1, stroke=0)
    canvas.setFillColor(VIOLET)
    canvas.rect(0, H - 52 * mm, W, 1.5, fill=1, stroke=0)
    canvas.restoreState()


def generate(job, recommendation: dict) -> bytes:
    """Build the full report for a finished job."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    st = _styles()
    usable = W - 36 * mm
    els: list = []

    profile = job.profile or {}
    config = (job.config or {}).get("config") if isinstance(job.config, dict) and "config" in job.config else job.config or {}
    training = job.training or {}

    # ── Cover ────────────────────────────────────────────────────────────────
    els += [
        Spacer(1, 8 * mm),
        Paragraph("Deep Learning 1.0", st["title"]),
        Paragraph("Unsupervised Pattern Discovery Report", st["subtitle"]),
        Spacer(1, 2 * mm),
        HRFlowable(width=usable, thickness=1, color=VIOLET),
        Spacer(1, 4 * mm),
    ]

    meta = [
        ["Dataset", job.filename],
        ["Generated", datetime.now().strftime("%Y-%m-%d  %H:%M")],
        ["Engine", training.get("engine", "—")],
        ["Mode", "Unsupervised — no target column required"],
    ]
    tbl = Table([[Paragraph(k, st["label"]), Paragraph(str(v), st["value"])] for k, v in meta],
                colWidths=[35 * mm, usable - 35 * mm])
    tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                             ("TOPPADDING", (0, 0), (-1, -1), 4),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    els += [tbl, Spacer(1, 7 * mm)]

    # ── Dataset summary ──────────────────────────────────────────────────────
    els.append(Paragraph("Dataset Summary", st["h2"]))
    summary = [
        ["Rows", "Columns", "Used", "Excluded", "Missing values"],
        [f"{profile.get('rows', 0):,}", str(profile.get("columns_total", 0)),
         str(profile.get("columns_used", 0)), str(profile.get("columns_dropped", 0)),
         f"{profile.get('missing_total', 0):,}"],
    ]
    t = Table(summary, colWidths=[usable / 5] * 5)
    t.setStyle(_table_style())
    els += [t, Spacer(1, 6 * mm)]

    # ── Model configuration ──────────────────────────────────────────────────
    els.append(Paragraph("Model Configuration (auto-selected)", st["h2"]))
    cfg_rows = [
        ["Parameter", "Value", "Parameter", "Value"],
        ["Hidden layers", " → ".join(map(str, config.get("hidden_layers", []))),
         "Latent dimensions", str(config.get("latent_dim", "—"))],
        ["Epochs (budget)", str(config.get("epochs", "—")),
         "Epochs run", f"{training.get('epochs_run', '—')}"
                       f"{' (early stop)' if training.get('stopped_early') else ''}"],
        ["Learning rate", str(config.get("learning_rate", "—")),
         "Batch size", str(config.get("batch_size", "—"))],
        ["Activation", str(config.get("activation", "—")).upper(),
         "Optimizer", str(config.get("optimizer", "—")).capitalize()],
        ["Loss function", str(config.get("loss_function", "—")).upper(),
         "Dropout", str(config.get("dropout", "—"))],
        ["Parameters", f"{training.get('n_params', 0):,}",
         "Training time", str(training.get("training_time", "—"))],
    ]
    t = Table(cfg_rows, colWidths=[usable * 0.26, usable * 0.24] * 2, repeatRows=1)
    t.setStyle(_table_style(header_color=PURPLE))
    els += [t, Spacer(1, 5 * mm)]

    if config.get("rationale"):
        els.append(Paragraph("Why these values", st["h2"]))
        for line in config["rationale"]:
            els.append(Paragraph(f"• {line}", st["body"]))
        els.append(Spacer(1, 5 * mm))

    # ── Architecture + loss curve ────────────────────────────────────────────
    if training.get("architecture"):
        els.append(Paragraph("Network Architecture", st["h2"]))
        els.append(ArchitectureDiagram(training["architecture"], usable))
        els.append(Spacer(1, 4 * mm))

    if training.get("history"):
        els.append(Paragraph("Reconstruction Loss", st["h2"]))
        els.append(LossCurve(training["history"], usable))
        els.append(Spacer(1, 4 * mm))

    # ── Performance ──────────────────────────────────────────────────────────
    els.append(Paragraph("Performance Metrics", st["h2"]))
    gain = training.get("nonlinear_gain")
    # A negative gain means the linear baseline won. Reporting that as a negative
    # "advantage" reads as a broken number, so state the finding in words instead.
    if not isinstance(gain, (int, float)):
        gain_label, gain_meaning = "—", "Not available"
    elif gain > 0:
        gain_label = f"+{gain * 100:.1f}%"
        gain_meaning = "The network beats the linear baseline — structure is non-linear"
    else:
        gain_label = "None"
        gain_meaning = "Linear methods do as well — the structure here is linear"

    perf = [
        ["Metric", "Value", "Meaning"],
        ["Reconstruction error (MSE)", str(training.get("ae_error", "—")),
         "How well the network rebuilds the data"],
        ["Linear (PCA) baseline", str(training.get("pca_error", "—")),
         "Same compression, linear method"],
        ["Non-linear advantage", gain_label, gain_meaning],
        ["Best epoch", str(training.get("best_epoch", "—")),
         "Where validation loss bottomed out"],
    ]
    t = Table(perf, colWidths=[usable * 0.3, usable * 0.2, usable * 0.5], repeatRows=1)
    t.setStyle(_table_style(header_color=EMERALD))
    els += [t, Spacer(1, 6 * mm)]

    # ── Selected patterns ────────────────────────────────────────────────────
    selected = recommendation.get("selected_patterns", [])
    els.append(Paragraph("Selected Hidden Pattern(s)", st["h2"]))
    if not selected:
        els.append(Paragraph("No pattern was selected — the ranking below reflects all "
                             "discovered structure.", st["body"]))
    else:
        by_id = {p["id"]: p for p in (job.patterns or [])}
        for sel in selected:
            full = by_id.get(sel["id"], {})
            preferred = " (preferred)" if sel["id"] == job.preferred_pattern else ""
            block = [
                HRFlowable(width=usable, thickness=2, color=VIOLET, spaceAfter=4),
                Paragraph(f"{sel['title']}{preferred}", st["insight_title"]),
                Paragraph(full.get("description", ""), st["insight_text"]),
                Paragraph(f"<b>Confidence:</b> {sel['confidence'] * 100:.0f}% &nbsp;&nbsp; "
                          f"<b>Columns:</b> {', '.join(sel.get('columns', [])) or '—'}",
                          st["insight_text"]),
            ]
            if sel.get("recommendation"):
                block.append(Paragraph(f"<b>What to do:</b> {sel['recommendation']}",
                                       st["insight_text"]))
            block.append(Spacer(1, 4 * mm))
            els.append(KeepTogether(block))
    els.append(Spacer(1, 3 * mm))

    # ── Feature recommendation ───────────────────────────────────────────────
    els.append(Paragraph("Recommended Features", st["h2"]))
    used = recommendation.get("features_used", [])
    ignored = recommendation.get("features_ignored", [])
    els.append(Paragraph(
        f"<b>Use ({len(used)}):</b> {', '.join(used) if used else '—'}", st["body"]))
    els.append(Spacer(1, 2 * mm))
    els.append(Paragraph(
        f"<b>Ignore ({len(ignored)}):</b> {', '.join(ignored) if ignored else '—'}", st["body"]))
    els.append(Spacer(1, 4 * mm))

    ranking = recommendation.get("ranking", [])
    if ranking:
        rows = [["#", "Feature", "Unique information", "In selected pattern"]]
        for r in ranking[:25]:
            rows.append([str(r["rank"]), r["feature"], f"{r['information_pct']:.1f}%",
                         "yes" if r["in_selected_pattern"] else ""])
        t = Table(rows, colWidths=[usable * 0.08, usable * 0.42, usable * 0.25, usable * 0.25],
                  repeatRows=1)
        t.setStyle(_table_style(header_color=VIOLET))
        els += [t, Spacer(1, 5 * mm)]

    excluded = recommendation.get("excluded_at_load", [])
    if excluded:
        els.append(Paragraph("Columns Excluded Before Modelling", st["h2"]))
        rows = [["Column", "Reason"]] + [[e["column"], e["reason"]] for e in excluded]
        t = Table(rows, colWidths=[usable * 0.35, usable * 0.65], repeatRows=1)
        t.setStyle(_table_style(header_color=AMBER))
        els += [t, Spacer(1, 5 * mm)]

    # ── All discovered patterns ──────────────────────────────────────────────
    if job.patterns:
        els.append(Paragraph("All Discovered Patterns", st["h2"]))
        rows = [["Pattern", "Type", "Confidence", "Columns"]]
        for p in job.patterns:
            rows.append([p["title"], p["type"].replace("_", " "),
                         f"{p['confidence'] * 100:.0f}%",
                         ", ".join(p.get("columns", [])[:3])])
        t = Table(rows, colWidths=[usable * 0.4, usable * 0.18, usable * 0.14, usable * 0.28],
                  repeatRows=1)
        t.setStyle(_table_style())
        els += [t, Spacer(1, 5 * mm)]

    els += [
        HRFlowable(width=usable, thickness=0.5, color=SLATE_700),
        Spacer(1, 3 * mm),
        Paragraph(f"Generated by DataForge AI · Deep Learning 1.0 · "
                  f"{datetime.now().strftime('%Y-%m-%d')}",
                  ParagraphStyle("f", fontSize=7, textColor=SLATE_700,
                                 fontName="Helvetica", alignment=TA_CENTER)),
    ]

    doc.build(els, onFirstPage=_header, onLaterPages=_header)
    return buf.getvalue()
