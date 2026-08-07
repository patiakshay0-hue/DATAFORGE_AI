"""Deep Learning 2.0 PDF report.

Light-theme print palette matching core.exporter. All charts are native
reportlab vectors — no raster images.
"""

from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from core.exporter import (
    _styles,
    _table_style,
    _fmt,
    PAPER,
    PAPER_SOFT,
    INK,
    INK_MUTED,
    INK_FAINT,
    RULE,
    RULE_STRONG,
    BRAND,
    BRAND_DEEP,
    BRAND_SOFT,
    BRAND_MID,
    PURPLE,
    EMERALD,
    AMBER,
    RED,
    SKY,
    WHITE,
    W,
    H,
)

# ── Chart palette (light-theme, colour-blind accessible) ──────────────────────
TEAL    = BRAND_MID
VIOLET  = PURPLE
ROSE    = RED
CYAN    = SKY
ORANGE  = AMBER
LIME    = EMERALD
PINK    = colors.HexColor("#e11d48")
INDIGO  = colors.HexColor("#6366f1")

CLUSTER_PALETTE = [TEAL, VIOLET, CYAN, ROSE, ORANGE, INDIGO, PINK, LIME]

GRID_COLOR   = colors.HexColor("#e2e8f0")
LABEL_COLOR  = INK_MUTED
FILL_DEFAULT = BRAND


class LossCurve(Flowable):
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
        val   = [h.get("val_loss",   0) or 0 for h in self.history]
        lo = min(min(train), min(val))
        hi = max(max(train), max(val))
        span = (hi - lo) or 1.0

        pad = 8 * mm
        pw, ph = self.width - pad * 1.5, self.height - pad

        # Light grid
        c.setStrokeColor(GRID_COLOR)
        c.setLineWidth(0.3)
        for i in range(5):
            y = pad + ph * i / 4
            c.line(pad, y, pad + pw, y)

        def series(vals, colour, lw=1.4):
            c.setStrokeColor(colour)
            c.setLineWidth(lw)
            path = c.beginPath()
            for i, v in enumerate(vals):
                x = pad + pw * i / max(1, len(vals) - 1)
                y = pad + ph * (v - lo) / span
                path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
            c.drawPath(path)

        series(train, TEAL)
        series(val, ORANGE)

        c.setFont("Helvetica", 6)
        c.setFillColor(INK_FAINT)
        c.drawString(pad, 2 * mm, "epoch 1")
        c.drawRightString(pad + pw, 2 * mm, f"epoch {len(self.history)}")
        c.setFillColor(TEAL)
        c.drawString(pad, self.height - 3 * mm, "— train")
        c.setFillColor(ORANGE)
        c.drawString(pad + 16 * mm, self.height - 3 * mm, "— validation")


class ArchitectureDiagram(Flowable):
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
            is_latent = "Latent" in label
            bar_w = self.width * (0.35 if is_latent else 0.64)
            c.setFillColor(BRAND_SOFT if is_latent else PAPER_SOFT)
            c.setStrokeColor(BRAND if is_latent else RULE)
            c.setLineWidth(0.6)
            c.roundRect(
                (self.width - bar_w) / 2, y, bar_w, row_h - 1.6 * mm,
                2, fill=1, stroke=1,
            )
            c.setFillColor(BRAND_DEEP if is_latent else INK_MUTED)
            c.setFont("Helvetica-Bold" if is_latent else "Helvetica", 7)
            c.drawCentredString(self.width / 2, y + 1.6 * mm, str(label))


class BarChart(Flowable):
    def __init__(self, items, width, height=55 * mm, max_items=10,
                 value_key="error_pct"):
        super().__init__()
        self.items = (items or [])[:max_items]
        self.width = width
        self.height = height
        self.value_key = value_key

    def draw(self):
        c = self.canv
        if not self.items:
            return
        pad_l, pad_r = 45 * mm, 12 * mm
        pad_t, pad_b = 8 * mm, 8 * mm
        bar_h, gap = 5 * mm, 2 * mm

        max_val = max((i.get(self.value_key, 0) or 0) for i in self.items) or 1
        pw = self.width - pad_l - pad_r

        c.setStrokeColor(GRID_COLOR)
        c.setLineWidth(0.3)
        for i in range(5):
            x = pad_l + pw * i / 4
            c.line(x, pad_b, x, self.height - pad_t)

        for idx, item in enumerate(self.items):
            y = self.height - pad_t - (idx + 1) * (bar_h + gap)
            if y < pad_b:
                break
            name = str(item.get("feature", ""))[:20]
            val  = item.get(self.value_key, 0) or 0
            bw   = pw * val / max_val

            # Gradient-ish: use alternating brand shades
            fill = TEAL if idx % 2 == 0 else VIOLET
            c.setFillColor(fill)
            c.roundRect(pad_l, y, bw, bar_h, 1.5, fill=1, stroke=0)

            c.setFillColor(INK_MUTED)
            c.setFont("Helvetica", 7)
            c.drawRightString(pad_l - 2 * mm, y + 1.5 * mm, name)

            c.setFillColor(INK_FAINT)
            c.setFont("Helvetica", 6)
            c.drawString(pad_l + bw + 2 * mm, y + 1.5 * mm, f"{val:.1f}%")


class ScatterPlot(Flowable):
    def __init__(self, points, labels=None, width=100 * mm, height=60 * mm,
                 highlight_indices=None, title=None):
        super().__init__()
        self.points = (points or [])[:500]
        self.labels = (labels or [])[:500] if labels else None
        self.width = width
        self.height = height
        self.highlight_indices = set(highlight_indices or [])
        self.title = title

    def draw(self):
        c = self.canv
        if not self.points:
            return

        pad = 12 * mm
        pw, ph = self.width - pad * 1.5, self.height - pad * 1.5

        xs = [p.get("x", 0) for p in self.points]
        ys = [p.get("y", 0) for p in self.points]
        x_span = (max(xs) - min(xs)) or 1.0
        y_span = (max(ys) - min(ys)) or 1.0

        # Light grid
        c.setStrokeColor(GRID_COLOR)
        c.setLineWidth(0.3)
        for i in range(5):
            y = pad + ph * i / 4
            c.line(pad, y, pad + pw, y)

        label_colors = None
        if self.labels:
            ul = sorted(set(self.labels))
            label_colors = {l: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                            for i, l in enumerate(ul)}

        for i, pt in enumerate(self.points):
            px = pad + pw * (pt.get("x", 0) - min(xs)) / x_span
            py = pad + ph * (pt.get("y", 0) - min(ys)) / y_span
            if i in self.highlight_indices:
                c.setFillColor(ROSE)
                c.circle(px, py, 2.5, fill=1, stroke=0)
            elif label_colors and self.labels and i < len(self.labels):
                c.setFillColor(label_colors[self.labels[i]])
                c.circle(px, py, 1.6, fill=1, stroke=0)
            else:
                c.setFillColor(FILL_DEFAULT)
                c.circle(px, py, 1.6, fill=1, stroke=0)

        c.setFillColor(INK_FAINT)
        c.setFont("Helvetica", 6)
        c.drawString(pad, 3 * mm, f"n={len(self.points)} points")
        if self.title:
            c.setFillColor(INK_MUTED)
            c.setFont("Helvetica-Bold", 8)
            c.drawCentredString(self.width / 2, self.height - 4 * mm, self.title)


class HeatmapGrid(Flowable):
    """Correlation heatmap — diverging red/green, light-theme variant."""

    def __init__(self, matrix, columns, width, height=60 * mm, max_cols=12):
        super().__init__()
        self.matrix = matrix
        self.columns = (columns or [])[:max_cols]
        self.width = width
        self.height = height
        self.max_cols = max_cols

    def draw(self):
        c = self.canv
        if not self.matrix or not self.columns:
            return
        n = min(len(self.columns), self.max_cols)
        if n < 2:
            return

        pad_l, pad_t = 30 * mm, 8 * mm
        cell = min(8 * mm, (self.width - pad_l) / n)

        def _color(val):
            r = abs(val)
            if val >= 0.8:
                return colors.HexColor("#15803d")   # dark green
            if val >= 0.5:
                return colors.HexColor("#22c55e")   # green
            if val >= 0.2:
                return colors.HexColor("#86efac")   # light green
            if val <= -0.8:
                return colors.HexColor("#b91c1c")   # dark red
            if val <= -0.5:
                return colors.HexColor("#ef4444")   # red
            if val <= -0.2:
                return colors.HexColor("#fca5a5")   # light red
            return PAPER_SOFT                       # neutral

        for i in range(n):
            for j in range(n):
                val = (self.matrix[i][j]
                       if i < len(self.matrix) and j < len(self.matrix[i]) else 0)
                x = pad_l + j * cell
                y = self.height - pad_t - (i + 1) * cell
                c.setFillColor(_color(val))
                c.rect(x, y, cell - 0.5, cell - 0.5, fill=1, stroke=0)
                # Inline correlation value
                c.setFillColor(INK if abs(val) >= 0.5 else INK_FAINT)
                c.setFont("Helvetica", max(4.5, min(6, cell / 2.2)))
                c.drawCentredString(x + cell / 2 - 0.25, y + cell / 3,
                                    f"{val:.2f}")

        c.setFillColor(INK_MUTED)
        c.setFont("Helvetica", 5)
        for i, col in enumerate(self.columns):
            y = self.height - pad_t - (i + 1) * cell + cell / 2
            c.drawRightString(pad_l - 2 * mm, y, col[:15])
            c.saveState()
            c.translate(pad_l + i * cell + cell / 2, self.height - 2 * mm)
            c.rotate(45)
            c.drawString(0, 0, col[:10])
            c.restoreState()

        c.setFont("Helvetica", 6)
        c.setFillColor(INK_FAINT)
        c.drawString(pad_l, 3 * mm,
                     "Green: positive correlation | Red: negative")


def _draw_header(canvas, doc):
    canvas.saveState()
    # Page fill
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)

    if doc.page == 1:
        canvas.setFillColor(BRAND_DEEP)
        canvas.rect(0, H - 48 * mm, W, 48 * mm, fill=1, stroke=0)
        canvas.setFillColor(BRAND_MID)
        canvas.rect(0, H - 48 * mm, W, 1.8, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#115e59"))
        canvas.rect(0, H - 8 * mm, W, 8 * mm, fill=1, stroke=0)
    else:
        canvas.setStrokeColor(BRAND)
        canvas.setLineWidth(1.2)
        canvas.line(18 * mm, H - 10 * mm, W - 18 * mm, H - 10 * mm)
        canvas.setFillColor(BRAND_DEEP)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(18 * mm, H - 8 * mm, "DataForge AI")
        canvas.setFillColor(INK_FAINT)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(W - 18 * mm, H - 8 * mm,
                               "Deep Learning 2.0 Report")

    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 12 * mm, W - 18 * mm, 12 * mm)
    canvas.setFillColor(INK_FAINT)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(18 * mm, 7 * mm, "DataForge AI · Confidential")
    canvas.drawRightString(W - 18 * mm, 7 * mm, f"Page {doc.page}")
    canvas.restoreState()


def generate(job, recommendation: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=18 * mm,
    )
    st = _styles()
    usable = W - 36 * mm
    els: list = []

    profile   = job.profile or {}
    config    = ((job.config or {}).get("config")
                 if isinstance(job.config, dict) and "config" in job.config
                 else job.config or {})
    training  = job.training or {}

    # ── Cover ────────────────────────────────────────────────────────────────
    els += [
        Spacer(1, 6 * mm),
        Paragraph("Deep Learning 2.0", st["title"]),
        Paragraph("Unsupervised Pattern Discovery Report", st["subtitle"]),
        Spacer(1, 14 * mm),
    ]

    # KPI strip (from exporter._kpi_row reuse)
    from core.exporter import _kpi_row
    els.append(_kpi_row([
        ("DATASET", job.filename),
        ("ENGINE", training.get("engine", "—")),
        ("MODE", "Unsupervised"),
        ("GENERATED", datetime.now().strftime("%Y-%m-%d  %H:%M")),
    ], usable, st))
    els.append(Spacer(1, 7 * mm))

    # ── Dataset Summary ─────────────────────────────────────────────────────
    els.append(Paragraph("Dataset Summary", st["h2"]))
    els.append(HRFlowable(width=usable, thickness=0.8, color=BRAND, spaceAfter=4))
    summary = [
        ["Rows", "Columns", "Used", "Excluded", "Missing"],
        [f"{profile.get('rows', 0):,}", str(profile.get("columns_total", 0)),
         str(profile.get("columns_used", 0)),
         str(profile.get("columns_dropped", 0)),
         f"{profile.get('missing_total', 0):,}"],
    ]
    t = Table(summary, colWidths=[usable / 5] * 5)
    t.setStyle(_table_style())
    els += [t, Spacer(1, 6 * mm)]

    # ── Model Config ────────────────────────────────────────────────────────
    els.append(Paragraph("Model Configuration (auto-selected)", st["h2"]))
    els.append(HRFlowable(width=usable, thickness=0.8, color=PURPLE, spaceAfter=4))
    cfg_rows = [
        ["Parameter", "Value", "Parameter", "Value"],
        ["Hidden layers",
         " → ".join(map(str, config.get("hidden_layers", []))),
         "Latent dimensions", str(config.get("latent_dim", "—"))],
        ["Epochs (budget)", str(config.get("epochs", "—")),
         "Epochs run",
         f"{training.get('epochs_run', '—')}"
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
    t = Table(cfg_rows, colWidths=[usable * 0.26, usable * 0.24] * 2,
              repeatRows=1)
    t.setStyle(_table_style(BRAND))
    els += [t, Spacer(1, 5 * mm)]

    if config.get("rationale"):
        els.append(Paragraph("Why these values", st["h2"]))
        for line in config["rationale"]:
            els.append(Paragraph(f"• {line}", st["body"]))
        els.append(Spacer(1, 5 * mm))

    # ── Architecture + Loss Curve ───────────────────────────────────────────
    if training.get("architecture"):
        els.append(Paragraph("Network Architecture", st["h2"]))
        els.append(HRFlowable(width=usable, thickness=0.8, color=BRAND,
                              spaceAfter=4))
        els.append(ArchitectureDiagram(training["architecture"], usable))
        els.append(Spacer(1, 4 * mm))

    if training.get("history"):
        els.append(Paragraph("Reconstruction Loss", st["h2"]))
        els.append(HRFlowable(width=usable, thickness=0.8, color=ORANGE,
                              spaceAfter=4))
        els.append(LossCurve(training["history"], usable))
        els.append(Spacer(1, 4 * mm))

    # ── Performance ─────────────────────────────────────────────────────────
    els.append(Paragraph("Performance Metrics", st["h2"]))
    els.append(HRFlowable(width=usable, thickness=0.8, color=EMERALD,
                          spaceAfter=4))
    gain = training.get("nonlinear_gain")
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
    t = Table(perf, colWidths=[usable * 0.3, usable * 0.2, usable * 0.5],
              repeatRows=1)
    t.setStyle(_table_style(EMERALD))
    els += [t, Spacer(1, 6 * mm)]

    # ── Selected Patterns ──────────────────────────────────────────────────
    selected = recommendation.get("selected_patterns", [])
    els.append(Paragraph("Selected Hidden Pattern(s)", st["h2"]))
    els.append(HRFlowable(width=usable, thickness=0.8, color=PURPLE,
                          spaceAfter=4))
    if not selected:
        els.append(Paragraph(
            "No pattern was selected — the ranking below reflects all "
            "discovered structure.", st["body"]))
    else:
        by_id = {p["id"]: p for p in (job.patterns or [])}
        for sel in selected:
            full = by_id.get(sel["id"], {})
            preferred = " (preferred)" if sel["id"] == job.preferred_pattern else ""
            block = [
                HRFlowable(width=usable, thickness=2, color=BRAND, spaceAfter=4),
                Paragraph(f"{sel['title']}{preferred}", st["insight_title"]),
                Paragraph(full.get("description", ""), st["insight_text"]),
                Paragraph(
                    f"<b>Confidence:</b> {sel['confidence'] * 100:.0f}% "
                    f"&nbsp;&nbsp; "
                    f"<b>Columns:</b> {', '.join(sel.get('columns', [])) or '—'}",
                    st["insight_text"]),
            ]
            if sel.get("recommendation"):
                block.append(Paragraph(
                    f"<b>What to do:</b> {sel['recommendation']}",
                    st["insight_text"]))
            block.append(Spacer(1, 3 * mm))

            pdata = full.get("data", {})
            pviz  = full.get("visualization")
            ptype = full.get("type", "")

            if pviz == "bar" and pdata.get("items"):
                block.append(BarChart(pdata["items"], usable,
                                      value_key="error_pct"))
                block.append(Spacer(1, 3 * mm))

            elif pviz == "scatter" and pdata.get("points"):
                highlight = pdata.get("indices") if ptype == "anomalies" else None
                title = ("Anomaly Distribution" if ptype == "anomalies"
                         else "Cluster Structure")
                block.append(ScatterPlot(pdata["points"], pdata.get("labels"),
                                         usable, 60 * mm, highlight, title))
                block.append(Spacer(1, 3 * mm))

                if ptype in ("clusters", "segments") and pdata.get("profiles"):
                    block.append(Paragraph("<b>Cluster Profiles:</b>", st["body"]))
                    for prof in pdata["profiles"]:
                        traits = ", ".join(
                            f"{t['feature']} ({'+' if t['z'] > 0 else ''}"
                            f"{t['z']:.2f}σ)"
                            for t in prof.get("traits", [])[:3])
                        block.append(Paragraph(
                            f"Cluster {prof['cluster']}: "
                            f"{prof['size']} rows "
                            f"({prof['share'] * 100:.1f}%) — {traits}",
                            st["body"]))
                    block.append(Spacer(1, 2 * mm))

                if ptype == "anomalies":
                    count   = pdata.get("count", 0)
                    share   = pdata.get("share", 0)
                    ratio   = pdata.get("error_ratio", 1)
                    drivers = pdata.get("drivers", [])
                    block.append(Paragraph(
                        f"<b>Details:</b> {count} anomalous rows "
                        f"({share * 100:.1f}% of data) "
                        f"reconstruct {ratio:.1f}x worse than normal.",
                        st["body"]))
                    if drivers:
                        dtxt = ", ".join(
                            f"{d['feature']} ({'+' if d['deviation'] > 0 else ''}"
                            f"{d['deviation']:.2f}σ)"
                            for d in drivers[:4])
                        block.append(Paragraph(
                            f"<b>Key drivers:</b> {dtxt}", st["body"]))

            elif pviz == "heatmap":
                pairs  = pdata.get("pairs", [])
                matrix = pdata.get("matrix")
                cols   = pdata.get("columns", [])
                if pairs:
                    block.append(Paragraph("<b>Correlated Pairs:</b>", st["body"]))
                    pair_rows = [["Feature A", "Feature B", "Correlation"]]
                    for p in pairs[:8]:
                        pair_rows.append([
                            p.get("a", ""), p.get("b", ""),
                            f"{p.get('r', 0):.3f}"])
                    pt = Table(pair_rows,
                               colWidths=[usable * 0.35, usable * 0.35,
                                          usable * 0.3],
                               repeatRows=1)
                    pt.setStyle(_table_style(EMERALD))
                    block.append(pt)
                    block.append(Spacer(1, 2 * mm))
                if matrix and cols and len(cols) <= 15:
                    block.append(Paragraph("<b>Correlation Matrix:</b>",
                                           st["body"]))
                    block.append(HeatmapGrid(matrix, cols, usable))
                    block.append(Spacer(1, 3 * mm))

            if ptype == "compressibility" and pdata:
                latent    = pdata.get("latent_dim", 0)
                n_feat    = pdata.get("n_features", 0)
                reduction = pdata.get("reduction_pct", 0)
                block.append(Paragraph(
                    f"<b>Compression:</b> {n_feat} features → {latent} "
                    f"latent dimensions ({reduction:.1f}% reduction)",
                    st["body"]))

            if ptype in ("non_linear", "linear_structure") and pdata:
                ae_err = pdata.get("ae_error", 0)
                pca_err = pdata.get("pca_error", 0)
                gain_v = pdata.get("gain", 0)
                block.append(Paragraph(
                    f"<b>Error comparison:</b> Autoencoder = {ae_err:.4f}, "
                    f"PCA = {pca_err:.4f} "
                    f"(difference: {'+' if gain_v > 0 else ''}"
                    f"{gain_v * 100:.1f}%)", st["body"]))

            block.append(Spacer(1, 4 * mm))
            els.append(KeepTogether(block))
    els.append(Spacer(1, 3 * mm))

    # ── Feature Recommendation ─────────────────────────────────────────────
    els.append(Paragraph("Recommended Features", st["h2"]))
    els.append(HRFlowable(width=usable, thickness=0.8, color=SKY, spaceAfter=4))
    used    = recommendation.get("features_used", [])
    ignored = recommendation.get("features_ignored", [])
    els.append(Paragraph(
        f"<b>Use ({len(used)}):</b> {', '.join(used) if used else '—'}",
        st["body"]))
    els.append(Spacer(1, 2 * mm))
    els.append(Paragraph(
        f"<b>Ignore ({len(ignored)}):</b> {', '.join(ignored) if ignored else '—'}",
        st["body"]))
    els.append(Spacer(1, 4 * mm))

    ranking = recommendation.get("ranking", [])
    if ranking:
        rows_data = [["#", "Feature", "Unique info", "In pattern"]]
        for r in ranking[:25]:
            rows_data.append([
                str(r["rank"]), r["feature"],
                f"{r['information_pct']:.1f}%",
                "yes" if r["in_selected_pattern"] else ""])
        t = Table(rows_data,
                  colWidths=[usable * 0.08, usable * 0.44,
                             usable * 0.24, usable * 0.24],
                  repeatRows=1)
        t.setStyle(_table_style(BRAND))
        els += [t, Spacer(1, 5 * mm)]

    excluded = recommendation.get("excluded_at_load", [])
    if excluded:
        els.append(Paragraph("Columns Excluded Before Modelling", st["h2"]))
        els.append(HRFlowable(width=usable, thickness=0.8, color=AMBER,
                              spaceAfter=4))
        rows_data = [["Column", "Reason"]] + [
            [e["column"], e["reason"]] for e in excluded]
        t = Table(rows_data,
                  colWidths=[usable * 0.35, usable * 0.65], repeatRows=1)
        t.setStyle(_table_style(AMBER))
        els += [t, Spacer(1, 5 * mm)]

    # ── All Discovered Patterns ────────────────────────────────────────────
    if job.patterns:
        els.append(Paragraph("All Discovered Patterns", st["h2"]))
        els.append(HRFlowable(width=usable, thickness=0.8, color=INK_FAINT,
                              spaceAfter=4))
        for p in job.patterns:
            pblock = [
                HRFlowable(width=usable, thickness=0.6, color=RULE, spaceAfter=2),
                Paragraph(
                    f"<b>{p['title']}</b> "
                    f"({p['type'].replace('_', ' ')}, "
                    f"{p['confidence'] * 100:.0f}%)",
                    st["body"]),
            ]
            if p.get("description"):
                pblock.append(Paragraph(p["description"][:220], st["insight_text"]))

            pdata = p.get("data", {})
            pviz  = p.get("visualization")
            ptype = p.get("type", "")

            if pviz == "bar" and pdata.get("items"):
                pblock.append(Spacer(1, 2 * mm))
                pblock.append(BarChart(pdata["items"], usable * 0.85,
                                       40 * mm, max_items=6))
            elif pviz == "scatter" and pdata.get("points"):
                pblock.append(Spacer(1, 2 * mm))
                highlight = pdata.get("indices") if ptype == "anomalies" else None
                pblock.append(ScatterPlot(
                    pdata["points"], pdata.get("labels"),
                    usable * 0.8, 45 * mm, highlight))
                if ptype == "anomalies":
                    pblock.append(Paragraph(
                        f"<font size='7'>{pdata.get('count', 0)} anomalies, "
                        f"{pdata.get('share', 0) * 100:.1f}% of data</font>",
                        st["body"]))
                elif ptype in ("clusters", "segments") and pdata.get("profiles"):
                    n_cl = len(pdata["profiles"])
                    sizes = [f"{pr['size']}" for pr in pdata["profiles"]]
                    pblock.append(Paragraph(
                        f"<font size='7'>{n_cl} clusters: "
                        f"{', '.join(sizes)} rows each</font>",
                        st["body"]))
            elif pviz == "heatmap" and pdata.get("pairs"):
                pair_list = ", ".join(
                    f"{p['a']}:{p['b']}({p['r']:.2f})"
                    for p in pdata["pairs"][:4])
                pblock.append(Paragraph(
                    f"<font size='7'>Pairs: {pair_list}</font>", st["body"]))

            pblock.append(Spacer(1, 3 * mm))
            els.append(KeepTogether(pblock))
        els.append(Spacer(1, 5 * mm))

    # ── Footer ──────────────────────────────────────────────────────────────
    els += [
        HRFlowable(width=usable, thickness=0.6, color=RULE_STRONG),
        Spacer(1, 2 * mm),
        Paragraph(
            f"Generated by DataForge AI · Deep Learning 2.0 · "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ParagraphStyle("f", fontSize=7, textColor=INK_FAINT,
                           fontName="Helvetica", alignment=TA_CENTER)),
    ]

    doc.build(els, onFirstPage=_draw_header, onLaterPages=_draw_header)
    return buf.getvalue()
