"""Deep Learning 2.0 PDF report.

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
ROSE = colors.HexColor("#f43f5e")
CYAN = colors.HexColor("#06b6d4")
ORANGE = colors.HexColor("#f97316")
LIME = colors.HexColor("#84cc16")
PINK = colors.HexColor("#ec4899")
TEAL = colors.HexColor("#14b8a6")
INDIGO = colors.HexColor("#6366f1")

CLUSTER_PALETTE = [VIOLET, CYAN, ROSE, EMERALD, AMBER, INDIGO, PINK, TEAL]


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
            is_latent = "Latent" in label
            bar_w = self.width * (0.32 if is_latent else 0.62)
            c.setFillColor(VIOLET if is_latent else colors.HexColor("#1e293b"))
            c.roundRect((self.width - bar_w) / 2, y, bar_w,
                        row_h - 1.6 * mm, 1.5, fill=1, stroke=0)
            c.setFillColor(WHITE if is_latent else SLATE_400)
            c.setFont("Helvetica-Bold" if is_latent else "Helvetica", 7)
            c.drawCentredString(self.width / 2, y + 1.6 * mm, str(label))


class BarChart(Flowable):
    """Horizontal bar chart for feature importance, redundancy, etc."""

    def __init__(self, items, width, height=55 * mm, max_items=10, value_key="error_pct"):
        super().__init__()
        self.items = items[:max_items] if items else []
        self.width = width
        self.height = height
        self.value_key = value_key

    def draw(self):
        c = self.canv
        if not self.items:
            return

        pad_left = 45 * mm
        pad_right = 12 * mm
        pad_top = 8 * mm
        pad_bottom = 8 * mm
        bar_h = 5 * mm
        gap = 2 * mm

        max_val = max((i.get(self.value_key, 0) or 0) for i in self.items) or 1
        plot_w = self.width - pad_left - pad_right
        n = len(self.items)

        c.setStrokeColor(colors.HexColor("#e2e8f0"))
        c.setLineWidth(0.3)
        for i in range(5):
            x = pad_left + plot_w * i / 4
            c.line(x, pad_bottom, x, self.height - pad_top)

        for idx, item in enumerate(self.items):
            y = self.height - pad_top - (idx + 1) * (bar_h + gap)
            if y < pad_bottom:
                break

            name = str(item.get("feature", ""))[:20]
            val = item.get(self.value_key, 0) or 0
            bar_w = plot_w * val / max_val

            c.setFillColor(VIOLET)
            c.roundRect(pad_left, y, bar_w, bar_h, 1.5, fill=1, stroke=0)

            c.setFillColor(SLATE_700)
            c.setFont("Helvetica", 7)
            c.drawRightString(pad_left - 2 * mm, y + 1.5 * mm, name)

            c.setFillColor(SLATE_400)
            c.setFont("Helvetica", 6)
            c.drawString(pad_left + bar_w + 2 * mm,
                         y + 1.5 * mm, f"{val:.1f}%")


class ScatterPlot(Flowable):
    """Scatter plot for clusters, segments, and anomalies in latent space."""

    def __init__(self, points, labels=None, width=100 * mm, height=60 * mm,
                 highlight_indices=None, title=None):
        super().__init__()
        self.points = points[:500] if points else []
        self.labels = labels[:500] if labels else None
        self.width = width
        self.height = height
        self.highlight_indices = set(highlight_indices or [])
        self.title = title

    def draw(self):
        c = self.canv
        if not self.points:
            return

        pad = 12 * mm
        plot_w = self.width - pad * 1.5
        plot_h = self.height - pad * 1.5

        xs = [p.get("x", 0) for p in self.points]
        ys = [p.get("y", 0) for p in self.points]
        x_lo, x_hi = min(xs), max(xs)
        y_lo, y_hi = min(ys), max(ys)
        x_span = (x_hi - x_lo) or 1.0
        y_span = (y_hi - y_lo) or 1.0

        c.setStrokeColor(colors.HexColor("#e2e8f0"))
        c.setLineWidth(0.3)
        for i in range(5):
            y = pad + plot_h * i / 4
            c.line(pad, y, pad + plot_w, y)

        if self.labels:
            unique_labels = sorted(set(self.labels))
            label_colors = {lbl: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                            for i, lbl in enumerate(unique_labels)}
        else:
            label_colors = None

        for i, pt in enumerate(self.points):
            px = pad + plot_w * (pt.get("x", 0) - x_lo) / x_span
            py = pad + plot_h * (pt.get("y", 0) - y_lo) / y_span

            if self.highlight_indices and i in self.highlight_indices:
                c.setFillColor(ROSE)
                c.circle(px, py, 2.5, fill=1, stroke=0)
            elif label_colors and self.labels and i < len(self.labels):
                c.setFillColor(label_colors[self.labels[i]])
                c.circle(px, py, 1.5, fill=1, stroke=0)
            else:
                c.setFillColor(VIOLET)
                c.circle(px, py, 1.5, fill=1, stroke=0)

        c.setFillColor(SLATE_400)
        c.setFont("Helvetica", 6)
        c.drawString(pad, 3 * mm, f"n={len(self.points)} points")

        if self.title:
            c.setFillColor(SLATE_700)
            c.setFont("Helvetica-Bold", 8)
            c.drawCentredString(
                self.width / 2, self.height - 4 * mm, self.title)


class HeatmapGrid(Flowable):
    """Correlation heatmap as a colored grid."""

    def __init__(self, matrix, columns, width, height=60 * mm, max_cols=12):
        super().__init__()
        self.matrix = matrix
        self.columns = columns[:max_cols] if columns else []
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

        pad_left = 30 * mm
        pad_top = 8 * mm
        cell = min(8 * mm, (self.width - pad_left) / n)
        grid_h = n * cell

        def corr_color(val):
            r = abs(val)
            if val >= 0.8:
                return colors.HexColor("#166534")
            elif val >= 0.5:
                return colors.HexColor("#22c55e")
            elif val >= 0.2:
                return colors.HexColor("#86efac")
            elif val <= -0.8:
                return colors.HexColor("#991b1b")
            elif val <= -0.5:
                return colors.HexColor("#ef4444")
            elif val <= -0.2:
                return colors.HexColor("#fca5a5")
            else:
                return colors.HexColor("#f1f5f9")

        for i in range(n):
            for j in range(n):
                val = self.matrix[i][j] if i < len(
                    self.matrix) and j < len(self.matrix[i]) else 0
                x = pad_left + j * cell
                y = self.height - pad_top - (i + 1) * cell
                c.setFillColor(corr_color(val))
                c.rect(x, y, cell - 0.5, cell - 0.5, fill=1, stroke=0)

        c.setFillColor(SLATE_700)
        c.setFont("Helvetica", 5)
        for i, col in enumerate(self.columns):
            y = self.height - pad_top - (i + 1) * cell + cell / 2
            c.drawRightString(pad_left - 2 * mm, y, col[:15])
            c.drawString(pad_left + i * cell + cell / 2,
                         self.height - 4 * mm, col[:8])

        c.setFont("Helvetica", 6)
        c.setFillColor(SLATE_400)
        c.drawString(
            pad_left, 3 * mm, "Green: positive correlation | Red: negative | Size indicates strength")


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
    config = (job.config or {}).get("config") if isinstance(
        job.config, dict) and "config" in job.config else job.config or {}
    training = job.training or {}

    # ── Cover ────────────────────────────────────────────────────────────────
    els += [
        Spacer(1, 8 * mm),
        Paragraph("Deep Learning 2.0", st["title"]),
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
         str(profile.get("columns_used", 0)), str(
             profile.get("columns_dropped", 0)),
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
    t = Table(cfg_rows, colWidths=[usable * 0.26,
              usable * 0.24] * 2, repeatRows=1)
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
    t = Table(perf, colWidths=[usable * 0.3,
              usable * 0.2, usable * 0.5], repeatRows=1)
    t.setStyle(_table_style(header_color=EMERALD))
    els += [t, Spacer(1, 6 * mm)]

    # ── Selected patterns with visualizations ─────────────────────────────────────
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
                HRFlowable(width=usable, thickness=2,
                           color=VIOLET, spaceAfter=4),
                Paragraph(f"{sel['title']}{preferred}", st["insight_title"]),
                Paragraph(full.get("description", ""), st["insight_text"]),
                Paragraph(f"<b>Confidence:</b> {sel['confidence'] * 100:.0f}% &nbsp;&nbsp; "
                          f"<b>Columns:</b> {', '.join(sel.get('columns', [])) or '—'}",
                          st["insight_text"]),
            ]
            if sel.get("recommendation"):
                block.append(Paragraph(f"<b>What to do:</b> {sel['recommendation']}",
                                       st["insight_text"]))
            block.append(Spacer(1, 3 * mm))

            pdata = full.get("data", {})
            pviz = full.get("visualization")
            ptype = full.get("type", "")

            if pviz == "bar" and pdata.get("items"):
                block.append(
                    BarChart(pdata["items"], usable, value_key="error_pct"))
                block.append(Spacer(1, 3 * mm))

            elif pviz == "scatter" and pdata.get("points"):
                highlight = pdata.get(
                    "indices") if ptype == "anomalies" else None
                title = "Anomaly Distribution" if ptype == "anomalies" else "Cluster Structure"
                block.append(ScatterPlot(pdata["points"], pdata.get("labels"),
                                         usable, 60 * mm, highlight, title))
                block.append(Spacer(1, 3 * mm))

                if ptype in ("clusters", "segments") and pdata.get("profiles"):
                    block.append(
                        Paragraph("<b>Cluster Profiles:</b>", st["body"]))
                    for prof in pdata["profiles"]:
                        traits = ", ".join(
                            f"{t['feature']} ({'+' if t['z'] > 0 else ''}{t['z']:.2f}σ)"
                            for t in prof.get("traits", [])[:3]
                        )
                        block.append(Paragraph(
                            f"Cluster {prof['cluster']}: {prof['size']} rows ({prof['share']*100:.1f}%) — {traits}",
                            st["body"]))
                    block.append(Spacer(1, 2 * mm))

                if ptype == "anomalies":
                    count = pdata.get("count", 0)
                    share = pdata.get("share", 0)
                    ratio = pdata.get("error_ratio", 1)
                    drivers = pdata.get("drivers", [])
                    block.append(Paragraph(
                        f"<b>Details:</b> {count} anomalous rows ({share*100:.1f}% of data) "
                        f"reconstruct {ratio:.1f}x worse than normal.", st["body"]))
                    if drivers:
                        driver_text = ", ".join(
                            f"{d['feature']} ({'+' if d['deviation'] > 0 else ''}{d['deviation']:.2f}σ)"
                            for d in drivers[:4]
                        )
                        block.append(
                            Paragraph(f"<b>Key drivers:</b> {driver_text}", st["body"]))

            elif pviz == "heatmap":
                pairs = pdata.get("pairs", [])
                matrix = pdata.get("matrix")
                cols = pdata.get("columns", [])
                if pairs:
                    block.append(
                        Paragraph("<b>Correlated Pairs:</b>", st["body"]))
                    pair_rows = [["Feature A", "Feature B", "Correlation"]]
                    for p in pairs[:8]:
                        pair_rows.append([p.get("a", ""), p.get(
                            "b", ""), f"{p.get('r', 0):.3f}"])
                    pt = Table(pair_rows, colWidths=[usable * 0.35, usable * 0.35, usable * 0.3],
                               repeatRows=1)
                    pt.setStyle(_table_style(header_color=EMERALD))
                    block.append(pt)
                    block.append(Spacer(1, 2 * mm))
                if matrix and cols and len(cols) <= 15:
                    block.append(
                        Paragraph("<b>Correlation Matrix:</b>", st["body"]))
                    block.append(HeatmapGrid(matrix, cols, usable))
                    block.append(Spacer(1, 3 * mm))

            if ptype == "compressibility" and pdata:
                latent = pdata.get("latent_dim", 0)
                n_feat = pdata.get("n_features", 0)
                reduction = pdata.get("reduction_pct", 0)
                block.append(Paragraph(
                    f"<b>Compression:</b> {n_feat} features → {latent} latent dimensions "
                    f"({reduction:.1f}% reduction)", st["body"]))

            if ptype in ("non_linear", "linear_structure") and pdata:
                ae_err = pdata.get("ae_error", 0)
                pca_err = pdata.get("pca_error", 0)
                gain = pdata.get("gain", 0)
                block.append(Paragraph(
                    f"<b>Error comparison:</b> Autoencoder = {ae_err:.4f}, PCA = {pca_err:.4f} "
                    f"(difference: {'+' if gain > 0 else ''}{gain*100:.1f}%)", st["body"]))

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
        rows = [["Column", "Reason"]] + [[e["column"], e["reason"]]
                                         for e in excluded]
        t = Table(rows, colWidths=[usable * 0.35, usable * 0.65], repeatRows=1)
        t.setStyle(_table_style(header_color=AMBER))
        els += [t, Spacer(1, 5 * mm)]

    # ── All discovered patterns with details ───────────────────────────────────
    if job.patterns:
        els.append(Paragraph("All Discovered Patterns", st["h2"]))
        for idx, p in enumerate(job.patterns):
            pblock = [
                HRFlowable(width=usable, thickness=1,
                           color=SLATE_400, spaceAfter=2),
                Paragraph(f"<b>{p['title']}</b> ({p['type'].replace('_', ' ')}, {p['confidence']*100:.0f}%)",
                          st["body"]),
            ]
            if p.get("description"):
                pblock.append(
                    Paragraph(p["description"][:200], st["insight_text"]))

            pdata = p.get("data", {})
            pviz = p.get("visualization")
            ptype = p.get("type", "")

            if pviz == "bar" and pdata.get("items"):
                pblock.append(Spacer(1, 2 * mm))
                pblock.append(BarChart(pdata["items"], usable * 0.8, 40 * mm, max_items=6,
                                       value_key="error_pct"))
            elif pviz == "scatter" and pdata.get("points"):
                pblock.append(Spacer(1, 2 * mm))
                highlight = pdata.get(
                    "indices") if ptype == "anomalies" else None
                pblock.append(ScatterPlot(pdata["points"], pdata.get("labels"),
                                          usable * 0.75, 45 * mm, highlight))
                if ptype == "anomalies":
                    pblock.append(Paragraph(
                        f"<font size='7'>{pdata.get('count', 0)} anomalies, "
                        f"{pdata.get('share', 0)*100:.1f}% of data</font>", st["body"]))
                elif ptype in ("clusters", "segments") and pdata.get("profiles"):
                    n_clusters = len(pdata["profiles"])
                    sizes = [f"{p['size']}" for p in pdata["profiles"]]
                    pblock.append(Paragraph(
                        f"<font size='7'>{n_clusters} clusters: {', '.join(sizes)} rows each</font>",
                        st["body"]))
            elif pviz == "heatmap" and pdata.get("pairs"):
                pair_list = ", ".join(f"{p['a']}:{p['b']}({p['r']:.2f})"
                                      for p in pdata["pairs"][:4])
                pblock.append(
                    Paragraph(f"<font size='7'>Pairs: {pair_list}</font>", st["body"]))

            pblock.append(Spacer(1, 3 * mm))
            els.append(KeepTogether(pblock))
        els.append(Spacer(1, 5 * mm))

    els += [
        HRFlowable(width=usable, thickness=0.5, color=SLATE_700),
        Spacer(1, 3 * mm),
        Paragraph(f"Generated by DataForge AI · Deep Learning 2.0 · "
                  f"{datetime.now().strftime('%Y-%m-%d')}",
                  ParagraphStyle("f", fontSize=7, textColor=SLATE_700,
                                 fontName="Helvetica", alignment=TA_CENTER)),
    ]

    doc.build(els, onFirstPage=_header, onLaterPages=_header)
    return buf.getvalue()
