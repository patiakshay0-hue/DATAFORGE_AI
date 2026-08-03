"""Pattern narration — turn a SignalBundle into the human-readable patterns of §4.

Every detector here is a **pure function over already-computed signals**. No detector
touches the dataframe or fits a model, which is what keeps nine pattern categories
cheap after the single heavy pass in `signals.py`.

Each returns zero or more patterns shaped as:

    {id, type, title, description, confidence, columns, visualization}

`confidence` is a 0-1 score meaning "how strongly the data supports this claim" — it
is a calibrated read of the underlying statistic, not a model probability.

Adding a category for a future version means writing one function and appending it to
DETECTORS; nothing else changes.
"""

from __future__ import annotations

import numpy as np

from .signals import SignalBundle


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _pattern(pid, ptype, title, description, confidence, columns, visualization=None, data=None):
    p = {
        "id": pid,
        "type": ptype,
        "title": title,
        "description": description,
        "confidence": round(_clamp(confidence), 3),
        "columns": list(columns),
        "visualization": visualization,
    }
    if data is not None:
        p["data"] = data
    return p


# ── 1. Feature importance ─────────────────────────────────────────────────────
def _feature_importance(b: SignalBundle) -> list[dict]:
    imp = [d for d in b.permutation_importance if d.get("importance_pct", 0) > 0]
    if not imp:
        return []
    top = imp[:5]
    lead = top[0]
    share = sum(d.get("importance_pct", 0) for d in top[:3]) / 100.0

    return [_pattern(
        "importance", "feature_importance",
        f"'{lead['feature']}' dominates the model's predictions",
        f"Permutation testing ranks {lead['feature']} highest at "
        f"{lead.get('importance_pct', 0):.1f}% of total importance. "
        f"The top {min(3, len(top))} features together account for {share * 100:.0f}% "
        f"of predictive power, so the model leans heavily on a small subset of inputs.",
        # Confident when one feature clearly leads the pack.
        confidence=_clamp(0.5 + share / 2),
        columns=[d["feature"] for d in top],
        visualization="bar",
        data={"items": top},
    )]


# ── 2. Correlation patterns ───────────────────────────────────────────────────
def _correlation(b: SignalBundle) -> list[dict]:
    if not b.collinear_pairs:
        return []
    pair = b.collinear_pairs[0]
    others = len(b.collinear_pairs) - 1
    extra = f" {others} further strongly-correlated pair(s) were found." if others > 0 else ""

    return [_pattern(
        "correlation", "correlation",
        f"'{pair['a']}' and '{pair['b']}' carry nearly the same information",
        f"These two features correlate at r = {pair['r']:.2f}. Highly correlated inputs "
        f"add little independent signal and can make importance scores unstable, since "
        f"the model may split credit arbitrarily between them.{extra}",
        confidence=_clamp(abs(pair["r"])),
        columns=[pair["a"], pair["b"]],
        visualization="heatmap",
        data={"pairs": b.collinear_pairs[:10],
              "matrix": b.corr_matrix, "columns": b.corr_columns},
    )]


# ── 3. Hidden clusters ────────────────────────────────────────────────────────
def _clusters(b: SignalBundle) -> list[dict]:
    c = b.clusters
    if not c or not c.get("profiles"):
        return []
    sil = c.get("silhouette_score") or 0.0
    k = c.get("n_clusters", 0)
    biggest = max(c["profiles"], key=lambda p: p["size"])
    traits = ", ".join(t["feature"] for t in biggest.get("traits", [])[:3])

    return [_pattern(
        "clusters", "clusters",
        f"The data separates into {k} natural groups",
        f"K-means found {k} well-separated clusters (silhouette {sil:.2f}). The largest "
        f"holds {biggest['share'] * 100:.0f}% of rows and is characterised mainly by "
        f"{traits}. These groupings were not labelled in the data — the model can exploit "
        f"them even though nobody defined them.",
        # Silhouette above ~0.5 is a genuinely clean separation.
        confidence=_clamp(sil * 1.6),
        columns=[t["feature"] for t in biggest.get("traits", [])[:4]],
        visualization="scatter",
        data={"pca": b.pca, "clusters": c},
    )]


# ── 4. Non-linear relationships ───────────────────────────────────────────────
def _non_linear(b: SignalBundle) -> list[dict]:
    """High mutual information with weak linear correlation means a curved relationship."""
    found = []
    for name in b.feature_names:
        mi = b.mutual_info.get(name, 0.0)
        lin = abs(b.pearson.get(name, 0.0))
        if mi > 0.35 and lin < 0.25:
            found.append({"feature": name, "mutual_info": mi, "pearson": round(lin, 4),
                          "gap": round(mi - lin, 4)})
    if not found:
        return []
    found.sort(key=lambda f: -f["gap"])
    lead = found[0]

    return [_pattern(
        "non_linear", "non_linear",
        f"'{lead['feature']}' relates to the target non-linearly",
        f"{lead['feature']} shows strong mutual information ({lead['mutual_info']:.2f}) "
        f"with the target while its linear correlation is only {lead['pearson']:.2f}. "
        f"A linear model would miss this feature almost entirely — it is exactly the kind "
        f"of relationship a neural network is worth using for.",
        confidence=_clamp(lead["gap"] + 0.3),
        columns=[f["feature"] for f in found[:4]],
        visualization="scatter",
        data={"items": found[:8]},
    )]


# ── 5. Feature interactions ───────────────────────────────────────────────────
def _interactions(b: SignalBundle) -> list[dict]:
    if not b.interactions:
        return []
    top = b.interactions[0]
    a, c = top["features"]

    return [_pattern(
        "interactions", "interactions",
        f"'{a}' and '{c}' matter more together than apart",
        f"Combined, these two features correlate with the target at {top['joint']:.2f}, "
        f"against {top['best_single']:.2f} for the better one alone — a gain of "
        f"{top['gain']:.2f}. Their joint effect is what carries the signal, so treating "
        f"them independently understates both.",
        confidence=_clamp(0.4 + top["gain"] * 2),
        columns=[a, c],
        visualization="bar",
        data={"items": b.interactions[:6]},
    )]


# ── 6. Influential variables ──────────────────────────────────────────────────
def _influential(b: SignalBundle) -> list[dict]:
    """Features with a strong direct association with the target, by any measure."""
    scored = []
    for name in b.feature_names:
        strength = max(abs(b.pearson.get(name, 0.0)),
                       abs(b.spearman.get(name, 0.0)),
                       b.mutual_info.get(name, 0.0))
        if strength >= 0.3:
            scored.append({
                "feature": name,
                "strength": round(strength, 4),
                "pearson": b.pearson.get(name, 0.0),
                "spearman": b.spearman.get(name, 0.0),
                "mutual_info": b.mutual_info.get(name, 0.0),
            })
    if not scored:
        return []
    scored.sort(key=lambda s: -s["strength"])
    lead = scored[0]
    names = ", ".join(s["feature"] for s in scored[:3])

    return [_pattern(
        "influential", "influential",
        f"{len(scored)} variable(s) move the target directly",
        f"{names} show the strongest direct association with the target "
        f"(peak strength {lead['strength']:.2f}). These are the levers most likely to "
        f"change the outcome if acted upon.",
        confidence=_clamp(lead["strength"]),
        columns=[s["feature"] for s in scored[:5]],
        visualization="bar",
        data={"items": scored[:10]},
    )]


# ── 7. Data segments ──────────────────────────────────────────────────────────
def _segments(b: SignalBundle) -> list[dict]:
    """Clusters whose target behaviour differs sharply from the rest."""
    profiles = (b.clusters or {}).get("profiles") or []
    if len(profiles) < 2:
        return []

    if b.task == "classification":
        pure = [p for p in profiles if p.get("purity", 0) >= 0.75]
        if not pure:
            return []
        best = max(pure, key=lambda p: p.get("purity", 0))
        traits = ", ".join(t["feature"] for t in best.get("traits", [])[:3])
        return [_pattern(
            "segments", "segments",
            f"A {best['share'] * 100:.0f}% segment is {best['purity'] * 100:.0f}% one class",
            f"Cluster {best['cluster']} covers {best['size']} rows and is "
            f"{best['purity'] * 100:.0f}% class {best.get('dominant_class')}. It is defined "
            f"chiefly by {traits}. A segment this pure is often actionable on its own — "
            f"it can be targeted with a simple rule rather than the full model.",
            confidence=_clamp(best.get("purity", 0)),
            columns=[t["feature"] for t in best.get("traits", [])[:4]],
            visualization="scatter",
            data={"profiles": profiles},
        )]

    means = [p.get("target_mean", 0.0) for p in profiles]
    spread = max(means) - min(means)
    if spread <= 0:
        return []
    hi = max(profiles, key=lambda p: p.get("target_mean", 0.0))
    lo = min(profiles, key=lambda p: p.get("target_mean", 0.0))
    traits = ", ".join(t["feature"] for t in hi.get("traits", [])[:3])
    return [_pattern(
        "segments", "segments",
        "Distinct segments show very different target levels",
        f"Cluster {hi['cluster']} averages {hi.get('target_mean'):.2f} on the target while "
        f"cluster {lo['cluster']} averages {lo.get('target_mean'):.2f}. The high group is "
        f"characterised by {traits}. Segment membership alone explains a large share of the "
        f"variation.",
        confidence=_clamp(0.5 + min(0.5, spread / (abs(max(means)) + 1e-9))),
        columns=[t["feature"] for t in hi.get("traits", [])[:4]],
        visualization="scatter",
        data={"profiles": profiles},
    )]


# ── 8. Anomaly groups ─────────────────────────────────────────────────────────
def _anomalies(b: SignalBundle) -> list[dict]:
    a = b.anomalies
    if not a or not a.get("count"):
        return []
    drivers = ", ".join(d["feature"] for d in a.get("drivers", [])[:3])

    return [_pattern(
        "anomalies", "anomalies",
        f"{a['count']} rows behave unlike the rest",
        f"Isolation Forest flags {a['count']} rows ({a['share'] * 100:.1f}%) as outliers, "
        f"driven mainly by {drivers}. These can be genuine rare cases worth studying or "
        f"data-quality problems worth fixing — either way they distort training if left "
        f"unexamined.",
        # Confidence tracks how cleanly separated the outliers are.
        confidence=_clamp(0.45 + abs(a.get("mean_score", 0)) / 2),
        columns=[d["feature"] for d in a.get("drivers", [])[:4]],
        visualization="scatter",
        data=a,
    )]


# ── 9. Target-driving features ────────────────────────────────────────────────
def _target_drivers(b: SignalBundle) -> list[dict]:
    """Where model importance and statistical association agree — the robust drivers."""
    imp = {d["feature"]: d.get("importance_pct", 0) for d in b.permutation_importance}
    if not imp:
        return []

    agreed = []
    for name, pct in imp.items():
        assoc = max(abs(b.pearson.get(name, 0.0)), b.mutual_info.get(name, 0.0))
        if pct >= 10 and assoc >= 0.2:
            agreed.append({"feature": name, "importance_pct": pct,
                           "association": round(assoc, 4)})
    if not agreed:
        return []
    agreed.sort(key=lambda a: -a["importance_pct"])
    names = ", ".join(a["feature"] for a in agreed[:3])

    return [_pattern(
        "target_drivers", "target_drivers",
        f"{len(agreed)} feature(s) drive the target on both measures",
        f"{names} rank highly in the trained model's permutation importance *and* show "
        f"strong direct association with the target. Agreement between an independent "
        f"statistical measure and the model itself makes these the most trustworthy "
        f"drivers to act on.",
        confidence=_clamp(0.55 + agreed[0]["association"] / 2),
        columns=[a["feature"] for a in agreed[:5]],
        visualization="bar",
        data={"items": agreed[:10]},
    )]


# Order here is the order patterns are presented in the UI.
DETECTORS = [
    _target_drivers,
    _feature_importance,
    _influential,
    _non_linear,
    _interactions,
    _clusters,
    _segments,
    _anomalies,
    _correlation,
]


def discover(bundle: SignalBundle) -> list[dict]:
    """Run every detector over the bundle, newest-confidence-first.

    A detector that finds nothing returns an empty list, and one that raises is
    skipped rather than failing the whole run — a single bad statistic should never
    cost the user their trained model.
    """
    patterns: list[dict] = []
    for detect in DETECTORS:
        try:
            patterns.extend(detect(bundle))
        except Exception:
            continue

    patterns.sort(key=lambda p: -p["confidence"])
    return patterns
