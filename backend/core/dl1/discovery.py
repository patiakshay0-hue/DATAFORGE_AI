"""Unsupervised pattern discovery — hidden structure found without any target.

The supervised detectors in `patterns.py` all lean on a target column (correlation
with y, permutation importance, class purity). None of that exists here, so this
module derives its evidence from the trained autoencoder instead:

  * per-feature reconstruction error  → which columns carry unique information
  * per-row reconstruction error      → anomalies
  * the latent code                   → clusters and segments in learned space
  * autoencoder error vs PCA error    → whether the structure is non-linear
  * latent width vs feature count     → how compressible the data really is

Plus two purely statistical detectors (collinearity, mutual-information coupling)
that need no model at all.

Each pattern is a claim the user can accept or reject, carrying the columns it
implicates — which is what drives the "features to use" section of the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import silhouette_score

MIN_CLUSTER_MEMBERS = 20
MAX_CLUSTER_K = 8
SILHOUETTE_TOLERANCE = 0.03
ANOMALY_PERCENTILE = 97.5
MAX_MI_FEATURES = 12
MAX_MI_ROWS = 5000
MAX_CLUSTER_ROWS = 10000


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _safe(v, default=0.0) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _pattern(pid, ptype, title, description, confidence, columns,
             visualization=None, data=None, recommendation=None):
    return {
        "id": pid,
        "type": ptype,
        "title": title,
        "description": description,
        "confidence": round(_clamp(confidence), 3),
        "columns": list(columns),
        "visualization": visualization,
        "recommendation": recommendation,
        "data": data or {},
    }


# ── Latent-space clustering (shared by two detectors) ─────────────────────────
def _cluster_latent(latent: np.ndarray, names: list[str], X: np.ndarray):
    """K-means over the learned code, with per-cluster profiles in original terms.

    Clustering the latent space rather than the raw columns is the point of using a
    network at all: groups that are tangled in the input space are often linearly
    separable once the encoder has unfolded them.
    """
    if len(latent) < MIN_CLUSTER_MEMBERS * 2:
        return {}

    # Sample for clustering to avoid O(n) blow-up on large datasets
    if len(latent) > MAX_CLUSTER_ROWS:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(latent), MAX_CLUSTER_ROWS, replace=False)
        latent_sample = latent[sample_idx]
        X_sample = X[sample_idx]
    else:
        latent_sample = latent
        X_sample = X
        sample_idx = None

    k_max = int(min(MAX_CLUSTER_K, max(2, len(latent_sample) // MIN_CLUSTER_MEMBERS)))
    scored, elbow = [], []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=3)
        labels = km.fit_predict(latent_sample)
        elbow.append({"k": k, "inertia": round(_safe(km.inertia_), 2)})
        if len(set(labels)) > 1:
            scored.append((k, _safe(silhouette_score(latent_sample, labels), -1.0)))

    if not scored:
        return {}

    best_score = max(s for _, s in scored)
    # Prefer the simplest grouping that is essentially as good (see signals.py).
    best_k = min(k for k, s in scored if s >= best_score - SILHOUETTE_TOLERANCE)
    best_score = dict(scored)[best_k]

    km = KMeans(n_clusters=best_k, random_state=42, n_init=5)
    sample_labels = km.fit_predict(latent_sample)

    # Assign labels to full dataset using nearest centroid
    if sample_idx is not None:
        labels = km.predict(latent)
    else:
        labels = sample_labels

    profiles = []
    global_mean, global_std = X.mean(axis=0), X.std(axis=0) + 1e-9
    for c in range(best_k):
        mask = labels == c
        if not mask.any():
            continue
        z = (X[mask].mean(axis=0) - global_mean) / global_std
        traits = sorted(
            ({"feature": n, "z": round(_safe(z[i]), 3)} for i, n in enumerate(names)),
            key=lambda t: -abs(t["z"]),
        )[:5]
        profiles.append({
            "cluster": int(c),
            "size": int(mask.sum()),
            "share": round(float(mask.sum() / len(latent)), 4),
            "traits": traits,
        })

    # 2-D projection for plotting the groups (on sample for speed)
    if latent_sample.shape[1] >= 2:
        pts = PCA(n_components=2, random_state=42).fit_transform(latent_sample)
    else:
        pts = np.column_stack([latent_sample[:, 0], np.zeros(len(latent_sample))])

    return {
        "n_clusters": best_k,
        "silhouette_score": round(best_score, 4),
        "elbow": elbow,
        "labels": labels[:500].tolist(),
        "profiles": profiles,
        "points": [{"x": round(_safe(p[0]), 4), "y": round(_safe(p[1]), 4)}
                   for p in pts[:500]],
    }


# ── 1. Which features carry unique information ────────────────────────────────
def _information_value(ctx) -> list[dict]:
    fe = ctx["feature_error"]
    if len(fe) < 2:
        return []
    top = fe[:5]
    lead = top[0]
    share = sum(f["error_pct"] for f in top[:3])

    return [_pattern(
        "information_value", "information_value",
        f"'{lead['feature']}' carries the most unique information",
        f"The autoencoder reconstructs {lead['feature']} least accurately "
        f"({lead['error_pct']:.1f}% of total reconstruction error), meaning it cannot be "
        f"inferred from the other columns — it holds independent signal. The top "
        f"{min(3, len(top))} such columns account for {share:.0f}% of the error.",
        confidence=_clamp(0.5 + share / 200),
        columns=[f["feature"] for f in top],
        visualization="bar",
        data={"items": fe},
        recommendation="Keep these columns — they are the backbone of any model built on this data.",
    )]


# ── 2. Redundant features ─────────────────────────────────────────────────────
def _redundancy(ctx) -> list[dict]:
    """Near-zero reconstruction error means the column is predictable from the rest."""
    fe = ctx["feature_error"]
    if len(fe) < 3:
        return []
    errors = np.array([f["reconstruction_error"] for f in fe])
    if errors.max() <= 0:
        return []
    cutoff = errors.max() * 0.15
    redundant = [f for f in fe if f["reconstruction_error"] <= cutoff]
    if not redundant:
        return []

    names = ", ".join(f["feature"] for f in redundant[:4])
    return [_pattern(
        "redundancy", "redundancy",
        f"{len(redundant)} column(s) duplicate information already present",
        f"{names} are reconstructed almost perfectly from the other columns, so they add "
        f"little the model cannot already infer. Dropping them usually simplifies a model "
        f"without measurable loss — though keep any that are cheap to collect and easy "
        f"to interpret.",
        confidence=_clamp(0.5 + (1 - cutoff / (errors.max() or 1)) / 2),
        columns=[f["feature"] for f in redundant],
        visualization="bar",
        data={"items": redundant},
        recommendation="Candidates to drop — verify against domain knowledge first.",
    )]


# ── 3. Hidden clusters ────────────────────────────────────────────────────────
def _clusters(ctx) -> list[dict]:
    c = ctx["clusters"]
    if not c or not c.get("profiles"):
        return []
    sil = c.get("silhouette_score") or 0.0
    biggest = max(c["profiles"], key=lambda p: p["size"])
    traits = ", ".join(t["feature"] for t in biggest.get("traits", [])[:3])

    return [_pattern(
        "clusters", "clusters",
        f"The data falls into {c['n_clusters']} natural groups",
        f"Clustering the network's learned representation reveals {c['n_clusters']} "
        f"distinct groups (silhouette {sil:.2f}) that nobody labelled. The largest covers "
        f"{biggest['share'] * 100:.0f}% of rows and is distinguished mainly by {traits}.",
        confidence=_clamp(sil * 1.6),
        columns=[t["feature"] for t in biggest.get("traits", [])[:4]],
        visualization="scatter",
        data=c,
        recommendation="Use the columns above to describe and target these groups.",
    )]


# ── 4. Data segments ──────────────────────────────────────────────────────────
def _segments(ctx) -> list[dict]:
    """The most *distinctive* group, described in plain feature terms."""
    c = ctx["clusters"]
    profiles = (c or {}).get("profiles") or []
    if len(profiles) < 2:
        return []

    def sharpness(p):
        return max((abs(t["z"]) for t in p.get("traits", [])), default=0.0)

    best = max(profiles, key=sharpness)
    peak = sharpness(best)
    if peak < 0.75:                     # nothing stands out enough to be worth naming
        return []

    traits = best.get("traits", [])[:3]
    described = ", ".join(
        f"{t['feature']} {'well above' if t['z'] > 0 else 'well below'} average"
        for t in traits
    )

    return [_pattern(
        "segments", "segments",
        f"A distinct {best['share'] * 100:.0f}% segment stands apart",
        f"Cluster {best['cluster']} holds {best['size']} rows and is characterised by "
        f"{described} (peak deviation {peak:.1f} standard deviations). A segment this "
        f"sharply defined can usually be acted on with a simple rule rather than a model.",
        confidence=_clamp(0.4 + peak / 4),
        columns=[t["feature"] for t in traits],
        visualization="scatter",
        data={"profiles": profiles, "points": (c or {}).get("points", []),
              "labels": (c or {}).get("labels", [])},
        recommendation="Segment on these columns to isolate this group.",
    )]


# ── 5. Anomaly groups ─────────────────────────────────────────────────────────
def _anomalies(ctx) -> list[dict]:
    row_error = ctx["row_error"]
    names, X = ctx["feature_names"], ctx["X"]
    if len(row_error) < 20:
        return []

    threshold = float(np.percentile(row_error, ANOMALY_PERCENTILE))
    mask = row_error > threshold
    if not mask.any():
        return []

    # What makes them unusual: largest standardised gap from the normal rows.
    normal_mean = X[~mask].mean(axis=0) if (~mask).any() else X.mean(axis=0)
    normal_std = X[~mask].std(axis=0) + 1e-9 if (~mask).any() else X.std(axis=0) + 1e-9
    z = (X[mask].mean(axis=0) - normal_mean) / normal_std
    drivers = sorted(
        ({"feature": n, "deviation": round(_safe(z[i]), 3)} for i, n in enumerate(names)),
        key=lambda d: -abs(d["deviation"]),
    )[:5]

    ratio = float(row_error[mask].mean() / (row_error[~mask].mean() + 1e-12)) if (~mask).any() else 1.0
    driver_names = ", ".join(d["feature"] for d in drivers[:3])

    return [_pattern(
        "anomalies", "anomalies",
        f"{int(mask.sum())} rows do not fit the learned structure",
        f"These rows reconstruct {ratio:.1f}x worse than the rest, meaning the network "
        f"could not express them in terms of the patterns it learned. They deviate most on "
        f"{driver_names}. Such rows are either genuinely rare cases worth studying or data "
        f"errors worth correcting.",
        confidence=_clamp(0.45 + min(0.5, ratio / 20)),
        columns=[d["feature"] for d in drivers[:4]],
        visualization="scatter",
        data={"count": int(mask.sum()),
              "share": round(float(mask.sum() / len(row_error)), 4),
              "threshold": round(threshold, 6),
              "error_ratio": round(ratio, 2),
              "drivers": drivers,
              "indices": np.flatnonzero(mask)[:100].tolist()},
        recommendation="Review these rows before training — they distort most models.",
    )]


# ── 6. Non-linear structure ───────────────────────────────────────────────────
def _non_linear(ctx) -> list[dict]:
    gain = ctx.get("nonlinear_gain")
    if gain is None or gain <= 0.05:
        return []

    return [_pattern(
        "non_linear", "non_linear",
        "The structure in this data is non-linear",
        f"At the same compression width, the neural autoencoder reconstructs the data "
        f"{gain * 100:.0f}% more accurately than PCA, a purely linear method "
        f"(error {ctx['ae_error']:.4f} vs {ctx['pca_error']:.4f}). Linear models will "
        f"systematically underperform here — the relationships bend.",
        confidence=_clamp(0.5 + gain),
        columns=[f["feature"] for f in ctx["feature_error"][:4]],
        visualization="bar",
        data={"ae_error": ctx["ae_error"], "pca_error": ctx["pca_error"], "gain": gain},
        recommendation="Prefer neural or tree-based models over linear regression here.",
    )]


# ── 7. Intrinsic dimensionality ───────────────────────────────────────────────
def _linear_structure(ctx) -> list[dict]:
    """The converse of _non_linear: a linear projection matches or beats the network.

    Worth stating plainly. It means the relationships here are straight, and simple,
    fast, interpretable models will do the job — a genuinely useful finding rather
    than a failure.
    """
    gain = ctx.get("nonlinear_gain")
    if gain is None or gain > -0.05:
        return []

    return [_pattern(
        "linear_structure", "linear_structure",
        "The structure in this data is essentially linear",
        f"A linear projection (PCA) reconstructs this data at least as accurately as the "
        f"neural network at the same compression width "
        f"(error {ctx['pca_error']:.4f} vs {ctx['ae_error']:.4f}). The relationships here "
        f"are straight lines, not curves — so simpler models will match a neural network "
        f"while training faster and staying interpretable.",
        confidence=_clamp(0.6 + min(0.35, abs(gain) / 20)),
        columns=[f["feature"] for f in ctx["feature_error"][:4]],
        visualization="bar",
        data={"ae_error": ctx["ae_error"], "pca_error": ctx["pca_error"]},
        recommendation="Linear or tree-based models are a better fit than a deep network here.",
    )]


def _compressibility(ctx) -> list[dict]:
    latent_dim, n_features = ctx["latent_dim"], len(ctx["feature_names"])
    if n_features < 3:
        return []
    ratio = latent_dim / n_features
    if ratio > 0.7:                    # barely compressible — not an interesting claim
        return []

    return [_pattern(
        "compressibility", "compressibility",
        f"{n_features} columns really contain about {latent_dim} dimensions of information",
        f"The autoencoder compresses all {n_features} features into {latent_dim} latent "
        f"dimension(s) and still reconstructs them accurately — a {(1 - ratio) * 100:.0f}% "
        f"reduction. The columns are far from independent; a handful of underlying factors "
        f"drives most of what you see.",
        confidence=_clamp(0.5 + (1 - ratio) / 2),
        columns=[f["feature"] for f in ctx["feature_error"][:5]],
        visualization="bar",
        data={"latent_dim": latent_dim, "n_features": n_features,
              "reduction_pct": round((1 - ratio) * 100, 1)},
        recommendation="A compact feature set will capture nearly everything here.",
    )]


# ── 8. Correlated pairs ───────────────────────────────────────────────────────
def _correlation(ctx) -> list[dict]:
    pairs = ctx["collinear_pairs"]
    if not pairs:
        return []
    lead = pairs[0]
    extra = f" {len(pairs) - 1} further pair(s) are similarly linked." if len(pairs) > 1 else ""

    return [_pattern(
        "correlation", "correlation",
        f"'{lead['a']}' and '{lead['b']}' move together",
        f"These columns correlate at r = {lead['r']:.2f}. Keeping both adds little "
        f"independent information and can destabilise feature-importance estimates, since "
        f"credit gets split arbitrarily between them.{extra}",
        confidence=_clamp(abs(lead["r"])),
        columns=[lead["a"], lead["b"]],
        visualization="heatmap",
        data={"pairs": pairs[:10], "matrix": ctx["corr_matrix"], "columns": ctx["feature_names"]},
        recommendation="Keep one of each pair unless both are independently meaningful.",
    )]


# ── 9. Non-linear feature coupling ────────────────────────────────────────────
def _coupling(ctx) -> list[dict]:
    """Pairs with high mutual information but low correlation — curved dependence."""
    coupled = ctx.get("coupled_pairs") or []
    if not coupled:
        return []
    lead = coupled[0]

    return [_pattern(
        "coupling", "coupling",
        f"'{lead['a']}' and '{lead['b']}' are linked, but not linearly",
        f"These two share substantial mutual information ({lead['mi']:.2f}) while their "
        f"linear correlation is only {lead['r']:.2f}. One predicts the other through a "
        f"curved relationship that a correlation matrix would miss entirely.",
        confidence=_clamp(0.4 + lead["mi"]),
        columns=[lead["a"], lead["b"]],
        visualization="scatter",
        data={"items": coupled[:8]},
        recommendation="Worth plotting directly — the shape of this relationship matters.",
    )]


DETECTORS = [
    _information_value,
    _compressibility,
    _non_linear,
    _linear_structure,
    _clusters,
    _segments,
    _anomalies,
    _coupling,
    _correlation,
    _redundancy,
]


def _statistical_context(X: np.ndarray, names: list[str]):
    """Model-free signals: collinearity and non-linear coupling between features."""
    df = pd.DataFrame(X, columns=names)
    corr = df.corr(method="pearson").fillna(0.0)
    matrix = [[round(_safe(corr.iloc[i, j]), 4) for j in range(len(names))]
              for i in range(len(names))]

    collinear = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = _safe(corr.iloc[i, j])
            if abs(r) >= 0.8:
                collinear.append({"a": names[i], "b": names[j], "r": round(r, 4)})
    collinear.sort(key=lambda p: -abs(p["r"]))

    # Mutual information between feature pairs, restricted to the highest-variance
    # columns so this stays O(k^2) with small k. Sample rows to avoid O(n) blow-up.
    coupled = []
    try:
        order = np.argsort(-X.var(axis=0))[:MAX_MI_FEATURES]
        n_rows = X.shape[0]
        if n_rows > MAX_MI_ROWS:
            idx = np.random.RandomState(42).choice(n_rows, MAX_MI_ROWS, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
        for pos_a in range(len(order)):
            for pos_b in range(pos_a + 1, len(order)):
                ia, ib = int(order[pos_a]), int(order[pos_b])
                r = abs(_safe(corr.iloc[ia, ib]))
                if r >= 0.5:
                    continue
                mi = float(mutual_info_regression(
                    X_sample[:, [ia]], X_sample[:, ib], random_state=42)[0])
                if mi > 0.3:
                    coupled.append({"a": names[ia], "b": names[ib],
                                    "mi": round(mi, 4), "r": round(r, 4)})
    except Exception:
        coupled = []
    coupled.sort(key=lambda c: -c["mi"])

    return matrix, collinear[:20], coupled[:10]


def discover(encoded, trained: dict) -> tuple[list[dict], dict]:
    """Run every unsupervised detector. Returns (patterns, context_for_report).

    A detector that finds nothing returns an empty list; one that raises is skipped
    rather than losing the whole run.
    """
    X, names = encoded.X, encoded.feature_names
    matrix, collinear, coupled = _statistical_context(X, names)

    ctx = {
        "X": X,
        "feature_names": names,
        "feature_error": trained["feature_error"],
        "row_error": trained["row_error"],
        "latent_dim": trained["config"]["latent_dim"],
        "ae_error": trained["ae_error"],
        "pca_error": trained["pca_error"],
        "nonlinear_gain": trained["nonlinear_gain"],
        "clusters": _cluster_latent(trained["latent"], names, X),
        "corr_matrix": matrix,
        "collinear_pairs": collinear,
        "coupled_pairs": coupled,
    }

    patterns: list[dict] = []
    for detect in DETECTORS:
        try:
            patterns.extend(detect(ctx))
        except Exception:
            continue

    patterns.sort(key=lambda p: -p["confidence"])

    # Everything the report needs, minus the big arrays.
    context = {
        "clusters": ctx["clusters"],
        "corr_matrix": matrix,
        "corr_columns": names,
        "collinear_pairs": collinear,
        "coupled_pairs": coupled,
    }
    return patterns, context
