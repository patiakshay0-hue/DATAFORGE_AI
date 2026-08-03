"""SignalBundle — one heavy numeric pass over the dataset.

Spec §4 asks for nine categories of "hidden pattern". Computing each independently
would re-run the expensive parts (KMeans sweep, PCA, mutual information) nine times.

Instead this module does the heavy work **once** and hands back a bundle of raw
signals; `patterns.py` then interprets that bundle nine different ways with cheap
pure functions. O(1 heavy pass) instead of O(9).

Unlike `deep_trainer.discover_patterns`, this operates on the **encoded** feature
matrix from `_prepare_rich`, so categorical columns take part in clustering,
segmentation and anomaly detection rather than being silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Keep the heavy passes bounded regardless of dataset size.
SAMPLE_CAP = 5_000
MAX_CLUSTER_K = 8
MAX_INTERACTION_FEATURES = 8

# Minimum rows per cluster for the grouping to be worth reporting.
MIN_CLUSTER_MEMBERS = 20
# How much silhouette a larger k must gain to justify the extra complexity.
SILHOUETTE_TOLERANCE = 0.03


@dataclass
class SignalBundle:
    """Everything the pattern detectors need, computed once."""

    feature_names: list[str]
    task: str
    n_samples: int

    # Linear and monotonic association with the target.
    pearson: dict[str, float] = field(default_factory=dict)
    spearman: dict[str, float] = field(default_factory=dict)
    mutual_info: dict[str, float] = field(default_factory=dict)

    # Feature-to-feature correlation (for redundancy / correlation patterns).
    corr_matrix: list[list[float]] = field(default_factory=list)
    corr_columns: list[str] = field(default_factory=list)
    collinear_pairs: list[dict] = field(default_factory=list)

    # Structure.
    pca: dict = field(default_factory=dict)
    clusters: dict = field(default_factory=dict)
    anomalies: dict = field(default_factory=dict)

    # Model-derived.
    permutation_importance: list[dict] = field(default_factory=list)
    interactions: list[dict] = field(default_factory=list)

    variance_rank: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "task": self.task,
            "n_samples": self.n_samples,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "mutual_info": self.mutual_info,
            "corr_matrix": self.corr_matrix,
            "corr_columns": self.corr_columns,
            "collinear_pairs": self.collinear_pairs,
            "pca": self.pca,
            "clusters": self.clusters,
            "anomalies": self.anomalies,
            "permutation_importance": self.permutation_importance,
            "interactions": self.interactions,
            "variance_rank": self.variance_rank,
        }


def _safe(value, default=0.0) -> float:
    """NaN/inf-proof float conversion — these feed JSON responses."""
    try:
        f = float(value)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _subsample(X: np.ndarray, y: np.ndarray, cap: int = SAMPLE_CAP):
    if len(X) <= cap:
        return X, y
    idx = np.random.RandomState(42).choice(len(X), cap, replace=False)
    return X[idx], y[idx]


def _associations(X: np.ndarray, y: np.ndarray, names: list[str], task: str):
    """Pearson (linear), Spearman (monotonic) and mutual information (any dependence).

    The gap between them is what later identifies non-linear relationships: a feature
    with high mutual information but near-zero Pearson is non-linearly informative.
    """
    pearson, spearman, minfo = {}, {}, {}

    df = pd.DataFrame(X, columns=names)
    y_s = pd.Series(y)

    for i, name in enumerate(names):
        col = df[name]
        if col.nunique() <= 1:
            pearson[name] = spearman[name] = 0.0
            continue
        pearson[name] = round(_safe(col.corr(y_s, method="pearson")), 4)
        spearman[name] = round(_safe(col.corr(y_s, method="spearman")), 4)

    # Mutual information captures non-monotonic dependence that correlation misses.
    try:
        fn = mutual_info_classif if task == "classification" else mutual_info_regression
        scores = fn(X, y, random_state=42)
        top = max(scores) if len(scores) and max(scores) > 0 else 1.0
        for name, s in zip(names, scores):
            minfo[name] = round(_safe(s / top), 4)   # normalised 0-1
    except Exception:
        minfo = {n: 0.0 for n in names}

    return pearson, spearman, minfo


def _correlation_structure(X: np.ndarray, names: list[str], threshold: float = 0.8):
    """Feature-to-feature correlation plus the strongly collinear pairs."""
    df = pd.DataFrame(X, columns=names)
    corr = df.corr(method="pearson").fillna(0.0)

    matrix = [[round(_safe(corr.iloc[i, j]), 4) for j in range(len(names))]
              for i in range(len(names))]

    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = _safe(corr.iloc[i, j])
            if abs(r) >= threshold:
                pairs.append({"a": names[i], "b": names[j], "r": round(r, 4)})
    pairs.sort(key=lambda p: -abs(p["r"]))
    return matrix, names, pairs[:20]


def _pca(Xs: np.ndarray, names: list[str]):
    n_components = int(min(3, Xs.shape[1], Xs.shape[0]))
    if n_components < 1:
        return {}
    model = PCA(n_components=n_components, random_state=42)
    proj = model.fit_transform(Xs)

    loadings = []
    for i, name in enumerate(names):
        loadings.append({
            "feature": name,
            "pc1": round(_safe(model.components_[0][i]), 4),
            "pc2": round(_safe(model.components_[1][i]), 4) if n_components >= 2 else 0.0,
        })

    return {
        "explained_variance_ratio": [round(_safe(v), 4) for v in model.explained_variance_ratio_],
        "cumulative": round(_safe(model.explained_variance_ratio_.sum()), 4),
        "n_components": n_components,
        "loadings": loadings,
        "points": [{"x": round(_safe(r[0]), 4),
                    "y": round(_safe(r[1]), 4) if n_components >= 2 else 0.0}
                   for r in proj[:500]],
    }


def _clusters(Xs: np.ndarray, names: list[str], y: np.ndarray, task: str):
    """Sweep k, pick the best silhouette, then profile each cluster.

    The profile is what turns an anonymous cluster into a describable segment: which
    features are unusually high or low, and how the target behaves inside it.
    """
    if len(Xs) < 10:
        return {}

    # A cluster needs enough members to mean anything. Without this a 30-row dataset
    # happily "finds" 8 groups of four rows each.
    k_max = int(min(MAX_CLUSTER_K, max(2, len(Xs) // MIN_CLUSTER_MEMBERS)))

    elbow, scored = [], []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=3)
        labels = km.fit_predict(Xs)
        elbow.append({"k": k, "inertia": round(_safe(km.inertia_), 2)})
        if len(set(labels)) > 1:
            scored.append((k, _safe(silhouette_score(Xs, labels), -1.0)))

    if not scored:
        return {}

    # Prefer parsimony: take the *smallest* k whose silhouette is within tolerance of
    # the best. Raw argmax drifts to large k, splitting one real group into shards
    # that score marginally higher but describe nothing.
    best_score = max(s for _, s in scored)
    best_k = min(k for k, s in scored if s >= best_score - SILHOUETTE_TOLERANCE)
    best_score = dict(scored)[best_k]

    km = KMeans(n_clusters=best_k, random_state=42, n_init=5)
    labels = km.fit_predict(Xs)

    # Per-cluster profile: deviation from the global mean, in standard deviations.
    profiles = []
    for c in range(best_k):
        mask = labels == c
        if not mask.any():
            continue
        centroid = Xs[mask].mean(axis=0)
        traits = sorted(
            ({"feature": n, "z": round(_safe(centroid[i]), 3)} for i, n in enumerate(names)),
            key=lambda t: -abs(t["z"]),
        )[:5]
        entry = {
            "cluster": c,
            "size": int(mask.sum()),
            "share": round(float(mask.sum() / len(Xs)), 4),
            "traits": traits,
        }
        if task == "classification":
            vals, counts = np.unique(y[mask], return_counts=True)
            entry["dominant_class"] = int(vals[counts.argmax()])
            entry["purity"] = round(float(counts.max() / counts.sum()), 4)
        else:
            entry["target_mean"] = round(_safe(y[mask].mean()), 4)
        profiles.append(entry)

    return {
        "n_clusters": best_k,
        "silhouette_score": round(best_score, 4) if best_score > -1 else None,
        "elbow": elbow,
        "labels": labels[:500].tolist(),
        "profiles": profiles,
    }


def _anomalies(Xs: np.ndarray, names: list[str], y: np.ndarray, task: str):
    """Isolation Forest — points that are cheap to isolate are outliers."""
    if len(Xs) < 20:
        return {}
    try:
        iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        flags = iso.fit_predict(Xs)          # -1 == outlier
        scores = iso.score_samples(Xs)
        mask = flags == -1
        if not mask.any():
            return {}

        # What makes them anomalous: features furthest from the inlier mean.
        inlier_mean = Xs[~mask].mean(axis=0) if (~mask).any() else Xs.mean(axis=0)
        outlier_mean = Xs[mask].mean(axis=0)
        drivers = sorted(
            ({"feature": n, "deviation": round(_safe(outlier_mean[i] - inlier_mean[i]), 3)}
             for i, n in enumerate(names)),
            key=lambda d: -abs(d["deviation"]),
        )[:5]

        out = {
            "count": int(mask.sum()),
            "share": round(float(mask.sum() / len(Xs)), 4),
            "mean_score": round(_safe(scores[mask].mean()), 4),
            "drivers": drivers,
        }
        if task == "classification" and mask.any():
            vals, counts = np.unique(y[mask], return_counts=True)
            out["dominant_class"] = int(vals[counts.argmax()])
        return out
    except Exception:
        return {}


def _interactions(X: np.ndarray, y: np.ndarray, names: list[str],
                  importance: list[dict], task: str):
    """Find feature pairs that predict jointly better than either does alone.

    Cheap synergy proxy: correlate the product of two standardised features with the
    target and compare against the better of the two individual correlations. A pair
    whose product beats both singles carries interaction signal.

    Restricted to the top few features by importance so this stays O(k^2) with small k.
    """
    ranked = [d["feature"] for d in importance][:MAX_INTERACTION_FEATURES]
    if len(ranked) < 2:
        return []

    idx = {n: i for i, n in enumerate(names)}
    Xz = StandardScaler().fit_transform(X)
    y_s = pd.Series(y.astype(float))

    results = []
    for a_pos in range(len(ranked)):
        for b_pos in range(a_pos + 1, len(ranked)):
            a, b = ranked[a_pos], ranked[b_pos]
            if a not in idx or b not in idx:
                continue
            va, vb = Xz[:, idx[a]], Xz[:, idx[b]]
            solo = max(abs(_safe(pd.Series(va).corr(y_s))),
                       abs(_safe(pd.Series(vb).corr(y_s))))
            joint = abs(_safe(pd.Series(va * vb).corr(y_s)))
            gain = joint - solo
            if gain > 0.02:
                results.append({
                    "features": [a, b],
                    "joint": round(joint, 4),
                    "best_single": round(solo, 4),
                    "gain": round(gain, 4),
                })

    results.sort(key=lambda r: -r["gain"])
    return results[:10]


def build(X: np.ndarray, y: np.ndarray, feature_names: list[str], task: str,
          permutation_importance: list[dict] | None = None) -> SignalBundle:
    """Run every heavy computation once and return the bundle.

    X must be the encoded feature matrix from `deep_trainer._prepare_rich` — that is
    what brings categorical columns into clustering and anomaly detection.
    """
    permutation_importance = permutation_importance or []

    Xw, yw = _subsample(np.asarray(X, dtype=float), np.asarray(y))
    Xs = StandardScaler().fit_transform(Xw)

    pearson, spearman, minfo = _associations(Xw, yw, feature_names, task)
    corr_matrix, corr_columns, collinear = _correlation_structure(Xw, feature_names)

    variance = np.var(Xw, axis=0)
    variance_rank = sorted(
        ({"feature": n, "variance": round(_safe(variance[i]), 4)}
         for i, n in enumerate(feature_names)),
        key=lambda v: -v["variance"],
    )

    return SignalBundle(
        feature_names=list(feature_names),
        task=task,
        n_samples=int(len(Xw)),
        pearson=pearson,
        spearman=spearman,
        mutual_info=minfo,
        corr_matrix=corr_matrix,
        corr_columns=corr_columns,
        collinear_pairs=collinear,
        pca=_pca(Xs, feature_names),
        clusters=_clusters(Xs, feature_names, yw, task),
        anomalies=_anomalies(Xs, feature_names, yw, task),
        permutation_importance=permutation_importance,
        interactions=_interactions(Xw, yw, feature_names, permutation_importance, task),
        variance_rank=variance_rank,
    )
