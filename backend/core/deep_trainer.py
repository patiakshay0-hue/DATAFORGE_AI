"""Deep learning engine for tabular data.

Trains a feedforward neural network (MLP) on tabular datasets and reports a
per-epoch training history so the frontend can render a live loss curve.

Uses PyTorch when available for a genuine, fully-controlled training loop and
falls back to scikit-learn's MLP (also a real neural network) when torch is not
installed — mirroring the HAS_XGB pattern in trainer.py.

Beyond training it also powers:
  • recommend_targets  — suggest which column to predict
  • permutation importance + confusion matrix / ROC / residuals in the result
  • predict            — live inference on the last-trained network (playground)
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, roc_curve, auc,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.neural_network import MLPClassifier, MLPRegressor


# ── Config ───────────────────────────────────────────────────────────────────
DEFAULTS = {
    "hidden_layers": [64, 32],
    "epochs": 50,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "batch_size": 32,
}

LIMITS = {
    "epochs": (1, 300),
    "learning_rate": (1e-5, 1.0),
    "dropout": (0.0, 0.9),
    "batch_size": (1, 1024),
    "max_layers": 6,
    "max_units": 512,
}

# The last successfully trained network, kept in memory for the prediction
# playground. Single-user app, mirrors main.py's `data_store` pattern.
LAST_MODEL = {}

# Column-name hints that a column is a label worth predicting / an identifier
_LABEL_HINTS = ["target", "label", "class", "outcome", "churn", "result", "fraud",
                "survived", "default", "approved", "status", "category", "type"]
_ID_HINTS = ["id", "uuid", "guid", "index", "key", "code"]


def _clean_config(cfg: dict | None) -> dict:
    cfg = {**DEFAULTS, **(cfg or {})}
    layers = cfg.get("hidden_layers") or DEFAULTS["hidden_layers"]
    layers = [int(u) for u in layers if int(u) > 0][: LIMITS["max_layers"]]
    layers = [min(u, LIMITS["max_units"]) for u in layers] or [64]
    cfg["hidden_layers"] = layers
    lo, hi = LIMITS["epochs"]
    cfg["epochs"] = int(np.clip(cfg["epochs"], lo, hi))
    lo, hi = LIMITS["learning_rate"]
    cfg["learning_rate"] = float(np.clip(cfg["learning_rate"], lo, hi))
    lo, hi = LIMITS["dropout"]
    cfg["dropout"] = float(np.clip(cfg["dropout"], lo, hi))
    lo, hi = LIMITS["batch_size"]
    cfg["batch_size"] = int(np.clip(cfg["batch_size"], lo, hi))
    return cfg


# ── Feature preparation (keeps encoders + schema, unlike trainer._prepare_data) ─
def _prepare_rich(df: pd.DataFrame, target_column: str):
    """Encode features while retaining the metadata needed for inference.

    Returns X, y, task, feature_names, spec, encoders, target_meta.
    """
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    y_raw = df[target_column]

    feature_names, spec, encoders, cols = [], [], {}, []
    for col in df.columns:
        if col == target_column:
            continue
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        # Non-numeric (object, pandas StringDtype, category, bool) → categorical.
        # Using is_numeric_dtype rather than == "object" so pandas 3.x string
        # columns aren't misread as numeric.
        if not pd.api.types.is_numeric_dtype(series):
            n_uniq = series.nunique()
            if n_uniq > min(50, max(2, len(df) // 3)):
                continue  # drop high-cardinality free text
            cats = sorted(series.fillna(
                "__na__").astype(str).unique().tolist())
            mapping = {c: i for i, c in enumerate(cats)}
            encoders[col] = {"type": "categorical", "mapping": mapping}
            cols.append(series.fillna("__na__").astype(
                str).map(mapping).astype(float).values)
            feature_names.append(col)
            spec.append({"name": col, "kind": "categorical",
                        "categories": cats, "default": cats[0]})
        else:
            vals = pd.to_numeric(series, errors="coerce")
            median = vals.median()
            median = float(median) if pd.notna(median) else 0.0
            encoders[col] = {"type": "numeric", "median": median}
            cols.append(vals.fillna(median).values.astype(float))
            feature_names.append(col)
            spec.append({
                "name": col, "kind": "numeric",
                "min": float(vals.min()) if pd.notna(vals.min()) else 0.0,
                "max": float(vals.max()) if pd.notna(vals.max()) else 0.0,
                "default": round(float(vals.mean()) if pd.notna(vals.mean()) else median, 4),
            })

    if not cols:
        raise ValueError(
            "No usable feature columns found after preprocessing.")

    X = np.column_stack(cols)

    n_unique = y_raw.nunique()
    if pd.api.types.is_numeric_dtype(y_raw) and n_unique > 10:
        task = "regression"
        y = pd.to_numeric(y_raw, errors="coerce").fillna(
            y_raw.median()).values.astype(float)
        target_meta = {"type": "regression"}
    else:
        task = "classification"
        classes = sorted(y_raw.astype(str).unique().tolist())
        cmap = {c: i for i, c in enumerate(classes)}
        y = y_raw.astype(str).map(cmap).values.astype(int)
        target_meta = {"type": "classification", "classes": classes}

    return X, y, task, feature_names, spec, encoders, target_meta


def _encode_row(inputs: dict, feature_names, encoders) -> np.ndarray:
    """Transform a dict of raw feature values into the model's input vector."""
    row = []
    for name in feature_names:
        enc = encoders[name]
        val = inputs.get(name)
        if enc["type"] == "categorical":
            key = "__na__" if val is None else str(val)
            row.append(float(enc["mapping"].get(key, 0)))
        else:
            try:
                row.append(float(val))
            except (TypeError, ValueError):
                row.append(float(enc["median"]))
    return np.array(row, dtype=float).reshape(1, -1)


# ── Target recommendation ────────────────────────────────────────────────────
def recommend_targets(df: pd.DataFrame) -> dict:
    n = len(df)
    suggestions = []
    for col in df.columns:
        s = df[col]
        nu = int(s.nunique(dropna=True))
        if nu <= 1:
            continue  # constant column is useless
        name = str(col).lower()
        is_id = (nu == n) or any(name == h or name.endswith("_" + h)
                                 or name.startswith(h + "_") for h in _ID_HINTS)
        if is_id:
            continue
        is_num = pd.api.types.is_numeric_dtype(s)

        score, task, reason, n_classes = 0, None, "", None
        if any(h in name for h in _LABEL_HINTS):
            score += 40

        if is_num and nu > 10:
            task, n_classes = "regression", None
            score += 20
            reason = "continuous numeric — regression"
        elif 2 <= nu <= 20:
            task, n_classes = "classification", nu
            score += 25 + (12 if nu == 2 else 0)
            reason = f"{nu} distinct classes — classification"
        else:
            continue

        if any(h in name for h in _LABEL_HINTS):
            reason = "name looks like a label · " + reason
        suggestions.append({
            "column": col, "task": task, "n_classes": n_classes,
            "reason": reason, "_score": score,
        })

    suggestions.sort(key=lambda s: s["_score"], reverse=True)
    top = suggestions[:3]
    for i, s in enumerate(top):
        s["recommended"] = (i == 0)
        del s["_score"]
    return {"suggestions": top, "engine": "PyTorch" if HAS_TORCH else "scikit-learn"}


def suggest_config(df, target_column: str | None) -> dict:
    engine = "PyTorch" if HAS_TORCH else "scikit-learn"
    if not target_column or target_column not in df.columns:
        rec = recommend_targets(df)
        return {"task": None, "note": "Select a target column — a neural network needs something to predict.",
                "config": DEFAULTS, "engine": engine, "recommendations": rec["suggestions"]}
    try:
        X, y, task, feats, *_ = _prepare_rich(df, target_column)
    except Exception as e:
        return {"task": None, "note": str(e), "config": DEFAULTS, "engine": engine}

    n_features, n_rows = X.shape[1], X.shape[0]
    width = int(min(LIMITS["max_units"], max(
        32, 2 ** int(np.ceil(np.log2(max(8, n_features * 2)))))))
    hidden = [width, max(16, width // 2)
              ] if n_rows >= 200 else [max(16, width // 2)]
    epochs = 80 if n_rows < 1000 else 50
    n_classes = int(len(np.unique(y))) if task == "classification" else None

    return {
        "task": task, "n_features": int(n_features), "n_rows": int(n_rows), "n_classes": n_classes,
        "config": {**DEFAULTS, "hidden_layers": hidden, "epochs": epochs},
        "engine": engine, "note": f"Detected a {task} task with {n_features} usable features.",
    }


def _final_metrics(task, y_true, y_pred):
    if task == "classification":
        n_classes = len(np.unique(y_true))
        avg = "binary" if n_classes == 2 else "macro"
        return {
            "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_score":  round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        }, "accuracy"
    return {
        "r2_score": round(float(r2_score(y_true, y_pred)), 4),
        "mae":      round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mse":      round(float(mean_squared_error(y_true, y_pred)), 4),
    }, "r2_score"


def _describe_arch(in_dim, hidden, out_dim, dropout, engine="PyTorch"):
    arch = [f"Input · {in_dim} features"]
    for u in hidden:
        arch.append(f"Dense {u} · BatchNorm · ReLU · Dropout {dropout}" if engine == "PyTorch"
                    else f"Dense {u} · ReLU · L2 reg")
    arch.append(f"Output · {out_dim} unit{'s' if out_dim != 1 else ''}")
    return arch


# ── Shared post-training analysis ─────────────────────────────────────────────
def _permutation_importance(predict_scaled, Xs_val, y_val, task, feature_names, rng):
    """Model-agnostic permutation importance on the (scaled) validation set."""
    def score(y_pred):
        return accuracy_score(y_val, y_pred) if task == "classification" else r2_score(y_val, y_pred)

    base = score(predict_scaled(Xs_val)[0])
    out = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(3):  # average a few shuffles for stability
            Xp = Xs_val.copy()
            rng.shuffle(Xp[:, j])
            drops.append(base - score(predict_scaled(Xp)[0]))
        out.append({"feature": name, "importance": round(
            max(0.0, float(np.mean(drops))), 4)})
    total = sum(o["importance"] for o in out) or 1.0
    for o in out:
        o["importance_pct"] = round(100 * o["importance"] / total, 1)
    out.sort(key=lambda o: o["importance"], reverse=True)
    return out


def _evaluation(task, y_test, y_pred, proba, target_meta):
    """Confusion matrix + ROC (classification) or predicted-vs-actual (regression)."""
    if task == "classification":
        labels = list(range(len(target_meta.get("classes", []))))
        cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
        ev = {"kind": "classification",
              "labels": target_meta["classes"], "confusion_matrix": cm}
        if len(labels) == 2 and proba is not None:
            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
            ev["roc"] = {
                "points": [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
                           for f, t in zip(fpr, tpr)],
                "auc": round(float(auc(fpr, tpr)), 4),
            }
        return ev
    # regression: sample points for a predicted-vs-actual scatter
    idx = np.arange(len(y_test))
    if len(idx) > 200:
        idx = np.random.RandomState(42).choice(idx, 200, replace=False)
    points = [{"actual": round(float(y_test[i]), 4), "predicted": round(
        float(y_pred[i]), 4)} for i in idx]
    return {"kind": "regression", "points": points}


# ── PyTorch path ─────────────────────────────────────────────────────────────
def _build_torch_mlp(in_dim, out_dim, hidden, dropout):
    layers, prev = [], in_dim
    for units in hidden:
        layers += [nn.Linear(prev, units), nn.BatchNorm1d(units),
                   nn.ReLU(), nn.Dropout(dropout)]
        prev = units
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _train_torch(X, y, task, cfg, feature_names, target_meta):
    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.RandomState(42)
    n_classes = int(len(np.unique(y))) if task == "classification" else 1
    out_dim = n_classes if task == "classification" else 1

    test_size = max(0.15, min(0.25, 20 / len(X)))
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_tr)
    X_tr, X_val, X_test = scaler.transform(
        X_tr), scaler.transform(X_val), scaler.transform(X_test)

    xt = torch.tensor(X_tr, dtype=torch.float32)
    xv = torch.tensor(X_val, dtype=torch.float32)
    if task == "classification":
        yt = torch.tensor(y_tr, dtype=torch.long)
        yv = torch.tensor(y_val, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
    else:
        yt = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
        yv = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        criterion = nn.MSELoss()

    model = _build_torch_mlp(
        X.shape[1], out_dim, cfg["hidden_layers"], cfg["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    n_params = sum(p.numel() for p in model.parameters())
    batch = min(cfg["batch_size"], len(xt))

    def predict_scaled(Xs):
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(Xs, dtype=torch.float32))
            if task == "classification":
                proba = torch.softmax(out, dim=1).numpy()
                return proba.argmax(1), proba
            return out.view(-1).numpy(), None

    history = []
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(xt))
        epoch_loss, nb = 0.0, 0
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            optimizer.zero_grad()
            loss = criterion(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            nb += 1
        model.eval()
        with torch.no_grad():
            val_out = model(xv)
            val_loss = float(criterion(val_out, yv).item())
            val_metric = (accuracy_score(y_val, val_out.argmax(1).numpy()) if task == "classification"
                          else r2_score(y_val, val_out.view(-1).numpy()))
        history.append({"epoch": epoch + 1, "train_loss": round(epoch_loss / max(1, nb), 4),
                        "val_loss": round(val_loss, 4), "val_metric": round(float(val_metric), 4)})
    elapsed = round(time.time() - t0, 2)

    y_pred, proba = predict_scaled(X_test)
    importance = _permutation_importance(
        predict_scaled, X_val, y_val, task, feature_names, rng)
    evaluation = _evaluation(task, y_test, y_pred, proba, target_meta)
    metrics, primary = _final_metrics(task, y_test, y_pred)

    return _assemble(model, scaler, "PyTorch", task, metrics, primary, history,
                     X.shape[1], cfg, out_dim, n_params, elapsed, importance, evaluation,
                     feature_names, target_meta, predict_scaled)


# ── scikit-learn fallback ────────────────────────────────────────────────────
def _train_sklearn(X, y, task, cfg, feature_names, target_meta):
    rng = np.random.RandomState(42)
    test_size = max(0.15, min(0.25, 20 / len(X)))
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_tr)
    X_tr, X_val, X_test = scaler.transform(
        X_tr), scaler.transform(X_val), scaler.transform(X_test)

    common = dict(hidden_layer_sizes=tuple(cfg["hidden_layers"]),
                  learning_rate_init=cfg["learning_rate"],
                  batch_size=min(cfg["batch_size"], len(X_tr)), random_state=42)
    if task == "classification":
        model = MLPClassifier(**common)
        classes = np.unique(y)
    else:
        model = MLPRegressor(**common)
        classes = None

    def predict_scaled(Xs):
        if task == "classification":
            proba = model.predict_proba(Xs)
            return proba.argmax(1), proba
        return model.predict(Xs), None

    history = []
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        if classes is not None:
            model.partial_fit(X_tr, y_tr, classes=classes)
            vm = accuracy_score(y_val, model.predict(X_val))
        else:
            model.partial_fit(X_tr, y_tr)
            vm = r2_score(y_val, model.predict(X_val))
        history.append({"epoch": epoch + 1, "train_loss": round(float(getattr(model, "loss_", 0.0)), 4),
                        "val_loss": None, "val_metric": round(float(vm), 4)})
    elapsed = round(time.time() - t0, 2)

    y_pred, proba = predict_scaled(X_test)
    importance = _permutation_importance(
        predict_scaled, X_val, y_val, task, feature_names, rng)
    evaluation = _evaluation(task, y_test, y_pred, proba, target_meta)
    metrics, primary = _final_metrics(task, y_test, y_pred)

    out_dim = len(np.unique(y)) if task == "classification" else 1
    n_params = int(sum(c.size for c in model.coefs_) +
                   sum(b.size for b in model.intercepts_))
    return _assemble(model, scaler, "scikit-learn", task, metrics, primary, history,
                     X.shape[1], cfg, out_dim, n_params, elapsed, importance, evaluation,
                     feature_names, target_meta, predict_scaled)


def _assemble(model, scaler, engine, task, metrics, primary, history, in_dim, cfg,
              out_dim, n_params, elapsed, importance, evaluation, feature_names,
              target_meta, predict_scaled):
    LAST_MODEL.clear()
    LAST_MODEL.update({
        "engine": engine, "task": task, "model": model, "scaler": scaler,
        "predict_scaled": predict_scaled, "feature_names": feature_names,
        "target_meta": target_meta,
    })
    return {
        "status": "success", "engine": engine, "task": task, "metrics": metrics,
        "primary_metric": primary, "metric_label": "Accuracy" if task == "classification" else "Val R²",
        "history": history,
        "architecture": _describe_arch(in_dim, cfg["hidden_layers"], out_dim, cfg["dropout"], engine),
        "n_params": n_params, "training_time": f"{elapsed}s", "config": cfg,
        "feature_importance": importance, "evaluation": evaluation,
    }


def train_neural_network(df, target_column, config=None):
    if not target_column or target_column not in df.columns:
        return {"status": "error", "note": "A target column is required to train a neural network."}
    cfg = _clean_config(config)
    try:
        X, y, task, feats, spec, encoders, target_meta = _prepare_rich(
            df, target_column)
    except Exception as e:
        return {"status": "error", "note": str(e)}
    if len(X) < 20:
        return {"status": "error", "note": "Dataset too small for deep learning (need ≥ 20 rows)."}

    try:
        fn = _train_torch if HAS_TORCH else _train_sklearn
        result = fn(X, y, task, cfg, feats, target_meta)
    except Exception as e:
        return {"status": "error", "note": str(e)[:300]}

    # attach the input schema so the playground can build its form
    LAST_MODEL.update({"encoders": encoders, "spec": spec,
                      "target_column": target_column})
    result["feature_spec"] = spec
    result["target_column"] = target_column
    return result


# ── Auto-optimize config: algorithm to find best hidden layers, epochs, LR ───
def auto_optimize_config(df: pd.DataFrame, target_column: str = None) -> dict:
    n_samples, n_features = df.shape
    n_features -= 1  # exclude target if provided

    task = "auto"
    n_classes = None

    if target_column and target_column in df.columns:
        y_raw = df[target_column]
        n_unique = y_raw.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(y_raw) and n_unique > 10:
            task = "regression"
        elif 2 <= n_unique <= 20:
            task = "classification"
            n_classes = n_unique

    # Hidden layers algorithm based on data size and complexity
    base_width = int(
        min(256, max(16, 2 ** int(np.ceil(np.log2(max(8, n_features * 1.5)))))))
    if n_samples < 500:
        hidden_layers = [base_width]
    elif n_samples < 2000:
        hidden_layers = [base_width, max(16, base_width // 2)]
    elif n_samples < 10000:
        hidden_layers = [base_width, max(
            32, base_width // 2), max(16, base_width // 4)]
    else:
        hidden_layers = [min(512, base_width * 2),
                         base_width, max(32, base_width // 2)]
    if task == "classification" and n_classes and n_classes > 5:
        hidden_layers = hidden_layers + [max(16, hidden_layers[-1] // 2)]

    # Epochs algorithm
    if n_samples < 500:
        epochs = 150
    elif n_samples < 2000:
        epochs = 100
    elif n_samples < 10000:
        epochs = 60
    else:
        epochs = 40

    # Learning rate algorithm
    lr = 0.001
    if n_features > 50:
        lr = 0.0005
    if n_samples > 10000:
        lr = min(lr, 0.0003)
    if task == "classification" and n_classes and n_classes > 10:
        lr = min(lr, 0.0005)

    return {
        "config": {
            "hidden_layers": hidden_layers,
            "epochs": epochs,
            "learning_rate": lr,
            "dropout": 0.2 if n_samples > 1000 else 0.1,
            "batch_size": min(256, max(16, 2 ** int(np.log2(min(n_samples // 10, 64))))),
        },
        "data_profile": {
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "task": task,
            "n_classes": n_classes,
        },
    }


# ── Pattern discovery: PCA, clusters, correlations ───────────────────────────
PATTERN_SAMPLE_CAP = 5000


def _pattern_sample(df: pd.DataFrame) -> pd.DataFrame:
    return df if len(df) <= PATTERN_SAMPLE_CAP else df.sample(n=PATTERN_SAMPLE_CAP, random_state=42)


def discover_patterns(df: pd.DataFrame) -> dict:
    df = df.select_dtypes(include=[np.number]).copy()
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns for pattern discovery."}
    df = df.fillna(df.median(numeric_only=True))

    sample_df = _pattern_sample(df)

    # Correlation matrix (top features by variance)
    variances = df.var().sort_values(ascending=False)
    top_cols = variances.head(min(20, len(variances))).index.tolist()
    corr_df = sample_df[top_cols].corr()

    # PCA on sampled data
    n_components = min(3, sample_df.shape[1], sample_df.shape[0])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(sample_df.values)

    # Feature ranking by variance (uses full data, cheap O(n) per column)
    var_df = df.var().sort_values(ascending=False)
    feature_rank = [{"feature": col, "variance": round(float(v), 4)}
                    for col, v in var_df.head(30).items()]

    # PCA loadings
    loadings = []
    for i, col in enumerate(sample_df.columns[:20]):
        loadings.append({
            "feature": col,
            "pc1": round(float(pca.components_[0][i]), 4) if n_components >= 1 else 0,
            "pc2": round(float(pca.components_[1][i]), 4) if n_components >= 2 else 0,
        })

    # Fast elbow + silhouette on sampled data
    best_k, best_score = 2, -1
    elbow = []
    vals = sample_df.values
    ks = range(2, min(10, len(sample_df)))
    if len(sample_df) >= 10:
        from sklearn.metrics import silhouette_score
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=3)
            labels = km.fit_predict(vals)
            elbow.append({"k": k, "inertia": round(float(km.inertia_), 2)})
            if len(set(labels)) > 1:
                s = silhouette_score(vals, labels)
                if s > best_score:
                    best_score, best_k = s, k

    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=3)
    cluster_labels = final_km.fit_predict(vals).tolist()

    return {
        "pca": {
            "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "points": [{"x": round(float(r[0]), 4), "y": round(float(r[1]), 4) if n_components >= 2 else 0,
                        "z": round(float(r[2]), 4) if n_components >= 3 else 0}
                       for r in pca_result[:500]],
            "loadings": loadings,
        },
        "clusters": {
            "n_clusters": best_k,
            "silhouette_score": round(float(best_score), 4) if best_score > 0 else None,
            "labels": cluster_labels[:500],
            "elbow": elbow,
        },
        "correlation": {
            "columns": top_cols,
            "values": [[round(float(corr_df.iloc[i, j]), 4) for j in range(len(top_cols))]
                       for i in range(len(top_cols))],
        },
        "feature_rank": feature_rank,
    }


# ── Prediction playground ─────────────────────────────────────────────────────
def predict(inputs: dict) -> dict:
    if "predict_scaled" not in LAST_MODEL:
        return {"status": "error", "note": "No trained network in memory. Train a model first."}
    try:
        row = _encode_row(
            inputs, LAST_MODEL["feature_names"], LAST_MODEL["encoders"])
        Xs = LAST_MODEL["scaler"].transform(row)
        preds, proba = LAST_MODEL["predict_scaled"](Xs)
        task = LAST_MODEL["task"]
        if task == "classification":
            classes = LAST_MODEL["target_meta"]["classes"]
            idx = int(preds[0])
            dist = ([{"class": classes[i], "prob": round(float(p), 4)} for i, p in enumerate(proba[0])]
                    if proba is not None else [])
            dist.sort(key=lambda d: d["prob"], reverse=True)
            return {"status": "success", "task": "classification", "prediction": classes[idx],
                    "confidence": round(float(proba[0][idx]), 4) if proba is not None else None,
                    "distribution": dist, "target": LAST_MODEL.get("target_column")}
        return {"status": "success", "task": "regression",
                "prediction": round(float(preds[0]), 4), "target": LAST_MODEL.get("target_column")}
    except Exception as e:
        return {"status": "error", "note": str(e)[:300]}
