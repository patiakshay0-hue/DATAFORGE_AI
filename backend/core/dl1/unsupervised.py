"""Target-free deep learning — autoencoder training and auto-configuration.

Supervised training needs a label. An **autoencoder** does not: it is trained to
reconstruct its own input through a narrow bottleneck, so the only supervision is the
data itself. Whatever survives that bottleneck is the structure the data actually
contains, which is what makes it the right tool for "just give me the patterns".

What the bottleneck buys us:

  * reconstruction error per row      → anomalies, without labelling any
  * reconstruction error per feature  → which columns carry information
  * the latent code                   → clusters/segments in a learned, non-linear space
  * autoencoder vs PCA error          → proof that structure is non-linear

Everything degrades to a scikit-learn implementation when torch is unavailable, in
line with the rest of this codebase (see requirements.txt).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MAX_UNITS = 512
MAX_LAYERS = 4
# Fraction of variance the latent space should retain when sizing the bottleneck.
TARGET_VARIANCE = 0.95
# Autoencoders are budgeted by gradient updates rather than epochs — see auto_config.
TARGET_UPDATES = 5_000
MIN_EPOCHS = 40
MAX_EPOCHS = 400
# An epoch must cut reconstruction loss by at least this fraction to count as progress.
REL_IMPROVEMENT = 0.002
# Columns with more distinct values than this (relative to row count) are treated as
# free-text/identifiers and excluded from modelling.
HIGH_CARDINALITY_RATIO = 0.5
# Standardised feature values are clipped to +/- this many standard deviations.
CLIP_SIGMA = 6.0


# ── Preprocessing (no target column anywhere) ────────────────────────────────
@dataclass
class Encoded:
    X: np.ndarray                       # scaled numeric matrix, model-ready
    feature_names: list[str]
    spec: list[dict]                    # per-column metadata for the report
    dropped: list[dict]                 # columns excluded, with the reason
    scaler: StandardScaler
    n_rows: int


def prepare(df: pd.DataFrame) -> Encoded:
    """Encode every usable column with no notion of a target.

    Numeric columns are median-imputed; categoricals are ordinal-encoded; constant,
    all-null and high-cardinality text columns are dropped and reported so the user
    can see exactly what was and was not used (spec: "what all column to be used").
    """
    n_rows = int(len(df))
    columns, names, spec, dropped = [], [], [], []

    for col in df.columns:
        series = df[col]

        if series.isnull().all():
            dropped.append({"column": col, "reason": "every value is missing"})
            continue
        if series.nunique(dropna=True) <= 1:
            dropped.append({"column": col, "reason": "constant — no information"})
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            dropped.append({"column": col, "reason": "datetime — not modelled in 1.0"})
            continue

        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce")
            median = values.median()
            median = float(median) if pd.notna(median) else 0.0
            columns.append(values.fillna(median).values.astype(float))
            names.append(col)
            spec.append({
                "name": col, "kind": "numeric",
                "missing": int(series.isnull().sum()),
                "unique": int(series.nunique(dropna=True)),
                "min": round(float(values.min()), 4) if pd.notna(values.min()) else 0.0,
                "max": round(float(values.max()), 4) if pd.notna(values.max()) else 0.0,
                "mean": round(float(values.mean()), 4) if pd.notna(values.mean()) else 0.0,
            })
        else:
            n_unique = int(series.nunique(dropna=True))
            if n_unique > max(2, n_rows * HIGH_CARDINALITY_RATIO):
                dropped.append({
                    "column": col,
                    "reason": f"high-cardinality text ({n_unique} distinct values) — likely an identifier",
                })
                continue
            cats = sorted(series.fillna("__na__").astype(str).unique().tolist())
            mapping = {c: i for i, c in enumerate(cats)}
            columns.append(series.fillna("__na__").astype(str).map(mapping).astype(float).values)
            names.append(col)
            spec.append({
                "name": col, "kind": "categorical",
                "missing": int(series.isnull().sum()),
                "unique": n_unique,
                "categories": cats[:20],
            })

    if not columns:
        raise ValueError(
            "No usable columns found. The dataset needs at least one numeric or "
            "low-cardinality categorical column."
        )

    X_raw = np.column_stack(columns)
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    # Bound the standardised range. A handful of extreme rows can otherwise hold a
    # large share of the total squared magnitude, and because the autoencoder is
    # trained on MSE by gradient descent it will spend its capacity on those few rows
    # and underfit everything else — while PCA, a closed-form least-squares fit,
    # absorbs the same direction for free. Clipping keeps the comparison meaningful.
    # Outliers stay the worst-reconstructed rows, so anomaly detection is unaffected.
    X = np.clip(X, -CLIP_SIGMA, CLIP_SIGMA)

    return Encoded(X=X, feature_names=names, spec=spec,
                   dropped=dropped, scaler=scaler, n_rows=n_rows)


# ── Auto-configuration (the algorithm the request asks for) ──────────────────
@dataclass
class AutoEncoderConfig:
    """Hyperparameters chosen from the data alone — no target, no manual tuning."""

    hidden_layers: list[int]
    latent_dim: int
    epochs: int
    learning_rate: float
    batch_size: int
    activation: str
    optimizer: str
    dropout: float
    loss_function: str
    early_stopping: dict = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _intrinsic_dim(X: np.ndarray) -> tuple[int, float, list[float]]:
    """How many dimensions the data genuinely occupies.

    This is the key decision for an autoencoder: the bottleneck should be wide enough
    to hold the real structure and narrow enough to force generalisation. PCA gives a
    principled starting point — the component count needed to retain TARGET_VARIANCE.
    Compressing below that destroys information; going far above it just learns the
    identity function.
    """
    n_features = X.shape[1]
    if n_features < 2:
        return 1, 1.0, [1.0]

    n_comp = int(min(n_features, X.shape[0]))
    pca = PCA(n_components=n_comp, random_state=42).fit(X)
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)

    needed = int(np.searchsorted(cumulative, TARGET_VARIANCE) + 1)
    needed = int(np.clip(needed, 1, max(1, n_features - 1)))
    return needed, float(cumulative[needed - 1]), [round(float(r), 4) for r in ratios[:10]]


def auto_config(enc: Encoded) -> AutoEncoderConfig:
    """Pick hidden layers, epochs and learning rate (plus the rest) from the data.

    Deterministic heuristics — one PCA pass, no search — so a configuration comes
    back in milliseconds.
    """
    n_samples, n_features = enc.X.shape
    why: list[str] = []

    # ── Bottleneck: driven by the data's intrinsic dimensionality ────────────
    latent, retained, _ = _intrinsic_dim(enc.X)
    latent = int(np.clip(latent, 2, max(2, n_features - 1)))
    why.append(
        f"Latent dimension {latent} — PCA shows {latent} component(s) retain "
        f"{retained * 100:.0f}% of variance, so that is the real width of this data."
    )

    # ── Encoder depth: more rows support a deeper funnel ─────────────────────
    if n_samples < 500 or n_features <= 4:
        depth = 1
        why.append(f"Single encoder layer — {n_samples} rows / {n_features} features is a small problem.")
    elif n_samples < 5_000:
        depth = 2
        why.append(f"Two encoder layers for {n_samples} rows.")
    else:
        depth = 3
        why.append(f"Three encoder layers for {n_samples:,} rows.")
    depth = int(min(depth, MAX_LAYERS))

    # Geometric taper from the input width down to the latent width. A smooth ratio
    # avoids a sudden collapse that would throw away information in one step.
    start = int(min(MAX_UNITS, max(latent * 2, 2 ** int(np.ceil(np.log2(max(4, n_features * 2)))))))
    if depth == 1:
        hidden = [start]
    else:
        ratio = (latent / start) ** (1 / (depth + 1))
        hidden = []
        width = start
        for _ in range(depth):
            width = int(max(latent + 1, round(width * ratio)))
            hidden.append(int(min(MAX_UNITS, max(2, width))))
        hidden[0] = start
    why.append(f"Encoder funnel {' -> '.join(map(str, hidden))} -> {latent} (mirrored to decode).")

    batch = int(np.clip(2 ** int(np.floor(np.log2(max(8, min(256, n_samples // 10))))), 8, 256))

    # ── Epochs: budget by gradient *updates*, not passes ─────────────────────
    # A row count alone is misleading. 2,000 rows at batch 256 is only 8 updates per
    # epoch, so a "100 epoch" budget is 800 updates and the network is still
    # improving when the loop ends. Sizing against a target update count and letting
    # early stopping cut the run is both safer and usually faster.
    batches_per_epoch = max(1, int(n_samples * 0.8) // batch)
    epochs = int(np.clip(round(TARGET_UPDATES / batches_per_epoch), MIN_EPOCHS, MAX_EPOCHS))
    why.append(
        f"Up to {epochs} epochs — {batches_per_epoch} update(s) per epoch, targeting "
        f"~{TARGET_UPDATES:,} gradient steps. Early stopping ends the run once "
        f"reconstruction plateaus."
    )

    # ── Learning rate: scale down as the problem grows ──────────────────────
    lr = 1e-3
    if n_features > 50:
        lr = 5e-4
    if n_samples > 10_000:
        lr = min(lr, 3e-4)
    if n_samples < 300:
        lr = 2e-3
    # A very aggressive compression ratio benefits from smaller, steadier steps.
    if n_features and latent / n_features < 0.2:
        lr = min(lr, 5e-4)
    why.append(f"Learning rate {lr} — set from {n_samples} rows x {n_features} features "
               f"and a {latent}/{n_features} compression ratio.")

    why.append(f"Batch size {batch} — about a tenth of the rows, snapped to a power of two.")

    dropout = 0.0 if n_samples < 1_000 else 0.1
    why.append("No dropout — too little data to regularise." if dropout == 0
               else f"Light dropout ({dropout}) to keep the code general.")

    # Tanh keeps very small autoencoders numerically stable; ReLU is faster otherwise.
    activation = "tanh" if n_samples < 300 else "relu"
    why.append(f"{activation.upper()} activation.")

    optimizer = "adamw" if n_samples >= 2_000 else "adam"
    why.append(f"{optimizer.capitalize()} optimizer.")

    patience = int(np.clip(round(epochs * 0.15), 5, 25))
    why.append(f"Early stopping on reconstruction loss, patience {patience}.")

    return AutoEncoderConfig(
        hidden_layers=hidden,
        latent_dim=latent,
        epochs=epochs,
        learning_rate=float(lr),
        batch_size=batch,
        activation=activation,
        optimizer=optimizer,
        dropout=float(dropout),
        loss_function="mse",
        early_stopping={"enabled": True, "monitor": "val_loss", "patience": patience,
                        "min_delta": 1e-5, "restore_best_weights": True},
        rationale=why,
    )


# ── Training ─────────────────────────────────────────────────────────────────
def _activation_module(name: str):
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU,
            "elu": nn.ELU, "silu": nn.SiLU}.get(name, nn.ReLU)()


def _build_autoencoder(n_features: int, cfg: AutoEncoderConfig):
    """Symmetric encoder/decoder around the latent bottleneck."""
    enc_layers, prev = [], n_features
    for units in cfg.hidden_layers:
        enc_layers += [nn.Linear(prev, units), _activation_module(cfg.activation)]
        if cfg.dropout > 0:
            enc_layers.append(nn.Dropout(cfg.dropout))
        prev = units
    enc_layers.append(nn.Linear(prev, cfg.latent_dim))

    dec_layers, prev = [], cfg.latent_dim
    for units in reversed(cfg.hidden_layers):
        dec_layers += [nn.Linear(prev, units), _activation_module(cfg.activation)]
        prev = units
    dec_layers.append(nn.Linear(prev, n_features))

    return nn.Sequential(*enc_layers), nn.Sequential(*dec_layers)


def _train_torch(X: np.ndarray, cfg: AutoEncoderConfig) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)

    n = len(X)
    split = max(1, int(n * 0.2))
    perm = np.random.RandomState(42).permutation(n)
    val_idx, tr_idx = perm[:split], perm[split:]
    if len(tr_idx) == 0:
        tr_idx = perm

    xt = torch.tensor(X[tr_idx], dtype=torch.float32)
    xv = torch.tensor(X[val_idx], dtype=torch.float32)

    encoder, decoder = _build_autoencoder(X.shape[1], cfg)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = (torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=1e-2)
                 if cfg.optimizer == "adamw" else torch.optim.Adam(params, lr=cfg.learning_rate))
    criterion = nn.MSELoss()

    es = cfg.early_stopping or {}
    patience = int(es.get("patience", 10))
    min_delta = float(es.get("min_delta", 1e-5))
    best, best_epoch, best_state, stale, stopped_early = None, 0, None, 0, False

    batch = min(cfg.batch_size, max(2, len(xt)))
    history, t0 = [], time.time()

    for epoch in range(cfg.epochs):
        encoder.train(); decoder.train()
        order = torch.randperm(len(xt))
        total, nb = 0.0, 0
        for i in range(0, len(xt), batch):
            idx = order[i:i + batch]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            batch_x = xt[idx]
            loss = criterion(decoder(encoder(batch_x)), batch_x)
            loss.backward()
            optimizer.step()
            total += float(loss.item()); nb += 1

        encoder.eval(); decoder.eval()
        with torch.no_grad():
            val_loss = float(criterion(decoder(encoder(xv)), xv).item()) if len(xv) else 0.0
        history.append({"epoch": epoch + 1,
                        "train_loss": round(total / max(1, nb), 6),
                        "val_loss": round(val_loss, 6)})

        # Relative threshold: reconstruction loss shrinks by orders of magnitude
        # during training, so a fixed absolute delta keeps the run alive on
        # negligible gains. Require a proportional improvement instead.
        threshold = max(min_delta, abs(best) * REL_IMPROVEMENT) if best is not None else 0.0
        if best is None or val_loss < best - threshold:
            best, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = ({k: v.detach().clone() for k, v in encoder.state_dict().items()},
                          {k: v.detach().clone() for k, v in decoder.state_dict().items()})
        else:
            stale += 1
            if stale >= patience:
                stopped_early = True
                break

    if best_state is not None:
        encoder.load_state_dict(best_state[0])
        decoder.load_state_dict(best_state[1])
    elapsed = round(time.time() - t0, 2)

    with torch.no_grad():
        full = torch.tensor(X, dtype=torch.float32)
        latent = encoder(full).numpy()
        recon = decoder(encoder(full)).numpy()

    n_params = sum(p.numel() for p in params)
    return {
        "engine": "PyTorch", "latent": latent, "reconstruction": recon,
        "history": history, "n_params": int(n_params), "training_time": f"{elapsed}s",
        "epochs_run": len(history), "epochs_requested": cfg.epochs,
        "stopped_early": stopped_early, "best_epoch": best_epoch or len(history),
        "final_loss": round(history[-1]["val_loss"], 6) if history else None,
    }


def _train_sklearn(X: np.ndarray, cfg: AutoEncoderConfig) -> dict:
    """Fallback for deployments without torch.

    An MLPRegressor trained to predict its own input is an autoencoder; forcing the
    hidden layers through `latent_dim` gives the same bottleneck. Per-epoch history is
    obtained via partial_fit so the loss curve still renders.
    """
    from sklearn.neural_network import MLPRegressor

    layers = tuple(cfg.hidden_layers) + (cfg.latent_dim,) + tuple(reversed(cfg.hidden_layers))
    model = MLPRegressor(hidden_layer_sizes=layers, learning_rate_init=cfg.learning_rate,
                         activation="tanh" if cfg.activation == "tanh" else "relu",
                         solver="adam", batch_size=min(cfg.batch_size, len(X)),
                         random_state=42, max_iter=1)

    es = cfg.early_stopping or {}
    patience = int(es.get("patience", 10))
    min_delta = float(es.get("min_delta", 1e-5))
    best, best_epoch, stale, stopped_early = None, 0, 0, False

    history, t0 = [], time.time()
    for epoch in range(cfg.epochs):
        model.partial_fit(X, X)
        loss = float(getattr(model, "loss_", 0.0))
        history.append({"epoch": epoch + 1, "train_loss": round(loss, 6),
                        "val_loss": round(loss, 6)})
        threshold = max(min_delta, abs(best) * REL_IMPROVEMENT) if best is not None else 0.0
        if best is None or loss < best - threshold:
            best, best_epoch, stale = loss, epoch + 1, 0
        else:
            stale += 1
            if stale >= patience:
                stopped_early = True
                break

    elapsed = round(time.time() - t0, 2)
    recon = model.predict(X)
    if recon.ndim == 1:
        recon = recon.reshape(-1, 1)

    # sklearn exposes no encoder half, so approximate the latent code with PCA at the
    # same width — enough for clustering and plotting.
    latent = PCA(n_components=min(cfg.latent_dim, X.shape[1]), random_state=42).fit_transform(X)

    n_params = int(sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_))
    return {
        "engine": "scikit-learn", "latent": latent, "reconstruction": recon,
        "history": history, "n_params": n_params, "training_time": f"{elapsed}s",
        "epochs_run": len(history), "epochs_requested": cfg.epochs,
        "stopped_early": stopped_early, "best_epoch": best_epoch or len(history),
        "final_loss": round(history[-1]["val_loss"], 6) if history else None,
    }


def train(enc: Encoded, cfg: AutoEncoderConfig) -> dict:
    """Train the autoencoder and derive the quantities the pattern engine needs."""
    result = _train_torch(enc.X, cfg) if HAS_TORCH else _train_sklearn(enc.X, cfg)

    X, recon = enc.X, result["reconstruction"]
    if recon.shape != X.shape:                     # sklearn edge case on 1-column data
        recon = np.resize(recon, X.shape)

    squared = (X - recon) ** 2
    row_error = squared.mean(axis=1)
    col_error = squared.mean(axis=0)

    # A feature the network reconstructs well is one it has learned to explain from
    # the others — i.e. it is redundant. Poorly reconstructed features carry unique
    # information. Inverted here so "importance" means "carries unique signal".
    total = float(col_error.sum()) or 1.0
    feature_error = sorted(
        ({"feature": name,
          "reconstruction_error": round(float(col_error[i]), 6),
          "error_pct": round(float(col_error[i] / total * 100), 2)}
         for i, name in enumerate(enc.feature_names)),
        key=lambda d: -d["reconstruction_error"],
    )

    # How much better the non-linear autoencoder is than a linear projection of the
    # same width — direct evidence of non-linear structure.
    pca_error = None
    try:
        k = min(cfg.latent_dim, X.shape[1])
        pca = PCA(n_components=k, random_state=42).fit(X)
        pca_error = float(((X - pca.inverse_transform(pca.transform(X))) ** 2).mean())
    except Exception:
        pass

    ae_error = float(squared.mean())
    result.update({
        "row_error": row_error,
        "feature_error": feature_error,
        "ae_error": round(ae_error, 6),
        "pca_error": round(pca_error, 6) if pca_error is not None else None,
        "nonlinear_gain": (round((pca_error - ae_error) / pca_error, 4)
                           if pca_error and pca_error > 0 else None),
        "architecture": _describe(enc.X.shape[1], cfg),
        "config": cfg.to_dict(),
    })
    return result


def _describe(n_features: int, cfg: AutoEncoderConfig) -> list[str]:
    act = cfg.activation.upper()
    arch = [f"Input · {n_features} features"]
    for u in cfg.hidden_layers:
        arch.append(f"Encode {u} · {act}")
    arch.append(f"Latent · {cfg.latent_dim} dimensions (bottleneck)")
    for u in reversed(cfg.hidden_layers):
        arch.append(f"Decode {u} · {act}")
    arch.append(f"Output · {n_features} features (reconstruction)")
    return arch
