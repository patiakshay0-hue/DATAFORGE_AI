"""AutoModelConfig — pick a full neural-network configuration from the data alone.

The existing `deep_trainer.auto_optimize_config` chose width/depth/epochs/lr/dropout/
batch size. This completes the set the spec requires by also choosing the **optimizer,
activation, loss function and early-stopping criteria**, all of which were previously
hardcoded in the training loop (Adam / ReLU / CrossEntropy-MSE / no early stopping).

Everything is heuristic and deterministic — no search, no extra training runs — so a
config comes back in microseconds. The goal is a *strong baseline without manual
tuning*, not a tuned optimum.

Usage:
    cfg = AutoModelConfig.from_dataframe(df, "target").to_dict()
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

# Ceilings that keep an auto-chosen network sane on a small free-tier box.
MAX_UNITS = 512
MAX_LAYERS = 6
MAX_EPOCHS = 300


@dataclass
class DataProfile:
    """What the config decision is based on (spec §3: size, features, distribution,
    target type, complexity)."""

    n_samples: int
    n_features: int
    n_numeric: int
    n_categorical: int
    missing_ratio: float
    task: str                      # regression | binary | multiclass
    n_classes: int | None
    class_balance: float | None    # min/max class frequency; 1.0 == perfectly balanced
    skew: float                    # mean |skewness| across numeric features
    complexity: float              # 0-1, drives capacity and regularisation

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelConfig:
    """The full hyperparameter set (spec §3)."""

    hidden_layers: list[int]
    neurons: int
    activation: str
    learning_rate: float
    optimizer: str
    batch_size: int
    epochs: int
    dropout: float
    loss_function: str
    early_stopping: dict = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def infer_task(y: pd.Series) -> str:
    """Decide regression vs classification for a target column.

    Single source of truth — `deep_trainer._prepare_rich` imports this so the config
    we report always matches the model that actually gets trained.

    The naive rule ("numeric and more than 10 distinct values means regression")
    misreads integer class labels: a 12-class target of ints 0-11 is classification,
    not regression. So integer-valued columns with a small, repeating set of values
    are treated as classes, while genuinely continuous columns fall through.
    """
    y = y.dropna()
    if len(y) == 0:
        return "regression"

    n_unique = int(y.nunique())
    if n_unique <= 1:
        return "regression"

    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return "classification"

    vals = pd.to_numeric(y, errors="coerce").dropna().values
    if len(vals) == 0:
        return "classification"

    # Integer-valued and drawn from a small repeating set → class labels.
    integral = bool(np.allclose(vals, np.round(vals)))
    if integral and n_unique <= 20 and n_unique < len(vals) * 0.5:
        return "classification"

    # Very few distinct values is classification even when fractional.
    if n_unique <= 10:
        return "classification"

    return "regression"


def _profile(df: pd.DataFrame, target_column: str | None) -> DataProfile:
    """Measure the dataset. Cheap: one pass of column-wise stats."""
    n_samples = int(len(df))
    features = [c for c in df.columns if c != target_column]

    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]

    cells = max(1, n_samples * max(1, len(features)))
    missing_ratio = float(df[features].isnull().sum().sum() / cells) if features else 0.0

    # Skewness signals how non-normal the inputs are; heavy skew favours
    # normalisation-friendly activations and a gentler learning rate.
    skew = 0.0
    if numeric:
        vals = df[numeric].skew(numeric_only=True).abs()
        vals = vals[np.isfinite(vals)]
        skew = float(vals.mean()) if len(vals) else 0.0

    task, n_classes, balance = "regression", None, None
    if target_column and target_column in df.columns:
        y = df[target_column].dropna()
        if infer_task(y) == "classification":
            n_classes = int(y.nunique())
            task = "binary" if n_classes == 2 else "multiclass"
            counts = y.value_counts()
            balance = float(counts.min() / counts.max()) if len(counts) and counts.max() else None

    # Complexity blends the drivers that justify more capacity: wide feature space,
    # many classes, skewed inputs, missingness, and scarce rows per feature.
    rows_per_feature = n_samples / max(1, len(features))
    complexity = float(np.clip(
        0.30 * min(1.0, len(features) / 50)
        + 0.25 * min(1.0, (n_classes or 2) / 20)
        + 0.20 * min(1.0, skew / 3)
        + 0.15 * min(1.0, missing_ratio * 5)
        + 0.10 * (1 - min(1.0, rows_per_feature / 50)),
        0.0, 1.0,
    ))

    return DataProfile(
        n_samples=n_samples,
        n_features=len(features),
        n_numeric=len(numeric),
        n_categorical=len(categorical),
        missing_ratio=round(missing_ratio, 4),
        task=task,
        n_classes=n_classes,
        class_balance=round(balance, 4) if balance is not None else None,
        skew=round(skew, 4),
        complexity=round(complexity, 4),
    )


def _architecture(p: DataProfile) -> tuple[list[int], list[str]]:
    """Depth from sample count, width from feature count, both nudged by complexity."""
    why = []

    # Width: next power of two above ~1.5x the feature count, clamped.
    base = int(min(MAX_UNITS, max(16, 2 ** int(np.ceil(np.log2(max(8, p.n_features * 1.5)))))))
    if p.complexity > 0.6:
        base = int(min(MAX_UNITS, base * 2))
        why.append(f"Widened to {base} units — high data complexity ({p.complexity:.2f}).")

    # Depth: more rows support more layers without overfitting.
    if p.n_samples < 500:
        layers = [base]
        why.append(f"Single hidden layer — only {p.n_samples} rows, deeper nets would overfit.")
    elif p.n_samples < 2_000:
        layers = [base, max(16, base // 2)]
        why.append(f"Two hidden layers for {p.n_samples} rows.")
    elif p.n_samples < 10_000:
        layers = [base, max(32, base // 2), max(16, base // 4)]
        why.append(f"Three hidden layers for {p.n_samples} rows.")
    else:
        layers = [min(MAX_UNITS, base * 2), base, max(32, base // 2)]
        why.append(f"Wide three-layer funnel for {p.n_samples:,} rows.")

    # Many classes need extra capacity near the output to separate them.
    if p.n_classes and p.n_classes > 5:
        layers = layers + [max(16, layers[-1] // 2)]
        why.append(f"Extra layer added — {p.n_classes} classes to separate.")

    return layers[:MAX_LAYERS], why


def _activation(p: DataProfile) -> tuple[str, str]:
    """ReLU by default; GELU for genuinely complex problems; Tanh for tiny data.

    Tanh is bounded, which keeps very small networks numerically stable, whereas GELU's
    smooth gradient helps deep nets on skewed, high-dimensional inputs.
    """
    if p.n_samples < 300:
        return "tanh", "Tanh — bounded activation is stable on very small datasets."
    if p.complexity > 0.6 or p.skew > 2.0:
        return "gelu", f"GELU — smoother gradients suit complex/skewed data (skew {p.skew:.2f})."
    return "relu", "ReLU — fast, reliable default for tabular data."


def _optimizer(p: DataProfile) -> tuple[str, str]:
    """AdamW once regularisation matters, plain Adam otherwise.

    AdamW decouples weight decay from the gradient update, which is the better
    behaved choice when the model has enough capacity to overfit.
    """
    if p.n_samples >= 2_000 and p.complexity > 0.4:
        return "adamw", "AdamW — decoupled weight decay helps on larger, complex data."
    if p.n_samples < 300:
        return "adam", "Adam — robust on tiny datasets without extra tuning."
    return "adam", "Adam — strong general-purpose default."


def _learning_rate(p: DataProfile, optimizer: str) -> tuple[float, str]:
    lr = 1e-3
    why = "Base learning rate 0.001."
    if p.n_features > 50:
        lr = 5e-4
        why = f"Lowered to {lr} — {p.n_features} features increase gradient variance."
    if p.n_samples > 10_000:
        lr = min(lr, 3e-4)
        why = f"Lowered to {lr} — many rows mean many updates per epoch."
    if p.n_classes and p.n_classes > 10:
        lr = min(lr, 5e-4)
        why = f"Lowered to {lr} — {p.n_classes} classes need finer steps."
    if p.n_samples < 300:
        lr = max(lr, 2e-3)
        why = f"Raised to {lr} — few batches per epoch on a small dataset."
    return float(lr), why


def _batch_size(p: DataProfile) -> tuple[int, str]:
    """Roughly n/10, snapped to a power of two, bounded to [8, 256]."""
    if p.n_samples < 100:
        return 8, "Batch size 8 — very few rows available."
    target = max(8, min(256, p.n_samples // 10))
    size = int(2 ** int(np.floor(np.log2(target))))
    size = int(np.clip(size, 8, 256))
    return size, f"Batch size {size} — about a tenth of {p.n_samples} rows, snapped to a power of two."


def _epochs(p: DataProfile) -> tuple[int, str]:
    """Small data needs more passes; early stopping trims the excess anyway."""
    if p.n_samples < 500:
        n = 150
    elif p.n_samples < 2_000:
        n = 100
    elif p.n_samples < 10_000:
        n = 60
    else:
        n = 40
    n = int(min(MAX_EPOCHS, n))
    return n, f"Up to {n} epochs — early stopping ends the run sooner if validation stalls."


def _dropout(p: DataProfile) -> tuple[float, str]:
    """Regularise in proportion to overfitting risk (capacity vs. rows available)."""
    if p.n_samples < 500:
        rate, why = 0.10, "Light dropout (0.10) — too little data to regularise aggressively."
    elif p.n_samples < 5_000:
        rate, why = 0.20, "Standard dropout (0.20)."
    else:
        rate, why = 0.30, "Stronger dropout (0.30) — plenty of data, guard against overfitting."
    if p.complexity > 0.7:
        rate = min(0.5, rate + 0.1)
        why = f"Dropout raised to {rate:.2f} — high complexity increases overfitting risk."
    return float(rate), why


def _loss(p: DataProfile) -> tuple[str, str]:
    """Loss follows the task, with a class-imbalance correction for classification."""
    if p.task == "regression":
        if p.skew > 2.0:
            return "huber", f"Huber loss — skewed target (skew {p.skew:.2f}) makes MSE outlier-sensitive."
        return "mse", "Mean squared error — standard regression objective."
    if p.class_balance is not None and p.class_balance < 0.5:
        return ("weighted_cross_entropy",
                f"Class-weighted cross-entropy — classes are imbalanced ({p.class_balance:.2f} min/max ratio).")
    return "cross_entropy", "Cross-entropy — standard classification objective."


def _early_stopping(p: DataProfile, epochs: int) -> tuple[dict, str]:
    """Patience scales with epoch budget; noisy small data gets more slack."""
    patience = int(np.clip(round(epochs * 0.15), 5, 25))
    if p.n_samples < 500:
        patience = min(30, patience + 5)
    monitor = "val_loss"
    return (
        {
            "enabled": True,
            "monitor": monitor,
            "patience": patience,
            "min_delta": 1e-4,
            "restore_best_weights": True,
        },
        f"Early stopping on {monitor} with patience {patience} and best-weight restore.",
    )


class AutoModelConfig:
    """Dataset in, complete neural-network configuration out.

    Deliberately a class rather than a function so 2.0 can subclass and override a
    single decision (e.g. swap `_optimizer` for a learned policy) without touching
    the rest of the pipeline.
    """

    def __init__(self, profile: DataProfile, config: ModelConfig):
        self.profile = profile
        self.config = config

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, target_column: str | None = None) -> "AutoModelConfig":
        p = _profile(df, target_column)

        layers, why_arch = _architecture(p)
        activation, why_act = _activation(p)
        optimizer, why_opt = _optimizer(p)
        lr, why_lr = _learning_rate(p, optimizer)
        batch, why_batch = _batch_size(p)
        epochs, why_epochs = _epochs(p)
        dropout, why_drop = _dropout(p)
        loss, why_loss = _loss(p)
        stopping, why_stop = _early_stopping(p, epochs)

        cfg = ModelConfig(
            hidden_layers=layers,
            neurons=int(sum(layers)),
            activation=activation,
            learning_rate=lr,
            optimizer=optimizer,
            batch_size=batch,
            epochs=epochs,
            dropout=dropout,
            loss_function=loss,
            early_stopping=stopping,
            rationale=[*why_arch, why_act, why_opt, why_lr, why_batch,
                       why_epochs, why_drop, why_loss, why_stop],
        )
        return cls(p, cfg)

    def to_dict(self) -> dict:
        return {"config": self.config.to_dict(), "data_profile": self.profile.to_dict()}
