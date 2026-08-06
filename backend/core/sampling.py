"""Shared guards that keep memory-hungry sklearn calls inside a small box.

The deployed backend runs on a 512 MB container. Several metrics in scikit-learn are
O(n^2) in *memory*, not just time, and silently allocate gigabytes on datasets the
rest of the app handles comfortably. When that happens the container is OOM-killed
mid-request — from the browser it looks like the server simply vanished, which is
exactly the "lost contact with the server" failure this module exists to prevent.

`silhouette_score` is the worst offender: it materialises the full pairwise distance
matrix, so n rows cost n^2 * 8 bytes. Measured on this codebase:

      n =  5,000  ->   200 MB
      n = 10,000  ->   800 MB      <- what DL 2.0 was doing, on a 3.6 MB CSV
      n = 50,000  ->  20.0 GB      <- what the K-Means model card was doing

Scoring a random subsample is the standard remedy and costs nothing in accuracy —
silhouette is a mean over rows, so a few thousand of them estimate it to more decimal
places than the UI ever shows.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Rows any *supervised* trainer will fit on. This one is a time budget rather than
# a memory one, and it exists because a synchronous HTTP request has to come back:
# SVM is between quadratic and cubic in row count, so a 50k-row dataset does not
# take a long time, it effectively never finishes. Even the well-behaved models
# overshoot — a PyTorch net on 50k rows measured 191s on a fast desktop, and a
# free-tier container is an order of magnitude slower than that. Whatever proxy
# sits in front of the app will have hung up long before, which the browser
# reports as the connection dropping.
TRAIN_MAX_ROWS = int(os.getenv("TRAIN_MAX_ROWS", "20000"))

# Rows used for any O(n^2) metric. 2,000 rows is a ~32 MB distance matrix and
# estimates a mean-over-rows statistic to well within the 2 decimals we display.
SILHOUETTE_SAMPLE = 2_000

# Rows used to *fit* clustering models. KMeans is O(n) in memory so this is a
# time budget rather than a memory one — the free tier gets ~0.1 of a CPU.
CLUSTER_FIT_ROWS = 5_000


def safe_silhouette(X, labels, default: float = -1.0) -> float:
    """silhouette_score with a bounded memory footprint.

    Returns `default` instead of raising: a missing quality score should never cost
    the caller its whole run.
    """
    from sklearn.metrics import silhouette_score

    X = np.asarray(X)
    labels = np.asarray(labels)

    # Fewer than two populated clusters makes the score undefined.
    if len(X) < 3 or len(set(labels.tolist())) < 2:
        return default

    try:
        if len(X) > SILHOUETTE_SAMPLE:
            # sample_size makes sklearn score a random subset, so the distance
            # matrix is bounded by SILHOUETTE_SAMPLE^2 regardless of len(X).
            score = silhouette_score(
                X, labels, sample_size=SILHOUETTE_SAMPLE, random_state=42)
        else:
            score = silhouette_score(X, labels)
        score = float(score)
        return score if np.isfinite(score) else default
    except Exception:
        return default


# Wall-clock a synchronous training request may spend in its epoch loop. Row and
# epoch caps bound the *work*, but not the time, because how long that work takes
# depends entirely on the box: the same 20k-row fit measured 77s on a desktop and
# is several times that on a shared-CPU container. A time budget is the only guard
# that holds on hardware we cannot measure in advance. Training stops at the epoch
# boundary and keeps everything learned so far, which is a usable model — an
# aborted request is nothing at all.
TRAIN_TIME_BUDGET = float(os.getenv("TRAIN_TIME_BUDGET_SECONDS", "60"))


def cap_training_rows(df: pd.DataFrame, cap: int = TRAIN_MAX_ROWS):
    """Trim a dataframe to `cap` rows for training. Returns (df, note_or_None).

    The note is meant to be handed straight back to the user. Quietly training on
    a fraction of someone's data and reporting the accuracy as if it came from all
    of it would be the wrong kind of fix.
    """
    n = int(len(df))
    if n <= cap:
        return df, None
    return (
        df.sample(n=cap, random_state=42),
        f"Trained on a random {cap:,}-row sample of {n:,} rows to keep training "
        f"within the request timeout.",
    )


def subsample_rows(*arrays, cap: int, seed: int = 42):
    """Take the same random `cap` rows from each array. Returns (arrays..., index).

    `index` is None when no sampling happened, so callers can tell whether the
    result covers the full dataset.
    """
    n = len(arrays[0])
    if n <= cap:
        return (*arrays, None)
    idx = np.random.RandomState(seed).choice(n, cap, replace=False)
    return (*(np.asarray(a)[idx] for a in arrays), idx)
