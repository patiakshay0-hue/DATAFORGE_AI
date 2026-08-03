"""Deep Learning 1.0 pipeline — the full target-free run, as one job.

Stages, in order:

    preprocess → configure → train → discover → ready

Everything happens inside a single job so the dataframe is encoded once and the
autoencoder is trained once, no matter how many times the UI asks about progress.
The job runs on a worker thread and reports progress into `app.dl1_store`, which the
frontend polls.

Deviation from the original plan: progress is delivered by polling a status endpoint
rather than Server-Sent Events. The important half of that optimisation — one job
instead of six round-trips — is unchanged, and polling avoids a streaming path that
would need its own buffering and reconnect handling for a run that lasts seconds.
"""

from __future__ import annotations

import threading
import traceback

import pandas as pd

from app import dl1_store
from . import discovery, unsupervised

# Weight of each stage in the overall progress bar. Training dominates wall-clock
# time, so it owns the widest span.
STAGE_PROGRESS = {
    "preprocess": (5, 15),
    "configure": (15, 25),
    "train": (25, 75),
    "discover": (75, 95),
    "ready": (95, 100),
}


def _set(job_id: str, stage: str, message: str, pct: int | None = None) -> None:
    lo, hi = STAGE_PROGRESS.get(stage, (0, 100))
    dl1_store.update(job_id, stage=stage, message=message,
                     progress=int(pct if pct is not None else lo), status="running")


def run(job_id: str, df: pd.DataFrame) -> None:
    """Execute the whole pipeline, recording progress and results on the job."""
    try:
        # ── 1. Preprocess ────────────────────────────────────────────────────
        _set(job_id, "preprocess", "Detecting column types and handling missing values")
        encoded = unsupervised.prepare(df)

        profile = {
            "rows": encoded.n_rows,
            "columns_total": int(len(df.columns)),
            "columns_used": len(encoded.feature_names),
            "columns_dropped": len(encoded.dropped),
            "numeric": sum(1 for s in encoded.spec if s["kind"] == "numeric"),
            "categorical": sum(1 for s in encoded.spec if s["kind"] == "categorical"),
            "missing_total": int(df.isnull().sum().sum()),
            "features_used": encoded.feature_names,
            "features_dropped": encoded.dropped,
            "column_spec": encoded.spec,
        }
        dl1_store.update(job_id, profile=profile)

        # ── 2. Configure ─────────────────────────────────────────────────────
        _set(job_id, "configure", "Choosing architecture, epochs and learning rate")
        cfg = unsupervised.auto_config(encoded)
        dl1_store.update(job_id, config=cfg.to_dict())

        # ── 3. Train ─────────────────────────────────────────────────────────
        _set(job_id, "train",
             f"Training autoencoder — {'→'.join(map(str, cfg.hidden_layers))}→{cfg.latent_dim}")
        trained = unsupervised.train(encoded, cfg)

        # The heavy arrays stay out of the stored payload; only what the UI renders.
        dl1_store.update(job_id, training={
            "engine": trained["engine"],
            "architecture": trained["architecture"],
            "history": trained["history"],
            "n_params": trained["n_params"],
            "training_time": trained["training_time"],
            "epochs_run": trained["epochs_run"],
            "epochs_requested": trained["epochs_requested"],
            "stopped_early": trained["stopped_early"],
            "best_epoch": trained["best_epoch"],
            "final_loss": trained["final_loss"],
            "ae_error": trained["ae_error"],
            "pca_error": trained["pca_error"],
            "nonlinear_gain": trained["nonlinear_gain"],
            "feature_error": trained["feature_error"],
        })

        # ── 4. Discover ──────────────────────────────────────────────────────
        _set(job_id, "discover", "Searching for hidden patterns")
        patterns, context = discovery.discover(encoded, trained)

        dl1_store.update(job_id, patterns=patterns, signals=context)

        # ── 5. Done ──────────────────────────────────────────────────────────
        dl1_store.update(job_id, status="done", stage="ready", progress=100,
                         message=f"Found {len(patterns)} pattern(s)")

    except Exception as exc:                       # noqa: BLE001 - surfaced to the user
        dl1_store.update(
            job_id, status="error", stage="error", progress=100,
            error=str(exc) or exc.__class__.__name__,
            message="Run failed",
        )
        traceback.print_exc()


def start(df: pd.DataFrame, filename: str) -> dl1_store.DL1Job:
    """Create a job and run the pipeline on a background thread."""
    job = dl1_store.create(filename=filename, target_column="")
    thread = threading.Thread(target=run, args=(job.id, df.copy()), daemon=True)
    thread.start()
    return job


# ── Feature recommendation (drives the report's "columns to use") ────────────
def recommend_features(job: dl1_store.DL1Job) -> dict:
    """Decide which columns to use, based on the patterns the user selected.

    Ranking is by unique information carried (autoencoder reconstruction error).
    Columns implicated by a selected pattern are promoted — the user's choice of
    pattern is what makes this recommendation theirs rather than generic.
    """
    training = job.training or {}
    profile = job.profile or {}
    feature_error = training.get("feature_error", [])

    chosen_ids = set(job.selected_patterns or [])
    if job.preferred_pattern:
        chosen_ids.add(job.preferred_pattern)

    selected = [p for p in (job.patterns or []) if p["id"] in chosen_ids]
    implicated: set[str] = set()
    for p in selected:
        implicated.update(p.get("columns", []))

    # Columns a selected redundancy pattern explicitly nominates for removal.
    flagged_redundant: set[str] = set()
    for p in selected:
        if p["type"] == "redundancy":
            flagged_redundant.update(p.get("columns", []))

    ranked, used, ignored = [], [], []
    for rank, item in enumerate(feature_error, start=1):
        name = item["feature"]
        in_pattern = name in implicated
        drop = name in flagged_redundant and not in_pattern

        entry = {
            "rank": rank,
            "feature": name,
            "information_pct": item["error_pct"],
            "in_selected_pattern": in_pattern,
        }
        ranked.append(entry)
        (ignored if drop else used).append(name)

    # Promote pattern-implicated columns to the top of the ranking.
    ranked.sort(key=lambda e: (not e["in_selected_pattern"], -e["information_pct"]))
    for i, entry in enumerate(ranked, start=1):
        entry["rank"] = i

    return {
        "features_used": used,
        "features_ignored": ignored + [d["column"] for d in profile.get("features_dropped", [])],
        "ranking": ranked,
        "excluded_at_load": profile.get("features_dropped", []),
        "selected_patterns": [
            {"id": p["id"], "title": p["title"], "type": p["type"],
             "confidence": p["confidence"], "columns": p.get("columns", []),
             "recommendation": p.get("recommendation")}
            for p in selected
        ],
    }
