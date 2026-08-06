"""Per-job state for the Deep Learning 2.0 pipeline.

Why this exists instead of reusing `app.store.data_store`:

`data_store` is a single module-level dict shared by every request. That is fine for
the one-shot upload → analyse flow, but DL 1.0 is a *stateful, multi-step* workflow —
the user trains, then browses patterns, then selects one, then generates a report,
often minutes apart. A second upload in another tab would clobber a run mid-flow.

Jobs are keyed by an opaque id, carry their own copy of everything the later steps
need, and expire on a TTL so long-lived processes don't leak trained models.
"""

from __future__ import annotations

import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Any

# How long a finished job stays retrievable, and how many we keep at most.
JOB_TTL_SECONDS = 60 * 60          # 1 hour
MAX_JOBS = 20

# Pipeline stages, in execution order. The frontend renders progress from these,
# so the names are part of the API contract.
STAGES = [
    ("preprocess", "Analysing & preprocessing data"),
    ("configure",  "Selecting hyperparameters"),
    ("train",      "Training neural network"),
    ("patterns",   "Discovering hidden patterns"),
    ("report",     "Preparing report"),
]


@dataclass
class DL1Job:
    """One end-to-end Deep Learning 2.0 run."""

    id: str
    filename: str
    target_column: str
    created_at: float = field(default_factory=time.time)

    status: str = "pending"        # pending | running | done | error
    stage: str = ""                # current stage key from STAGES
    progress: int = 0              # 0-100
    message: str = ""
    error: str | None = None

    # Filled in as the pipeline advances.
    profile: dict | None = None        # dataset profile (§2)
    config: dict | None = None         # AutoModelConfig output (§3)
    training: dict | None = None       # train_neural_network result (§6)
    patterns: list[dict] = field(
        default_factory=list)   # narrated patterns (§4)
    signals: dict | None = None        # raw SignalBundle, kept for the report

    # User choices from the pattern-selection step (§5).
    selected_patterns: list[str] = field(default_factory=list)
    preferred_pattern: str | None = None

    def public(self) -> dict:
        """Status view — cheap to poll, excludes heavy payloads."""
        return {
            "job_id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "filename": self.filename,
            "target_column": self.target_column,
        }

    def result(self) -> dict:
        """Full payload once the run has finished."""
        return {
            **self.public(),
            "profile": self.profile,
            "config": self.config,
            "training": self.training,
            "patterns": self.patterns,
            "selected_patterns": self.selected_patterns,
            "preferred_pattern": self.preferred_pattern,
        }


# Guarded because the pipeline runs in a worker thread while HTTP handlers read it.
_lock = threading.Lock()
_jobs: dict[str, DL1Job] = {}


def _evict_locked() -> None:
    """Drop expired jobs, then oldest-first if still over capacity. Caller holds lock."""
    now = time.time()
    for jid in [j for j, job in _jobs.items() if now - job.created_at > JOB_TTL_SECONDS]:
        _jobs.pop(jid, None)

    if len(_jobs) > MAX_JOBS:
        for jid, _ in sorted(_jobs.items(), key=lambda kv: kv[1].created_at)[: len(_jobs) - MAX_JOBS]:
            _jobs.pop(jid, None)


def create(filename: str, target_column: str) -> DL1Job:
    job = DL1Job(id=uuid.uuid4().hex[:12],
                 filename=filename, target_column=target_column)
    with _lock:
        _evict_locked()
        _jobs[job.id] = job
    return job


def get(job_id: str) -> DL1Job | None:
    with _lock:
        return _jobs.get(job_id)


def update(job_id: str, **fields: Any) -> DL1Job | None:
    """Apply field updates atomically so a polling reader never sees a torn state."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        for key, value in fields.items():
            setattr(job, key, value)
        return job


def clear() -> None:
    """Test helper — drop everything."""
    with _lock:
        _jobs.clear()
