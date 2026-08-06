# Deploying DataForge AI

Written against a 512 MB / shared-CPU container (Render free tier), which is the
constraint every number below comes from. On a larger box the limits can be
raised with the environment variables listed at the end.

## Start command

```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

**One worker. Not two.** Uploaded data and Deep Learning 2.0 runs live in the
process's own memory, and workers do not share memory. With two of them, an
upload lands in one process and the next request is routed to the other, which
has never heard of it — so the app intermittently reports "No data uploaded" and
loses training runs, only under load, with nothing in the logs. The backend logs
a warning at startup if `WEB_CONCURRENCY` is set above 1.

To confirm a deployment is running a single worker, call `/health` a few times:
`boot_id` identifies the process, so a value that changes between calls means
more than one process is answering.

## What was wrong, and what fixed it

Two independent bugs both surfaced as **"Lost contact with the server"** after a
Deep Learning 2.0 run.

### 1. The container was being OOM-killed

`silhouette_score` builds the full pairwise distance matrix — n rows cost n² × 8
bytes. Pattern discovery was scoring 10,000 rows, once per candidate cluster
count:

| rows scored | distance matrix |
|---|---|
| 5,000 | 200 MB |
| 10,000 | **800 MB** |
| 50,000 | 20 GB |

Measured peak for one DL 2.0 run on a 3.6 MB, 50,000-row CSV: **1,948 MB**. The
512 MB container was killed part-way through, the browser's next status poll got
no answer, and the run vanished. Nothing was logged, because the process was
gone.

Scoring is now done on a bounded random subsample (`core/sampling.py`), which
estimates the same statistic to more decimal places than the UI displays. Same
run, same data, after the fix: **419 MB peak**, and pattern discovery went from
19.3s to 2.4s.

Two more allocations of the same kind were found and capped: scikit-learn's
`working_memory` (defaults to 1024 MB per chunk — twice the whole container) and
SVM's kernel cache (defaults to 200 MB, measured at +234 MB for one model).

Full workload — upload, 4 ML models, deep training, a DL 2.0 run, both PDF
reports — on a 22 MB / 300,000-row CSV now peaks at **327 MB**.

### 2. Heavy requests froze the whole server

Every route outside `data_routes.py` was declared `async def` while doing
blocking CPU work. FastAPI runs an `async def` handler directly on the event
loop, so nothing else is served until it returns; a plain `def` handler goes to a
threadpool instead. Measured with a 5-second handler and a health poll running
throughout:

| handler | polls answered | median latency |
|---|---|---|
| `async def` | 1 | 5,018 ms |
| `def` | 14 | 150 ms |

So while the ML or Deep Learning tab trained — minutes, on a shared CPU — the DL
2.0 status poll got no reply at all, and reported the server lost. Handlers are
now `def`, or push their blocking work to the threadpool with
`run_in_threadpool` when they need to `await` an upload first.

### 3. The poll gave up too easily

The status poll ran every 700 ms and treated **any** failure — a dropped packet,
a proxy hiccup, a 404 — as fatal, ending the run on the first one. It now polls
every 1.5s, tolerates 8 consecutive failures (~12s) while showing a
"Reconnecting" note, and distinguishes a 404 (the run is genuinely gone) from
silence (worth retrying).

### 4. Uploads

- Rejected before sending if the format or size is wrong, rather than after.
- Size enforced server-side while reading, so an oversized file is refused at the
  limit instead of being buffered whole.
- One automatic retry when the server never answered — a sleeping container drops
  the request that wakes it. A 4xx is never retried; it would fail again.
- `/health` is pinged on page load, so the ~50s cold start happens while the user
  is choosing a file rather than inside their upload.

## Work limits

Long jobs are bounded so a synchronous request always comes back. Every limit is
reported in the UI rather than applied silently — sampled training says so on the
results panel, and a shortened run says how many epochs it managed.

| Setting | Default | What it bounds |
|---|---|---|
| `MAX_UPLOAD_MB` | 50 | Upload size ceiling |
| `TRAIN_MAX_ROWS` | 20,000 | Rows for ML + supervised deep learning |
| `TRAIN_TIME_BUDGET_SECONDS` | 60 | Wall clock in the epoch loop |
| `SVM_MAX_ROWS` | 5,000 | Rows for kernel SVM alone (it scales quadratically) |
| `SVM_CACHE_MB` | 64 | SVM kernel cache |
| `DL1_MAX_TRAIN_ROWS` | 25,000 | Rows for the Deep Learning 2.0 autoencoder |
| `SKLEARN_WORKING_MEMORY_MB` | 64 | scikit-learn chunked pairwise ops |
| `CORS_ORIGINS` | `*` | Comma-separated origins, or `*` |

Effect of the training limits on a 50,000-row dataset: `/deep/train` went from
191s to 53s, `/train` with 4 models from 26.5s to 2.2s for the SVM alone.

## Keeping the backend awake

A free-tier container sleeps after 15 idle minutes and takes the better part of a
minute to answer its next request, which is most of what "uploads sometimes take
forever" is. `/health` is cheap and allocates nothing — point an uptime pinger at
it every 10 minutes to keep the container warm.

## torch is not installed in production

`requirements.txt` omits `torch`/`torchvision` because they exceed the free-tier
image size, so production runs the scikit-learn fallbacks everywhere — a
different code path from a dev machine that has torch installed. To test what is
actually deployed, install without torch, or block the import.

Dependencies are pinned to version ranges. Unpinned, a redeploy that changes
nothing in this repo can still come up broken, because it resolves to whatever
was published that day.
