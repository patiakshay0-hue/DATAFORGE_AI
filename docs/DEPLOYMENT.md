# Deploying DataForge AI

Written against a 512 MB / shared-CPU container, which is the constraint every
number below comes from. Koyeb Free and Render Free are the same shape — 512 MB
RAM, ~0.1 vCPU — so the tuning applies to both. On a larger box the limits can
be raised with the environment variables listed at the end.

## Backend on Koyeb (Docker)

[`backend/Dockerfile`](../backend/Dockerfile) builds the image. Koyeb clones the
repo and builds it, so nothing needs pushing to a registry.

### First deploy, step by step

Control panel route — no CLI needed, which is the easier path on Windows since
Koyeb documents CLI installation only for macOS and Linux.

**1. Store the API key as a secret before creating the service.**
Go to [Secrets](https://app.koyeb.com/secrets) → *Create Secret*.
Name it `anthropic-api-key`, paste the key from `backend/.env` as the value.
Doing this first means it is selectable in step 4 instead of being pasted as
plain text. ([Secrets docs](https://www.koyeb.com/docs/reference/secrets))

**2. Create the service.** [Control panel](https://app.koyeb.com) → *Create Web
Service* → **GitHub** → authorise Koyeb if prompted → pick
`patiakshay0-hue/DATAFORGE_AI`, branch `main`.
([Deploy from Git docs](https://www.koyeb.com/docs/build-and-deploy/deploy-with-git))

**3. Set the builder.** In *Builder*, switch from Buildpack to **Dockerfile**.
Then open the builder options and set:

- **Work directory** → `backend`
- **Dockerfile location** → `Dockerfile`

Both are required. Without the work directory Koyeb builds from the repo root,
where there is no Dockerfile and no `requirements.txt`, and the build fails.

**4. Environment variables.**

- `PORT` = `8000` — plain value. Koyeb does not inject this; the container
  defaults to 8000 but the service must agree with it.
- `ANTHROPIC_API_KEY` — choose **Secret** as the type, then pick
  `anthropic-api-key`.
  ([Env var docs](https://www.koyeb.com/docs/build-and-deploy/environment-variables))

**5. Instance and region.** *Free* instance type. Region **Frankfurt** or
**Washington, D.C.** — the Free instance runs nowhere else.
([Instance specs](https://www.koyeb.com/docs/reference/instances))

**6. Exposed port and route.** Port `8000`, protocol `HTTP`, path `/`.

**7. Health check.** On the port entry, switch the protocol from **TCP** to
**HTTP** and set the path to `/health`. Then open the advanced settings and
raise **Grace period** to `120` seconds.

This step is the one people skip, and it is the one that will fail the deploy.
The default grace period is 5 seconds; importing pandas, scipy and scikit-learn
on a 0.1 vCPU instance measured **97 seconds** here. Leave it at 5 and Koyeb
kills the container before it has finished booting, over and over, and the
deploy never goes healthy. ([Health check docs](https://www.koyeb.com/docs/run-and-scale/health-checks))

**8. Deploy.** First build takes a few minutes — it installs the whole
scientific Python stack. Watch the runtime logs for
`Uvicorn running on http://0.0.0.0:8000`.

**9. Verify** at `https://<service>-<org>.koyeb.app/health`. A JSON body with
`status: ok` and a `boot_id` means it is up. Call it twice: if `boot_id` changes
between calls, more than one instance is answering and the per-process state
this app relies on will misbehave — see "One worker" below.

**10. Point the frontend at it.** In the Vercel project, set `VITE_API_URL` to
the Koyeb URL with no trailing slash, then **redeploy** — Vite bakes env vars in
at build time, so changing the variable alone does nothing until the frontend is
rebuilt. ([Vercel env vars](https://vercel.com/docs/environment-variables))

CORS already allows any origin, so nothing else needs changing. Set
`CORS_ORIGINS` to the Vercel domain if you want to lock it down.

### Service settings

| Setting | Value |
|---|---|
| Builder | **Dockerfile** |
| Work directory | `backend` |
| Dockerfile location | `Dockerfile` (relative to the work directory) |
| Exposed port | `8000`, protocol `http` |
| Route | `/` → `8000` |
| Health check | **HTTP**, path `/health` |
| Environment | `PORT=8000`, `ANTHROPIC_API_KEY` (as a **secret**) |

`PORT` is not injected automatically on Koyeb — you set it yourself, and it must
match the port you expose. The `CMD` falls back to 8000 if it is unset.

Set the health check to HTTP `/health` rather than leaving it on the default TCP
probe. TCP only proves something accepted a connection; `/health` proves the app
booted and its routes are mounted, so a container that is up but broken gets
restarted instead of quietly serving errors. Give it a **grace period of 120s** —
a cold start on a 0.1 vCPU instance measured 97 seconds, and a shorter grace
period will kill the container before it has finished importing pandas.

Equivalent CLI:

```sh
koyeb app init dataforge \
  --git github.com/patiakshay0-hue/DATAFORGE_AI \
  --git-branch main \
  --git-builder docker \
  --git-workdir backend \
  --git-docker-dockerfile Dockerfile \
  --ports 8000:http \
  --routes /:8000 \
  --env PORT=8000 \
  --env ANTHROPIC_API_KEY=@anthropic-api-key
```

(`@name` references a Koyeb secret. Create it first with
`koyeb secrets create anthropic-api-key`. `--env KEY={{secret.name}}` is an
equivalent syntax.)

### After it is up

Point the frontend at the new URL: set `VITE_API_URL` to
`https://<service>-<org>.koyeb.app` in the Vercel project's environment
variables and redeploy. The backend allows any origin by default, so no CORS
change is needed; set `CORS_ORIGINS` if you want to lock it to the Vercel domain.

### Free instance facts worth knowing

512 MB RAM, 0.1 vCPU, 2 GB SSD. One per organisation, Frankfurt or Washington
D.C. only, and **it scales to zero after an hour without traffic** — so the cold
start above is a normal occurrence, not a fault. The frontend already pings
`/health` on page load to absorb it; an uptime pinger every ~30 minutes avoids it
entirely.

### The image

807 MB, from `python:3.11-slim` via a two-stage build so pip's cache and wheels
never reach the runtime layer. It started at 1.98 GB — against a 2 GB disk — and
two things accounted for the difference:

- **`nvidia-nccl-cu12`, 400 MB.** A transitive dependency of the default
  `xgboost` Linux wheel: CUDA libraries, on an instance with no GPU. The
  requirements file now selects `xgboost-cpu` on Linux and plain `xgboost`
  elsewhere, via an environment marker — both import as `xgboost`, and
  `xgboost-cpu` publishes Linux wheels only, so a Windows or macOS dev machine
  still resolves the normal package.
- **`plotly`, 70 MB.** Listed as a dependency but imported nowhere in the
  backend — charts are built as plain JSON series in `core/analyzer.py` and
  drawn by recharts in the browser. Removed.

The image also pins `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`
and `NUMEXPR_NUM_THREADS` to 1. numpy, scipy, scikit-learn and xgboost size their
thread pools from the *host's* core count, which on a shared node is dozens — so
a container with a tenth of a CPU would otherwise spawn dozens of threads that do
nothing but contend for it, each with its own stack and scratch buffers.

`.env` is excluded by [`.dockerignore`](../backend/.dockerignore): it holds a
real API key, and anything copied into an image is readable by anyone who can
pull it — and survives in that layer even if a later step deletes it. Supply the
key as a Koyeb secret instead. The container runs as a non-root user (uid 10001),
which matters here because the app parses untrusted uploaded files.

### Measured in the container

Under `--memory=512m`, the full suite passes — every endpoint, on a 60,000-row
CSV — with **peak RSS 314 MB of 512 MB**, no OOM kill, no restarts, health check
green. uvicorn runs as PID 1 (via `exec`), so it receives SIGTERM directly and
stops in 2.6s instead of being killed at the end of the grace period.

At a hard **0.1 CPU** cap, the same workload:

| | time |
|---|---|
| Cold start to first healthy response | 97s |
| Upload + EDA, 60k rows | 8.6s |
| `/train`, Random Forest + XGBoost | 92.8s |
| `/deep/train` | 64.0s (time budget held) |
| Deep Learning 2.0, full run | 173.3s |

Deep Learning 2.0 is a background job you poll, so three minutes is fine. The one
to watch is `/train`: it is a single synchronous request, and 93 seconds is close
to the edge of what proxies tolerate. If you hit timeouts there, lower
`TRAIN_MAX_ROWS` or select fewer models per run — Random Forest and SVM dominate
that number.

## Backend on Render

Render builds from `requirements.txt` rather than the Dockerfile, with:

```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Everything below the "One worker" rule applies here too; what the Dockerfile
sets as environment defaults, you set in Render's dashboard — in particular
`OMP_NUM_THREADS=1` and its siblings, which Render will not set for you.

## One worker, on any host

Uploaded data and Deep Learning 2.0 runs live in the process's own memory, and
workers do not share memory. With two of them, an upload lands in one process and
the next request is routed to the other, which has never heard of it — so the app
intermittently reports "No data uploaded" and loses training runs, only under
load, with nothing in the logs. The backend logs a warning at startup if
`WEB_CONCURRENCY` is set above 1; the Dockerfile pins it to 1.

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

### 4. Image-zip uploads killed the container

A 290 MB image archive returned 502 and restarted the backend. `await
file.read()` was the cause: Starlette has already written the request body to a
`SpooledTemporaryFile` by the time a handler runs — on disk for anything over
1 MB — so reading it does not fetch the upload, it copies it into RAM a second
time. 290 MB resident before a single image had been decoded, then the decode on
top of that.

`/vision/upload` now hands the file object to `zipfile`, which seeks around it on
disk. Two other paths had the same shape and are bounded through the shared
`app/uploads.py`: `/convert/inspect` (which also keeps its bytes for the rest of
the session) and `/vision/predict`.

Decoded images are kept at 160px on the longest edge rather than 256px.
Everything downstream shrinks them further — the CNN transform to 128, the
scikit-learn descriptor to 32, thumbnails to 96 — so 256 was four times the
pixels of the largest consumer, and 236 MB of resident images at the 1,200-image
cap. Now 92 MB.

Same 290 MB archive inside a 512 MB container after the fix: **succeeds twice in
a row at peak process memory 235 MB**, no OOM kill.

### 5. Uploads

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
| `MAX_UPLOAD_MB` | 50 | Tabular upload (parsed into a DataFrame held in memory) |
| `MAX_ZIP_MB` | 300 | Image archives — larger, because they stream from disk |
| `MAX_UNCOMPRESSED_MB` | 2048 | What an archive may expand to, read from the zip index |
| `TRAIN_MAX_ROWS` | 20,000 | Rows for ML + supervised deep learning |
| `TRAIN_TIME_BUDGET_SECONDS` | 60 | Wall clock in the epoch loop |
| `SVM_MAX_ROWS` | 5,000 | Rows for kernel SVM alone (it scales quadratically) |
| `SVM_CACHE_MB` | 64 | SVM kernel cache |
| `DL1_MAX_TRAIN_ROWS` | 25,000 | Rows for the Deep Learning 2.0 autoencoder |
| `SKLEARN_WORKING_MEMORY_MB` | 64 | scikit-learn chunked pairwise ops |
| `CORS_ORIGINS` | `*` | Comma-separated origins, or `*` |
| `WEB_CONCURRENCY` | 1 | Must stay 1 — see above |
| `PORT` | 8000 | Port uvicorn binds (Koyeb requires it set explicitly) |

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
