# DataForge AI backend — built for a Koyeb Free instance:
# 512 MB RAM, 0.1 vCPU, 2 GB SSD. Every choice below follows from one of those.
#
# Build:  docker build -t dataforge-backend .          (from backend/)
# Run:    docker run -p 8000:8000 -e PORT=8000 dataforge-backend

# ── Build stage ───────────────────────────────────────────────────────────────
# Separate stage so pip's cache, wheels and build metadata stay out of the final
# image. Every dependency here ships a manylinux wheel for Python 3.11, so no
# compiler is needed — which is also why the runtime stage can stay slim.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied on its own so the dependency layer is cached and only reinstalls when
# requirements.txt actually changes — application edits then rebuild in seconds.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Thread caps. This is the single most important runtime setting on a 0.1 vCPU
# box. numpy, scipy, scikit-learn and xgboost all size their thread pools from
# the *host's* core count, which on a shared node is dozens — so a container with
# a tenth of a CPU spawns dozens of threads that do nothing but contend for it,
# each carrying its own stack and scratch buffers. Pinning them to one trades
# nothing (there is no second core to use) for a large drop in memory and
# context-switching.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    # Uploaded data and DL 2.0 runs live in process memory and are not shared
    # between workers — see the lifespan warning in main.py.
    WEB_CONCURRENCY=1 \
    # Unbuffered, so logs reach Koyeb as they happen rather than when a buffer
    # fills. On a crash, a buffered log is a lost log.
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Non-root: nothing here needs to write outside /tmp, and the app parses
# untrusted uploads, which is exactly the code you least want running as root.
RUN useradd --create-home --uid 10001 dataforge
COPY --chown=dataforge:dataforge . .
USER dataforge

# Documentation only — Koyeb routes by the port you declare on the service, and
# the CMD below honours $PORT whatever it is set to.
EXPOSE 8000

# Local convenience. Koyeb runs its own health check against the service port;
# point that one at /health too, so a container that is up but broken is
# restarted rather than left serving errors.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",\"8000\")}/health', timeout=4).status == 200 else 1)"

# Shell form so ${PORT} expands — Koyeb does not inject PORT automatically, so
# this defaults to 8000 and you set the variable on the service to match the
# port you expose. `exec` hands PID 1 to uvicorn so it receives SIGTERM directly
# and shuts down cleanly instead of being killed after the grace period.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
