import os
import time

from fastapi import APIRouter

router = APIRouter()

# Set once at import. A change in this value across two responses means the
# process restarted — which on a small container usually means it was OOM-killed
# mid-run, and is exactly what the frontend needs to distinguish from a dropped
# packet before it tells the user a run has been lost.
BOOT_ID = os.urandom(8).hex()
STARTED_AT = time.time()


@router.get("/")
async def root():
    return {"message": "DataForge AI Backend is running"}


@router.get("/health")
async def health():
    """Cheap liveness probe. Safe to poll — touches no data and allocates nothing.

    Also the endpoint to point an uptime pinger at: a free-tier container that
    sleeps after 15 idle minutes takes the better part of a minute to answer its
    next request, which is what makes a first upload feel broken.
    """
    return {
        "status": "ok",
        "boot_id": BOOT_ID,
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
    }
