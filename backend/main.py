import logging
import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.config import APP_TITLE, CORS_CONFIG
from app.routes.health import router as health_router
from app.routes.data_routes import router as data_router
from app.routes.ml_routes import router as ml_router
from app.routes.deep_routes import router as deep_router
from app.routes.vision_routes import router as vision_router
from app.routes.convert_routes import router as convert_router
from app.routes.export_routes import router as export_router
from app.routes.chat_routes import router as chat_router
from app.routes.dl1_routes import router as dl1_router

logger = logging.getLogger("dataforge")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Refuse to let a silent multi-worker misconfiguration go unnoticed.

    Uploaded data and Deep Learning 2.0 runs live in this process's memory. With
    more than one worker, requests are spread across processes that cannot see
    each other's state, so an upload lands in worker A and the next request hits
    worker B and is told no data was ever uploaded — intermittently, and only
    under load, which is the worst kind of bug to chase. One worker is the
    correct setting for this app until that state moves to Redis or a database.
    """
    workers = int(os.getenv("WEB_CONCURRENCY", "1") or 1)
    if workers > 1:
        logger.warning(
            "WEB_CONCURRENCY=%s. This app keeps uploaded data and DL 2.0 runs in "
            "process memory, which is not shared between workers — expect "
            "intermittent 'No data uploaded' and lost training runs. Set it to 1.",
            workers,
        )
    yield


app = FastAPI(title=APP_TITLE, lifespan=lifespan)

app.add_middleware(CORSMiddleware, **CORS_CONFIG)
# The EDA payload (summary + correlation matrix + chart series) is mostly
# repeated numbers and column names, which gzip flattens to a fraction of the
# wire size. Skips small responses so short JSON isn't compressed for nothing.
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.include_router(health_router)
app.include_router(data_router)
app.include_router(ml_router)
app.include_router(deep_router)
app.include_router(vision_router)
app.include_router(convert_router)
app.include_router(export_router)
app.include_router(chat_router)
app.include_router(dl1_router)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Turn a crash into a readable answer instead of a bare 500.

    Without this, an unexpected error reaches the browser as the string
    "Internal Server Error" with no JSON body — the frontend reads `detail`,
    finds nothing, and shows a generic message, so a deployed failure gives
    neither the user nor the logs anything to go on. The traceback goes to the
    server log; the client gets one sentence and the exception type.
    """
    logger.error("Unhandled error on %s %s\n%s", request.method,
                 request.url.path, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"The server hit an unexpected error "
                           f"({type(exc).__name__}). Please try again — if it "
                           f"persists, try a smaller dataset."},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
