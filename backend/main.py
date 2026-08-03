from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(title=APP_TITLE)

app.add_middleware(CORSMiddleware, **CORS_CONFIG)

app.include_router(health_router)
app.include_router(data_router)
app.include_router(ml_router)
app.include_router(deep_router)
app.include_router(vision_router)
app.include_router(convert_router)
app.include_router(export_router)
app.include_router(chat_router)
app.include_router(dl1_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
