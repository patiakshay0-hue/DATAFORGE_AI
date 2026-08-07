from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool

from app.uploads import enforce_size, rewound, read_capped, MAX_ZIP_MB
from core.vision_trainer import (
    load_image_zip, train_classifier, predict_image, current_dataset as vision_dataset,
    VISION_AVAILABLE as VISION_READY, HAS_TORCH, ENGINE_NAME as VISION_ENGINE,
)
from app.models.schemas import VisionTrainRequest

router = APIRouter()


def _vision_has_dataset():
    from core.vision_trainer import VISION_STORE
    return bool(VISION_STORE.get("images"))


@router.get("/vision/status")
def vision_status():
    return {
        "ready": VISION_READY,
        "engine": VISION_ENGINE,
        "cnn": HAS_TORCH,
        "has_dataset": bool(_vision_has_dataset()),
    }


@router.post("/vision/upload")
async def vision_upload(file: UploadFile = File(...)):
    if not VISION_READY:
        raise HTTPException(status_code=400, detail="Image tools need Pillow installed on the backend.")
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip archive of labelled image folders.")

    enforce_size(file, MAX_ZIP_MB, "archive")
    # The archive is handed over as a file object rather than as bytes. Starlette
    # has already spooled it to disk, so this reads it from there instead of
    # loading a second copy into memory — which is what used to take the
    # container down on a large upload.
    result = await run_in_threadpool(load_image_zip, rewound(file))
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Could not read the image archive"))
    return result


@router.post("/vision/train")
def vision_train(request: VisionTrainRequest):
    if not VISION_READY:
        raise HTTPException(status_code=400, detail="Image tools need Pillow installed on the backend.")
    result = train_classifier(request.config)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Training failed"))
    return result


@router.post("/vision/predict")
async def vision_predict(file: UploadFile = File(...)):
    # One image: small enough to hold, but still bounded rather than unlimited.
    content = await run_in_threadpool(read_capped, file, 25, "image")
    result = await run_in_threadpool(predict_image, content)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Prediction failed"))
    return result


@router.get("/vision/dataset")
def vision_get_dataset():
    ds = vision_dataset()
    return ds if ds else {"status": "empty"}
