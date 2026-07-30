from fastapi import APIRouter, UploadFile, File, HTTPException
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
async def vision_status():
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
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip archive of labelled image folders.")
    content = await file.read()
    result = load_image_zip(content)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Could not read the image archive"))
    return result


@router.post("/vision/train")
async def vision_train(request: VisionTrainRequest):
    if not VISION_READY:
        raise HTTPException(status_code=400, detail="Image tools need Pillow installed on the backend.")
    result = train_classifier(request.config)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Training failed"))
    return result


@router.post("/vision/predict")
async def vision_predict(file: UploadFile = File(...)):
    content = await file.read()
    result = predict_image(content)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Prediction failed"))
    return result


@router.get("/vision/dataset")
async def vision_get_dataset():
    ds = vision_dataset()
    return ds if ds else {"status": "empty"}
