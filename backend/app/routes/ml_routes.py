from fastapi import APIRouter, HTTPException
from app.store import data_store
from core.trainer import suggest_models, train_selected_models
from app.models.schemas import SuggestRequest, TrainRequest

router = APIRouter()


@router.post("/suggest")
async def suggest(request: SuggestRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    return suggest_models(data_store["current_df"], request.target_column)


@router.post("/train")
async def train(request: TrainRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    if not request.models:
        raise HTTPException(status_code=400, detail="No models selected")
    return train_selected_models(data_store["current_df"], request.models, request.target_column)
