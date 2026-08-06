from fastapi import APIRouter, HTTPException
from app.store import data_store
from core.trainer import suggest_models, train_selected_models
from app.models.schemas import SuggestRequest, TrainRequest

router = APIRouter()

# Sync `def`, not `async def` — same reasoning as data_routes. Model training is
# seconds of blocking CPU work; on the event loop it would freeze every other
# request for its whole duration, including the Deep Learning progress polls,
# which the frontend reads as the server having gone away.


@router.post("/suggest")
def suggest(request: SuggestRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    return suggest_models(data_store["current_df"], request.target_column)


@router.post("/train")
def train(request: TrainRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    if not request.models:
        raise HTTPException(status_code=400, detail="No models selected")
    return train_selected_models(data_store["current_df"], request.models, request.target_column)
