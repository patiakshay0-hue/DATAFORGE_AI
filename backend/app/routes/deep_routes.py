from fastapi import APIRouter, HTTPException
from app.store import data_store
from core.deep_trainer import (
    suggest_config, train_neural_network, recommend_targets,
    predict as deep_predict, auto_optimize_config, discover_patterns,
)
from app.models.schemas import (
    DeepSuggestRequest, DeepTrainRequest, DeepPredictRequest, AutoConfigRequest,
)

router = APIRouter()


@router.post("/deep/suggest")
async def deep_suggest(request: DeepSuggestRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    return suggest_config(data_store["current_df"], request.target_column)


@router.post("/deep/train")
async def deep_train(request: DeepTrainRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    if not request.target_column:
        raise HTTPException(status_code=400, detail="A target column is required for deep learning")
    result = train_neural_network(data_store["current_df"], request.target_column, request.config)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Training failed"))
    return result


@router.get("/deep/recommend-targets")
async def deep_recommend_targets():
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    return recommend_targets(data_store["current_df"])


@router.post("/deep/predict")
async def deep_predict_route(request: DeepPredictRequest):
    result = deep_predict(request.inputs)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Prediction failed"))
    return result


@router.post("/deep/auto-config")
async def deep_auto_config(request: AutoConfigRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    config = auto_optimize_config(data_store["current_df"], request.target_column)
    patterns = discover_patterns(data_store["current_df"])
    return {**config, "patterns": patterns}
