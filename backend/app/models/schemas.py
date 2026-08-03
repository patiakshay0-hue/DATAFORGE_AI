from pydantic import BaseModel
from typing import List, Optional


class SuggestRequest(BaseModel):
    target_column: Optional[str] = None


class TrainRequest(BaseModel):
    models: List[str]
    target_column: Optional[str] = None


class DeepSuggestRequest(BaseModel):
    target_column: Optional[str] = None


class DeepTrainRequest(BaseModel):
    target_column: Optional[str] = None
    config: Optional[dict] = None


class DeepPredictRequest(BaseModel):
    inputs: dict


class AutoConfigRequest(BaseModel):
    target_column: Optional[str] = None


class CleanRequest(BaseModel):
    strategies: List[dict]


class VisionTrainRequest(BaseModel):
    config: Optional[dict] = None


class ConvertChoice(BaseModel):
    choice: Optional[dict] = None


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = []


class DL1SelectRequest(BaseModel):
    """Patterns the user picked from a Deep Learning 1.0 run."""
    pattern_ids: Optional[List[str]] = []
    preferred: Optional[str] = None
