from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import List, Optional
import io
import pandas as pd
import uvicorn
from dotenv import load_dotenv

from core.loader import load_data, get_schema
from core.analyzer import perform_eda
from core.insights import generate_insights
from core.trainer import suggest_models, train_selected_models
from core.cleaner import missing_report, handle_missing
from core.deep_trainer import suggest_config, train_neural_network, recommend_targets, predict as deep_predict
from core.vision_trainer import (
    load_image_zip, train_classifier, predict_image, current_dataset as vision_dataset,
    HAS_TORCH as VISION_READY,
)
from core import converter
from core.exporter import generate_pdf_report
from core.chat import stream_chat_response, check_api_key

load_dotenv()  # load ANTHROPIC_API_KEY from backend/.env

app = FastAPI(title="DataForge AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_store = {}   # { current_df, schema, eda, filename }


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "DataForge AI Backend is running"}


# ── Upload ─────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = load_data(content, file.filename)
        schema = get_schema(df)
        eda_results = perform_eda(df)

        data_store["current_df"] = df
        data_store["schema"]     = schema
        data_store["eda"]        = eda_results
        data_store["filename"]   = file.filename

        return {
            "filename": file.filename,
            "schema": schema,
            "eda": eda_results,
            "preview": df.head(10).fillna("").to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Insights ───────────────────────────────────────────────────────────────────
@app.get("/insights")
async def get_data_insights():
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    return generate_insights(data_store["current_df"])


# ── EDA re-run ─────────────────────────────────────────────────────────────────
@app.get("/analyze")
async def analyze_data():
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    return perform_eda(data_store["current_df"])


# ── ML: suggest + train ────────────────────────────────────────────────────────
class SuggestRequest(BaseModel):
    target_column: Optional[str] = None

class TrainRequest(BaseModel):
    models: List[str]
    target_column: Optional[str] = None

@app.post("/suggest")
async def suggest(request: SuggestRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    return suggest_models(data_store["current_df"], request.target_column)

@app.post("/train")
async def train(request: TrainRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    if not request.models:
        raise HTTPException(status_code=400, detail="No models selected")
    return train_selected_models(data_store["current_df"], request.models, request.target_column)


# ── Deep Learning: suggest + train ───────────────────────────────────────────
class DeepSuggestRequest(BaseModel):
    target_column: Optional[str] = None

class DeepTrainRequest(BaseModel):
    target_column: Optional[str] = None
    config: Optional[dict] = None

@app.post("/deep/suggest")
async def deep_suggest(request: DeepSuggestRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    return suggest_config(data_store["current_df"], request.target_column)

@app.post("/deep/train")
async def deep_train(request: DeepTrainRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    if not request.target_column:
        raise HTTPException(status_code=400, detail="A target column is required for deep learning")
    result = train_neural_network(data_store["current_df"], request.target_column, request.config)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Training failed"))
    return result

# ── Data cleaning: handle missing values ──────────────────────────────────────
@app.get("/clean/missing-report")
async def clean_missing_report():
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    return missing_report(data_store["current_df"])

class CleanRequest(BaseModel):
    strategies: List[dict]

@app.post("/clean/apply")
async def clean_apply(request: CleanRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    cleaned, summary = handle_missing(data_store["current_df"], request.strategies)
    if cleaned.shape[1] == 0:
        raise HTTPException(status_code=400, detail="That would drop every column — adjust your choices.")
    schema = get_schema(cleaned)
    eda_results = perform_eda(cleaned)
    data_store["current_df"] = cleaned
    data_store["schema"]     = schema
    data_store["eda"]        = eda_results
    return {
        "filename": data_store.get("filename", "dataset"),
        "schema": schema,
        "eda": eda_results,
        "preview": cleaned.head(10).fillna("").to_dict(orient="records"),
        "clean_summary": summary,
    }


@app.get("/deep/recommend-targets")
async def deep_recommend_targets():
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    return recommend_targets(data_store["current_df"])

class DeepPredictRequest(BaseModel):
    inputs: dict

@app.post("/deep/predict")
async def deep_predict_route(request: DeepPredictRequest):
    result = deep_predict(request.inputs)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Prediction failed"))
    return result


# ── Vision: image classification (upload zip → train CNN → predict) ───────────
class VisionTrainRequest(BaseModel):
    config: Optional[dict] = None

@app.get("/vision/status")
async def vision_status():
    return {"ready": VISION_READY, "has_dataset": bool(_vision_has_dataset())}

def _vision_has_dataset():
    from core.vision_trainer import VISION_STORE
    return bool(VISION_STORE.get("images"))

@app.post("/vision/upload")
async def vision_upload(file: UploadFile = File(...)):
    if not VISION_READY:
        raise HTTPException(status_code=400, detail="Image classification needs torch + torchvision installed on the backend.")
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip archive of labelled image folders.")
    content = await file.read()
    result = load_image_zip(content)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Could not read the image archive"))
    return result

@app.post("/vision/train")
async def vision_train(request: VisionTrainRequest):
    if not VISION_READY:
        raise HTTPException(status_code=400, detail="Image classification needs torch + torchvision installed on the backend.")
    result = train_classifier(request.config)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Training failed"))
    return result

@app.post("/vision/predict")
async def vision_predict(file: UploadFile = File(...)):
    content = await file.read()
    result = predict_image(content)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Prediction failed"))
    return result

@app.get("/vision/dataset")
async def vision_get_dataset():
    ds = vision_dataset()
    return ds if ds else {"status": "empty"}


# ── Import & Convert: any format → CSV, or route an image zip ─────────────────
class ConvertChoice(BaseModel):
    choice: Optional[dict] = None

@app.post("/convert/inspect")
async def convert_inspect(file: UploadFile = File(...)):
    content = await file.read()
    return converter.inspect_upload(content, file.filename)

@app.post("/convert/convert")
async def convert_run(request: ConvertChoice):
    result = converter.convert(request.choice)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Conversion failed"))
    return result

@app.post("/convert/image-metadata")
async def convert_image_metadata():
    result = converter.image_metadata()
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Could not build metadata"))
    return result

@app.get("/convert/download")
async def convert_download():
    csv_bytes, name, _ = converter.get_converted()
    if csv_bytes is None:
        raise HTTPException(status_code=404, detail="Nothing converted yet")
    return StreamingResponse(
        io.BytesIO(csv_bytes), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )

@app.post("/convert/load")
async def convert_load():
    """Push the converted table into the main analysis pipeline (like /upload)."""
    csv_bytes, name, df = converter.get_converted()
    if df is None:
        raise HTTPException(status_code=400, detail="Convert a file to CSV first")
    schema = get_schema(df)
    eda_results = perform_eda(df)
    data_store["current_df"] = df
    data_store["schema"]     = schema
    data_store["eda"]        = eda_results
    data_store["filename"]   = name
    return {
        "filename": name, "schema": schema, "eda": eda_results,
        "preview": df.head(10).fillna("").to_dict(orient="records"),
    }

@app.post("/convert/send-to-vision")
async def convert_send_to_vision():
    if not VISION_READY:
        raise HTTPException(status_code=400, detail="Image classification needs torch + torchvision installed.")
    content = converter.get_stored_zip()
    if content is None:
        raise HTTPException(status_code=400, detail="Upload an image zip first")
    result = load_image_zip(content)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Could not load images"))
    return result


# ── Export PDF ─────────────────────────────────────────────────────────────────
@app.get("/export")
async def export_report():
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded. Upload a file first.")
    try:
        df       = data_store["current_df"]
        schema   = data_store.get("schema", [])
        eda      = data_store.get("eda", {})
        filename = data_store.get("filename", "dataset")
        insights = generate_insights(df)

        pdf_bytes = generate_pdf_report(filename, schema, eda, insights, df)
        safe_name = filename.rsplit(".", 1)[0].replace(" ", "_")

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}_report.pdf"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# ── Chat ───────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    history:  Optional[List[dict]] = []

@app.get("/chat/status")
async def chat_status():
    return check_api_key()

@app.post("/chat")
async def chat(request: ChatRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded. Upload a file first.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    df       = data_store["current_df"]
    schema   = data_store.get("schema", [])
    eda      = data_store.get("eda", {})
    filename = data_store.get("filename", "dataset")

    return StreamingResponse(
        stream_chat_response(
            request.question, request.history or [],
            df, schema, eda, filename,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
