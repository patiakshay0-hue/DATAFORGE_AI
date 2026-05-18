from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import uvicorn
from dotenv import load_dotenv

from core.loader import load_data, get_schema
from core.analyzer import perform_eda
from core.insights import generate_insights
from core.trainer import suggest_models, train_selected_models
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
