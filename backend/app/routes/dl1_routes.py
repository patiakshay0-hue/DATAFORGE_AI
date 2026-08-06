"""Deep Learning 2.0 routes — target-free training, pattern selection, reporting.

Flow:
    POST /dl1/run           upload a file (or reuse the loaded dataset) → job_id
    GET  /dl1/status/{id}   cheap poll for stage + progress
    GET  /dl1/result/{id}   full payload once finished
    POST /dl1/select/{id}   record the patterns the user picked
    GET  /dl1/report/{id}   PDF report
"""

import io

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from app import dl1_store
from app.store import data_store
from core.dl1 import pipeline, report as dl1_report
from core.loader import load_data
from app.models.schemas import DL1SelectRequest

router = APIRouter()


@router.post("/dl1/run")
async def dl1_run(file: UploadFile = File(None)):
    """Start a run. Accepts a fresh upload, or falls back to the already-loaded data.

    No target column is involved anywhere — that is the point of this module.
    """
    if file is not None:
        try:
            content = await file.read()
            df = load_data(content, file.filename)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        filename = file.filename
    elif "current_df" in data_store:
        df = data_store["current_df"]
        filename = data_store.get("filename", "dataset")
    else:
        raise HTTPException(
            status_code=400,
            detail="Upload a dataset to analyse, or load one from the Upload tab first.",
        )

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="The dataset is empty.")
    if len(df) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 10 rows to learn anything — this dataset has {len(df)}.",
        )

    job = pipeline.start(df, filename)
    return job.public()


@router.get("/dl1/status/{job_id}")
async def dl1_status(job_id: str):
    job = dl1_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="Run not found or expired.")
    return job.public()


@router.get("/dl1/result/{job_id}")
async def dl1_result(job_id: str):
    job = dl1_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="Run not found or expired.")
    if job.status == "error":
        raise HTTPException(status_code=400, detail=job.error or "Run failed.")
    if job.status != "done":
        raise HTTPException(
            status_code=409, detail="Run is still in progress.")
    return job.result()


@router.post("/dl1/select/{job_id}")
async def dl1_select(job_id: str, request: DL1SelectRequest):
    job = dl1_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="Run not found or expired.")
    if job.status != "done":
        raise HTTPException(
            status_code=409, detail="Run is still in progress.")

    valid = {p["id"] for p in (job.patterns or [])}
    chosen = [pid for pid in (request.pattern_ids or []) if pid in valid]
    preferred = request.preferred if request.preferred in valid else None
    if preferred and preferred not in chosen:
        chosen.append(preferred)

    dl1_store.update(job_id, selected_patterns=chosen,
                     preferred_pattern=preferred)
    job = dl1_store.get(job_id)
    return {**job.public(), "recommendation": pipeline.recommend_features(job)}


@router.get("/dl1/recommendation/{job_id}")
async def dl1_recommendation(job_id: str):
    job = dl1_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="Run not found or expired.")
    if job.status != "done":
        raise HTTPException(
            status_code=409, detail="Run is still in progress.")
    return pipeline.recommend_features(job)


@router.get("/dl1/report/{job_id}")
async def dl1_report_pdf(job_id: str):
    job = dl1_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="Run not found or expired.")
    if job.status != "done":
        raise HTTPException(
            status_code=409, detail="Run is still in progress.")

    try:
        pdf = dl1_report.generate(job, pipeline.recommend_features(job))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not build report: {exc}")

    name = (job.filename or "dataset").rsplit(".", 1)[0]
    return StreamingResponse(
        io.BytesIO(pdf), media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{name}_deep_learning_1.0.pdf"'},
    )
