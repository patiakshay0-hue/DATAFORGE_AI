import os

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.store import data_store
from core.loader import load_data, get_schema
from core.analyzer import perform_eda
from core.insights import generate_insights
from core.cleaner import missing_report, handle_missing
from app.models.schemas import CleanRequest

router = APIRouter()

# Hard ceiling on an upload, matching what the UI advertises. Enforced while
# reading rather than after, so a 2 GB file is rejected at ~50 MB instead of
# being buffered in full and taking the container's memory with it.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
_CHUNK = 1024 * 1024

ALLOWED_EXTENSIONS = ("csv", "xlsx", "xls", "json")


def _read_capped(upload: UploadFile) -> bytes:
    """Read the upload, refusing anything over the size cap."""
    chunks, total = [], 0
    while True:
        chunk = upload.file.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"That file is larger than the {MAX_UPLOAD_MB} MB limit. "
                       f"Try a sample of the rows, or split it into smaller files.",
            )
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="That file is empty.")
    return b"".join(chunks)

# These handlers are deliberately sync `def`, not `async def`. Every one of them
# does blocking CPU work (pandas parsing, describe, correlations). An `async def`
# handler runs directly on the event loop, so that work would stall every other
# request — including the health check — until it finished. Declared sync,
# FastAPI runs them in its threadpool and the server stays responsive.


@router.post("/upload")
def upload_file(file: UploadFile = File(...)):
    filename = file.filename or "dataset"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}' is not a supported format. "
                   f"Upload a CSV, XLSX, XLS or JSON file.",
        )

    content = _read_capped(file)

    try:
        df = load_data(content, filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read '{filename}': {e}. Check that it is a valid "
                   f"{ext.upper()} file and that the first row holds column names.",
        )

    if df is None or df.empty:
        raise HTTPException(
            status_code=400, detail="That file parsed successfully but contains no rows.")

    try:
        schema = get_schema(df)
        eda_results = perform_eda(df)
    except MemoryError:
        # Reported honestly rather than as a bad request — the file was fine, the
        # box was not. 500 also stops the frontend retrying a doomed upload.
        raise HTTPException(
            status_code=500,
            detail="The server ran out of memory analysing this dataset. "
                   "Try again with fewer rows or columns.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not analyse '{filename}': {e}")

    data_store["current_df"] = df
    data_store["schema"] = schema
    data_store["eda"] = eda_results
    data_store["filename"] = filename

    return {
        "filename": filename,
        "schema": schema,
        "eda": eda_results,
        "preview": df.head(10).fillna("").to_dict(orient="records"),
    }


@router.get("/insights")
def get_data_insights():
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    return generate_insights(data_store["current_df"])


@router.get("/analyze")
def analyze_data():
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    return perform_eda(data_store["current_df"])


@router.get("/clean/missing-report")
def clean_missing_report():
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    return missing_report(data_store["current_df"])


@router.post("/clean/apply")
def clean_apply(request: CleanRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded")
    cleaned, summary = handle_missing(data_store["current_df"], request.strategies)
    if cleaned.shape[1] == 0:
        raise HTTPException(status_code=400, detail="That would drop every column — adjust your choices.")
    schema = get_schema(cleaned)
    eda_results = perform_eda(cleaned)
    data_store["current_df"] = cleaned
    data_store["schema"] = schema
    data_store["eda"] = eda_results
    return {
        "filename": data_store.get("filename", "dataset"),
        "schema": schema,
        "eda": eda_results,
        "preview": cleaned.head(10).fillna("").to_dict(orient="records"),
        "clean_summary": summary,
    }
