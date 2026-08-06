from fastapi import APIRouter, UploadFile, File, HTTPException
from app.store import data_store
from core.loader import load_data, get_schema
from core.analyzer import perform_eda
from core.insights import generate_insights
from core.cleaner import missing_report, handle_missing
from app.models.schemas import CleanRequest

router = APIRouter()

# These handlers are deliberately sync `def`, not `async def`. Every one of them
# does blocking CPU work (pandas parsing, describe, correlations). An `async def`
# handler runs directly on the event loop, so that work would stall every other
# request — including the health check — until it finished. Declared sync,
# FastAPI runs them in its threadpool and the server stays responsive.


@router.post("/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        content = file.file.read()
        df = load_data(content, file.filename)
        schema = get_schema(df)
        eda_results = perform_eda(df)

        data_store["current_df"] = df
        data_store["schema"] = schema
        data_store["eda"] = eda_results
        data_store["filename"] = file.filename

        return {
            "filename": file.filename,
            "schema": schema,
            "eda": eda_results,
            "preview": df.head(10).fillna("").to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
