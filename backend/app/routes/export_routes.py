from fastapi import APIRouter, HTTPException, Response
from app.store import data_store
from core.insights import generate_insights
from core.exporter import generate_pdf_report

router = APIRouter()


@router.get("/export")
async def export_report():
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded. Upload a file first.")
    try:
        df = data_store["current_df"]
        schema = data_store.get("schema", [])
        eda = data_store.get("eda", {})
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
