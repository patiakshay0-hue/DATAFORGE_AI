import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from core import converter
from app.store import data_store
from core.loader import get_schema
from core.analyzer import perform_eda
from core.vision_trainer import (
    load_image_zip, VISION_AVAILABLE as VISION_READY,
)
from app.models.schemas import ConvertChoice

router = APIRouter()


@router.post("/convert/inspect")
async def convert_inspect(file: UploadFile = File(...)):
    content = await file.read()
    return converter.inspect_upload(content, file.filename)


@router.post("/convert/convert")
async def convert_run(request: ConvertChoice):
    result = converter.convert(request.choice)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Conversion failed"))
    return result


@router.post("/convert/image-metadata")
async def convert_image_metadata():
    result = converter.image_metadata()
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Could not build metadata"))
    return result


@router.get("/convert/download")
async def convert_download():
    csv_bytes, name, _ = converter.get_converted()
    if csv_bytes is None:
        raise HTTPException(status_code=404, detail="Nothing converted yet")
    return StreamingResponse(
        io.BytesIO(csv_bytes), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.post("/convert/load")
async def convert_load():
    csv_bytes, name, df = converter.get_converted()
    if df is None:
        raise HTTPException(status_code=400, detail="Convert a file to CSV first")
    schema = get_schema(df)
    eda_results = perform_eda(df)
    data_store["current_df"] = df
    data_store["schema"] = schema
    data_store["eda"] = eda_results
    data_store["filename"] = name
    return {
        "filename": name, "schema": schema, "eda": eda_results,
        "preview": df.head(10).fillna("").to_dict(orient="records"),
    }


@router.post("/convert/send-to-vision")
async def convert_send_to_vision():
    if not VISION_READY:
        raise HTTPException(status_code=400, detail="Image tools need Pillow installed on the backend.")
    content = converter.get_stored_zip()
    if content is None:
        raise HTTPException(status_code=400, detail="Upload an image zip first")
    result = load_image_zip(content)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("note", "Could not load images"))
    return result
