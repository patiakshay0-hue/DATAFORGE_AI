from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.store import data_store
from core.chat import stream_chat_response, check_api_key
from app.models.schemas import ChatRequest

router = APIRouter()


@router.get("/chat/status")
async def chat_status():
    return check_api_key()


@router.post("/chat")
async def chat(request: ChatRequest):
    if "current_df" not in data_store:
        raise HTTPException(status_code=404, detail="No data uploaded. Upload a file first.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    df = data_store["current_df"]
    schema = data_store.get("schema", [])
    eda = data_store.get("eda", {})
    filename = data_store.get("filename", "dataset")

    return StreamingResponse(
        stream_chat_response(
            request.question, request.history or [],
            df, schema, eda, filename,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
