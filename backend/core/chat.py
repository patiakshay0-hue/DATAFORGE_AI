import os
import json
import pandas as pd
import numpy as np
from typing import AsyncGenerator

import anthropic

MODEL = "claude-haiku-4-5-20251001"

def _build_data_context(df: pd.DataFrame, schema: list, eda: dict, filename: str) -> str:
    """Build a rich text context block about the uploaded dataset."""
    lines = [
        f"DATASET: {filename}",
        f"ROWS: {len(df)}  |  COLUMNS: {len(df.columns)}",
        "",
        "=== SCHEMA ===",
    ]
    for col in schema:
        lines.append(
            f"  {col['name']} [{col['type']}]  unique={col['unique']}  missing={col['missing']}"
        )

    # Summary statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        lines += ["", "=== NUMERIC SUMMARY ==="]
        desc = df[numeric_cols].describe().round(3)
        lines.append(desc.to_string())

    # Categorical value counts (top 5 per column)
    cat_cols = df.select_dtypes(
        exclude=[np.number, "datetime", "datetimetz", "timedelta"]
    ).columns
    if len(cat_cols):
        lines += ["", "=== CATEGORICAL COLUMNS ==="]
        for col in cat_cols[:5]:
            top = df[col].value_counts().head(5).to_dict()
            lines.append(f"  {col}: {top}")

    # Correlation for numeric
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().round(3)
        lines += ["", "=== CORRELATION MATRIX ===", corr.to_string()]

    # Sample rows
    lines += ["", "=== FIRST 10 ROWS (CSV) ==="]
    lines.append(df.head(10).to_csv(index=False))

    return "\n".join(lines)


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to backend/.env as: ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=api_key)


async def stream_chat_response(
    question: str,
    history: list,
    df: pd.DataFrame,
    schema: list,
    eda: dict,
    filename: str,
) -> AsyncGenerator[str, None]:
    """
    Yields Server-Sent Event strings.
    Each chunk: 'data: <text>\n\n'
    Final chunk: 'data: [DONE]\n\n'
    """
    try:
        client = _get_client()
    except ValueError as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
        return

    data_context = _build_data_context(df, schema, eda, filename)

    system_prompt = (
        "You are DataForge AI, an expert data analyst assistant. "
        "The user has uploaded a dataset and you have full access to its contents below. "
        "Answer questions about the data accurately. "
        "When referencing specific values, cite the actual numbers from the dataset. "
        "Be concise but thorough. Use markdown formatting (bold, bullets, tables) when helpful.\n\n"
        "--- DATASET CONTEXT ---\n"
        + data_context
    )

    # Convert history to Anthropic messages format
    messages = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
        ) as stream:
            for text_chunk in stream.text_stream:
                yield f"data: {json.dumps({'text': text_chunk})}\n\n"

    except anthropic.AuthenticationError:
        yield f"data: {json.dumps({'error': 'Invalid API key. Check your ANTHROPIC_API_KEY in backend/.env'})}\n\n"
    except anthropic.RateLimitError:
        yield f"data: {json.dumps({'error': 'Rate limit reached. Please wait a moment and try again.'})}\n\n"
    except Exception as e:
        # str(e) on an API error is the whole response dict — status code, nested
        # error object, request id — rendered straight into the chat bubble. The
        # one case a user can actually act on is an empty account, so name it.
        text = str(e)
        if "credit balance is too low" in text:
            text = ("This Anthropic account has no credit left, so Chat with Data "
                    "cannot answer. Add credit at console.anthropic.com — every "
                    "other tab works without it.")
        yield f"data: {json.dumps({'error': text})}\n\n"

    yield "data: [DONE]\n\n"


def check_api_key() -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return {"configured": False, "message": "ANTHROPIC_API_KEY not set in backend/.env"}
    if not key.startswith("sk-ant-"):
        return {"configured": False, "message": "API key format looks incorrect"}
    return {"configured": True, "message": "API key is configured"}
