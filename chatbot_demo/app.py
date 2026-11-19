"""FastAPI entrypoint for the single-screen Guardrailed AI Chatbot demo."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chatbot_demo.guardrails_service import GuardrailedChatbotService

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="VinBank Guardrailed Chatbot Demo", version="1.0.0")
service = GuardrailedChatbotService(max_requests=15, window_seconds=60)


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=4000)
    session_id: str = Field(default="default", min_length=1, max_length=128)


class ResetRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=128)


@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model": service.model,
        "api_configured": bool(service.api_key),
        "guardrails": [
            "rate_limiter",
            "injection_guard",
            "topic_filter",
            "output_guardrail",
        ],
    }


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest) -> dict:
    return await service.chat(payload.message, session_id=payload.session_id)


@app.post("/api/reset")
async def reset_endpoint(payload: ResetRequest) -> dict:
    service.reset_session(payload.session_id)
    return {"status": "reset", "session_id": payload.session_id}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "chatbot_demo.app:app",
        host="127.0.0.1",
        port=port,
        reload=False,
    )
