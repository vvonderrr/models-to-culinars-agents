"""
HTTP API для мобильного приложения «Ням Шеф».

POST /api/v1/chat    { "message": "...", "history": [{"role": "user"|"assistant", "content": "..."}] }
                  -> { "reply": "..." }
GET  /api/v1/health  -> { "status": "ok", "llm": true|false }

Запуск: uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

load_dotenv()  # подхватывает переменные из файла .env рядом с main.py

from chef_llm_client import (
    ChefLlmClient,
    LlmBadResponseError,
    LlmTimeoutError,
    LlmUnavailableError,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("nyam_chef")


# ---------- Схемы запроса и ответа (валидацию делает Pydantic) ----------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=50)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Сообщение пустое")
        return value


class ChatResponse(BaseModel):
    reply: str


# ---------- Жизненный цикл: один клиент модели на всё приложение ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = ChefLlmClient()
    try:
        yield
    finally:
        await app.state.llm.close()


app = FastAPI(title="Ням Шеф API", lifespan=lifespan)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    llm: ChefLlmClient = request.app.state.llm
    history = [m.model_dump() for m in body.history]

    try:
        reply = await llm.reply(body.message, history=history)
    except LlmUnavailableError as exc:
        logger.error("LLM недоступна: %s", exc)
        raise HTTPException(status_code=503, detail="ИИ-помощник сейчас недоступен, попробуйте позже")
    except LlmTimeoutError as exc:
        logger.error("LLM таймаут: %s", exc)
        raise HTTPException(status_code=504, detail="ИИ-помощник думает слишком долго, попробуйте ещё раз")
    except LlmBadResponseError as exc:
        logger.error("LLM ошибка ответа: %s", exc)
        raise HTTPException(status_code=502, detail="Не удалось получить ответ от ИИ-помощника")

    return ChatResponse(reply=reply)


@app.get("/api/v1/health")
async def health(request: Request) -> dict[str, object]:
    llm: ChefLlmClient = request.app.state.llm
    return {
        "status": "ok",
        "llm": await llm.is_available(),
        "provider": llm.provider_name,
        "model": llm.model,
    }
