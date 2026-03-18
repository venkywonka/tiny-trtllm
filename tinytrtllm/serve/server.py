"""OpenAI-compatible FastAPI server for tiny-trtllm.

Supports:
- /v1/models — list models
- /v1/chat/completions — streaming (SSE) and non-streaming
- /health — health check
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from tinytrtllm.config import SamplingParams
from tinytrtllm.engine.request import LlmRequest
from tinytrtllm.llm import LLM


# ---------------------------------------------------------------------------
# Request/Response models (OpenAI-compatible)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, gt=0, le=4096)
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class StreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "tiny-trtllm"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_app(llm: Optional[LLM] = None, model_name: str = "default") -> FastAPI:
    """Create FastAPI app with an LLM backend."""
    app = FastAPI(title="tiny-trtllm", version="0.1.0")
    app.state.llm = llm
    app.state.model_name = model_name

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return ModelListResponse(
            data=[
                ModelInfo(
                    id=app.state.model_name,
                    created=int(time.time()),
                )
            ]
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if app.state.llm is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Build prompt from messages
        try:
            prompt = _format_messages(request.messages)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        sp = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if request.stream:
            return StreamingResponse(
                _stream_response(
                    app.state.llm, prompt, sp, completion_id, created, request.model
                ),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming
            outputs = await asyncio.get_running_loop().run_in_executor(
                None, lambda: app.state.llm.generate([prompt], sp)
            )
            out = outputs[0]
            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessage(role="assistant", content=out.text),
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=out.prompt_token_count,
                    completion_tokens=len(out.token_ids),
                    total_tokens=out.prompt_token_count + len(out.token_ids),
                ),
            )

    return app


async def _stream_response(
    llm: LLM,
    prompt: str,
    sp: SamplingParams,
    completion_id: str,
    created: int,
    model: str,
) -> AsyncGenerator[str, None]:
    """Stream SSE chunks."""
    # Run generation in thread pool
    loop = asyncio.get_running_loop()
    outputs = await loop.run_in_executor(
        None, lambda: llm.generate([prompt], sp)
    )

    out = outputs[0]
    text = out.text

    # Send role chunk
    chunk = StreamChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
    )
    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    # Send content in small chunks
    chunk_size = max(1, len(text) // 5) if text else 1
    for i in range(0, max(1, len(text)), chunk_size):
        content = text[i : i + chunk_size]
        if not content:
            break
        chunk = StreamChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[StreamChoice(delta=DeltaMessage(content=content))],
        )
        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    # Send finish chunk
    chunk = StreamChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )
    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
    yield "data: [DONE]\n\n"


def _format_messages(messages: list[ChatMessage]) -> str:
    """Format chat messages into a prompt string."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")
        else:
            raise ValueError(f"Unsupported message role: {msg.role}")
    parts.append("Assistant:")
    return "\n".join(parts)


def run_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    **kwargs,
) -> None:
    """Run the server with a model."""
    import uvicorn

    llm = LLM(model_path, **kwargs)
    app = create_app(llm, model_name=model_path)
    uvicorn.run(app, host=host, port=port)
