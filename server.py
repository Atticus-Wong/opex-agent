import asyncio
import json
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent import build_agent

app = FastAPI()

# Allow your Next.js dev server to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],   # or ["POST"] if you want to be strict
    allow_headers=["*"],
)

agent = build_agent()  # compile graph once per



def serialize_message_content(message) -> str:
    """
    Convert LangChain message content to a plain string so it fits the response model.
    """
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict):
                if chunk.get("type") == "text":
                    parts.append(chunk.get("text", ""))
                else:
                    parts.append(str(chunk))
            else:
                parts.append(str(chunk))

        joined = "\n".join(filter(None, parts))
        return joined if joined else str(content)

    return str(content)


class RunRequest(BaseModel):
    chat_session_id: str
    prompt: str


class RunResponse(BaseModel):
    chat_session_id: str
    document: str | None
    diagram: str | None
    messages: list[str]


@app.post("/run", response_model=RunResponse)
async def run_workflow(body: RunRequest):
    try:
        result = await agent.ainvoke(
            {
                "messages": [HumanMessage(content=body.prompt)],
                "chat_session_id": body.chat_session_id,
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    session_id: str = result.get("chat_session_id") or body.chat_session_id

    return RunResponse(
        chat_session_id=session_id,
        document=result.get("document"),
        diagram=result.get("diagram"),
        messages=[serialize_message_content(msg) for msg in result["messages"]],
    )

async def _sse_event_stream(body: RunRequest):
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def handler(payload: dict[str, Any]):
        await queue.put(f"data: {json.dumps(payload)}\n\n")

    async def invoke_agent():
        try:
            result = await agent.ainvoke(
                {
                    "messages": [HumanMessage(content=body.prompt)],
                    "chat_session_id": body.chat_session_id,
                    "stream_handler": handler,
                }
            )

            session_id: str = result.get("chat_session_id") or body.chat_session_id
            summary = {
                "type": "result",
                "chat_session_id": session_id,
                "document": result.get("document"),
                "diagram": result.get("diagram"),
                "messages": [
                    serialize_message_content(msg) for msg in result["messages"]
                ],
            }
            await queue.put(f"data: {json.dumps(summary)}\n\n")
        except Exception as exc:  # noqa: BLE001 - surface agent errors to client
            error_payload = {"type": "error", "message": str(exc)}
            await queue.put(f"data: {json.dumps(error_payload)}\n\n")
        finally:
            await queue.put("data: [DONE]\n\n")
            await queue.put(None)

    task = asyncio.create_task(invoke_agent())

    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@app.post("/chat")
async def chat_stream(body: RunRequest):
    response = StreamingResponse(
        _sse_event_stream(body), 
        media_type="text/event-stream"
    )
    # Crucial headers to disable buffering
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no" # For Nginx
    response.headers["Connection"] = "keep-alive"
    return response
