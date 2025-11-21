import asyncio
import json
from contextlib import suppress
from typing import Any
from context import logger
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent import build_agent
from nodes import STEP_LABELS

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
    logger.info("_sse_event_stream entered")
    summary_payload = None
    final_state = None
    current_step = None
    
    try:
        # Use astream_events with version v2 for better event handling
        async for event in agent.astream_events(
            {
                "messages": [HumanMessage(content=body.prompt)],
                "chat_session_id": body.chat_session_id,
            },
            version="v2",
        ):
            event_type = event.get("event")
            payload = None

            # --- Node Start Events ---
            if event_type == "on_chain_start":
                name = event.get("name")
                # Filter for actual node names (not the overall graph)
                if name and name in STEP_LABELS:
                    current_step = name
                    payload = {
                        "type": "status",
                        "step": name,
                        "label": STEP_LABELS.get(name, name),
                        "status": "start",
                    }

            # --- Node End Events ---
            elif event_type == "on_chain_end":
                name = event.get("name")
                if name and name in STEP_LABELS:
                    payload = {
                        "type": "status",
                        "step": name,
                        "label": STEP_LABELS.get(name, name),
                        "status": "end",
                    }
                    # Capture output data if this is a node
                    output = event.get("data", {}).get("output")
                    if output and isinstance(output, dict):
                        final_state = output

            # --- LLM Token Streaming ---
            elif event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                text = serialize_message_content(chunk)
                if text and current_step:
                    payload = {
                        "type": "chunk",
                        "step": current_step,
                        "label": STEP_LABELS.get(current_step, current_step),
                        "delta": text,
                    }

            # --- Send event immediately ---
            if payload:
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0)  # Force context switch for flushing

        # --- Handle Final Result ---
        # If we didn't capture final_state from events, invoke once to get it
        if not final_state:
            logger.info("No final_state captured, invoking agent")
            final_state = await agent.ainvoke(
                {
                    "messages": [HumanMessage(content=body.prompt)],
                    "chat_session_id": body.chat_session_id,
                }
            )

        if final_state:
            logger.info("Preparing summary payload")
            session_id = final_state.get("chat_session_id") or body.chat_session_id
            messages_list = final_state.get("messages", [])
            
            # Get final assistant message as plain text
            assistant_message = ""
            if messages_list:
                assistant_message = serialize_message_content(messages_list[-1])

            summary_payload = {
                "type": "response",
                "chat_session_id": session_id,
                "assistant_message": assistant_message,
                "document": final_state.get("document"),
                "diagram": final_state.get("diagram"),
                "messages": [
                    serialize_message_content(msg)
                    for msg in messages_list
                ],
            }

    except Exception as exc:
        logger.error(f"Error in stream: {exc}")
        error_payload = {"type": "error", "message": str(exc)}
        yield f"data: {json.dumps(error_payload)}\n\n"
    
    finally:
        if summary_payload:
            logger.info("Sending summary payload")
            yield f"data: {json.dumps(summary_payload)}\n\n"
        yield "data: [DONE]\n\n"


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
