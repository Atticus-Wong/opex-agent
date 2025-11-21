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
    try:
        final_state = None
        
        # We iterate directly over the agent's stream
        async for event in agent.astream_events(
            {
                "messages": [HumanMessage(content=body.prompt)],
                "chat_session_id": body.chat_session_id,
            },
            stream_mode="messages",
            version="v1",
        ):
            event_type = event.get("event")
            payload = None

            # --- Logic to format your events ---
            if event_type == "on_node_start":
                step = event.get("name")
                if step:
                    payload = {
                        "type": "status",
                        "step": step,
                        "label": STEP_LABELS.get(step, step),
                        "status": "start",
                    }

            elif event_type == "on_node_end":
                step = event.get("name")
                if step:
                    payload = {
                        "type": "status",
                        "step": step,
                        "label": STEP_LABELS.get(step, step),
                        "status": "end",
                    }

            elif event_type == "on_chat_model_stream":
                metadata = event.get("metadata") or {}
                tags = metadata.get("tags") or []
                step = tags[-1] if tags else metadata.get("langgraph_node")
                
                if step:
                    chunk = event.get("data", {}).get("chunk")
                    # Note: Ensure serialize_message_content is available here
                    text = serialize_message_content(chunk)
                    if text:
                        payload = {
                            "type": "chunk",
                            "step": step,
                            "label": STEP_LABELS.get(step, step),
                            "delta": text,
                        }

            elif event_type == "on_graph_end":
                final_state = event.get("data", {}).get("output")

            # --- The Critical Change: Yield Immediately ---
            if payload:
                yield f"data: {json.dumps(payload)}\n\n"
                # Optional: forceful context switch to ensure flush (usually not needed but helps debug)
                # await asyncio.sleep(0) 
        # --- Handle Final Result ---
        if final_state:
            session_id = final_state.get("chat_session_id") or body.chat_session_id
            messages_list = final_state.get("messages", [])
            
            # Get final assistant message as plain text
            assistant_message = ""
            if messages_list:
                assistant_message = serialize_message_content(messages_list[-1])

            summary = {
                "type": "response",  # IMPORTANT CHANGE
                "chat_session_id": session_id,
                "assistant_message": assistant_message,
                "document": final_state.get("document"),  # for DB metadata
                "diagram": final_state.get("diagram"),    # for DB metadata
                "messages": [
                    serialize_message_content(msg)
                    for msg in messages_list
                ],
            }

            # Send final SSE event
            yield f"data: {json.dumps(summary)}\n\n"


    except Exception as exc:
        error_payload = {"type": "error", "message": str(exc)}
        yield f"data: {json.dumps(error_payload)}\n\n"
    
    finally:
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
