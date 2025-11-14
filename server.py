from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from context import checkpointer, logger
import traceback
from agent import build_agent

from concurrent.futures import ThreadPoolExecutor
import asyncio

# At the top of your file
executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI()
agent = build_agent(checkpointer)


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
        config = {"configurable": {"thread_id": body.chat_session_id}}
        result = await agent.invoke(
            {
                "messages": [HumanMessage(content=body.prompt)],
                "chat_session_id": body.chat_session_id,
            },
            config
        )
        
        # Add logging to see what result looks like
        logger.info(f"Result: {result}")
        
        session_id: str = result.get("chat_session_id") or body.chat_session_id
        
        # Serialize messages - this might be failing
        try:
            serialized_messages = [serialize_message_content(msg) for msg in result["messages"]]
        except Exception as msg_exc:
            logger.error(f"Error serializing messages: {msg_exc}")
            logger.error(traceback.format_exc())
            raise
        
        return RunResponse(
            chat_session_id=session_id,
            document=result.get("document"),
            diagram=result.get("diagram"),
            messages=serialized_messages,
        )
    except Exception as exc:
        logger.error(f"Error in run_workflow: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {str(exc) or 'Unknown error'}")
