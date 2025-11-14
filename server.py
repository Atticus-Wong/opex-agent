from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent import build_agent

app = FastAPI()
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
