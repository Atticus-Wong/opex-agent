from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent import build_agent

origins = [
    "http://localhost:3000",  # Example: your React app running on localhost
    "https://yourfrontenddomain.com", # Example: your deployed frontend
    # Add more origins as needed
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True, # Allow cookies, authorization headers, etc.
    allow_methods=["*"],    # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],    # Allow all headers

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
