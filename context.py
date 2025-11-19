import os
import logging
from typing import Any, Awaitable, Callable, Dict

from langgraph.graph import MessagesState
from composio import Composio
from composio_langchain import LangchainProvider
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

composio = Composio(provider=LangchainProvider())
COMPOSIO_USER_ID = os.environ["COMPOSIO_USER_ID"]

tools = composio.tools.get(user_id=COMPOSIO_USER_ID, tools=["GMAIL_FETCH_EMAILS", "GMAIL_SEND_EMAIL"])
# tools = composio.tools.get(user_id=COMPOSIO_USER_ID, toolkits=["NOTION"], tools=["GMAIL_SEND_EMAIL"])

llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")
# llm = ChatOpenAI(model="gpt-5-nano")

logger = logging.getLogger("opex-agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logger.addHandler(handler)
logger.propagate = False

StreamPayload = Dict[str, Any]
StreamHandler = Callable[[StreamPayload], Awaitable[None]]


class Context(MessagesState):
    diagram: str
    document: str
    is_satisfied: bool
    chat_session_id: str
    stream_handler: StreamHandler | None
