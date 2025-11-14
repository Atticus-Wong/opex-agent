import os
import logging
from langgraph.graph import MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from composio import Composio
from composio_langchain import LangchainProvider
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ["DB_URL"]
composio = Composio(provider=LangchainProvider())
COMPOSIO_USER_ID = os.environ["COMPOSIO_USER_ID"]

tools = composio.tools.get(user_id=COMPOSIO_USER_ID, tools=["GMAIL_FETCH_EMAILS", "GMAIL_SEND_EMAIL"])
# tools = composio.tools.get(user_id=COMPOSIO_USER_ID, toolkits=["NOTION"], tools=["GMAIL_SEND_EMAIL"])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
# llm = ChatOpenAI(model="gpt-5-nano")

logger = logging.getLogger("opex-agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logger.addHandler(handler)
logger.propagate = False

class Context(MessagesState):
    diagram: str 
    document: str 
    is_satisfied: bool


def initialize_db():
    pool = ConnectionPool(
        conninfo=DB_URL,
        min_size=1,
        max_size=5,
        kwargs={
            "sslmode": "require",
            "row_factory": dict_row,
            "autocommit": True
        },
    )
    
    checkpointer = PostgresSaver(pool)  # type: ignore
    checkpointer.setup()
    return checkpointer
checkpointer = initialize_db()
