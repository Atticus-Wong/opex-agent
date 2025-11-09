import os
import requests
import asyncio
import logging
from typing import Any, Dict, List
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
# ----------
from composio import Composio
from composio.exceptions import ValidationError
from composio_langchain import LangchainProvider


llm = ChatOpenAI(model="gpt-5-nano")
class Context(MessagesState):
    diagram: str = ""
    document: str = ""
    is_satisfied: bool = False
  

# ---------- BUILD WORKFLOW ---------- #
graph = StateGraph(Context)
graph.add_node("intentParser", intentParserNode)
graph.add_node("generateProcessDiagram", generateProcessDiagramNode)
graph.add_node("generateDocument", generateDocumentNode)
graph.add_node("validation", validationNode)
graph.add_node("processIteration", processIterationNode)
graph.add_node("docIteration", docIterationNode)
graph.add_node("tools", toolNode)

graph.add_edge(START, "intentParser")
graph.add_edge("intentParser", "generateProcessDiagram")
graph.add_edge("generateProcessDiagram", "generateDocument")
graph.add_edge("generateDocument", "validation")

def should_iterate(ctx: Context) -> str:
    if ctx["is_satisfied"]:
        return "tools"
    else:
        return "docIteration"

graph.add_conditional_edges(
    "validation",
    should_iterate,
    {
        "tools": "tools",
        "docIteration": "docIteration"
    }
)

graph.add_edge("docIteration", "processIteration")
graph.add_edge("processIteration", "validation")

graph.add_edge("tools", END)

agent = graph.compile()


# Invoke with a dictionary (not Context object)
result = agent.invoke({
    "messages": [HumanMessage(content="create an onboarding process for new designers and developer at Opex AI for me")]
})

# Access results from the dictionary
final_messages = result["messages"]
document = result.get("document")
diagram = result.get("diagram")

print("Final output:", final_messages[-1].content)