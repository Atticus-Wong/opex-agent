from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from nodes import (
    intentParserNode,
    generateDocumentNode,
    generateProcessDiagramNode,
    validationNode,
    processIterationNode,
    docIterationNode,
    toolNode,
)
from context import Context
from ws_manager import manager
import asyncio

def wrap_node(fn, step_label: str):
    async def wrapped(ctx: Context):
        # get session id from state
        chat_session_id = ctx.get("chat_session_id")

        if chat_session_id:
            # send status over websocket
            await manager.send_status(chat_session_id, step_label)

        # call the original (sync) node
        # it's fine to call a sync function from an async one
        return fn(ctx)

    return wrapped



def build_agent():
    graph = StateGraph(Context)

    graph.add_node("intentParser",
                   wrap_node(intentParserNode, "Gathering user intent"))
    graph.add_node("generateProcessDiagram",
                   wrap_node(generateProcessDiagramNode, "Generating process diagram"))
    graph.add_node("generateDocument",
                   wrap_node(generateDocumentNode, "Drafting process document"))
    graph.add_node("validation",
                   wrap_node(validationNode, "Validating workflow"))
    graph.add_node("processIteration",
                   wrap_node(processIterationNode, "Refining diagram"))
    graph.add_node("docIteration",
                   wrap_node(docIterationNode, "Refining document"))
    graph.add_node("tools",
                   wrap_node(toolNode, "Sending final outputs"))

    graph.add_edge(START, "intentParser")
    graph.add_edge("intentParser", "generateProcessDiagram")
    graph.add_edge("generateProcessDiagram", "generateDocument")
    graph.add_edge("generateDocument", "validation")

    def should_iterate(ctx: Context) -> str:
        return "tools" if ctx["is_satisfied"] else "docIteration"

    graph.add_conditional_edges(
        "validation",
        should_iterate,
        {"tools": "tools", "docIteration": "docIteration"},
    )

    graph.add_edge("docIteration", "processIteration")
    graph.add_edge("processIteration", "validation")
    graph.add_edge("tools", END)

    return graph.compile()

if __name__ == "__main__":
    agent = build_agent()
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "create an onboarding process for new designers "
                        "and developer at Opex AI for me"
                    )
                )
            ]
        }
    )

    final_messages = result["messages"]
    document = result.get("document")
    diagram = result.get("diagram")

    print("Final output:", final_messages[-1].content)
