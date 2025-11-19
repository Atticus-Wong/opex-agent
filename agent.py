import inspect

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
    STEP_LABELS,
)
from context import Context

def wrap_node(fn, step_key: str):
    async def wrapped(ctx: Context):
        handler = ctx.get("stream_handler")
        label = STEP_LABELS.get(step_key, step_key)

        if handler:
            await handler({
                "type": "status",
                "step": step_key,
                "label": label,
                "status": "start",
            })

        result = fn(ctx)
        if inspect.isawaitable(result):
            result = await result

        if handler:
            await handler({
                "type": "status",
                "step": step_key,
                "label": label,
                "status": "end",
            })

        return result

    return wrapped



def build_agent():
    graph = StateGraph(Context)

    graph.add_node("intentParser", wrap_node(intentParserNode, "intentParser"))
    graph.add_node(
        "generateProcessDiagram",
        wrap_node(generateProcessDiagramNode, "generateProcessDiagram"),
    )
    graph.add_node(
        "generateDocument", wrap_node(generateDocumentNode, "generateDocument")
    )
    graph.add_node("validation", wrap_node(validationNode, "validation"))
    graph.add_node(
        "processIteration", wrap_node(processIterationNode, "processIteration")
    )
    graph.add_node("docIteration", wrap_node(docIterationNode, "docIteration"))
    graph.add_node("tools", wrap_node(toolNode, "tools"))

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
