from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
# ----------
from nodes import intentParserNode, generateDocumentNode, generateProcessDiagramNode, validationNode, processIterationNode, docIterationNode, toolNode 
from context import Context

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

result = agent.invoke({
    "messages": [HumanMessage(content="create an onboarding process for new designers and developer at Opex AI for me")]
})

# Access results from the dictionary
final_messages = result["messages"]
document = result.get("document")
diagram = result.get("diagram")

print("Final output:", final_messages[-1].content)
