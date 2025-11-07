# main.py
from pathlib import Path
from dotenv import load_dotenv
from nodes import Agent, InputNode, MCPBootstrapNode, LLMPlannerNode, MCPExecutorNode

# Load env
load_dotenv(Path(__file__).resolve().parent / ".env.local")

def run_once(prompt: str):
    pipeline = [
        InputNode(prompt),
        MCPBootstrapNode(server_alias="notion_server"),  # uses composio_helper to bring up Notion tools
        LLMPlannerNode(),                                # OpenAI chooses Notion tool + args
        MCPExecutorNode(),                               # Executes via MCP
    ]
    agent = Agent(pipeline)
    result = agent.run(prompt)

    print("\n=== Agent Result ===")
    print("status:", result["status"])
    print("answer:", result["answer"])

    print("\nArtifacts:")
    for k, v in result["artifacts"].items():
        if k == "mcp_client":
            print(f" - {k}: <MCP client>")
        else:
            print(f" - {k}: {v}")

    print("\nSteps:")
    for step in result["steps"]:
        print(" -", step)

if __name__ == "__main__":
    # Example prompt for Notion (the LLM will map to your MCP Notion tool):
    run_once("Create a Notion page titled 'Agent Demo' with a paragraph: This was created by the MCP agent.")
