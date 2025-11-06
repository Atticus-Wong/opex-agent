# main.py
from pathlib import Path
from dotenv import load_dotenv
from nodes import Agent, InputNode, NotionSinkNode

# 1) load env from .env.local at project root
load_dotenv(Path(__file__).resolve().parent / ".env.local")

def run_once(prompt: str):
    pipeline = [InputNode(prompt), NotionSinkNode()]
    agent = Agent(pipeline)
    result = agent.run(prompt)
    print("\n=== Agent Result ===")
    print("status:", result["status"])
    print("answer:", result["answer"])
    print("artifacts:", {k: v for k, v in result["artifacts"].items() if k != "secrets"})
    print("\nSteps:")
    for step in result["steps"]:
        print(" -", step)

if __name__ == "__main__":
    run_once("log this run to notion (connectivity test)")
