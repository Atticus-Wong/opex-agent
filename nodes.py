"""
nodes.py
A tiny, beginner-friendly "node graph" + Notion connectivity check.

Run:
  python nodes.py
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

# 1) load env from .env.local (keeps secrets out of code)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env.local")
except Exception:
    # It's okay if dotenv isn't installed; we also allow your shell env
    pass

import requests


# ---------------------------
# Shared context passed between nodes
# ---------------------------
@dataclass
class Context:
    prompt: str
    meta: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)


# ---------------------------
# Base Node
# ---------------------------
class Node:
    name: str = "node"

    def run(self, ctx: Context) -> Context:
        """Each node does one small thing and returns the updated context."""
        raise NotImplementedError


# ---------------------------
# Input Node (just saves the prompt)
# ---------------------------
class InputNode(Node):
    name = "input"

    def __init__(self, prompt: str):
        self._prompt = prompt

    def run(self, ctx: Context) -> Context:
        ctx.logs.append(f"[{self.name}] received prompt: {self._prompt!r}")
        ctx.artifacts["input_prompt"] = self._prompt
        return ctx


# ---------------------------
# Notion client (very small, only /v1/users/me today)
# ---------------------------
class NotionClient:
    def __init__(self, token: Optional[str] = None):
        token = token or os.getenv("NOTION_TOKEN")
        if not token:
            raise RuntimeError(
                "NOTION_TOKEN is missing. Put it in .env.local or your shell env."
            )
        self._token = token
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": "2022-06-28",  # stable version
            "Content-Type": "application/json",
        })

    def whoami(self) -> Dict[str, Any]:
        """GET /v1/users/me — proves the token is valid."""
        resp = self._session.get("https://api.notion.com/v1/users/me", timeout=20)
        resp.raise_for_status()
        return resp.json()


# ---------------------------
# Notion Sink Node (for now: check token works)
# Later: can write a page or DB row.
# ---------------------------
class NotionSinkNode(Node):
    name = "notion_sink"

    def __init__(self, client: Optional[NotionClient] = None):
        self.client = client or NotionClient()

    def run(self, ctx: Context) -> Context:
        ctx.logs.append(f"[{self.name}] checking Notion token via /v1/users/me ...")
        data = self.client.whoami()
        # Save a tiny, non-sensitive piece of info into context
        user_name = data.get("name") or data.get("bot", {}).get("owner", {}).get("workspace_name")
        ctx.artifacts["notion_user"] = user_name
        ctx.logs.append(f"[{self.name}] Notion token OK. User/Workspace: {user_name!r}")
        return ctx


# ---------------------------
# Agent (runs nodes in sequence)
# ---------------------------
class Agent:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    def run(self, prompt: str) -> Dict[str, Any]:
        ctx = Context(prompt=prompt)
        # first node can also be InputNode that records the prompt
        for node in self.nodes:
            ctx = node.run(ctx)
        # shape a well-defined response (simple version)
        return {
            "status": "ok",
            "answer": f"Received prompt: {ctx.artifacts.get('input_prompt')!r}",
            "steps": ctx.logs,
            "artifacts": ctx.artifacts,
        }


# ---------------------------
# Quick demo
# ---------------------------
if __name__ == "__main__":
    prompt = "log this run to notion (for now just test connectivity)"
    pipeline = [
        InputNode(prompt),
        NotionSinkNode(),   # validates NOTION_TOKEN works
    ]
    agent = Agent(pipeline)
    result = agent.run(prompt)

    # Print a safe, structured output (no secrets)
    print("\n=== Agent Result ===")
    print("status:", result["status"])
    print("answer:", result["answer"])
    print("artifacts:", {k: v for k, v in result["artifacts"].items() if k != "secrets"})
    print("\nSteps:")
    for step in result["steps"]:
        print(" -", step)
