# nodes.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os, requests

# ---------- shared context ----------
@dataclass
class Context:
    prompt: str
    meta: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

# ---------- base node ----------
class Node:
    name: str = "node"
    def run(self, ctx: Context) -> Context:
        raise NotImplementedError

# ---------- input node ----------
class InputNode(Node):
    name = "input"
    def __init__(self, prompt: str): self._prompt = prompt
    def run(self, ctx: Context) -> Context:
        ctx.logs.append(f"[{self.name}] received prompt: {self._prompt!r}")
        ctx.artifacts["input_prompt"] = self._prompt
        return ctx

# ---------- minimal Notion client ----------
class NotionClient:
    def __init__(self, token: Optional[str] = None):
        token = token or os.getenv("NOTION_TOKEN")
        if not token:
            raise RuntimeError("NOTION_TOKEN is missing (set it in .env.local or your shell).")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        })
    def whoami(self) -> Dict[str, Any]:
        r = self._session.get("https://api.notion.com/v1/users/me", timeout=20)
        r.raise_for_status()
        return r.json()

# ---------- notion sink node ----------
class NotionSinkNode(Node):
    name = "notion_sink"
    def __init__(self, client: Optional[NotionClient] = None):
        self.client = client or NotionClient()
    def run(self, ctx: Context) -> Context:
        ctx.logs.append(f"[{self.name}] checking Notion token via /v1/users/me ...")
        data = self.client.whoami()
        user_name = data.get("name") or data.get("bot", {}).get("owner", {}).get("workspace_name")
        ctx.artifacts["notion_user"] = user_name
        ctx.logs.append(f"[{self.name}] Notion token OK. User/Workspace: {user_name!r}")
        return ctx

# ---------- agent orchestrator ----------
class Agent:
    def __init__(self, nodes: List[Node]): self.nodes = nodes
    def run(self, prompt: str) -> Dict[str, Any]:
        ctx = Context(prompt=prompt)
        for node in self.nodes:
            ctx = node.run(ctx)
        return {
            "status": "ok",
            "answer": f"Received prompt: {ctx.artifacts.get('input_prompt')!r}",
            "steps": ctx.logs,
            "artifacts": ctx.artifacts,
        }
