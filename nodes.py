# nodes.py
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from composio_helper import initialize_composio_mcp

from composio import MultiServerMCPClient

@dataclass
class Context:
    prompt: str
    meta: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

class Node:
    name: str = "node"
    def run(self, ctx: Context) -> Context:
        raise NotImplementedError

class InputNode(Node):
    name = "input"
    def __init__(self, prompt: str): self._prompt = prompt
    def run(self, ctx: Context) -> Context:
        ctx.logs.append(f"[{self.name}] received prompt: {self._prompt!r}")
        ctx.artifacts["input_prompt"] = self._prompt
        return ctx

class MCPBootstrapNode(Node):
    name = "mcp_bootstrap"
    def __init__(self, server_alias: str = "notion_server"):
        self.server_alias = server_alias

    async def _async_setup_and_fetch(self):
        client_config = initialize_composio_mcp()
        mcp_client = MultiServerMCPClient({self.server_alias: client_config})
        tools = await mcp_client.get_tools()
        return mcp_client, tools

    def run(self, ctx: Context) -> Context:
        ctx.logs.append(f"[{self.name}] initializing Composio MCP and fetching Notion tools ...")
        mcp_client, tools = asyncio.run(self._async_setup_and_fetch())
        ctx.artifacts["mcp_client"] = mcp_client
        ctx.artifacts["mcp_tools"] = tools

        llm_tools_schema: List[Dict[str, Any]] = []
        tool_names: List[str] = []
        for t in tools:
            name = t.get("name") or t.get("toolName") or t.get("id") or "unknown_tool"
            desc = t.get("description") or f"MCP tool {name}"
            schema = t.get("inputSchema") or t.get("parameters") or {"type": "object", "properties": {}}
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
            llm_tools_schema.append({
                "type": "function",
                "function": {"name": name, "description": desc, "parameters": schema}
            })
            tool_names.append(name)

        ctx.artifacts["llm_tools_schema"] = llm_tools_schema
        ctx.artifacts["mcp_tool_names"] = tool_names
        ctx.logs.append(f"[{self.name}] tools available: {tool_names}")
        return ctx

class LLMPlannerNode(Node):
    name = "llm_planner"
    def __init__(self, model: Optional[str] = None) -> None:
        load_dotenv()
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing (set it in .env.local or your shell).")

    def run(self, ctx: Context) -> Context:
        from openai import OpenAI
        client = OpenAI()

        tools_schema = ctx.artifacts.get("llm_tools_schema", [])
        if not tools_schema:
            ctx.logs.append(f"[{self.name}] no tools available from MCP; skipping planning")
            ctx.artifacts["tool_call"] = None
            return ctx

        system = (
            "You are an agent planner. You have Notion tools available via MCP. "
            "When the user asks to create or update content in Notion (e.g., create page, append blocks, list databases), "
            "select exactly one tool and provide valid, concise arguments per the tool's JSON schema."
        )

        resp = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": ctx.prompt},
            ],
            tools=tools_schema,
            tool_choice="auto",
            temperature=0.1,
        )

        choice = resp.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)

        if tool_calls:
            call = tool_calls[0]
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except Exception:
                args = {}
            ctx.artifacts["tool_call"] = {"name": name, "args": args}
            ctx.logs.append(f"[{self.name}] model selected tool: {name} with args={args}")
        else:
            ctx.artifacts["tool_call"] = None
            ctx.logs.append(f"[{self.name}] model did not request any tool call")
        return ctx

class MCPExecutorNode(Node):
    name = "mcp_executor"
    async def _async_execute(self, mcp_client: MultiServerMCPClient, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return await mcp_client.call_tool(tool_name, args or {})

    def run(self, ctx: Context) -> Context:
        call = ctx.artifacts.get("tool_call")
        if not call:
            ctx.logs.append(f"[{self.name}] no tool call to execute")
            return ctx

        mcp_client = ctx.artifacts.get("mcp_client")
        if not mcp_client:
            raise RuntimeError("[mcp_executor] MCP client missing in context. Did MCPBootstrapNode run?")

        name = call.get("name")
        args = call.get("args", {}) or {}

        ctx.logs.append(f"[{self.name}] executing MCP tool {name!r} with args={args}")
        result = asyncio.run(self._async_execute(mcp_client, name, args))

        ctx.artifacts["mcp_result"] = result
        url = None
        if isinstance(result, dict):
            url = result.get("url") or result.get("public_url") or result.get("htmlLink") or result.get("link")
        ctx.artifacts["mcp_result_url"] = url
        ctx.logs.append(f"[{self.name}] tool result: {'(no url)' if not url else url}")
        return ctx

class Agent:
    def __init__(self, nodes: List[Node]): self.nodes = nodes
    def run(self, prompt: str) -> Dict[str, Any]:
        ctx = Context(prompt=prompt)
        for node in self.nodes:
            ctx = node.run(ctx)
        return {
            "status": "ok",
            "answer": self._compose_answer(ctx),
            "steps": ctx.logs,
            "artifacts": ctx.artifacts,
        }

    def _compose_answer(self, ctx: Context) -> str:
        if ctx.artifacts.get("mcp_result_url"):
            return f"Done via MCP (Notion). Result: {ctx.artifacts['mcp_result_url']}"
        if ctx.artifacts.get("mcp_result"):
            return "Done via MCP (Notion). Result payload captured."
        return f"Received: {ctx.artifacts.get('input_prompt')!r} (no tool run)"
