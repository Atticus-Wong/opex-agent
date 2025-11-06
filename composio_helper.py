from typing import Any, Dict
import logging
import os

from composio import Composio
from composio.exceptions import ValidationError
from composio_client.types.tool_router_create_session_params import ConfigToolkit

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- COMPOSIO HELPER FUNCTIONS ----------
def _extract_client_config(server: Any) -> Dict[str, Any]:
    """
    Normalise the server response to the config shape expected by MultiServerMCPClient.
    """
    # Collect possible dictionary representations from the server object.
    candidate_maps = []
    transport_fallback = "streamable_http"
    if hasattr(server, "mcp_url"):
        return {"url": getattr(server, "mcp_url"), "transport": transport_fallback}
    if hasattr(server, "client_config") and getattr(server, "client_config"):
        return getattr(server, "client_config")
    if hasattr(server, "model_dump"):
        candidate_maps.append(server.model_dump())
    if hasattr(server, "dict"):
        candidate_maps.append(server.dict())
    if isinstance(server, dict):
        candidate_maps.append(server)
    if hasattr(server, "__dict__"):
        candidate_maps.append({k: getattr(server, k) for k in vars(server) if not k.startswith("_")})

    for data in candidate_maps:
        if not isinstance(data, dict):
            continue
        for key in ("client_config", "clientConfig"):
            value = data.get(key)
            if isinstance(value, dict):
                return value
        # Some responses already match the expected schema (url + transport).
        if {"url", "transport"} <= data.keys():
            return data
        if "mcp_url" in data:
            transport = data.get("type") or transport_fallback
            return {"url": data["mcp_url"], "transport": transport}

    raise ValueError("Could not determine MCP client configuration from server response.")

def _find_existing_server(composio_client: Composio, server_name: str):
    """
    Look up an existing MCP server with the given name.
    """
    response = composio_client.mcp.list(name=server_name)
    items = response.get("items", []) if isinstance(response, dict) else getattr(response, "items", [])
    for entry in items:
        entry_name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
        if entry_name == server_name:
            return entry
    return None

def initialize_composio_mcp():
    logger.info("Preparing Composio MCP client setup")
    composio_client = Composio(api_key=os.environ["COMPOSIO_API_KEY"])

    server_name = "mcp-config-73840"
    logger.debug("Using Composio server name %s", server_name)

    created_new = True
    try:
        logger.info("Creating MCP server %s", server_name)
        server = composio_client.mcp.create(
            name=server_name,
            toolkits=[
                ConfigToolkit(**{
                    "toolkit": "googlecalendar",
                    "auth_config": "ac_FKnlhoa1rCHO",
                })
            ]
        )
        logger.info("Successfully created MCP server %s", server_name)
    except ValidationError as exc:
        cause = getattr(exc, "__cause__", None)
        cause_text = str(cause) if cause else str(exc)
        if "already exists" not in cause_text.lower():
            raise
        logger.info("Server %s already exists; attempting to reuse configuration", server_name)
        created_new = False
        server = _find_existing_server(composio_client, server_name)
        if server is None:
            logger.error("Server %s already exists but could not be retrieved", server_name)
            raise RuntimeError(
                f"MCP server named {server_name!r} already exists, but could not retrieve it."
            ) from exc

    server_id = getattr(server, "id", None)
    if server_id is None and isinstance(server, dict):
        server_id = server.get("id")
    if server_id:
        status = "created" if created_new else "reused existing"
        logger.info("Server ready (%s): %s", status, server_id)
    else:
        logger.warning("Server ready but id unavailable in response")

    client_config = _extract_client_config(server)
    if "url" not in client_config or "transport" not in client_config:
        raise ValueError(f"Client config missing url/transport keys: {client_config}")
    logger.debug(
        "Resolved client config for %s: url=%s, transport=%s",
        server_name,
        client_config.get("url"),
        client_config.get("transport"),
    )
    return client_config

    # mcp_client = MultiServerMCPClient({"google_calendar": client_config})
    # mcp_tools = asyncio.run(mcp_client.get_tools())
