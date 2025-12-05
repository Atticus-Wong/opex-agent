import os
from supabase import create_client, Client
from typing import Optional, Tuple

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def test_supabase_connection() -> None:
    """
    Very basic test: tries to query 1 row from the 'messages' table.
    Adjust the table name if yours is different.
    """
    try:
        resp = supabase.table("messages").select("*").limit(1).execute()
        print("✅ Supabase connection OK")
        print("Data returned:", resp.data)
    except Exception as e:
        print("❌ Supabase connection FAILED")
        print("Error:", e)

def get_latest_diagram_and_document(chat_session_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (diagram, document) for the most recent *assistant* message
    in this chat_session_id, or (None, None) if nothing found.

    Assumes metadata looks like:
      {
        "diagram": "<string>",
        "document": "<string>"
      }
    """
    resp = (
        supabase.table("messages")
        .select("metadata, created_at, sender_type")
        .eq("chat_session_id", chat_session_id)
        .eq("sender_type", "assistant")   # 🔴 filter only assistant messages
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    rows = resp.data or []
    if not rows:
        return None, None

    metadata = rows[0].get("metadata") or {}
    diagram = metadata.get("diagram")
    document = metadata.get("document")
    return diagram, document
