# ws_manager.py
from typing import Dict, List
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, List[WebSocket]] = {}

    async def connect(self, chat_session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active.setdefault(chat_session_id, []).append(websocket)

    def disconnect(self, chat_session_id: str, websocket: WebSocket):
        conns = self.active.get(chat_session_id)
        if not conns:
            return
        if websocket in conns:
            conns.remove(websocket)
        if not conns:
            self.active.pop(chat_session_id, None)

    async def send_status(self, chat_session_id: str, step: str):
        conns = self.active.get(chat_session_id, [])
        if not conns:
            return
        message = {"step": step}
        still_alive: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_json(message)
                still_alive.append(ws)
            except Exception:
                # ignore dead sockets
                pass
        if still_alive:
            self.active[chat_session_id] = still_alive
        else:
            self.active.pop(chat_session_id, None)

manager = ConnectionManager()
