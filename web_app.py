from __future__ import annotations

from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dialogue_service import DialogueWebService


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web_static"


class SessionRequest(BaseModel):
    npc_id: str
    player_id: str = "demo_player"
    mode: Literal["mock", "live"] = "mock"


class ChatRequest(BaseModel):
    session_id: str
    npc_id: str
    player_id: str = "demo_player"
    message: str
    mode: Literal["mock", "live"] = "mock"


def create_app(service: DialogueWebService | None = None) -> FastAPI:
    app = FastAPI(title="GameNPCDialogue Web Demo")
    dialogue_service = service or DialogueWebService()

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="web_static/index.html not found")
        return FileResponse(index_path)

    @app.get("/api/npcs")
    def get_npcs():
        return {"npcs": dialogue_service.list_npcs()}

    @app.post("/api/sessions")
    def create_session(request: SessionRequest):
        try:
            return dialogue_service.create_session(
                npc_id=request.npc_id,
                player_id=request.player_id,
                mode=request.mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        try:
            return dialogue_service.chat(
                session_id=request.session_id,
                npc_id=request.npc_id,
                player_id=request.player_id,
                message=request.message,
                mode=request.mode,
            )
        except ValueError as exc:
            message = str(exc)
            status_code = 400 if "Message cannot be empty" in message else 404
            raise HTTPException(status_code=status_code, detail=message) from exc

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=False)
