from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from intent_router import (
    GM_FEEDBACK,
    INVENTORY_QUERY,
    LORE_QUESTION,
    PERSONA_CHAT,
    QUEST_REQUEST,
    classify_intent,
)
from memory_store import DialogueMemorySession
from npc_dialogue import load_all_npc_configs, load_npc_config


VALID_MODES = {"mock", "live"}


class DialogueWebService:
    def __init__(self, log_dir: str | Path = "logs/web_sessions") -> None:
        self.log_dir = Path(log_dir)
        self.sessions: dict[str, DialogueMemorySession] = {}

    def list_npcs(self) -> list[dict[str, Any]]:
        configs = load_all_npc_configs()
        return [
            self._public_npc_payload(npc_id, npc)
            for npc_id, npc in configs.items()
        ]

    def create_session(
        self,
        npc_id: str,
        player_id: str = "demo_player",
        mode: str = "mock",
    ) -> dict[str, Any]:
        self._validate_mode(mode)
        npc = self._load_npc(npc_id)
        session = DialogueMemorySession(
            npc_id=npc_id,
            player_id=player_id,
            log_dir=self.log_dir,
        )
        self.sessions[session.session_id] = session
        return {
            "session_id": session.session_id,
            "npc": self._public_npc_payload(npc_id, npc),
            "mode": mode,
        }

    def chat(
        self,
        session_id: str,
        npc_id: str,
        player_id: str = "demo_player",
        message: str = "",
        mode: str = "mock",
    ) -> dict[str, Any]:
        self._validate_mode(mode)
        if not message.strip():
            raise ValueError("Message cannot be empty")
        npc = self._load_npc(npc_id)
        session = self._get_or_create_session(session_id, npc_id, player_id)

        requested_mode = mode
        fallback = None
        if mode == "live" and not os.environ.get("ANTHROPIC_API_KEY"):
            mode = "mock"
            fallback = "ANTHROPIC_API_KEY is not configured"

        route = classify_intent(message)
        reply = self._mock_reply(npc=npc, intent=route.intent, message=message)
        rag_used = route.intent == LORE_QUESTION
        tool_used = route.intent == INVENTORY_QUERY

        memory_recorded = False
        try:
            session.append_turn(
                user_input=message,
                assistant_reply=reply,
                intent=route.intent,
                used_rag=rag_used,
                used_tool_calling=tool_used,
                rag_sources=["mock_lore_context"] if rag_used else [],
                tool_names=["mock_inventory_check"] if tool_used else [],
            )
            memory_recorded = True
        except OSError:
            memory_recorded = False

        return {
            "reply": reply,
            "metadata": {
                "intent": route.intent,
                "intent_reason": route.reason,
                "rag_used": rag_used,
                "tool_used": tool_used,
                "memory_recorded": memory_recorded,
                "session_id": session.session_id,
                "turn_count": session.turn_count,
                "mode": mode,
                "requested_mode": requested_mode,
                "fallback": fallback,
            },
        }

    def _load_npc(self, npc_id: str) -> dict[str, Any]:
        try:
            return load_npc_config(npc_id)
        except ValueError as exc:
            raise ValueError(f"Unknown NPC ID: {npc_id}") from exc

    def _get_or_create_session(
        self,
        session_id: str,
        npc_id: str,
        player_id: str,
    ) -> DialogueMemorySession:
        if session_id in self.sessions:
            return self.sessions[session_id]
        session = DialogueMemorySession(
            npc_id=npc_id,
            player_id=player_id,
            log_dir=self.log_dir,
            session_id=session_id or None,
        )
        self.sessions[session.session_id] = session
        return session

    def _public_npc_payload(self, npc_id: str, npc: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": npc_id,
            "name": npc["name"],
            "title": npc["title"],
            "greeting": npc["greeting"],
            "personality": npc.get("personality", ""),
            "topics": npc.get("topics", []),
        }

    def _validate_mode(self, mode: str) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown mode: {mode}")

    def _mock_reply(self, npc: dict[str, Any], intent: str, message: str) -> str:
        name = npc["name"]
        if intent == INVENTORY_QUERY:
            return (
                f"[Mock] {name}: 我先查了一下你的背包记录。"
                "这个演示分支会标记为 tool_used，真实模式会调用工具层。"
            )
        if intent == QUEST_REQUEST:
            return (
                f"[Mock] {name}: 我可以给你一个小任务："
                "沿着镇北的小路调查异常线索，再回来告诉我发现。"
            )
        if intent == LORE_QUESTION:
            return (
                f"[Mock] {name}: 这个问题会触发 lore/RAG 路径。"
                f"当前 mock 模式先用本地占位知识回答：{message}"
            )
        if intent == GM_FEEDBACK:
            return (
                f"[Mock] {name}: 我会把这条反馈记录为 GM review 输入，"
                "并指出任务目标、触发条件和奖励说明是否清楚。"
            )
        if intent == PERSONA_CHAT:
            return f"[Mock] {name}: {npc['greeting']} 你刚才说的是：{message}"
        return f"[Mock] {name}: 我听到了。"
