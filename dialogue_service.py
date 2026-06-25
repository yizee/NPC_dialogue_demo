from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from game_tools import (
    check_inventory,
    format_inventory_response,
    get_player_profile,
)
from intent_router import (
    GM_FEEDBACK,
    INVENTORY_QUERY,
    LORE_QUESTION,
    PERSONA_CHAT,
    QUEST_REQUEST,
    classify_intent,
)
from memory_store import DialogueMemorySession
from npc_dialogue import (
    extract_inventory_item_name,
    generate_gm_feedback_safely,
    generate_quest,
    load_all_npc_configs,
    load_npc_config,
    run_claude_tool_turn_with_metadata,
    should_use_rag_for_intent,
)
from prompts import build_system_prompt
from rag_pipeline import RagService


VALID_MODES = {"mock", "live"}


class DialogueWebService:
    def __init__(self, log_dir: str | Path = "logs/web_sessions") -> None:
        self.log_dir = Path(log_dir)
        self.sessions: dict[str, DialogueMemorySession] = {}
        self.rag_service: RagService | None = None
        self.rag_disabled = False

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
        tool_names: list[str] = []
        rag_sources: list[str] = []

        if mode == "live":
            try:
                live_result = self._live_reply(
                    npc_id=npc_id,
                    npc=npc,
                    player_id=player_id,
                    session=session,
                    intent=route.intent,
                    message=message,
                )
                reply = live_result["reply"]
                rag_used = live_result["rag_used"]
                tool_used = live_result["tool_used"]
                rag_sources = live_result["rag_sources"]
                tool_names = live_result["tool_names"]
            except Exception as exc:
                mode = "mock"
                fallback = f"Live mode failed: {exc}"
                reply = self._mock_reply(npc=npc, intent=route.intent, message=message)
                rag_used = route.intent == LORE_QUESTION
                tool_used = route.intent == INVENTORY_QUERY
        else:
            reply = self._mock_reply(npc=npc, intent=route.intent, message=message)
            rag_used = route.intent == LORE_QUESTION
            tool_used = route.intent == INVENTORY_QUERY
            rag_sources = ["mock_lore_context"] if rag_used else []
            tool_names = ["mock_inventory_check"] if tool_used else []

        memory_recorded = False
        try:
            session.append_turn(
                user_input=message,
                assistant_reply=reply,
                intent=route.intent,
                used_rag=rag_used,
                used_tool_calling=tool_used,
                rag_sources=rag_sources,
                tool_names=tool_names,
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

    def _player_context(self, player_id: str) -> dict[str, Any]:
        player_context = get_player_profile(player_id)
        player_context.setdefault("quest_status", "刚刚抵达小镇，尚未接取任务")
        player_context.setdefault("reputation", "中立")
        return player_context

    def _live_reply(
        self,
        *,
        npc_id: str,
        npc: dict[str, Any],
        player_id: str,
        session: DialogueMemorySession,
        intent: str,
        message: str,
    ) -> dict[str, Any]:
        player_context = self._player_context(player_id)

        if intent == QUEST_REQUEST:
            reply = generate_quest(
                npc,
                player_context,
                list(session.recent_conversation_history),
            )
            return self._live_result(reply=reply)

        if intent == INVENTORY_QUERY:
            try:
                tool_turn = run_claude_tool_turn_with_metadata(
                    npc,
                    player_context,
                    message,
                )
                return self._live_result(
                    reply=tool_turn["reply"],
                    tool_used=True,
                    tool_names=tool_turn.get("tool_names", []),
                )
            except Exception:
                item_name = extract_inventory_item_name(message)
                inventory_result = check_inventory(player_context, item_name)
                return self._live_result(
                    reply=format_inventory_response(inventory_result),
                    tool_used=True,
                    tool_names=["check_inventory"],
                )

        if intent == GM_FEEDBACK:
            if npc_id == "gm_advisor":
                feedback_history = list(session.full_conversation_history)
                feedback_history.append({"role": "user", "content": message})
                reply = generate_gm_feedback_safely(
                    npc,
                    player_context,
                    feedback_history,
                )
            else:
                reply = (
                    "[GM Workflow] 已记录你的反馈。"
                    "如果你想做完整复盘，可以退出后选择 gm_advisor 找档案官赛琳。"
                )
            return self._live_result(reply=reply)

        retrieved_context = ""
        rag_sources: list[str] = []
        if should_use_rag_for_intent(intent):
            retrieval = self._retrieve_context(message, npc_id)
            retrieved_context = retrieval["context"]
            rag_sources = retrieval["sources"]

        system_prompt = build_system_prompt(
            npc,
            player_context,
            retrieved_context=retrieved_context,
            memory_summary=session.session_summary,
            quest_state=session.quest_state,
        )
        live_history = list(session.recent_conversation_history)
        live_history.append({"role": "user", "content": message})
        reply = self._create_claude_message(
            system_prompt=system_prompt,
            messages=live_history,
            max_tokens=300,
        )
        return self._live_result(
            reply=reply,
            rag_used=bool(retrieved_context),
            rag_sources=rag_sources,
        )

    def _retrieve_context(self, query: str, npc_id: str) -> dict[str, Any]:
        if self.rag_disabled:
            return {"context": "", "sources": []}
        if self.rag_service is None:
            try:
                self.rag_service = RagService()
            except Exception:
                self.rag_disabled = True
                return {"context": "", "sources": []}
        try:
            retrieval = self.rag_service.retrieve(query, npc_id=npc_id)
        except Exception:
            self.rag_disabled = True
            return {"context": "", "sources": []}
        return {
            "context": retrieval.context,
            "sources": [match.source for match in retrieval.matches],
        }

    def _create_claude_message(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int = 300,
    ) -> str:
        live_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = live_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    def _live_result(
        self,
        *,
        reply: str,
        rag_used: bool = False,
        tool_used: bool = False,
        rag_sources: list[str] | None = None,
        tool_names: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "reply": reply,
            "rag_used": rag_used,
            "tool_used": tool_used,
            "rag_sources": rag_sources or [],
            "tool_names": tool_names or [],
        }

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
