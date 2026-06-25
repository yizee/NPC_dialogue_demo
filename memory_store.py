from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re


MAX_RECENT_MESSAGES = 12
COMPACT_EVERY_N_TURNS = 8


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def create_session_id(npc_id: str, player_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{_safe_id(npc_id)}_{_safe_id(player_id)}"


@dataclass
class DialogueMemorySession:
    npc_id: str
    player_id: str
    log_dir: str | Path = "logs/sessions"
    max_recent_messages: int = MAX_RECENT_MESSAGES
    compact_every_n_turns: int = COMPACT_EVERY_N_TURNS
    session_id: str | None = None
    recent_conversation_history: list[dict] = field(default_factory=list)
    session_summary: str = ""
    full_conversation_history: list[dict] = field(default_factory=list)
    story_events: list[dict] = field(default_factory=list)
    quest_state: dict = field(default_factory=lambda: {
        "active_quest": None,
        "completed_quests": [],
        "failed_quests": [],
        "known_clues": [],
        "pending_commitments": [],
    })
    turn_count: int = 0
    compact_pending: bool = False

    def __post_init__(self) -> None:
        if self.session_id is None:
            self.session_id = create_session_id(self.npc_id, self.player_id)
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.raw_log_path = self.log_dir / f"{self.session_id}.jsonl"

    def append_turn(
        self,
        user_input: str,
        assistant_reply: str,
        intent: str,
        used_rag: bool = False,
        used_tool_calling: bool = False,
        rag_sources: list[str] | None = None,
        tool_names: list[str] | None = None,
    ) -> dict:
        self.turn_count += 1
        raw_turn = {
            "session_id": self.session_id,
            "turn_index": self.turn_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "npc_id": self.npc_id,
            "player_id": self.player_id,
            "user_input": user_input,
            "assistant_reply": assistant_reply,
            "intent": intent,
            "used_rag": used_rag,
            "used_tool_calling": used_tool_calling,
            "rag_sources": rag_sources or [],
            "tool_names": tool_names or [],
        }
        with self.raw_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(raw_turn, ensure_ascii=False) + "\n")
        self.recent_conversation_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_reply},
        ])
        self.full_conversation_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_reply},
        ])
        if len(self.recent_conversation_history) > self.max_recent_messages:
            self.compact_pending = True
        self.trim_recent_history()
        return raw_turn

    def trim_recent_history(self) -> None:
        if len(self.recent_conversation_history) > self.max_recent_messages:
            self.recent_conversation_history = (
                self.recent_conversation_history[-self.max_recent_messages:]
            )

    def should_compact(self, force: bool = False) -> bool:
        if force:
            return True
        if self.compact_pending:
            return True
        if len(self.recent_conversation_history) > self.max_recent_messages:
            return True
        return (
            self.turn_count > 0
            and self.turn_count % self.compact_every_n_turns == 0
        )

    def add_story_event(self, event_type: str, summary: str) -> dict:
        event = {
            "event_type": event_type,
            "summary": summary,
        }
        self.story_events.append(event)
        return event

    def update_quest_state(self, **updates) -> dict:
        for key, value in updates.items():
            if key not in self.quest_state:
                raise KeyError(f"Unknown quest_state key: {key}")
            self.quest_state[key] = value
        return self.quest_state

    def mark_compacted(self) -> None:
        self.compact_pending = False

    def build_compact_input(self) -> dict:
        return {
            "session_id": self.session_id,
            "npc_id": self.npc_id,
            "player_id": self.player_id,
            "session_summary": self.session_summary,
            "full_conversation_history": list(self.full_conversation_history),
            "recent_conversation_history": list(self.recent_conversation_history),
            "story_events": list(self.story_events),
            "quest_state": dict(self.quest_state),
        }
