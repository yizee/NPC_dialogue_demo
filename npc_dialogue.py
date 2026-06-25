"""
NPC Dialogue + Quest Generator — 方案 A+B 合体版
新增：/quest 指令触发任务生成，完全融合在对话中
"""

import json
import os
import re

from anthropic import Anthropic

from claude_tools import CLAUDE_TOOLS
from game_tools import (
    check_inventory,
    format_inventory_response,
    get_player_profile,
)
from intent_router import (
    GM_FEEDBACK,
    INVENTORY_QUERY,
    LORE_QUESTION,
    QUEST_REQUEST,
    classify_intent,
)
from memory_store import DialogueMemorySession
from prompts import build_gm_prompt, build_quest_prompt, build_system_prompt
from rag_pipeline import RagService
from tool_executor import execute_tool_call, tool_result_block

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def load_npc_config(npc_id: str) -> dict:
    with open("npc_config.json", "r", encoding="utf-8") as f:
        configs = json.load(f)
    if npc_id not in configs:
        raise ValueError(f"NPC '{npc_id}' 不存在，可选: {list(configs.keys())}")
    return configs[npc_id]


def load_all_npc_configs() -> dict:
    with open("npc_config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def safety_check(player_input: str) -> bool:
    forbidden_keywords = ["hack", "exploit", "bypass", "ignore your instructions"]
    return not any(kw in player_input.lower() for kw in forbidden_keywords)


def should_use_rag_for_intent(intent: str) -> bool:
    return intent == LORE_QUESTION


def should_use_claude_tools(intent: str) -> bool:
    return intent == INVENTORY_QUERY


def is_gm_advisor(npc_id: str) -> bool:
    return npc_id == "gm_advisor"


def record_memory_turn(
    memory_session: DialogueMemorySession,
    user_input: str,
    assistant_reply: str,
    intent: str,
    used_rag: bool = False,
    used_tool_calling: bool = False,
    rag_sources: list[str] | None = None,
    tool_names: list[str] | None = None,
) -> None:
    try:
        memory_session.append_turn(
            user_input=user_input,
            assistant_reply=assistant_reply,
            intent=intent,
            used_rag=used_rag,
            used_tool_calling=used_tool_calling,
            rag_sources=rag_sources or [],
            tool_names=tool_names or [],
        )
    except Exception as exc:
        print(f"[Memory 警告] 对话日志保存失败，将继续本轮对话：{exc}\n")


def get_live_conversation_history(memory_session: DialogueMemorySession) -> list[dict]:
    return list(memory_session.recent_conversation_history)


def compact_memory_if_needed(
    memory_session: DialogueMemorySession,
    force: bool = False,
) -> None:
    if not memory_session.should_compact(force=force):
        return
    # Phase 4 MVP defines the compact contract and trigger point.
    # Claude-based compact generation can be added after storage is stable.
    memory_session.trim_recent_history()
    memory_session.mark_compacted()


def extract_inventory_item_name(player_input: str) -> str:
    text = player_input.strip()
    replacements = [
        "我背包里有",
        "背包里有",
        "我有没有",
        "有没有",
        "是否拥有",
        "这个道具",
        "这个物品",
        "吗",
        "？",
        "?",
    ]
    for token in replacements:
        text = text.replace(token, "")

    lowered = text.lower()
    lowered = re.sub(r"\bdo i have\b", "", lowered)
    lowered = re.sub(r"\bin my inventory\b", "", lowered)
    lowered = re.sub(r"\bin my backpack\b", "", lowered)
    lowered = re.sub(r"\bthe\b", "", lowered)
    return " ".join(lowered.split()) if lowered.strip() else text.strip()


def generate_gm_feedback(npc: dict, player_context: dict, conversation_history: list) -> str:
    """
    对话结束后，从 GM 视角生成玩家行为反馈报告
    """
    gm_prompt = build_gm_prompt(npc, player_context, conversation_history)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=gm_prompt,
        messages=[{"role": "user", "content": "请生成本次对话的玩家反馈报告。"}]
    )

    usage = response.usage
    print(f"\n[Token 使用: 输入 {usage.input_tokens} / 输出 {usage.output_tokens}]\n")
    return response.content[0].text


def generate_gm_feedback_safely(
    npc: dict,
    player_context: dict,
    conversation_history: list,
) -> str:
    try:
        return generate_gm_feedback(npc, player_context, conversation_history)
    except Exception as exc:
        return (
            "[GM Workflow] 完整对话记录已保存。"
            f"自动 GM 反馈生成失败：{exc}"
        )


def generate_quest(npc: dict, player_context: dict, conversation_history: list) -> str:
    """
    触发任务生成
    把对话历史也传进去，让任务内容和当前对话有关联感
    """
    quest_prompt = build_quest_prompt(npc, player_context, conversation_history)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=quest_prompt,
        messages=[{"role": "user", "content": "请给我一个任务。"}]
    )

    usage = response.usage
    print(f"\n[Token 使用: 输入 {usage.input_tokens} / 输出 {usage.output_tokens}]\n")
    return response.content[0].text


def execute_tool_uses(tool_uses: list) -> dict:
    tool_results = []
    tool_names = []
    for tool_use in tool_uses:
        tool_name = tool_use.name
        execution = execute_tool_call({
            "name": tool_name,
            "input": tool_use.input,
        })
        tool_names.append(tool_name)
        tool_results.append(tool_result_block(tool_use.id, execution))
    return {
        "tool_results": tool_results,
        "tool_names": tool_names,
    }


def build_tool_system_prompt(npc: dict, player_context: dict) -> str:
    system_prompt = build_system_prompt(npc, player_context)
    return f"""{system_prompt}

== 工具调用规则 ==
当前玩家 ID：{player_context.get('player_id', 'player_001')}
当玩家询问背包、物品、下一步行动、任务元数据或任务完成条件时，优先使用可用工具。
工具结果是游戏后端状态，必须优先于猜测。不要向玩家暴露 Python 错误或内部实现。
"""


def run_claude_tool_turn_with_metadata(
    npc: dict,
    player_context: dict,
    player_input: str,
) -> dict:
    system_prompt = build_tool_system_prompt(npc, player_context)
    messages = [{"role": "user", "content": player_input}]
    first_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=system_prompt,
        messages=messages,
        tools=CLAUDE_TOOLS,
    )

    tool_uses = [
        block for block in first_response.content
        if getattr(block, "type", None) == "tool_use"
    ]
    if not tool_uses:
        return {
            "reply": first_response.content[0].text,
            "tool_names": [],
        }

    executed = execute_tool_uses(tool_uses)

    follow_up = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=system_prompt,
        messages=[
            messages[0],
            {"role": "assistant", "content": first_response.content},
            {"role": "user", "content": executed["tool_results"]},
        ],
    )
    return {
        "reply": follow_up.content[0].text,
        "tool_names": executed["tool_names"],
    }


def run_claude_tool_turn(npc: dict, player_context: dict, player_input: str) -> str:
    return run_claude_tool_turn_with_metadata(
        npc,
        player_context,
        player_input,
    )["reply"]


def chat_with_npc(npc_id: str, player_context: dict):
    npc = load_npc_config(npc_id)
    player_id = player_context.get("player_id", "player_001")
    memory_session = DialogueMemorySession(npc_id=npc_id, player_id=player_id)
    rag_service = None
    rag_disabled = False

    def get_rag_service():
        nonlocal rag_service, rag_disabled
        if rag_disabled:
            return None
        if rag_service is None:
            try:
                rag_service = RagService()
            except Exception as exc:
                rag_disabled = True
                print(f"[RAG 警告] 知识辅助初始化失败，将继续使用普通 NPC 对话：{exc}\n")
                return None
        return rag_service

    print(f"\n{'='*50}")
    print(f"你遇到了：{npc['name']} — {npc['title']}")
    print(f"{'='*50}")
    print(f"{npc['name']}: {npc['greeting']}")
    print("\n(输入 'quit' 退出 | 输入 '/quest' 获取任务)\n")

    while True:
        player_input = input("你: ").strip()

        if player_input.lower() == "quit":
            print(f"\n{npc['name']}: {npc['farewell']}")
            compact_memory_if_needed(memory_session, force=True)
            break

        if not player_input:
            continue

        if not safety_check(player_input):
            print("[系统提示] 该输入包含不允许的内容，请重新输入。\n")
            continue

        # /quest 指令：触发任务生成
        if player_input.lower() == "/quest":
            print(f"\n{npc['name']} 沉吟片刻……\n")
            quest_text = generate_quest(
                npc,
                player_context,
                get_live_conversation_history(memory_session),
            )
            print(f"{npc['name']}: {quest_text}\n")

            record_memory_turn(
                memory_session=memory_session,
                user_input="（玩家请求一个任务）",
                assistant_reply=quest_text,
                intent=QUEST_REQUEST,
            )
            compact_memory_if_needed(memory_session)
            continue

        route = classify_intent(player_input)
        print(f"[Router] intent={route.intent} reason={route.reason}")

        if route.intent == QUEST_REQUEST:
            print(f"\n{npc['name']} 沉吟片刻……\n")
            quest_text = generate_quest(
                npc,
                player_context,
                get_live_conversation_history(memory_session),
            )
            print(f"{npc['name']}: {quest_text}\n")
            record_memory_turn(
                memory_session=memory_session,
                user_input=player_input,
                assistant_reply=quest_text,
                intent=route.intent,
            )
            compact_memory_if_needed(memory_session)
            continue

        if should_use_claude_tools(route.intent):
            tool_names = []
            try:
                tool_turn = run_claude_tool_turn_with_metadata(
                    npc,
                    player_context,
                    player_input,
                )
                tool_response = tool_turn["reply"]
                tool_names = tool_turn["tool_names"]
            except Exception as exc:
                item_name = extract_inventory_item_name(player_input)
                inventory_result = check_inventory(player_context, item_name)
                tool_response = format_inventory_response(inventory_result)
                tool_names = ["check_inventory"]
                print(f"[Tool Calling 警告] Claude 工具调用失败，已使用本地工具兜底：{exc}\n")
            print(f"\n{npc['name']}: {tool_response}\n")
            record_memory_turn(
                memory_session=memory_session,
                user_input=player_input,
                assistant_reply=tool_response,
                intent=route.intent,
                used_tool_calling=True,
                tool_names=tool_names,
            )
            compact_memory_if_needed(memory_session)
            continue

        if route.intent == GM_FEEDBACK:
            if is_gm_advisor(npc_id):
                print("\n[GM 正在分析你的反馈…]\n")
                feedback_history = memory_session.full_conversation_history + [
                    {"role": "user", "content": player_input}
                ]
                feedback_response = generate_gm_feedback_safely(
                    npc,
                    player_context,
                    feedback_history,
                )
            else:
                feedback_response = (
                    "[GM Workflow] 已记录你的反馈。"
                    "如果你想做完整复盘，可以退出后选择 gm_advisor 找档案官赛琳。"
                )
            print(f"\n{feedback_response}\n")
            record_memory_turn(
                memory_session=memory_session,
                user_input=player_input,
                assistant_reply=feedback_response,
                intent=route.intent,
            )
            compact_memory_if_needed(memory_session)
            continue

        # 普通对话
        retrieved_context = ""
        active_rag_service = get_rag_service() if should_use_rag_for_intent(route.intent) else None
        if active_rag_service is not None:
            try:
                retrieval = active_rag_service.retrieve(player_input, npc_id=npc_id)
                retrieved_context = retrieval.context
            except Exception as exc:
                rag_disabled = True
                print(f"[RAG 警告] 本轮知识辅助失败，本次会话将使用普通对话：{exc}\n")

        system_prompt = build_system_prompt(
            npc,
            player_context,
            retrieved_context=retrieved_context,
            memory_summary=memory_session.session_summary,
            quest_state=memory_session.quest_state,
        )
        live_history = get_live_conversation_history(memory_session)
        live_history.append({"role": "user", "content": player_input})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=system_prompt,
            messages=live_history
        )

        npc_reply = response.content[0].text
        record_memory_turn(
            memory_session=memory_session,
            user_input=player_input,
            assistant_reply=npc_reply,
            intent=route.intent,
            used_rag=bool(retrieved_context),
        )
        compact_memory_if_needed(memory_session)

        print(f"\n{npc['name']}: {npc_reply}\n")

        usage = response.usage
        print(f"[Token 使用: 输入 {usage.input_tokens} / 输出 {usage.output_tokens}]\n")


def main():
    player_context = get_player_profile("player_001")
    player_context["quest_status"] = "刚刚抵达小镇，尚未接取任务"
    player_context["reputation"] = "中立"

    print("=== NPC 对话 + 任务生成原型 ===")
    available_npcs = " / ".join(load_all_npc_configs())
    print(f"可用 NPC: {available_npcs}")
    npc_choice = input("选择 NPC: ").strip() or "blacksmith"

    chat_with_npc(npc_choice, player_context)


if __name__ == "__main__":
    main()
