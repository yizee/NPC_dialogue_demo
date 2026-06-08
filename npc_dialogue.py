"""
NPC Dialogue + Quest Generator — 方案 A+B 合体版
新增：/quest 指令触发任务生成，完全融合在对话中
"""

import json
import os
import re

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
    QUEST_REQUEST,
    classify_intent,
)
from prompts import build_gm_prompt, build_quest_prompt, build_system_prompt
from rag_pipeline import RagService

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def load_npc_config(npc_id: str) -> dict:
    with open("npc_config.json", "r", encoding="utf-8") as f:
        configs = json.load(f)
    if npc_id not in configs:
        raise ValueError(f"NPC '{npc_id}' 不存在，可选: {list(configs.keys())}")
    return configs[npc_id]


def safety_check(player_input: str) -> bool:
    forbidden_keywords = ["hack", "exploit", "bypass", "ignore your instructions"]
    return not any(kw in player_input.lower() for kw in forbidden_keywords)


def should_use_rag_for_intent(intent: str) -> bool:
    return intent == LORE_QUESTION


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


def chat_with_npc(npc_id: str, player_context: dict):
    npc = load_npc_config(npc_id)
    conversation_history = []
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
            if conversation_history:
                print("\n[GM 正在生成反馈报告…]\n")
                feedback = generate_gm_feedback(npc, player_context, conversation_history)
                print(f"{'='*50}\n[GM 反馈报告]\n{feedback}\n{'='*50}\n")
            break

        if not player_input:
            continue

        if not safety_check(player_input):
            print("[系统提示] 该输入包含不允许的内容，请重新输入。\n")
            continue

        # /quest 指令：触发任务生成
        if player_input.lower() == "/quest":
            print(f"\n{npc['name']} 沉吟片刻……\n")
            quest_text = generate_quest(npc, player_context, conversation_history)
            print(f"{npc['name']}: {quest_text}\n")

            # 把任务对话也加入历史，保持上下文连贯
            conversation_history.append({"role": "user", "content": "（玩家请求一个任务）"})
            conversation_history.append({"role": "assistant", "content": quest_text})
            continue

        route = classify_intent(player_input)
        print(f"[Router] intent={route.intent} reason={route.reason}")

        if route.intent == QUEST_REQUEST:
            print(f"\n{npc['name']} 沉吟片刻……\n")
            quest_text = generate_quest(npc, player_context, conversation_history)
            print(f"{npc['name']}: {quest_text}\n")
            conversation_history.append({"role": "user", "content": player_input})
            conversation_history.append({"role": "assistant", "content": quest_text})
            continue

        if route.intent == INVENTORY_QUERY:
            item_name = extract_inventory_item_name(player_input)
            inventory_result = check_inventory(player_context, item_name)
            tool_response = format_inventory_response(inventory_result)
            print(f"\n{tool_response}\n")
            conversation_history.append({"role": "user", "content": player_input})
            conversation_history.append({"role": "assistant", "content": tool_response})
            continue

        if route.intent == GM_FEEDBACK:
            feedback_response = (
                "[GM Workflow] 已记录你的反馈。"
                "如果这是任务或剧情问题，GM 会优先检查任务目标、奖励和触发条件是否清晰。"
            )
            if conversation_history:
                feedback_response += " 你也可以输入 quit 结束本轮对话并生成完整 GM 反馈报告。"
            print(f"\n{feedback_response}\n")
            conversation_history.append({"role": "user", "content": player_input})
            conversation_history.append({"role": "assistant", "content": feedback_response})
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
        )
        conversation_history.append({"role": "user", "content": player_input})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=system_prompt,
            messages=conversation_history
        )

        npc_reply = response.content[0].text
        conversation_history.append({"role": "assistant", "content": npc_reply})

        print(f"\n{npc['name']}: {npc_reply}\n")

        usage = response.usage
        print(f"[Token 使用: 输入 {usage.input_tokens} / 输出 {usage.output_tokens}]\n")


def main():
    player_context = get_player_profile("player_001")
    player_context["quest_status"] = "刚刚抵达小镇，尚未接取任务"
    player_context["reputation"] = "中立"

    print("=== NPC 对话 + 任务生成原型 ===")
    print("可用 NPC: blacksmith / innkeeper / mysterious_wizard")
    npc_choice = input("选择 NPC: ").strip() or "blacksmith"

    chat_with_npc(npc_choice, player_context)


if __name__ == "__main__":
    main()
