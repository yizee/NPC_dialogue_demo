"""
NPC Dialogue Prototype — 方案 A
使用 Anthropic API 生成游戏 NPC 对话
"""

import os
import json
from anthropic import Anthropic
from prompts import build_system_prompt, SAFETY_CHECK_PROMPT

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def load_npc_config(npc_id: str) -> dict:
    """加载 NPC 配置"""
    with open("npc_config.json", "r", encoding="utf-8") as f:
        configs = json.load(f)
    if npc_id not in configs:
        raise ValueError(f"NPC '{npc_id}' 不存在，可选: {list(configs.keys())}")
    return configs[npc_id]


def safety_check(player_input: str) -> bool:
    """
    简单的 safety 检查：检测玩家输入是否包含不适合游戏的内容
    返回 True 表示安全，False 表示需要过滤
    """
    forbidden_keywords = ["hack", "exploit", "bypass", "ignore your instructions"]
    lowered = player_input.lower()
    return not any(kw in lowered for kw in forbidden_keywords)


def chat_with_npc(npc_id: str, player_context: dict):
    """
    与指定 NPC 进行多轮对话
    player_context: 包含玩家信息，如 name, level, quest_status
    """
    npc = load_npc_config(npc_id)
    system_prompt = build_system_prompt(npc, player_context)
    conversation_history = []

    print(f"\n{'='*50}")
    print(f"你遇到了：{npc['name']} — {npc['title']}")
    print(f"{'='*50}")
    print(f"{npc['name']}: {npc['greeting']}")
    print("\n(输入 'quit' 退出对话)\n")

    while True:
        player_input = input("你: ").strip()

        if player_input.lower() == "quit":
            print(f"\n{npc['name']}: {npc['farewell']}")
            break

        if not player_input:
            continue

        # Safety 检查
        if not safety_check(player_input):
            print(f"[系统提示] 该输入包含不允许的内容，请重新输入。\n")
            continue

        # 加入对话历史
        conversation_history.append({
            "role": "user",
            "content": player_input
        })

        # 调用 API
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=system_prompt,
            messages=conversation_history
        )

        npc_reply = response.content[0].text

        # 将回复加入历史
        conversation_history.append({
            "role": "assistant",
            "content": npc_reply
        })

        print(f"\n{npc['name']}: {npc_reply}\n")

        # 显示 token 用量（方便你追踪 API 消耗）
        usage = response.usage
        print(f"[Token 使用: 输入 {usage.input_tokens} / 输出 {usage.output_tokens}]\n")


def main():
    # 玩家上下文（可以改成动态输入）
    player_context = {
        "name": "旅行者",
        "level": 5,
        "quest_status": "刚刚抵达小镇，尚未接取任务",
        "reputation": "中立"
    }

    # 可用 NPC 列表
    print("=== NPC 对话原型 ===")
    print("可用 NPC: blacksmith（铁匠）/ innkeeper（酒馆老板）/ mysterious_wizard（神秘法师）")
    npc_choice = input("选择 NPC: ").strip() or "blacksmith"

    chat_with_npc(npc_choice, player_context)


if __name__ == "__main__":
    main()
