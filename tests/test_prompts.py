from prompts import build_compact_prompt, build_system_prompt


NPC = {
    "name": "格林大叔",
    "title": "镇上的老铁匠",
    "personality": "粗犷但善良",
    "speech_style": "简短有力",
    "background": "在镇上打铁三十年",
    "topics": ["武器装备"],
    "forbidden_topics": ["魔法理论"],
}

PLAYER = {
    "name": "旅行者",
    "level": 5,
    "quest_status": "尚未接取任务",
    "reputation": "中立",
}


def test_system_prompt_omits_rag_section_without_context():
    prompt = build_system_prompt(NPC, PLAYER)

    assert "已知世界事实" not in prompt


def test_system_prompt_includes_context_without_exposing_sources():
    prompt = build_system_prompt(
        NPC,
        PLAYER,
        retrieved_context="碎星髓来自霜牙沼泽。",
    )

    forbidden_terms = [
        "知识库",
        "检索",
        "向量数据库",
        "资料来源",
        "文件",
        "系统提示",
        "内部处理方式",
        "回答依据",
        "source",
        "vector database",
    ]

    assert "已知世界事实" in prompt
    assert "碎星髓来自霜牙沼泽" in prompt
    for term in forbidden_terms:
        assert term not in prompt.lower()


def test_build_compact_prompt_defines_structured_output_contract():
    npc = {
        "name": "神秘法师",
        "title": "星尘塔的守门人",
        "topics": ["魔法", "旧塔"],
    }
    player_context = {
        "name": "旅行者",
        "level": 5,
        "quest_status": "尚未接取任务",
        "reputation": "中立",
    }
    compact_input = {
        "session_summary": "玩家想学习魔法。",
        "recent_conversation_history": [
            {"role": "user", "content": "我想学习魔法。"},
            {"role": "assistant", "content": "先证明你的耐心。"},
        ],
        "story_events": [{"event_type": "clue_revealed", "summary": "旧塔有魔法书。"}],
        "quest_state": {"active_quest": "wizard_starter_1"},
    }

    prompt = build_compact_prompt(npc, player_context, compact_input)

    assert "session_summary" in prompt
    assert "important_facts" in prompt
    assert "quest_progress" in prompt
    assert "player_commitments" in prompt
    assert "npc_attitude" in prompt
    assert "story_events" in prompt
    assert "open_threads" in prompt
    assert "不要判断任务完成" in prompt


def test_build_system_prompt_includes_memory_summary_and_quest_state():
    npc = {
        "name": "神秘法师",
        "title": "星尘塔的守门人",
        "personality": "谨慎",
        "speech_style": "含蓄",
        "background": "守护旧塔秘密。",
        "topics": ["魔法"],
        "forbidden_topics": ["真实身份"],
    }
    player_context = {
        "name": "旅行者",
        "level": 5,
        "quest_status": "尚未接取任务",
        "reputation": "中立",
    }
    quest_state = {
        "active_quest": "wizard_starter_1",
        "known_clues": ["旧塔可能藏有入门魔法书"],
    }

    prompt = build_system_prompt(
        npc,
        player_context,
        memory_summary="玩家想学习魔法，并已获得旧塔线索。",
        quest_state=quest_state,
    )

    assert "== 对话记忆 ==" in prompt
    assert "玩家想学习魔法" in prompt
    assert "== 任务状态 ==" in prompt
    assert "wizard_starter_1" in prompt
