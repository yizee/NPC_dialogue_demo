from prompts import build_system_prompt


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
