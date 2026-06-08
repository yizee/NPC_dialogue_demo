from game_tools import (
    check_inventory,
    format_inventory_response,
    get_player_profile,
    recommend_next_action,
)


def test_get_player_profile_returns_default_profile():
    profile = get_player_profile("player_001")

    assert profile["name"] == "旅行者"
    assert profile["level"] == 5
    assert profile["class"] == "未选择"
    assert "月盐药剂" in profile["inventory"]


def test_check_inventory_finds_known_item():
    profile = get_player_profile("player_001")

    result = check_inventory(profile, "月盐药剂")

    assert result["has_item"] is True
    assert result["item_name"] == "月盐药剂"


def test_check_inventory_supports_common_english_alias():
    profile = get_player_profile("player_001")

    result = check_inventory(profile, "black iron tongs")

    assert result["has_item"] is True
    assert result["item_name"] == "黑铁夹具"


def test_check_inventory_rejects_missing_item():
    profile = get_player_profile("player_001")

    result = check_inventory(profile, "魔龙之心")

    assert result["has_item"] is False
    assert result["item_name"] == "魔龙之心"


def test_check_inventory_rejects_empty_item_name():
    profile = get_player_profile("player_001")

    result = check_inventory(profile, "  ")

    assert result["has_item"] is False
    assert result["item_name"] == ""


def test_format_inventory_response_is_deterministic():
    profile = get_player_profile("player_001")
    found = check_inventory(profile, "月盐药剂")
    missing = check_inventory(profile, "魔龙之心")

    assert "你有月盐药剂" in format_inventory_response(found)
    assert "你还没有魔龙之心" in format_inventory_response(missing)


def test_recommend_next_action_for_unselected_class():
    profile = get_player_profile("player_001")

    recommendation = recommend_next_action(profile)

    assert "选择一个主要职业" in recommendation
