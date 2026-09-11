"""Unit tests for database/get_games.py's playoff-field extraction (_extract_playoff_fields).

No network, no DB: `_extract_playoff_fields` is a pure function over a single dict, pulled out
of get_games_by_year_week's request/DB-dependent body specifically so this loop can be exercised
in isolation. The real, previously-unnoticed bug this guards against: get_games.py used to read
playoff.get("round_order"), a key that does not exist on GamePlayoff at all -- round_order lives
on a different endpoint's PlayoffMatchup object -- so it silently returned None for every row,
every season, for a year, because nothing under tests/ exercised this extraction in isolation
before. Fixed by reading `round` instead; the DB column name (playoff_round_order) is unchanged.

Run: python -m pytest tests/ -q   (or: python tests/test_get_games_playoff_extraction.py)
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.get_games import _extract_playoff_fields  # noqa: E402


def test_extraction_populates_every_field_from_a_full_playoff_object():
    """A CFP row shaped like CFBD's real /games response: a full playoff object, no round_order
    key at all (matching real 2025 data -- see tests/test_postseason_slots.py)."""
    game = {
        "id": 1,
        "playoff": {
            "competition": "CFP",
            "format": "12-team",
            "round": 2,
            "round_name": "Quarterfinal",
            "bracket_slot": "QF1",
            "home_seed": 1,
            "away_seed": 8,
            "bowl_name": "Rose Bowl",
        },
    }
    _extract_playoff_fields(game)
    assert game["playoffRoundName"] == "Quarterfinal"
    assert game["playoffRoundOrder"] == 2
    assert game["playoffBracketSlot"] == "QF1"
    assert game["playoffBowlName"] == "Rose Bowl"
    assert game["playoffHomeSeed"] == 1
    assert game["playoffAwaySeed"] == 8


def test_extraction_never_reads_the_nonexistent_round_order_key():
    """THE regression this file exists to guard against: GamePlayoff has no round_order field --
    it lives on a different endpoint's PlayoffMatchup. A row carrying a round_order key (as if
    from the old, wrong assumption about the schema) but no `round` key must still yield None,
    not the round_order value -- proving the extraction reads `round`, never `round_order`."""
    game = {"id": 2, "playoff": {"round_order": 99, "round_name": "Semifinal"}}
    _extract_playoff_fields(game)
    assert game["playoffRoundOrder"] is None


def test_extraction_degrades_to_none_for_a_null_playoff_object():
    game = {"id": 3, "playoff": None}
    _extract_playoff_fields(game)
    assert game["playoffRoundName"] is None
    assert game["playoffRoundOrder"] is None
    assert game["playoffBracketSlot"] is None
    assert game["playoffBowlName"] is None
    assert game["playoffHomeSeed"] is None
    assert game["playoffAwaySeed"] is None


def test_extraction_degrades_to_none_when_the_playoff_key_is_absent_entirely():
    """The vast majority of real rows -- every regular-season, non-CFP-postseason game -- have
    no `playoff` key at all, not even a null one."""
    game = {"id": 4}
    _extract_playoff_fields(game)
    assert game["playoffRoundName"] is None
    assert game["playoffRoundOrder"] is None
    assert game["playoffBracketSlot"] is None
    assert game["playoffBowlName"] is None
    assert game["playoffHomeSeed"] is None
    assert game["playoffAwaySeed"] is None


def test_extraction_over_a_mixed_synthetic_games_data_list_never_raises():
    """The shape get_games_by_year_week actually iterates: a real CFP row, a null-playoff row,
    and a no-playoff-key row in the same list, exactly as CFBD's real /games response mixes
    them."""
    games_data = [
        {"id": 1, "playoff": {"round": 1, "round_name": "First Round", "bracket_slot": "FR1",
                               "bowl_name": "College Football Playoff First Round Game",
                               "home_seed": 4, "away_seed": 5}},
        {"id": 2, "playoff": None},
        {"id": 3},
    ]
    for game in games_data:
        _extract_playoff_fields(game)
    assert games_data[0]["playoffRoundOrder"] == 1
    assert games_data[0]["playoffHomeSeed"] == 4
    assert games_data[0]["playoffAwaySeed"] == 5
    assert games_data[1]["playoffRoundName"] is None
    assert games_data[2]["playoffBowlName"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
