"""The elimination test must not say "eliminated" on banked wins alone when teams are on
different numbers of counted conference games.

More wins is a better record only on the same game count: 8-2 is more wins than 7-1 and a worse
percentage. And under the two tie definitions wider than equal percentage (ACC: same wins OR same
losses; CUSA: within one win AND equal losses) a team on a different game count can be TIED with
one that has more wins, and a tiebreaker could then put it ahead.

Every fixture is W-L-remaining conference records for a five-team conference, top two advance.
"""
import pytest  # type: ignore

from artifacts.schedule_standings import (
    _surely_finishes_ahead,
    compute_conference_championship_status,
)

CONF = "Testing Conference"
QUALIFYING = {CONF: "top2"}


def _rec(wins, losses, remaining):
    return {"conference": CONF, "wins": wins, "losses": losses, "conf_wins": wins,
            "conf_losses": losses, "conf_games_remaining": remaining}


def _status(records, tie_definition=None):
    tie_definitions = None if tie_definition is None else {CONF: tie_definition}
    result = compute_conference_championship_status(
        records, qualifying_conferences=QUALIFYING, tie_definitions=tie_definitions)
    return {team: entry["status"] for team, entry in result.items()}


def test_same_game_count_is_the_original_wins_only_test():
    """Everyone on eight games: two teams with more banked wins than Echo can reach end it."""
    records = {
        "Alpha": _rec(7, 0, 1), "Bravo": _rec(6, 1, 1), "Charlie": _rec(4, 3, 1),
        "Delta": _rec(3, 4, 1), "Echo": _rec(4, 4, 0),
    }
    for definition in (None, "win_pct", "acc_alternate_games", "cusa_within_one_win"):
        assert _status(records, definition)["Echo"] == "eliminated", definition


def test_more_wins_on_a_longer_schedule_is_not_ahead_on_percentage():
    """Echo can finish 7-1 (.875). Alpha and Bravo have banked 8 wins on ten-game schedules and
    could finish 8-2 (.800): more wins, worse record. Echo is alive."""
    records = {
        "Alpha": _rec(8, 0, 2), "Bravo": _rec(8, 0, 2), "Charlie": _rec(2, 6, 0),
        "Delta": _rec(1, 7, 0), "Echo": _rec(6, 1, 1),
    }
    assert _status(records, "win_pct")["Echo"] == "possible"
    assert _status(records)["Echo"] == "possible"  # legacy callers read as win_pct


def test_a_longer_schedule_still_eliminates_once_the_percentage_is_out_of_reach():
    """Same shape, but Echo's best is 5-3 (.625) and both leaders' worst is 8-2 (.800)."""
    records = {
        "Alpha": _rec(8, 0, 2), "Bravo": _rec(8, 0, 2), "Charlie": _rec(2, 6, 0),
        "Delta": _rec(1, 7, 0), "Echo": _rec(4, 3, 1),
    }
    assert _status(records, "win_pct")["Echo"] == "eliminated"


@pytest.mark.parametrize("definition", ["acc_alternate_games", "cusa_within_one_win", "something_new"])
def test_a_possible_shared_loss_total_keeps_the_team_alive_under_wider_tie_rules(definition):
    """Echo (eight games) is 5-2 with one to play: best 6-2. Alpha and Bravo (nine games) have
    banked 7 wins and are 7-1 with one to play: worst 7-2, better on percentage than 6-2. Under
    equal percentage Echo is out. But 7-2 and 6-2 share a loss total, which the wider
    definitions call a tie -- so a tiebreaker could still put Echo in. Stay "possible"."""
    records = {
        "Alpha": _rec(7, 1, 1), "Bravo": _rec(7, 1, 1), "Charlie": _rec(2, 6, 0),
        "Delta": _rec(1, 7, 0), "Echo": _rec(5, 2, 1),
    }
    assert _status(records, "win_pct")["Echo"] == "eliminated"
    assert _status(records, definition)["Echo"] == "possible"


def test_wider_tie_rules_still_eliminate_when_no_loss_total_can_be_shared():
    """As above, but Echo already has four losses: nobody ahead of it can get there."""
    records = {
        "Alpha": _rec(7, 1, 1), "Bravo": _rec(7, 1, 1), "Charlie": _rec(2, 6, 0),
        "Delta": _rec(1, 7, 0), "Echo": _rec(3, 4, 1),
    }
    for definition in ("acc_alternate_games", "cusa_within_one_win"):
        assert _status(records, definition)["Echo"] == "eliminated", definition


def test_only_teams_surely_ahead_are_counted_toward_the_cutoff():
    """One leader is surely ahead of Echo, the other is not (longer schedule, worse worst case):
    one is fewer than the two berths, so Echo is alive."""
    records = {
        "Alpha": _rec(8, 0, 0), "Bravo": _rec(8, 0, 2), "Charlie": _rec(2, 6, 0),
        "Delta": _rec(1, 7, 0), "Echo": _rec(6, 1, 1),
    }
    assert _status(records, "win_pct")["Echo"] == "possible"


def test_helper_requires_more_banked_wins_than_the_team_can_reach():
    W, B, N = {"A": 6, "T": 5}, {"A": 8, "T": 6}, {"A": 8, "T": 8}
    assert _surely_finishes_ahead("A", "T", W, B, N, "win_pct") is False
    W["A"] = 7
    assert _surely_finishes_ahead("A", "T", W, B, N, "win_pct") is True


def test_unknown_tie_rule_also_needs_a_clear_win_of_separation():
    """An unrecognised definition is read as strictly as _clear_of_other_game_counts reads it:
    a tie could mean "within one win", so one more banked win than the team can reach is not
    enough across different game counts."""
    W, B, N = {"A": 7, "T": 3}, {"A": 7, "T": 6}, {"A": 9, "T": 7}
    # A is 7-2 final; T can reach 6-1. Percentage: 7/9 < 6/7, so not ahead under any rule.
    assert _surely_finishes_ahead("A", "T", W, B, N, "something_new") is False
    W, B, N = {"A": 9, "T": 3}, {"A": 9, "T": 8}, {"A": 10, "T": 12}
    # A is 9-1; T can reach 8-4. Ahead on percentage, no shared loss total, but only one win clear.
    assert _surely_finishes_ahead("A", "T", W, B, N, "win_pct") is True
    assert _surely_finishes_ahead("A", "T", W, B, N, "acc_alternate_games") is True
    assert _surely_finishes_ahead("A", "T", W, B, N, "something_new") is False


def test_a_named_title_game_participant_the_math_eliminated_rejects_the_participant_set():
    """Review finding. CUSA, season over: Bravo and Charlie are 9-0, Delta 8-0, Alpha 7-1, and
    the "title game" is said to be Alpha against Delta. The stricter clinch test leaves Bravo
    and Charlie "possible" (unequal game counts under CUSA's tie rule), so the older guard --
    "the override would eliminate a clinched team" -- no longer fires. Alpha, a named
    participant, computes as eliminated: that contradiction must reject the participant set
    instead of letting it eliminate the two unbeaten teams."""
    records = {
        "Alpha": _rec(7, 1, 0), "Bravo": _rec(9, 0, 0), "Charlie": _rec(9, 0, 0), "Delta": _rec(8, 0, 0),
    }
    result = compute_conference_championship_status(
        records, qualifying_conferences=QUALIFYING, tie_definitions={CONF: "cusa_within_one_win"},
        ccg_participants={CONF: {"Alpha", "Delta"}},
    )
    assert result["Alpha"]["status"] == "eliminated"
    assert result["Bravo"]["status"] != "eliminated"
    assert result["Charlie"]["status"] != "eliminated"
