"""Tests for artifacts/tiebreaker_steps.py (T2 of docs/conference-tiebreakers/plan.yaml): the
shared operation library for conference championship tiebreakers.

Every primitive is tested in isolation with synthetic rows -- no live DB, no real 2025 games
replayed play-by-play -- but several fixtures deliberately reproduce REAL published numbers as
correctness targets, named in each test:
  - sec.txt Appendix A's own worked example for capped_relative_scoring_margin (+30.9).
  - The real 2025 ACC five-way tie's opponents_cumulative_conf_pct numbers (Duke 32-32 best of
    the five; Georgia Tech/Miami 28-36; Pittsburgh/SMU 27-37).
  - common_opponents_record's gate: the real 2025 ACC common set of exactly one opponent
    (Syracuse) -> None, and the real 2025 MAC three-team case (Miami (OH)/Toledo/Ohio against
    Ball State/Northern Illinois/Western Michigan) -> a real partition.
  - vs_placed_opponents against its two published sub-rules (sec.txt C.1's
    head_to_head_then_combine vs big12.txt step c's combine).

Run: uv run pytest tests/test_tiebreaker_steps.py -q
     (or: uv run pytest tests/ -q for the full suite)
"""
import sys
from pathlib import Path

import logging

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifacts.tiebreaker_steps import (  # noqa: E402
    STEP_REGISTRY,
    TiebreakContext,
    _capped_relative_margin_for_team,
    capped_relative_scoring_margin,
    common_opponents_record,
    external_ranking,
    head_to_head,
    opponents_cumulative_conf_pct,
    random_draw,
    sub_group_record,
    _final_conference_week,
    conditional_external_ranking,
    divisional_record,
    overall_win_pct,
    sweep_in_out,
    total_wins_capped,
    vs_placed_opponents,
)

SEASON = 2025


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _row(team, opponent, status, team_score=None, opp_score=None, conference_game=True,
         season=SEASON, season_type="regular", game_id=None, week=None):
    return {
        "season": season,
        "team": team,
        "opponent": opponent,
        "status": status,
        "conference_game": conference_game,
        "team_score": team_score,
        "opp_score": opp_score,
        "season_type": season_type,
        "game_id": game_id,
        "week": week,
    }


def _game(team_a, score_a, team_b, score_b, conference_game=True, season=SEASON):
    """Both team-oriented rows of one game -- the shape schedule_grid actually produces (one
    played game becomes two rows, opposite perspectives, one game_id)."""
    status_a = "win" if score_a > score_b else "loss"
    status_b = "win" if score_b > score_a else "loss"
    return [
        _row(team_a, team_b, status_a, score_a, score_b, conference_game, season),
        _row(team_b, team_a, status_b, score_b, score_a, conference_game, season),
    ]


def _ctx(rows, frozen_order=None, conf_records=None, team_ranks=None, conference="TEST",
         non_fbs_teams=None, placement_excluded_game_ids=frozenset(), divisions=None):
    """`non_fbs_teams` defaults to None, meaning NOT SUPPLIED -- deliberately distinct from an
    empty frozenset, which means supplied-and-nobody-is-non-FBS. total_wins_capped branches on
    exactly that difference, so the default must stay None rather than frozenset()."""
    return TiebreakContext(
        rows=rows,
        season=SEASON,
        conference=conference,
        frozen_order=frozen_order or [],
        conf_records=conf_records or {},
        team_ranks=team_ranks or {},
        placement_excluded_game_ids=placement_excluded_game_ids,
        divisions=divisions or {},
        non_fbs_teams=non_fbs_teams,
    )


# ===========================================================================
# 1. head_to_head
# ===========================================================================
def test_head_to_head_single_meeting_counted_once():
    rows = _game("A", 24, "B", 10)
    result = head_to_head(["A", "B"], _ctx(rows))
    assert result == [["A"], ["B"]]


def test_head_to_head_split_series_is_none():
    rows = _game("A", 24, "B", 10) + _game("A", 10, "B", 24)
    result = head_to_head(["A", "B"], _ctx(rows))
    assert result is None


def test_head_to_head_no_meeting_is_none():
    assert head_to_head(["A", "B"], _ctx([])) is None


def test_head_to_head_wrong_group_size_is_none():
    rows = _game("A", 24, "B", 10)
    assert head_to_head(["A", "B", "C"], _ctx(rows)) is None


# ===========================================================================
# 2. sub_group_record
# ===========================================================================
def test_sub_group_record_complete_round_robin_separates():
    rows = _game("A", 20, "B", 10) + _game("A", 20, "C", 10) + _game("B", 20, "C", 10)
    ctx = _ctx(rows)
    result = sub_group_record(["A", "B", "C"], ctx, require_round_robin=True)
    assert result == [["A"], ["B"], ["C"]]


def test_sub_group_record_incomplete_and_gated_returns_none():
    # A beat B and C; B and C never played each other -- not a complete round robin.
    rows = _game("A", 20, "B", 10) + _game("A", 20, "C", 10)
    ctx = _ctx(rows)
    assert sub_group_record(["A", "B", "C"], ctx, require_round_robin=True) is None


def test_sub_group_record_incomplete_without_gate_uses_games_played():
    rows = _game("A", 20, "B", 10) + _game("A", 20, "C", 10)
    ctx = _ctx(rows)
    result = sub_group_record(["A", "B", "C"], ctx, require_round_robin=False)
    # A: 2-0 among tied (1.0); B: 0-1 (0.0); C: 0-1 (0.0) -- B and C tied at the bottom.
    assert result == [["A"], ["B", "C"]]


def test_sub_group_record_true_tie_returns_single_group_not_none():
    # A cycle: A beats B, B beats C, C beats A -- every team 1-1 among tied, a REAL computed tie.
    rows = _game("A", 20, "B", 10) + _game("B", 20, "C", 10) + _game("C", 20, "A", 10)
    ctx = _ctx(rows)
    result = sub_group_record(["A", "B", "C"], ctx, require_round_robin=True)
    assert result == [["A", "B", "C"]]


def test_sub_group_record_fewer_than_three_is_none():
    rows = _game("A", 20, "B", 10)
    assert sub_group_record(["A", "B"], _ctx(rows)) is None


# ===========================================================================
# 3. sweep_in_out
# ===========================================================================
def test_sweep_in_out_promotes_sweeper_and_demotes_total_losers():
    """A beat all three others, D lost to all three others, and B vs C was never played -- so the
    round robin is incomplete and both the promote and demote halves fire."""
    rows = (
        _game("A", 20, "B", 10) + _game("A", 20, "C", 10) + _game("A", 20, "D", 10)
        + _game("B", 20, "D", 10) + _game("C", 20, "D", 10)
    )
    ctx = _ctx(rows)
    assert sweep_in_out(["A", "B", "C", "D"], ctx) == [["A"], ["B", "C"], ["D"]]


def test_sweep_in_out_needs_a_result_against_every_other_tied_team():
    """A beat the two tied teams it played but never played D, so it has NOT "defeated each of
    the other Tied Teams" (acc.txt) / "beat all the other tied teams" (sec.txt). Nobody in this
    shape qualifies, so the step must decline.

    The looser reading -- swept everyone it happened to play -- promotes A on two games out of a
    possible three and is what this test exists to forbid."""
    rows = _game("A", 20, "B", 10) + _game("A", 20, "C", 10)
    ctx = _ctx(rows)
    assert sweep_in_out(["A", "B", "C", "D"], ctx) is None


def test_sweep_in_out_is_silent_on_the_real_2025_acc_five_way():
    """The regression case, from live data. Georgia Tech played exactly one of the other four
    tied teams (Duke) and won it; Duke played exactly one (Georgia Tech) and lost it; the other
    three pairs never met. Under "beat all the ones it played", Georgia Tech is promoted and Duke
    demoted to last on the strength of one game among five teams -- and Duke actually won this
    tie, four steps later, on its opponents' combined conference record (32-32, .500).

    So this step must have no opinion here."""
    five = ["Duke", "Georgia Tech", "Miami", "Pittsburgh", "SMU"]
    ctx = _ctx(_game("Georgia Tech", 27, "Duke", 20))
    assert sweep_in_out(five, ctx) is None


def test_sweep_in_out_promote_only_leaves_the_total_loser_in_the_middle():
    """`sides="promote_only"` is what five of the ten documents actually publish: a lone sweeper
    advances and nothing whatsoever is said about a team that lost to everyone. Demoting one
    anyway would order those conferences by a rule they never wrote down.

    Same fixture as the both-sides test above, so the ONLY difference is the parameter: A still
    goes top, but D must land in the middle group with B and C instead of alone at the bottom."""
    rows = (
        _game("A", 20, "B", 10) + _game("A", 20, "C", 10) + _game("A", 20, "D", 10)
        + _game("B", 20, "D", 10) + _game("C", 20, "D", 10)
    )
    ctx = _ctx(rows)
    assert sweep_in_out(["A", "B", "C", "D"], ctx, sides="both") == [["A"], ["B", "C"], ["D"]]
    assert sweep_in_out(["A", "B", "C", "D"], ctx, sides="promote_only") == [["A"], ["B", "C", "D"]]


def test_sweep_in_out_promote_only_still_declines_when_nobody_swept():
    """Dropping the demote half must not turn "no opinion" into an opinion: with no sweeper, the
    step returns None under either setting rather than partitioning on the demote side alone."""
    rows = _game("A", 20, "B", 10) + _game("B", 20, "C", 10) + _game("C", 20, "A", 10)
    ctx = _ctx(rows, )
    assert sweep_in_out(["A", "B", "C", "D"], ctx, sides="promote_only") is None


def test_sweep_in_out_rejects_an_unknown_sides_value():
    """A typo in a config must not silently fall through to the more aggressive both-sides
    behaviour, which would demote a team on a rule its conference never published."""
    ctx = _ctx(_game("A", 20, "B", 10))
    with pytest.raises(ValueError, match="sides"):
        sweep_in_out(["A", "B", "C"], ctx, sides="promote-only")     # hyphen, not underscore


def test_sweep_in_out_no_separation_returns_none():
    # A 4-cycle, each team plays exactly 2 of 3 possible tied opponents (incomplete), and every
    # team both won and lost once -- nobody swept, nobody was swept.
    rows = (
        _game("A", 20, "B", 10)
        + _game("B", 20, "C", 10)
        + _game("C", 20, "D", 10)
        + _game("D", 20, "A", 10)
    )
    ctx = _ctx(rows)
    assert sweep_in_out(["A", "B", "C", "D"], ctx) is None


def test_sweep_in_out_none_when_round_robin_is_complete():
    # Complete round robin among 3 -- sub_group_record's job, not sweep_in_out's; even though A
    # incidentally beat everyone it played, the precondition (incomplete) fails.
    rows = _game("A", 20, "B", 10) + _game("A", 20, "C", 10) + _game("B", 20, "C", 10)
    ctx = _ctx(rows)
    assert sweep_in_out(["A", "B", "C"], ctx) is None


def test_sweep_in_out_fewer_than_three_is_none():
    rows = _game("A", 20, "B", 10)
    assert sweep_in_out(["A", "B"], _ctx(rows)) is None


# ===========================================================================
# 4. common_opponents_record
# ===========================================================================
def test_common_opponents_record_acc_gate_returns_none():
    """Real 2025 ACC five-way tie: the common set is exactly one team (Syracuse, beaten by all
    five) -- below the default min_sample of 2, so this step must have no opinion."""
    teams = ["Duke", "Georgia Tech", "Miami", "Pittsburgh", "SMU"]
    rows = []
    for t in teams:
        rows += _game(t, 30, "Syracuse", 10)
        rows += _game(t, 20, f"{t} Only Opponent", 15)  # breaks any other accidental commonality
    ctx = _ctx(rows)
    assert common_opponents_record(teams, ctx) is None


def test_common_opponents_record_mac_three_way_partition():
    """Real 2025 MAC case: Miami (OH) 3-0, Toledo 2-1, Ohio 1-2 against the common opponents
    Ball State, Northern Illinois, Western Michigan."""
    rows = (
        _game("Miami (OH)", 30, "Ball State", 10)
        + _game("Miami (OH)", 30, "Northern Illinois", 10)
        + _game("Miami (OH)", 30, "Western Michigan", 10)
        + _game("Toledo", 30, "Ball State", 10)
        + _game("Toledo", 30, "Northern Illinois", 10)
        + _game("Toledo", 10, "Western Michigan", 30)
        + _game("Ohio", 30, "Ball State", 10)
        + _game("Ohio", 10, "Northern Illinois", 30)
        + _game("Ohio", 10, "Western Michigan", 30)
    )
    ctx = _ctx(rows)
    result = common_opponents_record(["Miami (OH)", "Toledo", "Ohio"], ctx, min_sample=2)
    assert result == [["Miami (OH)"], ["Toledo"], ["Ohio"]]


def test_common_opponents_record_respects_custom_min_sample():
    rows = _game("A", 30, "Z", 10) + _game("B", 30, "Z", 10)
    ctx = _ctx(rows)
    assert common_opponents_record(["A", "B"], ctx, min_sample=2) is None
    assert common_opponents_record(["A", "B"], ctx, min_sample=1) is not None


# ===========================================================================
# 5. vs_placed_opponents
# ===========================================================================
def test_vs_placed_opponents_no_common_opponents_is_none():
    rows = _game("X", 30, "OnlyX", 10) + _game("Y", 30, "OnlyY", 10)
    assert vs_placed_opponents(["X", "Y"], _ctx(rows)) is None


def test_vs_placed_opponents_single_best_opponent_separates():
    rows = _game("X", 30, "Z", 10) + _game("Y", 10, "Z", 30)
    ctx = _ctx(rows, frozen_order=["Z"], conf_records={"Z": (5, 3)})
    result = vs_placed_opponents(["X", "Y"], ctx)
    assert result == [["X"], ["Y"]]


def test_vs_placed_opponents_head_to_head_then_combine_resolves_tied_opponents():
    """sec.txt C.1: opponents P and Q are themselves tied in the standings; P beat Q
    head-to-head, so the position resolves to P alone -- which DOES separate X and Y, even
    though their combined record against {P, Q} is even."""
    rows = (
        _game("X", 30, "P", 10)      # X beat P
        + _game("X", 10, "Q", 30)    # X lost to Q
        + _game("Y", 10, "P", 30)    # Y lost to P
        + _game("Y", 30, "Q", 10)    # Y beat Q
        + _game("P", 20, "Q", 17)    # P beat Q head-to-head
    )
    ctx = _ctx(
        rows,
        frozen_order=["P", "Q"],
        conf_records={"P": (5, 3), "Q": (5, 3)},
    )
    result = vs_placed_opponents(["X", "Y"], ctx, tied_opponent_handling="head_to_head_then_combine")
    assert result == [["X"], ["Y"]]


def test_vs_placed_opponents_combine_mode_keeps_tied_opponents_together():
    """big12.txt step c: no head-to-head sub-step -- P and Q stay combined even though P beat Q
    head-to-head, and X/Y's combined record against {P, Q} is even, so no separation."""
    rows = (
        _game("X", 30, "P", 10)
        + _game("X", 10, "Q", 30)
        + _game("Y", 10, "P", 30)
        + _game("Y", 30, "Q", 10)
        + _game("P", 20, "Q", 17)
    )
    ctx = _ctx(
        rows,
        frozen_order=["P", "Q"],
        conf_records={"P": (5, 3), "Q": (5, 3)},
    )
    result = vs_placed_opponents(["X", "Y"], ctx, tied_opponent_handling="combine")
    assert result is None


def test_vs_placed_opponents_exhausts_all_common_opponents_by_default():
    rows = (
        _game("X", 30, "R", 10) + _game("Y", 30, "R", 10)   # tied at R
        + _game("X", 30, "S", 10) + _game("Y", 10, "S", 30)  # separates at S
    )
    ctx = _ctx(
        rows,
        frozen_order=["R", "S"],
        conf_records={"R": (6, 2), "S": (4, 4)},
    )
    result = vs_placed_opponents(["X", "Y"], ctx)  # exhaust_all_opponents=True default
    assert result == [["X"], ["Y"]]


def test_vs_placed_opponents_stops_at_first_position_when_not_exhausting():
    rows = (
        _game("X", 30, "R", 10) + _game("Y", 30, "R", 10)
        + _game("X", 30, "S", 10) + _game("Y", 10, "S", 30)
    )
    ctx = _ctx(
        rows,
        frozen_order=["R", "S"],
        conf_records={"R": (6, 2), "S": (4, 4)},
    )
    result = vs_placed_opponents(["X", "Y"], ctx, exhaust_all_opponents=False)
    assert result is None


def test_vs_placed_opponents_invalid_mode_raises():
    with pytest.raises(ValueError):
        vs_placed_opponents(["X", "Y"], _ctx([]), tied_opponent_handling="bogus")


def test_vs_placed_opponents_direction_descending_is_harmless_noop():
    rows = _game("X", 30, "Z", 10) + _game("Y", 10, "Z", 30)
    ctx = _ctx(rows, frozen_order=["Z"], conf_records={"Z": (5, 3)})
    result = vs_placed_opponents(["X", "Y"], ctx, direction="descending")
    assert result == [["X"], ["Y"]]


def test_vs_placed_opponents_unsupported_direction_raises():
    with pytest.raises(ValueError):
        vs_placed_opponents(["X", "Y"], _ctx([]), direction="ascending")


def test_vs_placed_opponents_advance_on_unequal_games_skips_a_mismatched_position():
    """cusa.txt section D, which CUSA alone states: if the tied teams "PLAYED AN UNEQUAL NUMBER OF
    GAMES against the teams within the tied group, immediately advance to the team(s) with the
    next highest conference winning percentage."

    Best-placed common opponent is Top. A played it twice (1-1, .500); B played it once and won
    (1.000). Default behaviour compares those percentages and hands the position to B. With the
    CUSA rule the position is SKIPPED as incomparable, and the next position down -- Mid, which
    both played once -- decides it, where A won and B lost. So the two settings give opposite
    answers on the same data, which is what makes this discriminating."""
    rows = (
        _game("A", 20, "Top", 10) + _game("Top", 20, "A", 10)     # A: 1-1 vs Top
        + _game("B", 20, "Top", 10)                                # B: 1-0 vs Top
        + _game("A", 20, "Mid", 10)                                # A: 1-0 vs Mid
        + _game("Mid", 20, "B", 10)                                # B: 0-1 vs Mid
    )
    # Top outranks Mid in the frozen order, and both are common opponents of A and B.
    ctx = _ctx(rows, frozen_order=["Top", "Mid", "A", "B"])
    assert vs_placed_opponents(["A", "B"], ctx) == [["B"], ["A"]]
    assert vs_placed_opponents(["A", "B"], ctx, advance_on_unequal_games=True) == [["A"], ["B"]]


# ===========================================================================
# 6. opponents_cumulative_conf_pct
# ===========================================================================
def test_opponents_cumulative_conf_pct_acc_numbers():
    """Real 2025 ACC five-way tie, the step that actually decided it: Duke's eight ACC
    opponents went a combined 32-32 (.500, best); Georgia Tech and Miami 28-36 (.4375);
    Pittsburgh and SMU 27-37 (.4219)."""
    duke_opps = {f"D{i}": (4, 4) for i in range(1, 9)}                     # 32-32
    gt_opps = {f"G{i}": (4, 4) for i in range(1, 8)}
    gt_opps["G8"] = (0, 8)                                                  # 28-36
    miami_opps = {f"M{i}": (4, 4) for i in range(1, 8)}
    miami_opps["M8"] = (0, 8)                                               # 28-36
    pitt_opps = {f"P{i}": (4, 4) for i in range(1, 7)}
    pitt_opps["P7"] = (3, 5)
    pitt_opps["P8"] = (0, 8)                                                # 27-37
    smu_opps = {f"S{i}": (4, 4) for i in range(1, 7)}
    smu_opps["S7"] = (3, 5)
    smu_opps["S8"] = (0, 8)                                                 # 27-37

    conf_records = {**duke_opps, **gt_opps, **miami_opps, **pitt_opps, **smu_opps}

    def opp_rows(team, opp_names):
        return [_row(team, opp, "win", 21, 14) for opp in opp_names]

    rows = (
        opp_rows("Duke", duke_opps)
        + opp_rows("Georgia Tech", gt_opps)
        + opp_rows("Miami", miami_opps)
        + opp_rows("Pittsburgh", pitt_opps)
        + opp_rows("SMU", smu_opps)
    )
    ctx = _ctx(rows, conf_records=conf_records)
    tied = ["Duke", "Georgia Tech", "Miami", "Pittsburgh", "SMU"]
    result = opponents_cumulative_conf_pct(tied, ctx)
    assert result == [["Duke"], ["Georgia Tech", "Miami"], ["Pittsburgh", "SMU"]]


def test_opponents_cumulative_conf_pct_unbalanced_schedule_gated_by_default():
    # A played 2 conference opponents, B played 1 -- a count mismatch this step is silent on
    # unless explicitly told to ignore it.
    conf_records = {"OppA1": (4, 4), "OppA2": (4, 4), "OppB1": (8, 0)}
    rows = [
        _row("A", "OppA1", "win"),
        _row("A", "OppA2", "win"),
        _row("B", "OppB1", "win"),
    ]
    ctx = _ctx(rows, conf_records=conf_records)
    assert opponents_cumulative_conf_pct(["A", "B"], ctx) is None
    result = opponents_cumulative_conf_pct(["A", "B"], ctx, ignore_opponent_count_mismatch=True)
    assert result == [["B"], ["A"]]  # A: 8-8 = .500; B: 8-0 = 1.000 -- B ranks first


# ===========================================================================
# 7. capped_relative_scoring_margin
# ===========================================================================
def test_capped_relative_scoring_margin_sec_appendix_a_worked_example():
    """sec.txt Appendix A's own worked example, pinned exactly: Team A defeats Team B 31-28;
    Team B averaged 24 scored / 21 allowed for the season -> Team A's margin is +30.9."""
    rows = (
        _game("A", 31, "B", 28)                # Week 1 meeting
        + [_row("B", "C", "win", 20, 11)]        # B's other game -> season avg 24 scored/21 allowed
    )
    ctx = _ctx(rows)
    margin = _capped_relative_margin_for_team(ctx, "A")
    assert margin == pytest.approx(30.9, abs=0.05)


def test_capped_relative_scoring_margin_offense_capped_at_200():
    rows = (
        _game("X", 100, "Y", 0)
        # Y's season: 3 games total (100 allowed vs X, 0 allowed in each of two others) ->
        # avg allowed = 100/3 = 33.3, avg scored = 20/3 = 6.7.
        + [_row("Y", "Z1", "win", 10, 0), _row("Y", "Z2", "win", 10, 0)]
    )
    ctx = _ctx(rows)
    margin = _capped_relative_margin_for_team(ctx, "X")
    # raw offense would be (100/33.3)*100 = ~300% -- capped to 200; defense (0/6.7)*100 = 0.
    assert margin == pytest.approx(200.0, abs=0.05)


def test_capped_relative_scoring_margin_public_function_partitions_with_unscored_team():
    rows = _game("A", 31, "B", 28) + [_row("B", "C", "win", 20, 11)]
    ctx = _ctx(rows)
    result = capped_relative_scoring_margin(["A", "D"], ctx)  # D has no conference games at all
    assert result == [["A"], ["D"]]


def test_capped_relative_scoring_margin_offense_cap_is_configurable():
    rows = (
        _game("X", 100, "Y", 0)
        + [_row("Y", "Z1", "win", 10, 0), _row("Y", "Z2", "win", 10, 0)]
    )
    ctx = _ctx(rows)
    default_margin = _capped_relative_margin_for_team(ctx, "X")
    custom_margin = _capped_relative_margin_for_team(ctx, "X", offense_cap=150.0)
    assert default_margin == pytest.approx(200.0, abs=0.05)
    assert custom_margin == pytest.approx(150.0, abs=0.05)
    assert custom_margin != default_margin


# ===========================================================================
# 8. total_wins_capped
# ===========================================================================
def test_total_wins_capped_uncapped_total_and_documents_limitation():
    rows = [
        _row("A", "O1", "win"), _row("A", "O2", "win"), _row("A", "O3", "win"),
        _row("A", "O4", "loss"), _row("A", "O5", "loss"),  # losses must NOT be counted as wins
        _row("B", "O1", "win"), _row("B", "O2", "win"), _row("B", "O3", "win"), _row("B", "O4", "win"),
        _row("B", "O5", "win"),
    ]
    ctx = _ctx(rows)
    assert total_wins_capped(["A", "B"], ctx) == [["B"], ["A"]]


def test_total_wins_capped_equal_wins_is_single_group():
    rows = [_row("A", "O1", "win"), _row("B", "O1", "win")]
    ctx = _ctx(rows)
    assert total_wins_capped(["A", "B"], ctx) == [["A", "B"]]


def test_total_wins_capped_max_games_clamps_the_total():
    rows = [_row("A", f"O{i}", "win") for i in range(1, 15)]  # 14 wins, unrealistic but defensive
    ctx = _ctx(rows)
    assert total_wins_capped(["A"], ctx, max_games=12) == [["A"]]  # still a single group of one
    # Clamp is visible via a second team that would otherwise be strictly fewer wins.
    rows2 = rows + [_row("B", f"P{i}", "win") for i in range(1, 13)]  # B: exactly 12 wins
    ctx2 = _ctx(rows2)
    assert total_wins_capped(["A", "B"], ctx2) == [["A"], ["B"]]  # unclamped: 14 > 12
    assert total_wins_capped(["A", "B"], ctx2, max_games=12) == [["A", "B"]]  # clamped: 12 == 12


def test_total_wins_capped_excludes_postseason_and_championship_games():
    """The Big 12's step e is a STANDINGS measure, so bowl wins and the conference title game
    must not count toward it -- the same rule its sibling overall_win_pct and the engine's
    fallback both apply. It did not, until an independent review caught the inconsistency.

    A and B are each 1-0 in the regular season. A then wins the title game and a bowl, which
    would put it ahead if either exclusion were missing; with both applied they stay level."""
    rows = [
        _row("A", "O1", "win"), _row("B", "P1", "win"),
        _row("A", "Champ", "win", game_id=777),
        _row("A", "Bowl", "win", season_type="postseason"),
    ]
    ctx = _ctx(rows, placement_excluded_game_ids=frozenset({777}))
    assert total_wins_capped(["A", "B"], ctx) == [["A", "B"]]
    # Without the championship-game exclusion A pulls ahead, which is what makes this an
    # assertion about the filter rather than about the fixture.
    unguarded = _ctx(rows)
    assert total_wins_capped(["A", "B"], unguarded) == [["A"], ["B"]]


def test_conference_game_rows_exclude_the_championship_game():
    """_conf_game_rows feeds eight primitives -- head-to-head, intra-group record, sweeps,
    round-robin detection, common opponents, placed opponents, opponents' records and divisional
    record -- so a title game counted here reaches every one of them. TiebreakContext's own field
    documents those ids as "game_ids that must not count toward STANDINGS PLACEMENT".

    A and B split their season: B won in the regular season, A won the title game. Head-to-head
    must read as B's win, not as a 1-1 wash."""
    rows = (
        _game("B", 21, "A", 14)
        + [
            {"season": SEASON, "team": "A", "opponent": "B", "status": "win",
             "conference_game": True, "season_type": "regular", "game_id": 777,
             "team_score": 28, "opp_score": 10},
            {"season": SEASON, "team": "B", "opponent": "A", "status": "loss",
             "conference_game": True, "season_type": "regular", "game_id": 777,
             "team_score": 10, "opp_score": 28},
        ]
    )
    ctx = _ctx(rows, placement_excluded_game_ids=frozenset({777}))
    assert head_to_head(["A", "B"], ctx) == [["B"], ["A"]]
    # Counting the title game makes it a 1-1 split, which head_to_head correctly reads as a wash
    # -- so the exclusion is the only thing standing between "B won the tie" and "nobody did".
    assert head_to_head(["A", "B"], _ctx(rows)) is None


def test_total_wins_capped_without_the_roster_declines_rather_than_reporting_uncapped(caplog):
    """cap_fcs_wins=True with no non_fbs_teams roster must return None -- the engine's "this step
    has no opinion" signal, which the driver skips. The failure mode this guards against is
    returning an UNCAPPED total that looks like a capped one, which would silently order a Big 12
    tie by the wrong number."""
    ctx = _ctx([_row("A", "O1", "win")])
    with caplog.at_level(logging.WARNING, logger="cfb_lp"):
        assert total_wins_capped(["A"], ctx, cap_fcs_wins=True) is None
    assert "non_fbs_teams" in caplog.text


def test_total_wins_capped_counts_only_one_fcs_win():
    """The Big 12's step e: 'Only one win against a team from the NCAA Football Championship
    Subdivision or lower division will be counted annually.' A has 3 wins but two are over FCS
    opponents, so it counts 2; B has 2 wins over FBS opponents and counts 2. They tie."""
    rows = [
        _row("A", "FCS1", "win"), _row("A", "FCS2", "win"), _row("A", "FBS1", "win"),
        _row("B", "FBS2", "win"), _row("B", "FBS3", "win"),
    ]
    roster = frozenset({"FCS1", "FCS2"})
    ctx = _ctx(rows, non_fbs_teams=roster)
    assert total_wins_capped(["A", "B"], ctx) == [["A"], ["B"]]          # uncapped: 3 > 2
    assert total_wins_capped(["A", "B"], ctx, cap_fcs_wins=True) == [["A", "B"]]  # capped: 2 == 2


def test_total_wins_capped_keeps_the_first_fcs_win():
    """The cap strikes only the SECOND and later FCS wins -- one is counted, not zero. Without
    this, a team with a single FCS win would be under-credited by one."""
    rows = [_row("A", "FCS1", "win"), _row("A", "FBS1", "win"), _row("B", "FBS2", "win")]
    ctx = _ctx(rows, non_fbs_teams=frozenset({"FCS1"}))
    assert total_wins_capped(["A", "B"], ctx, cap_fcs_wins=True) == [["A"], ["B"]]


def test_total_wins_capped_empty_roster_is_supplied_not_missing():
    """An empty frozenset means the roster WAS loaded and nobody here played a non-FBS team, so
    the step computes normally. Only None means it was never loaded. Collapsing the two would
    make a legitimately FCS-free conference silently skip the step."""
    rows = [_row("A", "O1", "win"), _row("A", "O2", "win"), _row("B", "O3", "win")]
    ctx = _ctx(rows, non_fbs_teams=frozenset())
    assert total_wins_capped(["A", "B"], ctx, cap_fcs_wins=True) == [["A"], ["B"]]


# ===========================================================================
# 9. external_ranking
# ===========================================================================
def test_external_ranking_gated_below_min_conference_games():
    conf_records = {"A": (2, 1), "B": (5, 3)}
    ctx = _ctx([], conf_records=conf_records, team_ranks={"A": 3, "B": 10})
    assert external_ranking(["A", "B"], ctx, min_conference_games=5) is None


def test_external_ranking_orders_ascending_unranked_last():
    conf_records = {"A": (5, 3), "B": (5, 3), "C": (5, 3)}
    ctx = _ctx([], conf_records=conf_records, team_ranks={"A": 10, "B": 3})
    result = external_ranking(["A", "B", "C"], ctx, min_conference_games=0)
    assert result == [["B"], ["A"], ["C"]]  # C is unranked (missing from team_ranks) -> last


# ===========================================================================
# 10. random_draw
# ===========================================================================
def test_random_draw_always_none():
    assert random_draw(["A", "B", "C"], _ctx([])) is None


# ===========================================================================
# overall_win_pct
# ===========================================================================
def _nonconf(team, opponent, status, season=SEASON, season_type="regular", game_id=None):
    return {
        "season": season, "team": team, "opponent": opponent, "status": status,
        "conference_game": False, "season_type": season_type, "week": 1,
        "game_id": game_id, "team_score": None, "opp_score": None,
    }


def test_overall_win_pct_plain_is_the_american_variant():
    """american.txt 10.5.9: plain overall percentage, conference and non-conference. A is 3-1
    (.750), B is 2-2 (.500)."""
    rows = [
        _nonconf("A", "O1", "win"), _nonconf("A", "O2", "win"),
        _nonconf("A", "O3", "win"), _nonconf("A", "O4", "loss"),
        _nonconf("B", "P1", "win"), _nonconf("B", "P2", "win"),
        _nonconf("B", "P3", "loss"), _nonconf("B", "P4", "loss"),
    ]
    assert overall_win_pct(["A", "B"], _ctx(rows)) == [["A"], ["B"]]


def test_overall_win_pct_fcs_cap_is_the_mountain_west_variant():
    """mountainwest.txt two-team 3: "a maximum of ONE win against a team from the NCAA Football
    Championship Subdivision shall be included". A's 3-0 includes two FCS wins, so it counts 2-0
    (1.000); B is 2-0 over FBS teams (1.000). Uncapped A looks better on volume; capped they are
    level and the step separates nobody."""
    rows = [
        _nonconf("A", "FCS1", "win"), _nonconf("A", "FCS2", "win"), _nonconf("A", "FBS1", "win"),
        _nonconf("B", "FBS2", "win"), _nonconf("B", "FBS3", "win"),
    ]
    ctx = _ctx(rows, non_fbs_teams=frozenset({"FCS1", "FCS2"}))
    # Both 1.000 either way here, so use a loss to make the cap visible in the percentage.
    rows2 = rows + [_nonconf("A", "FBS9", "loss"), _nonconf("B", "FBS8", "loss")]
    ctx2 = _ctx(rows2, non_fbs_teams=frozenset({"FCS1", "FCS2"}))
    assert overall_win_pct(["A", "B"], ctx2) == [["A"], ["B"]]          # 3-1 .750 vs 2-1 .667
    assert overall_win_pct(["A", "B"], ctx2, fcs_win_cap=1) == [["A", "B"]]   # 2-1 vs 2-1


def test_overall_win_pct_fcs_cap_does_not_forgive_a_loss_to_an_fcs_team():
    """The rule caps what a team can bank from playing down; it says nothing about excusing a
    LOSS to an FCS opponent. A cap that dropped such losses too would quietly reward the worst
    result in college football."""
    rows = [
        _nonconf("A", "FCS1", "win"), _nonconf("A", "FCS2", "loss"),
        _nonconf("B", "FBS1", "win"),
    ]
    ctx = _ctx(rows, non_fbs_teams=frozenset({"FCS1", "FCS2"}))
    assert overall_win_pct(["A", "B"], ctx, fcs_win_cap=1) == [["B"], ["A"]]   # 1.000 vs .500


def test_overall_win_pct_fbs_only_is_the_sun_belt_variant():
    """sunbelt.txt step 9: overall percentage "against FBS teams" -- the non-FBS game is dropped
    entirely, win or loss, rather than capped. A is 1-1 against FBS (.500) once its FCS win is
    removed; B is 2-0 (1.000)."""
    rows = [
        _nonconf("A", "FCS1", "win"), _nonconf("A", "FBS1", "win"), _nonconf("A", "FBS2", "loss"),
        _nonconf("B", "FBS3", "win"), _nonconf("B", "FBS4", "win"),
    ]
    ctx = _ctx(rows, non_fbs_teams=frozenset({"FCS1"}))
    assert overall_win_pct(["A", "B"], ctx, fbs_only=True) == [["B"], ["A"]]
    # Counting the FCS win would make A 2-1 (.667) -- still behind, so assert the difference
    # against the uncapped call directly to prove the parameter did something.
    assert overall_win_pct(["A", "B"], ctx) == [["B"], ["A"]]
    plain = overall_win_pct(["A"], ctx)
    fbs = overall_win_pct(["A"], ctx, fbs_only=True)
    assert plain == fbs == [["A"]]          # single team: shape only, values asserted above


def test_overall_win_pct_excludes_postseason_and_championship_games():
    """Bowl and playoff results must not affect standings placement, and a championship game is a
    'regular' row in CFBD's data so the caller passes its game_id.

    The fixture is built so the two readings give OPPOSITE answers. A is 1-1 in the regular
    season (.500) against B's 3-2 (.600), so B leads on placement record -- but A also won the
    title game and a bowl, which would make it 3-1 (.750) and put it AHEAD if the exclusions
    stopped working. An earlier version of this test used a fixture where the favoured team led
    under both readings, which proved nothing about the exclusions at all.
    """
    rows = [
        _nonconf("A", "O1", "win"), _nonconf("A", "O2", "loss"),
        _nonconf("A", "Champ", "win", game_id=777),                       # excluded by game_id
        _nonconf("A", "Bowl", "win", season_type="postseason"),           # excluded by type
        _nonconf("B", "P1", "win"), _nonconf("B", "P2", "win"), _nonconf("B", "P3", "win"),
        _nonconf("B", "P4", "loss"), _nonconf("B", "P5", "loss"),
    ]
    ctx = _ctx(rows, placement_excluded_game_ids=frozenset({777}))
    assert overall_win_pct(["A", "B"], ctx) == [["B"], ["A"]]             # .500 vs .600


def test_overall_win_pct_championship_exclusion_is_load_bearing_on_its_own():
    """A championship game is season_type 'regular', so season_type alone cannot exclude it --
    only the caller's game_id set can. A is 1-1 (.500) versus B's 3-2 (.600) with the exclusion,
    and 2-1 (.667) versus .600 without it, so dropping this one guard inverts the result."""
    rows = [
        _nonconf("A", "O1", "win"), _nonconf("A", "O2", "loss"),
        _nonconf("A", "Champ", "win", game_id=777),
        _nonconf("B", "P1", "win"), _nonconf("B", "P2", "win"), _nonconf("B", "P3", "win"),
        _nonconf("B", "P4", "loss"), _nonconf("B", "P5", "loss"),
    ]
    with_guard = _ctx(rows, placement_excluded_game_ids=frozenset({777}))
    without_guard = _ctx(rows)
    assert overall_win_pct(["A", "B"], with_guard) == [["B"], ["A"]]      # .500 vs .600
    assert overall_win_pct(["A", "B"], without_guard) == [["A"], ["B"]]   # .667 vs .600


def test_overall_win_pct_postseason_exclusion_is_load_bearing_on_its_own():
    """The season_type guard, isolated: no excluded ids are passed at all, so only it can keep
    the bowl win out. Same .500-versus-.600 shape, inverting to .667 if the guard is dropped."""
    rows = [
        _nonconf("A", "O1", "win"), _nonconf("A", "O2", "loss"),
        _nonconf("A", "Bowl", "win", season_type="postseason"),
        _nonconf("B", "P1", "win"), _nonconf("B", "P2", "win"), _nonconf("B", "P3", "win"),
        _nonconf("B", "P4", "loss"), _nonconf("B", "P5", "loss"),
    ]
    assert overall_win_pct(["A", "B"], _ctx(rows)) == [["B"], ["A"]]      # .500 vs .600


def test_overall_win_pct_declines_without_the_roster_when_an_adjustment_is_requested():
    """Same contract as total_wins_capped: no roster means no opinion, rather than an unadjusted
    percentage presented as an adjusted one. Plain mode needs no roster and still works."""
    rows = [_nonconf("A", "O1", "win"), _nonconf("B", "P1", "loss")]
    ctx = _ctx(rows)                       # non_fbs_teams is None
    assert overall_win_pct(["A", "B"], ctx, fcs_win_cap=1) is None
    assert overall_win_pct(["A", "B"], ctx, fbs_only=True) is None
    assert overall_win_pct(["A", "B"], ctx) == [["A"], ["B"]]


# ===========================================================================
# divisional_record, and the division-scoped modes of the shared primitives
# ===========================================================================
# A Sun Belt-shaped conference: two divisions, and the tied pair sits inside one of them.
DIVISIONS = {
    "A": "East", "B": "East", "InEast1": "East", "InEast2": "East",
    "W1": "West", "W2": "West", "W3": "West",
}


def test_divisional_record_counts_only_same_division_games():
    """sunbelt.txt step 2, "highest overall DIVISIONAL winning percentage". A and B are level on
    all conference games (2-1 each), but A went 2-0 inside the East while B went 1-1 there, so
    the divisional-only measure separates them where the primary key cannot.

    That is the whole reason the step exists: the Sun Belt decides a division champion on ALL
    conference games, divisional and not, so this narrower record breaks a tie in the broader one.
    """
    rows = (
        _game("A", 21, "InEast1", 14) + _game("A", 21, "InEast2", 14)   # A: 2-0 divisional
        + _game("W1", 21, "A", 14)                                       # A: 0-1 cross
        + _game("B", 21, "InEast1", 14) + _game("InEast2", 21, "B", 14)  # B: 1-1 divisional
        + _game("B", 21, "W1", 14)                                       # B: 1-0 cross
    )
    ctx = _ctx(rows, divisions=DIVISIONS)
    assert divisional_record(["A", "B"], ctx) == [["A"], ["B"]]          # 1.000 vs .500


def test_divisional_record_declines_across_divisions_or_without_a_map():
    """A cross-division comparison answers a question no rule asks, and an absent map must not be
    read as "everyone shares a division"."""
    rows = _game("A", 21, "InEast1", 14) + _game("W1", 21, "W2", 14)
    assert divisional_record(["A", "W1"], _ctx(rows, divisions=DIVISIONS)) is None
    assert divisional_record(["A", "B"], _ctx(rows)) is None             # no divisions supplied


def test_common_opponents_scope_non_divisional_is_the_sun_belt_variant():
    """sunbelt.txt step 4 asks for common NON-DIVISIONAL opponents only, because step 2 has
    already compared divisional records. Here the two common opponents sit in different
    divisions and give OPPOSITE answers, so the scope decides the result.

    A beat the East common opponent and lost to the West one; B did the reverse. Unscoped, the
    two cancel and nobody is separated; scoped to non-divisional, only the West game counts and B
    wins; scoped to divisional, only the East game counts and A wins."""
    rows = (
        _game("A", 21, "InEast1", 14) + _game("W1", 21, "A", 14)
        + _game("InEast1", 21, "B", 14) + _game("B", 21, "W1", 14)
    )
    ctx = _ctx(rows, divisions=DIVISIONS)
    assert common_opponents_record(["A", "B"], ctx, min_sample=2) == [["A", "B"]]   # .500 each
    assert common_opponents_record(
        ["A", "B"], ctx, min_sample=1, scope="non_divisional") == [["B"], ["A"]]
    assert common_opponents_record(
        ["A", "B"], ctx, min_sample=1, scope="divisional") == [["A"], ["B"]]


def test_common_opponents_scope_declines_rather_than_widening_to_all():
    """Without a usable division map a scoped call must return None, never quietly answer the
    unscoped question -- which is a different question from the one the conference asked."""
    rows = _game("A", 21, "X", 14) + _game("B", 21, "X", 14)
    assert common_opponents_record(
        ["A", "B"], _ctx(rows), min_sample=1, scope="non_divisional") is None


def test_common_opponents_rejects_an_unknown_scope():
    with pytest.raises(ValueError, match="scope"):
        common_opponents_record(["A", "B"], _ctx([]), scope="nondivisional")


def test_vs_placed_opponents_divisional_scope_walks_only_the_division():
    """sunbelt.txt step 3 walks the DIVISIONAL standings, not the conference standings. The
    best-placed common opponent overall is W1 (out of division) and it separates the pair one
    way; the best-placed DIVISIONAL common opponent is InEast1 and it separates them the other.
    So the scope flips the answer."""
    rows = (
        _game("W1", 21, "A", 14) + _game("A", 21, "InEast1", 14)
        + _game("B", 21, "W1", 14) + _game("InEast1", 21, "B", 14)
    )
    ctx = _ctx(
        rows,
        conf_records={"W1": (6, 0), "InEast1": (3, 3)},   # W1 places above InEast1
        frozen_order=["W1", "InEast1", "A", "B"],
        divisions=DIVISIONS,
    )
    assert vs_placed_opponents(["A", "B"], ctx) == [["B"], ["A"]]        # W1 decides: B won it
    assert vs_placed_opponents(
        ["A", "B"], ctx, standings_scope="divisional") == [["A"], ["B"]]  # InEast1: A won it


def test_vs_placed_opponents_rejects_an_unknown_standings_scope():
    with pytest.raises(ValueError, match="standings_scope"):
        vs_placed_opponents(["A", "B"], _ctx([]), standings_scope="division")


# ===========================================================================
# 11. conditional_external_ranking
# ===========================================================================
# _final_conference_week scopes to THIS conference's members via ctx.conf_records, so every
# cascade fixture must declare them -- ctx.rows is the whole league's grid in production, and a
# context with no members would legitimately find no final week at all.
_CASCADE_MEMBERS = {
    "A": (4, 2), "B": (4, 2),
    "Early1": (2, 4), "Early2": (2, 4), "FinalA": (2, 4), "FinalB": (2, 4),
}


def _cascade_rows(a_result, b_result, final_week=10):
    """A and B each play one earlier conference game plus a final-week game whose outcome is set
    per team. `None` means idle in the final week (a bye), which is the case that separates the
    two `condition` values."""
    rows = _row_pair("A", "Early1", "win", week=1) + _row_pair("B", "Early2", "win", week=1)
    for team, result in (("A", a_result), ("B", b_result)):
        if result is None:
            continue
        opp = f"Final{team}"
        rows += _row_pair(team, opp, result, week=final_week)
    return rows


def _cascade_ctx(rows, team_ranks, conf_records=None, placement_excluded_game_ids=frozenset()):
    return _ctx(
        rows,
        conf_records=conf_records or dict(_CASCADE_MEMBERS),
        team_ranks=team_ranks,
        placement_excluded_game_ids=placement_excluded_game_ids,
    )


def _row_pair(team, opponent, status, week):
    other = "loss" if status == "win" else "win"
    return [
        {"season": SEASON, "team": team, "opponent": opponent, "status": status,
         "conference_game": True, "season_type": "regular", "week": week,
         "team_score": None, "opp_score": None},
        {"season": SEASON, "team": opponent, "opponent": team, "status": other,
         "conference_game": True, "season_type": "regular", "week": week,
         "team_score": None, "opp_score": None},
    ]


def test_cascade_prefers_the_ranked_team_that_survived_the_final_weekend():
    """THE WHOLE POINT OF THE STEP. B is better rated, but B lost in the final weekend and A did
    not -- so A is selected despite the worse rating. A plain rating comparison returns the
    opposite order, which is what makes this discriminating."""
    rows = _cascade_rows(a_result="win", b_result="loss")
    ctx = _cascade_ctx(rows, {"A": 20, "B": 5})
    assert conditional_external_ranking(["A", "B"], ctx) == [["A"], ["B"]]
    assert external_ranking(["A", "B"], ctx) == [["B"], ["A"]]


def test_cascade_falls_back_to_the_rating_when_nobody_survives():
    """If every ranked tied team lost in the final weekend, the documents revert to a composite
    average over the whole group -- which under the K6 substitution is the same rating. So the
    step must behave exactly like external_ranking here, not return no opinion."""
    rows = _cascade_rows(a_result="loss", b_result="loss")
    ctx = _cascade_ctx(rows, {"A": 20, "B": 5})
    assert conditional_external_ranking(["A", "B"], ctx) == [["B"], ["A"]]


def test_cascade_falls_back_to_the_rating_when_everybody_survives():
    """The mirror case: survival that splits nobody is no information, so the group is ordered by
    rating alone rather than being reported as separated for the wrong reason."""
    rows = _cascade_rows(a_result="win", b_result="win")
    ctx = _cascade_ctx(rows, {"A": 20, "B": 5})
    assert conditional_external_ranking(["A", "B"], ctx) == [["B"], ["A"]]


def test_cascade_ignores_a_team_outside_the_ranked_cutoff():
    """"Was ranked going into the final weekend" is a real filter in the source documents -- the
    CFP poll holds 25 teams. A survived the final weekend but sits outside the cutoff, so it is
    not a ranked survivor and cannot be promoted over the better-rated B on that basis.

    Without the cutoff our rating would make EVERY team "ranked" and the whole cascade would
    collapse into a plain rating comparison, which is the failure this pins."""
    rows = _cascade_rows(a_result="win", b_result="loss")
    ctx = _cascade_ctx(rows, {"A": 90, "B": 5})
    assert conditional_external_ranking(["A", "B"], ctx, ranked_cutoff=25) == [["B"], ["A"]]
    # Widen the cutoff to include A and the survival condition decides instead.
    assert conditional_external_ranking(["A", "B"], ctx, ranked_cutoff=100) == [["A"], ["B"]]


def test_cascade_condition_splits_on_an_idle_final_weekend():
    """A bye is exactly where the two published wordings disagree. A is idle in the final week:
    it "does not lose" (american.txt 10.5.3) but does not "win" (mountainwest.txt 2(a)). B, worse
    rated, won. So the two settings give opposite answers on identical data."""
    rows = _cascade_rows(a_result=None, b_result="win")
    ctx = _cascade_ctx(rows, {"A": 5, "B": 20})
    # does_not_lose: both qualify -> nobody is split off -> ordered by rating, A first.
    assert conditional_external_ranking(
        ["A", "B"], ctx, condition="does_not_lose") == [["A"], ["B"]]
    # wins: only B qualifies -> B is promoted above the better-rated A.
    assert conditional_external_ranking(["A", "B"], ctx, condition="wins") == [["B"], ["A"]]


def test_cascade_keeps_teams_it_cannot_tell_apart_in_one_group():
    """Non-survivors it has no ranking for must stay together, not be exploded into singletons.

    An earlier version returned one singleton per team, which did two wrong things at once: it
    ordered UNRANKED teams relative to each other in whatever sequence they arrived in -- so the
    same data gave different answers depending on input order -- and, because each was a
    singleton, the driver recorded them as resolved and skipped every later step in the chain.
    """
    rows = _cascade_rows(a_result="win", b_result="loss")
    members = dict(_CASCADE_MEMBERS)
    members.update({"C": (4, 2), "D": (4, 2)})
    ranks = {"A": 5, "B": None, "C": None, "D": None}
    forwards = conditional_external_ranking(
        ["A", "B", "C", "D"], _cascade_ctx(rows, ranks, conf_records=members))
    backwards = conditional_external_ranking(
        ["A", "D", "C", "B"], _cascade_ctx(rows, ranks, conf_records=members))
    assert forwards == [["A"], ["B", "C", "D"]], forwards
    # The unranked group's membership must not depend on the order it was handed in.
    assert sorted(backwards[-1]) == ["B", "C", "D"], backwards
    assert len(backwards) == len(forwards) == 2


def test_cascade_is_gated_by_min_conference_games():
    rows = _cascade_rows(a_result="win", b_result="loss")
    ctx = _cascade_ctx(rows, {"A": 20, "B": 5}, conf_records={"A": (1, 1), "B": (1, 1)})
    assert conditional_external_ranking(["A", "B"], ctx, min_conference_games=4) is None


def test_cascade_declines_when_the_season_has_no_conference_games():
    """With no conference games there is no final weekend, so the condition is unanswerable. It
    must return None rather than silently degrading into a plain rating comparison -- otherwise
    an early-season tie would be decided by a condition nobody could have met."""
    ctx = _cascade_ctx([], {"A": 20, "B": 5})
    assert conditional_external_ranking(["A", "B"], ctx) is None


def test_cascade_rejects_an_unknown_condition():
    ctx = _cascade_ctx(_cascade_rows("win", "loss"), {"A": 1, "B": 2})
    with pytest.raises(ValueError, match="condition"):
        conditional_external_ranking(["A", "B"], ctx, condition="doesnt_lose")


def test_cascade_ignores_the_conference_championship_game_as_the_final_weekend():
    """A championship game must not be mistaken for the final weekend of the conference REGULAR
    season. It is excluded by its game_id, NOT by season_type -- this repo's own comment says "a
    championship game is always season_type=='regular'" (artifacts/schedule.py), so an earlier
    version of this test built it as 'postseason' and was therefore testing an input that cannot
    occur. It passed for the wrong reason and gave false assurance on exactly the shape that
    later turned out to be broken.

    Here A wins in week 10 and loses the week-15 title game. With the title game excluded the
    final week is 10, A survives it, and A is selected despite the worse rating."""
    rows = _cascade_rows(a_result="win", b_result="loss", final_week=10)
    rows += [
        {"season": SEASON, "team": "A", "opponent": "B", "status": "loss",
         "conference_game": True, "season_type": "regular", "week": 15, "game_id": 777,
         "team_score": None, "opp_score": None},
        {"season": SEASON, "team": "B", "opponent": "A", "status": "win",
         "conference_game": True, "season_type": "regular", "week": 15, "game_id": 777,
         "team_score": None, "opp_score": None},
    ]
    ctx = _cascade_ctx(rows, {"A": 20, "B": 5}, placement_excluded_game_ids=frozenset({777}))
    assert _final_conference_week(ctx) == 10
    assert conditional_external_ranking(["A", "B"], ctx) == [["A"], ["B"]]

    # Without the exclusion the final week becomes 15, where A LOST -- so A stops surviving and
    # the step hands the tie to the better-rated B. Opposite answer, same rows.
    unguarded = _cascade_ctx(rows, {"A": 20, "B": 5})
    assert _final_conference_week(unguarded) == 15
    assert conditional_external_ranking(["A", "B"], unguarded) == [["B"], ["A"]]


def test_final_conference_week_is_scoped_to_this_conference_and_to_played_games():
    """The bug an independent review caught, pinned. ctx.rows is the WHOLE LEAGUE's grid in
    production, including unplayed fixtures, so the final week must be derived only from THIS
    conference's PLAYED games.

    Unscoped, either of the two intruders below moves the final week past this conference's
    finale, every tied team reads as idle, and the whole cascade silently degrades into a plain
    rating comparison -- which never ties, and so pre-empts every remaining step in three
    conferences' chains."""
    rows = _cascade_rows(a_result="win", b_result="loss", final_week=10)
    intruders = [
        # Another conference, later week, already played.
        {"season": SEASON, "team": "OtherConfTeam", "opponent": "OtherConfFoe", "status": "win",
         "conference_game": True, "season_type": "regular", "week": 13, "game_id": None,
         "team_score": None, "opp_score": None},
        # This conference, later week, NOT YET PLAYED.
        {"season": SEASON, "team": "A", "opponent": "Early1", "status": "upcoming",
         "conference_game": True, "season_type": "regular", "week": 14, "game_id": None,
         "team_score": None, "opp_score": None},
    ]
    ctx = _cascade_ctx(rows + intruders, {"A": 20, "B": 5})
    assert _final_conference_week(ctx) == 10
    # And the condition still does its job: A survived week 10, B did not.
    assert conditional_external_ranking(["A", "B"], ctx) == [["A"], ["B"]]
    assert external_ranking(["A", "B"], ctx) == [["B"], ["A"]], (
        "the two must differ here, or this fixture proves nothing about the condition"
    )


# ===========================================================================
# STEP_REGISTRY
# ===========================================================================
def test_step_registry_exact_keys_and_mapping():
    assert set(STEP_REGISTRY) == {
        "head_to_head",
        "sub_group_record",
        "sweep_in_out",
        "common_opponents_record",
        "vs_placed_opponents",
        "opponents_cumulative_conf_pct",
        "capped_relative_scoring_margin",
        "total_wins_capped",
        "conditional_external_ranking",
        "overall_win_pct",
        "divisional_record",
        "external_ranking",
        "random_draw",
    }
    assert STEP_REGISTRY["head_to_head"] is head_to_head
    assert STEP_REGISTRY["sub_group_record"] is sub_group_record
    assert STEP_REGISTRY["sweep_in_out"] is sweep_in_out
    assert STEP_REGISTRY["common_opponents_record"] is common_opponents_record
    assert STEP_REGISTRY["vs_placed_opponents"] is vs_placed_opponents
    assert STEP_REGISTRY["opponents_cumulative_conf_pct"] is opponents_cumulative_conf_pct
    assert STEP_REGISTRY["capped_relative_scoring_margin"] is capped_relative_scoring_margin
    assert STEP_REGISTRY["total_wins_capped"] is total_wins_capped
    assert STEP_REGISTRY["external_ranking"] is external_ranking
    assert STEP_REGISTRY["random_draw"] is random_draw


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
