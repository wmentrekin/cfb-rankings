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
    sweep_in_out,
    total_wins_capped,
    vs_placed_opponents,
)

SEASON = 2025


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _row(team, opponent, status, team_score=None, opp_score=None, conference_game=True, season=SEASON):
    return {
        "season": season,
        "team": team,
        "opponent": opponent,
        "status": status,
        "conference_game": conference_game,
        "team_score": team_score,
        "opp_score": opp_score,
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
         non_fbs_teams=None):
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
