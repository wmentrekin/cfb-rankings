"""Tests for the head-to-head clinch and the complement elimination in
artifacts/schedule_standings.compute_conference_championship_status (clinch-h2h-and-ccg-ingest, T1).

The wins-only clinch test could not call a team clinched while a rival could still TIE its
worst-case win total, even when the team owns the head-to-head and the conference's first
two-team tiebreak step IS head-to-head. 2026 week 12 is the live case: Wake Forest (ACC) and San
Diego State (Pac-12) had both clinched a title-game berth but showed "possible" because Louisville
and Texas State could still tie them.

Fixtures are the 2026 standings after week 12, written as W-L-remaining conference records.

Covers:
  1. ACC (top_n 2): Miami 8-0-1 and Wake Forest 7-1-1 (head-to-head over Louisville 7-2-0) clinched;
     everyone else, Louisville included, eliminated.
  2. Pac-12 (top_n 2, 7-game round robin, all played): Boise State and San Diego State (head-to-head
     over Texas State) clinched; the rest eliminated.
  3. SEC: Georgia 8-1-0 clinched; Alabama / Tennessee / Texas (6-2-1 each) all stay "possible" --
     the rule must not fire for any of them.
  4. Negatives, each staying "possible": a third team that can finish on the same win total; a
     rival that can PASS (B_i > L_T); head-to-head lost / not played / split / a rematch still to
     come / one-sided data; unequal conference game counts (the rival, and any other team that
     could out-rank T on percentage); conference whose first step is not head_to_head or that has
     no rule set; the rule disabled by default (output identical to the wins-only math).
  5. Division pool (top_n=1): the rule is sound there too -- tested in both directions.
  6. Complement elimination composing with the played-championship-game override and its guard.
  7. The wiring: compute_standings builds the series from rows (honouring excluded games), and
     schedule._head_to_head_first_conferences reads the real tiebreaker config.

MUTATION-CHECK LOG (one defect at a time injected into artifacts/schedule_standings.py, this
file re-run, change reverted; "caught" = the named tests failed):
  b  rival picked by head-to-head instead of "B_i == L_T"      caught: test_a_rival_that_can_pass_blocks_it,
                                                               test_head_to_head_lost_blocks_it,
                                                               test_division_pool_without_the_head_to_head_stays_possible
  c  head-to-head check dropped                                caught (8 tests): lost / not played / split / rematch /
                                                               one-sided / division negatives, Pac-12 fixture
  c' rematch tolerated on the winner's own rows only           caught: test_a_won_series_with_a_rematch_still_scheduled_blocks_it
  d  pair game-count equality dropped                          caught: test_unequal_conference_game_counts_for_the_pair_block_it
  f  proof-of-a-two-team-tie loop dropped                      caught: test_an_unplaced_team_that_banked_exactly_the_floor_blocks_it,
                                                               test_a_short_schedule_team_that_could_out_rank_on_percentage_blocks_it,
                                                               test_a_team_that_could_reach_the_floor_on_a_different_game_count_blocks_it
  f' percentage clause of (f) dropped                          caught: test_a_short_schedule_team_that_could_out_rank_on_percentage_blocks_it
  complement elimination never applied                         caught (6 tests): ACC, Pac-12, division, title-game, wiring
  complement not skipped for a rejected participant set        caught: test_a_rejected_participant_set_also_blocks_the_complement_elimination
  WIDER TIE DEFINITIONS (review F1; block = _clear_of_other_game_counts, condition f2 + complement fence):
  f2 wins-range check removed                                  caught: test_clear_helper_wins_and_losses_ranges_each_block_on_their_own,
                                                               test_clear_helper_cusa_and_unknown_widen_... (unit level only: in the two
                                                               end-to-end counterexamples the losses range overlaps too, so it alone blocks)
  f2 losses-range check removed                                caught: ACC counterexample (test_acc_alternate_games_tie_blocks_...),
                                                               the helper unit test, the compute_standings and build_schedule_payload wiring tests
  f2 CUSA widening removed (widen = 0 always)                  caught: test_clear_helper_cusa_and_unknown_widen_the_wins_range_by_one_but_acc_does_not
                                                               (unit level: the widening is strictly redundant with the losses check for the
                                                               real CUSA rule, which needs EQUAL losses -- it only adds conservatism)
  unrecognised definition not treated as strictest             caught: same unit test; a name missing from a supplied mapping read as
                                                               win_pct caught: test_unknown_tie_definition_is_read_as_the_strictest_case
  whole f2 block removed                                       caught (5 tests): both counterexamples, unknown definition, both wiring tests
  complement fence removed                                     caught: test_complement_is_skipped_when_a_clinched_team_is_not_clear_...
  a  "len(S) != top_n" weakened to "len(S) > top_n"            SURVIVES by construction (equivalent mutant): len(S) < top_n means the
                                                               wins-only test already clinched T, so the branch is never reached.

Run: python -m pytest tests/test_h2h_clinch.py -q
"""
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifacts import schedule as schedule_module  # noqa: E402
from artifacts import schedule_standings  # noqa: E402
from artifacts.schedule_standings import (  # noqa: E402
    build_head_to_head_series,
    compute_conference_championship_status,
    compute_standings,
)

ACC = "ACC"
PAC12 = "Pac-12"
SEC = "SEC"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _rec(conference, wins, losses, remaining):
    """One compute_team_records() entry; only the conference-tally fields are read."""
    return {
        "conference": conference,
        "wins": wins,
        "losses": losses,
        "conf_wins": wins,
        "conf_losses": losses,
        "conf_games_remaining": remaining,
    }


def _pool(conference, spec):
    """{team: (W, L, R)} -> records dict."""
    return {team: _rec(conference, *wlr) for team, wlr in spec.items()}


def _series(*results, upcoming=()):
    """Head-to-head facts from (winner, loser) played results and (a, b) still-scheduled pairs,
    in the shape build_head_to_head_series returns."""
    out = {}

    def bump(pair, idx):
        cur = list(out.get(pair, (0, 0, 0)))
        cur[idx] += 1
        out[pair] = tuple(cur)

    for winner, loser in results:
        bump((winner, loser), 0)
        bump((loser, winner), 1)
    for a, b in upcoming:
        bump((a, b), 2)
        bump((b, a), 2)
    return out


def _statuses(result):
    return {team: entry["status"] for team, entry in result.items()}


def _run(conference, spec, series=None, first=None, **kwargs):
    records = _pool(conference, spec)
    if series is not None:
        kwargs["head_to_head"] = series
    if first is not None:
        kwargs["head_to_head_first_conferences"] = first
    return compute_conference_championship_status(records, **kwargs)


# 2026 after week 12. Every ACC member has 9 conference games except Clemson (8 counted in the
# live data): the pct clause of condition (f) has to accept it, because Clemson 5-3 cannot reach
# Wake Forest's 7-of-9 floor on percentage either (5/8 < 7/9).
ACC_SPEC = {
    "Miami": (8, 0, 1),
    "Wake Forest": (7, 1, 1),
    "Louisville": (7, 2, 0),
    "SMU": (5, 3, 1),
    "Clemson": (5, 3, 0),
    "Duke": (4, 5, 0),
    "Pittsburgh": (4, 4, 1),
    "Virginia": (3, 6, 0),
}
ACC_H2H = _series(("Wake Forest", "Louisville"))

PAC12_SPEC = {
    "Boise State": (6, 1, 0),
    "Texas State": (5, 2, 0),
    "San Diego State": (5, 2, 0),
    "Oregon State": (4, 3, 0),
    "Utah State": (3, 4, 0),
    "Fresno State": (3, 4, 0),
    "Colorado State": (1, 6, 0),
    "Washington State": (1, 6, 0),
}
PAC12_H2H = _series(("San Diego State", "Texas State"))

SEC_SPEC = {
    "Georgia": (8, 1, 0),
    "Alabama": (6, 2, 1),
    "Tennessee": (6, 2, 1),
    "Texas": (6, 2, 1),
    "Vanderbilt": (3, 5, 1),
    "Auburn": (2, 6, 1),
}


# ---------------------------------------------------------------------------
# 1-3. Real-standings fixtures
# ---------------------------------------------------------------------------
def test_acc_week12_wake_forest_clinches_on_head_to_head_and_louisville_is_eliminated():
    out = _statuses(_run(ACC, ACC_SPEC, ACC_H2H, {ACC}))
    assert out["Miami"] == "clinched"
    assert out["Wake Forest"] == "clinched"
    assert {t for t, s in out.items() if s != "clinched"} == set(ACC_SPEC) - {"Miami", "Wake Forest"}
    assert all(s == "eliminated" for t, s in out.items() if t not in ("Miami", "Wake Forest"))
    assert out["Louisville"] == "eliminated"  # the complement step: the wins-only math leaves it possible


def test_pac12_week12_san_diego_state_clinches_and_texas_state_is_eliminated():
    out = _statuses(_run(PAC12, PAC12_SPEC, PAC12_H2H, {PAC12}))
    assert out["Boise State"] == "clinched"
    assert out["San Diego State"] == "clinched"
    for team in set(PAC12_SPEC) - {"Boise State", "San Diego State"}:
        assert out[team] == "eliminated", team


def test_sec_georgia_clinches_and_the_three_way_chase_stays_possible():
    # Even handed every possible head-to-head win, the rule must not fire: Alabama, Tennessee and
    # Texas can all finish on 7 wins, so any tie at that total is not a two-team tie.
    series = _series(("Alabama", "Tennessee"), ("Tennessee", "Texas"), ("Texas", "Alabama"))
    result = _run(SEC, SEC_SPEC, series, {SEC})
    out = _statuses(result)
    assert out["Georgia"] == "clinched"
    for team in ("Alabama", "Tennessee", "Texas"):
        assert out[team] == "possible", team
        assert result[team]["conditional_opponent"] == "Georgia"


# ---------------------------------------------------------------------------
# Disabled by default / wrong conference -> old output exactly
# ---------------------------------------------------------------------------
def test_rule_is_disabled_by_default_and_reproduces_the_wins_only_output():
    out = _statuses(_run(ACC, ACC_SPEC))
    assert out["Miami"] == "clinched"
    assert out["Wake Forest"] == "possible"
    assert out["Louisville"] == "possible"
    # exactly one clinched team -> the complement step does not fire either
    assert sum(1 for s in out.values() if s == "clinched") == 1


@pytest.mark.parametrize(
    "series, first",
    [
        (ACC_H2H, None),               # series but no conference list
        (None, {ACC}),                 # conference list but no series
        ({}, {ACC}),                   # empty series
        (ACC_H2H, set()),              # nobody's first step is head_to_head
        (ACC_H2H, {"Big Ten"}),        # a different conference's rule only: ACC has none / not h2h
    ],
)
def test_rule_stays_off_without_both_the_series_and_the_conference_rule(series, first):
    out = _statuses(_run(ACC, ACC_SPEC, series, first))
    assert out["Wake Forest"] == "possible"
    assert out["Louisville"] == "possible"


# ---------------------------------------------------------------------------
# 4. Negatives
# ---------------------------------------------------------------------------
def _wake_is_possible(spec, series, first=frozenset({ACC})):
    return _statuses(_run(ACC, spec, series, set(first)))["Wake Forest"] == "possible"


def test_a_third_team_that_can_finish_on_the_same_win_total_blocks_it():
    spec = dict(ACC_SPEC, NC_State=(6, 2, 1))  # B = 7 = Wake's floor, and W = 6 not above it
    assert _wake_is_possible(spec, _series(("Wake Forest", "Louisville"), ("Wake Forest", "NC_State")))


def test_a_second_team_that_ties_at_best_is_not_enough_even_with_both_head_to_heads():
    # Louisville and NC State both top out at Wake's floor of 7: S has 3 members, not top_n=2.
    spec = dict(ACC_SPEC, NC_State=(6, 2, 1))
    series = _series(("Wake Forest", "Louisville"), ("Wake Forest", "NC_State"))
    assert _wake_is_possible(spec, series)


def test_len_s_equal_top_n_with_two_teams_tying_at_best_is_blocked_by_the_proof_of_a_two_team_tie():
    # Miami drops out of the picture (below Wake): S = {Louisville, NC State}, both B == L_T == 7,
    # Wake beat both. |S| == top_n and a head-to-head win exists over each, yet a 3-way tie at 7
    # wins is possible, which head-to-head does not resolve.
    spec = dict(ACC_SPEC, Miami=(6, 3, 0), NC_State=(6, 2, 1))
    series = _series(("Wake Forest", "Louisville"), ("Wake Forest", "NC_State"))
    assert _wake_is_possible(spec, series)


def test_an_unplaced_team_that_banked_exactly_the_floor_blocks_it():
    # Condition (f): Miami 7-1-1 can reach 8 but has only banked 7 == Wake's floor, so a 7-win
    # finish ties Wake AND Louisville -- three teams at 7 wins.
    spec = dict(ACC_SPEC, Miami=(7, 1, 1))
    assert _wake_is_possible(spec, ACC_H2H)


def test_a_rival_that_can_pass_blocks_it():
    # Louisville 6-1-2 can reach 8 > Wake's floor of 7: it can finish ahead outright.
    spec = dict(ACC_SPEC, Louisville=(6, 1, 2))
    assert _wake_is_possible(spec, ACC_H2H)


def test_head_to_head_lost_blocks_it():
    assert _wake_is_possible(ACC_SPEC, _series(("Louisville", "Wake Forest")))


def test_head_to_head_not_played_blocks_it():
    assert _wake_is_possible(ACC_SPEC, {})
    assert _wake_is_possible(ACC_SPEC, _series(("Miami", "Louisville")))  # unrelated series only


def test_two_meetings_split_blocks_it():
    series = {
        ("Wake Forest", "Louisville"): (1, 1, 0),
        ("Louisville", "Wake Forest"): (1, 1, 0),
    }
    assert _wake_is_possible(ACC_SPEC, series)


def test_a_won_series_with_a_rematch_still_scheduled_blocks_it():
    series = _series(("Wake Forest", "Louisville"), upcoming=[("Wake Forest", "Louisville")])
    assert _wake_is_possible(ACC_SPEC, series)
    # ... and when only the winner's own rows show the pending rematch
    one_sided = {**_series(("Wake Forest", "Louisville")), ("Wake Forest", "Louisville"): (1, 0, 1)}
    assert _wake_is_possible(ACC_SPEC, one_sided)


def test_one_sided_series_data_blocks_it():
    # Wake's own row says win but Louisville's mirror row is missing: unknown, never a win.
    assert _wake_is_possible(ACC_SPEC, {("Wake Forest", "Louisville"): (1, 0, 0)})


def test_unequal_conference_game_counts_for_the_pair_block_it():
    # Louisville 7-1-0 has played 8 counted games to Wake's 9: equal wins is not equal percentage.
    spec = dict(ACC_SPEC, Louisville=(7, 1, 0))
    assert _wake_is_possible(spec, ACC_H2H)


def test_a_short_schedule_team_that_could_out_rank_on_percentage_blocks_it():
    # NC State 6-1-0 (7 games): B = 6 < 7 so it never reaches Wake's 7 WINS, but 6/7 > 7/9.
    spec = dict(ACC_SPEC, NC_State=(6, 1, 0))
    assert _wake_is_possible(spec, ACC_H2H)


def test_a_team_that_could_reach_the_floor_on_a_different_game_count_blocks_it():
    # Miami on 8 counted games: it has banked more than Wake's floor, but not on the same
    # denominator, so the "never tied at that total" argument no longer applies.
    spec = dict(ACC_SPEC, Miami=(8, 0, 0))
    assert _wake_is_possible(spec, ACC_H2H)


def test_a_short_schedule_team_that_cannot_reach_the_floor_even_on_percentage_does_not_block_it():
    # Clemson 5-3-0 is already in ACC_SPEC and is the live case; pin it explicitly.
    assert not _wake_is_possible(ACC_SPEC, ACC_H2H)
    assert ACC_SPEC["Clemson"][0] + ACC_SPEC["Clemson"][1] + ACC_SPEC["Clemson"][2] == 8


# ---------------------------------------------------------------------------
# 5. Division pool (top_n = 1)
# ---------------------------------------------------------------------------
DIV_CONF = "Sun Belt"
DIV_SPEC = {
    # East: Ashe 6-1-1 (floor 6) and Boone 6-2-0 (best case 6) -- Ashe beat Boone.
    "Ashe": (6, 1, 1), "Boone": (6, 2, 0), "Cato": (3, 5, 0), "Dunn": (2, 6, 0),
    # West: undecided.
    "Eli": (5, 2, 1), "Finn": (4, 3, 1), "Gus": (3, 5, 0), "Hal": (2, 6, 0),
}
DIV_MAP = {
    **{t: "East" for t in ("Ashe", "Boone", "Cato", "Dunn")},
    **{t: "West" for t in ("Eli", "Finn", "Gus", "Hal")},
}


def _run_div(series):
    return _run(
        DIV_CONF, DIV_SPEC, series, {DIV_CONF},
        qualifying_conferences={}, divisional_conferences={DIV_CONF: "test"}, divisions=DIV_MAP,
    )


def test_division_pool_clinches_on_head_to_head_and_eliminates_the_rival():
    result = _run_div(_series(("Ashe", "Boone")))
    out = _statuses(result)
    assert out["Ashe"] == "clinched"
    assert out["Boone"] == "eliminated"  # wins-only math: B == Ashe's W, so it stayed "possible"
    assert out["Cato"] == out["Dunn"] == "eliminated"
    # West is untouched and its possible teams name the clinched East team
    assert out["Eli"] == "possible"
    assert result["Eli"]["conditional_opponent"] == "Ashe"


def test_division_pool_without_the_head_to_head_stays_possible():
    out = _statuses(_run_div(_series(("Boone", "Ashe"))))
    assert out["Ashe"] == "possible"
    assert out["Boone"] == "possible"
    out = _statuses(_run_div({}))
    assert out["Ashe"] == out["Boone"] == "possible"


# ---------------------------------------------------------------------------
# 6. Complement elimination and the played-championship-game override
# ---------------------------------------------------------------------------
def test_played_title_game_between_the_two_clinched_teams_changes_nothing():
    plain = _run(ACC, ACC_SPEC, ACC_H2H, {ACC})
    with_game = _run(ACC, ACC_SPEC, ACC_H2H, {ACC}, ccg_participants={ACC: {"Miami", "Wake Forest"}})
    assert with_game == plain
    assert all(entry["conditional_opponent"] is None for entry in with_game.values())


def test_a_rejected_participant_set_also_blocks_the_complement_elimination(caplog):
    # SMU and Clemson are not the clinched teams: guard (a) sees Miami and Wake Forest -- both
    # "clinched" non-participants -- and distrusts the whole set. Louisville must then keep the
    # status the clinch math gave it ("possible") instead of being complement-eliminated.
    with caplog.at_level(logging.ERROR, logger="cfb_lp"):
        result = _run(ACC, ACC_SPEC, ACC_H2H, {ACC}, ccg_participants={ACC: {"SMU", "Clemson"}})
    out = _statuses(result)
    assert out["Miami"] == "clinched" and out["Wake Forest"] == "clinched"
    assert out["Louisville"] == "possible"
    assert any("would force team(s)" in r.getMessage() for r in caplog.records)


def test_a_played_game_participant_is_never_complement_eliminated():
    # A (malformed) participant set larger than top_n: Louisville is named, so the override
    # leaves it alone, and the complement step must too.
    result = _run(
        ACC, ACC_SPEC, ACC_H2H, {ACC},
        ccg_participants={ACC: {"Miami", "Wake Forest", "Louisville"}},
    )
    assert result["Louisville"] == {"status": "possible", "conditional_opponent": None}
    assert result["SMU"]["status"] == "eliminated"


def test_no_conditional_opponent_once_both_slots_are_clinched():
    result = _run(ACC, ACC_SPEC, ACC_H2H, {ACC})
    assert all(entry["conditional_opponent"] is None for entry in result.values())


# ---------------------------------------------------------------------------
# 7. Wiring
# ---------------------------------------------------------------------------
def _rows(spec, conference, season, pairs, first_game_id=1):
    """schedule_grid-shaped rows for a pool: `pairs` are real (winner, loser) meetings with mirrored
    rows; the rest of each team's W-L-R is padded with games against off-pool dummy opponents."""
    rows = []
    game_id = first_game_id

    def row(gid, team, opponent, status):
        return dict(
            game_id=gid, season=season, season_type="regular", team=team, opponent=opponent,
            conference=conference, conference_game=True, status=status,
        )

    used = {team: [0, 0] for team in spec}
    for winner, loser in pairs:
        rows.append(row(game_id, winner, loser, "win"))
        rows.append(row(game_id, loser, winner, "loss"))
        used[winner][0] += 1
        used[loser][1] += 1
        game_id += 1
    for team, (w, l, r) in spec.items():
        for status, count in (("win", w - used[team][0]), ("loss", l - used[team][1]), ("upcoming", r)):
            for _ in range(count):
                rows.append(row(game_id, team, f"Off-pool {game_id}", status))
                game_id += 1
    return rows


def test_compute_standings_builds_the_series_from_rows():
    rows = _rows(ACC_SPEC, ACC, 2026, [("Wake Forest", "Louisville")])
    on = compute_standings(rows, 2026, head_to_head_first_conferences={ACC})
    off = compute_standings(rows, 2026)
    assert on["Wake Forest"]["championship_status"] == "clinched"
    assert on["Louisville"]["championship_status"] == "eliminated"
    assert off["Wake Forest"]["championship_status"] == "possible"
    # a generator input is consumed once but used twice internally
    gen = compute_standings(iter(rows), 2026, head_to_head_first_conferences={ACC})
    assert gen == on


def test_compute_standings_with_the_head_to_head_game_excluded_does_not_fire():
    rows = _rows(ACC_SPEC, ACC, 2026, [("Wake Forest", "Louisville")])
    game_id = next(r["game_id"] for r in rows if r["team"] == "Wake Forest" and r["opponent"] == "Louisville")
    on = compute_standings(rows, 2026, excluded_conference_game_ids={game_id}, head_to_head_first_conferences={ACC})
    assert on["Wake Forest"]["championship_status"] != "clinched"


def test_build_head_to_head_series_counts_each_perspective_once_and_honours_exclusions():
    rows = _rows(ACC_SPEC, ACC, 2026, [("Wake Forest", "Louisville")])
    series = build_head_to_head_series(rows, 2026)
    assert series[("Wake Forest", "Louisville")] == (1, 0, 0)
    assert series[("Louisville", "Wake Forest")] == (0, 1, 0)
    gid = next(r["game_id"] for r in rows if r["team"] == "Wake Forest" and r["opponent"] == "Louisville")
    assert ("Wake Forest", "Louisville") not in build_head_to_head_series(rows, 2026, {gid})
    assert build_head_to_head_series(rows, 2025) == {}
    non_conf = [dict(r, conference_game=False) for r in rows]
    assert build_head_to_head_series(non_conf, 2026) == {}


def test_first_step_head_to_head_is_read_from_the_real_tiebreaker_config():
    first = schedule_module._head_to_head_first_conferences
    assert first({ACC, PAC12, SEC, "Not A Conference"}, 2026) == {ACC, PAC12, SEC}
    assert first({PAC12}, 2025) == set()  # Pac-12 has no rule set before 2026


def test_a_config_that_failed_to_load_leaves_the_rule_off(monkeypatch):
    monkeypatch.setattr(schedule_module, "_tiebreaker_config", lambda: None)
    assert schedule_module._head_to_head_first_conferences({ACC}, 2026) == set()


def test_build_schedule_payload_passes_the_conference_rules_through():
    """End to end through the call site in schedule.build_schedule_payload: the published
    championship status for the 2026 ACC fixture shows Wake Forest clinched."""
    rows = _rows(ACC_SPEC, ACC, 2026, [("Wake Forest", "Louisville")])
    for r in rows:
        r.update(start_date="2026-10-17 19:00:00", home_away="home", neutral_site=False)
    teams_meta = {t: {"conference": ACC, "division": None, "logos": None} for t in ACC_SPEC}
    payload = schedule_module.build_schedule_payload(rows, teams_meta, 2026)
    teams = {t["team"]: t for c in payload["conferences"] for t in c["teams"]}

    def ccg_status(team):
        slot = next(w for w in teams[team]["weeks"] if w["slot_id"] == "conf-championship")
        return slot["status"]

    assert ccg_status("Miami") == "clinched"
    assert ccg_status("Wake Forest") == "clinched"
    assert ccg_status("Louisville") == "eliminated"


# ---------------------------------------------------------------------------
# 8. Conference tie definitions wider than equal win percentage (review F1)
# ---------------------------------------------------------------------------
ACC_ALT = {ACC: "acc_alternate_games"}
CUSA = "Conference USA"
CUSA_DEF = {CUSA: "cusa_within_one_win"}

# Clemson 6-2 on 8 games shares Wake Forest's possible loss count (7-2 worst case): under the
# ACC's 2026 "same wins OR same losses" definition Clemson joins the tie, so it is 3-way.
ACC_CX_SPEC = {
    "Miami": (8, 0, 1), "Wake Forest": (7, 1, 1), "Louisville": (7, 2, 0),
    "Clemson": (6, 2, 0), "Duke": (4, 5, 0),
}
# "Short" 5-2 on 7 games is within one win of Jacksonville State's 6-2 worst case, equal losses.
CUSA_CX_SPEC = {
    "Liberty": (8, 0, 0), "Jacksonville State": (6, 1, 1), "WKU": (6, 2, 0),
    "Short": (5, 2, 0), "Other": (3, 5, 0),
}
CUSA_CX_H2H = _series(("Jacksonville State", "WKU"))


def test_acc_alternate_games_tie_blocks_the_head_to_head_clinch():
    series = _series(("Wake Forest", "Louisville"))
    out = _statuses(_run(ACC, ACC_CX_SPEC, series, {ACC}, tie_definitions=ACC_ALT))
    assert out["Wake Forest"] == "possible"
    assert out["Miami"] == "clinched"
    # same fixture under plain win percentage still clinches: the block is what stops it
    assert _statuses(_run(ACC, ACC_CX_SPEC, series, {ACC}))["Wake Forest"] == "clinched"
    assert _statuses(_run(ACC, ACC_CX_SPEC, series, {ACC}, tie_definitions={ACC: "win_pct"}))["Wake Forest"] == "clinched"


def test_cusa_within_one_win_tie_blocks_the_head_to_head_clinch():
    out = _statuses(_run(CUSA, CUSA_CX_SPEC, CUSA_CX_H2H, {CUSA}, tie_definitions=CUSA_DEF))
    assert out["Jacksonville State"] == "possible"
    assert _statuses(_run(CUSA, CUSA_CX_SPEC, CUSA_CX_H2H, {CUSA}))["Jacksonville State"] == "clinched"


def test_unknown_tie_definition_is_read_as_the_strictest_case():
    out = _statuses(_run(CUSA, CUSA_CX_SPEC, CUSA_CX_H2H, {CUSA}, tie_definitions={CUSA: "some_future_rule"}))
    assert out["Jacksonville State"] == "possible"
    # a conference missing from a supplied mapping is unknown too
    out = _statuses(_run(CUSA, CUSA_CX_SPEC, CUSA_CX_H2H, {CUSA}, tie_definitions={"Other Conf": "win_pct"}))
    assert out["Jacksonville State"] == "possible"


def test_live_acc_still_clinches_under_the_acc_alternate_games_definition():
    out = _statuses(_run(ACC, ACC_SPEC, ACC_H2H, {ACC}, tie_definitions=ACC_ALT))
    assert out == {
        "Miami": "clinched", "Wake Forest": "clinched", "Louisville": "eliminated", "SMU": "eliminated",
        "Clemson": "eliminated", "Duke": "eliminated", "Pittsburgh": "eliminated", "Virginia": "eliminated",
    }


def test_pac12_fixture_is_unchanged_under_win_pct():
    out = _statuses(_run(PAC12, PAC12_SPEC, PAC12_H2H, {PAC12}, tie_definitions={PAC12: "win_pct"}))
    assert out == _statuses(_run(PAC12, PAC12_SPEC, PAC12_H2H, {PAC12}))
    assert out["Boise State"] == out["San Diego State"] == "clinched"


def _ranges(**teams):
    """name -> (W, N, B) -> the W/B/N dicts the helper takes."""
    return (
        {t: v[0] for t, v in teams.items()}, {t: v[2] for t, v in teams.items()},
        {t: v[1] for t, v in teams.items()},
    )


def test_clear_helper_wins_and_losses_ranges_each_block_on_their_own():
    clear = schedule_standings._clear_of_other_game_counts
    # T: wins [7, 8] of 9 (losses [1, 2]).
    W, B, N = _ranges(T=(7, 9, 8), j=(8, 12, 8))      # wins [8,8] overlap; losses [4,4] clear
    assert clear("T", ["T", "j"], W, B, N, "acc_alternate_games") is False
    W, B, N = _ranges(T=(7, 9, 8), j=(2, 4, 2))       # wins [2,2] clear; losses [2,2] overlap
    assert clear("T", ["T", "j"], W, B, N, "acc_alternate_games") is False
    W, B, N = _ranges(T=(7, 9, 8), j=(5, 8, 5))       # wins [5,5]; losses [3,3]: clear both
    assert clear("T", ["T", "j"], W, B, N, "acc_alternate_games") is True
    # same game count is not examined at all
    W, B, N = _ranges(T=(7, 9, 8), j=(8, 9, 8))
    assert clear("T", ["T", "j"], W, B, N, "acc_alternate_games") is True


def test_clear_helper_cusa_and_unknown_widen_the_wins_range_by_one_but_acc_does_not():
    clear = schedule_standings._clear_of_other_game_counts
    # T wins [7, 8], losses [1, 2]; j wins [6, 6] (adjacent), losses [4, 4] (no loss overlap).
    W, B, N = _ranges(T=(7, 9, 8), j=(6, 10, 6))
    assert clear("T", ["T", "j"], W, B, N, "acc_alternate_games") is True
    assert clear("T", ["T", "j"], W, B, N, "cusa_within_one_win") is False
    assert clear("T", ["T", "j"], W, B, N, "brand_new_rule") is False


def test_complement_is_skipped_when_a_clinched_team_is_not_clear_of_other_game_counts(monkeypatch):
    real = schedule_standings._clear_of_other_game_counts
    calls = []

    def fake(team, *args, **kwargs):
        calls.append(team)
        return False if team == "Miami" else real(team, *args, **kwargs)

    monkeypatch.setattr(schedule_standings, "_clear_of_other_game_counts", fake)
    out = _statuses(_run(ACC, ACC_SPEC, ACC_H2H, {ACC}, tie_definitions=ACC_ALT))
    assert "Miami" in calls
    assert out["Miami"] == "clinched" and out["Wake Forest"] == "clinched"
    # the complement did not run: Louisville (the only team the wins-only math cannot eliminate)
    # stays possible; SMU / Clemson are eliminated by the wins-only test itself (best case 6 < 7)
    assert out["Louisville"] == "possible"
    assert out["SMU"] == out["Clemson"] == "eliminated"


def test_complement_is_not_fenced_for_win_pct(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("win_pct must not consult the wider-definition check")

    monkeypatch.setattr(schedule_standings, "_clear_of_other_game_counts", boom)
    out = _statuses(_run(ACC, ACC_SPEC, ACC_H2H, {ACC}, tie_definitions={ACC: "win_pct"}))
    assert out["Louisville"] == "eliminated"


def test_compute_standings_threads_tie_definitions_end_to_end():
    rows = _rows(ACC_CX_SPEC, ACC, 2026, [("Wake Forest", "Louisville")])
    on = compute_standings(rows, 2026, head_to_head_first_conferences={ACC}, tie_definitions=ACC_ALT)
    legacy = compute_standings(rows, 2026, head_to_head_first_conferences={ACC})
    assert on["Wake Forest"]["championship_status"] == "possible"
    assert legacy["Wake Forest"]["championship_status"] == "clinched"


def test_real_config_tie_definitions_reach_the_helper():
    defs = schedule_module._tie_definitions({ACC, CUSA, SEC, PAC12, "Not A Conference"}, 2026)
    assert defs == {
        ACC: "acc_alternate_games", CUSA: "cusa_within_one_win", SEC: "win_pct", PAC12: "win_pct",
    }
    assert schedule_module._tie_definitions({ACC}, 2025) == {ACC: "win_pct"}  # pre-amendment era


def test_a_config_that_failed_to_load_gives_no_tie_definitions(monkeypatch):
    monkeypatch.setattr(schedule_module, "_tiebreaker_config", lambda: None)
    assert schedule_module._tie_definitions({ACC}, 2026) == {}


def test_build_schedule_payload_passes_the_tie_definitions_through(monkeypatch):
    seen = {}
    real = schedule_standings.compute_standings

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(schedule_standings, "compute_standings", spy)
    rows = _rows(ACC_CX_SPEC, ACC, 2026, [("Wake Forest", "Louisville")])
    for r in rows:
        r.update(start_date="2026-10-17 19:00:00", home_away="home", neutral_site=False)
    teams_meta = {t: {"conference": ACC, "division": None, "logos": None} for t in ACC_CX_SPEC}
    payload = schedule_module.build_schedule_payload(rows, teams_meta, 2026)
    assert seen["tie_definitions"][ACC] == "acc_alternate_games"
    teams = {t["team"]: t for c in payload["conferences"] for t in c["teams"]}
    slot = next(w for w in teams["Wake Forest"]["weeks"] if w["slot_id"] == "conf-championship")
    assert slot["status"] != "clinched"
