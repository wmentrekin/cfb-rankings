"""T6: acceptance fixtures built from the real 2025 conference ties.

WHAT THESE ARE, AND WHAT THEY ARE NOT
-------------------------------------
These are NOT replays of the 2025 schedules. This environment has no database credentials, so
the games cannot be read back. What each fixture reproduces instead is the DECISIVE STRUCTURE of
a real tie -- how many teams, which of them actually played each other, how large the common
opponent set was, and the aggregate that separated them -- using values recomputed from live data
earlier in this feature's work and recorded in docs/conference-tiebreakers/source-rules/. Opponent
identities outside each tied group are synthetic; the numbers that decide the tie are the real
ones.

That distinction matters because it bounds what a pass proves. A green run here says the
configured chain reaches the right answer FOR THE RIGHT REASON on the shape the season actually
produced. It does not say the 2025 payload renders correctly end to end -- that needs a reingest
and a republish against the real rows.

WHY resolved_by IS ASSERTED EVERYWHERE
--------------------------------------
Plan risk R2: the engine can reproduce the right ORDER for the wrong REASON and nobody notices.
That is not hypothetical here. An early reading of the ACC's procedure explained Duke's win by
head-to-head and survived an entire research pass before being caught -- Duke was LAST in the
group on head-to-head. Asserting only the order would have passed that reading too. So every test
below pins the step that decided, not just who came first.

Run: python -m pytest tests/test_tiebreaker_acceptance_2025.py -q
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from artifacts.tiebreaker_engine import order_tied_group  # noqa: E402
from artifacts.tiebreaker_rules import (  # noqa: E402
    KNOWN_STEPS,
    load_conference_rules,
    rules_for,
)
from artifacts.tiebreaker_steps import TiebreakContext  # noqa: E402

SEASON = 2025


def _row(team, opponent, status, week=1, conference_game=True):
    return {
        "season": SEASON, "team": team, "opponent": opponent, "status": status,
        "conference_game": conference_game, "season_type": "regular", "week": week,
        "game_id": None, "team_score": None, "opp_score": None,
    }


def _pair(winner, loser, week=1, conference_game=True):
    """Both team-oriented rows of one played game."""
    return [
        _row(winner, loser, "win", week, conference_game),
        _row(loser, winner, "loss", week, conference_game),
    ]


# ===========================================================================
# ACC 2025 -- the five-way tie at 6-2
# ===========================================================================
# Duke, Georgia Tech, Miami, Pittsburgh and SMU all finished 6-2 in the ACC. Three facts about
# that group were recomputed from live data during this feature's discovery and are what the
# fixture below reproduces:
#
#   1. It was NOT a round robin. Exactly one pair among the five met: Georgia Tech beat Duke.
#      So Duke was 0-1 on head-to-head -- LAST in the group -- and still won the tie.
#   2. The set of opponents common to all five was exactly ONE team: Syracuse, beaten by all
#      five. That is below the min_sample of 2, so common_opponents_record is gated off -- but
#      the gate is NOT what decides this case, and an earlier version of this comment overclaimed
#      that it was. Because all five BEAT Syracuse, an ungated step would compute 1.000 for
#      everyone and separate nobody, so the chain would move on either way. The gate itself is
#      pinned separately, in test_common_opponents_record_acc_gate_returns_none.
#   3. Duke's eight ACC opponents went 32-32 (.500), the best of the five. That is the aggregate
#      that actually decided it.
#
# The opponents' individual identities are synthetic; their records are chosen so each team's
# total hits the real aggregate.

ACC_FIVE = ["Duke", "Georgia Tech", "Miami", "Pittsburgh", "SMU"]

# Each tied team's opponents' combined conference record. Duke's is the real recomputed value;
# the other four are given descending totals so that every team separates at this step and the
# assertion can name a full order rather than only a winner.
_ACC_OPPONENT_TOTALS = {
    "Duke": (32, 32),            # .500 -- the real value, best of the five
    "Georgia Tech": (30, 34),
    "Miami": (29, 35),
    "Pittsburgh": (28, 36),
    "SMU": (27, 37),
}


def _acc_fixture():
    """Build rows and conference records for the ACC five-way.

    Every tied team plays eight conference games: Syracuse, plus seven others. Duke and Georgia
    Tech each spend one of those seven on the other, which is the single intra-group meeting.
    Opponent records are assigned so each tied team's opponents sum to its entry in
    _ACC_OPPONENT_TOTALS.
    """
    rows = []
    conf_records = {team: (6, 2) for team in ACC_FIVE}
    conf_records["Syracuse"] = (2, 6)

    # The one intra-group game, and the reason a head-to-head reading of this tie is wrong.
    rows += _pair("Georgia Tech", "Duke")

    def _fill(team, extra_opponents, target):
        """Give `team` its remaining opponents, records summing to `target` once Syracuse (and
        the intra-group game, already counted by the caller) are included."""
        target_w, target_l = target
        # Syracuse is every tied team's opponent; the caller has already accounted for any
        # intra-group opponent in `accounted`.
        remaining_w = target_w - accounted[team][0]
        remaining_l = target_l - accounted[team][1]
        n = len(extra_opponents)
        # Distribute as 4-4 records, downgrading some to 3-5 until the totals match. Each
        # downgrade moves one win to a loss, so the number of downgrades is fixed by the gap.
        downgrades = (4 * n) - remaining_w
        assert 0 <= downgrades <= n, (
            f"{team}: cannot hit {target} with {n} opponents (downgrades={downgrades})"
        )
        assert remaining_l == (4 * n) + downgrades, (
            f"{team}: losses {remaining_l} inconsistent with wins {remaining_w}"
        )
        for i, opponent in enumerate(extra_opponents):
            conf_records[opponent] = (3, 5) if i < downgrades else (4, 4)

    # What each tied team's opponents already contribute before the generic fill: Syracuse for
    # everyone, plus the intra-group opponent for Duke and Georgia Tech.
    accounted = {}
    for team in ACC_FIVE:
        w, l = conf_records["Syracuse"]
        if team == "Duke":
            w += conf_records["Georgia Tech"][0]
            l += conf_records["Georgia Tech"][1]
        elif team == "Georgia Tech":
            w += conf_records["Duke"][0]
            l += conf_records["Duke"][1]
        accounted[team] = (w, l)

    for team in ACC_FIVE:
        # Every tied team beat Syracuse -- which is exactly why that lone common opponent
        # separates nobody even when a step does look at it.
        rows += _pair(team, "Syracuse")

        intra = 1 if team in ("Duke", "Georgia Tech") else 0
        n_extra = 7 - intra
        extras = [f"{team.replace(' ', '')}Opp{i}" for i in range(n_extra)]
        _fill(team, extras, _ACC_OPPONENT_TOTALS[team])

        # Each team must finish 6-2 in conference. Wins already banked: Syracuse, plus the
        # Georgia Tech-Duke result. Losses already banked: that same result for Duke.
        wins_banked = 1 + (1 if team == "Georgia Tech" else 0)
        losses_banked = 1 if team == "Duke" else 0
        wins_left = 6 - wins_banked
        losses_left = 2 - losses_banked
        assert wins_left + losses_left == n_extra, (team, wins_left, losses_left, n_extra)
        for i, opponent in enumerate(extras):
            if i < wins_left:
                rows += _pair(team, opponent)
            else:
                rows += _pair(opponent, team)

    return rows, conf_records


# Production hands every primitive the WHOLE LEAGUE's schedule grid for the season, including
# fixtures that have not been played. Every fixture in this suite originally contained only the
# tied group's own conference and only completed games, and that single shared blind spot is what
# hid a real bug: the cascade's "final weekend" was derived league-wide and from scheduled rather
# than played games, so it landed on a week the tied teams had no game in, every team read as
# idle, and the whole condition silently stopped applying. Both fixtures below now carry these.
def _league_noise():
    """Rows that must not influence any conference's tiebreak: another conference's games, and an
    unplayed fixture in a later week than anything in the fixture proper."""
    return [
        _row("Some SEC Team", "Another SEC Team", "win", week=14),
        _row("Another SEC Team", "Some SEC Team", "loss", week=14),
        _row("Some Big Ten Team", "Another Big Ten Team", "upcoming", week=15),
    ]


def _acc_context():
    rows, conf_records = _acc_fixture()
    rows = rows + _league_noise()
    # frozen_order is by conference win percentage ALONE (K4). The five tied teams lead at .750;
    # everyone else follows. Only its relative order matters to vs_placed_opponents.
    others = sorted(t for t in conf_records if t not in ACC_FIVE)
    return TiebreakContext(
        rows=rows,
        season=SEASON,
        conference="ACC",
        frozen_order=ACC_FIVE + others,
        conf_records=conf_records,
        team_ranks={team: None for team in ACC_FIVE},
        non_fbs_teams=frozenset(),
    )


def _acc_rules():
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    rules = rules_for(cfg, "ACC", SEASON)
    assert rules is not None, "the ACC's pre-amendment era must cover 2025"
    return rules


def test_acc_2025_fixture_reproduces_the_three_facts_it_claims():
    """The fixture is only worth anything if it actually has the structure it says it has, so
    check that first. Every later assertion rests on these three."""
    rows, conf_records = _acc_fixture()

    # 1. Every tied team really is 6-2 in the conference, as built from the rows.
    for team in ACC_FIVE:
        wins = sum(1 for r in rows if r["team"] == team and r["status"] == "win")
        losses = sum(1 for r in rows if r["team"] == team and r["status"] == "loss")
        assert (wins, losses) == (6, 2), (team, wins, losses)
        assert conf_records[team] == (6, 2), team

    # 2. Exactly one pair among the five met, and Duke lost it.
    met = {
        (r["team"], r["opponent"]) for r in rows
        if r["team"] in ACC_FIVE and r["opponent"] in ACC_FIVE
    }
    assert met == {("Georgia Tech", "Duke"), ("Duke", "Georgia Tech")}, met
    duke_vs_gt = [
        r for r in rows if r["team"] == "Duke" and r["opponent"] == "Georgia Tech"
    ]
    assert [r["status"] for r in duke_vs_gt] == ["loss"]

    # 3. The common opponent set is exactly {Syracuse}, and Duke's opponents went 32-32.
    opponent_sets = [
        {r["opponent"] for r in rows if r["team"] == team and r["opponent"] not in ACC_FIVE}
        for team in ACC_FIVE
    ]
    assert set.intersection(*opponent_sets) == {"Syracuse"}
    for team, expected in _ACC_OPPONENT_TOTALS.items():
        opponents = [r["opponent"] for r in rows if r["team"] == team]
        total_w = sum(conf_records[o][0] for o in opponents)
        total_l = sum(conf_records[o][1] for o in opponents)
        assert (total_w, total_l) == expected, (team, total_w, total_l, expected)
    assert _ACC_OPPONENT_TOTALS["Duke"] == (32, 32)


def test_acc_2025_five_way_is_won_by_duke_on_opponents_combined_record():
    """AC1. The headline acceptance case, and the one that caught a wrong reading once already.

    Duke finishes FIRST despite being last in the group on head-to-head, and the step that puts
    it there must be opponents_cumulative_conf_pct -- not head-to-head, not common opponents.
    """
    outcome = order_tied_group(ACC_FIVE, _acc_context(), _acc_rules())
    assert outcome.flat[0] == "Duke", outcome.flat
    assert outcome.resolved_by["Duke"] == "opponents_cumulative_conf_pct", outcome.resolved_by


def test_acc_2025_the_lone_common_opponent_could_not_have_separated_them_anyway():
    """Keeps the comment above honest. The min_sample gate stops common_opponents_record before
    it computes anything -- but even ungated it would find every tied team at 1.000 against
    Syracuse, because all five beat it. So this step is silent for two independent reasons, and
    only one of them is the gate.

    Worth pinning because the opposite would be a real hazard: if the five had split against
    that single common opponent, an ungated step would separate them on ONE game and pre-empt
    the measure that actually decided the tie."""
    ctx = _acc_context()
    for team in ACC_FIVE:
        against_syracuse = [
            r["status"] for r in ctx.rows
            if r["team"] == team and r["opponent"] == "Syracuse"
        ]
        assert against_syracuse == ["win"], (team, against_syracuse)


def test_acc_2025_no_earlier_step_in_the_chain_speaks():
    """The other half of the same claim: Duke wins because the three steps BEFORE the deciding
    one are all silent on this shape. If any of them spoke, the answer would be reached for the
    wrong reason even when the order happened to come out right.

    Each is silent for its own documented reason -- the group is not a round robin, nobody swept
    it, and the common opponent set is a single team, below min_sample."""
    outcome = order_tied_group(ACC_FIVE, _acc_context(), _acc_rules())
    silent = {"sub_group_record", "sweep_in_out", "common_opponents_record", "vs_placed_opponents"}
    assert not (set(outcome.resolved_by.values()) & silent), outcome.resolved_by


def test_acc_2025_full_order_follows_the_opponents_combined_record():
    """With distinct opponent totals every team separates at the same step, so the whole group's
    order is determined by it -- a stronger claim than "Duke first" alone."""
    outcome = order_tied_group(ACC_FIVE, _acc_context(), _acc_rules())
    assert outcome.flat == ["Duke", "Georgia Tech", "Miami", "Pittsburgh", "SMU"]
    assert set(outcome.resolved_by.values()) == {"opponents_cumulative_conf_pct"}


def test_acc_2025_a_head_to_head_first_reading_would_have_placed_duke_last():
    """Pins the counterfactual that makes the test above meaningful. Duke is 0-1 inside the group
    and everyone else is unbeaten-or-unplayed there, so any procedure that ranked this group by
    intra-group record would put Duke at the BOTTOM -- the exact inversion of what happened.

    This is why sub_group_record is gated on a complete round robin and why sweep_in_out requires
    a result against every other tied team rather than every one a team happened to play."""
    ctx = _acc_context()
    intra_group_record = {}
    for team in ACC_FIVE:
        wins = sum(
            1 for r in ctx.rows
            if r["team"] == team and r["opponent"] in ACC_FIVE and r["status"] == "win"
        )
        losses = sum(
            1 for r in ctx.rows
            if r["team"] == team and r["opponent"] in ACC_FIVE and r["status"] == "loss"
        )
        intra_group_record[team] = (wins, losses)
    assert intra_group_record["Duke"] == (0, 1)
    assert intra_group_record["Georgia Tech"] == (1, 0)
    assert all(intra_group_record[t] == (0, 0) for t in ("Miami", "Pittsburgh", "SMU"))


# ===========================================================================
# Mountain West 2025 -- the four-way tie
# ===========================================================================
# Recorded during discovery as unreachable by any record-based measure: UNLV advanced holding the
# WORST head-to-head record in the group. mountainwest.txt then explained it exactly -- its
# multi-team step 1 does not rank a partially-played group at all ("the group of teams SHALL
# REMAIN TIED, unless one team defeated all other tied teams"), so the tie went straight to the
# CFP/composite ranking at step 2.
#
# Only two things about that group are established well enough to assert: it was not a complete
# round robin, and UNLV was 0-2 within it. The other three teams' identities are not pinned down
# here, so they are named generically rather than guessed at.

MW_FOUR = ["UNLV", "MW Rival A", "MW Rival B", "MW Rival C"]


def _mw_context(ranks=None, final_week_results=None):
    """UNLV is 0-2 inside the group; the other pairs never met, so it is not a round robin and
    nobody defeated all the others.

    `final_week_results` maps each tied team to 'win' or 'loss' in the LAST conference week,
    which the Mountain West's step 2 turns on: only a ranked team that WINS the final weekend is
    selected. Every earlier game sits in its own earlier week, so the final week is a single
    unambiguous game per team -- without that the step reads whichever row happens to come first.
    """
    ranks = ranks or {"UNLV": 8, "MW Rival A": 20, "MW Rival B": None, "MW Rival C": None}
    final_week_results = final_week_results or {
        # The real shape: UNLV is the ranked team that won on the final weekend.
        "UNLV": "win", "MW Rival A": "loss", "MW Rival B": "win", "MW Rival C": "loss",
    }
    FINAL_WEEK = 12

    rows = _pair("MW Rival A", "UNLV", week=2) + _pair("MW Rival B", "UNLV", week=3)
    conf_records = {team: (6, 2) for team in MW_FOUR}
    for team in MW_FOUR:
        for j in range(4):
            opponent = f"{team.replace(' ', '')}Opp{j}"
            conf_records[opponent] = (4, 4)
            week = 4 + j
            rows += _pair(team, opponent, week=week) if j < 2 else _pair(opponent, team, week=week)
        # The final-weekend game, one per team, in a week nothing else occupies.
        closer = f"{team.replace(' ', '')}Closer"
        conf_records[closer] = (4, 4)
        if final_week_results[team] == "win":
            rows += _pair(team, closer, week=FINAL_WEEK)
        else:
            rows += _pair(closer, team, week=FINAL_WEEK)
    rows += _league_noise()
    return TiebreakContext(
        rows=rows,
        season=SEASON,
        conference="Mountain West",
        frozen_order=MW_FOUR,
        conf_records=conf_records,
        team_ranks=ranks,
        non_fbs_teams=frozenset(),
    )


def _mw_rules():
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    rules = rules_for(cfg, "Mountain West", SEASON)
    assert rules is not None
    return rules


def test_mountain_west_2025_worst_head_to_head_still_advances():
    """AC2, and the case that was an open question until the primary text arrived. UNLV is 0-2
    inside the group and must still come first, because the Mountain West's opening step is void
    on a partially-played group and the ranking at step 2 decides instead."""
    outcome = order_tied_group(MW_FOUR, _mw_context(), _mw_rules())
    assert outcome.flat[0] == "UNLV", outcome.flat
    assert outcome.resolved_by["UNLV"] == "conditional_external_ranking", outcome.resolved_by


def test_mountain_west_2025_opening_step_is_void_not_merely_outvoted():
    """The distinction that matters: the opening intra-group step must have NO OPINION here, not
    an opinion that a later step overrides. If it spoke, UNLV would be placed last and no later
    step could lift it -- recursion happens strictly within a group."""
    outcome = order_tied_group(MW_FOUR, _mw_context(), _mw_rules())
    assert "sub_group_record" not in outcome.resolved_by.values()
    assert "sweep_in_out" not in outcome.resolved_by.values()


def test_mountain_west_2025_is_not_a_round_robin_and_nobody_swept_it():
    """The two properties that make the opening step void, asserted directly against the fixture
    so a later change to it cannot quietly invalidate the test above."""
    ctx = _mw_context()
    met = {
        frozenset((r["team"], r["opponent"])) for r in ctx.rows
        if r["team"] in MW_FOUR and r["opponent"] in MW_FOUR
    }
    assert len(met) == 2, "only two of the six possible intra-group pairs met"
    for team in MW_FOUR:
        beaten = {
            r["opponent"] for r in ctx.rows
            if r["team"] == team and r["opponent"] in MW_FOUR and r["status"] == "win"
        }
        assert beaten != set(MW_FOUR) - {team}, f"{team} swept the group; fixture is wrong"


def test_mountain_west_an_unranked_team_does_not_advance_on_the_ranking_step():
    """Control one: the ranked cutoff is load-bearing.

    UNLV is the ONLY team to win the final weekend, so the survival half of the condition favours
    it outright -- but at rank 60 it is outside the cutoff and cannot be a ranked survivor, so it
    must not be promoted. An earlier version of this control gave a rival the best rank as well,
    which meant widening the cutoff changed nothing and the assertion held either way; this
    version fails if the cutoff stops being applied."""
    ranks = {"UNLV": 60, "MW Rival A": 20, "MW Rival B": None, "MW Rival C": None}
    final = {"UNLV": "win", "MW Rival A": "loss", "MW Rival B": "loss", "MW Rival C": "loss"}
    outcome = order_tied_group(MW_FOUR, _mw_context(ranks, final), _mw_rules())
    assert outcome.flat[0] != "UNLV", outcome.flat
    assert outcome.flat[0] == "MW Rival A", outcome.flat


def test_mountain_west_a_ranked_team_that_loses_the_final_weekend_does_not_advance():
    """Control two, and the one that proves the FINAL-WEEKEND half of the condition is doing
    work rather than the ranking alone. UNLV keeps the best rank but loses its last conference
    game, so the lower-ranked team that won it is selected instead."""
    ranks = {"UNLV": 8, "MW Rival A": 20, "MW Rival B": None, "MW Rival C": None}
    final = {"UNLV": "loss", "MW Rival A": "win", "MW Rival B": "loss", "MW Rival C": "loss"}
    outcome = order_tied_group(MW_FOUR, _mw_context(ranks, final), _mw_rules())
    assert outcome.flat[0] == "MW Rival A", outcome.flat
    assert outcome.resolved_by["MW Rival A"] == "conditional_external_ranking"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
