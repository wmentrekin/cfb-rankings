"""Tests for artifacts/tiebreaker_engine.py -- the recursive peel-off driver.

These are the driver's own tests: that it walks a chain in order, recurses into sub-groups at the
right chain, terminates, and always returns a permutation of its input. The individual measures
are tested in tests/test_tiebreaker_steps.py and the config in tests/test_tiebreaker_rules.py;
nothing here re-tests those.

Most tests build a SYNTHETIC rule set out of real step names rather than loading the shipped
config, so that a change to a conference's transcribed procedure cannot make a driver test fail
for a reason that has nothing to do with the driver. The exceptions are the two tests at the end
that deliberately run the real Power 4 chains, which is what catches the driver and the config
disagreeing about a step's parameters.

Run: python -m pytest tests/test_tiebreaker_engine.py -q
"""
import itertools
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from artifacts.tiebreaker_engine import (  # noqa: E402
    FALLBACK_LABEL,
    order_tied_group,
)
from artifacts.tiebreaker_rules import (  # noqa: E402
    KNOWN_STEPS,
    MultiTeamRules,
    RuleSet,
    Step,
    load_conference_rules,
    rules_for,
)
from artifacts.tiebreaker_steps import TiebreakContext  # noqa: E402

SEASON = 2025


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _row(team, opponent, status, team_score=None, opp_score=None, conference_game=True,
         season=SEASON, season_type="regular", game_id=None):
    return {
        "season": season, "team": team, "opponent": opponent, "status": status,
        "team_score": team_score, "opp_score": opp_score, "conference_game": conference_game,
        "season_type": season_type, "game_id": game_id,
    }


def _game(team_a, score_a, team_b, score_b, conference_game=True, season=SEASON,
          season_type="regular", game_id=None):
    """Both team-oriented rows of one game, as schedule_grid actually produces them."""
    a_won = score_a > score_b
    return [
        _row(team_a, team_b, "win" if a_won else "loss", score_a, score_b, conference_game,
             season, season_type, game_id),
        _row(team_b, team_a, "loss" if a_won else "win", score_b, score_a, conference_game,
             season, season_type, game_id),
    ]


def _ctx(rows=None, conf_records=None, team_ranks=None, frozen_order=None, conference="TEST",
         non_fbs_teams=None, placement_excluded_game_ids=frozenset()):
    return TiebreakContext(
        rows=rows or [],
        season=SEASON,
        conference=conference,
        frozen_order=frozen_order or [],
        conf_records=conf_records or {},
        team_ranks=team_ranks or {},
        placement_excluded_game_ids=placement_excluded_game_ids,
        non_fbs_teams=non_fbs_teams,
    )


def _rules(two_team=None, multi_team=None, restart_at="size_appropriate_restart",
           eliminated_teams_locked=True, tie_definition="win_pct"):
    """A synthetic rule set. Defaults to head-to-head only, on both chains."""
    two = tuple(two_team if two_team is not None else [Step(step="head_to_head")])
    multi = tuple(multi_team if multi_team is not None else [Step(step="head_to_head")])
    return RuleSet(
        season_min=None, season_max=None, provenance="search_derived",
        source_file="synthetic", notes="test fixture",
        two_team=two,
        multi_team=MultiTeamRules(
            restart_at=restart_at, steps=multi,
            eliminated_teams_locked=eliminated_teams_locked,
        ),
        tie_definition=tie_definition,
    )


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tied", [[], ["Solo"]], ids=["empty", "single"])
def test_zero_or_one_team_is_returned_unchanged(tied):
    outcome = order_tied_group(tied, _ctx(), _rules())
    assert outcome.flat == tied
    assert outcome.resolved_by == {}


# ---------------------------------------------------------------------------
# The basic contract: a step's partition orders the group
# ---------------------------------------------------------------------------
def test_two_team_head_to_head_orders_the_winner_first():
    ctx = _ctx(_game("A", 24, "B", 10))
    outcome = order_tied_group(["B", "A"], ctx, _rules())
    assert outcome.flat == ["A", "B"]
    assert outcome.resolved_by == {"A": "head_to_head", "B": "head_to_head"}


def test_a_step_that_declines_falls_through_to_the_next_one():
    """Head-to-head is silent on teams that never met, so the chain must reach step 2. Without
    fall-through, the whole chain would stall on its first uninformative step."""
    ctx = _ctx(
        _game("A", 30, "X", 0) + _game("B", 3, "Y", 0),   # A and B never played each other
        team_ranks={"A": 9, "B": 2},
    )
    rules = _rules(two_team=[
        Step(step="head_to_head"),
        Step(step="external_ranking", params={"min_conference_games": 0}),
    ])
    outcome = order_tied_group(["A", "B"], ctx, rules)
    assert outcome.flat == ["B", "A"]                       # rank 2 beats rank 9
    assert outcome.resolved_by["B"] == "external_ranking"


def test_a_step_that_separates_nobody_is_treated_as_no_opinion():
    """A step can compute an answer and find every team equal. That is a single-group partition,
    which is no progress: acting on it would recurse on the same group forever. It must advance
    to the next step instead, and this test is what proves the chain does not stall there."""
    ctx = _ctx(
        _game("A", 21, "X", 0) + _game("B", 21, "Y", 0),   # identical 1-0 records, never met
        team_ranks={"A": 4, "B": 1},
    )
    rules = _rules(two_team=[
        Step(step="sub_group_record"),                       # computes; finds no separation
        Step(step="external_ranking", params={"min_conference_games": 0}),
    ])
    outcome = order_tied_group(["A", "B"], ctx, rules)
    assert outcome.flat == ["B", "A"]
    assert outcome.resolved_by["B"] == "external_ranking"


# ---------------------------------------------------------------------------
# Peel-off and restart -- the behaviour the source documents actually describe
# ---------------------------------------------------------------------------
def test_peeling_one_team_off_restarts_the_rest_on_the_two_team_chain():
    """Three tied teams, incomplete round robin: A beat both others, so A is peeled off. The
    source texts then send the remaining two to the FIRST step of the TWO-TEAM procedures, not to
    the next multi-team step.

    This is the test that distinguishes a real restart from a flat walk down one chain. B and C
    never played, so the two-team chain's head-to-head is silent and only its SECOND step can
    separate them -- and that step is absent from the multi-team chain entirely. B and C can
    therefore only come out ordered if the driver actually switched chains on the remainder.
    """
    rows = _game("A", 20, "B", 10) + _game("A", 20, "C", 10)   # B vs C never played
    ctx = _ctx(rows, team_ranks={"B": 3, "C": 12})
    rules = _rules(
        multi_team=[Step(step="sweep_in_out", when="not_round_robin_among_tied")],
        two_team=[
            Step(step="head_to_head"),                             # silent: B and C never met
            Step(step="external_ranking", params={"min_conference_games": 0}),
        ],
    )
    outcome = order_tied_group(["B", "C", "A"], ctx, rules)
    assert outcome.flat == ["A", "B", "C"]
    assert outcome.resolved_by["A"] == "sweep_in_out"
    # B and C could only be separated by a step that exists ONLY on the two-team chain, so this
    # is what proves the driver switched chains rather than walking one list to the end.
    assert outcome.resolved_by["B"] == "external_ranking"
    assert outcome.resolved_by["C"] == "external_ranking"


def test_when_predicate_gates_a_step_off_for_the_wrong_group_shape():
    """All three pairs played, so `round_robin_among_tied` fires and the not-round-robin step is
    skipped. Paired with the test below, which drives the opposite branch on an incomplete group;
    together they prove the two `when` values actually gate anything."""
    rows = _game("A", 20, "B", 10) + _game("A", 30, "C", 0) + _game("B", 30, "C", 0)
    ctx = _ctx(rows)
    rules = _rules(multi_team=[
        Step(step="sub_group_record", when="round_robin_among_tied"),
        Step(step="sweep_in_out", when="not_round_robin_among_tied"),
    ])
    outcome = order_tied_group(["C", "B", "A"], ctx, rules)
    assert outcome.flat == ["A", "B", "C"]                    # 2-0, 1-1, 0-2
    assert set(outcome.resolved_by.values()) == {"sub_group_record"}


def test_not_round_robin_branch_is_taken_when_the_group_is_incomplete():
    """The mirror of the test above, on a group where one pair never met. The round-robin step
    must be skipped and the sweep step must decide -- otherwise the two `when` values are not
    actually doing anything."""
    rows = _game("A", 20, "B", 10) + _game("A", 30, "C", 0)   # B and C never met
    ctx = _ctx(rows)
    rules = _rules(multi_team=[
        Step(step="sub_group_record", when="round_robin_among_tied"),
        Step(step="sweep_in_out", when="not_round_robin_among_tied"),
    ])
    outcome = order_tied_group(["C", "B", "A"], ctx, rules)
    assert outcome.flat[0] == "A"                              # A swept both
    assert outcome.resolved_by["A"] == "sweep_in_out"


def test_a_team_placed_below_can_never_climb_above_one_placed_higher():
    """The Big Ten's "shall not be pulled back into the tiebreaker for any future step"
    (bigten.txt:39-40). Here the later step would rank D best of all four if it ever saw the
    whole group -- D has the best rating -- but D was placed in the bottom band by the first
    step, so recursion only ever compares it against its own band.
    """
    rows = (
        _game("A", 20, "B", 10) + _game("A", 20, "C", 10) + _game("A", 20, "D", 10)
        + _game("B", 20, "C", 10) + _game("B", 20, "D", 10)
        + _game("C", 20, "D", 10)
    )
    ctx = _ctx(rows, team_ranks={"A": 40, "B": 30, "C": 20, "D": 1})
    rules = _rules(
        multi_team=[
            Step(step="sub_group_record", when="round_robin_among_tied"),
            Step(step="external_ranking", params={"min_conference_games": 0}),
        ],
        two_team=[Step(step="external_ranking", params={"min_conference_games": 0})],
    )
    outcome = order_tied_group(["A", "B", "C", "D"], ctx, rules)
    assert outcome.flat == ["A", "B", "C", "D"]      # 3-0, 2-1, 1-2, 0-3
    assert outcome.flat[-1] == "D", "best-rated team must stay last; its band was fixed earlier"


# ---------------------------------------------------------------------------
# Fallback (K7) and the no-config case (R4/AC7)
# ---------------------------------------------------------------------------
def test_exhausted_chain_falls_back_to_overall_record():
    """Two teams who never met, with no rating: only the overall-record fallback can order them.
    B is 3-0 overall against A's 1-2."""
    rows = (
        _game("A", 20, "X", 10, conference_game=False)
        + _game("A", 0, "Y", 10, conference_game=False)
        + _game("A", 0, "Z", 10, conference_game=False)
        + _game("B", 20, "P", 0, conference_game=False)
        + _game("B", 20, "Q", 0, conference_game=False)
        + _game("B", 20, "R", 0, conference_game=False)
    )
    ctx = _ctx(rows)
    outcome = order_tied_group(["A", "B"], ctx, _rules())
    assert outcome.flat == ["B", "A"]
    assert outcome.resolved_by == {"A": FALLBACK_LABEL, "B": FALLBACK_LABEL}


def test_fallback_ranks_a_bigger_record_above_an_equal_percentage():
    """The rule from the closed PR #18, which the repo owner stated as "if n is a number of
    wins, then N+1 and 0 is better than N and 0": 2-0 must beat 1-0, not tie with it. Both are
    1.000, so a percentage-only fallback would fall through to team name and order them
    alphabetically -- which is how USC ended up below a row of 1-0 teams on the live 2026 grid.
    """
    rows = (
        _game("Zeta", 20, "X", 0, conference_game=False)
        + _game("Zeta", 20, "Y", 0, conference_game=False)      # Zeta 2-0
        + _game("Alpha", 20, "P", 0, conference_game=False)     # Alpha 1-0
    )
    ctx = _ctx(rows)
    outcome = order_tied_group(["Alpha", "Zeta"], ctx, _rules())
    assert outcome.flat == ["Zeta", "Alpha"]


def test_fallback_prefers_a_perfect_short_record_over_a_longer_winning_one():
    """The other side of the same key: percentage comes FIRST, so 8-0 beats 9-3. Reversing the
    two terms would invert this, which is why both tests exist."""
    rows = []
    for i in range(8):
        rows += _game("Perfect", 20, f"O{i}", 0, conference_game=False)
    for i in range(9):
        rows += _game("Longer", 20, f"P{i}", 0, conference_game=False)
    for i in range(3):
        rows += _game("Longer", 0, f"Q{i}", 20, conference_game=False)
    ctx = _ctx(rows)
    assert order_tied_group(["Longer", "Perfect"], ctx, _rules()).flat == ["Perfect", "Longer"]


def test_fallback_ignores_postseason_results():
    """Bowl and playoff results must not affect standings placement. A goes 1-2 in the regular
    season and then wins three postseason games; B goes 2-1 and loses one. On regular-season
    record alone B is ahead, and the postseason must not overturn that.

    This is the rule the fallback is most likely to break quietly, because it only applies once
    the conference's own procedure has run out of opinions."""
    rows = (
        _game("A", 20, "W", 0, conference_game=False)
        + _game("A", 0, "X", 20, conference_game=False)
        + _game("A", 0, "Y", 20, conference_game=False)
        + _game("A", 40, "P1", 0, conference_game=False, season_type="postseason")
        + _game("A", 40, "P2", 0, conference_game=False, season_type="postseason")
        + _game("A", 40, "P3", 0, conference_game=False, season_type="postseason")
        + _game("B", 20, "M", 0, conference_game=False)
        + _game("B", 20, "N", 0, conference_game=False)
        + _game("B", 0, "O", 20, conference_game=False)
        + _game("B", 0, "P4", 40, conference_game=False, season_type="postseason")
    )
    ctx = _ctx(rows)
    assert order_tied_group(["A", "B"], ctx, _rules()).flat == ["B", "A"]


def test_fallback_ignores_conference_championship_games():
    """A championship game is season_type 'regular' in CFBD's data, so season_type alone does not
    exclude it -- the caller passes its game_id. Here A's only win is the title game and B's only
    win is a regular one, so excluding it must put B ahead."""
    rows = (
        _game("A", 20, "B", 10, conference_game=False, game_id=999)      # the championship game
        + _game("B", 20, "Z", 0, conference_game=False, game_id=1)
    )
    ctx = _ctx(rows, placement_excluded_game_ids=frozenset({999}))
    assert order_tied_group(["A", "B"], ctx, _rules(two_team=[])).flat == ["B", "A"]
    # Without the exclusion the same data orders them the other way, which is what makes the
    # assertion above about the exclusion rather than about anything else in the fixture.
    bare = _ctx(rows)
    assert order_tied_group(["A", "B"], bare, _rules(two_team=[])).flat == ["A", "B"]


def test_fallback_uses_rating_before_name_and_puts_unranked_last():
    """Identical records, so rating decides; and an unranked team must sort LAST rather than
    being treated as rank 0, which would make it best."""
    ctx = _ctx([], team_ranks={"Ranked": 5, "Unranked": None})
    assert order_tied_group(["Unranked", "Ranked"], ctx, _rules()).flat == ["Ranked", "Unranked"]


def test_no_rule_set_orders_by_fallback_and_invents_nothing():
    """`rules=None` is the conference-without-a-supplied-procedure case, and the ACC before it
    dropped divisions (plan R4/AC7). Every team must be labelled as fallback-resolved, so a
    caller can tell this apart from a real tiebreaker result."""
    rows = _game("A", 20, "B", 10)
    # A is 1-0 overall and B is 0-1, so the record term of the fallback puts A first. B has the
    # far better RATING, which is what makes this discriminating: rating is consulted only after
    # record, so a fallback that checked rating first would return ["B", "A"] here.
    ctx = _ctx(rows, team_ranks={"A": 50, "B": 1})
    outcome = order_tied_group(["A", "B"], ctx, rules=None)
    assert outcome.flat == ["A", "B"]
    assert outcome.restart_policy is None
    assert set(outcome.resolved_by.values()) == {FALLBACK_LABEL}


# ---------------------------------------------------------------------------
# AC8: termination
# ---------------------------------------------------------------------------
def test_fully_tied_conference_where_no_step_separates_anyone_still_terminates():
    """AC8. Eighteen teams, no games at all, a chain of every real step name, and no ratings --
    so nothing can separate anyone. It must return a complete order rather than hang or raise."""
    teams = [f"Team{i:02d}" for i in range(18)]
    ctx = _ctx([])
    every_step = [
        Step(step="head_to_head"),
        Step(step="sub_group_record"),
        Step(step="sweep_in_out"),
        Step(step="common_opponents_record", params={"min_sample": 2}),
        Step(step="vs_placed_opponents", params={"direction": "descending"}),
        Step(step="opponents_cumulative_conf_pct"),
        Step(step="capped_relative_scoring_margin"),
        Step(step="total_wins_capped"),
        Step(step="external_ranking", params={"min_conference_games": 4}),
        Step(step="random_draw"),
    ]
    outcome = order_tied_group(teams, ctx, _rules(two_team=every_step, multi_team=every_step))
    assert sorted(outcome.flat) == sorted(teams)
    assert len(outcome.flat) == len(teams)
    assert set(outcome.resolved_by.values()) == {FALLBACK_LABEL}


def test_every_step_name_in_the_registry_is_driveable():
    """Each real step name, alone in a chain, must run without raising on a group of three with
    no games -- the shape a week-1 conference actually has. This is what catches a primitive
    whose signature the driver cannot call (a missing **params, a required argument)."""
    ctx = _ctx([])
    for name in sorted(KNOWN_STEPS):
        rules = _rules(two_team=[Step(step=name)], multi_team=[Step(step=name)])
        outcome = order_tied_group(["A", "B", "C"], ctx, rules)
        assert sorted(outcome.flat) == ["A", "B", "C"], name


def test_unknown_step_name_is_skipped_not_raised(caplog):
    """Config validation makes this unreachable, but a hand-built rule set must degrade to the
    fallback rather than take down an artifact publish."""
    import logging

    ctx = _ctx([], team_ranks={"A": 1, "B": 2})
    rules = _rules(two_team=[Step(step="not_a_real_step")])
    with caplog.at_level(logging.ERROR, logger="cfb_lp"):
        outcome = order_tied_group(["A", "B"], ctx, rules)
    assert outcome.flat == ["A", "B"]
    assert "not_a_real_step" in caplog.text


# ---------------------------------------------------------------------------
# Property test: the output is always a permutation of the input
# ---------------------------------------------------------------------------
def test_output_is_always_a_permutation_for_randomised_groups():
    """The plan's property test. Random schedules, random ratings, random group sizes, driven
    through the real SEC chain. `order_tied_group` asserts this internally too; running it over
    randomised input is what makes that assertion worth having."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    sec = rules_for(cfg, "SEC", SEASON)
    rng = random.Random(20250911)

    for trial in range(300):
        size = rng.randint(2, 7)
        teams = [f"T{i}" for i in range(size)]
        rows = []
        for a, b in itertools.combinations(teams, 2):
            if rng.random() < 0.6:                      # a partially-played schedule
                sa, sb = rng.randint(0, 45), rng.randint(0, 45)
                if sa == sb:
                    sa += 1
                rows += _game(a, sa, b, sb)
        conf_records = {}
        for team in teams:
            wins = sum(1 for r in rows if r["team"] == team and r["status"] == "win")
            losses = sum(1 for r in rows if r["team"] == team and r["status"] == "loss")
            conf_records[team] = (wins, losses)
        ctx = _ctx(
            rows,
            conf_records=conf_records,
            team_ranks={t: rng.choice([rng.randint(1, 130), None]) for t in teams},
            frozen_order=sorted(teams, key=lambda t: -conf_records[t][0]),
            conference="SEC",
        )
        outcome = order_tied_group(teams, ctx, sec)
        assert sorted(outcome.flat) == sorted(teams), f"trial {trial}: {outcome.flat}"
        assert set(outcome.resolved_by) == set(teams), f"trial {trial} missing resolved_by"


# ---------------------------------------------------------------------------
# Against the real shipped config
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("conference", ["SEC", "Big 12", "Big Ten", "ACC"])
def test_real_power4_chains_run_end_to_end(conference):
    """Every shipped chain must be driveable with the parameters the config actually sets. This
    is the seam test between T1's config and T2's primitives: a param the config passes that a
    primitive does not accept, or a value it rejects, fails here and nowhere else."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    rules = rules_for(cfg, conference, SEASON if conference != "ACC" else 2026)
    assert rules is not None, conference
    teams = ["A", "B", "C", "D"]
    rows = []
    for a, b in itertools.combinations(teams, 2):
        rows += _game(a, 21, b, 14)                 # a complete round robin, decisive scores
    ctx = _ctx(
        rows,
        conf_records={t: (3, 0) for t in teams},
        team_ranks={"A": 4, "B": 3, "C": 2, "D": 1},
        frozen_order=teams,
        conference=conference,
        non_fbs_teams=frozenset(),
    )
    outcome = order_tied_group(teams, ctx, rules)
    assert sorted(outcome.flat) == sorted(teams)
    assert set(outcome.resolved_by) == set(teams)


def test_acc_2025_five_way_is_not_decided_by_head_to_head():
    """The real 2025 ACC five-way tie at 6-2. Duke was LAST within the group on head-to-head
    (0-1) and still advanced, because the pre-amendment chain reached the opponents'-combined-
    conference-record step, where Duke's eight ACC opponents went 32-32 (.500), the best of the
    five. So whatever the opening step does here, it must NOT resolve the group -- and Duke in
    particular must not be placed last by it.

    This test asserts the negative because the positive needs the full 2025 ACC schedule, which
    belongs in T6's acceptance fixtures. What it pins is the property that made the original
    reading of this tie wrong for an entire research pass.
    """
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    acc_2025 = rules_for(cfg, "ACC", 2025)
    assert acc_2025 is not None

    five = ["Duke", "Georgia Tech", "Miami", "Pittsburgh", "SMU"]
    # Duke played exactly one of the other four, and lost it. Nobody else in the group met.
    rows = _game("Georgia Tech", 27, "Duke", 20)
    ctx = _ctx(
        rows,
        conf_records={t: (6, 2) for t in five},
        team_ranks={t: None for t in five},
        frozen_order=five,
        conference="ACC",
    )
    outcome = order_tied_group(five, ctx, acc_2025)
    opening_steps = {"sub_group_record", "sweep_in_out"}
    assert outcome.resolved_by["Duke"] not in opening_steps, (
        "the opening intra-group step must have no opinion on this shape; a bare head-to-head "
        "reading would have placed Duke last and it actually won the tie"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
