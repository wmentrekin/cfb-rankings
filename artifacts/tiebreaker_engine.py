"""
tiebreaker_engine.py -- T3 of docs/conference-tiebreakers/plan.yaml: the recursive peel-off
driver that turns a conference's configured step chain into an ordered partition of a tied group.

WHAT THIS MODULE IS FOR
-----------------------
`artifacts/tiebreaker_steps.py` implements the individual MEASURES (head-to-head, common
opponents, ...). `artifacts/conference_tiebreakers.json` records, per conference and per season,
WHICH measures to apply and in WHAT ORDER. This module is the thing that walks that chain: it
applies steps in order, and when one separates the group it recurses into each resulting
sub-group, which is what the source documents mean by "revert to the beginning of the applicable
procedures".

THE ONE REFRAMING THAT MATTERS
------------------------------
Every published tiebreaker procedure is written to SELECT CHAMPIONSHIP PARTICIPANTS -- it answers
"which one or two teams go to the title game", and its language is about teams being "selected"
or "eliminated". What this project needs is different and strictly harder: a TOTAL ORDER over the
whole tied group, all season long, because these are the conference standings and not a
December bracket decision (the repo owner: "these tie breakers always apply not just for the
final week title game stuff, these rules dictate conference standings at all times").

The bridge is that a step's partition is read as an ORDERING between groups rather than as a
selection event:

  - "Team X is selected"  -> X is placed above everyone remaining.
  - "Team Y is eliminated" -> Y is placed below everyone still in contention. Y is NOT dropped;
                              it still needs a position, so the chain re-runs on the teams that
                              share its band.
  - "The remaining teams revert to the beginning of the applicable procedures"
                           -> recurse on that sub-group, which re-enters at step 1 of the chain
                              matching ITS size (two teams -> the two-team chain; three or more
                              -> the multi-team chain).

Under that reading, the Big Ten's rule that an eliminated team "SHALL NOT be pulled back into the
tiebreaker for any future step" (bigten.txt:39-40) is structural rather than a flag to check:
recursion happens strictly WITHIN a group, so a team can never cross above a team that an earlier
step placed in a higher group. `MultiTeamRules.eliminated_teams_locked` is therefore not consulted
here -- see LIMITS below, where the one config field this module deliberately does not act on is
explained rather than quietly ignored.

TERMINATION
-----------
A step's result is only acted on when it splits the group into two or more non-empty sub-groups.
Every such sub-group is therefore STRICTLY smaller than the group that produced it, so recursion
depth is bounded by the initial group size and each recursive call runs on a smaller input. A step
that computes an answer but separates nobody (a single group containing everyone) is treated
exactly like a step that declined to speak: advance to the next step, do not recurse. That is the
whole termination argument, and `_MAX_DEPTH` below is belt-and-braces against a future primitive
that violates the partition contract, not load-bearing.

FALLBACK (K7)
-------------
When the chain is exhausted -- every step gated, silent, or unable to separate -- ordering falls
back to overall record, then our own rating, then team name. This is what the repo owner asked
for: "overall win-loss record should only matter before the tie breaker is applicable (like early
in the season now when a lot of teams havent played a conference game yet, or there isnt enough
info to relate teams)". Because team names are unique, the fallback is total, so this module
always returns a complete ordering and never reports an unresolved group.

The overall-record term compares (win percentage, then win COUNT), which is deliberate: it is the
rule from the closed PR #18 that a bigger identical-percentage record is better, so 2-0 sorts
above 1-0 rather than tying with it.

LIMITS -- things this module does NOT do, stated rather than implied
--------------------------------------------------------------------
1. `RuleSet.tie_definition` and the ACC's `restart_at: "redefine_tied_teams"` are not acted on
   here, because neither is answerable from a single tied group. The ACC defines its tied set as
   the best-win-percentage team(s) PLUS any team that played a different number of conference
   games and has the same number of wins or the same number of losses (acc.txt section 1), and
   its restart re-derives that set each time. Both need the whole conference table, which is the
   CALLER's data, not this module's input. `order_tied_group` reports the policy back on its
   result so the caller can act on it; it does not pretend to have implemented it.
2. `MultiTeamRules.eliminated_teams_locked` is structurally guaranteed rather than enforced, per
   the reframing above.
3. The SEC and Big Ten distinguish the two-team procedure FOR FIRST PLACE from the two-team
   procedure FOR SECOND PLACE. The config carries a single `two_team` chain per conference, so
   that distinction is not represented; a group of two always runs `two_team`. No source text
   supplied so far gives the two variants different STEPS, only different entry contexts.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from artifacts.tiebreaker_rules import RuleSet, Step
from artifacts.tiebreaker_steps import STEP_REGISTRY, TiebreakContext, satisfies_when

logger = logging.getLogger("cfb_lp")

# Defensive only: termination is guaranteed by strict group-size decrease (see module docstring).
# This catches a future primitive that returns a "partition" containing a group as large as its
# input, which would otherwise recurse forever.
_MAX_DEPTH = 32

# The label recorded in `resolved_by` for a team whose position the configured chain could not
# fix, and which the overall-record/rating/name fallback ordered instead (K8).
FALLBACK_LABEL = "fallback"


@dataclass(frozen=True)
class TiebreakOutcome:
    """The result of running a conference's chain over one tied group.

    `order` is an ordered partition, best group first. Because the fallback is total, every group
    is a singleton in practice; the nested shape is kept so a caller can still distinguish "these
    teams are genuinely tied" from "these teams are adjacent" if a future fallback stops being
    total.

    `resolved_by` maps each team to the name of the step that fixed its position, or
    FALLBACK_LABEL. This is the field that makes a test able to assert the RIGHT step decided,
    not merely that the order came out right by luck -- the failure mode that let an incorrect
    reading of the ACC's procedure look correct for an entire research pass (plan R2).

    `restart_policy` echoes the rule set's `multi_team.restart_at` so a caller can act on the
    ACC's `redefine_tied_teams` regrouping, which this module cannot do alone (LIMITS 1).
    """

    order: List[List[str]]
    resolved_by: Dict[str, str] = field(default_factory=dict)
    restart_policy: Optional[str] = None

    @property
    def flat(self) -> List[str]:
        """The ordering as a flat list, best first."""
        return [team for group in self.order for team in group]


# ---------------------------------------------------------------------------
# Fallback ordering (K7)
# ---------------------------------------------------------------------------
def _overall_record(ctx: TiebreakContext, team: str) -> Tuple[int, int]:
    """(wins, losses) across this team's played games this season, conference or not, EXCLUDING
    anything that must not affect standings placement.

    Two exclusions, both required by the repo owner's rule that bowl and playoff results must not
    affect where a team is placed in its conference standings:

      - any row with season_type == 'postseason';
      - any row whose game_id is in `ctx.placement_excluded_game_ids`, which the caller populates
        with the season's conference championship games. Those are season_type 'regular' in
        CFBD's data, so season_type alone does not catch them.

    This mirrors `artifacts/schedule.py::_placement_pct`, which applies the same two exclusions
    to the primary standings key. Without it the fallback would quietly reintroduce postseason
    results at exactly the point where the conference's own procedure ran out of opinions --
    the least visible place for that rule to break.

    Counts only the team's own-perspective rows. Every schedule_grid game appears twice, once
    from each side, so filtering on `team` is what keeps a single game from being counted for
    both participants -- the same discipline `_conf_game_rows` applies in tiebreaker_steps.
    """
    wins = losses = 0
    for row in ctx.rows:
        if row.get("season") != ctx.season or row.get("team") != team:
            continue
        if row.get("season_type") == "postseason":
            continue
        if row.get("game_id") in ctx.placement_excluded_game_ids:
            continue
        status = row.get("status")
        if status == "win":
            wins += 1
        elif status == "loss":
            losses += 1
    return wins, losses


def _fallback_sort_key(ctx: TiebreakContext, team: str) -> Tuple[float, int, float, str]:
    """Overall record, then our rating, then name -- in that order, and total by construction.

    The record term is (win percentage, win count) rather than win percentage alone. That second
    element is the fix from the closed PR #18: at equal percentage the larger record is better, so
    2-0 sorts above 1-0 instead of tying with it. Percentage has to come first regardless, or a
    9-3 team would outrank an unbeaten 8-0 one on raw win count.

    A team with no played games gets a percentage of 0.0, which places it below any team with a
    win and above nothing -- it cannot be distinguished further by record, so rating decides.

    Rating is `ctx.team_ranks` (1 = best), so it is negated nowhere and sorted ascending; an
    unranked team sorts last via infinity rather than being treated as rank 0 (the best possible),
    which is the sign error this comment exists to prevent.
    """
    wins, losses = _overall_record(ctx, team)
    played = wins + losses
    win_pct = (wins / played) if played else 0.0
    rank = ctx.team_ranks.get(team)
    rank_key = float(rank) if rank is not None else float("inf")
    return (-win_pct, -wins, rank_key, team)


def _fallback_order(group: List[str], ctx: TiebreakContext) -> List[List[str]]:
    """Total order over `group`, one team per position."""
    return [[team] for team in sorted(group, key=lambda t: _fallback_sort_key(ctx, t))]


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------
def _chain_for(group: List[str], rules: RuleSet) -> Tuple[Step, ...]:
    """The step list applicable to a group of this size.

    Two teams take the two-team chain; three or more take the multi-team chain. This is the
    "applicable procedures" of every source document's restart language, and it is re-evaluated
    on each recursive call, which is what makes a three-way tie reduced to two teams continue
    under the two-team rules rather than the multi-team ones.
    """
    return rules.two_team if len(group) == 2 else rules.multi_team.steps


def _apply_step(step: Step, group: List[str], ctx: TiebreakContext) -> Optional[List[List[str]]]:
    """Run one configured step, or return None if it does not apply or has no opinion.

    Three distinct reasons for None, all treated the same by the caller (advance to the next
    step) but logged differently because they mean different things when debugging a standings
    order that looks wrong:
      - the `when` predicate excludes this step for this group;
      - the primitive returned None, i.e. a gate inside it was not met (too few common opponents,
        too few conference games played, a roster it needed was absent);
      - the step name is not in STEP_REGISTRY, which config validation should have made
        impossible and so is logged as an error.
    """
    if not satisfies_when(step.when, group, ctx):
        return None

    fn = STEP_REGISTRY.get(step.step)
    if fn is None:
        # load_conference_rules validates every step name against STEP_REGISTRY, so reaching here
        # means the config was built by hand or the registry shrank. Log and skip rather than
        # raise: an unknown step must not take down an entire artifact publish, and skipping
        # degrades to the next step and ultimately the total fallback.
        logger.error(
            "tiebreaker_engine: step %r is not in STEP_REGISTRY; skipping it for "
            "conference=%s season=%s group=%s",
            step.step, ctx.conference, ctx.season, group,
        )
        return None

    result = fn(group, ctx, **step.params)
    if result is None:
        return None

    partition = [list(sub) for sub in result if sub]
    if len(partition) <= 1:
        # The step computed an answer and separated nobody. Distinct from None in meaning (see
        # tiebreaker_steps' contract) but identical in consequence: no progress, so the next step
        # gets a turn. Acting on it would recurse on the same group and never terminate.
        return None
    return partition


def _resolve(
    group: List[str],
    ctx: TiebreakContext,
    rules: RuleSet,
    resolved_by: Dict[str, str],
    depth: int,
) -> List[List[str]]:
    """Order `group`, recording in `resolved_by` which step fixed each team's position."""
    if len(group) <= 1:
        return [list(group)] if group else []

    if depth >= _MAX_DEPTH:
        # Unreachable while every primitive honours the partition contract; see TERMINATION.
        logger.error(
            "tiebreaker_engine: recursion depth %s reached for conference=%s season=%s "
            "group=%s; falling back. A step primitive is returning a sub-group no smaller "
            "than its input.",
            depth, ctx.conference, ctx.season, group,
        )
        return _mark_fallback(group, ctx, resolved_by)

    for step in _chain_for(group, rules):
        partition = _apply_step(step, group, ctx)
        if partition is None:
            continue

        ordered: List[List[str]] = []
        for sub in partition:
            if len(sub) == 1:
                resolved_by[sub[0]] = step.step
                ordered.append(sub)
            else:
                # Guaranteed by len(partition) >= 2: every sub-group is strictly smaller than
                # `group`, so this recursion terminates. Asserted because it is the property the
                # whole module's termination rests on.
                assert len(sub) < len(group), (
                    f"step {step.step!r} returned sub-group {sub} not smaller than its input "
                    f"{group}; this breaks the termination guarantee"
                )
                ordered.extend(_resolve(sub, ctx, rules, resolved_by, depth + 1))
        return ordered

    return _mark_fallback(group, ctx, resolved_by)


def _mark_fallback(
    group: List[str], ctx: TiebreakContext, resolved_by: Dict[str, str]
) -> List[List[str]]:
    for team in group:
        resolved_by[team] = FALLBACK_LABEL
    return _fallback_order(group, ctx)


def order_tied_group(
    tied: List[str], ctx: TiebreakContext, rules: Optional[RuleSet]
) -> TiebreakOutcome:
    """Order a group of teams tied on conference win percentage, per `rules`.

    Args:
        tied: the team names to order. One or zero teams is returned unchanged.
        ctx: the game rows, records and ranks the step primitives read.
        rules: the conference's rule set for this season, from
            `tiebreaker_rules.rules_for(...)`. **None means no configured procedure** -- for a
            conference whose rules nobody has supplied, or a season outside the era a rule set
            covers, such as the ACC before it dropped divisions. In that case the group is
            ordered by the fallback alone, every team is labelled FALLBACK_LABEL, and nothing is
            invented on the conference's behalf (plan R4/AC7).

    Returns:
        TiebreakOutcome. Always a complete ordering of `tied`, because the fallback is total.
    """
    resolved_by: Dict[str, str] = {}
    restart_policy = rules.multi_team.restart_at if rules is not None else None

    if len(tied) <= 1:
        return TiebreakOutcome(
            order=[[team] for team in tied],
            resolved_by={},
            restart_policy=restart_policy,
        )

    if rules is None:
        logger.info(
            "tiebreaker_engine: no rule set for conference=%s season=%s; ordering %s teams by "
            "the overall-record fallback.",
            ctx.conference, ctx.season, len(tied),
        )
        order = _mark_fallback(list(tied), ctx, resolved_by)
    else:
        order = _resolve(list(tied), ctx, rules, resolved_by, depth=0)

    _assert_permutation(tied, order)
    return TiebreakOutcome(order=order, resolved_by=resolved_by, restart_policy=restart_policy)


def _assert_permutation(tied: List[str], order: List[List[str]]) -> None:
    """The output must contain exactly the input teams, once each.

    Cheap, and it catches the failure mode that matters most in a standings renderer: a team
    silently vanishing from, or being duplicated within, its conference table because a
    primitive returned an overlapping or incomplete partition.
    """
    flat = [team for group in order for team in group]
    if sorted(flat) != sorted(tied):
        raise AssertionError(
            f"tiebreaker_engine produced {flat} for input {list(tied)}; the result must be a "
            "permutation of the input"
        )


def describe_outcome(outcome: TiebreakOutcome) -> str:
    """One line per team, for logs and for reading a test failure without a debugger."""
    return "\n".join(
        f"{position:>2}. {team:<28} resolved_by={outcome.resolved_by.get(team, '-')}"
        for position, team in enumerate(outcome.flat, 1)
    )


__all__ = [
    "FALLBACK_LABEL",
    "TiebreakOutcome",
    "describe_outcome",
    "order_tied_group",
]
