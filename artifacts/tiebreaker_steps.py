"""
tiebreaker_steps.py -- T2 of docs/conference-tiebreakers/plan.yaml: the shared, generic
operation library for conference championship tiebreakers.

SCOPE (hard, per the T2 handoff): this module implements the ten step PRIMITIVES only. It knows
nothing about the ORDER in which a conference calls them, or what "restart at step N" means --
that sequencing is the recursive peel-off driver (T3, a separate module, separate agent, not
touched here). Different conferences call the same operations in different orders; this module
is the operations.

SOURCE OF TRUTH: docs/conference-tiebreakers/source-rules/{sec,big12,bigten,acc}.txt, supplied
directly by the repo owner. They are authoritative over both the original task brief AND a
mid-task correction message from the orchestrator -- both were checked against the actual text
below, and every place they turned out to disagree with the source files is called out inline
and in the task report, not silently reconciled.

CONTRACT
--------
StepResult = Optional[List[List[str]]]   # ordered partition of team names, best group first.
                                          # None = "this step has no opinion" -> driver skips it.
                                          # A single group containing everyone is a DIFFERENT,
                                          # deliberate outcome from None: it means the step was
                                          # actually computed and found no separation, as opposed
                                          # to a gate (min_sample, round-robin completeness, ...)
                                          # that never got far enough to compute anything at all.
                                          # Every primitive below preserves that distinction.

Every function has the signature (tied: List[str], ctx: TiebreakContext, **params) ->
StepResult, and is registered in STEP_REGISTRY under the exact name a config's `step` key names.

ROW-LEVEL DISCIPLINE (reused, not imported): artifacts/schedule.py's `_head_to_head_winner`
already fixed the "one played game becomes two team-oriented rows" bug once (T2/K8 in that
module's own history) -- a split series must be a wash, and tallying both perspectives of the
same game reads a single meeting as a 2-0 sweep. This module re-derives that same discipline
independently in `_conf_game_rows` / `_two_team_h2h` below, rather than importing from
artifacts/schedule.py: that file is forbidden scope for T2, and more importantly the eventual
driver (T3) gets wired INTO schedule.py by T4, so this module importing schedule.py would set up
a real circular-import risk. Every helper below filters to ONE team's own-perspective rows
before tallying, exactly like the original fix.

INPUT ROW SHAPE (schedule_grid, see database/migrations/0003_schedule_grid_view.sql): each row
carries at least season (int), team (str), opponent (str), status ('win'|'loss'|'upcoming'),
conference_game (bool), team_score (int|None), opp_score (int|None). A row's `conference` value
is the TEAM's own conference for that game, not necessarily ctx.conference -- but since
conference_game is True only when both sides share a conference, and every team this module is
asked about is itself a member of ctx.conference, filtering on conference_game alone is
sufficient (matching the existing _head_to_head_winner convention, which does the same).
"""

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("cfb_lp")

StepResult = Optional[List[List[str]]]


@dataclass(frozen=True)
class TiebreakContext:
    rows: List[Dict[str, Any]]                  # schedule_grid rows, all season_types
    season: int
    conference: str                              # raw conference value, e.g. "SEC"
    frozen_order: List[str]                       # teams ordered by conf win pct ALONE, frozen
                                                   # before any tiebreaker step ran -- never the
                                                   # in-progress order (K4, circular otherwise).
    conf_records: Dict[str, Tuple[int, int]]       # team -> (conf wins, conf losses)
    team_ranks: Dict[str, Optional[int]]           # our own model rank; 1 = best; may be missing
    non_fbs_teams: Optional[frozenset] = None      # school names in the `non_fbs_teams` table for
                                                   # this season -- the FCS/lower-division roster.
                                                   # None means NOT SUPPLIED (the caller did not
                                                   # load it), which is deliberately distinct from
                                                   # an empty frozenset (supplied, and nobody in
                                                   # the conference played a non-FBS opponent).
                                                   # total_wins_capped relies on that distinction.


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _conf_game_rows(ctx: TiebreakContext, team: str) -> List[Dict[str, Any]]:
    """`team`'s own-perspective, played (win/loss), conference-game rows this season. Every
    helper below builds from this rather than scanning ctx.rows directly, so the
    own-perspective/played/conference-game filter is applied exactly once, consistently."""
    return [
        r for r in ctx.rows
        if r.get("season") == ctx.season
        and r.get("team") == team
        and r.get("conference_game")
        and r.get("status") in ("win", "loss")
    ]


def _two_team_h2h(ctx: TiebreakContext, team_a: str, team_b: str) -> Optional[str]:
    """Series winner between two teams this season, or None on 0-0 (never played) or any genuine
    split (1-1, 2-2, ...) -- a split series is a wash, not a tiebreak. Tallies ONLY team_a's own
    rows filtered to opponent == team_b, so the same game's row on team_b's side (opposite
    `status`, same game_id) is never separately counted -- see module docstring."""
    wins_a = wins_b = 0
    for row in _conf_game_rows(ctx, team_a):
        if row.get("opponent") != team_b:
            continue
        if row["status"] == "win":
            wins_a += 1
        else:
            wins_b += 1
    if wins_a > wins_b:
        return team_a
    if wins_b > wins_a:
        return team_b
    return None


def _is_complete_round_robin(ctx: TiebreakContext, tied: List[str]) -> bool:
    """True iff every pair of `tied` teams played each other at least once this season."""
    for a, b in itertools.combinations(tied, 2):
        if not any(r.get("opponent") == b for r in _conf_game_rows(ctx, a)):
            return False
    return True


def _group_by_value(ordered_teams: List[str], value_of: Dict[str, float]) -> List[List[str]]:
    """Collapses a value-sorted team list into contiguous groups of equal value. Sort is stable,
    so within a group teams keep their relative order from the caller's original `tied` list --
    deterministic, but not itself a ranking signal (a group IS a tie)."""
    groups: List[List[str]] = []
    for team in ordered_teams:
        if groups and abs(value_of[groups[-1][-1]] - value_of[team]) < 1e-9:
            groups[-1].append(team)
        else:
            groups.append([team])
    return groups


def _partition_by_value(
    tied: List[str], value_of: Dict[str, Optional[float]], descending: bool = True
) -> StepResult:
    """Turns a team -> Optional[float] map into a StepResult: teams with a real value are sorted
    (best first) and grouped by exact-tie value; teams with value None (no data for this step --
    e.g. unranked, or no games against the relevant opponent set) form one trailing group,
    "equally no data" rather than ranked among themselves. Returns None only when NO team has a
    real value at all (the step could not compute anything, as opposed to computing and finding
    everyone equal, which returns a real single group -- see module docstring)."""
    scored = [t for t in tied if value_of.get(t) is not None]
    unscored = [t for t in tied if value_of.get(t) is None]
    if not scored:
        return None
    scored.sort(key=lambda t: value_of[t], reverse=descending)
    groups = _group_by_value(scored, value_of)
    if unscored:
        groups.append(sorted(unscored))
    return groups


# ---------------------------------------------------------------------------
# 1. head_to_head
# ---------------------------------------------------------------------------
def head_to_head(tied: List[str], ctx: TiebreakContext, **params) -> StepResult:
    """Exactly two teams: the one that won their conference series. None if they never played,
    or split (a split series is a wash -- see _two_team_h2h). Any group size other than 2 is out
    of this primitive's defined scope and returns None (a config that reaches this step with a
    3+ group is a sequencing bug for sub_group_record/sweep_in_out to have caught first)."""
    if len(tied) != 2:
        return None
    team_a, team_b = tied
    winner = _two_team_h2h(ctx, team_a, team_b)
    if winner is None:
        return None
    loser = team_b if winner == team_a else team_a
    return [[winner], [loser]]


# ---------------------------------------------------------------------------
# 2. sub_group_record
# ---------------------------------------------------------------------------
def sub_group_record(
    tied: List[str], ctx: TiebreakContext, require_round_robin: bool = False, **params
) -> StepResult:
    """Win percentage in games among the tied teams only (three or more; for two, see
    head_to_head). `require_round_robin=True` gates the whole step on a COMPLETE round robin
    among the tied teams (every pair played at least once) -- if incomplete, returns None so the
    driver falls through to sweep_in_out, matching every source document's shape: sec.txt A.1 vs
    A.2, big12.txt multi-team step a (implicit complete case vs explicit a.1/a.2 incomplete
    case), bigten.txt B.1 (a)/(b) -- all three describe "if complete, compare win pct; if not,
    use the sweep rule instead", which is exactly [sub_group_record(require_round_robin=True),
    sweep_in_out] in this vocabulary. `require_round_robin=False` (default) skips that gate and
    computes win pct from whatever tied-group games were actually played, regardless of
    completeness -- kept for a config whose text does not make the same complete/incomplete
    distinction; not exercised by any of the four supplied source documents, all of which do."""
    if len(tied) < 3:
        return None
    if require_round_robin and not _is_complete_round_robin(ctx, tied):
        return None
    tied_set = set(tied)
    pct: Dict[str, Optional[float]] = {}
    for team in tied:
        wins = losses = 0
        for row in _conf_game_rows(ctx, team):
            if row.get("opponent") not in tied_set:
                continue
            if row["status"] == "win":
                wins += 1
            else:
                losses += 1
        pct[team] = (wins / (wins + losses)) if (wins + losses) else None
    return _partition_by_value(tied, pct)


# ---------------------------------------------------------------------------
# 3. sweep_in_out
# ---------------------------------------------------------------------------
def sweep_in_out(tied: List[str], ctx: TiebreakContext, **params) -> StepResult:
    """For an INCOMPLETE round robin among 3+ tied teams: a team that beat every tied opponent it
    actually played is promoted into its own top group; a team that lost to every tied opponent
    it actually played is demoted into its own bottom group; everyone else (split, or never
    played a tied opponent) lands in one unseparated middle group. Returns None -- not a guess --
    when the round robin IS complete (sub_group_record applies instead, per every source
    document), when fewer than 3 teams are tied, or when no team swept or was swept (matching
    sec.txt A.2.c: "if no team either beat all or lost to all -> all tied teams advance to the
    next step", i.e. this step had no opinion).

    BOTH-SIDED, per source text in sec.txt A.2(a)/(b) and acc.txt Sec 2.b[sic].ii.1 explicitly
    ("the tied team which lost to each of the other Tied Teams is removed from the tie").

    SOURCE DISCREPANCY, reported rather than smoothed over: big12.txt multi-team step a ONLY
    contains the promote half -- a.1 promotes a lone sweeper ("removed from the tiebreaker...
    remaining teams revert to the beginning"); a.2 ("no team defeated all other teams... move to
    the next step") has NO demote/eliminate branch for a lone total-loser at all. An
    orchestrator correction mid-task asserted Big 12 confirms both halves; the supplied
    big12.txt text does not support that claim for the demote half, and per this module's
    instructions the source file wins. This primitive still implements both halves as a single
    general operation (SEC and ACC alone justify that), but a Big 12 config step that reaches
    this function should not rely on its demote output producing a real Big 12 outcome -- that
    is a config-authoring concern for a later task, not something enforceable from inside a
    primitive with no knowledge of which conference called it.

    WORDING NUANCE, also reported: Big 12 says the sweeper is "removed... and the remaining teams
    revert to the beginning" (placed atop a STILL-CONTINUING ranking); SEC says "selected"
    (chosen directly as a championship participant, ending its involvement). Both are
    driver-level sequencing concerns; this primitive only returns an ordered partition, best
    group first, and is agnostic to what the driver does with a solo top group.

    MULTIPLE SIMULTANEOUS SWEEPERS/TOTAL-LOSERS (e.g. A beat B, C beat D, A and C never played):
    not addressed by any source document. All qualifying sweepers are grouped together as one
    tied top group (not guessed at an order among them); likewise for total-losers at the bottom.
    """
    if len(tied) < 3:
        return None
    if _is_complete_round_robin(ctx, tied):
        return None
    tied_set = set(tied)
    top: List[str] = []
    bottom: List[str] = []
    middle: List[str] = []
    for team in tied:
        played, beat, lost = set(), set(), set()
        for row in _conf_game_rows(ctx, team):
            opp = row.get("opponent")
            if opp not in tied_set or opp == team:
                continue
            played.add(opp)
            (beat if row["status"] == "win" else lost).add(opp)
        if played and beat == played:
            top.append(team)
        elif played and lost == played:
            bottom.append(team)
        else:
            middle.append(team)
    if not top and not bottom:
        return None
    groups: List[List[str]] = []
    if top:
        groups.append(top)
    if middle:
        groups.append(middle)
    if bottom:
        groups.append(bottom)
    return groups


# ---------------------------------------------------------------------------
# 4. common_opponents_record
# ---------------------------------------------------------------------------
def common_opponents_record(
    tied: List[str], ctx: TiebreakContext, min_sample: int = 2, **params
) -> StepResult:
    """Win percentage against opponents faced by ALL tied teams (excluding the tied teams
    themselves). Returns None if the common set is smaller than `min_sample` (default 2).

    This gate is load-bearing and empirically justified (per task brief): the real 2025 ACC
    five-way tie had a common set of exactly one team (Syracuse, beaten by all five) -- a gate
    of 2 correctly skips this step and lets the NEXT step (opponents_cumulative_conf_pct)
    decide, which is what actually happened. See test_common_opponents_record_acc_gate."""
    tied_set = set(tied)
    opp_sets = [
        {r.get("opponent") for r in _conf_game_rows(ctx, team) if r.get("opponent") not in tied_set}
        for team in tied
    ]
    common = set.intersection(*opp_sets) if opp_sets else set()
    if len(common) < min_sample:
        return None
    pct: Dict[str, Optional[float]] = {}
    for team in tied:
        wins = losses = 0
        for row in _conf_game_rows(ctx, team):
            if row.get("opponent") in common:
                if row["status"] == "win":
                    wins += 1
                else:
                    losses += 1
        pct[team] = (wins / (wins + losses)) if (wins + losses) else None
    return _partition_by_value(tied, pct)


# ---------------------------------------------------------------------------
# 5. vs_placed_opponents
# ---------------------------------------------------------------------------
def _conf_win_pct(ctx: TiebreakContext, team: str) -> Optional[float]:
    rec = ctx.conf_records.get(team)
    if not rec:
        return None
    wins, losses = rec
    total = wins + losses
    return wins / total if total else None


def vs_placed_opponents(
    tied: List[str],
    ctx: TiebreakContext,
    tied_opponent_handling: str = "combine",
    exhaust_all_opponents: bool = True,
    **params,
) -> StepResult:
    """Record against the best-placed common Conference opponent, proceeding down
    ctx.frozen_order among the (excluding-tied-teams) common-opponent set -- same common-opponent
    definition as common_opponents_record, but evaluated one standings position at a time rather
    than aggregated all at once.

    THE OPPONENT SUB-RULE (there are TWO published versions, not one -- a mid-task correction
    caught a brief that had conflated them; verified independently against the source text):

    - sec.txt C.1: when the opponents at the current position are THEMSELVES tied in the
      standings, break THEIR tie by head-to-head between just the two of them; if that fails
      (they split, or never played), fall back to combining the tied teams' records against the
      whole tied-opponent group. -> tied_opponent_handling="head_to_head_then_combine".
    - big12.txt two-team step c / multi-team step c: no head-to-head sub-step at all -- straight
      to "use each team's win percentage against the collective tied teams AS A GROUP". ->
      tied_opponent_handling="combine".
    - bigten.txt B.3 specifies neither; this function defaults to "combine" for an unconfigured
      conference, since it is the reading that invents no additional step beyond what the text
      says (head_to_head_then_combine is an SEC-specific elaboration, not a safe universal
      default) -- flagged in the task report for reconciliation, not assumed silently.

    Both modes fall back to "combine" (whole tied-opponent group, no attempted resolution)
    whenever 3 or more opponents are tied at a position -- sec.txt's own text ("enter the
    two-team or three-team [opponent] procedures") would require recursively re-running the
    ENTIRE tiebreak procedure on the OPPONENTS themselves to fully resolve, which is a
    bootstrapping problem out of scope for a single primitive; documented here as a deliberate,
    reported simplification rather than a silent guess.

    "if 3 or more opponents are tied but not all are common, only the record against common
    opponents is considered" (sec.txt C.1) is automatically satisfied: only common opponents ever
    enter the position ordering below, so a same-standings-position opponent that is not itself a
    common opponent never appears in a "tied block" at all.

    "if ALL tied teams have the same record against the best-placed common opponent(s), proceed
    to the NEXT common opponent by order of finish, and continue until all common opponents are
    exhausted" (sec.txt C) is `exhaust_all_opponents=True` (default): keep advancing down
    ctx.frozen_order's common opponents until a position separates the tied teams or the list is
    exhausted (-> None). `exhaust_all_opponents=False` checks only the single best-placed
    position/block and returns immediately (separated or not) -- provided for a conference config
    that wants to stop there instead; not exercised by any of the four supplied documents, all of
    which "proceed through the standings."

    APPENDIX B NOTE: sec.txt itself says Appendix B "contains ~25 worked examples... and should
    be transcribed into the test suite" -- but the supplied sec.txt file does NOT include that
    appendix's actual text, only a reference to its existence. No fixtures could be transcribed
    from it; this is reported as a limitation, not silently skipped. Tests below instead build
    synthetic fixtures that exercise each documented branch (single common opponent, two tied
    opponents resolved by head-to-head, two tied opponents requiring combine, exhaustion through
    multiple positions).
    """
    if tied_opponent_handling not in ("head_to_head_then_combine", "combine"):
        raise ValueError(f"unknown tied_opponent_handling: {tied_opponent_handling!r}")
    direction = params.get("direction")
    if direction is not None and direction != "descending":
        # T1's loader (artifacts/tiebreaker_rules.py) allows an explicit `direction` param on
        # this step, restricted to "descending" -- this primitive only ever implements
        # best-placed-first traversal (every source document proceeds that way), so an absent or
        # "descending" value is a harmless no-op; anything else means a config asked for
        # behavior this primitive does not have.
        raise ValueError(f"vs_placed_opponents only supports direction='descending', got {direction!r}")

    tied_set = set(tied)
    opp_sets = [
        {r.get("opponent") for r in _conf_game_rows(ctx, team) if r.get("opponent") not in tied_set}
        for team in tied
    ]
    common = set.intersection(*opp_sets) if opp_sets else set()
    if not common:
        return None

    order = [t for t in ctx.frozen_order if t in common]
    order += sorted(common - set(order))  # defensive: a common opponent frozen_order omits

    idx = 0
    while idx < len(order):
        anchor = order[idx]
        anchor_pct = _conf_win_pct(ctx, anchor)
        block = [anchor]
        j = idx + 1
        while j < len(order) and anchor_pct is not None and _conf_win_pct(ctx, order[j]) == anchor_pct:
            block.append(order[j])
            j += 1

        if len(block) == 1:
            opp_group = set(block)
        elif len(block) == 2 and tied_opponent_handling == "head_to_head_then_combine":
            winner = _two_team_h2h(ctx, block[0], block[1])
            opp_group = {winner} if winner else set(block)
        else:
            opp_group = set(block)  # "combine", or 3+ under either mode

        pct: Dict[str, Optional[float]] = {}
        for team in tied:
            wins = losses = 0
            for row in _conf_game_rows(ctx, team):
                if row.get("opponent") in opp_group:
                    if row["status"] == "win":
                        wins += 1
                    else:
                        losses += 1
            pct[team] = (wins / (wins + losses)) if (wins + losses) else None

        result = _partition_by_value(tied, pct)
        if result is not None and len(result) > 1:
            return result
        if not exhaust_all_opponents:
            return None
        idx = j
    return None


# ---------------------------------------------------------------------------
# 6. opponents_cumulative_conf_pct
# ---------------------------------------------------------------------------
def opponents_cumulative_conf_pct(
    tied: List[str], ctx: TiebreakContext, ignore_opponent_count_mismatch: bool = False, **params
) -> StepResult:
    """Cumulative conference winning percentage of each tied team's OWN conference opponents:
    for each conference game a tied team played, sum that opponent's own (conf_wins, conf_losses)
    -- NOT averaged per opponent, genuinely cumulative -- then take one pct per tied team.

    THIS IS THE STEP THAT DECIDED THE REAL 2025 ACC FIVE-WAY TIE (single most important
    correctness target): Duke's eight ACC opponents went a combined 32-32 (.500, best of the
    five); Georgia Tech and Miami 28-36 (.4375); Pittsburgh and SMU 27-37 (.4219). See
    test_opponents_cumulative_conf_pct_acc_numbers.

    bigten.txt A.5(a)/B.4(a): "In the event of an unbalanced schedule (fewer than nine conference
    games played), compare on the best cumulative conference winning percentage of all conference
    opponents REGARDLESS of how many conference opponents each team played." Because this
    function is already a raw cumulative sum (never averaged or scaled by the tied team's own
    opponent COUNT), there would be nothing left for `ignore_opponent_count_mismatch=True` to
    change if the only alternative were "normalize by count" -- so it is instead interpreted as
    a GATE: when False (default), if the tied teams' own conference-games-played counts differ
    at all, return None (no opinion -- an unbalanced comparison this step's text does not
    address); when True (Big Ten's config), skip that gate and compare the raw cumulative pct
    regardless of the mismatch, exactly per A.5(a). sec.txt and big12.txt are silent on
    unbalanced schedules -- both conferences' schedules make it largely moot -- so False (the
    more conservative choice, declining to compare rather than inventing behavior neither text
    specifies) is the default, reported here rather than assumed to be Big Ten's override
    universally."""
    games_played: Dict[str, int] = {}
    pct: Dict[str, Optional[float]] = {}
    for team in tied:
        total_wins = total_losses = 0
        n_opponents = 0
        for row in _conf_game_rows(ctx, team):
            rec = ctx.conf_records.get(row.get("opponent"))
            if rec is None:
                continue
            w, l = rec
            total_wins += w
            total_losses += l
            n_opponents += 1
        games_played[team] = n_opponents
        pct[team] = (total_wins / (total_wins + total_losses)) if (total_wins + total_losses) else None

    if not ignore_opponent_count_mismatch and len(set(games_played.values())) > 1:
        return None
    return _partition_by_value(tied, pct)


# ---------------------------------------------------------------------------
# 7. capped_relative_scoring_margin
# ---------------------------------------------------------------------------
def _season_scoring_averages(ctx: TiebreakContext, team: str) -> Tuple[Optional[float], Optional[float]]:
    """(avg points scored, avg points allowed) across ALL of `team`'s played games this season
    (not conference-only -- Appendix A's own worked example describes plain "season" averages,
    with no conference restriction on the averaging input, only on which of the TIED team's own
    games the margin itself is computed over)."""
    scored = allowed = games = 0
    for row in ctx.rows:
        if row.get("season") != ctx.season or row.get("team") != team:
            continue
        if row.get("status") not in ("win", "loss"):
            continue
        ts, os_ = row.get("team_score"), row.get("opp_score")
        if ts is None or os_ is None:
            continue
        scored += ts
        allowed += os_
        games += 1
    if games == 0:
        return None, None
    return scored / games, allowed / games


def _capped_relative_margin_for_team(
    ctx: TiebreakContext, team: str, offense_cap: float = 200.0, defense_floor: float = 0.0
) -> Optional[float]:
    """Per-game capped relative scoring margin, averaged across `team`'s conference games this
    season, per sec.txt Appendix A. For each conference game: relative offense =
    (points scored / opponent's season average points allowed) * 100, capped at `offense_cap`
    (200, per Appendix A); relative defense = (points allowed / opponent's season average points
    scored) * 100, floored at `defense_floor` (0, per Appendix A); margin = offense - defense.
    Offense/defense are each rounded to 1 decimal place BEFORE subtracting, matching the
    conference's own worked computation exactly (147.6 - 116.7 = +30.9; computing on unrounded
    ratios instead gives +31.0, which the published example does not). The cap/floor are exposed
    as parameters (rather than hardcoded) only so a config can state sec.txt's own 200/0 values
    explicitly for provenance (artifacts/tiebreaker_rules.py's STEP_PARAMS allows
    `offense_cap`/`defense_floor` on this step) -- the DEFAULTS are what Appendix A actually
    specifies, and no supplied source document describes a conference using different values."""
    margins: List[float] = []
    for row in _conf_game_rows(ctx, team):
        ts, os_ = row.get("team_score"), row.get("opp_score")
        if ts is None or os_ is None:
            continue
        opp_avg_scored, opp_avg_allowed = _season_scoring_averages(ctx, row.get("opponent"))
        if not opp_avg_scored or not opp_avg_allowed:
            continue
        offense = round(min(offense_cap, (ts / opp_avg_allowed) * 100), 1)
        defense = round(max(defense_floor, (os_ / opp_avg_scored) * 100), 1)
        margins.append(offense - defense)
    if not margins:
        return None
    return sum(margins) / len(margins)


def capped_relative_scoring_margin(
    tied: List[str],
    ctx: TiebreakContext,
    offense_cap: float = 200.0,
    defense_floor: float = 0.0,
    **params,
) -> StepResult:
    """The SEC's step E (sec.txt Appendix A), averaged across each tied team's conference games.
    See _capped_relative_margin_for_team for the per-game formula, the `offense_cap`/
    `defense_floor` parameters, and the pinned worked example (Team A beats Team B 31-28; Team B
    averaged 24 scored / 21 allowed for the season -> +30.9)."""
    values = {
        team: _capped_relative_margin_for_team(ctx, team, offense_cap, defense_floor)
        for team in tied
    }
    return _partition_by_value(tied, values)


# ---------------------------------------------------------------------------
# 8. total_wins_capped
# ---------------------------------------------------------------------------
def total_wins_capped(
    tied: List[str],
    ctx: TiebreakContext,
    max_games: Optional[int] = None,
    cap_fcs_wins: bool = False,
    **params,
) -> StepResult:
    """The Big 12's step e: total wins in a 12-game season, with at most one win against an
    FCS/lower-division opponent counted annually, excluding any game exempted from the annual
    maximum-contests limit under NCAA Bylaw 17.10.5.2.1 (in practice the Hawaii exemption).

    `max_games`: if given, clamps each team's reported win count to `min(actual_wins, max_games)`
    -- a defensive bound matching the "12-game season" framing (real data should never exceed a
    team's actual game count, but this guards against a duplicate-row data issue producing an
    inflated total). None (default) applies no clamp, i.e. the plain total.

    `cap_fcs_wins`: counts at most ONE win against an FCS/lower-division opponent, per the rule
    text. The classification comes from `ctx.non_fbs_teams`, the season's `non_fbs_teams` roster
    (populated by database/get_non_fbs_teams.py, which splits on CFBD's own `classification`
    field and keeps everything that is not `fbs`).

    That roster is a POSITIVE list of non-FBS schools, which is why it is sound here even though
    a superficially similar negative test is not. database/migrations/0003_schedule_grid_view.sql
    investigated and explicitly rejected "opponent not in teams.school" as an FCS signal, because
    teams.school lists only FBS teams and the absence of a name proves nothing about why it is
    absent. Membership in non_fbs_teams is an assertion rather than an absence, so it does not
    inherit that flaw.

    When `cap_fcs_wins` is True but `ctx.non_fbs_teams` is None -- the roster was never loaded --
    this returns None (no opinion) and logs a warning, rather than either raising or silently
    reporting an uncapped total dressed up as a capped one. Returning None is the engine's
    established "this step cannot speak" signal, identical to common_opponents_record below its
    min_sample gate: the driver skips the step and the chain continues to the next one. An
    exception here would instead abort the whole standings sort, taking down an artifact publish
    over a single conference's tiebreak.

    STILL NOT CAPTURED: the rule also excludes games exempted from the annual maximum-contests
    limit under NCAA Bylaw 17.10.5.2.1 (in practice the Hawaii exemption). No data available to
    this module flags an exempt game. The effect is bounded -- it can only matter for a team that
    played a 13th regular-season game -- but it is a real, known gap, not a solved one.

    `max_games`: if given, clamps each team's reported win count to `min(actual_wins, max_games)`
    -- a defensive bound matching the "12-game season" framing (real data should never exceed a
    team's actual game count, but this guards against a duplicate-row data issue producing an
    inflated total). None (default) applies no clamp, i.e. the plain total."""
    if cap_fcs_wins and ctx.non_fbs_teams is None:
        logger.warning(
            "total_wins_capped: cap_fcs_wins=True but ctx.non_fbs_teams was not supplied for "
            "season=%s conference=%s; skipping the step rather than reporting an uncapped total "
            "as if it were capped. Populate TiebreakContext.non_fbs_teams from the "
            "`non_fbs_teams` table to enable it.",
            ctx.season, ctx.conference,
        )
        return None

    non_fbs = ctx.non_fbs_teams or frozenset()
    wins: Dict[str, Optional[float]] = {}
    for team in tied:
        won = [
            r for r in ctx.rows
            if r.get("season") == ctx.season and r.get("team") == team and r.get("status") == "win"
        ]
        if cap_fcs_wins:
            fcs_wins = sum(1 for r in won if r.get("opponent") in non_fbs)
            # Every FCS win beyond the first is struck from the total; one is kept.
            total = len(won) - max(0, fcs_wins - 1)
        else:
            total = len(won)
        wins[team] = min(total, max_games) if max_games is not None else total
    return _partition_by_value(tied, wins)


# ---------------------------------------------------------------------------
# 9. external_ranking
# ---------------------------------------------------------------------------
def external_ranking(
    tied: List[str], ctx: TiebreakContext, min_conference_games: int = 0, **params
) -> StepResult:
    """Substitutes our own model rating (ctx.team_ranks; 1 = best) wherever a conference cites a
    proprietary ranking (SportSource Team Success/Rating Score, CFP committee, SP+, SOR, KPI).
    Orders ascending (rank 1 first), unranked (None) last.

    `min_conference_games` gates the WHOLE step on the least-experienced tied team: if any tied
    team has played fewer conference games than this threshold (from ctx.conf_records), returns
    None. This gate is essential, not decorative (per task brief) -- ratings are continuous and
    essentially never tie exactly, so without a gate this step would ALWAYS resolve every group,
    pre-empting the overall-record fallback the repo owner explicitly wants available in the
    thin-information early season. Default 0 (no gating) when a config omits the threshold."""
    games_played = []
    for team in tied:
        rec = ctx.conf_records.get(team)
        games_played.append((rec[0] + rec[1]) if rec else 0)
    if games_played and min(games_played) < min_conference_games:
        return None
    ranks: Dict[str, Optional[float]] = {team: ctx.team_ranks.get(team) for team in tied}
    return _partition_by_value(tied, ranks, descending=False)


# ---------------------------------------------------------------------------
# 10. random_draw
# ---------------------------------------------------------------------------
def random_draw(tied: List[str], ctx: TiebreakContext, **params) -> StepResult:
    """Genuinely not computable -- exists so a conference's published tiebreaker chain can be
    TRANSCRIBED faithfully, including its terminal step (every source document ends in a
    commissioner's draw or coin toss), rather than silently truncated at the last computable
    step. Always returns None; the driver's own fallback chain (outside this module's scope)
    is what actually resolves a tie that reaches here."""
    return None


STEP_REGISTRY: Dict[str, Callable] = {
    "head_to_head": head_to_head,
    "sub_group_record": sub_group_record,
    "sweep_in_out": sweep_in_out,
    "common_opponents_record": common_opponents_record,
    "vs_placed_opponents": vs_placed_opponents,
    "opponents_cumulative_conf_pct": opponents_cumulative_conf_pct,
    "capped_relative_scoring_margin": capped_relative_scoring_margin,
    "total_wins_capped": total_wins_capped,
    "external_ranking": external_ranking,
    "random_draw": random_draw,
}
