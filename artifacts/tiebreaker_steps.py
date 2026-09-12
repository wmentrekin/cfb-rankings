"""
tiebreaker_steps.py -- the shared, generic
operation library for conference championship tiebreakers.

SCOPE (hard, per the T2 handoff): this module implements the ten step PRIMITIVES only. It knows
nothing about the ORDER in which a conference calls them, or what "restart at step N" means --
that sequencing is the recursive peel-off driver (T3, a separate module, separate agent, not
touched here). Different conferences call the same operations in different orders; this module
is the operations.

SOURCE OF TRUTH: the conferences' own published tiebreaker policies, supplied directly by the
repo owner. Those documents are not committed to this repository, so the decisive wording is
quoted inline -- here in the docstrings and in each configured step's `cites` field -- and every
place a transcription interpreted rather than copied is called out rather than silently
reconciled. Filenames like `sec.txt` and `acc.txt` appear throughout as labels for those supplied
documents, not as paths to files in the tree.

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
from dataclasses import dataclass, field
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
    placement_excluded_game_ids: frozenset = frozenset()
                                                   # game_ids that must not count toward STANDINGS
                                                   # PLACEMENT -- in practice the conference
                                                   # championship games. Postseason rows are
                                                   # excluded by season_type and need no entry
                                                   # here. Read by the driver's overall-record
                                                   # fallback; see tiebreaker_engine.
    divisions: Dict[str, Optional[str]] = field(default_factory=dict)
                                                   # team -> its division name, or None for a
                                                   # conference that plays none. Only the Sun Belt
                                                   # still has divisions among the ten, and only
                                                   # its chain reads this; an empty mapping means
                                                   # "no division information", which the
                                                   # division-scoped steps treat as no opinion
                                                   # rather than as "everyone shares a division".
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
def _counts_for_placement(ctx: TiebreakContext, row: Dict[str, Any]) -> bool:
    """Whether a row may contribute to STANDINGS PLACEMENT.

    Two exclusions, applied in exactly one place so every measure inherits them: postseason rows,
    and any row whose game_id the caller listed in `placement_excluded_game_ids` -- in practice
    the conference championship games, which CFBD reports as season_type 'regular' (see
    artifacts/schedule.py: "A championship game is always season_type=='regular'"), so
    season_type alone does not catch them.

    This exists because the exclusions were originally applied only in the engine's fallback and
    in overall_win_pct, leaving every row-scanning primitive counting a title game toward the
    head-to-head, common-opponent and intra-group measures that feed the standings it is supposed
    not to affect.
    """
    if row.get("season_type") == "postseason":
        return False
    return row.get("game_id") not in ctx.placement_excluded_game_ids


def _conf_game_rows(ctx: TiebreakContext, team: str) -> List[Dict[str, Any]]:
    """`team`'s own-perspective, played (win/loss), placement-eligible conference-game rows this
    season. Every helper below builds from this rather than scanning ctx.rows directly, so the
    own-perspective/played/conference-game/placement filter is applied exactly once,
    consistently."""
    return [
        r for r in ctx.rows
        if r.get("season") == ctx.season
        and r.get("team") == team
        and r.get("conference_game")
        and r.get("status") in ("win", "loss")
        and _counts_for_placement(ctx, r)
    ]


def _division_of(ctx: TiebreakContext, team: str) -> Optional[str]:
    return ctx.divisions.get(team)


def _shared_division(ctx: TiebreakContext, tied: List[str]) -> Optional[str]:
    """The division every tied team belongs to, or None if they differ or it is unknown.

    The division-scoped steps only ever run on a group inside one division -- artifacts/schedule.py
    sorts each division in its own call -- so a group spanning two divisions means the division
    map is wrong or absent, and returning None makes the step decline instead of inventing a
    comparison across divisions that no rule asks for.
    """
    divisions = {_division_of(ctx, team) for team in tied}
    if len(divisions) != 1:
        return None
    only = divisions.pop()
    return only if only is not None else None


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
def sweep_in_out(
    tied: List[str], ctx: TiebreakContext, sides: str = "both", **params
) -> StepResult:
    """For an INCOMPLETE round robin among 3+ tied teams: a team that beat EVERY OTHER TIED TEAM
    is promoted into its own top group; a team that lost to EVERY OTHER TIED TEAM is demoted into
    its own bottom group; everyone else lands in one unseparated middle group. Returns None --
    not a guess -- when the round robin IS complete (sub_group_record applies instead, per every
    source document), when fewer than 3 teams are tied, or when no team swept or was swept
    (matching sec.txt A.2.c: "if no team either beat all or lost to all -> all tied teams advance
    to the next step", i.e. this step had no opinion).

    "EVERY OTHER TIED TEAM" MEANS ALL OF THEM, NOT ALL THE ONES IT PLAYED. This is the whole
    subtlety of the step, and getting it wrong is not an academic matter -- it inverts a real
    result. The source texts are explicit: acc.txt Sec 2.b[sic].ii.1 says "The Tied Team which
    defeated EACH OF THE OTHER Tied Teams", and sec.txt A.2(a) says "One team beat ALL THE OTHER
    tied teams". A team that played one tied opponent and won has not beaten all the others; it
    has beaten one of them.

    The 2025 ACC five-way tie at 6-2 is the disproof of the looser reading. Georgia Tech played
    exactly one of the other four tied teams (Duke) and won; Duke played exactly one (Georgia
    Tech) and lost. Under "all the ones it played", Georgia Tech is promoted to the top and Duke
    is demoted to the bottom, and the step resolves the tie on the strength of a single game
    between two of five teams. Duke actually WON that tie, on the opponents'-combined-conference-
    record step four places later (its eight ACC opponents went 32-32, .500, the best of the
    five). Requiring a result against every other tied team makes this step correctly silent on
    that shape, so the chain reaches the step that really decided it.

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

    `sides` -- WHICH HALVES THE CONFERENCE ACTUALLY PUBLISHES. Across the ten supplied documents
    there are three distinct readings of this step, so it cannot be one fixed behaviour:

      "both"          the team that beat all others is promoted AND the team that lost to all
                      others is demoted. Stated explicitly by only TWO conferences: sec.txt
                      A.2(a)/(b) and acc.txt Sec 2.b[sic].ii.1.
      "promote_only"  only the promote half is published. big12.txt multi-team a.1, bigten.txt
                      B.1(a), mac.txt C.2, mountainwest.txt multi step 1 and american.txt 10.6.3
                      all describe a lone sweeper advancing and say nothing whatsoever about a
                      team that lost to everyone. Demoting one anyway would order a conference's
                      standings by a rule it never wrote down, so a total-loser stays in the
                      unseparated middle group.

    The third reading needs no value here because it needs no step: pac12.txt's multi-team step 1
    has NO sweep clause at all -- if the tied teams did not all play one another "the process
    moves to the next criterion" -- so the Pac-12 config simply omits this step rather than
    configuring it.
    """
    if sides not in ("both", "promote_only"):
        raise ValueError(
            f"sweep_in_out: unknown sides={sides!r}; expected 'both' or 'promote_only'"
        )
    if len(tied) < 3:
        return None
    if _is_complete_round_robin(ctx, tied):
        return None
    tied_set = set(tied)
    top: List[str] = []
    bottom: List[str] = []
    middle: List[str] = []
    for team in tied:
        others = tied_set - {team}
        beat, lost = set(), set()
        for row in _conf_game_rows(ctx, team):
            opp = row.get("opponent")
            if opp not in tied_set or opp == team:
                continue
            (beat if row["status"] == "win" else lost).add(opp)
        # Compared against `others`, NOT against the subset this team happened to play. Both
        # comparisons therefore require a result against EVERY other tied team; a team with an
        # unplayed tied opponent can be neither a sweeper nor a total-loser. See the docstring.
        if beat == others:
            top.append(team)
        elif lost == others and sides == "both":
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
    tied: List[str], ctx: TiebreakContext, min_sample: int = 2, scope: str = "all", **params
) -> StepResult:
    """Win percentage against opponents faced by ALL tied teams (excluding the tied teams
    themselves). Returns None if the common set is smaller than `min_sample` (default 2).

    This gate is load-bearing and empirically justified (per task brief): the real 2025 ACC
    five-way tie had a common set of exactly one team (Syracuse, beaten by all five) -- a gate
    of 2 correctly skips this step and lets the NEXT step (opponents_cumulative_conf_pct)
    decide, which is what actually happened. See test_common_opponents_record_acc_gate.

    `scope` -- nine of the ten conferences say "all common conference opponents" and take the
    default "all". The Sun Belt is the exception, because it is the only one that still plays
    divisions: sunbelt.txt two-team step 4 asks for "combined highest winning percentage against
    all COMMON NON-DIVISIONAL Conference opponents", a deliberately narrower set that exists
    because its step 2 has already compared divisional records. `scope="non_divisional"` keeps
    only common opponents OUTSIDE the tied teams' own division; `scope="divisional"` is the
    complement, provided for symmetry.

    Both scoped modes need ctx.divisions and a group that sits inside ONE division. Without
    either the step returns None rather than silently falling back to "all", which would answer a
    different question from the one the conference asked.
    """
    if scope not in ("all", "divisional", "non_divisional"):
        raise ValueError(
            f"common_opponents_record: unknown scope={scope!r}; expected 'all', 'divisional' "
            "or 'non_divisional'"
        )
    tied_set = set(tied)
    opp_sets = [
        {r.get("opponent") for r in _conf_game_rows(ctx, team) if r.get("opponent") not in tied_set}
        for team in tied
    ]
    common = set.intersection(*opp_sets) if opp_sets else set()

    if scope != "all":
        division = _shared_division(ctx, tied)
        if division is None:
            logger.warning(
                "common_opponents_record: scope=%r needs a division for every tied team, but "
                "season=%s conference=%s group=%s does not share one; skipping the step.",
                scope, ctx.season, ctx.conference, tied,
            )
            return None
        if scope == "divisional":
            common = {o for o in common if _division_of(ctx, o) == division}
        else:
            common = {o for o in common if _division_of(ctx, o) != division}

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
    advance_on_unequal_games: bool = False,
    standings_scope: str = "conference",
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

    `advance_on_unequal_games` -- CUSA-specific, from cusa.txt section D: "If the tied teams have
    the same record OR PLAYED AN UNEQUAL NUMBER OF GAMES against the teams within the tied group,
    immediately advance to the team(s) with the next highest conference winning percentage." The
    "same record" half is what every conference does and is already the default behaviour; the
    unequal-games half is stated by CUSA alone. With it enabled, a position where the tied teams
    played a DIFFERENT NUMBER of games against that position's opponent(s) is skipped outright
    rather than compared -- a 1-0 record and a 2-1 record against the same block are not treated
    as comparable percentages. Default False, so no other conference's chain changes.

    `standings_scope` -- which standings the traversal walks. Nine conferences walk the conference
    standings and take the default "conference". The Sun Belt walks its DIVISIONAL standings:
    sunbelt.txt two-team step 3 is "each team's winning percentage vs the team occupying the next
    highest position in the final DIVISIONAL standings". `standings_scope="divisional"` filters
    ctx.frozen_order to the tied teams' own division before traversing it, and returns None when
    they do not share one.

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
    if standings_scope not in ("conference", "divisional"):
        raise ValueError(
            f"vs_placed_opponents: unknown standings_scope={standings_scope!r}; expected "
            "'conference' or 'divisional'"
        )
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

    if standings_scope == "divisional":
        division = _shared_division(ctx, tied)
        if division is None:
            logger.warning(
                "vs_placed_opponents: standings_scope='divisional' needs a division for every "
                "tied team, but season=%s conference=%s group=%s does not share one; skipping.",
                ctx.season, ctx.conference, tied,
            )
            return None
        common = {o for o in common if _division_of(ctx, o) == division}
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
        played_counts: Dict[str, int] = {}
        for team in tied:
            wins = losses = 0
            for row in _conf_game_rows(ctx, team):
                if row.get("opponent") in opp_group:
                    if row["status"] == "win":
                        wins += 1
                    else:
                        losses += 1
            played_counts[team] = wins + losses
            pct[team] = (wins / (wins + losses)) if (wins + losses) else None

        if advance_on_unequal_games and len(set(played_counts.values())) > 1:
            # CUSA section D: unequal games against this position means the comparison is not
            # made at all, rather than made on percentages drawn from different sample sizes.
            if not exhaust_all_opponents:
                return None
            idx = j
            continue

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
    """(avg points scored, avg points allowed) across `team`'s played games this season -- not
    conference-only, because Appendix A's own worked example describes plain "season" averages,
    with no conference restriction on the averaging input, only on which of the TIED team's own
    games the margin itself is computed over.

    Placement exclusions still apply: a bowl or the conference title game must not move the
    averages that decide a standings position, for the same reason it must not move any other
    measure here."""
    scored = allowed = games = 0
    for row in ctx.rows:
        if row.get("season") != ctx.season or row.get("team") != team:
            continue
        if row.get("status") not in ("win", "loss"):
            continue
        if not _counts_for_placement(ctx, row):
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
            if r.get("season") == ctx.season and r.get("team") == team
            and r.get("status") == "win"
            # Same placement exclusions as overall_win_pct and the engine's fallback. Without
            # them this step counts bowl wins and the conference title game toward a standings
            # measure, which it is the one place in the chain that must not do.
            and _counts_for_placement(ctx, r)
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


# ---------------------------------------------------------------------------
# 11. divisional_record
# ---------------------------------------------------------------------------
def divisional_record(tied: List[str], ctx: TiebreakContext, **params) -> StepResult:
    """Conference win percentage counting ONLY games against teams in the same division.

    sunbelt.txt two-team step 2 and multi step 2: "the team with the highest overall DIVISIONAL
    winning percentage shall be the division champion". The Sun Belt is the only one of the ten
    conferences that still plays divisions, so this is the only chain that uses it.

    IT IS NOT THE PRIMARY KEY, AND THAT IS THE WHOLE POINT. The same document defines a division
    champion as "the team with the highest winning percentage in ALL CONFERENCE GAMES, BOTH
    DIVISIONAL AND NON-DIVISIONAL" -- which is the conference record this project already sorts
    by. So the divisional-only record is a genuinely separate, narrower measure used to break a
    tie in the broader one, not a restatement of it.

    Returns None when the tied teams do not share one division or the division map is absent,
    for the same reason the scoped modes of common_opponents_record do: a cross-division
    comparison answers a question no rule asked.
    """
    division = _shared_division(ctx, tied)
    if division is None:
        logger.warning(
            "divisional_record: needs a division for every tied team, but season=%s "
            "conference=%s group=%s does not share one; skipping the step.",
            ctx.season, ctx.conference, tied,
        )
        return None

    pct: Dict[str, Optional[float]] = {}
    for team in tied:
        wins = losses = 0
        for row in _conf_game_rows(ctx, team):
            if _division_of(ctx, row.get("opponent")) != division:
                continue
            if row["status"] == "win":
                wins += 1
            else:
                losses += 1
        pct[team] = (wins / (wins + losses)) if (wins + losses) else None
    return _partition_by_value(tied, pct)


# ---------------------------------------------------------------------------
# 12. overall_win_pct
# ---------------------------------------------------------------------------
def overall_win_pct(
    tied: List[str],
    ctx: TiebreakContext,
    fcs_win_cap: Optional[int] = None,
    fbs_only: bool = False,
    **params,
) -> StepResult:
    """Overall winning percentage, conference and non-conference, in three published variants.

    Three conferences reach for "overall record" late in their chains and each adjusts it
    differently, which is why this is one primitive with parameters rather than three:

      american.txt 10.5.9 / 10.6.10   plain overall percentage, "excluding exempt games"
      mountainwest.txt two-team 3     overall percentage, "a maximum of ONE win against a team
                                      from the NCAA Football Championship Subdivision shall be
                                      included"                       -> fcs_win_cap=1
      sunbelt.txt step 9              overall percentage "against FBS teams"  -> fbs_only=True

    Note this is a PERCENTAGE, and distinct from `total_wins_capped`, which is the Big 12's
    12-game win COUNT with its own FCS cap. Four conferences, four different adjustments to the
    same underlying idea.

    STANDINGS-PLACEMENT EXCLUSIONS APPLY. Postseason rows and
    ctx.placement_excluded_game_ids (the conference championship games) are left out, matching
    artifacts/schedule.py::_placement_pct and the engine's own fallback. None of the three source
    documents says so explicitly -- they predate or ignore the question -- but the repo owner's
    rule that bowl and playoff results must not affect standings placement applies to every
    measure that places a team, not only to the primary key.

    "EXCLUDING EXEMPT GAMES" IS NOT IMPLEMENTED and is not pretended to be. american.txt 10.5.9
    and big12.txt step e both defer to NCAA Bylaw 17.10.5.2.1 (in practice the Hawaii
    exemption); no data available to this module flags an exempt game. The effect is bounded to
    teams that played a thirteenth regular-season game, and the American's config records the gap
    in that step's `cites` rather than leaving it to be rediscovered.

    Both `fcs_win_cap` and `fbs_only` need ctx.non_fbs_teams. When it was not supplied (None, as
    opposed to an empty set) the step returns None and warns, rather than reporting an unadjusted
    percentage as if it had been adjusted -- the same contract as total_wins_capped.

    WHAT THE ROSTER COVERS, PRECISELY. ctx.non_fbs_teams comes from the `non_fbs_teams` table,
    which database/get_non_fbs_teams.py fills from CFBD's plain /teams endpoint keeping every
    classification EXCEPT fbs -- so fcs, ii, ii/iii and iii are all in it. What is NOT in it is a
    school CFBD does not return for the season at all, an NAIA opponent being the realistic case.
    Such a team is absent from the roster and therefore counts as FBS here: its win would survive
    `fbs_only` and would not be capped by `fcs_win_cap`. There is no data available to this module
    that would distinguish it, so the gap is recorded rather than papered over. It is narrow --
    FBS teams schedule outside Division I rarely, and CFBD usually lists those opponents anyway.
    """
    if (fcs_win_cap is not None or fbs_only) and ctx.non_fbs_teams is None:
        logger.warning(
            "overall_win_pct: fcs_win_cap/fbs_only requested but ctx.non_fbs_teams was not "
            "supplied for season=%s conference=%s; skipping the step rather than reporting an "
            "unadjusted percentage as an adjusted one.",
            ctx.season, ctx.conference,
        )
        return None

    non_fbs = ctx.non_fbs_teams or frozenset()
    pct: Dict[str, Optional[float]] = {}
    for team in tied:
        wins: List[Dict[str, Any]] = []
        losses = 0
        for row in ctx.rows:
            if row.get("season") != ctx.season or row.get("team") != team:
                continue
            if row.get("season_type") == "postseason":
                continue
            if row.get("game_id") in ctx.placement_excluded_game_ids:
                continue
            status = row.get("status")
            if status not in ("win", "loss"):
                continue
            opponent_is_non_fbs = row.get("opponent") in non_fbs
            if fbs_only and opponent_is_non_fbs:
                # Sun Belt step 9: the game is not counted at all, win or loss.
                continue
            if status == "win":
                wins.append(row)
            else:
                losses += 1

        win_count = len(wins)
        if fcs_win_cap is not None:
            # Mountain West: wins over non-FBS opponents count at most `fcs_win_cap` times. The
            # LOSSES are untouched -- the rule caps what a team can bank from playing down, and
            # says nothing about forgiving a loss to an FCS team.
            fcs_wins = sum(1 for r in wins if r.get("opponent") in non_fbs)
            win_count -= max(0, fcs_wins - fcs_win_cap)

        played = win_count + losses
        pct[team] = (win_count / played) if played else None
    return _partition_by_value(tied, pct)


# ---------------------------------------------------------------------------
# 13. conditional_external_ranking
# ---------------------------------------------------------------------------
def _final_conference_week(ctx: TiebreakContext) -> Optional[int]:
    """THIS conference's last PLAYED regular-season conference week, or None if it has none.

    Derived from the rows rather than from a calendar, because the week number is not constant:
    get_cfb_week()'s anchor moves each year, so the final conference weekend is a different week
    number in different seasons (artifacts/schedule.py documents the same thing for the
    championship-week label).

    THREE FILTERS, EACH LOAD-BEARING, and the first two were missing until an independent review
    caught it. `ctx.rows` in production is the WHOLE LEAGUE's schedule grid for the season,
    including unplayed rows:

      - `team in ctx.conf_records` scopes to this conference. Without it, `max` returns the last
        week any FBS conference is scheduled to play, which is routinely later than this
        conference's own finale.
      - `status in ("win", "loss")` excludes fixtures that have not happened. Without it, a week
        that no game has yet been played in still counts as "the final weekend".
      - `_counts_for_placement` drops postseason rows and the conference championship games,
        which CFBD reports as season_type 'regular'.

    Getting any of these wrong does not fail loudly. It moves the final week to one in which the
    tied teams have no game, so every team reads as idle, and `conditional_external_ranking`
    degrades silently into a plain rating comparison -- which never ties, and therefore pre-empts
    every remaining step in three conferences' chains.
    """
    members = set(ctx.conf_records)
    weeks = [
        r.get("week") for r in ctx.rows
        if r.get("season") == ctx.season
        and r.get("team") in members
        and r.get("conference_game")
        and r.get("status") in ("win", "loss")
        and _counts_for_placement(ctx, r)
        and r.get("week") is not None
    ]
    return max(weeks) if weeks else None


def _final_week_outcome(ctx: TiebreakContext, team: str, final_week: int) -> Optional[str]:
    """'win', 'loss', or None if the team had no conference game that week (a bye).

    The bye case is why `condition` below has two values rather than one: a team on a bye
    literally "does not lose" in the final weekend but does not "win" either, and the source
    documents split on exactly that wording.
    """
    for row in ctx.rows:
        if (
            row.get("season") == ctx.season
            and row.get("team") == team
            and row.get("week") == final_week
            and row.get("conference_game")
            and row.get("season_type") != "postseason"
            and row.get("status") in ("win", "loss")
        ):
            return row["status"]
    return None


def conditional_external_ranking(
    tied: List[str],
    ctx: TiebreakContext,
    min_conference_games: int = 0,
    ranked_cutoff: int = 25,
    condition: str = "does_not_lose",
    **params,
) -> StepResult:
    """An outside ranking CONDITIONED ON THE FINAL WEEKEND'S RESULT -- the step that dominates the
    Mountain West, Sun Belt and American procedures.

    All three enumerate the same operation case by case rather than stating it once
    (mountainwest.txt two-team 2 and multi 2; sunbelt.txt steps 5-8; american.txt 10.5.3-10.5.7
    and 10.6.4-10.6.8 -- five consecutive clauses there before any record-based measure appears).
    Collapsed, the published logic is:

      - a tied team that was RANKED going into the final weekend, and then won it (or merely did
        not lose it -- the documents differ, see `condition`), is selected;
      - if no ranked tied team survives the final weekend, the comparison reverts to a COMPOSITE
        AVERAGE of computer rankings over all the tied teams.

    Read as an ordering rather than a selection (see tiebreaker_engine's reframing), that is a
    two-tier partition: the surviving ranked teams first, ordered among themselves, then everyone
    else. When the first tier is empty, or contains every tied team, this step is exactly
    `external_ranking` -- so it can only change an outcome when survival splits the group, which
    is the case its tests pin.

    WHAT THE SUBSTITUTION COSTS, STATED PLAINLY. Per K6 this project's own rating stands in for
    every outside service, and here that erases a distinction the documents rely on: the CFP poll
    ranks 25 teams, so "was ranked" is a real filter, while our rating ranks everyone, so it is
    none. Left alone, every tied team would count as ranked and the cascade would collapse into
    plain `external_ranking`, losing the final-weekend condition entirely.

    `ranked_cutoff` (default 25) is the deliberate proxy: a team counts as "ranked going into the
    final weekend" if our rank is within the cutoff, mirroring the poll's size. It is a proxy and
    not the thing itself -- our top 25 is not the committee's -- and it is the single largest
    interpretive liberty in any of the ten transcriptions.

    The second substitution is invisible rather than lossy: the documents fall back from the CFP
    poll to a composite of named computer rankings (Anderson & Hester, Massey, Colley, Wolfe for
    the Sun Belt and Mountain West; Connolly SP+, SportSource TR116 SOR, ESPN SOR, KPI for the
    American), and both the poll and the composite become the same rating here, so the
    distinction between "use the ranking" and "use the composite" has no effect.

    ALSO NOT MODELLED: the documents specify the poll AS OF A DATE before the final weekend
    (mountainwest.txt names November 21). ctx.team_ranks is a single current snapshot with no
    as-of dimension, so the rank used is the one we hold now, not the one we held then. The
    repository does store weekly ratings, so this is a plumbing gap rather than an impossible one.

    `condition` -- the wording genuinely differs and a bye is the case that separates them:
      "does_not_lose"  american.txt 10.5.3/10.5.5/10.6.4 say "doesn't lose". A team idle in the
                       final weekend satisfies this, as does a tie.
      "wins"           mountainwest.txt 2(a)/(b) and sunbelt.txt 5-8 say "wins". An idle team
                       does not qualify.

    `min_conference_games` gates the whole step exactly as `external_ranking` does, so it cannot
    fire in the thin-information early season.
    """
    if condition not in ("does_not_lose", "wins"):
        raise ValueError(
            f"conditional_external_ranking: unknown condition={condition!r}; expected "
            "'does_not_lose' or 'wins'"
        )

    games_played = []
    for team in tied:
        rec = ctx.conf_records.get(team)
        games_played.append((rec[0] + rec[1]) if rec else 0)
    if games_played and min(games_played) < min_conference_games:
        return None

    final_week = _final_conference_week(ctx)
    if final_week is None:
        # No conference games at all this season: the condition is unanswerable, so this step has
        # no opinion rather than degrading silently into a plain rating comparison.
        return None

    survivors: List[str] = []
    for team in tied:
        rank = ctx.team_ranks.get(team)
        if rank is None or rank > ranked_cutoff:
            continue
        outcome = _final_week_outcome(ctx, team, final_week)
        if condition == "wins":
            qualifies = outcome == "win"
        else:
            qualifies = outcome != "loss"        # a win or a bye; only a loss disqualifies
        if qualifies:
            survivors.append(team)

    def _rank_key(team: str) -> float:
        rank = ctx.team_ranks.get(team)
        return float(rank) if rank is not None else float("inf")

    if not survivors or len(survivors) == len(tied):
        # Nothing to split on: fall through to the composite over the whole group, which under
        # the K6 substitution is the same rating. Identical to external_ranking's behaviour.
        ranks: Dict[str, Optional[float]] = {t: ctx.team_ranks.get(t) for t in tied}
        return _partition_by_value(tied, ranks, descending=False)

    rest = [t for t in tied if t not in survivors]
    # Both halves go through _partition_by_value rather than being exploded into singletons, so
    # teams this step cannot tell apart stay in one group and the chain continues on them. An
    # earlier version returned one singleton per team, which separated UNRANKED teams from each
    # other in whatever order they arrived in -- a non-deterministic ordering presented as a
    # decision, and one that also recorded them as resolved and so skipped every later step.
    survivor_ranks: Dict[str, Optional[float]] = {t: ctx.team_ranks.get(t) for t in survivors}
    rest_ranks: Dict[str, Optional[float]] = {t: ctx.team_ranks.get(t) for t in rest}
    return (
        (_partition_by_value(survivors, survivor_ranks, descending=False) or [list(survivors)])
        + (_partition_by_value(rest, rest_ranks, descending=False) or [list(rest)])
    )


def satisfies_when(when: Optional[str], tied: List[str], ctx: TiebreakContext) -> bool:
    """Whether a config step's `when` predicate holds for this group, so the driver can gate a
    step without itself knowing the row shape.

    Lives here rather than in the driver because the only thing either predicate asks about is
    whether the tied teams all played each other, which is a question about game rows -- the
    subject of this module. The vocabulary is validated at config load
    (tiebreaker_rules.WHEN_PREDICATES); an unrecognised value reaching here is a programming
    error, so it is refused loudly rather than silently treated as "always".

    None means the step is unconditional."""
    if when is None:
        return True
    if when == "round_robin_among_tied":
        return _is_complete_round_robin(ctx, tied)
    if when == "not_round_robin_among_tied":
        return not _is_complete_round_robin(ctx, tied)
    raise ValueError(
        f"satisfies_when: unrecognised predicate {when!r}. Valid values are "
        "'round_robin_among_tied', 'not_round_robin_among_tied', or None."
    )


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
    "conditional_external_ranking": conditional_external_ranking,
    "overall_win_pct": overall_win_pct,
    "divisional_record": divisional_record,
}
