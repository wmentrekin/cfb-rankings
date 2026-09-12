"""
schedule_standings.py -- pure-function "brains" module for the Season Grid
feature's computed columns (conference-championship and bowl-eligibility
status), plus the record/conf_record tallies both depend on.

SCOPE / BOUNDARY (hard):
  - This module does NO DB or HTTP I/O of its own. Every public function
    here takes `schedule_grid` rows (already fetched by the caller -- T4b)
    as plain dicts, plus a `season`, and returns plain Python data
    structures (dicts of dicts). Anything this module needs that is NOT in
    schedule_grid is INJECTED as an argument by the caller rather than
    queried here -- notably team `division` (a `teams` column with no
    schedule_grid counterpart), which artifacts/schedule.py passes down. The
    boundary is about I/O, not about arity. It knows nothing about JSON shape, logos,
    R2, conference display names/sort order, or postseason column
    bucketing -- all of that is T4b's job.
  - This module MUST NEVER be imported by, or share a code path with,
    model/ or the rankings_games view. It is a display-only computation
    over schedule_grid (the season-grid feature's own silver-layer view,
    see database/migrations/0003_schedule_grid_view.sql), entirely
    separate from the rating model's data path.

INPUT ROW SHAPE: each row is a dict (or dict-like mapping) with at least
these schedule_grid columns populated: season (int), team (str),
conference (str | None), status (str: 'win'|'loss'|'upcoming'|... --
this module only ever inspects 'win'/'loss'/'upcoming'), conference_game
(bool). Extra columns (game_id, week, opponent, etc.) are ignored here --
T4b uses those for its own JSON shaping.
"""

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional

# Reuse the same logger name main.py configures via utils.setup_logging (see
# artifacts/rankings.py for the identical convention), so warnings surface
# through the pipeline's existing handlers when run via main.py, and still
# fall back to Python's default stderr handler when this module is used
# standalone (e.g. from a unit test).
logger = logging.getLogger("cfb_lp")


# ---------------------------------------------------------------------------
# QUALIFYING_CHAMPIONSHIP_CONFERENCES
#
# Conferences confirmed (via live web research done during T4a, September
# 2026) to decide their football championship game strictly by "no
# divisions, top 2 teams by conference win-loss record play each other" --
# the exact shape the ELIMINATED/CLINCHED algorithm below models. A
# conference NOT in this dict gets NO possible/eliminated/clinched entry
# for any of its teams -- a deliberate blank fallback (see
# compute_conference_championship_status), never a guessed status.
#
# FLAG (per plan.yaml key_decisions and the T4a handoff): conference
# membership AND format change most years with realignment. This list must
# be RE-VERIFIED EACH OFFSEASON, not treated as permanent. Do not silently
# carry it forward season over season without re-checking.
#
# Per-conference basis (all checked live, not from training-data memory --
# search queries run during T4a, results below; see the T4a handoff report
# for the actual source links):
#   - SEC: divisions eliminated 2024; #1 vs #2 in conference standings.
#   - Big Ten: divisions eliminated 2024; top 2 by conference record.
#   - Big 12: divisions eliminated 2011ish, still top-2/no-divisions in
#     2025 (16 members after realignment).
#   - ACC: divisions eliminated 2023; top 2 by conference record.
#   - American Athletic (AAC): divisions eliminated ~2020 (after UConn's
#     departure); top 2 teams meet.
#   - Conference USA: single-division format; top 2 finishers meet.
#   - Mountain West: top 2 by conference record; a 2025 4-way tie at the
#     top was broken by a composite of external metrics (SP+/SOR/KPI/
#     SportSource), NOT head-to-head-based reseeding -- that tiebreak
#     detail doesn't affect this module (W_T is still each team's plain
#     conference win count), it's just evidence the "top 2 by record"
#     description is accurate.
#   - Mid-American (MAC): divisions eliminated after the 2023 season; the
#     MAC's own materials describe the criterion as "best conference
#     WINNING PERCENTAGE," not raw win total. CAVEAT, flagged rather than
#     silently ignored: this module's W_T/B_T/L_T counting is win-TOTAL
#     based per the handoff's exact algorithm spec. Win total and win
#     percentage rank identically whenever every team in the conference
#     has played the same number of conference games (the normal case),
#     and diverge only if conference game counts differ across members
#     mid-season (e.g. a newly-joined member on an odd schedule) -- a real
#     but narrow edge case. Included here because the "top 2, no
#     divisions" shape still matches; re-verify this caveat doesn't matter
#     for the specific season being computed if conference game counts are
#     known to be uneven.
#
# Explicitly EXCLUDED, and why (checked, not assumed):
#   - Sun Belt: STILL DIVISIONAL as of the 2025 season (East division
#     champion vs. West division champion) -- confirmed via the actual
#     2025 Sun Belt Championship Game participants (James Madison, East
#     champion, vs. Troy, West champion). Does not match the top-2/no-
#     divisions shape this algorithm assumes. Still excluded from THIS
#     dict for that reason; it is handled by its own format dict,
#     DIVISIONAL_CHAMPIONSHIP_CONFERENCES below.
#   - Pac-12 (2026): 8 members, 7-game full round robin with no divisions.
#     Top 2 by conference record play in championship game on Dec 4, 2026.
#     Verified via web-search snippets quoting pac-12.com and Wikipedia,
#     NOT from direct read of primary source (this environment's egress
#     policy blocked those domains). Re-verify the format for future seasons.
#   - FBS Independents: not a conference -- no championship game exists to
#     model. Also excluded structurally: compute_team_records() always
#     sets conf_wins/conf_losses to None for Independents, and this
#     module only considers teams with a non-None conf_wins.
#
# Membership-count qualification (>= 4 members) is NOT hardcoded here as a
# static number -- it's checked dynamically at runtime in
# compute_conference_championship_status() against however many teams
# actually appear for that conference in the season's input rows. This is
# deliberately realignment-robust: a conference losing/gaining members
# season to season doesn't require an update to a hardcoded count, only
# the format confirmation above needs annual re-verification.
# ---------------------------------------------------------------------------
QUALIFYING_CHAMPIONSHIP_CONFERENCES: Dict[str, str] = {
    "SEC": "top-2, no divisions since 2024",
    "Big Ten": "top-2, no divisions since 2024",
    "Big 12": "top-2, no divisions",
    "ACC": "top-2, no divisions since 2023",
    "American Athletic": "top-2, no divisions since ~2020",
    "Conference USA": "top-2, single division",
    "Mountain West": "top-2 by conference record",
    "Mid-American": "top-2 by conference win pct (see MAC caveat above)",
    "Pac-12": "top-2, no divisions (7-game full round robin as of 2026)",
}

# ---------------------------------------------------------------------------
# DIVISIONAL_CHAMPIONSHIP_CONFERENCES
#
# Conferences that still use DIVISIONS, and whose championship game is
# "each division's champion vs. the other division's champion" -- a
# different shape from the flat "top 2 of one pool" that
# QUALIFYING_CHAMPIONSHIP_CONFERENCES above describes. The two dicts are
# disjoint by construction: a conference has one format or the other, never
# both.
#
# A conference listed here still produces NO status at all unless the caller
# injects a division map covering it (compute_conference_championship_status's
# `divisions` argument -- division lives in the `teams` table, which this
# module deliberately cannot read; see the SCOPE / BOUNDARY note at the top)
# AND the data matches the modeled shape: exactly two divisions, every member
# placed in one of them, each division meeting min_members. Any other shape
# falls back to blank for the WHOLE conference -- never a guessed race.
#
# FLAG, same standing warning as the dict above: this must be RE-VERIFIED
# EACH OFFSEASON. Divisions are precisely the format detail realignment keeps
# changing -- the SEC, Big Ten, ACC and MAC all dropped theirs within the
# last few seasons, leaving the Sun Belt as the only FBS holdout in 2026. Do
# not silently carry this forward season over season without re-checking.
#
# Sun Belt basis, and the honest limits of that verification:
#   - Format: East and West, 7 teams each in 2026; the title game is the East
#     champion vs. the West champion.
#   - THE LOAD-BEARING DETAIL: a Sun Belt division champion is decided by
#     winning percentage across ALL CONFERENCE GAMES -- divisional AND
#     non-divisional alike -- NOT by a division-only record. Divisional
#     record enters only as a tiebreaker. The intuitive assumption (rank each
#     division by its members' records against each other) is the WRONG one
#     and is the single most likely way to produce a false label here. See
#     the pool-construction comment inside
#     compute_conference_championship_status, and
#     tests/test_division_championship_status.py's
#     test_division_race_ranks_by_overall_conference_record, which exists to
#     fail loudly if anyone later "fixes" this to division-only.
#   - Published tiebreaker order: head-to-head between the tied teams; then
#     highest divisional winning percentage; for 3+ team ties, win% among the
#     tied teams, then record vs. the next-highest-placed team in the
#     division cascading down, then combined win% vs. common non-divisional
#     conference opponents. This module models NONE of it -- it only asserts
#     clinched/eliminated when the outcome holds no matter how ties break.
#   - VERIFICATION CAVEAT, stated plainly rather than overclaimed: the above
#     comes from web-search snippets quoting sunbeltsports.org, NOT from a
#     direct read of the primary source -- this environment's egress policy
#     blocked those domains. The tiebreaker page also carries a 2018 slug
#     that may have been superseded by later realignment. Treat this as
#     corroborated secondhand, not confirmed at the source.
# ---------------------------------------------------------------------------
DIVISIONAL_CHAMPIONSHIP_CONFERENCES: Dict[str, str] = {
    "Sun Belt": (
        "East/West division champions meet; a division champion is the best record "
        "across ALL conference games, not division-only (2026)"
    ),
}

MIN_QUALIFYING_MEMBERS = 4

# A team's `conference` value that means "not actually in a conference" --
# matches the raw value confirmed live against schedule_grid/teams (see
# artifacts/rankings.py's CONFERENCE_DISPLAY_NAMES, which maps this same
# raw string to a display name).
INDEPENDENT_CONFERENCE_VALUE = "FBS Independents"


def compute_team_records(rows: Iterable[Dict[str, Any]], season: int) -> Dict[str, Dict[str, Any]]:
    """
    Tally each team's overall record, conference record, and remaining
    conference-game count from schedule_grid rows, for one season.

    Args:
        rows: schedule_grid rows as dicts (see module docstring for the
              columns this function reads). Rows whose `season` doesn't
              match the `season` argument are ignored defensively (this
              function does its own season filtering rather than trusting
              the caller pre-filtered).
        season: the season to compute records for.

    Returns:
        Dict keyed by team name:
            {
              "conference": str | None,        # the team's own conference this
                                                 # season (the mode of its rows'
                                                 # `conference` values; None only
                                                 # if the team has no rows with a
                                                 # non-null conference at all)
              "wins": int, "losses": int,       # overall record: all rows with
                                                 # status in ('win', 'loss'),
                                                 # regardless of conference_game
              "conf_wins": int | None,
              "conf_losses": int | None,        # conference record: restricted to
                                                 # conference_game=true rows. None
                                                 # (never "0-0") when the team's
                                                 # conference is Independents or
                                                 # unknown -- there is no conference
                                                 # to tally against.
              "conf_games_remaining": int,      # count of conference_game=true,
                                                 # status='upcoming' rows -- this is
                                                 # R_T for the championship algorithm
                                                 # below (0, and unused, for
                                                 # Independents).
            }
    """
    by_team_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("season") != season:
            continue
        team = row.get("team")
        if not team:
            continue
        by_team_rows[team].append(row)

    records: Dict[str, Dict[str, Any]] = {}
    for team, team_rows in by_team_rows.items():
        conference_counts = Counter(r["conference"] for r in team_rows if r.get("conference"))
        conference = conference_counts.most_common(1)[0][0] if conference_counts else None

        wins = sum(1 for r in team_rows if r.get("status") == "win")
        losses = sum(1 for r in team_rows if r.get("status") == "loss")

        conf_rows = [r for r in team_rows if r.get("conference_game")]
        is_independent = conference is None or conference == INDEPENDENT_CONFERENCE_VALUE
        if is_independent:
            conf_wins: Optional[int] = None
            conf_losses: Optional[int] = None
        else:
            conf_wins = sum(1 for r in conf_rows if r.get("status") == "win")
            conf_losses = sum(1 for r in conf_rows if r.get("status") == "loss")

        conf_games_remaining = sum(1 for r in conf_rows if r.get("status") == "upcoming")

        records[team] = {
            "conference": conference,
            "wins": wins,
            "losses": losses,
            "conf_wins": conf_wins,
            "conf_losses": conf_losses,
            "conf_games_remaining": conf_games_remaining,
        }
    return records


def compute_conference_championship_status(
    records: Dict[str, Dict[str, Any]],
    qualifying_conferences: Optional[Dict[str, str]] = None,
    min_members: int = MIN_QUALIFYING_MEMBERS,
    divisions: Optional[Dict[str, Optional[str]]] = None,
    divisional_conferences: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute conference-championship status (possible/eliminated/clinched)
    per team, for qualifying conferences only. Implements the exact
    conservative-by-construction algorithm described below -- this is NOT a
    simulation/brute-force search, and deliberately does not attempt
    joint-feasibility reasoning across teams' remaining games against each
    other. Both tests are O(N log N) per pool (a sort).

    POOLS. A "pool" is one set of teams competing for ONE championship-game
    slot, plus how many of them reach it (top_n):
      - a conference in `qualifying_conferences` (flat, no divisions) is a
        single pool of all its teams, top_n=2;
      - a conference in `divisional_conferences` is one pool PER DIVISION,
        top_n=1 each, because only the division champion advances.
    Everything below is written against pools, so both formats share one code
    path -- and top_n=2 reproduces the pre-division arithmetic exactly.

    For each team T in a pool:
        W_T = T's current banked conference wins (conf_wins)
        R_T = T's remaining scheduled conference games (conf_games_remaining)
        B_T (best case)  = W_T + R_T   (T wins out)
        L_T (worst case) = W_T         (T loses out)

    ELIMINATED: T is eliminated if B_T is less than the top_n'th-highest W_i
    among every OTHER team i in the pool -- i.e. at least top_n other teams
    have ALREADY banked more wins than T could possibly ever reach. This only
    ever relies on already-secured facts (other teams' current win counts),
    never on projecting how their remaining games (including games between
    two OTHER teams) turn out, so it sidesteps the classic joint-infeasibility
    problem entirely. It may leave a team "possible" longer than a fully
    rigorous analysis would -- that's expected and safe. It must NEVER assert
    "eliminated" incorrectly.

    CLINCHED: T has clinched if the count of other teams i where
    B_i >= L_T (each team i's own best case checked independently against
    T's worst case -- no joint-feasibility assumption) is <= top_n - 1. At
    most top_n - 1 other teams could possibly finish at or above T's
    guaranteed floor, so T is guaranteed a top-top_n finish. May under-report
    an early clinch; must never over-report one. Note that for a division
    pool (top_n=1) this means "nobody else in the division can even reach my
    banked win total", so T finishes strictly first and no tiebreaker can
    take the division away from it.

    WIN TOTALS vs. WIN PERCENTAGE: as with the MAC caveat on
    QUALIFYING_CHAMPIONSHIP_CONFERENCES, both tests count WINS, and the Sun
    Belt's rule is stated as winning PERCENTAGE. The two rank identically
    whenever every team in the pool ends the season having played the same
    number of conference games -- true of the Sun Belt's 8-game conference
    slate, verified for 2026 against schedule_grid (all 14 members have
    exactly 8 conference games scheduled). If a season's data ever has
    uneven conference game counts within a pool, re-check this before
    trusting a clinch/elimination.

    The standings-sort's own head-to-head tiebreak (T4b's job, purely
    cosmetic ordering of "possible" teams within a conference) is NOT used
    here at all -- clinch/eliminate are computed independently per team via
    the counting rules above; this function never tries to resolve 3+-way
    ties, and never models any published tiebreaker.

    CONDITIONAL_OPPONENT: the still-"possible" teams of a pool get
    conditional_opponent set to the name of the team that has already
    clinched the OTHER title-game slot, when exactly one team has:
      - flat conference: the other slot comes from the SAME pool, so this is
        "once exactly one team in the conference clinches, every other
        still-possible team in that conference names it" (unchanged);
      - divisional conference: the other slot is the OTHER DIVISION's
        champion, so an East team names the clinched WEST team -- never
        anyone from its own division, which is the very slot it is competing
        for. (With top_n=1, a division that has a clinched team has no
        "possible" teams left in it at all: every other member is
        necessarily eliminated.)
    A clinched or eliminated team's own entry has conditional_opponent=None
    (T4b maps a clinched team's own opponent field to null). If zero teams
    have clinched the other slot, or -- a rare edge case -- more than one
    has, conditional_opponent is None for every "possible" team in the pool:
    naming just one of two would be arbitrary, so it is left unset rather
    than guessed.

    Args:
        records: output of compute_team_records().
        qualifying_conferences: mapping of conference name -> format note for
            FLAT (no-division, top-2) conferences; defaults to the
            module-level QUALIFYING_CHAMPIONSHIP_CONFERENCES constant.
            Overridable so callers (and tests) can pass a synthetic/fake
            conference without touching the real config.
        min_members: minimum number of teams (with a known conference
            record) a qualifying pool must actually have, in the input data,
            to get championship statuses at all -- per division for a
            divisional conference. Checked dynamically against the input, not
            a hardcoded per-conference number, so it stays correct across
            realignment without needing updates here.
        divisions: optional Dict[team -> division name | None], injected by
            the caller. This module does no DB I/O and schedule_grid has no
            division column, so division must be passed in (from the `teams`
            table, via artifacts/schedule.py) -- passing data in as an
            argument is not a breach of that boundary, doing I/O here would
            be. When omitted (or missing/None for a team), a divisional
            conference produces no statuses at all rather than a guess.
        divisional_conferences: mapping of conference name -> format note for
            DIVISIONAL conferences; defaults to the module-level
            DIVISIONAL_CHAMPIONSHIP_CONFERENCES constant. Overridable for the
            same reason as qualifying_conferences.

    Returns:
        Dict keyed by team name, present ONLY for teams belonging to a
        qualifying conference that also meets min_members in this data --
        every other team (Independents, non-qualifying/unconfirmed-format
        conferences, a qualifying-format conference with too few teams in
        this data, or a divisional conference whose division data doesn't
        match the modeled shape) is simply ABSENT from the returned dict. A
        missing key means "not applicable," never "possible" by default --
        callers must not assume a missing key means anything but N/A.
            {
              "status": "possible" | "eliminated" | "clinched",
              "conditional_opponent": str | None,
            }
    """
    if qualifying_conferences is None:
        qualifying_conferences = QUALIFYING_CHAMPIONSHIP_CONFERENCES
    if divisional_conferences is None:
        divisional_conferences = DIVISIONAL_CHAMPIONSHIP_CONFERENCES
    if divisions is None:
        divisions = {}

    flat_by_conference: Dict[str, List[str]] = defaultdict(list)
    divisional_by_conference: Dict[str, Dict[Optional[str], List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for team, rec in records.items():
        conf = rec.get("conference")
        if rec.get("conf_wins") is None:
            continue
        if conf in qualifying_conferences:
            flat_by_conference[conf].append(team)
        elif conf in divisional_conferences:
            # elif, not a second if: the two format dicts are disjoint by
            # construction, and if an override ever lists a conference in
            # both, the flat (pre-existing) reading wins rather than the
            # newer code path silently taking over.
            divisional_by_conference[conf][divisions.get(team)].append(team)

    pools: List[Dict[str, Any]] = []
    for conf, teams in flat_by_conference.items():
        if len(teams) < min_members:
            logger.info(
                "Conference %r has only %d team(s) with a conference record in this "
                "data (< min_members=%d) -- skipping championship status for it even "
                "though it is in the qualifying-format list; likely incomplete data "
                "for this season, or a genuinely small membership.",
                conf, len(teams), min_members,
            )
            continue
        pools.append({"conference": conf, "division": None, "teams": teams, "top_n": 2})

    for conf, teams_by_division in divisional_by_conference.items():
        # Conservative gate: a divisional conference is computed ONLY when the
        # data matches the shape the algorithm models. Any mismatch skips the
        # WHOLE conference (both divisions), because a division race computed
        # from a partial or unexpected division map could assert something
        # false -- and a blank Sun Belt column is an acceptable outcome where
        # a wrong "Eliminated" is not.
        if None in teams_by_division:
            logger.info(
                "Conference %r is configured as divisional but %d of its team(s) have no "
                "division in the injected division map -- skipping championship status for "
                "the whole conference rather than guessing a division race.",
                conf, len(teams_by_division[None]),
            )
            continue
        if len(teams_by_division) != 2:
            logger.info(
                "Conference %r is configured as divisional but this data has %d division(s) "
                "(%s), not the 2 that a division-champion-vs-division-champion title game "
                "models -- skipping championship status for the whole conference.",
                conf, len(teams_by_division), sorted(teams_by_division),
            )
            continue
        undersized = {d: len(t) for d, t in teams_by_division.items() if len(t) < min_members}
        if undersized:
            logger.info(
                "Conference %r has division(s) with fewer than min_members=%d teams carrying "
                "a conference record in this data (%s) -- skipping championship status for "
                "the whole conference; likely incomplete data for this season.",
                conf, min_members, undersized,
            )
            continue
        for division in sorted(teams_by_division):
            pools.append({
                "conference": conf,
                "division": division,
                # Only the division CHAMPION reaches the title game, so top_n=1
                # where a flat conference uses 2.
                "top_n": 1,
                "teams": teams_by_division[division],
            })

    statuses_by_pool: List[Dict[str, str]] = []
    for pool in pools:
        teams = pool["teams"]
        top_n = pool["top_n"]
        pool_label = (
            pool["conference"]
            if pool["division"] is None
            else f"{pool['conference']} ({pool['division']} division)"
        )

        # W/R/B/L come straight from each team's conference-wide tallies --
        # ALL of its conference games, divisional and non-divisional alike.
        # For a DIVISION pool that is DELIBERATE, not an oversight or a
        # convenient approximation: the Sun Belt's division champion is the
        # division member with the best record across the full conference
        # schedule, and a team's record inside its own division only ever
        # breaks ties (see DIVISIONAL_CHAMPIONSHIP_CONFERENCES above).
        # Recomputing these tallies over intra-division games only would be
        # WRONG; tests/test_division_championship_status.py's
        # test_division_race_ranks_by_overall_conference_record is the guard
        # against that regression.
        W = {t: records[t]["conf_wins"] for t in teams}
        R = {t: records[t]["conf_games_remaining"] for t in teams}
        B = {t: W[t] + R[t] for t in teams}
        L = W  # worst case for T is simply its current banked win count

        statuses: Dict[str, str] = {}
        for t in teams:
            other_w_desc = sorted((W[i] for i in teams if i != t), reverse=True)
            # Generalized from the pre-division hardcoded [1]: with top_n
            # teams advancing, T is out once top_n OTHER teams have banked
            # more wins than T can ever reach. top_n=2 -> index 1 (identical
            # to the previous behavior for every flat conference); top_n=1
            # (division race) -> index 0.
            #
            # min_members >= 4 guarantees >= 3 other teams, so the index
            # always exists in practice; the -1 fallback is a defensive belt-
            # and-braces value that can never trigger a false "eliminated"
            # (B_t is never negative) if this function is ever called with a
            # smaller min_members override.
            nth_highest_other_w = other_w_desc[top_n - 1] if len(other_w_desc) >= top_n else -1
            eliminated = B[t] < nth_highest_other_w

            # Generalized the same way from the pre-division hardcoded <= 1.
            others_reaching_floor = sum(1 for i in teams if i != t and B[i] >= L[t])
            clinched = others_reaching_floor <= top_n - 1

            if eliminated and clinched:
                # Structurally should not happen -- if it ever does, it means
                # the two tests' inputs disagree with each other, which is a
                # bug worth knowing about loudly. Prefer the more
                # conservative-sounding call rather than picking silently.
                logger.error(
                    "Team %r in pool %r tested BOTH eliminated and clinched "
                    "simultaneously (should be impossible) -- treating as eliminated. "
                    "top_n=%d W=%s R=%s B=%s L=%s", t, pool_label, top_n, W, R, B, L,
                )
                statuses[t] = "eliminated"
            elif eliminated:
                statuses[t] = "eliminated"
            elif clinched:
                statuses[t] = "clinched"
            else:
                statuses[t] = "possible"
        statuses_by_pool.append(statuses)

    clinched_by_pool = [
        [t for t in pool["teams"] if statuses[t] == "clinched"]
        for pool, statuses in zip(pools, statuses_by_pool)
    ]

    result: Dict[str, Dict[str, Any]] = {}
    for idx, (pool, statuses) in enumerate(zip(pools, statuses_by_pool)):
        if pool["division"] is None:
            # Flat conference: both title-game slots come out of this one
            # pool, so the clinched team to name is in the pool itself.
            source_pool_indices = [idx]
        else:
            # Division race: the opponent slot belongs to the OTHER division,
            # so look there and never in this team's own division. The gate
            # above guarantees a divisional conference contributes exactly two
            # pools, so this resolves to exactly one sibling.
            source_pool_indices = [
                j for j, other in enumerate(pools)
                if other["conference"] == pool["conference"] and j != idx
            ]
        clinched_candidates = [t for j in source_pool_indices for t in clinched_by_pool[j]]
        conditional_opponent_name = (
            clinched_candidates[0] if len(clinched_candidates) == 1 else None
        )

        for t in pool["teams"]:
            if statuses[t] == "possible":
                result[t] = {"status": "possible", "conditional_opponent": conditional_opponent_name}
            else:
                # both "clinched" and "eliminated" carry no conditional_opponent
                result[t] = {"status": statuses[t], "conditional_opponent": None}

    return result


def compute_bowl_eligibility(records: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Compute bowl-eligibility status per team, current season, independent
    of conference-championship status -- every team gets an entry
    (Independents and non-qualifying-format conferences included).

    Default "possible". "eligible" once overall wins >= 6. "ineligible"
    once overall losses >= 7 (7 losses in a ~12-13 game season makes 6 wins
    mathematically impossible -- confirmed reading per the T4a handoff;
    wins>=6 is checked first so a team that has already banked its 6th win
    stays "eligible" even if it goes on to also collect a 7th loss).

    Args:
        records: output of compute_team_records().

    Returns:
        Dict keyed by team name -> "possible" | "eligible" | "ineligible".
    """
    result: Dict[str, str] = {}
    for team, rec in records.items():
        wins = rec["wins"]
        losses = rec["losses"]
        if wins >= 6:
            result[team] = "eligible"
        elif losses >= 7:
            result[team] = "ineligible"
        else:
            result[team] = "possible"
    return result


def compute_standings(
    rows: Iterable[Dict[str, Any]],
    season: int,
    divisions: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Top-level convenience entry point tying the three computations above
    together. T4b can call this directly rather than orchestrating the
    sub-functions itself; the sub-functions remain independently callable
    (and are what the synthetic test below exercises directly) for
    testability and for callers that only need one piece.

    Args:
        rows: schedule_grid rows as dicts (see module docstring).
        season: the season to compute standings for.
        divisions: optional Dict[team -> division name | None] from the
            caller (schedule_grid has no division column -- see the module
            docstring). Required for a divisional conference such as the Sun
            Belt to get any championship status at all; omitting it leaves
            those teams' championship_status None, exactly as before
            divisions were supported.

    Returns:
        Dict keyed by team name:
            {
              "conference": str | None,
              "record": {"wins": int, "losses": int},
              "conf_record": {"wins": int, "losses": int} | None,
              "championship_status": "possible" | "eliminated" | "clinched" | None,
              "conditional_opponent": str | None,
              "bowl_status": "possible" | "eligible" | "ineligible",
            }
        championship_status/conditional_opponent are None for every team
        not in a qualifying conference (see compute_conference_championship_status) --
        this is the deliberate blank fallback, not a guessed "possible".
    """
    records = compute_team_records(rows, season)
    championship = compute_conference_championship_status(records, divisions=divisions)
    bowl = compute_bowl_eligibility(records)

    combined: Dict[str, Dict[str, Any]] = {}
    for team, rec in records.items():
        champ = championship.get(team)
        conf_record = (
            None
            if rec["conf_wins"] is None
            else {"wins": rec["conf_wins"], "losses": rec["conf_losses"]}
        )
        combined[team] = {
            "conference": rec["conference"],
            "record": {"wins": rec["wins"], "losses": rec["losses"]},
            "conf_record": conf_record,
            "championship_status": champ["status"] if champ else None,
            "conditional_opponent": champ["conditional_opponent"] if champ else None,
            "bowl_status": bowl[team],
        }
    return combined
