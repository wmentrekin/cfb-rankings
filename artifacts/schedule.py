"""
schedule.py -- Season Grid artifact builder + R2 publisher (T4b).

Queries schedule_grid + teams for one season, calls
artifacts/schedule_standings.py's compute_standings() for record/conf_record
and championship/bowl status, resolves logos via artifacts/rankings.py's
_resolve_logo(), groups/sorts teams by conference, buckets postseason rows
into the fixed CFP-round slots, identifies each qualifying conference's
championship game and the Army-Navy game (see the two heuristic sections
below), synthesizes bye-week cells so every team has exactly one cell per
canonical column, and publishes the resulting JSON to R2 under a
season-scoped key layout (schedule/{season}/latest.json + schedule/index.json
-- NOT rankings' per-week snapshot scheme).

The coordination artifacts this module was specified from have been removed with
their feature's scratch directory; the contract they described is the public
surface below plus tests/test_season_schedule_publish.py, which pins it.

EXCEPTION DISCIPLINE: every public function here follows artifacts/r2.py's
exact never-raise, log-and-continue contract -- a schedule-artifact publish
failure must never break the rest of the pipeline run (model training, DB
insert, rankings publish have already happened by the time this runs).
"""

import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
from dotenv import load_dotenv  # type: ignore
from sqlalchemy import create_engine  # type: ignore

from artifacts import schedule_standings
from artifacts.bowl_names import short_bowl_name
from artifacts.r2 import get_r2_client, upload_json
from artifacts.rankings import CONFERENCE_DISPLAY_NAMES, _resolve_logo, compute_rank_and_delta
from artifacts.tiebreaker_engine import order_tied_group
from artifacts.tiebreaker_rules import RuleSet, TiebreakerConfigError, load_conference_rules, rules_for
from artifacts.tiebreaker_steps import TiebreakContext
from utils import football_day, get_cfb_week

# Reuse the same logger name main.py configures via utils.setup_logging, so warnings from this
# module surface through the pipeline's existing stdout/file handlers when run via main.py, and
# still fall back to Python's default stderr handler when this module is used standalone.
logger = logging.getLogger("cfb_lp")


# ---------------------------------------------------------------------------
# Fixed conference group order (plan.yaml key_decisions / T4b handoff
# implementation_notes) -- a small ordered constant, not inferred at query
# time, so it's easy for the user to reorder later. Uses the RAW
# teams.conference / schedule_grid.conference string values (matching
# CONFERENCE_DISPLAY_NAMES' keys in artifacts/rankings.py), not the display
# names -- display mapping happens at payload-build time via
# _display_conference_name() below. A conference with zero FBS teams in a
# given season's data simply produces no entry -- not all 11 are assumed to
# always have members every season.
# ---------------------------------------------------------------------------
CONFERENCE_ORDER: List[str] = [
    "ACC",
    "Big 12",
    "Big Ten",
    "SEC",
    schedule_standings.INDEPENDENT_CONFERENCE_VALUE,  # "FBS Independents"
    "American Athletic",
    "Conference USA",
    "Mid-American",
    "Mountain West",
    "Pac-12",
    "Sun Belt",
]

# Confirmed live (SELECT DISTINCT team FROM schedule_grid WHERE team ILIKE
# '%army%' OR '%navy%') -- exact spellings, not "Army West Point" or similar.
ARMY_TEAM = "Army"
NAVY_TEAM = "Navy"

# ---------------------------------------------------------------------------
# FLEX_WEEK_TBD_CONFIG (T6) -- curated (season, conference, week-bucket) rule
#
# The 2026 Pac-12 plays a 7-game round robin that concludes in week 12, per the
# Pac-12's own 2026 schedule announcement, which also states week 13's games are
# flex games that do NOT count toward conference standings (they will arrive as
# conference_game=false, so they cannot affect the clinch/eliminate math or
# identify_conference_championship_games -- no extra guard is needed for that).
#
# Nothing in schedule_grid distinguishes a flex week from an ordinary bye: bucket
# 13 is simply absent from the data for all 8 Pac-12 teams today, exactly like a
# real bye week would be. This config is what turns that absence into a `tbd`
# cell (reusing the existing `tbd` status -- no new status value was introduced) instead of `bye`,
# scoped narrowly to this one (season, conference, week-bucket) triple so no
# other conference and no other season is affected.
#
# SELF-CLEARING: _build_team_weeks only ever consults this config in the
# fallback branch reached when a team has NO real schedule_grid row for that
# slot (see _bye_cell's call site). Once a real flex game is ingested for a
# team, that row is present in team_slot_rows and renders as a normal game --
# this rule is never consulted for that team/slot again. No per-team
# maintenance: membership is derived from CONFERENCE_ORDER's raw "Pac-12" key
# at payload-build time (via _flex_week_tbd_slot_ids), not a hardcoded roster.
#
# RE-VERIFY EACH OFFSEASON: this is a season-specific scheduling fact, not a
# structural rule, and it WILL be wrong for a future season -- confirm the
# flex week's existence and its week bucket fresh each year (or once week 13
# is actually announced and populated) rather than assuming this entry still
# applies. Same discipline as QUALIFYING_CHAMPIONSHIP_CONFERENCES in
# artifacts/schedule_standings.py, which carries the identical warning.
# ---------------------------------------------------------------------------
FLEX_WEEK_TBD_CONFIG: List[Dict[str, Any]] = [
    {"season": 2026, "conference": "Pac-12", "week_bucket": 13},
]


def _flex_week_tbd_slot_ids(season: int, raw_conference: Optional[str]) -> set:
    """slot_ids (e.g. {'week-13'}) that should render `tbd` instead of `bye` for a team in
    `raw_conference` this `season`, per FLEX_WEEK_TBD_CONFIG. Empty for every conference/season
    not explicitly configured."""
    return {
        f"week-{entry['week_bucket']}"
        for entry in FLEX_WEEK_TBD_CONFIG
        if entry["season"] == season and entry["conference"] == raw_conference
    }

# slot_id is the stable contract the frontend keys off; the LABEL is what the user
# reads. The championship/Army-Navy labels are derived per season in
# build_canonical_columns() rather than hardcoded, because the week number they fall
# on is not constant: get_cfb_week()'s anchor (the Tuesday on/before Aug 24) lands on
# a different date each year, so the conference-championship weekend is week 15 in
# 2024 and 2025 but week 14 in 2026. The labels in these two constants are fallbacks
# used only when a season has no regular-season rows at all to derive a number from.
CONF_CHAMPIONSHIP_SLOT_ID = "conf-championship"
ARMY_NAVY_SLOT_ID = "army-navy"
CONF_CHAMPIONSHIP_SLOT: Tuple[str, str] = (CONF_CHAMPIONSHIP_SLOT_ID, "Conference Championship")
ARMY_NAVY_SLOT: Tuple[str, str] = (ARMY_NAVY_SLOT_ID, "Army-Navy Game Week")
CFP_SLOTS: List[Tuple[str, str]] = [
    ("cfp-r1-bowls", "Bowls"),
    ("cfp-quarterfinals", "CFP Quarterfinals"),
    ("cfp-semifinals", "CFP Semifinals"),
    ("cfp-national-championship", "CFP National Championship"),
]
_CFP_SLOT_IDS = {slot_id for slot_id, _ in CFP_SLOTS}

# T4/AC5: distinct status for a team that earned a first-round CFP bye -- see _cfp_bye_cell and
# the cfp-r1-bowls branch of _build_team_weeks. Previously this slot NEVER fell through to a bye
# at all (it unconditionally emitted the bowl-eligibility placeholder, reading as merely
# "Eligible" -- indistinguishable from, and arguably worse than, a team that missed the playoff
# entirely). Deliberately deferred when the grid first shipped and added later; the behaviour
# it replaced is described above so the reason survives without the planning note.
CFP_BYE_STATUS = "cfp_bye"


# ---------------------------------------------------------------------------
# Date/datetime coercion -- schedule_grid rows arrive here either as plain
# dicts (synthetic/test input) or as pandas-derived dicts (real DB query via
# pd.read_sql_query(...).to_dict("records")), where start_date lands as a
# pandas Timestamp, not a plain datetime. utils.get_cfb_week expects a plain
# `date`, so every call site normalizes through this helper first.
# ---------------------------------------------------------------------------
def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "to_pydatetime"):  # pandas Timestamp
        return value.to_pydatetime().date()
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    raise ValueError(f"schedule.py: unsupported start_date value/type for date coercion: {value!r}")


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError(f"schedule.py: unsupported start_date value/type for datetime coercion: {value!r}")


def _get_week_slot(row: Dict[str, Any]) -> int:
    """Per-row display-week-slot via utils.get_cfb_week -- NOT raw `week`.

    T3 confirmed raw `week` collides across true-Week-0-vs-Week-1 games (129 real examples);
    this is the fix, reused as-is per the handoff (no reimplementation of the date-anchor math).

    Buckets on utils.football_day() rather than the raw UTC date, so a Monday-night kickoff
    (00:00 UTC Tuesday for an 8pm ET start) groups with the weekend it belongs to instead of
    opening the next week -- see that function for why midnight UTC is the wrong boundary.
    """
    return get_cfb_week(football_day(_to_datetime(row["start_date"])), None)


def _is_army_navy_pairing(team: Optional[str], opponent: Optional[str]) -> bool:
    return {team, opponent} == {ARMY_TEAM, NAVY_TEAM}


def _display_conference_name(raw_conference: Optional[str]) -> str:
    """
    Map a raw conference value to its standardized display string via
    artifacts.rankings.CONFERENCE_DISPLAY_NAMES -- the SAME dict the Rankings
    tab uses, imported directly rather than re-derived, so both tabs show
    identical conference labels for the same teams. Unmapped values are
    logged and passed through unchanged (mirrors rankings.py's own
    _display_conference() fallback, which is private to that module).
    """
    if raw_conference not in CONFERENCE_DISPLAY_NAMES:
        logger.warning("schedule.py: unmapped conference value %r; passing through unchanged.", raw_conference)
        return raw_conference
    return CONFERENCE_DISPLAY_NAMES[raw_conference]


# ---------------------------------------------------------------------------
# CONFERENCE-CHAMPIONSHIP GAME IDENTIFICATION
#
# A championship game is identified by a POSITIVE STRUCTURAL SIGNAL: it is the
# lone conference game occupying a week bucket by itself, strictly later than
# the bucket holding that conference's regular slate. If a conference's latest
# bucket holds more than one game, that is a regular slate and nothing is
# identified.
#
# WHY NOT "the latest conference game" (the rule this replaced): that rule had
# no notion of whether a championship game existed at all. For an in-progress
# season with none scheduled, it fell back to rivalry week and tie-broke
# alphabetically -- six ACC games share an identical start_date in 2026, and the
# tie-break published "Wake Forest vs Duke" as a determined championship matchup
# in the live artifact. Requiring separation from the slate is what makes the
# in-progress case correctly identify nothing.
#
# ARMY-NAVY CARVE-OUT (retained from the previous rule, still needed): Army
# joined the American Athletic Conference in 2024 and Navy has been a member
# since 2015, so the annual Army-Navy game is tagged conference_game=true,
# conference='American Athletic'. It falls LATER than the real AAC title game
# (2024-12-14 vs 2024-12-07), and it sits alone in its own week bucket -- so it
# satisfies the new rule's shape perfectly and would be identified as AAC's
# championship game. It is excluded unconditionally, for every conference,
# consistent with the dedicated Army-Navy column it is always diverted into.
#
# ACCEPTED LIMITATION: a make-up or postponed regular-season conference game
# scheduled into the championship weekend shares that bucket, so the conference
# identifies nothing that season and its title game renders in a week column
# instead of the championship column. That is the conservative direction --
# rendering a real game in the wrong column beats asserting a matchup that was
# never determined, which is the failure this rule exists to prevent.
#
# THE OTHER DIRECTION -- a real false positive, not just a misplaced column: a
# make-up or postponed game that instead sits ALONE in a late bucket (no
# scheduling collision with the real slate) satisfies this rule's shape
# perfectly and gets identified as the conference's championship game, even
# though it is not one. UPDATED CONSEQUENCE (season-grid-postseason-format):
# this used to be cosmetic -- the game rendered in the Conference Championship
# column instead of a week column, nothing else. It no longer is. The same
# identification now also feeds _exclude_championship_games_from_conf_records
# (subtracts this game from both participants' DISPLAYED conf_record) and
# _resolve_conference_championship_outcomes (crowns its winner and loser, who
# then sort first and second via _tier -- see _sort_conference_teams) -- so a
# false positive here misstates a real, played conference game's tally and
# promotes the wrong team to the top of its conference. Still accepted, not
# mitigated further in this pass: the
# structural signal (lone game, later than the regular slate) is the best
# available without hand-curating a real championship-game schedule, and a
# make-up game landing alone on/after what would be championship weekend is
# an edge case, not the common path.
#
# VERIFIED against live 2024, 2025 and 2026 data (Supabase project
# oyqgmbgwohlnrxodvilt): every real title game is still identified for both
# completed seasons, 2024's two-team Pac-12 remnant correctly identifies nothing,
# and 2026 identifies nothing for every conference.
#
# RE-VERIFIED for T4b (default `qualifying_conferences` widened to the union of
# QUALIFYING_CHAMPIONSHIP_CONFERENCES and DIVISIONAL_CHAMPIONSHIP_CONFERENCES,
# see the in-function comment below): 2024 and 2025 Sun Belt and Pac-12 title
# games are now identified where they previously were not, the rest of the
# 2024/2025 real-data picture is unchanged, and 2026 still identifies nothing
# for every conference.
#
# T1 (season-grid-standings-fixes, K1/K2/K3) -- AUTHORITATIVE NOTES SIGNAL:
# CFBD changed its 2025 data model so a conference championship game arrives
# with conference_game=FALSE (not TRUE) and notes='<Conference> Championship'.
# The structural rule above requires conference_game=TRUE to even nominate a
# candidate, so for 2025 it identifies NOTHING for any of the nine real
# championship games -- every one of them silently falls into a week column
# instead of the dedicated Conference Championship column.
#
# `notes` is checked FIRST, as an ADDITIONAL, higher-priority signal layered on
# top of the structural rule -- not a replacement for it. A row is an
# AUTHORITATIVE championship match when ALL of: season_type=='regular';
# notes, stripped, ends with "Championship"; the prefix (notes minus that
# trailing "Championship", stripped) resolves -- via
# _CHAMPIONSHIP_NOTES_CONFERENCE_ALIASES, else identity -- to that SAME row's
# own `conference` value; that conference is in `qualifying_conferences`; and
# the row is not the Army-Navy pairing (same carve-out as the structural rule,
# see above -- Army-Navy's own notes, when present, would never end with
# "Championship", but the carve-out costs nothing to keep for defense in
# depth).
#
# On an authoritative match, that conference's championship game is resolved
# from `notes` ALONE -- the conference_game gate, the MEMBER-COUNT gate and
# the bucket-isolation ("lone game in the latest bucket") rule are all
# bypassed for it, and the structural rule does not run for that conference at
# all this season. This is safe specifically BECAUSE those three gates guard
# an INFERENTIAL rule (a shape in the data that's usually, not certainly, a
# championship game), while a `notes` label naming the exact conference is
# AUTHORITATIVE -- CFBD is telling us directly, not leaving us to guess from
# game count and date ordering.
#
# In particular, bypassing the member-count gate is safe here ONLY because the
# 2025 two-team Pac-12 remnant (Oregon State/Washington State, who played each
# other twice and have no title game) carries NO notes row at all -- it has
# nothing to match, so it never reaches the bypass and still falls through to
# (and is still stopped by) the structural rule's member-count gate below.
# This is pinned by test_two_team_conference_playing_twice_is_not_a_championship
# in tests/test_championship_diversion_scope.py, extended with a notes=None
# assertion alongside the pre-existing conference_game=True one -- if a future
# CFBD data shape ever attached a notes value to a non-championship rematch
# like this, the member-count gate would no longer be there to catch it.
#
# If NO authoritative notes match exists for a conference, that conference
# falls through to today's structural rule COMPLETELY UNCHANGED -- this is how
# every pre-2025 season (notes is NULL on every row, 2014-2024 confirmed
# live) and 2026 (notes non-null on exactly 3 week-1 kickoff-classic rows,
# none ending in "Championship") keep behaving exactly as before this change.
#
# AMBIGUITY: notes matching is per-row, but a real game has two team-
# perspective rows (both carrying the same notes and the same conference for
# a championship game) -- collapsed to a single game_id naturally. If two or
# more DISTINCT game_ids in the same conference somehow BOTH produce an
# authoritative notes match in one season, that is a "shouldn't happen" (a
# conference plays at most one championship game): logged and the whole
# conference falls through to the structural rule rather than guessing which
# match is the real one.
#
# MISMATCH (K2): a row whose notes ends with "Championship" but whose prefix
# does NOT resolve to that row's OWN conference is logged and ignored (falls
# through to the structural rule for its conference) rather than silently
# treated as a non-match. This is deliberate, not paranoia: nobody has ever
# seen what CFBD actually calls a Pac-12 title game -- the real 2025 Pac-12
# has no championship game at all, and the 2026 Pac-12's has not been played
# yet -- so if CFBD's naming ever surprises us there (or for any other
# conference, on a future data-shape change), this is how that becomes a
# visible log line instead of a silent miss that quietly reintroduces this
# same defect.
# ---------------------------------------------------------------------------

# T1/K3: CFBD's `notes` abbreviates two conference names differently than the
# raw `conference` string schedule_grid rows otherwise carry (confirmed live,
# 2025 season): 'American Championship' for the American Athletic Conference,
# and 'MAC Championship' for the Mid-American Conference. The other seven 2025
# championship-conference notes values match their row's own `conference`
# string by plain equality after stripping " Championship" and need no alias.
# This is an IDENTIFICATION concern (resolving a notes string to the
# conference it names), not a standings-format concern, so it lives here
# beside the function that consumes it rather than in schedule_standings.py.
# RE-VERIFY EACH OFFSEASON, same discipline as QUALIFYING_CHAMPIONSHIP_CONFERENCES
# and FLEX_WEEK_TBD_CONFIG -- CFBD's notes abbreviations are not a documented,
# stable contract and could change or gain new curated conferences.
_CHAMPIONSHIP_NOTES_CONFERENCE_ALIASES: Dict[str, str] = {
    "American": "American Athletic",
    "MAC": "Mid-American",
}


def identify_conference_championship_games(
    rows: List[Dict[str, Any]],
    season: int,
    qualifying_conferences: Optional[Dict[str, str]] = None,
) -> Dict[str, int]:
    """
    Returns: Dict[conference name -> game_id] for the identified championship
    game of each qualifying conference that has at least one candidate row
    this season. A qualifying conference with no conference_game=true rows
    this season (e.g. incomplete data) simply has no entry.
    """
    # Deliberately gated to the UNION of the two curated format lists -- flat-format
    # QUALIFYING_CHAMPIONSHIP_CONFERENCES plus divisional-format
    # DIVISIONAL_CHAMPIONSHIP_CONFERENCES -- NOT either list alone, and NOT every
    # conference in the data. This is a deliberate middle ground:
    #
    #   - The flat list alone UNDER-covers: it gates the CLINCH/ELIMINATE status math,
    #     which only models the flat top-2-of-one-pool shape, so it has no entry for the
    #     Sun Belt (divisional). Using it to gate DIVERSION too meant a real Sun Belt
    #     title game was never pulled out of its week column -- the Sun Belt now gets a
    #     computed status in the Conference Championship column (via
    #     DIVISIONAL_CHAMPIONSHIP_CONFERENCES) AND its actual title game sitting in a
    #     week column, plus the near-empty week column that game creates. That is
    #     exactly the phantom-column bug this project already fixed once, recurring for
    #     a conference the earlier fix didn't cover.
    #
    #   - Widening to EVERY conference in the data was tried and REVERTED in a previous
    #     pass, for two reasons that still apply: (1) the default set built from raw
    #     `conference` values carries no FBS filter, so an FCS conference appearing in
    #     the rows would become eligible; (2) it widens the surface for this rule's
    #     known false positive -- a make-up or postponed game sitting alone in a late
    #     bucket gets identified as a championship game (see ACCEPTED LIMITATION above).
    #
    #   - The union of the two curated lists is the set of conferences that actually
    #     HAVE a championship game. It cannot admit an FCS conference, because both
    #     lists are hand-maintained FBS-only. It extends the make-up-game false-positive
    #     exposure only to the Pac-12 and Sun Belt (the two conferences newly added by
    #     using the union), which is proportionate -- that exposure already exists today
    #     for the eight conferences the flat list covers.
    #
    #   - REJECTED ALTERNATIVE: a `neutral_site` discriminator. It's tempting because
    #     "every FBS conference title game is neutral-site" sounds like a clean, format-
    #     agnostic signal -- but it's false. The Sun Belt, Mountain West and Conference
    #     USA all host their title game at a division/top-seed campus site, and the
    #     Pac-12's 2026 game is at the top seed's home stadium. Gating on neutral_site
    #     would silently un-divert exactly the conferences this task is fixing. Do not
    #     reintroduce it.
    #
    # Built as a fresh dict merge (not hardcoded) so it stays correct if either curated
    # list gains a member in a future offseason re-verification pass.
    if qualifying_conferences is None:
        qualifying_conferences = {
            **schedule_standings.QUALIFYING_CHAMPIONSHIP_CONFERENCES,
            **schedule_standings.DIVISIONAL_CHAMPIONSHIP_CONFERENCES,
        }

    # T1/K1/K2: AUTHORITATIVE NOTES SIGNAL -- see the module comment above for the full
    # rationale. Runs first and independently of the structural candidate-gathering below;
    # a conference resolved here is excluded from the structural pass entirely (see
    # `notes_resolved_conferences` below), not merely given a preferred candidate.
    _CHAMPIONSHIP_SUFFIX = "Championship"
    notes_game_ids_by_conf: Dict[str, set] = defaultdict(set)
    for row in rows:
        if row.get("season") != season:
            continue
        if row.get("season_type") != "regular":
            continue
        notes = row.get("notes")
        if not isinstance(notes, str):
            continue
        stripped_notes = notes.strip()
        if not stripped_notes.endswith(_CHAMPIONSHIP_SUFFIX):
            # K2, widened after the PR #16 review: the MATCH predicate stays a strict,
            # case-sensitive endswith, but the LOG trigger is deliberately looser. Without
            # this branch the "naming surprise becomes a visible log line" property below
            # only held for a PREFIX surprise; a SUFFIX one -- 'SEC Championship Game', or
            # 'SEC championship' -- fell out here in silence and silently reintroduced the
            # exact defect this notes path exists to fix, for every conference at once.
            # Matching loosely instead would be the wrong trade: it would let a genuinely
            # unrelated game carrying the word through. Log loudly, match strictly.
            if _CHAMPIONSHIP_SUFFIX.lower() in stripped_notes.lower():
                logger.warning(
                    "schedule.py: season=%s game_id=%s notes=%r contains %r but does not END "
                    "with it, so it is NOT treated as a championship-game label. If CFBD has "
                    "changed its naming convention, this conference will fall back to the "
                    "structural rule and may identify nothing at all -- update "
                    "_CHAMPIONSHIP_SUFFIX rather than ignoring this warning.",
                    season, row.get("game_id"), notes, _CHAMPIONSHIP_SUFFIX,
                )
            continue
        if _is_army_navy_pairing(row.get("team"), row.get("opponent")):
            continue  # see module docstring section above -- deliberate carve-out
        row_conf = row.get("conference")
        prefix = stripped_notes[: -len(_CHAMPIONSHIP_SUFFIX)].strip()
        resolved_conf = _CHAMPIONSHIP_NOTES_CONFERENCE_ALIASES.get(prefix, prefix)
        if resolved_conf != row_conf:
            # K2: a naming surprise, not a silent miss -- see module comment ("MISMATCH").
            logger.warning(
                "schedule.py: season=%s game_id=%s notes=%r (resolved prefix %r) does not match "
                "its own row's conference %r -- ignoring this row's notes signal for championship "
                "identification and falling through to the structural rule for %r.",
                season, row.get("game_id"), notes, resolved_conf, row_conf, row_conf,
            )
            continue
        if row_conf not in qualifying_conferences:
            continue
        notes_game_ids_by_conf[row_conf].add(row.get("game_id"))

    result: Dict[str, int] = {}
    notes_resolved_conferences: set = set()
    for conf, game_ids in notes_game_ids_by_conf.items():
        if len(game_ids) > 1:
            # "Shouldn't happen" -- a conference plays at most one championship game, so two
            # distinct game_ids both matching authoritatively means something is wrong with the
            # data, not with this function's logic. Fall through to the structural rule rather
            # than guess which of the two is real.
            logger.warning(
                "schedule.py: conference %r season=%s has %d distinct game_ids with an "
                "authoritative championship-notes match (%s) -- a conference plays at most one "
                "championship game, so this should be structurally impossible. Falling through "
                "to the structural rule for this conference instead of picking one arbitrarily.",
                conf, season, len(game_ids), sorted(game_ids),
            )
            continue
        result[conf] = next(iter(game_ids))
        notes_resolved_conferences.add(conf)

    candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("season") != season:
            continue
        if row.get("season_type") != "regular":
            continue
        conf = row.get("conference")
        if conf in notes_resolved_conferences:
            # T1/K1: an authoritative notes match already resolved this conference above --
            # the structural rule (and its conference_game/member-count/bucket-isolation gates)
            # does not run for it at all this season. See module comment.
            continue
        if not row.get("conference_game"):
            continue
        if conf not in qualifying_conferences:
            continue
        if _is_army_navy_pairing(row.get("team"), row.get("opponent")):
            continue  # see module docstring section above -- deliberate carve-out, not an oversight
        if row.get("start_date") is None:
            # Cannot be bucketed, so it can neither be nor rule out a championship game.
            continue
        candidates[conf].append(row)

    for conf, crows in candidates.items():
        # MEMBER-COUNT GATE. A conference too small to hold a championship game cannot have
        # one, however its schedule happens to be shaped. Without this, the 2025 Pac-12 --
        # a two-team remnant of Oregon State and Washington State that played each other
        # TWICE (Nov 1 and Nov 29) -- satisfies every structural test below: the second
        # meeting sits alone in a bucket, strictly later than the first. The rule cannot
        # tell a title game from a rematch, so it would publish a Pac-12 championship that
        # never existed. That is the same fabrication class as the "Wake Forest vs Duke"
        # matchup this rule was written to eliminate, just reached by a different route.
        #
        # Reuses schedule_standings.MIN_QUALIFYING_MEMBERS, the threshold the clinch/
        # eliminate math already applies for the same reason, rather than inventing a
        # second number that could drift from it. Members are counted from the rows
        # themselves so this function keeps its "schedule_grid rows only" input contract.
        conf_members = {r.get("team") for r in crows if r.get("team")}
        if len(conf_members) < schedule_standings.MIN_QUALIFYING_MEMBERS:
            logger.info(
                "schedule.py: conference %r season=%s has only %d member(s) with conference "
                "games (%s) -- too few to hold a championship game, so none is identified. "
                "A small conference's teams can meet twice, which otherwise looks exactly "
                "like a title game separated from the slate.",
                conf, season, len(conf_members), sorted(conf_members),
            )
            continue

        # POSITIVE SIGNAL, not "latest game". A conference championship game is
        # structurally distinctive: it is the LONE conference game sitting in a week
        # bucket by itself, strictly later than the bucket holding that conference's
        # regular slate. Requiring that separation is what makes an in-progress season
        # identify nothing instead of guessing.
        #
        # The previous rule -- max(start_date), tie-broken alphabetically -- had no
        # notion of whether a championship game existed at all. With none scheduled
        # yet it fell back to rivalry week, where (2026 ACC) six games share an
        # identical start_date, and the tie-break produced a confidently-wrong
        # "Wake Forest vs Duke" championship matchup in the published artifact.
        by_bucket: Dict[int, set] = defaultdict(set)
        for r in crows:
            by_bucket[_get_week_slot(r)].add(r["game_id"])

        if len(by_bucket) < 2:
            # Every conference game in one bucket -> nothing is separated from the
            # slate, so there is no championship game to identify.
            continue

        latest_bucket = max(by_bucket)
        latest_game_ids = by_bucket[latest_bucket]
        if len(latest_game_ids) != 1:
            logger.info(
                "schedule.py: conference %r season=%s has %d distinct conference games sharing "
                "its latest week bucket (week %s, game_ids=%s) -- that is a regular slate, not a "
                "championship game, so none is identified for this conference.",
                conf, season, len(latest_game_ids), latest_bucket, sorted(latest_game_ids),
            )
            continue

        result[conf] = next(iter(latest_game_ids))
    return result


def identify_army_navy_game(rows: List[Dict[str, Any]], season: int) -> Optional[int]:
    """Returns the game_id of Army-vs-Navy for this season (season_type='regular'), or None."""
    for row in rows:
        if row.get("season") != season:
            continue
        if row.get("season_type") != "regular":
            continue
        if _is_army_navy_pairing(row.get("team"), row.get("opponent")):
            return row.get("game_id")
    return None


# ---------------------------------------------------------------------------
# CFP-ROUND BUCKETING (postseason rows only)
#
# Case-insensitive substring match on playoff_round_name per the handoff:
#   contains 'quarterfinal'                                   -> CFP Quarterfinals
#   contains 'semifinal'                                       -> CFP Semifinals
#   contains 'national championship', or ('championship' and
#     NOT 'conference')                                        -> CFP National Championship
#   anything else (incl. null/empty, non-CFP bowls, 'first
#     round')                                                   -> CFP 1st Round + Other Bowls
#
# playoff_round_order is deliberately NOT used, even as a tie-break: T2's
# actual implementation never called CFBD's separate /playoffs/cfp/games
# endpoint (a documented, deliberate scope-narrowing from T2, not an
# oversight -- see plan.yaml/implementation-report.yaml), so
# playoff_round_name/order's exact real-world values were never live-
# verified (0 postseason rows exist in this sandbox's DB). Per the handoff's
# own caution, round_order's numeric convention is unconfirmed -- coding a
# tie-break against it here would be guessing at an unverified contract, so
# it's left unused. Flagged again in this task's report.
# ---------------------------------------------------------------------------
def _bucket_cfp_slot(row: Dict[str, Any]) -> Tuple[str, str]:
    round_name = (row.get("playoff_round_name") or "").strip().lower()
    if "quarterfinal" in round_name:
        return CFP_SLOTS[1]
    if "semifinal" in round_name:
        return CFP_SLOTS[2]
    if "national championship" in round_name or ("championship" in round_name and "conference" not in round_name):
        return CFP_SLOTS[3]
    return CFP_SLOTS[0]


def _game_name_for_row(row: Dict[str, Any]) -> Optional[str]:
    """game_name is non-null ONLY for a real, determined postseason game, per the JSON contract."""
    if row.get("season_type") != "postseason":
        return None
    return row.get("playoff_bowl_name") or row.get("notes") or None


def _playoff_round_for_row(row: Dict[str, Any]) -> Optional[str]:
    """K5: playoff_round is non-null ONLY for a real CFP-round postseason game -- same
    season_type gate as _game_name_for_row, plus playoff_round_name itself, which CFBD already
    leaves null for ordinary (non-CFP) bowls and every non-postseason row. Exists so the
    frontend can badge a CFP game without string-matching "College Football Playoff" inside
    game_name -- a label this same pass is shortening (K6)."""
    if row.get("season_type") != "postseason":
        return None
    return row.get("playoff_round_name")


# ---------------------------------------------------------------------------
# Per-row slot classification + per-team slot-row index
# ---------------------------------------------------------------------------
def _classify_row_slot(
    row: Dict[str, Any],
    champ_game_ids: set,
    army_navy_game_id: Optional[int],
) -> Optional[Tuple[str, str]]:
    """Returns (slot_id, label) this row belongs to, or None if season_type is unrecognized."""
    season_type = row.get("season_type")
    if season_type == "regular":
        if army_navy_game_id is not None and row.get("game_id") == army_navy_game_id:
            return ARMY_NAVY_SLOT
        if row.get("game_id") in champ_game_ids:
            return CONF_CHAMPIONSHIP_SLOT
        week_n = _get_week_slot(row)
        return (f"week-{week_n}", f"Week {week_n}")
    if season_type == "postseason":
        return _bucket_cfp_slot(row)
    logger.warning(
        "schedule.py: unrecognized season_type %r for game_id=%s team=%r; row excluded from the "
        "grid (falls through to bye/computed-placeholder fill for its slot).",
        season_type, row.get("game_id"), row.get("team"),
    )
    return None


def _build_team_slot_rows(
    rows: List[Dict[str, Any]],
    fbs_team_names: set,
    champ_game_ids: set,
    army_navy_game_id: Optional[int],
    week_slot_ids_sorted: List[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Dict[team -> Dict[slot_id -> schedule_grid row]], restricted to FBS teams for this season.

    COLLISION HANDLING (discovered during this task's live 2025 validation -- a real, not
    hypothetical, case): utils.get_cfb_week() buckets by a fixed 7-day window from one global
    season-start anchor, not each team's own natural Tue-Mon broadcast week. This means two of a
    team's real games can land in the SAME week-N slot even after the get_cfb_week() fix for the
    raw-`week`-collision problem T3 documented -- confirmed live for North Carolina 2025 (TCU,
    Tue 2025-09-02, and Charlotte, Sat 2025-09-06, both fall in the same fixed week-2 window).
    Silently overwriting one of these would make a real, played game vanish from the grid, which
    is worse than the cosmetic imperfection of a slightly-off week label. Resolution: keep the
    chronologically earlier game in its natural slot; walk the later game forward to the next
    week-N column (in canonical order) that this team doesn't already occupy. Only applies to
    week-N slots -- Conference Championship/Army-Navy/CFP-round slots are keyed by a single
    globally-unique game_id per team per season by construction, so a collision there would
    indicate a genuine data anomaly rather than this fixed-window artifact; such a case is logged
    and resolved by simple first-wins rather than shifted (there is no well-defined "next slot"
    for a one-off special column).
    """
    # Pass 1: collect every classified row per team/slot (allowing duplicates for week-N slots).
    by_team_slot: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        team = row.get("team")
        if team not in fbs_team_names:
            continue
        classified = _classify_row_slot(row, champ_game_ids, army_navy_game_id)
        if classified is None:
            continue
        slot_id, _label = classified
        by_team_slot[team][slot_id].append(row)

    result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for team, slot_rows in by_team_slot.items():
        occupied: set = set()
        # Non-colliding slots first, so "next open slot" search below sees them as taken.
        for slot_id, candidate_rows in slot_rows.items():
            if len(candidate_rows) == 1:
                result[team][slot_id] = candidate_rows[0]
                occupied.add(slot_id)

        for slot_id, candidate_rows in slot_rows.items():
            if len(candidate_rows) <= 1:
                continue
            candidate_rows = sorted(candidate_rows, key=lambda r: _to_datetime(r["start_date"]))
            if slot_id not in week_slot_ids_sorted:
                # A non-week slot collision (conf-championship/army-navy/a CFP-round slot) --
                # should be structurally impossible given how those are identified; first-wins.
                logger.error(
                    "schedule.py: team %r has %d rows colliding on non-week slot %r (game_ids=%s) "
                    "-- this should be structurally impossible (each such slot is keyed by a "
                    "single globally-unique game_id). Keeping the earliest by start_date.",
                    team, len(candidate_rows), slot_id, [r.get("game_id") for r in candidate_rows],
                )
                result[team][slot_id] = candidate_rows[0]
                occupied.add(slot_id)
                continue

            result[team][slot_id] = candidate_rows[0]
            occupied.add(slot_id)
            start_idx = week_slot_ids_sorted.index(slot_id)
            for later_row in candidate_rows[1:]:
                placed = False
                for candidate_slot in week_slot_ids_sorted[start_idx + 1:]:
                    if candidate_slot not in occupied:
                        logger.warning(
                            "schedule.py: team %r has two real games in the same get_cfb_week() "
                            "window (natural slot %r): game_id=%s kept there; game_id=%s "
                            "(start_date=%s) shifted forward to %r so neither game is dropped "
                            "from the grid.",
                            team, slot_id, candidate_rows[0].get("game_id"),
                            later_row.get("game_id"), later_row.get("start_date"), candidate_slot,
                        )
                        result[team][candidate_slot] = later_row
                        occupied.add(candidate_slot)
                        placed = True
                        break
                if not placed:
                    logger.error(
                        "schedule.py: team %r game_id=%s could not be placed in any open week "
                        "slot after %r (all later week columns already occupied) -- dropped from "
                        "the grid. Extremely unlikely; investigate this season's data.",
                        team, later_row.get("game_id"), slot_id,
                    )
    return result


# ---------------------------------------------------------------------------
# Canonical column list
# ---------------------------------------------------------------------------
def build_canonical_columns(
    rows: List[Dict[str, Any]],
    season: int,
    champ_game_ids: Optional[set] = None,
    army_navy_game_id: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Every DISTINCT regular-season display-week-slot present in this season's
    data (derived, not hardcoded), then Conference Championship, then
    Army-Navy Game Week, then the four fixed CFP slots. Every team gets this
    SAME list.

    Rows that _classify_row_slot() DIVERTS into a dedicated slot -- the Army-Navy
    game, and any identified conference-championship game -- must not also mint a
    week-N column. Otherwise a bucket whose only occupant is diverted leaves behind
    a column no team can ever fill: the empty "Week 15" seen in the 2026 artifact,
    where the Army-Navy game is the only game in bucket 15.

    champ_game_ids/army_navy_game_id default to "nothing is diverted" so the
    function stays callable standalone, but build_schedule_payload always passes
    the real values -- identification must run BEFORE column derivation.
    """
    champ_game_ids = champ_game_ids or set()

    weeks = set()
    champ_buckets = set()
    army_navy_bucket: Optional[int] = None
    for row in rows:
        if row.get("season") != season:
            continue
        if row.get("season_type") != "regular":
            continue
        if row.get("start_date") is None:
            continue
        bucket = _get_week_slot(row)
        game_id = row.get("game_id")
        if army_navy_game_id is not None and game_id == army_navy_game_id:
            army_navy_bucket = bucket
            continue
        if game_id in champ_game_ids:
            champ_buckets.add(bucket)
            continue
        weeks.add(bucket)

    week_columns = [(f"week-{n}", f"Week {n}") for n in sorted(weeks)]

    # Derived postseason labels -- see the CONF_CHAMPIONSHIP_SLOT_ID comment above for
    # why these are not constants. When no championship game is identified (an
    # in-progress season), fall forward from the last regular week instead.
    if champ_buckets:
        ccg_week = max(champ_buckets)
    elif weeks:
        ccg_week = max(weeks) + 1
    else:
        ccg_week = None

    if army_navy_bucket is not None:
        army_navy_week = army_navy_bucket
    elif ccg_week is not None:
        army_navy_week = ccg_week + 1
    else:
        army_navy_week = None

    # The two derived numbers can legitimately collide -- an Army-Navy game played on
    # championship weekend, or a season with no championship game where ccg_week is
    # inferred as max(weeks)+1. Suffix BOTH when they do, so the header never shows two
    # columns reading the same thing.
    labels_collide = (
        ccg_week is not None and army_navy_week is not None and ccg_week == army_navy_week
    )

    def _postseason_label(week_n: Optional[int], suffix: str, fallback: str, always_suffix: bool) -> str:
        if week_n is None:
            return fallback
        # Suffix when asked for it, and always when a real week column already carries
        # this number -- two columns reading "Week 15" would be worse than a long label.
        if always_suffix or week_n in weeks:
            return f"Week {week_n} ({suffix})"
        return f"Week {week_n}"

    ccg_label = _postseason_label(ccg_week, "CCG", CONF_CHAMPIONSHIP_SLOT[1], always_suffix=True)
    army_navy_label = _postseason_label(
        army_navy_week, "Army-Navy", ARMY_NAVY_SLOT[1], always_suffix=labels_collide
    )

    postseason_columns = [(CONF_CHAMPIONSHIP_SLOT_ID, ccg_label)]

    # OMIT the Army-Navy column entirely when this season has no Army-Navy game.
    # Nothing but that one game is ever diverted into this slot (_classify_row_slot's
    # army_navy_game_id branch is the only writer), so with no game to divert the column
    # is one no team can ever fill -- exactly the failure this function's docstring
    # describes for week columns, reached from the other direction. 2025 is the live case:
    # its regular season ends 2025-12-07 and its postseason opens 2025-12-14, so the real
    # 2025-12-13 Army-Navy game is simply absent from the data. Before this guard, that
    # season published a completely empty column (labelled "Week 17" when no championship
    # game was identified either, "Week 16" once they were) -- a user-reported defect.
    #
    # SELF-HEALING, and deliberately not gated on the season: this is keyed on the game's
    # absence, not on a hardcoded season number, so backfilling the missing 2025 row makes
    # the column reappear with no code change. An in-progress season is unaffected --
    # identify_army_navy_game does not filter on status, so a SCHEDULED Army-Navy game
    # still yields a game_id and still mints the column.
    if army_navy_game_id is not None:
        postseason_columns.append((ARMY_NAVY_SLOT_ID, army_navy_label))

    return week_columns + postseason_columns + CFP_SLOTS


# ---------------------------------------------------------------------------
# Cell construction
# ---------------------------------------------------------------------------
def _cell_from_row(row: Dict[str, Any], logos_by_team: Dict[str, Any]) -> Dict[str, Any]:
    opponent = row.get("opponent")
    opponent_logo = _resolve_logo(logos_by_team.get(opponent)) if opponent else None
    status = row.get("status")
    team_score = row.get("team_score")
    opp_score = row.get("opp_score")
    if status == "upcoming":
        # schedule_grid's own migration note: unplayed games are stored with score=0/0, not NULL --
        # passing that through as a literal "0-0" would misleadingly look like a real final score
        # for a game that hasn't happened yet, so scores are nulled out for 'upcoming' specifically.
        team_score = None
        opp_score = None
    else:
        team_score = int(team_score) if team_score is not None else None
        opp_score = int(opp_score) if opp_score is not None else None
    game_name = _game_name_for_row(row)
    return {
        "season_type": row.get("season_type"),
        "opponent": opponent,
        "opponent_logo_url": opponent_logo,
        "conditional_opponent": None,  # only ever set on a computed 'possible' placeholder cell
        "game_name": game_name,
        # K6: one-line form of game_name via artifacts/bowl_names.py -- null wherever game_name
        # is null (guarded here rather than inside short_bowl_name, which treats a null input as
        # a pass-through the same way, but this keeps the None-ness decision co-located with
        # every other field on this cell).
        "game_name_short": short_bowl_name(game_name) if game_name is not None else None,
        "home_away": row.get("home_away"),
        "neutral_site": bool(row.get("neutral_site")) if row.get("neutral_site") is not None else False,
        "status": status,
        "team_score": team_score,
        "opp_score": opp_score,
        # K5: non-null only for a real CFP-round game -- see _playoff_round_for_row.
        "playoff_round": _playoff_round_for_row(row),
        # cfp_seed: null on every real-row cell -- this field exists only to carry a CFP bye
        # team's seed (see _cfp_bye_cell); a real game's own seed is not surfaced here today
        # (out of scope -- no acceptance criterion asks for it on a played/scheduled game).
        "cfp_seed": None,
    }


def _placeholder_cell(logical_season_type: str, status: str, conditional_opponent: Optional[str]) -> Dict[str, Any]:
    """A T4a-computed status (possible/eliminated/clinched/eligible/ineligible) -- no real row yet."""
    return {
        "season_type": logical_season_type,
        "opponent": None,
        "opponent_logo_url": None,
        "conditional_opponent": conditional_opponent,
        "game_name": None,
        "game_name_short": None,
        "home_away": None,
        "neutral_site": False,
        "status": status,
        "team_score": None,
        "opp_score": None,
        "playoff_round": None,
        "cfp_seed": None,
    }


def _bye_cell(logical_season_type: str) -> Dict[str, Any]:
    return {
        "season_type": logical_season_type,
        "opponent": None,
        "opponent_logo_url": None,
        "conditional_opponent": None,
        "game_name": None,
        "game_name_short": None,
        "home_away": None,
        "neutral_site": False,
        "status": "bye",
        "team_score": None,
        "opp_score": None,
        "playoff_round": None,
        "cfp_seed": None,
    }


def _tbd_cell(logical_season_type: str) -> Dict[str, Any]:
    """A FLEX_WEEK_TBD_CONFIG-driven placeholder (T6): a real game is expected in this slot but
    not yet announced/ingested. Reuses the existing `tbd` status (opponent=null, no separate flag
    needed -- same contract as a real scheduled-but-undetermined-opponent game, per plan.yaml)."""
    return {
        "season_type": logical_season_type,
        "opponent": None,
        "opponent_logo_url": None,
        "conditional_opponent": None,
        "game_name": None,
        "game_name_short": None,
        "home_away": None,
        "neutral_site": False,
        "status": "tbd",
        "team_score": None,
        "opp_score": None,
        "playoff_round": None,
        "cfp_seed": None,
    }


def _team_seed_from_row(row: Dict[str, Any]) -> Optional[int]:
    """This team's CFP seed, from a schedule_grid row it appears in (its cfp-quarterfinals row,
    for the bye case).

    Reads schedule_grid's team_seed directly rather than picking between
    playoff_home_seed/playoff_away_seed on home_away. Migration 0006 flips the seed in the view,
    the same way it already flips team_score/opp_score and conference -- re-deriving "which side
    am I?" here would duplicate, in Python, the one thing that view exists to do once.

    K8: home_seed/away_seed's existence on CFBD's GamePlayoff is confirmed only at the schema
    level (read from the OpenAPI-generated client) -- nobody has confirmed they're actually
    POPULATED for real games, and this environment cannot call CFBD to check. So an absent or
    non-numeric value degrades to None here rather than guessing -- the caller (_cfp_bye_cell)
    turns that into a plain label instead of a wrong seed number.
    """
    seed = row.get("team_seed")
    if seed is None:
        return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None


def _cfp_bye_cell(logical_season_type: str, seed: Optional[int]) -> Dict[str, Any]:
    """T4/AC5: a team with a real cfp-quarterfinals row but no real cfp-r1-bowls row earned a
    first-round CFP bye. Distinct from BOTH _bye_cell (no game exists at all that slot) and the
    bowl-eligibility placeholder (which reads as merely "Eligible", indistinguishable from a
    team that never made the playoff).

    REVISED (fix-cycle-1 design correction): the seed is carried as its OWN nullable field,
    cfp_seed -- an integer when resolved, null when not -- rather than pre-composed into
    game_name as "Bye (No. N)" prose. Baking it into game_name conflicted with
    _game_name_for_row's own contract (non-null only for a real, determined game; this is
    neither) and inverted K5's whole argument: K5 exists precisely so the frontend does not have
    to derive structured meaning (a badge) from a label's prose, and writing an integer into a
    sentence two functions before the frontend reads it is that same coupling in the other
    direction. game_name/game_name_short stay null here, exactly like _bye_cell -- the frontend
    composes its own label from status == CFP_BYE_STATUS plus cfp_seed (present or absent), the
    same K8 degrade (a resolved seed renders it, an absent one falls back to a plain "CFP Bye"
    label) now expressed as a data question instead of a string-presence question.
    """
    return {
        "season_type": logical_season_type,
        "opponent": None,
        "opponent_logo_url": None,
        "conditional_opponent": None,
        "game_name": None,
        "game_name_short": None,
        "home_away": None,
        "neutral_site": False,
        "status": CFP_BYE_STATUS,
        "team_score": None,
        "opp_score": None,
        "playoff_round": None,
        "cfp_seed": seed,
    }


def _build_team_weeks(
    canonical_columns: List[Tuple[str, str]],
    team_slot_rows: Dict[str, Dict[str, Any]],
    championship_status: Optional[str],
    conditional_opponent: Optional[str],
    bowl_status: str,
    logos_by_team: Dict[str, Any],
    flex_tbd_slot_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    flex_tbd_slot_ids = flex_tbd_slot_ids or set()
    weeks = []
    for slot_id, label in canonical_columns:
        real_row = team_slot_rows.get(slot_id)
        if real_row is not None:
            cell = _cell_from_row(real_row, logos_by_team)
            weeks.append({"slot_id": slot_id, "label": label, **cell})
            continue

        logical_type = "postseason" if slot_id in _CFP_SLOT_IDS else "regular"

        if slot_id == CONF_CHAMPIONSHIP_SLOT[0] and championship_status is not None:
            weeks.append({"slot_id": slot_id, "label": label, **_placeholder_cell(logical_type, championship_status, conditional_opponent)})
            continue
        if slot_id == CFP_SLOTS[0][0]:
            # T4/AC5: the only available bye signal is structural -- a real row in
            # cfp-quarterfinals (checked here) with no real row in cfp-r1-bowls (already true,
            # or this branch would never have been reached: the `real_row is not None` check
            # above always wins once a real cfp-r1-bowls row exists) means this team earned a
            # first-round bye, not that it merely "made a bowl."
            quarterfinal_row = team_slot_rows.get(CFP_SLOTS[1][0])
            if quarterfinal_row is not None:
                seed = _team_seed_from_row(quarterfinal_row)
                weeks.append({"slot_id": slot_id, "label": label, **_cfp_bye_cell(logical_type, seed)})
                continue
            # Bowl eligibility (possible/eligible/ineligible) is computed unconditionally for
            # every team, so absent a detected bye above, this slot NEVER falls through to a
            # plain 'bye' -- unlike Conference Championship/Army-Navy/the other 3 CFP slots,
            # which do.
            weeks.append({"slot_id": slot_id, "label": label, **_placeholder_cell(logical_type, bowl_status, None)})
            continue
        if slot_id in flex_tbd_slot_ids:
            # T6: FLEX_WEEK_TBD_CONFIG-driven -- a real game is expected here but not yet
            # announced/ingested. Only reached when team_slot_rows has no real row for this
            # slot (the `real_row is not None` branch above always wins once one exists), so
            # this self-clears the moment a real flex game is ingested for this team.
            weeks.append({"slot_id": slot_id, "label": label, **_tbd_cell(logical_type)})
            continue

        weeks.append({"slot_id": slot_id, "label": label, **_bye_cell(logical_type)})
    return weeks


# ---------------------------------------------------------------------------
# Within-conference sort (approximation, documented limitation -- NOT real
# tiebreaker bylaws, per plan.yaml key_decisions):
#   _tier desc (2 = conference-championship winner, 1 = CCG loser, 0 = everyone
#   else -- K5, see _tier in _sort_conference_teams and _is_champion/
#   _is_ccg_loser in build_schedule_payload) -> conf win% desc -> head-to-head
#   IF exactly two teams tied on conf win% AND on _tier (K4/K5 -- see the
#   grouping loop below for why a champion or a CCG loser must never be
#   grouped with a team of a different tier) and played each other this
#   season -> placement win% desc (K6: EXCLUDES postseason rows and the
#   identified championship game itself, so bowl/playoff results and the
#   CCG's own result never affect placement) -> model rank asc, with
#   unranked teams sorting last (K7) -> name asc.
# Independents: placement win% desc -> model rank asc -> name asc (no
# conference tiebreak question for them).
#
# T2/K7 ordering note: model rank is compared AFTER placement win%, so a
# real difference in record always outranks a difference in rating. This was
# decided explicitly by the user, and the two orderings are NOT equivalent --
# 2025 SEC is a live example. Oklahoma and Vanderbilt both sit at 6-2
# conference / 10-2 placement, and Texas at 6-2 conference / 9-3 placement,
# with ranks 10, 14 and 11 respectively. Record-first gives Oklahoma,
# Vanderbilt, Texas; rank-first would give Oklahoma, Texas, Vanderbilt,
# promoting a 9-3 team over a 10-2 one on rating alone. Record-first is what
# conventional standings do, and it matches the scope of the rank tiebreak as
# it was actually chosen: rank was picked to settle teams tied on BOTH
# conference and regular-season record (2025's Ole Miss and Texas A&M, both
# 7-1 / 11-1, who never played each other), not to override a record gap.
# Rank still decides that exhausted case, which is the whole reason it is in
# the key -- it just no longer reaches past a record difference to do it.
# ---------------------------------------------------------------------------
def _head_to_head_winner(rows: List[Dict[str, Any]], season: int, team_a: str, team_b: str) -> Optional[str]:
    """
    T2/K8: tallies every conference_game meeting between team_a and team_b this season and
    returns whichever team has strictly more wins in the series, or None on any tie -- including
    0-0 (no meeting at all) and, per the user's rule, a genuine split like 1-1 or 2-2 (a split
    series is a wash, not a tiebreak).

    THE TRAP THIS GUARDS AGAINST: every played game contributes TWO team-oriented rows sharing
    one game_id (team_a's row and team_b's row, opposite `status` values). Tallying rows from
    BOTH perspectives would therefore read a single meeting as a 2-0 sweep. Filtering to ONLY
    team_a's own-perspective rows (team == team_a and opponent == team_b) below sidesteps this
    entirely -- team_b's row of the same game_id carries no information team_a's row doesn't
    already have (a win for one side is a loss for the other), so it is simply never consulted.
    """
    wins_a = wins_b = 0
    for row in rows:
        if row.get("season") != season:
            continue
        if not row.get("conference_game"):
            continue
        if row.get("status") not in ("win", "loss"):
            continue
        if row.get("team") != team_a or row.get("opponent") != team_b:
            continue  # K8: team_a's own-perspective rows only -- see docstring.
        if row["status"] == "win":
            wins_a += 1
        else:
            wins_b += 1
    if wins_a > wins_b:
        return team_a
    if wins_b > wins_a:
        return team_b
    return None


# T2/K7: a large integer sentinel for a null `rank` -- an unranked team must sort LAST among
# otherwise-tied teams, not first, so it needs a value bigger than any real rank (ranks are
# small positive integers, at most a few hundred FBS+FCS teams). Not float('inf'): the sort key
# tuple mixes this with plain ints/floats and an explicit large int keeps the key trivially
# JSON/repr-friendly for debugging, with no behavioral difference from inf here.
_UNRANKED_SORT_SENTINEL = 10**9


def _placement_pct(entry: Dict[str, Any], rows: List[Dict[str, Any]], season: int, champ_game_ids: set) -> float:
    """
    T2/K6: this team's win percentage for STANDINGS PLACEMENT only -- excludes any postseason
    (season_type=='postseason') row AND any row whose game_id is one of the identified
    conference-championship games, per the user's rule that "bowl/playoff results must not
    affect standings placement." Does NOT touch the DISPLAYED record (entry["record"]) -- that
    keeps coming from schedule_standings.compute_team_records, untouched by this function.

    Implemented as a SUBTRACTION from entry["record"] (which already counts every row for this
    team this season, per compute_team_records) rather than an independent tally over `rows`,
    deliberately: the two are mathematically identical whenever `rows` actually contains this
    team's games (always true in the real pipeline, and in every fixture built from
    build_schedule_payload), but the subtraction degrades gracefully -- rather than exploding to
    the 0.5 "no games" sentinel -- for a hand-built entry passed with an empty/unrelated `rows`
    list and a pre-set "record" (every fixture in tests/test_conference_sort_and_pac12.py): with
    nothing to subtract, it reproduces entry["record"]'s own percentage exactly, so none of
    those tests needed to change for this task.

    Same 0.5 sentinel as the original _overall_pct for a team with no counted games after
    exclusion.
    """
    team = entry["team"]
    postseason_wins = postseason_losses = 0
    champ_wins = champ_losses = 0
    for row in rows:
        if row.get("season") != season or row.get("team") != team:
            continue
        status = row.get("status")
        if status not in ("win", "loss"):
            continue
        if row.get("season_type") == "postseason":
            if status == "win":
                postseason_wins += 1
            else:
                postseason_losses += 1
        elif row.get("game_id") in champ_game_ids:
            # A championship game is always season_type=='regular' (see the module comment on
            # identify_conference_championship_games), so this elif never double-subtracts a row
            # already counted above.
            if status == "win":
                champ_wins += 1
            else:
                champ_losses += 1

    # Clamped at 0. The subtraction assumes entry["record"] already counts every row this
    # loop can find, which compute_team_records guarantees in the real pipeline (it tallies
    # every win/loss row for the team regardless of season_type or conference_game, from this
    # same `rows` list). A hand-built entry whose "record" disagrees with `rows` breaks that
    # assumption and can drive either term negative -- record=1-3 against two postseason win
    # rows yields -1 wins and a pct of -0.5, which would sort BELOW a genuine 0.000 team
    # rather than above it. Unreachable from the pipeline, cheap to foreclose, and a silently
    # wrong ORDER is exactly the failure class this function was added to fix.
    w = max(0, entry["record"]["wins"] - postseason_wins - champ_wins)
    l = max(0, entry["record"]["losses"] - postseason_losses - champ_losses)
    return (w / (w + l)) if (w + l) > 0 else 0.5


def _non_fbs_roster(non_fbs_logos: Optional[Dict[str, Any]]) -> Optional[frozenset]:
    """The FCS/lower-division roster the Big 12's total-wins step needs, derived from the table
    already read for logos. database/get_non_fbs_teams.py filters on CFBD's own `classification`
    field, so membership here is a positive assertion that a school is not FBS -- which is why it
    is sound where "this name is absent from teams.school" is not (see
    database/migrations/0003_schedule_grid_view.sql, which investigated and rejected that proxy).

    An EMPTY mapping returns None, not frozenset(), and the difference is load-bearing:
    total_wins_capped reads None as "roster unavailable, decline the step" and an empty set as
    "roster loaded, nobody qualifies". An empty mapping at this point is far more likely to mean
    the non_fbs_teams read failed -- _fetch_non_fbs_logos handles that non-fatally by substituting
    {} -- than to mean that no FBS team played a non-FBS opponent all season, which essentially
    never happens. Declining is the safe reading of that ambiguity: it costs one step in one
    conference's chain, where the alternative silently reports uncapped win totals as capped.
    """
    return frozenset(non_fbs_logos) if non_fbs_logos else None


@dataclass(frozen=True)
class _TiebreakInputs:
    """Everything the tiebreaker engine needs that is scoped to the WHOLE conference, built once
    per conference and passed down into each divisional `_sort_conference_teams` call.

    Why whole-conference rather than per-group: two primitives reach outside the tied group.
    `opponents_cumulative_conf_pct` looks up each tied team's OPPONENTS' conference records, and
    `vs_placed_opponents` walks the conference's order of finish. Handing either only the tied
    teams' own records would silently score every outside opponent as having no record at all.

    `frozen_order` is the conference ordered by conference win percentage ALONE, computed before
    any tiebreaker step runs and never updated mid-resolution (plan K4). "Record against the
    next-highest-placed team" is circular otherwise.

    `rules` is None for a conference with no configured procedure. All ten FBS conferences now
    have one, so in practice this means a season outside a conference's configured era -- the
    2014-2022 divisional era for most of them -- or FBS Independents. The caller keeps its
    pre-engine ordering in that case rather than having a procedure invented for it.

    `divisions` maps every member to its division, and matters for exactly one conference: the
    Sun Belt is the only one of the ten that still plays them, and three of its steps are
    division-scoped (divisional record, common NON-divisional opponents, and a traversal of the
    DIVISIONAL rather than conference standings). Built from the whole conference for the same
    reason as the other two fields -- those steps ask about opponents outside the tied group.
    """

    conference: str
    rules: Optional[RuleSet]
    conf_records: Dict[str, Tuple[int, int]]
    frozen_order: List[str]
    divisions: Dict[str, Optional[str]]
    non_fbs_teams: Optional[frozenset]


_TIEBREAKER_CONFIG = None
_TIEBREAKER_CONFIG_FAILED = False


def _tiebreaker_config():
    """The parsed rule config, loaded once per process.

    A config that fails to load is reported once and then treated as "no rules for any
    conference", which degrades every conference to its pre-engine ordering. That is deliberate:
    a malformed rule file must not take down the artifact publish, and the pre-engine ordering is
    a known-good behaviour rather than a guess.
    """
    global _TIEBREAKER_CONFIG, _TIEBREAKER_CONFIG_FAILED
    if _TIEBREAKER_CONFIG is not None or _TIEBREAKER_CONFIG_FAILED:
        return _TIEBREAKER_CONFIG
    try:
        _TIEBREAKER_CONFIG = load_conference_rules()
    except (TiebreakerConfigError, OSError) as exc:
        _TIEBREAKER_CONFIG_FAILED = True
        logger.error(
            "schedule.py: could not load the conference tiebreaker config (%s); every "
            "conference will fall back to the pre-engine standings ordering.", exc,
        )
    return _TIEBREAKER_CONFIG


def _build_tiebreak_inputs(
    raw_conference: str,
    entries: List[Dict[str, Any]],
    season: int,
    non_fbs_teams: Optional[frozenset],
) -> _TiebreakInputs:
    """Assemble the whole-conference inputs from this conference's full member list."""
    conf_records: Dict[str, Tuple[int, int]] = {}
    for entry in entries:
        record = entry.get("conf_record")
        if record is not None:
            conf_records[entry["team"]] = (record["wins"], record["losses"])

    def _pct_and_played(team: str) -> Tuple[float, int]:
        wins, losses = conf_records.get(team, (0, 0))
        played = wins + losses
        # The 0.5 sentinel for an unplayed record matches _sort_conference_teams, but on its own
        # it would seat a team that has played NO conference games in mid-table -- ahead of every
        # sub-.500 team -- in the order vs_placed_opponents walks. The main sort key guards that
        # with a `-_conf_played` term and this must too, or "the next highest-placed team in the
        # standings" means something different here than it does in the standings themselves.
        return ((wins / played) if played else 0.5, 1 if played else 0)

    # Name is the final term purely for determinism -- frozen_order must not vary between runs
    # over identical data, or `vs_placed_opponents` becomes non-reproducible.
    frozen_order = sorted(
        (entry["team"] for entry in entries),
        key=lambda t: (-_pct_and_played(t)[0], -_pct_and_played(t)[1], t),
    )

    # Present for every member, including a None division for a conference that plays none, so
    # the division-scoped steps can tell "no divisions here" from "this team is missing".
    divisions: Dict[str, Optional[str]] = {
        entry["team"]: entry.get("division") for entry in entries
    }

    config = _tiebreaker_config()
    rules = rules_for(config, raw_conference, season) if config is not None else None
    if rules is not None:
        _warn_unhonoured_policies(raw_conference, season, rules)
    if rules is None:
        logger.info(
            "schedule.py: no tiebreaker rule set for conference=%r season=%s; using the "
            "pre-engine standings ordering for it.", raw_conference, season,
        )
    return _TiebreakInputs(
        conference=raw_conference,
        rules=rules,
        conf_records=conf_records,
        frozen_order=frozen_order,
        divisions=divisions,
        non_fbs_teams=non_fbs_teams,
    )


_UNHONOURED_POLICY_WARNED: set = set()


def _warn_unhonoured_policies(raw_conference: str, season: int, rules: RuleSet) -> None:
    """Say once, per conference and season, when a rule set declares something this caller does
    not implement.

    Two such fields exist, and both are grouping rules rather than steps: `tie_definition` other
    than plain win-percentage equality, and `restart_at: "redefine_tied_teams"`. The ACC defines
    its tied set to include teams on an alternate number of conference games with the same wins
    OR the same losses, and CUSA to include teams within one conference win with equal losses --
    neither of which this function's caller builds, since it groups on conference win percentage
    alone. The engine's own docstring is honest about not implementing them, but nothing in a
    running pipeline said so, and today falls inside the ACC's 2026 era, which is exactly the
    entry whose grouping rule is unimplemented.

    A warning rather than an error: the configured STEPS are still applied correctly to whatever
    group it is handed, so the result is a good answer to a slightly narrower question, not a
    wrong one.
    """
    key = (raw_conference, season)
    if key in _UNHONOURED_POLICY_WARNED:
        return
    unhonoured = []
    if rules.tie_definition != "win_pct":
        unhonoured.append(f"tie_definition={rules.tie_definition!r}")
    if rules.multi_team.restart_at == "redefine_tied_teams":
        unhonoured.append("restart_at='redefine_tied_teams'")
    if unhonoured:
        _UNHONOURED_POLICY_WARNED.add(key)
        logger.warning(
            "schedule.py: conference=%r season=%s declares %s, which this caller does not "
            "implement -- tied groups are still built on conference win percentage alone. The "
            "configured tiebreaker STEPS are applied normally; only the definition of who counts "
            "as tied is narrower than the conference's own.",
            raw_conference, season, " and ".join(unhonoured),
        )


def _engine_order_group(
    group: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    season: int,
    champ_game_ids: set,
    tiebreak: _TiebreakInputs,
) -> List[Dict[str, Any]]:
    """Order one tied group through the conference's configured procedure.

    Records the deciding step on each entry as `resolved_by`, so the published payload can say
    WHY a team sits where it does -- and so a test can assert that the right step decided rather
    than only that the order came out right.
    """
    by_team = {entry["team"]: entry for entry in group}
    ctx = TiebreakContext(
        rows=rows,
        season=season,
        conference=tiebreak.conference,
        frozen_order=tiebreak.frozen_order,
        conf_records=tiebreak.conf_records,
        team_ranks={entry["team"]: entry.get("rank") for entry in group},
        placement_excluded_game_ids=frozenset(champ_game_ids),
        divisions=tiebreak.divisions,
        non_fbs_teams=tiebreak.non_fbs_teams,
    )
    outcome = order_tied_group([entry["team"] for entry in group], ctx, tiebreak.rules)
    for team, step in outcome.resolved_by.items():
        if team in by_team:
            by_team[team]["resolved_by"] = step
    return [by_team[team] for team in outcome.flat]


def _sort_conference_teams(
    entries: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    season: int,
    champ_game_ids: Optional[set] = None,
    tiebreak: Optional[_TiebreakInputs] = None,
) -> List[Dict[str, Any]]:
    """
    champ_game_ids (T2/K6): the season's identified conference-championship game_ids, used only
    by _placement_pct's exclusion. Defaults to "exclude nothing" so every hand-built entry in
    tests/test_conference_sort_and_pac12.py -- which calls this function positionally with just
    (entries, rows, season) -- keeps sorting exactly as before.

    tiebreak (T4): the conference's configured tiebreaker procedure plus the whole-conference
    inputs it needs. Defaults to None, which keeps the PRE-ENGINE ordering exactly: placement
    percentage, then model rank, then name, with a head-to-head swap for a group of exactly two.
    That default is what every existing hand-built-entry test exercises, and it is also the live
    path for the six conferences whose published rules nobody has supplied yet -- so merging the
    engine changes nothing for them until their rules arrive (plan R4/AC7).
    """
    champ_game_ids = champ_game_ids or set()
    for e in entries:
        # T4: which step of the conference's procedure fixed this team's position. Stays None for
        # a team the conference win percentage separated on its own -- the common case -- and for
        # every team in a conference with no configured rules. Unlike the underscore-prefixed
        # scratch fields below, this one is NOT stripped: it is published.
        #
        # Assigned, not setdefault: this function is idempotent over the same entry dicts, and a
        # setdefault would carry a label from a previous call into a sort that no longer reaches
        # that step -- publishing a reason the current standings were not decided by.
        e["resolved_by"] = None
        e["_placement_pct"] = _placement_pct(e, rows, season, champ_game_ids)
        # Whether any conference game has been played, used only as a sort tiebreak below.
        e["_conf_played"] = bool(e["conf_record"] and (e["conf_record"]["wins"] + e["conf_record"]["losses"]) > 0)
        if e["conf_record"] is not None:
            cw, cl = e["conf_record"]["wins"], e["conf_record"]["losses"]
            # Use 0.5 as sentinel for unplayed conference record (neutral between win and loss),
            # matching the placement-pct logic. This fixes the NC State vs Duke case where
            # a team with 0-1 conference record should sort below a team with 0-0.
            e["_conf_pct"] = (cw / (cw + cl)) if (cw + cl) > 0 else 0.5
        else:
            e["_conf_pct"] = None
        # T2/K5: 2 = conference-championship winner, 1 = CCG loser, 0 = everyone else. Derived
        # from the two separate entry flags build_schedule_payload sets (both default False via
        # .get, so hand-built entries in existing tests that set only "_is_champion", or neither
        # flag at all, keep sorting exactly as before -- a champion still gets _tier 2, and a
        # plain entry still gets _tier 0).
        if e.get("_is_champion", False):
            e["_tier"] = 2
        elif e.get("_is_ccg_loser", False):
            e["_tier"] = 1
        else:
            e["_tier"] = 0

    def _rank_sort_key(e: Dict[str, Any]) -> int:
        # T2/K7: the team's own current model rank, ascending, with unranked teams (rank is
        # None whenever a season has no ratings rows for this team -- see build_schedule_payload's
        # docstring for `team_ranks`) sorting LAST via the sentinel rather than first (which a
        # bare `None` would do, since Python can't compare None to an int at all).
        # ASSUMPTION A2 (user-chosen, see plan.yaml key_decisions K7): valid only because the
        # ratings this rank is drawn from stop at week 15 for the 2025 season -- no postseason
        # week is ever rated -- so this rank reflects the regular season (bowls/playoffs
        # excluded) exactly like _placement_pct does. This would silently start incorporating
        # postseason results if a future pipeline run ever rated a postseason week; nothing here
        # guards against that (there is no season_type on a ratings-table row to check).
        rank = e.get("rank")
        return rank if rank is not None else _UNRANKED_SORT_SENTINEL

    has_conf_records = any(e["_conf_pct"] is not None for e in entries)
    if has_conf_records:
        # The None branch here is DEFENSIVE AND UNREACHABLE in practice, not a live rule.
        # conf_record is None only for Independents (schedule_standings sets conf_wins to
        # None for them and only for them), and this function is called once per conference,
        # so a single call sees either all-Independents -- in which case has_conf_records is
        # False and we take the else branch below -- or no Independents at all. The two
        # groups are never sorted against each other, so the fallback's value cannot affect
        # any real ordering. Left as -1.0 rather than 0.5 to keep it obviously a sentinel.
        # `-e["_conf_played"]` places a team that has actually played conference games above an
        # unplayed one at the SAME percentage. That is not cosmetic: the head-to-head tiebreak
        # below only fires on a group of exactly two, and giving an unplayed record 0.5 makes it
        # tie with every 1-1, 2-2, 3-3 team. Without this term a single 0-0 team joining two
        # tied teams grows the group to three and silently disables their head-to-head swap,
        # displaying the loser of that game above the winner. Ordering among played teams is
        # unchanged, since _conf_played is True for all of them.
        # T2/K4/K5: tier (default 0, so hand-built entries in existing tests that never set
        # _is_champion/_is_ccg_loser keep sorting exactly as before) is the TOP sort key -- a
        # conference-championship-game winner sorts first, its loser second, regardless of
        # conf_pct. Placement pct precedes model rank -- see the long comment above this
        # function for why, with the concrete 2025 Oklahoma/Vanderbilt/Texas example.
        entries.sort(key=lambda e: (-e["_tier"],
                                    -(e["_conf_pct"] if e["_conf_pct"] is not None else -1.0),
                                    -e["_conf_played"], -e["_placement_pct"], _rank_sort_key(e), e["team"]))
        i, n = 0, len(entries)
        while i < n:
            j = i
            # Group on percentage, whether the team has played, AND tier (T2/K4/K5).
            # Percentage alone is not enough: an unplayed record scores 0.5, which ties it with
            # every 1-1, 2-2 and 3-3 team, so one 0-0 team joining two genuinely tied teams grows
            # the group to three and silently cancels their head-to-head swap -- displaying the
            # loser of that game above the winner. A team that has played nobody cannot be part
            # of a head-to-head tie by definition, so it must never join the group.
            #
            # The _tier term guards a DIFFERENT failure: a champion or a CCG loser can easily tie
            # a team of a different tier on conf_pct (the championship game itself isn't the
            # only thing that separates them -- concretely, 2025 SEC's Alabama, the CCG loser,
            # ties Ole Miss and Texas A&M, both tier 0, at conf .875), and if that other-tier team
            # beat this one earlier in the regular season, the swap below would fire on that
            # meeting and undo the whole tier-first rule from the sort key above it -- silently,
            # since the swap has no notion of tier without this check. Both the champion and the
            # CCG-loser tier are singleton per conference by construction (identify_conference_
            # championship_games identifies at most one game per conference, which has exactly
            # one winner and one loser), so a unique tier-2 or tier-1 entry can therefore never
            # share a group with anyone else -- the swap below never touches it.
            #
            # INTERACTION WITH THE THREE-WAY PROTECTION ABOVE (fix-cycle-1, documented rather
            # than fixed -- the review judged the result arguably correct, just previously
            # unstated and untested): the "group grows to three, swap disabled" protection
            # described two paragraphs up is NOT unconditional once tier is part of the
            # grouping key. Three teams genuinely tied on conf_pct with one of them a champion do
            # NOT form one group of three -- the champion (tier 2) splits off into its own
            # single-team group, leaving the other two (both tier 0) as a group of exactly two,
            # which re-enables their head-to-head swap. Concrete case (see
            # tests/test_conference_championship_records.py's three-way-tie test): A, B, C all
            # 1-1, B beat A head-to-head. With no champion, all three group together and the
            # swap never fires -- name order, ['A', 'B', 'C']. With C the champion, C sorts alone
            # ahead of {A, B}, and THAT pair's now-exposed head-to-head swap fires on its own
            # meeting -- ['C', 'B', 'A']. This is consistent with the rule as specified (a
            # champion's own group can never be swapped; nothing protects the teams IT excludes
            # from the group it no longer joins), but is a real, non-obvious consequence of
            # narrowing the grouping key, not a re-derivation of the three-way protection itself.
            while j + 1 < n and entries[j + 1]["_conf_pct"] is not None and entries[i]["_conf_pct"] is not None \
                    and abs(entries[j + 1]["_conf_pct"] - entries[i]["_conf_pct"]) < 1e-9 \
                    and entries[j + 1]["_conf_played"] == entries[i]["_conf_played"] \
                    and entries[j + 1]["_tier"] == entries[i]["_tier"]:
                j += 1
            group = entries[i:j + 1]
            # A group is a genuine tie only if its members have actually played conference games:
            # an unplayed 0-0 record scores the 0.5 sentinel, which ties it with every 1-1 and
            # 2-2 team without either having any bearing on the other.
            is_real_tie = (
                len(group) >= 2
                and group[0]["_conf_pct"] is not None
                and group[0]["_conf_played"]
            )
            if is_real_tie and tiebreak is not None and tiebreak.rules is not None:
                # The conference's own published procedure, of any group size. This supersedes
                # the two-team head-to-head swap below, which was only ever the first step of
                # every one of those procedures applied to the one group size it could handle.
                entries[i:j + 1] = _engine_order_group(
                    group, rows, season, champ_game_ids, tiebreak
                )
            elif is_real_tie and len(group) == 2:
                # Pre-engine behaviour, retained verbatim for a conference with no configured
                # rules. Deliberately NOT extended to larger groups here: guessing at a
                # multi-team procedure is what this whole feature exists to stop doing.
                t1, t2 = group[0]["team"], group[1]["team"]
                winner = _head_to_head_winner(rows, season, t1, t2)
                if winner is not None:
                    # Recorded whichever way it fell: head-to-head fixed both positions just as
                    # much when the winner was already first as when they had to be swapped.
                    # Setting it only inside the swap published a null for half the cases.
                    group[0]["resolved_by"] = group[1]["resolved_by"] = "head_to_head"
                if winner == t2:
                    entries[i], entries[i + 1] = entries[i + 1], entries[i]
            i = j + 1
    else:
        entries.sort(key=lambda e: (-e["_placement_pct"], _rank_sort_key(e), e["team"]))

    for e in entries:
        del e["_placement_pct"]
        del e["_conf_pct"]
        del e["_conf_played"]
        del e["_tier"]
        # K10: scratch state, stripped before these entries reach the published payload --
        # pop (not del) because, unlike the four fields above (which this function itself always
        # sets), _is_champion/_is_ccg_loser are set by build_schedule_payload, not here, so
        # hand-built entries in existing tests that never set them at all must not raise a
        # KeyError here.
        e.pop("_is_champion", None)
        e.pop("_is_ccg_loser", None)
    return entries


# ---------------------------------------------------------------------------
# T1/K1 -- post-hoc conf_record exclusion (see the long comment inside the
# function body for why this cannot live inside compute_team_records's tally)
# ---------------------------------------------------------------------------
def _exclude_championship_games_from_conf_records(
    standings: Dict[str, Dict[str, Any]],
    rows: List[Dict[str, Any]],
    season: int,
    champ_game_ids: set,
) -> None:
    """
    Subtract the identified conference-championship game from each participant's DISPLAYED
    conf_record, in place, on the `standings` dict compute_standings() already returned.

    THIS CANNOT LIVE INSIDE compute_team_records'S TALLY INSTEAD -- the naive version (exclude
    the championship game_ids from the conf_rows tally itself) looks obviously correct, but
    compute_standings is the only caller of compute_team_records, and it feeds that SAME records
    dict straight into compute_conference_championship_status, which reads conf_wins and
    conf_games_remaining to decide clinched/eliminated/possible. identify_conference_championship_
    games does not gate on the game's status -- it happily identifies a scheduled-but-unplayed
    title game too. So stripping that row from the tally would also shrink conf_games_remaining
    (R_T) for BOTH participants in the week before kickoff. Since eliminated = B_T <
    nth_highest_other_W and B_T = W_T + R_T, a smaller R_T makes elimination MORE likely -- the two
    teams about to play for the conference title would be marked "Eliminated" during championship
    week, every season. compute_conference_championship_status's own docstring says it must never
    assert "eliminated" incorrectly. So the status math (compute_standings, called by this
    function's caller BEFORE this runs) always sees unmodified rows, and only the separate,
    already-built conf_record dict that gets DISPLAYED is adjusted here, afterward.

    Only rows with status in ('win', 'loss') are ever subtracted: an unplayed championship game
    was never counted in compute_team_records's tally to begin with (it only counts win/loss
    rows), so there is nothing to remove from the display for it either -- this is what keeps an
    identified-but-unplayed game from changing anything at all (see
    tests/test_conference_championship_records.py's K1 regression test, which pins conf_record
    itself in the unplayed case, not just championship_status -- fix-cycle-1 review found the
    status math had a passing test but the displayed record did not).

    DO NOT DELETE THE conference_game=True CHECK BELOW. It was added in fix-cycle-1 as
    "cheap insurance" against an upstream invariant being loosened, and described then as
    structurally redundant. That description is now INVERTED and the guard is load-bearing:
    CFBD reports a conference championship game as conference_game=FALSE (see the module
    comment on identify_conference_championship_games), so from the 2025 season onward a CCG
    row is never counted into conf_record in the first place -- schedule_standings.
    compute_team_records tallies conference records from conference_game rows only. This
    function subtracting it again would subtract a game that was never added.

    Concretely, with the guard removed, 2025 publishes Georgia at 6-1 and Alabama at 7-0 in
    the SEC; both are really 7-1. Mutation-verified during the PR #16 review.
    """
    if not champ_game_ids:
        return
    for row in rows:
        if row.get("season") != season:
            continue
        if row.get("game_id") not in champ_game_ids:
            continue
        if not row.get("conference_game"):
            continue
        status = row.get("status")
        if status not in ("win", "loss"):
            continue
        st = standings.get(row.get("team"))
        if st is None or st.get("conf_record") is None:
            continue
        key = "wins" if status == "win" else "losses"
        st["conf_record"][key] -= 1


# ---------------------------------------------------------------------------
# T2/K3/K4 -- resolve each identified championship game to its ACTUAL winner and loser
# ---------------------------------------------------------------------------
def _resolve_conference_championship_outcomes(
    champ_games_by_conf: Dict[str, int],
    rows: List[Dict[str, Any]],
    season: int,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns (champions, losers) -- both Dict[raw conference -> team name] -- for conferences
    whose identified championship game (identify_conference_championship_games) has actually
    been PLAYED.

    K3/K4: an identified-but-unplayed game contributes NO entry to EITHER dict -- a team that
    has merely clinched a spot in the title game is neither a champion nor a "CCG loser", so
    that conference's sort order stays untouched (see _sort_conference_teams's _tier default of
    0) until the game resolves.

    The loser is simply the OTHER team-perspective row of the same game_id, the one reading
    status='loss'. The "both sides win" (or "both sides lose") case -- two rows of the same
    game_id both reading the same status -- is unreachable: status is score-derived
    (schedule_grid's CASE expression), so exactly one of a game's two team-perspective rows
    reads 'win' and the other 'loss' once the game is played. Never a source of ambiguity here.
    """
    game_id_to_conf = {game_id: conf for conf, game_id in champ_games_by_conf.items()}
    if len(game_id_to_conf) != len(champ_games_by_conf):
        # "Shouldn't happen" -- champ_games_by_conf is Dict[conference -> game_id], one entry
        # per conference, but this inverts it to Dict[game_id -> conference], which is silently
        # LOSSY if two different conferences were ever identified against the SAME game_id:
        # whichever conference iterates last in champ_games_by_conf.items() wins the inversion,
        # and the other is dropped from champion/loser resolution entirely -- no exception, just
        # a missing champion (and loser) for that conference. Every neighbouring "shouldn't
        # happen" case in this file logs rather than silently proceeding; matching that here.
        collided = [conf for conf, gid in champ_games_by_conf.items() if game_id_to_conf.get(gid) != conf]
        logger.warning(
            "schedule.py: %d conference(s) were identified against a championship game_id shared "
            "with another conference -- %s -- and were dropped from champion resolution by the "
            "many-to-one game_id->conference inversion. Should be structurally impossible (a "
            "conference_game row belongs to exactly one conference).", len(collided), collided,
        )
    champions: Dict[str, str] = {}
    losers: Dict[str, str] = {}
    for row in rows:
        if row.get("season") != season:
            continue
        game_id = row.get("game_id")
        if game_id not in game_id_to_conf:
            continue
        status = row.get("status")
        if status == "win":
            champions[game_id_to_conf[game_id]] = row.get("team")
        elif status == "loss":
            losers[game_id_to_conf[game_id]] = row.get("team")
    return champions, losers


def _resolve_conference_champions(
    champ_games_by_conf: Dict[str, int],
    rows: List[Dict[str, Any]],
    season: int,
) -> Dict[str, str]:
    """
    Dict[raw conference -> winning team name], for conferences whose identified championship
    game has actually been PLAYED.

    T2/K4: a thin wrapper over _resolve_conference_championship_outcomes, kept as its own
    function (rather than inlined at every call site) specifically so
    tests/test_conference_championship_records.py::test_resolve_conference_champions_warns_on_a_
    shared_game_id_collision -- which calls this function directly and asserts a plain dict --
    keeps working unchanged.
    """
    champions, _losers = _resolve_conference_championship_outcomes(champ_games_by_conf, rows, season)
    return champions


# ---------------------------------------------------------------------------
# Top-level payload assembly (pure -- no I/O, fully testable with synthetic
# schedule_grid rows and a synthetic teams_meta dict)
# ---------------------------------------------------------------------------
def build_schedule_payload(
    rows: List[Dict[str, Any]],
    teams_meta: Dict[str, Dict[str, Any]],
    season: int,
    team_ranks: Optional[Dict[str, int]] = None,
    non_fbs_logos: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Args:
        rows: schedule_grid rows as dicts (all season_types) for `season`.
        teams_meta: Dict[school -> {"conference": str|None, "division": str|None, "logos": list|None}]
                    from the `teams` table for `season` -- the FBS team universe. `division` is
                    populated for divisional conferences and null elsewhere. In 2026 the Sun Belt
                    is the only one, but this is NOT a constant: eight conferences carry divisions
                    in `teams` for seasons between 2014 and 2023, so republishing an older season
                    will group those by division too, while the championship maths still treats
                    them as flat top-2. Do not assume "Sun Belt only". It drives both the division grouping below and, injected into
                    compute_standings(), the per-division championship status.
        season: the season to build the artifact for.
        team_ranks: Dict[school -> rank] from _fetch_team_ranks -- this season's current rank
                    (1 = best), derived from the latest week with ratings for `season`. Optional
                    and defaults to empty so this function stays callable with synthetic test
                    input that carries no ratings at all. A team not present in this dict --
                    including every team, when the season has no ratings rows whatsoever -- emits
                    rank: null rather than raising or being silently omitted from the grid.
        non_fbs_logos: Dict[school -> logos list|None] from the `non_fbs_teams` table -- the
                    FCS/DII/DIII opponents that appear on FBS schedules but are deliberately
                    absent from `teams`. Used for OPPONENT LOGOS ONLY and merged into
                    logos_by_team below, never into teams_meta: teams_meta's KEYS define which
                    teams get a Season Grid row and which conference each is grouped under
                    (see fbs_team_names and member_teams below), so a non-FBS entry there would
                    give every FCS opponent its own row in a fabricated conference. Optional and
                    defaults to empty, which reproduces the pre-existing behavior exactly (a
                    non-FBS opponent renders its name as text).
    Returns:
        The full Season Grid JSON payload per plan.yaml's contracts.interfaces.
    """
    team_ranks = team_ranks or {}
    non_fbs_logos = non_fbs_logos or {}
    non_fbs_team_names = _non_fbs_roster(non_fbs_logos)
    # Division is injected into the standings computation rather than looked up there:
    # schedule_standings does no DB I/O and schedule_grid carries no division column, so the
    # `teams`-sourced map has to come from here. It is what lets a divisional conference (the
    # Sun Belt today) get per-division championship statuses instead of a blank column; every
    # other conference's teams map to None and are computed exactly as before.
    divisions = {team: meta.get("division") for team, meta in teams_meta.items()}

    # T1/K2: identification must run BEFORE compute_standings, not after (it did, at the old
    # :914 vs :911) -- so its result is available for the post-hoc conf_record exclusion right
    # below. This reorder is inert on its own: the only statement previously between the two
    # call sites was the unrelated fbs_team_names assignment, simply moved down with it.
    champ_games_by_conf = identify_conference_championship_games(rows, season)
    champ_game_ids = set(champ_games_by_conf.values())

    standings = schedule_standings.compute_standings(rows, season, divisions=divisions)
    # T1/K1: compute_standings above ran on UNMODIFIED rows, so championship_status is safe (see
    # _exclude_championship_games_from_conf_records's docstring). Only the conf_record that gets
    # DISPLAYED is adjusted, here, afterward.
    _exclude_championship_games_from_conf_records(standings, rows, season, champ_game_ids)
    # T2/K3/K4: the identified game's actual winner and loser (absent for a conference with no
    # identified game, or one that hasn't been played yet) -- see
    # _resolve_conference_championship_outcomes.
    conference_champions, conference_ccg_losers = _resolve_conference_championship_outcomes(champ_games_by_conf, rows, season)

    fbs_team_names = set(teams_meta.keys())
    army_navy_game_id = identify_army_navy_game(rows, season)
    canonical_columns = build_canonical_columns(rows, season, champ_game_ids, army_navy_game_id)
    week_slot_ids_sorted = [slot_id for slot_id, _label in canonical_columns if slot_id.startswith("week-")]
    team_slot_rows = _build_team_slot_rows(rows, fbs_team_names, champ_game_ids, army_navy_game_id, week_slot_ids_sorted)
    # Non-FBS entries first so an FBS row always wins on a name collision. The two tables are
    # disjoint by construction (the ingest filters `classification != 'fbs'` before writing
    # non_fbs_teams), so there should be no collision at all -- this ordering just makes the
    # rating model's own team list authoritative if that invariant ever breaks upstream.
    logos_by_team: Dict[str, Any] = dict(non_fbs_logos)
    logos_by_team.update({team: meta.get("logos") for team, meta in teams_meta.items()})

    def team_conference(team: str) -> Optional[str]:
        st = standings.get(team)
        conf = st["conference"] if st else None
        if not conf:
            conf = teams_meta.get(team, {}).get("conference")
        return conf

    def team_division(team: str) -> Optional[str]:
        # Asymmetric with team_conference() above by necessity: standings (derived from
        # schedule_grid rows via schedule_standings.compute_standings) carries no division
        # concept at all -- schedule_grid has no division column, per the task's background --
        # so there is no schedule_grid-derived source to prefer or fall back from. teams_meta
        # (sourced from the `teams` table) is the ONLY source. Populated today only for the Sun
        # Belt (East/West), but eight conferences carry divisions in older seasons -- see
        # build_schedule_payload's docstring. Null for non-divisional conferences and Independents.
        return teams_meta.get(team, {}).get("division")

    conferences_out = []
    for raw_conf in CONFERENCE_ORDER:
        member_teams = [t for t in teams_meta if team_conference(t) == raw_conf]
        if not member_teams:
            continue

        flex_tbd_slot_ids = _flex_week_tbd_slot_ids(season, raw_conf)

        entries = []
        for team in member_teams:
            st = standings.get(team)
            logo = _resolve_logo(teams_meta.get(team, {}).get("logos"))
            if st is not None:
                record = st["record"]
                conf_record = st["conf_record"]
                championship_status = st["championship_status"]
                conditional_opponent = st["conditional_opponent"]
                bowl_status = st["bowl_status"]
            else:
                # Per the handoff, this "should not" happen for a real FBS team, but handled
                # defensively rather than raising -- e.g. a team newly added to `teams` with no
                # schedule_grid rows ingested yet.
                logger.warning(
                    "schedule.py: FBS team %r (season=%s) has zero schedule_grid rows; using a "
                    "default 0-0 record for the grid.", team, season,
                )
                record = {"wins": 0, "losses": 0}
                conf_record = None if raw_conf == schedule_standings.INDEPENDENT_CONFERENCE_VALUE else {"wins": 0, "losses": 0}
                championship_status = None
                conditional_opponent = None
                bowl_status = "possible"

            weeks = _build_team_weeks(
                canonical_columns,
                team_slot_rows.get(team, {}),
                championship_status,
                conditional_opponent,
                bowl_status,
                logos_by_team,
                flex_tbd_slot_ids,
            )
            entries.append({
                "team": team,
                "logo_url": logo,
                "rank": team_ranks.get(team),
                "record": record,
                "conf_record": conf_record,
                "division": team_division(team),
                # T2/K10: scratch state consumed by _sort_conference_teams and stripped there
                # before these entries are returned -- never reaches the published payload.
                "_is_champion": conference_champions.get(raw_conf) == team,
                # T2/K5: the identified championship game's LOSER (absent for a conference with
                # no identified game, or one not yet played) -- sorts second, below the champion
                # but above every other team, regardless of conf_pct. See _tier in
                # _sort_conference_teams.
                "_is_ccg_loser": conference_ccg_losers.get(raw_conf) == team,
                "weeks": weeks,
            })

        # Group by division before sorting, rather than adding division as a leading sort
        # key inside _sort_conference_teams: the head-to-head tiebreak in that function
        # (_head_to_head_winner, invoked when exactly two teams are tied on conf win%) should
        # only ever compare teams competing for the SAME division title. Verified against the
        # function body above -- it pairs up adjacent teams after sorting by conf_pct with no
        # awareness of division, so if it saw a full divisional conference in one pass, two
        # teams from OPPOSITE divisions that happen to tie on conf win% could be swapped based
        # on a head-to-head game that has nothing to do with either team's own division race.
        # Sorting each division's members through the existing, unmodified function in its own
        # call keeps that tiebreak scoped correctly and needs no change to the function itself.
        #
        # Divisions are ordered alphabetically (so "East" precedes "West"), matching the task's
        # requirement and today's only real case. For a conference with no divisions at all,
        # every team's division is None, so there is exactly one group (key None) and this is
        # a single _sort_conference_teams call over the full member list -- byte-identical to
        # the pre-existing behavior.
        divisions_present = sorted({e["division"] for e in entries}, key=lambda d: (d is None, d))
        # Built from the FULL member list, before the division split: the engine's
        # opponent-facing primitives need every member's conference record and the conference's
        # whole order of finish, not one division's slice of them.
        tiebreak = _build_tiebreak_inputs(raw_conf, entries, season, non_fbs_team_names)
        sorted_entries: List[Dict[str, Any]] = []
        for division in divisions_present:
            group = [e for e in entries if e["division"] == division]
            sorted_entries.extend(
                _sort_conference_teams(group, rows, season, champ_game_ids, tiebreak)
            )
        conferences_out.append({"name": _display_conference_name(raw_conf), "teams": sorted_entries})

    return {
        "season": season,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conferences": conferences_out,
    }


# ---------------------------------------------------------------------------
# DB fetch + R2 publish (I/O boundary -- everything above this line is a
# pure function over plain dicts, per the validation_expected split)
# ---------------------------------------------------------------------------
def _fetch_schedule_grid_rows(engine, season: int) -> List[Dict[str, Any]]:
    df = pd.read_sql_query(f"SELECT * FROM schedule_grid WHERE season = {int(season)};", engine)
    return df.to_dict("records")


def _fetch_teams_meta(engine, season: int) -> Dict[str, Dict[str, Any]]:
    df = pd.read_sql_query(f"SELECT school, conference, division, logos FROM teams WHERE season = {int(season)};", engine)

    def _null_to_none(value):
        # teams.division is NULL for every team outside a divisional conference, and
        # teams.conference can be NULL too. How pandas represents that NULL depends on the
        # version: 2.x yields None, but 3.x yields float('nan') for an object/text column.
        # nan is poison here -- `nan is None` is False and `nan == nan` is False -- so the
        # division grouping below silently matches NOTHING and every non-divisional
        # conference publishes with an empty teams list. No exception, no log, just a
        # vanished conference. Normalising at the boundary is the same thing _resolve_logo
        # already does for the logos column.
        return None if pd.isnull(value) else value

    return {
        row["school"]: {
            "conference": _null_to_none(row["conference"]),
            "division": _null_to_none(row["division"]),
            "logos": row["logos"],
        }
        for row in df.to_dict("records")
    }


def _fetch_non_fbs_logos(engine, season: int) -> Dict[str, Any]:
    """
    Opponent-logo lookup for the non-FBS Division-I teams that appear on FBS schedules.

    This is the consuming half of the `non_fbs_teams` table: the ingest
    (database/get_non_fbs_teams.py) populates it every run, and until this existed nothing
    read it, so an FCS opponent still rendered as plain text in the grid despite the data
    being present. Kept as its own query rather than a UNION into _fetch_teams_meta for the
    reason migration 0004 spells out at length -- teams_meta's key set IS the FBS universe for
    four different consumers, and this table must never merge into it.

    Returns:
        Dict[school -> logos]: empty dict if the table has no rows for `season` (a season
        ingested before migration 0004, say), which degrades to the previous text-name
        rendering rather than raising.
    """
    df = pd.read_sql_query(
        f"SELECT school, logos FROM non_fbs_teams WHERE season = {int(season)};", engine
    )
    return {row["school"]: row["logos"] for row in df.to_dict("records")}


def _fetch_team_ranks(engine, season: int) -> Dict[str, int]:
    """
    Season-scoped current-rank lookup (T5): the `ratings` table has no rank column of its own
    (team, rating, wins, losses, ties, season, week) -- rank is derived by ordering rating
    descending, and artifacts/rankings.py's compute_rank_and_delta already does exactly that
    (reused here with previous_df=None since this only needs the current rank, not a delta).

    Deliberately season-scoped, not (season, week): publish_schedule_artifact takes no week
    argument by design (the schedule artifact is a season-scoped key layout, unlike rankings'
    per-week snapshots), so this helper finds the latest week with ratings for `season` itself
    -- WHERE season = N AND week = (SELECT MAX(week) FROM ratings WHERE season = N) -- rather
    than requiring a week to be threaded in from the caller. main.py inserts ratings before
    publishing the schedule artifact in the same run, so a same-season query here sees that
    run's own just-inserted data.

    Returns:
        Dict[team -> rank]: empty dict if `season` has no ratings rows at all (MAX(week) is
        NULL) -- callers must treat a team missing from this dict as rank: null, never raise or
        guess. Never raises itself; the pipeline-facing caller wraps this the same as every other
        DB read in this module.
    """
    latest_week_df = pd.read_sql_query(
        f"SELECT MAX(week) AS week FROM ratings WHERE season = {int(season)};", engine
    )
    latest_week = latest_week_df["week"].iloc[0]
    if pd.isnull(latest_week):
        return {}
    latest_week = int(latest_week)

    ratings_df = pd.read_sql_query(
        f"SELECT team, rating FROM ratings WHERE season = {int(season)} AND week = {latest_week};",
        engine,
    )
    if ratings_df.empty:
        return {}

    ranked_df = compute_rank_and_delta(ratings_df, None)
    return dict(zip(ranked_df["team"], ranked_df["rank"].astype(int)))


_SEASON_KEY_RE = re.compile(r"^schedule/(\d+)/latest\.json$")


def build_schedule_index(client, bucket: str) -> Optional[Dict[str, Any]]:
    """
    Derive the schedule index from the bucket's actual schedule/{season}/latest.json keys,
    self-healing via listing (same philosophy as artifacts/r2.py's build_index(), own regex --
    not a literal reuse of rankings' per-week key shape).
    Returns:
        Optional[dict]: {generated_at_utc, latest_season, seasons: [int, ...]} (descending), or
                         None if the bucket has no schedule artifacts yet, or on any failure.
    """
    try:
        seasons = set()
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix="schedule/"):
            for obj in page.get("Contents", []):
                match = _SEASON_KEY_RE.match(obj["Key"])
                if match:
                    seasons.add(int(match.group(1)))

        if not seasons:
            return None

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "latest_season": max(seasons),
            "seasons": sorted(seasons, reverse=True),
        }
    except Exception as e:
        logger.warning("Failed to build schedule index: %s", e)
        return None


def refresh_schedule_index() -> None:
    """Rebuild and publish schedule/index.json from the bucket's current contents. Never raises."""
    try:
        load_dotenv()
        client = get_r2_client()
        if client is None:
            logger.warning("R2 not configured, skipping schedule index refresh")
            return

        bucket = os.getenv("R2_BUCKET_NAME")
        if not bucket:
            logger.warning("R2_BUCKET_NAME not configured, skipping schedule index refresh")
            return

        index = build_schedule_index(client, bucket)
        if index is None:
            logger.warning("No schedule artifacts found in R2, skipping schedule index publish")
            return

        if upload_json(client, bucket, "schedule/index.json", index):
            logger.info("Published schedule index to R2 key schedule/index.json")
        else:
            logger.warning("Schedule index publish failed")
    except Exception as e:
        logger.exception("refresh_schedule_index failed unexpectedly: %s", e)
        return


def publish_schedule_artifact(year: int) -> None:
    """
    Query schedule_grid + teams for `year`, assemble the Season Grid payload, and publish it to
    R2 under schedule/{year}/latest.json (+ refresh schedule/index.json).
    Args:
        year (int): Season year. No `week` argument -- this artifact is season-scoped, not a
                     per-week snapshot like the rankings artifact.
    Returns:
        None
    Notes:
        Fully exception-safe: any failure (DB read, assembly, R2 credentials, upload) is logged
        as a warning/exception and this function returns cleanly. It must never raise, since the
        rest of the pipeline run (model training, DB insert, rankings publish) has already
        succeeded by the time this runs and must not be put at risk.
    """
    try:
        load_dotenv()
        db_url = (
            f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
            "?sslmode=require"
        )
        engine = create_engine(db_url)
        try:
            rows = _fetch_schedule_grid_rows(engine, year)
            teams_meta = _fetch_teams_meta(engine, year)
            team_ranks = _fetch_team_ranks(engine, year)
            # Supplementary and non-fatal: a missing/empty non_fbs_teams table must not stop
            # the artifact publishing, it just means FCS opponents keep rendering as text.
            try:
                non_fbs_logos = _fetch_non_fbs_logos(engine, year)
            except Exception as e:
                logger.warning(
                    "Could not read non_fbs_teams for season=%s; FCS opponents will render "
                    "as text. Exception: %s", year, e,
                )
                non_fbs_logos = {}
        finally:
            engine.dispose()

        if not teams_meta:
            logger.warning("No teams found for season=%s; skipping schedule artifact publish", year)
            return
        if not rows:
            logger.warning("No schedule_grid rows found for season=%s; skipping schedule artifact publish", year)
            return

        payload = build_schedule_payload(rows, teams_meta, year, team_ranks, non_fbs_logos)

        client = get_r2_client()
        if client is None:
            logger.warning("R2 not configured, skipping schedule artifact publish")
            return

        bucket = os.getenv("R2_BUCKET_NAME")
        if not bucket:
            logger.warning("R2_BUCKET_NAME not configured, skipping schedule artifact publish")
            return

        key = f"schedule/{year}/latest.json"
        if upload_json(client, bucket, key, payload):
            logger.info("Published schedule artifact to R2 key %s", key)
        else:
            logger.warning("Schedule artifact publish failed for R2 key %s", key)

        refresh_schedule_index()
    except Exception as e:
        logger.exception("publish_schedule_artifact failed unexpectedly for year=%s: %s", year, e)
        return
