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

See docs/schedule-grid/plan.yaml (contracts.interfaces, T4b task block) and
docs/schedule-grid/handoffs/T4b-handoff.yaml for the full spec this module
implements.

EXCEPTION DISCIPLINE: every public function here follows artifacts/r2.py's
exact never-raise, log-and-continue contract -- a schedule-artifact publish
failure must never break the rest of the pipeline run (model training, DB
insert, rankings publish have already happened by the time this runs).
"""

import logging
import os
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
from dotenv import load_dotenv  # type: ignore
from sqlalchemy import create_engine  # type: ignore

from artifacts import schedule_standings
from artifacts.bowl_names import short_bowl_name
from artifacts.r2 import get_r2_client, upload_json
from artifacts.rankings import CONFERENCE_DISPLAY_NAMES, _resolve_logo, compute_rank_and_delta
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
# cell (reusing the existing status -- see contracts.interfaces in
# docs/season-grid-refinement/plan.yaml, no new status value) instead of `bye`,
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
# entirely). See docs/schedule-grid/implementation-report.yaml:119 for where this was deferred.
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
# ---------------------------------------------------------------------------
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

    candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("season") != season:
            continue
        if row.get("season_type") != "regular":
            continue
        if not row.get("conference_game"):
            continue
        conf = row.get("conference")
        if conf not in qualifying_conferences:
            continue
        if _is_army_navy_pairing(row.get("team"), row.get("opponent")):
            continue  # see module docstring section above -- deliberate carve-out, not an oversight
        if row.get("start_date") is None:
            # Cannot be bucketed, so it can neither be nor rule out a championship game.
            continue
        candidates[conf].append(row)

    result: Dict[str, int] = {}
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

    return week_columns + [
        (CONF_CHAMPIONSHIP_SLOT_ID, ccg_label),
        (ARMY_NAVY_SLOT_ID, army_navy_label),
    ] + CFP_SLOTS


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

    K8: the seed degrades the label, not the other way around -- when resolved, it becomes the
    literal display text via game_name/game_name_short ("Bye (No. N)"), read directly by the
    frontend exactly like a real game's bowl name; when absent, game_name stays null and the
    frontend's STATUS_TEXT_BY_SLOT[CFP_BYE_STATUS] fallback ("CFP Bye") renders instead. A wrong
    seed on a public page is worse than an absent one.
    """
    game_name = f"Bye (No. {seed})" if seed is not None else None
    return {
        "season_type": logical_season_type,
        "opponent": None,
        "opponent_logo_url": None,
        "conditional_opponent": None,
        "game_name": game_name,
        "game_name_short": game_name,  # already one line; no bowl-name sponsor text to strip
        "home_away": None,
        "neutral_site": False,
        "status": CFP_BYE_STATUS,
        "team_score": None,
        "opp_score": None,
        "playoff_round": None,
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
#   conference-championship winner first (K3/K9 -- see _is_champion in
#   build_schedule_payload) -> conf win% desc -> head-to-head IF exactly two
#   teams tied on conf win% AND on champion status (K4 -- see the grouping
#   loop below for why a champion must never be grouped with a non-champion)
#   and played each other this season -> overall win% desc -> name asc.
# Independents: overall win% desc -> name asc (no conference tiebreak
# question for them).
# ---------------------------------------------------------------------------
def _head_to_head_winner(rows: List[Dict[str, Any]], season: int, team_a: str, team_b: str) -> Optional[str]:
    for row in rows:
        if row.get("season") != season:
            continue
        if not row.get("conference_game"):
            continue
        if row.get("status") not in ("win", "loss"):
            continue
        team, opponent = row.get("team"), row.get("opponent")
        if team == team_a and opponent == team_b:
            return team_a if row["status"] == "win" else team_b
        if team == team_b and opponent == team_a:
            return team_b if row["status"] == "win" else team_a
    return None


def _sort_conference_teams(entries: List[Dict[str, Any]], rows: List[Dict[str, Any]], season: int) -> List[Dict[str, Any]]:
    for e in entries:
        w, l = e["record"]["wins"], e["record"]["losses"]
        # Use 0.5 as sentinel for unplayed overall record (neutral between win and loss),
        # not -1.0 (which sorts worse than any real percentage, even 0-1).
        # This ensures: team with 1-0 record > team with 0-0 record > team with 0-1 record.
        e["_overall_pct"] = (w / (w + l)) if (w + l) > 0 else 0.5
        # Whether any conference game has been played, used only as a sort tiebreak below.
        e["_conf_played"] = bool(e["conf_record"] and (e["conf_record"]["wins"] + e["conf_record"]["losses"]) > 0)
        if e["conf_record"] is not None:
            cw, cl = e["conf_record"]["wins"], e["conf_record"]["losses"]
            # Use 0.5 as sentinel for unplayed conference record (neutral between win and loss),
            # matching the overall record logic. This fixes the NC State vs Duke case where
            # a team with 0-1 conference record should sort below a team with 0-0.
            e["_conf_pct"] = (cw / (cw + cl)) if (cw + cl) > 0 else 0.5
        else:
            e["_conf_pct"] = None

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
        # T2/K4: champion status (default False, so hand-built entries in existing tests that
        # never set _is_champion keep sorting exactly as before) is the TOP sort tier -- a
        # conference-championship-game winner sorts first regardless of its conf_pct.
        entries.sort(key=lambda e: (-e.get("_is_champion", False),
                                    -(e["_conf_pct"] if e["_conf_pct"] is not None else -1.0),
                                    -e["_conf_played"], -e["_overall_pct"], e["team"]))
        i, n = 0, len(entries)
        while i < n:
            j = i
            # Group on percentage, whether the team has played, AND champion status (T2/K4).
            # Percentage alone is not enough: an unplayed record scores 0.5, which ties it with
            # every 1-1, 2-2 and 3-3 team, so one 0-0 team joining two genuinely tied teams grows
            # the group to three and silently cancels their head-to-head swap -- displaying the
            # loser of that game above the winner. A team that has played nobody cannot be part
            # of a head-to-head tie by definition, so it must never join the group.
            #
            # The _is_champion term guards a DIFFERENT failure: a champion can easily tie a
            # non-champion on conf_pct (the championship game itself isn't the only thing that
            # separates them), and if that non-champion beat the champion earlier in the regular
            # season, the swap below would fire on that meeting and undo the whole champion-first
            # rule from the sort key above -- silently, since the swap has no notion of
            # championship status without this check. A unique champion (True) can therefore
            # never share a group with anyone else (all False), so the swap below never touches it.
            while j + 1 < n and entries[j + 1]["_conf_pct"] is not None and entries[i]["_conf_pct"] is not None \
                    and abs(entries[j + 1]["_conf_pct"] - entries[i]["_conf_pct"]) < 1e-9 \
                    and entries[j + 1]["_conf_played"] == entries[i]["_conf_played"] \
                    and entries[j + 1].get("_is_champion", False) == entries[i].get("_is_champion", False):
                j += 1
            group = entries[i:j + 1]
            if len(group) == 2 and group[0]["_conf_pct"] is not None and group[0]["_conf_played"]:
                t1, t2 = group[0]["team"], group[1]["team"]
                winner = _head_to_head_winner(rows, season, t1, t2)
                if winner == t2:
                    entries[i], entries[i + 1] = entries[i + 1], entries[i]
            i = j + 1
    else:
        entries.sort(key=lambda e: (-e["_overall_pct"], e["team"]))

    for e in entries:
        del e["_overall_pct"]
        del e["_conf_pct"]
        del e["_conf_played"]
        # K10: scratch state, stripped before these entries reach the published payload --
        # pop (not del) because, unlike the three fields above, this function itself never sets
        # _is_champion unconditionally, so hand-built entries in existing tests that never set
        # it at all must not raise a KeyError here.
        e.pop("_is_champion", None)
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
    tests/test_conference_championship_records.py's K1 regression test).
    """
    if not champ_game_ids:
        return
    for row in rows:
        if row.get("season") != season:
            continue
        if row.get("game_id") not in champ_game_ids:
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
# T2/K3 -- resolve each identified championship game to its ACTUAL winner
# ---------------------------------------------------------------------------
def _resolve_conference_champions(
    champ_games_by_conf: Dict[str, int],
    rows: List[Dict[str, Any]],
    season: int,
) -> Dict[str, str]:
    """
    Dict[raw conference -> winning team name], for conferences whose identified championship
    game (identify_conference_championship_games) has actually been PLAYED.

    K3: an identified-but-unplayed game contributes NO entry here -- a team that has merely
    clinched a spot in the title game is not a champion, so that conference's sort order stays
    untouched (see _sort_conference_teams's _is_champion default of False) until the game
    resolves.
    """
    game_id_to_conf = {game_id: conf for conf, game_id in champ_games_by_conf.items()}
    champions: Dict[str, str] = {}
    for row in rows:
        if row.get("season") != season:
            continue
        game_id = row.get("game_id")
        if game_id not in game_id_to_conf:
            continue
        if row.get("status") != "win":
            continue
        champions[game_id_to_conf[game_id]] = row.get("team")
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
    # T2/K3: the identified game's actual winner (None for a conference with no identified game,
    # or one that hasn't been played yet) -- see _resolve_conference_champions.
    conference_champions = _resolve_conference_champions(champ_games_by_conf, rows, season)

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
        sorted_entries: List[Dict[str, Any]] = []
        for division in divisions_present:
            group = [e for e in entries if e["division"] == division]
            sorted_entries.extend(_sort_conference_teams(group, rows, season))
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
