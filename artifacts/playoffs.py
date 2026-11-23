"""
playoffs.py -- College Football Playoff bracket artifact builder + R2 publisher.

Produces `playoffs/{season}/latest.json` and the self-healing `playoffs/index.json`, the shared
contract documented in docs/clinch-h2h-and-ccg-ingest/playoffs-contract.md (the durable statement
of it is this module plus tests/test_playoffs_*.py).

Two kinds of season, both driven by SEASON_RULES below:
  - "actual"    -- the real bracket reconstructed from stored postseason games (playoff_round_name
                   set). Seeds, bowl names and scores are read from the rows.
  - "projected" -- "if the season ended today", computed from our model's rankings plus the
                   conference standings the Season Grid already shows.
A season configured "projected" (2026) SWITCHES to the actual bracket once the stored postseason
rows (playoff_round_name set) identify ALL field_size seeds, each seed on exactly one team, with no
participant lacking a seed (selection day onward; bye teams are seeded through their quarterfinal
rows). Anything less -- no CFP rows, a partial set of seeds, a duplicate seed, an unseeded
participant -- stays on the projection with one logged reason; a half-known bracket is never
published as "actual", and incomplete data never makes a projected season return None. A season
configured "actual" (2025) always uses the stored bracket and REFUSES to build (None, logged at
ERROR) on duplicate or missing seed data, since the payload's `teams` lookup would break.
Any season not in SEASON_RULES gets no artifact (the publisher skips it, the builder returns
None): the 2027 season moves to a 24-team format and every rule here is 12-team specific.

EXCEPTION DISCIPLINE: publish_playoffs_artifact / refresh_playoffs_index follow artifacts/r2.py's
never-raise, log-and-continue contract so a failure here can never break a pipeline run.
Everything above the "DB fetch + R2 publish" section is pure over plain dicts.
"""

import logging
import os
import re
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd  # type: ignore
from dotenv import load_dotenv  # type: ignore
from sqlalchemy import create_engine  # type: ignore

from artifacts import schedule_standings
from artifacts.bowl_names import _ROOT_NAMES_BY_LENGTH_DESC
from artifacts.r2 import get_r2_client, upload_json
from artifacts.rankings import CONFERENCE_DISPLAY_NAMES, _resolve_logo
from artifacts.schedule import (
    _fetch_schedule_grid_rows,
    _fetch_team_ranks,
    _fetch_teams_meta,
    _to_datetime,
    compute_season_standings,
)
from utils import football_day

logger = logging.getLogger("cfb_lp")

# The line publish_playoffs_artifact logs on success; .github/workflows/publish-playoffs-artifact.yml
# greps for it (tests/test_playoffs_publish.py pins the two together).
PUBLISHED_LOG_MESSAGE = "Published playoffs artifact to R2 key %s"

# ---------------------------------------------------------------------------
# Season-scoped rules.
#
# PROVENANCE: the 2026 rules are the owner's stated rules (playoffs-contract.md, "Owner rules")
# for the 12-team CFP of 2026-27: champions of the four AQ conferences regardless of rank; the
# highest-ranked Group of Six TEAM (champion or not); Notre Dame if ranked in the top 12 (its CFP
# MOU); then the next highest-ranked teams to 12; seeds strictly by rank, byes for seeds 1-4.
# Round dates/venues are from the 2026-27 CFP schedule (first round at campus sites Dec 18-19,
# quarterfinals Dec 30 - Jan 1, semifinals Jan 14-15, championship Jan 25 in Las Vegas).
# Conference names are the RAW teams.conference values, not display names.
# 2027 is a 24-team format: deliberately NOT configured, so no artifact is built for it.
# "actual" seasons read their dates from the stored games, so they carry no date config.
# ---------------------------------------------------------------------------
ROUND_FIRST = "first_round"
ROUND_QF = "quarterfinal"
ROUND_SF = "semifinal"
ROUND_CH = "championship"
ROUNDS: List[Tuple[str, str]] = [
    (ROUND_FIRST, "First Round"),
    (ROUND_QF, "Quarterfinals"),
    (ROUND_SF, "Semifinals"),
    (ROUND_CH, "National Championship"),
]

SEASON_RULES: Dict[int, Dict[str, Any]] = {
    2025: {
        "kind": "actual",
        "field_size": 12,
        "byes": 4,
        "round_notes": {ROUND_FIRST: "Campus Sites"},
    },
    2026: {
        # "projected" until the stored CFP rows identify all 12 seeds, then the ACTUAL bracket is
        # built instead (kind "actual"; see build_playoffs_payload). The round dates below keep
        # labelling the rounds in both modes: stored games may carry no dates yet, and a label
        # computed from a partial set of games would be wrong.
        "kind": "projected",
        "field_size": 12,
        "byes": 4,
        "aq_conferences": ["ACC", "Big Ten", "Big 12", "SEC"],
        "g6_conferences": [
            "American Athletic", "Conference USA", "Mid-American", "Mountain West", "Pac-12", "Sun Belt",
        ],
        "guarantee": {"Notre Dame": 12},
        "first_out_count": 4,
        "round_dates": {
            ROUND_FIRST: "Dec 18-19",
            ROUND_QF: "Dec 30 - Jan 1",
            ROUND_SF: "Jan 14-15",
            ROUND_CH: "Jan 25",
        },
        "round_notes": {ROUND_FIRST: "Campus Sites", ROUND_CH: "Las Vegas"},
    },
}

# Bracket geometry shared by both kinds (the real 2025 bracket has it too). Index = "quarter" =
# the quarterfinal order the first-round game feeds: 1 v 8/9, 4 v 5/12, 2 v 7/10, 3 v 6/11.
QUARTER_BYE_SEEDS = [1, 4, 2, 3]
QUARTER_FIRST_ROUND_SEEDS = [(8, 9), (5, 12), (7, 10), (6, 11)]
SEED_QUARTER: Dict[int, int] = {}
for _q, (_bye, (_hi, _lo)) in enumerate(zip(QUARTER_BYE_SEEDS, QUARTER_FIRST_ROUND_SEEDS)):
    SEED_QUARTER.update({_bye: _q, _hi: _q, _lo: _q})

BID_CHAMPION = "champion"
BID_G6 = "g6"
BID_NOTRE_DAME = "notre_dame"
BID_AT_LARGE = "at_large"
_BID_LABELS = {
    BID_G6: "Highest-Ranked Group of Six",
    BID_NOTRE_DAME: "Top-12 Guarantee",
    BID_AT_LARGE: "At-Large",
}


def _display_conference(raw: Optional[str]) -> Optional[str]:
    return CONFERENCE_DISPLAY_NAMES.get(raw, raw) if raw else raw


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _champion_label(raw_conference: str, actual: bool) -> str:
    return f"{raw_conference} Champion" if actual else f"Projected {raw_conference} Champion"


# ---------------------------------------------------------------------------
# Pure selection (projected seasons)
# ---------------------------------------------------------------------------
def _conference_win_pct(conf_record: Optional[Dict[str, int]]) -> float:
    if not conf_record:
        return 0.0
    games = conf_record["wins"] + conf_record["losses"]
    return conf_record["wins"] / games if games else 0.0


def resolve_projected_champions(
    ranks: Dict[str, int],
    standings: Dict[str, Dict[str, Any]],
    actual_champions: Dict[str, str],
    conferences: Iterable[str],
    title_game_pairs: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Champion (actual or projected) of each conference in `conferences` (RAW names).

    Owner's rule, taken literally:
      1. a conference whose title game has been PLAYED is decided: `actual_champions[conf]` wins
         outright, whatever the standings or ranks say;
      2. otherwise the field of contenders is NARROWED, in this order, BEFORE any rank filter:
           a. the two participants of the conference's IDENTIFIED championship game, played or
              not (`title_game_pairs[conf]`, from identify_conference_championship_games), when
              exactly two teams are given, both are members of the conference's standings and
              neither is already "eliminated" (an eliminated participant means the identified game
              is a false positive -- see schedule.identify_conference_championship_games);
           b. else the two teams that have CLINCHED (championship_status == "clinched"), when
              exactly two have;
           c. else every team in the conference.
         Only then are the narrowed teams filtered to the RANKED ones. So when one of a narrowed
         pair is unranked, the other is picked; when neither is, the conference gets no champion
         (logged as a warning) -- the narrowing never silently falls away to the whole conference,
         which could crown a team that is not in the title game;
      3. among the candidates, the best conference win percentage leads; ties at the top go to
         the team ranked highest by OUR model -- never head-to-head.

    `ranks`: team -> model rank (1 = best); a team missing from it is unranked and never a
    projected champion. `standings`: schedule_standings.compute_standings output.
    Returns {raw conference -> {"team": str, "actual": bool}}; a conference with no actual
    champion and no ranked candidate is absent.
    """
    title_game_pairs = title_game_pairs or {}
    result: Dict[str, Dict[str, Any]] = {}
    for conference in conferences:
        actual = actual_champions.get(conference)
        if actual:
            result[conference] = {"team": actual, "actual": True}
            continue
        members = [
            team for team, st in standings.items()
            if st.get("conference") == conference and st.get("conf_record") is not None
        ]
        narrowed, narrowed_by = members, None
        pair = title_game_pairs.get(conference) or set()
        if (
            len(pair) == 2 and pair <= set(members)
            and not any(standings[t].get("championship_status") == "eliminated" for t in pair)
        ):
            narrowed, narrowed_by = sorted(pair), "identified title game"
        else:
            clinched = [t for t in members if standings[t].get("championship_status") == "clinched"]
            if len(clinched) == 2:
                narrowed, narrowed_by = clinched, "two clinched teams"
        candidates = [t for t in narrowed if t in ranks]
        if not candidates:
            if narrowed_by:
                logger.warning(
                    "playoffs: neither of the %s for %s (%s) is ranked; no champion projected for it.",
                    narrowed_by, conference, sorted(narrowed),
                )
            continue
        best_pct = max(_conference_win_pct(standings[t]["conf_record"]) for t in candidates)
        leaders = [t for t in candidates if _conference_win_pct(standings[t]["conf_record"]) == best_pct]
        result[conference] = {"team": min(leaders, key=lambda t: ranks[t]), "actual": False}
    return result


def select_projected_field(
    ranks: Dict[str, int],
    team_conf_raw: Dict[str, Optional[str]],
    champions: Dict[str, Dict[str, Any]],
    rules: Optional[Dict[str, Any]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    The projected playoff field, seeded 1..N strictly by model rank.

    Qualifying, in precedence order when a team qualifies several ways
    (champion > g6 > notre_dame > at_large):
      - the champion of each AQ conference, regardless of rank (`champions` as returned by
        resolve_projected_champions);
      - the highest-ranked Group of Six TEAM, champion or not (sets are disjoint from the AQ
        champions by construction -- a team has one conference);
      - each guaranteed team (Notre Dame) ranked within its threshold -- a team already in the
        field consumes no extra slot;
      - then the next highest-ranked teams until the field is full. No duplicates.

    Teams with no rank are ignored. Fewer rankable teams than the field size -> None (logged).
    Returns a list of {team, rank, seed, bid, bid_label, bye} ordered by seed.
    """
    rules = rules or SEASON_RULES[2026]
    field_size = rules["field_size"]
    ordered = sorted(ranks, key=lambda t: ranks[t])
    if len(ordered) < field_size:
        logger.warning(
            "playoffs: only %d ranked teams, need %d for a projected field; no projection built.",
            len(ordered), field_size,
        )
        return None

    bids: Dict[str, Tuple[str, str]] = {}
    for conference in rules["aq_conferences"]:
        champion = champions.get(conference)
        if champion and champion["team"] in ranks:
            bids[champion["team"]] = (BID_CHAMPION, _champion_label(conference, champion["actual"]))
        elif champion:
            logger.warning("playoffs: %s champion %r has no rank; cannot be seeded.", conference, champion["team"])

    g6 = set(rules["g6_conferences"])
    g6_best = next((t for t in ordered if team_conf_raw.get(t) in g6), None)
    if g6_best is not None and g6_best not in bids:
        bids[g6_best] = (BID_G6, _BID_LABELS[BID_G6])

    for team, threshold in rules["guarantee"].items():
        if team in ranks and ranks[team] <= threshold and team not in bids:
            bids[team] = (BID_NOTRE_DAME, _BID_LABELS[BID_NOTRE_DAME])

    for team in ordered:
        if len(bids) >= field_size:
            break
        bids.setdefault(team, (BID_AT_LARGE, _BID_LABELS[BID_AT_LARGE]))

    selected = sorted(bids, key=lambda t: ranks[t])
    return [
        {
            "team": team, "rank": ranks[team], "seed": seed,
            "bid": bids[team][0], "bid_label": bids[team][1], "bye": seed <= rules["byes"],
        }
        for seed, team in enumerate(selected, start=1)
    ]


def select_first_out(ranks: Dict[str, int], field: List[Dict[str, Any]], count: int) -> List[str]:
    """The next `count` ranked teams outside the field, best first."""
    in_field = {entry["team"] for entry in field}
    return [t for t in sorted(ranks, key=lambda t: ranks[t]) if t not in in_field][:count]


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------
_SLOT_PREFIX = {ROUND_FIRST: "FR", ROUND_QF: "QF", ROUND_SF: "SF"}


def _slot(round_id: str, order: int) -> str:
    return "CH" if round_id == ROUND_CH else f"{_SLOT_PREFIX[round_id]}{order + 1}"


def _side(seed: Optional[int], team: Optional[str], score: Optional[int] = None, from_slot: Optional[str] = None):
    return {"seed": seed, "team": team, "score": score, "from": from_slot}


def _game(round_id, order, name, date_str, status, top, bottom, winner=None):
    return {
        "slot": _slot(round_id, order), "round": round_id, "order": order, "name": name,
        "date": date_str, "status": status, "top": top, "bottom": bottom, "winner": winner,
    }


def _feeder_slots(round_id: str, order: int) -> Tuple[Optional[str], Optional[str]]:
    """The `from` slots of a game's (top, bottom) side. Later rounds: games 2k and 2k+1 of the
    previous round feed game k. Quarterfinal: top is the bye seed (no feeder), bottom the
    first-round game of the same order."""
    if round_id == ROUND_QF:
        return None, _slot(ROUND_FIRST, order)
    if round_id == ROUND_SF:
        return _slot(ROUND_QF, 2 * order), _slot(ROUND_QF, 2 * order + 1)
    if round_id == ROUND_CH:
        return _slot(ROUND_SF, 0), _slot(ROUND_SF, 1)
    return None, None


def build_projected_games(field: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The 11 projected games for a seeded field: names and dates null (the round-level labels
    carry the dates), later-round sides unknown and linked by `from`."""
    team_by_seed = {e["seed"]: e["team"] for e in field}
    games: List[Dict[str, Any]] = []
    for q, (hi, lo) in enumerate(QUARTER_FIRST_ROUND_SEEDS):
        games.append(_game(
            ROUND_FIRST, q, None, None, "projected",
            _side(hi, team_by_seed.get(hi)), _side(lo, team_by_seed.get(lo)),
        ))
    for q, bye in enumerate(QUARTER_BYE_SEEDS):
        games.append(_game(
            ROUND_QF, q, None, None, "projected",
            _side(bye, team_by_seed.get(bye)), _side(None, None, from_slot=_slot(ROUND_FIRST, q)),
        ))
    for k in range(2):
        top_from, bottom_from = _feeder_slots(ROUND_SF, k)
        games.append(_game(
            ROUND_SF, k, None, None, "projected",
            _side(None, None, from_slot=top_from), _side(None, None, from_slot=bottom_from),
        ))
    top_from, bottom_from = _feeder_slots(ROUND_CH, 0)
    games.append(_game(
        ROUND_CH, 0, None, None, "projected",
        _side(None, None, from_slot=top_from), _side(None, None, from_slot=bottom_from),
    ))
    return games


def _round_id_for(round_name: str) -> Optional[str]:
    lowered = round_name.strip().lower()
    if "quarterfinal" in lowered:
        return ROUND_QF
    if "semifinal" in lowered:
        return ROUND_SF
    if "championship" in lowered:
        return ROUND_CH
    if "first" in lowered:
        return ROUND_FIRST
    return None


def _bowl_root_name(name: Optional[str]) -> Optional[str]:
    """Bowl display name without its sponsor but WITH the word "Bowl" ("Rose Bowl"), via the
    curated roots in artifacts/bowl_names.py. An unrecognised name passes through whole."""
    if not name or not name.strip():
        return None
    lowered = name.strip().lower()
    for root in _ROOT_NAMES_BY_LENGTH_DESC:
        if lowered == root.lower() or lowered.endswith(" " + root.lower()):
            return root
    return name.strip()


def _cfp_game_rows(rows: Iterable[Dict[str, Any]], season: int) -> Dict[Any, Dict[str, Any]]:
    """One schedule_grid row per CFP-bracket game (the table has two perspectives per game;
    the home row is kept when both are present -- orientation is irrelevant downstream)."""
    games: Dict[Any, Dict[str, Any]] = {}
    for row in rows:
        round_name = row.get("playoff_round_name")
        if row.get("season") != season or not isinstance(round_name, str) or not round_name.strip():
            continue
        game_id = row.get("game_id")
        if game_id not in games or row.get("home_away") == "home":
            games[game_id] = row
    return games


def _cfp_seed_map(rows: Iterable[Dict[str, Any]], season: int) -> Tuple[Dict[str, int], List[str]]:
    """({team: seed} from every CFP-bracket row, [problems]). Seeds come from team_seed/opp_seed on
    either perspective of any round (a bye team is seeded through its quarterfinal row). Problems:
    a team with two different seeds, two teams sharing a seed, and a participant with no seed on
    any row -- each would break the payload's `teams` lookup, so callers refuse to build on them."""
    seeds: Dict[str, int] = {}
    problems: List[str] = []
    participants: Set[str] = set()
    for row in rows:
        round_name = row.get("playoff_round_name")
        if row.get("season") != season or not isinstance(round_name, str) or not round_name.strip():
            continue
        for team_key, seed_key in (("team", "team_seed"), ("opponent", "opp_seed")):
            team, seed = row.get(team_key), _int_or_none(row.get(seed_key))
            if not team:
                continue
            participants.add(team)
            if seed is not None and seed in SEED_QUARTER:
                if seeds.setdefault(team, seed) != seed:
                    problems.append(f"{team} has conflicting seeds {seeds[team]} and {seed}")
    for team in sorted(participants - set(seeds)):
        problems.append(f"{team} has no seed on any row")
    by_seed: Dict[int, List[str]] = {}
    for team, seed in seeds.items():
        by_seed.setdefault(seed, []).append(team)
    for seed, teams in sorted(by_seed.items()):
        if len(teams) > 1:
            problems.append(f"seed {seed} is shared by {sorted(teams)}")
    return seeds, problems


def _complete_actual_bracket(rows: List[Dict[str, Any]], season: int, field_size: int) -> Optional[Dict[str, Any]]:
    """The actual bracket for a season configured as projected, or None (with ONE logged reason)
    when the stored rows do not yet identify the whole field: seeds 1..field_size each on exactly
    one team and no participant without a seed. Never raises, never returns a partial bracket."""
    if not _cfp_game_rows(rows, season):
        logger.info("playoffs: no CFP rows stored for season=%s yet; staying on the projection.", season)
        return None
    seeds, problems = _cfp_seed_map(rows, season)
    if problems:
        logger.warning(
            "playoffs: CFP rows for season=%s have bad seed data (%s); staying on the projection.",
            season, "; ".join(problems),
        )
        return None
    if sorted(seeds.values()) != list(range(1, field_size + 1)):
        logger.info(
            "playoffs: CFP rows for season=%s identify only seeds %s of %d; staying on the projection.",
            season, sorted(seeds.values()), field_size,
        )
        return None
    return build_actual_bracket(rows, season)


def build_actual_bracket(rows: List[Dict[str, Any]], season: int) -> Optional[Dict[str, Any]]:
    """
    Reconstruct the real bracket from schedule_grid rows whose playoff_round_name is set.

    Source choice: schedule_grid rows (already fetched for the Season Grid; team_seed/opp_seed are
    the per-perspective flips of games.playoff_home_seed/away_seed) rather than a dedicated query
    -- one DB round trip set, and the same rows the grid's CFP cells are built from.

    CFBD's bracket slot labels (FR1..FR4, QF1..) are NOT in bracket order, so nothing is read from
    them: every position derives from SEEDS. A team's seed fixes its "quarter" (SEED_QUARTER) --
    which first-round game it plays in, which quarterfinal, which semifinal half -- so the order
    of every game is the quarter of its teams (semifinal: quarter // 2). A seed missing from a row
    is filled from any other row of the same team; a game whose teams have no known seed at all is
    dropped with a warning.

    Incomplete brackets (e.g. mid-December) still build: all 11 slots are emitted, a slot with no
    stored game being a "scheduled" placeholder whose unknown sides have team/seed null and a
    `from` link (first-round and quarterfinal placeholders take team names from known seeds).
    No CFP rows at all -> None.

    Side placement: first round and quarterfinal put the higher seed on top (the quarterfinal's
    top is therefore the bye seed); semifinal and championship put the side fed by the earlier
    feeder game on top, so `from` always reads feeder 2k on top, 2k+1 on the bottom.

    Returns {"games": [11 game dicts], "seeds": {team: seed}, "round_dates": {round: [date]}}.
    """
    cfp_rows = _cfp_game_rows(rows, season)
    if not cfp_rows:
        return None

    seeds, problems = _cfp_seed_map(rows, season)
    if problems:
        logger.error(
            "playoffs: refusing to build the actual bracket for season=%s, bad CFP seed data: %s",
            season, "; ".join(problems),
        )
        return None
    team_by_seed = {seed: team for team, seed in seeds.items()}

    built: Dict[Tuple[str, int], Dict[str, Any]] = {}
    round_dates: Dict[str, List[str]] = {round_id: [] for round_id, _ in ROUNDS}
    for row in sorted(cfp_rows.values(), key=lambda r: str(r.get("game_id"))):
        round_id = _round_id_for(row["playoff_round_name"])
        if round_id is None:
            logger.warning("playoffs: unrecognised playoff_round_name %r on game %s", row["playoff_round_name"], row.get("game_id"))
            continue
        sides = [
            {"team": row.get("team"), "score": _int_or_none(row.get("team_score"))},
            {"team": row.get("opponent"), "score": _int_or_none(row.get("opp_score"))},
        ]
        for side in sides:
            side["seed"] = seeds.get(side["team"])
            side["quarter"] = SEED_QUARTER.get(side["seed"])
        quarters = [s["quarter"] for s in sides if s["quarter"] is not None]
        if not quarters:
            logger.warning("playoffs: game %s (%s) has no known seeds; dropped.", row.get("game_id"), round_id)
            continue
        order = {ROUND_FIRST: quarters[0], ROUND_QF: quarters[0], ROUND_SF: quarters[0] // 2, ROUND_CH: 0}[round_id]
        if (round_id, order) in built:
            logger.warning("playoffs: two games map to %s; keeping the first.", _slot(round_id, order))
            continue

        if round_id in (ROUND_SF, ROUND_CH):
            sides.sort(key=lambda s: (s["quarter"] is None, s["quarter"] or 0, s["seed"] or 99))
        else:
            sides.sort(key=lambda s: (s["seed"] is None, s["seed"] or 99))
        feeders = _feeder_slots(round_id, order)
        rendered = []
        for index, side in enumerate(sides):
            from_slot = feeders[index]
            if round_id == ROUND_QF and (side["seed"] or 99) <= len(QUARTER_BYE_SEEDS):
                from_slot = None
            rendered.append(_side(side["seed"], side["team"], side["score"], from_slot))

        played = row.get("status") in ("win", "loss") and all(s["score"] is not None for s in sides)
        winner = None
        if played and sides[0]["score"] != sides[1]["score"]:
            winner = "top" if sides[0]["score"] > sides[1]["score"] else "bottom"
        if round_id == ROUND_CH:
            name: Optional[str] = "National Championship"
        elif round_id == ROUND_FIRST:
            name = None
        else:
            name = _bowl_root_name(row.get("playoff_bowl_name") or row.get("notes"))
        date_str = None
        if row.get("start_date") is not None and not pd.isnull(row.get("start_date")):
            date_str = football_day(_to_datetime(row["start_date"])).isoformat()
            round_dates[round_id].append(date_str)
        built[(round_id, order)] = _game(
            round_id, order, name, date_str, "final" if played else "scheduled",
            rendered[0], rendered[1], winner,
        )

    games: List[Dict[str, Any]] = []
    counts = {ROUND_FIRST: 4, ROUND_QF: 4, ROUND_SF: 2, ROUND_CH: 1}
    for round_id, _label in ROUNDS:
        for order in range(counts[round_id]):
            game = built.get((round_id, order))
            if game is None:
                top_from, bottom_from = _feeder_slots(round_id, order)
                if round_id == ROUND_FIRST:
                    hi, lo = QUARTER_FIRST_ROUND_SEEDS[order]
                    top, bottom = _side(hi, team_by_seed.get(hi)), _side(lo, team_by_seed.get(lo))
                elif round_id == ROUND_QF:
                    bye = QUARTER_BYE_SEEDS[order]
                    top, bottom = _side(bye, team_by_seed.get(bye)), _side(None, None, from_slot=bottom_from)
                else:
                    top, bottom = _side(None, None, from_slot=top_from), _side(None, None, from_slot=bottom_from)
                game = _game(round_id, order, None, None, "scheduled", top, bottom)
            games.append(game)
    return {"games": games, "seeds": seeds, "round_dates": round_dates}


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------
def _dates_label(dates: List[str]) -> Optional[str]:
    """"Jan 25", "Dec 18-19" or "Dec 30 - Jan 1" from ISO dates; None when there are none."""
    if not dates:
        return None
    first, last = date.fromisoformat(min(dates)), date.fromisoformat(max(dates))
    if first == last:
        return f"{first:%b} {first.day}"
    if first.month == last.month:
        return f"{first:%b} {first.day}-{last.day}"
    return f"{first:%b} {first.day} - {last:%b} {last.day}"


def _rounds_block(rules: Dict[str, Any], round_dates: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
    configured = rules.get("round_dates", {})
    notes = rules.get("round_notes", {})
    return [
        {
            "round": round_id, "label": label,
            "dates_label": configured.get(round_id) or _dates_label((round_dates or {}).get(round_id, [])),
            "note": notes.get(round_id),
        }
        for round_id, label in ROUNDS
    ]


def _title_game_pairs(rows: List[Dict[str, Any]], champ_games_by_conf: Dict[str, Any]) -> Dict[str, Set[str]]:
    """{conference -> the two teams of its IDENTIFIED championship game}, played or not, read off
    the rows of the identified game ids. Conferences whose game does not show exactly two teams
    are omitted."""
    wanted = {game_id: conf for conf, game_id in champ_games_by_conf.items()}
    teams: Dict[str, Set[str]] = {}
    for row in rows:
        conf = wanted.get(row.get("game_id"))
        if conf is None:
            continue
        for key in ("team", "opponent"):
            if row.get(key):
                teams.setdefault(conf, set()).add(row[key])
    return {conf: pair for conf, pair in teams.items() if len(pair) == 2}


def build_playoffs_payload(
    rows: List[Dict[str, Any]],
    teams_meta: Dict[str, Dict[str, Any]],
    season: int,
    team_ranks: Optional[Dict[str, int]] = None,
    as_of_week: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    The `playoffs/{season}/latest.json` payload, or None when there is nothing to publish: a
    season outside SEASON_RULES, a projected season with too few ranked teams, or an actual
    season with no usable stored CFP games (none, or bad seed data). A projected-configured season
    whose stored CFP rows identify all 12 seeds is built as the actual bracket instead. Pure.
    """
    rules = SEASON_RULES.get(season)
    if rules is None:
        logger.info("playoffs: season %s has no playoff rules configured; nothing built.", season)
        return None
    team_ranks = {t: r for t, r in (team_ranks or {}).items() if t in teams_meta}
    prelude = compute_season_standings(rows, teams_meta, season)
    standings, actual_champions = prelude.standings, prelude.conference_champions

    def conference_raw(team: str) -> Optional[str]:
        st = standings.get(team)
        return (st["conference"] if st else None) or teams_meta.get(team, {}).get("conference")

    def team_row(team: str, seed: int, bid: str, bid_label: Optional[str]) -> Dict[str, Any]:
        st = standings.get(team)
        record = f"{st['record']['wins']}-{st['record']['losses']}" if st else "0-0"
        return {
            "seed": seed, "team": team, "rank": team_ranks.get(team), "record": record,
            "conference": _display_conference(conference_raw(team)),
            "logo_url": _resolve_logo(teams_meta.get(team, {}).get("logos")),
            "bid": bid, "bid_label": bid_label, "bye": seed <= rules["byes"],
        }

    def actual_payload(bracket: Dict[str, Any]) -> Dict[str, Any]:
        teams = []
        for team, seed in sorted(bracket["seeds"].items(), key=lambda kv: kv[1]):
            conference = conference_raw(team)
            champion_here = actual_champions.get(conference) if conference else None
            if champion_here == team:
                bid, label = BID_CHAMPION, _champion_label(conference, True)
            else:
                # "At-Large" is only asserted when we can see the conference's champion and it is
                # someone else (or the team has no conference); otherwise the label stays null.
                label = _BID_LABELS[BID_AT_LARGE] if (champion_here or conference in (None, schedule_standings.INDEPENDENT_CONFERENCE_VALUE)) else None
                bid = BID_AT_LARGE
            teams.append(team_row(team, seed, bid, label))
        return {
            "season": season, "kind": "actual", "as_of_week": None,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "teams": teams, "rounds": _rounds_block(rules, bracket["round_dates"]),
            "games": bracket["games"], "first_out": [],
        }

    if rules["kind"] == "actual":
        bracket = build_actual_bracket(rows, season)
        if bracket is None:
            logger.warning("playoffs: no usable CFP games stored for season=%s; nothing built.", season)
            return None
        return actual_payload(bracket)

    # Configured "projected": switch to the real bracket once the stored rows identify every seed.
    bracket = _complete_actual_bracket(rows, season, rules["field_size"])
    if bracket is not None:
        return actual_payload(bracket)

    champions = resolve_projected_champions(
        team_ranks, standings, actual_champions, rules["aq_conferences"],
        title_game_pairs=_title_game_pairs(rows, prelude.champ_games_by_conf),
    )
    field = select_projected_field(
        team_ranks, {t: conference_raw(t) for t in team_ranks}, champions, rules
    )
    if field is None:
        return None
    first_out = [
        {
            "team": team, "rank": team_ranks[team],
            "record": f"{standings[team]['record']['wins']}-{standings[team]['record']['losses']}" if team in standings else "0-0",
            "conference": _display_conference(conference_raw(team)),
            "logo_url": _resolve_logo(teams_meta.get(team, {}).get("logos")),
        }
        for team in select_first_out(team_ranks, field, rules["first_out_count"])
    ]
    return {
        "season": season, "kind": "projected", "as_of_week": as_of_week,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "teams": [team_row(e["team"], e["seed"], e["bid"], e["bid_label"]) for e in field],
        "rounds": _rounds_block(rules), "games": build_projected_games(field), "first_out": first_out,
    }


# ---------------------------------------------------------------------------
# DB fetch + R2 publish (I/O boundary)
# ---------------------------------------------------------------------------
def _fetch_ratings_week(engine, season: int) -> Optional[int]:
    """Latest ratings week for `season` -- the week _fetch_team_ranks ranks from. Its own tiny
    query because that helper returns ranks only."""
    df = pd.read_sql_query(f"SELECT MAX(week) AS week FROM ratings WHERE season = {int(season)};", engine)
    week = df["week"].iloc[0]
    return None if pd.isnull(week) else int(week)


def _engine():
    load_dotenv()
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )


def load_playoffs_payload(season: int) -> Optional[Dict[str, Any]]:
    """Read-only: fetch the inputs and build the payload (the exact one the publisher uploads and
    the script writes). None for an unsupported season or when nothing can be built. May raise
    on DB errors; publish_playoffs_artifact wraps it."""
    if season not in SEASON_RULES:
        return None
    engine = _engine()
    try:
        rows = _fetch_schedule_grid_rows(engine, season)
        teams_meta = _fetch_teams_meta(engine, season)
        team_ranks = _fetch_team_ranks(engine, season)
        as_of_week = _fetch_ratings_week(engine, season) if SEASON_RULES[season]["kind"] == "projected" else None
    finally:
        engine.dispose()
    if not teams_meta or not rows:
        logger.warning("playoffs: no teams/schedule rows for season=%s; nothing built.", season)
        return None
    return build_playoffs_payload(rows, teams_meta, season, team_ranks, as_of_week)


_SEASON_KEY_RE = re.compile(r"^playoffs/(\d+)/latest\.json$")


def build_playoffs_index(client, bucket: str) -> Optional[Dict[str, Any]]:
    """playoffs/index.json derived from the bucket's actual playoffs/{season}/latest.json keys
    (self-healing, same shape as schedule/index.json). None if there are none, or on failure."""
    try:
        seasons = set()
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="playoffs/"):
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
        logger.warning("Failed to build playoffs index: %s", e)
        return None


def refresh_playoffs_index() -> None:
    """Rebuild and publish playoffs/index.json from the bucket's current contents. Never raises."""
    try:
        load_dotenv()
        client = get_r2_client()
        if client is None:
            logger.warning("R2 not configured, skipping playoffs index refresh")
            return
        bucket = os.getenv("R2_BUCKET_NAME")
        if not bucket:
            logger.warning("R2_BUCKET_NAME not configured, skipping playoffs index refresh")
            return
        index = build_playoffs_index(client, bucket)
        if index is None:
            logger.warning("No playoffs artifacts found in R2, skipping playoffs index publish")
            return
        if upload_json(client, bucket, "playoffs/index.json", index):
            logger.info("Published playoffs index to R2 key playoffs/index.json")
        else:
            logger.warning("Playoffs index publish failed")
    except Exception as e:
        logger.exception("refresh_playoffs_index failed unexpectedly: %s", e)


def publish_playoffs_artifact(year: int) -> None:
    """
    Build the playoffs payload for `year` and publish it to R2 at playoffs/{year}/latest.json
    (+ refresh playoffs/index.json). A season outside SEASON_RULES is skipped. Fully
    exception-safe: every failure is logged and swallowed, never raised.
    """
    try:
        if year not in SEASON_RULES:
            logger.info("Skipping playoffs artifact publish: no playoff rules for season=%s", year)
            return
        payload = load_playoffs_payload(year)
        if payload is None:
            logger.warning("No playoffs payload for season=%s; skipping playoffs artifact publish", year)
            return
        load_dotenv()
        client = get_r2_client()
        if client is None:
            logger.warning("R2 not configured, skipping playoffs artifact publish")
            return
        bucket = os.getenv("R2_BUCKET_NAME")
        if not bucket:
            logger.warning("R2_BUCKET_NAME not configured, skipping playoffs artifact publish")
            return
        key = f"playoffs/{year}/latest.json"
        if upload_json(client, bucket, key, payload):
            logger.info(PUBLISHED_LOG_MESSAGE, key)
        else:
            logger.warning("Playoffs artifact publish failed for R2 key %s", key)
        refresh_playoffs_index()
    except Exception as e:
        logger.exception("publish_playoffs_artifact failed unexpectedly for year=%s: %s", year, e)
