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
    _head_to_head_first_conferences,
    _resolve_conference_championship_outcomes,
    _to_datetime,
    identify_army_navy_game,
    identify_conference_championship_games,
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
        "round_notes": {ROUND_FIRST: "Campus Sites"},
    },
    2026: {
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
) -> Dict[str, Dict[str, Any]]:
    """
    Champion (actual or projected) of each conference in `conferences` (RAW names).

    Owner's rule, taken literally:
      1. a conference whose title game has been PLAYED is decided: `actual_champions[conf]` wins
         outright, whatever the standings or ranks say;
      2. otherwise the candidates are that conference's ranked teams. When exactly two of them
         have CLINCHED a title-game berth (championship_status == "clinched") the candidates are
         restricted to those two;
      3. among the candidates, the best conference win percentage leads; ties at the top go to
         the team ranked highest by OUR model -- never head-to-head.

    `ranks`: team -> model rank (1 = best); a team missing from it is unranked and never a
    projected champion. `standings`: schedule_standings.compute_standings output.
    Returns {raw conference -> {"team": str, "actual": bool}}; a conference with no actual
    champion and no ranked candidate is absent.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for conference in conferences:
        actual = actual_champions.get(conference)
        if actual:
            result[conference] = {"team": actual, "actual": True}
            continue
        candidates = [
            team for team, st in standings.items()
            if st.get("conference") == conference and st.get("conf_record") is not None and team in ranks
        ]
        clinched = [t for t in candidates if standings[t].get("championship_status") == "clinched"]
        if len(clinched) == 2:
            candidates = clinched
        if not candidates:
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

    seeds: Dict[str, int] = {}
    for row in rows:
        if row.get("season") != season or not isinstance(row.get("playoff_round_name"), str):
            continue
        for team_key, seed_key in (("team", "team_seed"), ("opponent", "opp_seed")):
            team, seed = row.get(team_key), _int_or_none(row.get(seed_key))
            if team and seed is not None and seed in SEED_QUARTER:
                if seeds.setdefault(team, seed) != seed:
                    logger.warning("playoffs: %s has conflicting CFP seeds %s and %s", team, seeds[team], seed)
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


def _season_standings(
    rows: List[Dict[str, Any]], teams_meta: Dict[str, Dict[str, Any]], season: int
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """(standings, actual conference champions), computed exactly as build_schedule_payload does
    for the Season Grid (same Army-Navy exclusion, played-title-game override and head-to-head
    clinch inputs), so the status and records used here are the ones the grid shows."""
    champ_games = identify_conference_championship_games(rows, season)
    champions, losers = _resolve_conference_championship_outcomes(champ_games, rows, season)
    participants: Dict[str, Set[str]] = {}
    for conference, team in list(champions.items()) + list(losers.items()):
        participants.setdefault(conference, set()).add(team)
    army_navy = identify_army_navy_game(rows, season)
    standings = schedule_standings.compute_standings(
        rows, season,
        divisions={team: meta.get("division") for team, meta in teams_meta.items()},
        excluded_conference_game_ids={army_navy} if army_navy is not None else None,
        ccg_participants=participants,
        head_to_head_first_conferences=_head_to_head_first_conferences(
            {r["conference"] for r in rows if r.get("conference")}, season
        ),
    )
    return standings, champions


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
    season with no stored CFP games. Pure -- no I/O.
    """
    rules = SEASON_RULES.get(season)
    if rules is None:
        logger.info("playoffs: season %s has no playoff rules configured; nothing built.", season)
        return None
    team_ranks = {t: r for t, r in (team_ranks or {}).items() if t in teams_meta}
    standings, actual_champions = _season_standings(rows, teams_meta, season)

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
            "bid": bid, "bid_label": bid_label, "bye": seed <= 4,
        }

    if rules["kind"] == "actual":
        bracket = build_actual_bracket(rows, season)
        if bracket is None:
            logger.warning("playoffs: no CFP games stored for season=%s; nothing built.", season)
            return None
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

    champions = resolve_projected_champions(team_ranks, standings, actual_champions, rules["aq_conferences"])
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
