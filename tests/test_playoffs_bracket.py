"""Actual-bracket reconstruction and payload shape (artifacts/playoffs.py), pure -- no DB/R2.

The fixture is the real 2025 CFP data as stored in `games`/`schedule_grid` (verified live
2026-11-23): CFBD's slot labels (FR1..FR4, QF1..QF4, SF1, SF2) are NOT in bracket order, so the
expected geometry below is derived from seeds only.

MUTATION CHECK (run by hand; each mutant failed the named test, then reverted):
  - FR order from CFBD slot label instead of seeds        -> test_2025_games_follow_contract_geometry
  - QF order fixed to 0 instead of the quarter of the seeds -> test_2025_games_follow_contract_geometry
  - SF order by feeder QF index swapped (q // 2 -> 1 - q // 2) -> test_2025_games_follow_contract_geometry
  - SF/CH sides left in input order instead of by feeder  -> test_2025_games_follow_contract_geometry
  - winner flipped                                        -> test_2025_games_follow_contract_geometry
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifacts.playoffs import (  # noqa: E402
    SEASON_RULES,
    build_actual_bracket,
    build_playoffs_payload,
)

# (game_id, cfbd slot, round, bowl, away (seed, score), home (seed, score), UTC kickoff)
_GAMES = [
    (4, "FR4", "First Round", "College Football Playoff First Round Game", ("Alabama", 9, 34), ("Oklahoma", 8, 24), "2025-12-20T01:00"),
    (3, "FR3", "First Round", "College Football Playoff First Round Game", ("Miami", 10, 10), ("Texas A&M", 7, 3), "2025-12-20T17:00"),
    (2, "FR2", "First Round", "College Football Playoff First Round Game", ("Tulane", 11, 10), ("Ole Miss", 6, 41), "2025-12-20T21:00"),
    (1, "FR1", "First Round", "College Football Playoff First Round Game", ("James Madison", 12, 34), ("Oregon", 5, 51), "2025-12-21T01:00"),
    (13, "QF3", "Quarterfinal", "Cotton Bowl", ("Miami", 10, 24), ("Ohio State", 2, 14), "2026-01-01T01:00"),
    (12, "QF2", "Quarterfinal", "Orange Bowl", ("Oregon", 5, 23), ("Texas Tech", 4, 0), "2026-01-01T01:00"),
    (11, "QF1", "Quarterfinal", "Rose Bowl", ("Alabama", 9, 3), ("Indiana", 1, 38), "2026-01-01T21:00"),
    (14, "QF4", "Quarterfinal", "Sugar Bowl", ("Ole Miss", 6, 39), ("Georgia", 3, 34), "2026-01-02T01:00"),
    (22, "SF2", "Semifinal", "Fiesta Bowl", ("Miami", 10, 31), ("Ole Miss", 6, 27), "2026-01-09T01:00"),
    (21, "SF1", "Semifinal", "Peach Bowl", ("Oregon", 5, 22), ("Indiana", 1, 56), "2026-01-10T01:00"),
    (30, "CH", "National Championship", "College Football Playoff National Championship Presented by AT&T",
     ("Miami", 10, 21), ("Indiana", 1, 27), "2026-01-20T00:30"),
]
_CONF = {
    "Alabama": "SEC", "Oklahoma": "SEC", "Miami": "ACC", "Texas A&M": "SEC", "Tulane": "American Athletic",
    "Ole Miss": "SEC", "James Madison": "Sun Belt", "Oregon": "Big Ten", "Ohio State": "Big Ten",
    "Texas Tech": "Big 12", "Indiana": "Big Ten", "Georgia": "SEC",
}


def _rows(games=_GAMES):
    rows = []
    for gid, slot, round_name, bowl, (away, aseed, ascore), (home, hseed, hscore), kickoff in games:
        start = datetime.fromisoformat(kickoff).replace(tzinfo=timezone.utc)
        played = ascore is not None
        base = dict(
            game_id=gid, season=2025, season_type="postseason", start_date=start,
            playoff_round_name=round_name, playoff_bracket_slot=slot, playoff_bowl_name=bowl, notes=bowl,
            conference_game=False,
        )
        for team, opp, score, oscore, seed, oseed, side in (
            (home, away, hscore, ascore, hseed, aseed, "home"),
            (away, home, ascore, hscore, aseed, hseed, "away"),
        ):
            rows.append(dict(
                base, team=team, opponent=opp, team_score=score, opp_score=oscore, team_seed=seed,
                opp_seed=oseed, home_away=side, conference=_CONF[team],
                status=("win" if score > oscore else "loss") if played else "upcoming",
            ))
    return rows


def _teams_meta():
    return {t: {"conference": c, "division": None, "logos": [f"https://x/logos-dark/32/{t}.png"]} for t, c in _CONF.items()}


def _games_by_slot(bracket):
    return {g["slot"]: g for g in bracket["games"]}


def _brief(game):
    return (
        game["slot"], game["order"], game["name"], game["status"],
        (game["top"]["seed"], game["top"]["team"], game["top"]["score"], game["top"]["from"]),
        (game["bottom"]["seed"], game["bottom"]["team"], game["bottom"]["score"], game["bottom"]["from"]),
        game["winner"],
    )


def test_2025_games_follow_contract_geometry():
    bracket = build_actual_bracket(_rows(), 2025)
    got = [_brief(g) for g in bracket["games"]]
    assert got == [
        ("FR1", 0, None, "final", (8, "Oklahoma", 24, None), (9, "Alabama", 34, None), "bottom"),
        ("FR2", 1, None, "final", (5, "Oregon", 51, None), (12, "James Madison", 34, None), "top"),
        ("FR3", 2, None, "final", (7, "Texas A&M", 3, None), (10, "Miami", 10, None), "bottom"),
        ("FR4", 3, None, "final", (6, "Ole Miss", 41, None), (11, "Tulane", 10, None), "top"),
        ("QF1", 0, "Rose Bowl", "final", (1, "Indiana", 38, None), (9, "Alabama", 3, "FR1"), "top"),
        ("QF2", 1, "Orange Bowl", "final", (4, "Texas Tech", 0, None), (5, "Oregon", 23, "FR2"), "bottom"),
        ("QF3", 2, "Cotton Bowl", "final", (2, "Ohio State", 14, None), (10, "Miami", 24, "FR3"), "bottom"),
        ("QF4", 3, "Sugar Bowl", "final", (3, "Georgia", 34, None), (6, "Ole Miss", 39, "FR4"), "bottom"),
        ("SF1", 0, "Peach Bowl", "final", (1, "Indiana", 56, "QF1"), (5, "Oregon", 22, "QF2"), "top"),
        ("SF2", 1, "Fiesta Bowl", "final", (10, "Miami", 31, "QF3"), (6, "Ole Miss", 27, "QF4"), "top"),
        ("CH", 0, "National Championship", "final", (1, "Indiana", 27, "SF1"), (10, "Miami", 21, "SF2"), "top"),
    ]
    assert [g["round"] for g in bracket["games"]] == ["first_round"] * 4 + ["quarterfinal"] * 4 + ["semifinal"] * 2 + ["championship"]


def test_2025_dates_are_football_days_and_round_labels():
    bracket = build_actual_bracket(_rows(), 2025)
    by_slot = _games_by_slot(bracket)
    assert by_slot["FR1"]["date"] == "2025-12-19"  # Alabama @ Oklahoma, 8pm ET Friday
    assert by_slot["CH"]["date"] == "2026-01-19"
    payload = build_playoffs_payload(_rows(), _teams_meta(), 2025, {"Indiana": 1})
    labels = {r["round"]: r["dates_label"] for r in payload["rounds"]}
    assert labels["first_round"] == "Dec 19-20"
    assert labels["quarterfinal"] == "Dec 31 - Jan 1"
    assert labels["championship"] == "Jan 19"


def test_2025_payload_shape_and_champion():
    ranks = {t: i for i, t in enumerate(_CONF, start=1)}
    payload = build_playoffs_payload(_rows(), _teams_meta(), 2025, ranks)
    assert set(payload) == {"season", "kind", "as_of_week", "generated_at_utc", "teams", "rounds", "games", "first_out"}
    assert payload["kind"] == "actual" and payload["as_of_week"] is None and payload["first_out"] == []
    assert [r["round"] for r in payload["rounds"]] == ["first_round", "quarterfinal", "semifinal", "championship"]
    assert len(payload["teams"]) == 12 and len(payload["games"]) == 11
    assert [t["seed"] for t in payload["teams"]] == list(range(1, 13))
    assert [t["bye"] for t in payload["teams"]] == [True] * 4 + [False] * 8
    ch = _games_by_slot(payload)["CH"]
    assert ch["top"]["team"] == "Indiana" and ch["winner"] == "top"
    indiana = payload["teams"][0]
    assert indiana["team"] == "Indiana" and indiana["conference"] == "BIG 10"
    assert indiana["logo_url"].endswith("Indiana.png") and indiana["rank"] == ranks["Indiana"]
    # Records come from every stored row: Indiana is 3-0 in these fixture rows (no regular season stored).
    assert indiana["record"] == "3-0"
    # No title games stored, so no champion can be asserted and no "At-Large" label guessed.
    assert all(t["bid"] == "at_large" and t["bid_label"] is None for t in payload["teams"])


def test_actual_champion_bid_derived_only_from_played_title_game():
    rows = _rows()
    # A played Big Ten title game, which Indiana won.
    for team, opp, score, oscore, status in (("Indiana", "Ohio State", 13, 10, "win"), ("Ohio State", "Indiana", 10, 13, "loss")):
        rows.append(dict(
            game_id=900, season=2025, season_type="regular", week=15, team=team, opponent=opp,
            conference="Big Ten", home_away="home", team_score=score, opp_score=oscore, status=status,
            conference_game=True, notes="Big Ten Championship", neutral_site=True,
            start_date=datetime(2025, 12, 7, 0, 30, tzinfo=timezone.utc), playoff_round_name=None,
        ))
    payload = build_playoffs_payload(rows, _teams_meta(), 2025, {})
    teams = {t["team"]: t for t in payload["teams"]}
    assert teams["Indiana"]["bid"] == "champion" and teams["Indiana"]["bid_label"] == "Big Ten Champion"
    # Ohio State is in a conference whose champion is known and is someone else: provably at-large.
    assert teams["Ohio State"]["bid"] == "at_large" and teams["Ohio State"]["bid_label"] == "At-Large"


def test_incomplete_bracket_still_builds_with_placeholders():
    first_round_only = [g for g in _GAMES if g[2] == "First Round"]
    bracket = build_actual_bracket(_rows(first_round_only), 2025)
    assert len(bracket["games"]) == 11
    by_slot = _games_by_slot(bracket)
    assert by_slot["FR2"]["top"]["team"] == "Oregon" and by_slot["FR2"]["winner"] == "top"
    qf = by_slot["QF1"]
    assert qf["status"] == "scheduled" and qf["bottom"] == {"seed": None, "team": None, "score": None, "from": "FR1"}
    assert by_slot["SF1"]["top"] == {"seed": None, "team": None, "score": None, "from": "QF1"}
    assert by_slot["CH"]["bottom"]["from"] == "SF2"


def test_unplayed_game_is_scheduled_without_winner():
    upcoming = [g[:4] + (g[4][:2] + (None,), g[5][:2] + (None,), g[6]) if g[1] == "CH" else g for g in _GAMES]
    ch = _games_by_slot(build_actual_bracket(_rows(upcoming), 2025))["CH"]
    assert ch["status"] == "scheduled" and ch["winner"] is None
    assert ch["top"]["score"] is None and ch["top"]["team"] == "Indiana"


def test_no_cfp_rows_returns_none():
    assert build_actual_bracket([], 2025) is None
    assert build_playoffs_payload([], _teams_meta(), 2025, {}) is None


def test_unsupported_season_returns_none():
    assert build_playoffs_payload(_rows(), _teams_meta(), 2027, {"Indiana": 1}) is None
    assert 2027 not in SEASON_RULES and 2024 not in SEASON_RULES


def _projected_league():
    teams_meta, ranks = {}, {}
    plan = (
        [("Big 12", f"B12-{i}") for i in range(4)] + [("SEC", f"SEC-{i}") for i in range(4)]
        + [("Big Ten", f"B10-{i}") for i in range(4)] + [("ACC", f"ACC-{i}") for i in range(4)]
        + [("Mountain West", "MW-0"), ("FBS Independents", "Notre Dame")]
    )
    for rank, (conf, name) in enumerate(plan, start=1):
        teams_meta[name] = {"conference": conf, "division": None, "logos": None}
        ranks[name] = rank
    return teams_meta, ranks


def test_2026_projected_payload_shape():
    teams_meta, ranks = _projected_league()
    payload = build_playoffs_payload([], teams_meta, 2026, ranks, as_of_week=12)
    assert payload["kind"] == "projected" and payload["as_of_week"] == 12
    assert len(payload["teams"]) == 12 and [t["seed"] for t in payload["teams"]] == list(range(1, 13))
    assert [t["rank"] for t in payload["teams"]] == sorted(t["rank"] for t in payload["teams"])
    assert {"seed", "team", "rank", "record", "conference", "logo_url", "bid", "bid_label", "bye"} == set(payload["teams"][0])
    assert [r["dates_label"] for r in payload["rounds"]] == ["Dec 18-19", "Dec 30 - Jan 1", "Jan 14-15", "Jan 25"]
    games = payload["games"]
    assert len(games) == 11 and all(g["status"] == "projected" and g["name"] is None and g["date"] is None for g in games)
    assert _brief(games[0])[4:6] == ((8, payload["teams"][7]["team"], None, None), (9, payload["teams"][8]["team"], None, None))
    qf1 = games[4]
    assert qf1["top"]["seed"] == 1 and qf1["bottom"] == {"seed": None, "team": None, "score": None, "from": "FR1"}
    assert [g["bottom"]["from"] for g in games[8:]] == ["QF2", "QF4", "SF2"]
    assert len(payload["first_out"]) == 4 and not {f["team"] for f in payload["first_out"]} & {t["team"] for t in payload["teams"]}
    assert payload["first_out"][0]["rank"] == min(f["rank"] for f in payload["first_out"])


def test_2026_projected_short_field_returns_none():
    teams_meta, ranks = _projected_league()
    short = dict(list(ranks.items())[:8])
    assert build_playoffs_payload([], teams_meta, 2026, short) is None
