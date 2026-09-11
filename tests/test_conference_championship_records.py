"""Tests for T1 (conference records exclude the conference championship game) and T2
(conference champions sort first), per docs/season-grid-postseason-format/plan.yaml batch B1.

The conference-record exclusion (T1) had ZERO characterization before this file -- neither the
happy path nor the K1 regression it guards against. Every fixture below is invented; real team
names (Duke, Virginia, the Sun Belt's App State/Coastal Carolina/Troy/Arkansas State) are used
only where a test is reproducing a specific documented live case, never because the win/loss
numbers describe an actual season.

Covers:
  1. Duke 6-2 / Virginia 7-1 for a 2025-shaped ACC fixture (the exact live case
     docs/season-grid-postseason-format/requirements.yaml verified), and overall win/loss
     records left untouched by the exclusion.
  2. The championship game's winner (Duke) sorts first despite a WORSE displayed conference
     percentage than the team it beat (Virginia) -- K3.
  3. An identified-but-UNPLAYED championship game changes no team's championship_status (the K1
     regression) and crowns no champion, so the conference sorts exactly as it would with no
     knowledge of championships at all.
  4. A champion that lost the EARLIER regular-season head-to-head to the team it is now tied
     with on conference record still sorts first -- K4, the head-to-head grouping fix.
  5. A divisional conference (Sun Belt shape) whose champion has the WORSE in-division record
     leads its own division only; the other division's block is untouched -- K9.
  6. A conference with no identified championship game, mid-season: ordering identical to a
     plain re-sort that has never heard of championships -- AC3.
  7. Independents: no conference record, no championship status, unaffected by any of this --
     AC4.
  8. `_is_champion` never appears anywhere in the published payload -- K10.

Note: tests/test_division_championship_status.py's Sun Belt fixture was deliberately built to
AVOID identifying a championship game (see that file's own comments), so it cannot be reused or
lightly edited for case 5 above -- a fresh fixture that DOES trip
identify_conference_championship_games' "lone game, later than the regular slate" rule is needed.

MUTATION-CHECK LOG (defect injected into a working tree copy of artifacts/schedule.py, target
test re-run to confirm it fails, change reverted -- see the task report for the full table):
  every test function below was checked this way at least once.

Run: python -m pytest tests/ -q
     (or: python tests/test_conference_championship_records.py)
"""
import itertools
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifacts import schedule_standings  # noqa: E402
from artifacts.schedule import (  # noqa: E402
    CONF_CHAMPIONSHIP_SLOT_ID,
    _exclude_championship_games_from_conf_records,
    _head_to_head_winner,
    _placement_pct,
    _resolve_conference_champions,
    _sort_conference_teams,
    build_schedule_payload,
)

SEASON = 2025
ACC = "ACC"
SUN_BELT = "Sun Belt"

# A Saturday that lands in get_cfb_week's week-0 bucket for the 2025 anchor (Tuesday on/before
# Aug 24, 2025 = Aug 19). Every _row() below defaults week_offset to game_id - 1, so sequential
# game_ids land in sequential, strictly-increasing week buckets up to (and including) the
# week-16 cap -- exactly the "lone game, later than the regular slate" shape
# identify_conference_championship_games looks for, without needing an explicit same-bucket
# trick unless a fixture deliberately wants one (see the no-championship-identified fixture).
_SEASON_START = date(2025, 8, 23)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------
def _row(game_id, team, opponent, status, conference, week_offset=None, neutral_site=False):
    """One team-oriented schedule_grid row -- only the columns compute_team_records,
    identify_conference_championship_games and build_schedule_payload actually read."""
    offset = game_id - 1 if week_offset is None else week_offset
    return dict(
        game_id=game_id,
        season=SEASON,
        season_type="regular",
        team=team,
        opponent=opponent,
        conference=conference,
        conference_game=True,
        status=status,
        start_date=(_SEASON_START + timedelta(days=7 * offset)).isoformat() + " 19:00:00",
        home_away="home",
        neutral_site=neutral_site,
    )


def _played(game_id, winner, loser, conference, week_offset=None, neutral_site=False):
    """Both team-oriented rows for one completed conference game."""
    return [
        _row(game_id, winner, loser, "win", conference, week_offset=week_offset, neutral_site=neutral_site),
        _row(game_id, loser, winner, "loss", conference, week_offset=week_offset, neutral_site=neutral_site),
    ]


def _scheduled(game_id, team_a, team_b, conference, week_offset=None, neutral_site=False):
    """Both team-oriented rows for a not-yet-played conference game (status='upcoming')."""
    return [
        _row(game_id, team_a, team_b, "upcoming", conference, week_offset=week_offset, neutral_site=neutral_site),
        _row(game_id, team_b, team_a, "upcoming", conference, week_offset=week_offset, neutral_site=neutral_site),
    ]


def _teams_meta(teams, conference, divisions=None):
    divisions = divisions or {}
    return {t: {"conference": conference, "division": divisions.get(t), "logos": None} for t in teams}


def _conf_entries(payload, display_name):
    for conf in payload["conferences"]:
        if conf["name"] == display_name:
            return conf["teams"]
    raise AssertionError(
        f"{display_name!r} not present in payload conferences: "
        f"{[c['name'] for c in payload['conferences']]}"
    )


def _conf_names(payload, display_name):
    return [e["team"] for e in _conf_entries(payload, display_name)]


def _cell(entry, slot_id):
    return next(w for w in entry["weeks"] if w["slot_id"] == slot_id)


# ---------------------------------------------------------------------------
# Cases 1 & 2: the live Duke/Virginia 2025 case (T1's exclusion, T2's champion sort)
# ---------------------------------------------------------------------------
# 8 lightly-used opponents. Their own records are never asserted on -- they exist only to give
# Duke and Virginia their 8-game non-championship conference slates each, and to satisfy
# identify_conference_championship_games' >=4-distinct-member gate.
_FILLERS = [f"ACC Filler {i}" for i in range(1, 9)]


def _duke_virginia_2025_rows():
    """
    A 2025-shaped ACC fixture reproducing the exact live case
    docs/season-grid-postseason-format/requirements.yaml verified: Duke and Virginia each play
    an 8-game non-championship conference slate, then meet in the ACC title game (game_id 17,
    alone in the latest week bucket), which Duke wins.

    Duke's other 8 games: 6-2. Virginia's other 8 games: 7-1. Counting the championship game
    (today's bug), both read 7-2 -- exactly "Both teams currently show 7-2 in conference" from
    the requirements. Excluding it (T1) must give Duke 6-2, Virginia 7-1, while each team's
    OVERALL record (which does count the championship game) stays 7-2.
    """
    rows = []
    game_id = 1
    # Duke: 6 wins (vs Filler 1-6), then 2 losses (vs Filler 7-8).
    for opponent in _FILLERS[:6]:
        rows += _played(game_id, "Duke", opponent, ACC)
        game_id += 1
    for opponent in _FILLERS[6:]:
        rows += _played(game_id, opponent, "Duke", ACC)
        game_id += 1
    # Virginia: 7 wins (vs Filler 1-7), then 1 loss (vs Filler 8).
    for opponent in _FILLERS[:7]:
        rows += _played(game_id, "Virginia", opponent, ACC)
        game_id += 1
    rows += _played(game_id, _FILLERS[7], "Virginia", ACC)
    game_id += 1
    # ACC Championship: Duke over Virginia, neutral site -- alone in the latest week bucket.
    rows += _played(game_id, "Duke", "Virginia", ACC, neutral_site=True)
    return rows


def _duke_virginia_teams_meta():
    return _teams_meta(["Duke", "Virginia"] + _FILLERS, ACC)


def test_duke_virginia_2025_conf_record_excludes_championship_game():
    rows = _duke_virginia_2025_rows()
    payload = build_schedule_payload(rows, _duke_virginia_teams_meta(), SEASON)
    entries = {e["team"]: e for e in _conf_entries(payload, "ACC")}

    assert entries["Duke"]["conf_record"] == {"wins": 6, "losses": 2}, entries["Duke"]["conf_record"]
    assert entries["Virginia"]["conf_record"] == {"wins": 7, "losses": 1}, entries["Virginia"]["conf_record"]

    # Overall win/loss must be untouched by the exclusion -- both are 7-2 including the
    # championship game, matching the live 2025 numbers exactly.
    assert entries["Duke"]["record"] == {"wins": 7, "losses": 2}, entries["Duke"]["record"]
    assert entries["Virginia"]["record"] == {"wins": 7, "losses": 2}, entries["Virginia"]["record"]


def test_champion_sorts_above_better_conference_record():
    """K3/AC1: Duke -- the actual game's winner -- must sort above Virginia despite Virginia's
    better DISPLAYED conference percentage (7-1 = .875 vs Duke's 6-2 = .750), because Duke is
    the identified championship game's resolved winner."""
    rows = _duke_virginia_2025_rows()
    payload = build_schedule_payload(rows, _duke_virginia_teams_meta(), SEASON)
    names = _conf_names(payload, "ACC")

    assert names.index("Duke") < names.index("Virginia"), names


# ---------------------------------------------------------------------------
# Case 3: identified-but-UNPLAYED championship game (the K1 regression)
# ---------------------------------------------------------------------------
_K1_FILLERS = [f"K1 Filler {i}" for i in range(1, 15)]


def _k1_unplayed_championship_rows():
    """
    Boundary-exact regression fixture for K1. Duke and Virginia are the ACC's two unbeaten
    teams (3-0 conference, 1 game left -- the still-UNPLAYED ACC championship against each
    other). TeamC and TeamD have ALREADY banked 4 conference wins apiece with none left to play.

    Chosen so the CORRECT math sits exactly on the elimination boundary:
        B_Duke = W_Duke + R_Duke = 3 + 1 = 4; the 2nd-highest OTHER banked-win total in the
        4-team pool is also 4 (TeamC and TeamD) -- 4 < 4 is False, so Duke/Virginia are
        correctly "possible", never eliminated.
    The K1 bug (excluding the still-unplayed championship row from the TALLY compute_standings
    feeds compute_conference_championship_status, instead of only from the DISPLAYED conf_record
    afterward) would shrink R_Duke/R_Virginia to 0, making B_Duke = 3 < 4 -- wrongly
    "Eliminated", during championship week, exactly the defect K1 exists to prevent.

    Because Duke and Virginia are themselves playing in the identified game, their OWN
    conf-championship cell renders the real (still-upcoming) game, not a computed status -- so
    TeamC/TeamD, bystanders who are NOT in the game, are what actually surface the regression:
    the bug shrinks Duke/Virginia's best-case win total from 4 to 3, which flips TeamC/TeamD
    from correctly "possible" to a premature "clinched" (nobody else can reach their own banked
    4 wins once Duke/Virginia can supposedly no longer reach it either).
    """
    rows = []
    game_id = 1
    for opponent in _K1_FILLERS[0:3]:
        rows += _played(game_id, "Duke", opponent, ACC)
        game_id += 1
    for opponent in _K1_FILLERS[3:6]:
        rows += _played(game_id, "Virginia", opponent, ACC)
        game_id += 1
    for opponent in _K1_FILLERS[6:10]:
        rows += _played(game_id, "TeamC", opponent, ACC)
        game_id += 1
    for opponent in _K1_FILLERS[10:14]:
        rows += _played(game_id, "TeamD", opponent, ACC)
        game_id += 1
    # The ACC championship: NOT yet played. identify_conference_championship_games does not
    # gate on status, so this is still identified (alone in the latest week bucket).
    rows += _scheduled(game_id, "Duke", "Virginia", ACC)
    return rows


def _k1_teams_meta():
    return _teams_meta(["Duke", "Virginia", "TeamC", "TeamD"] + _K1_FILLERS, ACC)


def test_identified_but_unplayed_championship_game_does_not_change_status():
    rows = _k1_unplayed_championship_rows()

    # The math itself, independent of schedule.py's wiring: compute_standings must see the
    # UNMODIFIED rows (the still-unplayed championship game counted normally in
    # conf_games_remaining), so nobody in the pool is wrongly eliminated or prematurely clinched.
    direct = schedule_standings.compute_standings(rows, SEASON)
    for team in ("Duke", "Virginia", "TeamC", "TeamD"):
        assert direct[team]["championship_status"] == "possible", (team, direct[team])

    payload = build_schedule_payload(rows, _k1_teams_meta(), SEASON)
    entries = {e["team"]: e for e in _conf_entries(payload, "ACC")}

    # Fix-cycle-1: pin the DISPLAYED conf_record itself, not just championship_status -- an
    # unplayed championship game must leave conf_record identical to what compute_standings
    # produced, for both participants. Widening the win/loss gate in
    # _exclude_championship_games_from_conf_records to also match status='upcoming' passed every
    # existing assertion in this test (championship_status alone doesn't catch it) while actually
    # producing a negative loss count, e.g. {"wins": 3, "losses": -1}, during championship week.
    for team in ("Duke", "Virginia"):
        assert entries[team]["conf_record"] == direct[team]["conf_record"], (team, entries[team]["conf_record"])
        assert entries[team]["conf_record"] == {"wins": 3, "losses": 0}, entries[team]["conf_record"]

    # Duke/Virginia are playing IN the identified game, so their own cell is the real
    # (still-upcoming) game, not a placeholder status.
    assert _cell(entries["Duke"], CONF_CHAMPIONSHIP_SLOT_ID)["status"] == "upcoming"
    assert _cell(entries["Virginia"], CONF_CHAMPIONSHIP_SLOT_ID)["status"] == "upcoming"

    # TeamC/TeamD are the bystanders that actually surface the K1 regression (see the fixture's
    # docstring): they must still read "possible", never a premature "clinched".
    assert _cell(entries["TeamC"], CONF_CHAMPIONSHIP_SLOT_ID)["status"] == "possible", \
        _cell(entries["TeamC"], CONF_CHAMPIONSHIP_SLOT_ID)
    assert _cell(entries["TeamD"], CONF_CHAMPIONSHIP_SLOT_ID)["status"] == "possible", \
        _cell(entries["TeamD"], CONF_CHAMPIONSHIP_SLOT_ID)

    # K3: an identified-but-unplayed game crowns no champion, so the conference sorts exactly as
    # a plain re-sort that has never heard of championships would.
    baseline_entries = [
        {"team": e["team"], "record": e["record"], "conf_record": e["conf_record"]}
        for e in entries.values()
    ]
    baseline_order = [e["team"] for e in _sort_conference_teams(baseline_entries, rows, SEASON)]
    assert _conf_names(payload, "ACC") == baseline_order


def test_exclude_championship_games_ignores_a_non_conference_game_row():
    """Fix-cycle-1 hardening: _exclude_championship_games_from_conf_records now also requires
    conference_game=True on the row itself, duplicating the invariant
    identify_conference_championship_games already enforces when it builds champ_game_ids.
    Structurally unreachable through the real pipeline today (every row sharing a champ_game_ids
    game_id already IS a conference game, since both perspectives of one game share a
    conference_game value) -- so exercised directly against the function, the only way to
    construct the case: a row that shares a champ_game_ids game_id but carries
    conference_game=False must not be subtracted."""
    standings = {"Duke": {"conf_record": {"wins": 5, "losses": 1}}}
    rows = [dict(season=SEASON, game_id=99, team="Duke", status="win", conference_game=False)]
    _exclude_championship_games_from_conf_records(standings, rows, SEASON, {99})
    assert standings["Duke"]["conf_record"] == {"wins": 5, "losses": 1}


def test_resolve_conference_champions_warns_on_a_shared_game_id_collision(caplog):
    """Fix-cycle-1 hardening: champ_games_by_conf (Dict[conference -> game_id]) inverted into
    Dict[game_id -> conference] is silently LOSSY if two different conferences were ever
    identified against the SAME game_id -- should be structurally impossible (a conference_game
    row belongs to exactly one conference), but every neighbouring 'shouldn't happen' case in
    this file logs rather than silently proceeding. Exercised directly against the function,
    since the real pipeline cannot construct this input."""
    champ_games_by_conf = {"ACC": 1, "SEC": 1}
    rows = [dict(season=SEASON, game_id=1, team="Duke", status="win")]
    with caplog.at_level("WARNING"):
        champions = _resolve_conference_champions(champ_games_by_conf, rows, SEASON)
    assert any("shared" in record.getMessage() for record in caplog.records), caplog.records
    # Only one conference actually gets a champion out of this -- the drop itself is not fixed,
    # only now logged, per the coordinator's explicit ask.
    assert len(champions) == 1


# ---------------------------------------------------------------------------
# Case 4: champion tied with, and previously beaten by, the same team (K4)
# ---------------------------------------------------------------------------
def _k4_head_to_head_rows():
    """
    Alpha and Beta meet TWICE: an earlier regular-season game Beta wins, and later the ACC
    championship (alone in the latest bucket), which Alpha wins -- making Alpha the champion.
    Gamma and Delta exist ONLY to give Alpha a compensating win and Beta a compensating loss (so
    their DISPLAYED, post-exclusion conference records tie at 1-1) and to satisfy
    identify_conference_championship_games' >=4-distinct-member gate. They are deliberately left
    OUT of teams_meta (see _k4_teams_meta) so they never become their own sorted ACC entries,
    keeping Alpha and Beta the only two rows in the conference and therefore guaranteed adjacent
    after sorting -- which is what actually exercises the head-to-head grouping logic below.
    """
    rows = []
    rows += _played(1, "Beta", "Alpha", ACC)         # earlier regular-season meeting: Beta wins
    rows += _played(2, "Alpha", "Gamma", ACC)         # Alpha's compensating win
    rows += _played(3, "Delta", "Beta", ACC)          # Beta's compensating loss
    # ACC Championship: Alpha over Beta -- alone in the latest bucket.
    rows += _played(4, "Alpha", "Beta", ACC, neutral_site=True)
    return rows


def _k4_teams_meta():
    # Deliberately Alpha/Beta only -- see the fixture's docstring.
    return _teams_meta(["Alpha", "Beta"], ACC)


def test_champion_not_swapped_below_team_that_beat_it_head_to_head():
    rows = _k4_head_to_head_rows()
    payload = build_schedule_payload(rows, _k4_teams_meta(), SEASON)
    entries = {e["team"]: e for e in _conf_entries(payload, "ACC")}

    # Sanity-check the tie the test depends on.
    assert entries["Alpha"]["conf_record"] == {"wins": 1, "losses": 1}, entries["Alpha"]["conf_record"]
    assert entries["Beta"]["conf_record"] == {"wins": 1, "losses": 1}, entries["Beta"]["conf_record"]

    names = _conf_names(payload, "ACC")
    assert names == ["Alpha", "Beta"], (
        f"got {names}. Alpha is the actual champion (it won the LATER championship meeting) and "
        "must sort first despite Beta having won their EARLIER regular-season meeting. If this "
        "reads ['Beta', 'Alpha'] instead, the head-to-head tiebreak grouped a champion with a "
        "team tied on conference record without also requiring matching _is_champion (K4), and "
        "the group-of-two swap fired on their earlier meeting -- silently undoing the "
        "champion-first rule from the sort key above it."
    )


# ---------------------------------------------------------------------------
# Case 4b: _is_champion in the grouping key can split a three-way tie the pure-percentage
# grouping would otherwise protect (fix-cycle-1: documented in _sort_conference_teams, pinned
# here -- see the long comment there for the full mechanics).
# ---------------------------------------------------------------------------
def _three_way_tie_entries(champion_team=None):
    """A, B, C all tied 1-1 in conference record -- a group of exactly THREE on conf_pct, which
    _sort_conference_teams' grouping normally protects from the head-to-head swap (the swap only
    fires on a group of exactly two). champion_team, if given, marks one of them _is_champion."""
    return [
        {"team": t, "record": {"wins": 1, "losses": 1}, "conf_record": {"wins": 1, "losses": 1},
         "_is_champion": (t == champion_team)}
        for t in ("A", "B", "C")
    ]


def _three_way_tie_rows():
    # The only meeting _sort_conference_teams' head-to-head tiebreak can see: B beat A,
    # head-to-head, in a regular-season conference game.
    return _played(1, "B", "A", ACC)


def test_three_way_tie_with_no_champion_sorts_by_name_not_head_to_head():
    """Baseline: with no champion, all three group together (three-way, not two), so the
    B-beat-A head-to-head result never gets a chance to fire -- falls back to name order."""
    entries = _three_way_tie_entries()
    rows = _three_way_tie_rows()
    sorted_teams = [e["team"] for e in _sort_conference_teams(entries, rows, SEASON)]
    assert sorted_teams == ["A", "B", "C"], sorted_teams


def test_three_way_tie_with_a_champion_splits_the_group_and_exposes_head_to_head():
    """C as champion sorts first, alone (a unique True never groups with a False). That leaves
    A and B -- the two it excludes from its own group -- as a group of exactly two, which
    re-enables THEIR head-to-head swap: B beat A, so B now sorts above A. Different final order
    from the no-champion baseline above, from the SAME underlying record and head-to-head
    result -- a direct, documented consequence of narrowing the grouping key (see the comment in
    _sort_conference_teams), not a bug in the champion-first rule itself."""
    entries = _three_way_tie_entries(champion_team="C")
    rows = _three_way_tie_rows()
    sorted_teams = [e["team"] for e in _sort_conference_teams(entries, rows, SEASON)]
    assert sorted_teams == ["C", "B", "A"], sorted_teams


# ---------------------------------------------------------------------------
# Case 5: divisional champion leads only its own division (K9)
# ---------------------------------------------------------------------------
_EAST = ["App State", "Coastal Carolina"]
_WEST = ["Troy", "Arkansas State"]
_SUN_BELT_DIVISIONS = {**{t: "East" for t in _EAST}, **{t: "West" for t in _WEST}}


def _sun_belt_divisional_champion_rows():
    """
    A minimal Sun Belt fixture where the WEST champion (Troy) has the WORSE in-division record
    of the two West teams -- Arkansas State actually beat Troy in the regular season -- yet Troy
    is the team that wins the later championship game outright. This isolates K9: the champion
    tier must promote Troy to the top of its OWN division (West) only, while the East block
    (App State, Coastal Carolina) is untouched and still precedes West entirely -- the division
    enumeration/concatenation itself is out of scope for this batch, and this fixture would fail
    loudly if anything hoisted Troy above the East block instead of just to the top of West.
    """
    rows = []
    rows += _played(1, "App State", "Coastal Carolina", SUN_BELT)   # East, non-championship
    rows += _played(2, "Arkansas State", "Troy", SUN_BELT)          # West, non-championship
    # Sun Belt Championship: Troy (West) over App State (East) -- alone in the latest bucket.
    rows += _played(3, "Troy", "App State", SUN_BELT, neutral_site=True)
    return rows


def _sun_belt_teams_meta():
    return _teams_meta(_EAST + _WEST, SUN_BELT, divisions=_SUN_BELT_DIVISIONS)


def test_divisional_champion_leads_only_its_own_division():
    rows = _sun_belt_divisional_champion_rows()
    payload = build_schedule_payload(rows, _sun_belt_teams_meta(), SEASON)
    entries = {e["team"]: e for e in _conf_entries(payload, "SBC")}

    # Sanity-check the reversal the test depends on: Troy's DISPLAYED record is worse than
    # Arkansas State's, so if Troy still leads West it can only be because it is the champion.
    assert entries["Troy"]["conf_record"] == {"wins": 0, "losses": 1}, entries["Troy"]["conf_record"]
    assert entries["Arkansas State"]["conf_record"] == {"wins": 1, "losses": 0}, \
        entries["Arkansas State"]["conf_record"]

    names = _conf_names(payload, "SBC")
    assert names == ["App State", "Coastal Carolina", "Troy", "Arkansas State"], (
        f"got {names}. Expected the untouched East block (App State, Coastal Carolina) first, "
        "then the West block with Troy -- the actual champion, despite the worse in-division "
        "record -- leading it (K9): a divisional champion sorts first within its OWN division "
        "only, never hoisted above the other division's block."
    )


# ---------------------------------------------------------------------------
# Case 6: no identified championship game, in-progress season (AC3)
# ---------------------------------------------------------------------------
def _no_championship_identified_rows():
    """
    The shape identify_conference_championship_games is built to reject: the latest week bucket
    holds TWO distinct games (a make-up game sharing the weekend, or -- more commonly -- an
    in-progress season where nothing has separated out yet), so nothing is identified and the
    conference must sort exactly as it always has.
    """
    rows = []
    rows += _played(1, "A", "B", ACC)
    rows += _played(2, "C", "D", ACC)
    rows += _played(3, "A", "C", ACC)
    rows += _played(4, "B", "D", ACC, week_offset=2)  # deliberately shares game 3's bucket
    return rows


def test_conference_with_no_identified_championship_game_sorts_unchanged():
    rows = _no_championship_identified_rows()
    teams_meta = _teams_meta(["A", "B", "C", "D"], ACC)

    payload = build_schedule_payload(rows, teams_meta, SEASON)

    # No exclusion should have applied at all -- these are the raw, un-adjusted tallies.
    entries = {e["team"]: e for e in _conf_entries(payload, "ACC")}
    assert entries["A"]["conf_record"] == {"wins": 2, "losses": 0}
    assert entries["B"]["conf_record"] == {"wins": 1, "losses": 1}
    assert entries["C"]["conf_record"] == {"wins": 1, "losses": 1}
    assert entries["D"]["conf_record"] == {"wins": 0, "losses": 2}

    # Byte-identical to a sort that has never heard of championships: rebuild the same entries
    # WITHOUT an _is_champion key at all (exactly the pre-T2 shape) and re-sort independently.
    baseline = [
        {"team": e["team"], "record": e["record"], "conf_record": e["conf_record"]}
        for e in _conf_entries(payload, "ACC")
    ]
    baseline_order = [e["team"] for e in _sort_conference_teams(baseline, rows, SEASON)]
    assert _conf_names(payload, "ACC") == baseline_order == ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Case 7: Independents unaffected (AC4)
# ---------------------------------------------------------------------------
def _independents_rows():
    def _indep_row(game_id, team, opponent, status, week_offset):
        return dict(
            game_id=game_id,
            season=SEASON,
            season_type="regular",
            team=team,
            opponent=opponent,
            conference=schedule_standings.INDEPENDENT_CONFERENCE_VALUE,
            conference_game=False,
            status=status,
            start_date=(_SEASON_START + timedelta(days=7 * week_offset)).isoformat() + " 19:00:00",
            home_away="home",
            neutral_site=False,
        )

    return [
        _indep_row(1, "Indy Alpha", "Opponent X", "win", 0),
        _indep_row(2, "Indy Alpha", "Opponent Y", "win", 1),
        _indep_row(3, "Indy Beta", "Opponent Z", "loss", 0),
        _indep_row(4, "Indy Beta", "Opponent W", "loss", 1),
    ]


def test_independents_unaffected():
    rows = _independents_rows()
    teams_meta = _teams_meta(
        ["Indy Alpha", "Indy Beta"], schedule_standings.INDEPENDENT_CONFERENCE_VALUE
    )

    payload = build_schedule_payload(rows, teams_meta, SEASON)
    entries = {e["team"]: e for e in _conf_entries(payload, "FBS Independent")}

    assert entries["Indy Alpha"]["conf_record"] is None
    assert entries["Indy Beta"]["conf_record"] is None
    # No qualifying/divisional conference ever claims Independents, so championship_status is
    # always None, which renders as the plain 'bye' placeholder for that column.
    assert _cell(entries["Indy Alpha"], CONF_CHAMPIONSHIP_SLOT_ID)["status"] == "bye"
    assert _cell(entries["Indy Beta"], CONF_CHAMPIONSHIP_SLOT_ID)["status"] == "bye"

    names = _conf_names(payload, "FBS Independent")
    assert names == ["Indy Alpha", "Indy Beta"], (
        f"got {names}. Independents (2-0 and 0-2 overall) must still sort purely by overall "
        "record, exactly as before this batch."
    )

    # Defensive unit check on the T1 exclusion helper directly: even if a champion game_id were
    # -- incorrectly -- ever passed in for a team with no conference record at all, it must
    # never crash trying to subtract from a None conf_record.
    standings = {"Indy Alpha": {"conf_record": None}}
    _exclude_championship_games_from_conf_records(standings, rows, SEASON, champ_game_ids={1})
    assert standings["Indy Alpha"]["conf_record"] is None


# ---------------------------------------------------------------------------
# Case 8: _is_champion never leaks into the published payload (K10)
# ---------------------------------------------------------------------------
def _find_underscore_keys(obj, path=""):
    """Recursively collect every dict key starting with '_' anywhere in `obj`, with a path for
    a useful failure message. The published payload must contain none -- every underscore-
    prefixed field (_overall_pct, _conf_pct, _conf_played, _is_champion) is scratch state that
    _sort_conference_teams strips before its entries are returned."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key_path = f"{path}.{k}" if path else str(k)
            if isinstance(k, str) and k.startswith("_"):
                found.append(key_path)
            found += _find_underscore_keys(v, key_path)
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            found += _find_underscore_keys(v, f"{path}[{idx}]")
    return found


def test_is_champion_absent_from_published_payload():
    rows = _duke_virginia_2025_rows()
    payload = build_schedule_payload(rows, _duke_virginia_teams_meta(), SEASON)

    leaked = _find_underscore_keys(payload)
    assert leaked == [], (
        f"internal scratch field(s) leaked into the published payload: {leaked}. K10: "
        "_is_champion (and its neighbors _conf_pct/_conf_played/_overall_pct) are scratch state "
        "and must be popped/deleted before _sort_conference_teams returns its entries."
    )


# ---------------------------------------------------------------------------
# T2/K8: _head_to_head_winner direct tests -- the tallying fix for a split series.
# ---------------------------------------------------------------------------
def test_head_to_head_1_1_split_returns_none():
    """A split series is a WASH, not a tiebreak -- the exact defect (issue 7) this task fixes."""
    rows = _played(1, "Alpha", "Beta", ACC) + _played(2, "Beta", "Alpha", ACC)
    assert _head_to_head_winner(rows, SEASON, "Alpha", "Beta") is None


def test_head_to_head_2_2_split_returns_none():
    rows = (
        _played(1, "Alpha", "Beta", ACC)
        + _played(2, "Beta", "Alpha", ACC)
        + _played(3, "Alpha", "Beta", ACC)
        + _played(4, "Beta", "Alpha", ACC)
    )
    assert _head_to_head_winner(rows, SEASON, "Alpha", "Beta") is None


def test_head_to_head_2_0_sweep_returns_the_winner():
    rows = _played(1, "Alpha", "Beta", ACC) + _played(2, "Alpha", "Beta", ACC)
    assert _head_to_head_winner(rows, SEASON, "Alpha", "Beta") == "Alpha"
    # Order of the two teams passed in must not matter.
    assert _head_to_head_winner(rows, SEASON, "Beta", "Alpha") == "Alpha"


def test_head_to_head_no_meeting_returns_none():
    rows = _played(1, "Alpha", "Gamma", ACC) + _played(2, "Beta", "Delta", ACC)
    assert _head_to_head_winner(rows, SEASON, "Alpha", "Beta") is None


def test_head_to_head_single_meeting_counted_once_not_twice():
    """K8's actual trap: every played game contributes TWO team-oriented rows sharing one
    game_id. A single meeting must be tallied ONCE, not read as a 2-0 sweep."""
    rows = _played(1, "Alpha", "Beta", ACC)  # one game: Alpha's win row + Beta's loss row
    assert _head_to_head_winner(rows, SEASON, "Alpha", "Beta") == "Alpha"

    # Proof it wasn't double-counted: dropping Beta's own-perspective row (the one a correct
    # implementation never needed in the first place) must not change the outcome. If the
    # winner had depended on tallying wins_a=1 from Alpha's row AND wins_b(as a loss for
    # Alpha)=... this assertion isolates that only Alpha's own-perspective row ever mattered.
    only_alpha_perspective = [r for r in rows if r["team"] == "Alpha"]
    assert _head_to_head_winner(only_alpha_perspective, SEASON, "Alpha", "Beta") == "Alpha"


# ---------------------------------------------------------------------------
# T2/K5: a CCG LOSER (tier 1) tied with, and previously beaten by, a non-participant (tier 0)
# must not be swapped below it -- the loser-tier analog of the champion test above (K4), pinning
# that the head-to-head grouping predicate keys on _tier, not _is_champion.
# ---------------------------------------------------------------------------
def _k4_loser_tier_rows():
    """Gamma wins the ACC championship against Alpha, making Alpha the CCG LOSER (tier 1), not
    the champion. Beta ties Alpha 1-1 on (post-exclusion) conference record and beat it EARLIER
    in the regular season. Eta/Theta/Iota are fillers (Alpha's/Beta's/Gamma's compensating
    results) left OUT of teams_meta, same trick as _k4_head_to_head_rows, both to satisfy
    identify_conference_championship_games' >=4-distinct-member gate and to keep Alpha and Beta
    the only two teams_meta entries so they are guaranteed adjacent after sorting."""
    rows = []
    rows += _played(1, "Beta", "Alpha", ACC)   # earlier regular-season meeting: Beta wins
    rows += _played(2, "Alpha", "Eta", ACC)    # Alpha's compensating win
    rows += _played(3, "Theta", "Beta", ACC)   # Beta's compensating loss
    rows += _played(4, "Gamma", "Iota", ACC)   # Gamma's own conference involvement (unrelated)
    # ACC Championship: Gamma over Alpha -- alone in the latest bucket.
    rows += _played(5, "Gamma", "Alpha", ACC, neutral_site=True)
    return rows


def _k4_loser_tier_teams_meta():
    # Deliberately Alpha/Beta only -- see the fixture's docstring.
    return _teams_meta(["Alpha", "Beta"], ACC)


def test_ccg_loser_not_swapped_below_team_that_beat_it_head_to_head():
    rows = _k4_loser_tier_rows()
    payload = build_schedule_payload(rows, _k4_loser_tier_teams_meta(), SEASON)
    entries = {e["team"]: e for e in _conf_entries(payload, "ACC")}

    # Sanity-check the tie the test depends on.
    assert entries["Alpha"]["conf_record"] == {"wins": 1, "losses": 1}, entries["Alpha"]["conf_record"]
    assert entries["Beta"]["conf_record"] == {"wins": 1, "losses": 1}, entries["Beta"]["conf_record"]

    names = _conf_names(payload, "ACC")
    assert names == ["Alpha", "Beta"], (
        f"got {names}. Alpha is the CCG LOSER (tier 1) and must sort above Beta (tier 0) despite "
        "Beta having won their EARLIER regular-season meeting. If this reads ['Beta', 'Alpha'] "
        "instead, the head-to-head grouping predicate keyed on _is_champion instead of _tier "
        "(K5): Alpha is not a champion, so it would have grouped with Beta (both False) and the "
        "swap would have fired on their earlier meeting, silently undoing the loser-tier rule."
    )


# ---------------------------------------------------------------------------
# T2/K6: _placement_pct direct tests. These matter as their OWN tests, not just via the
# sort-order tests below: placement pct only changes an ORDER when two teams differ on it
# while tied on conference pct, so a fixture that happens to separate its teams earlier in
# the key would let a broken _placement_pct pass unnoticed. Exercised directly against the
# function so its arithmetic is pinned independently of any particular fixture's shape.
# ---------------------------------------------------------------------------
def test_placement_pct_excludes_postseason_rows():
    entry = {"team": "Team", "record": {"wins": 10, "losses": 3}}
    rows = [dict(season=SEASON, team="Team", season_type="postseason", status="loss", game_id=201)]
    # (10, 3) displayed, minus the 1 postseason loss found in `rows` -> (10, 2).
    assert _placement_pct(entry, rows, SEASON, champ_game_ids=set()) == 10 / 12


def test_placement_pct_excludes_the_identified_championship_game():
    entry = {"team": "Team", "record": {"wins": 10, "losses": 3}}
    rows = [dict(season=SEASON, team="Team", season_type="regular", status="win", game_id=99)]
    # (10, 3) displayed, minus the 1 championship-game win (game_id 99 is in champ_game_ids)
    # -> (9, 3). Deliberately season_type='regular' (a CCG always is) so this cannot pass merely
    # because of the postseason branch above.
    assert _placement_pct(entry, rows, SEASON, champ_game_ids={99}) == 9 / 12


def test_placement_pct_no_counted_games_uses_0_5_sentinel():
    entry = {"team": "Team", "record": {"wins": 1, "losses": 0}}
    # The team's only game is the championship game itself -- once excluded, 0 counted games.
    rows = [dict(season=SEASON, team="Team", season_type="regular", status="win", game_id=1)]
    assert _placement_pct(entry, rows, SEASON, champ_game_ids={1}) == 0.5


# ---------------------------------------------------------------------------
# T2/K4-K8: the full 2025 SEC shape -- champion, CCG loser, then the remaining teams by
# conference record with tiebreakers (issue 4), verified against the exact live 2025 numbers.
# This fixture is the one that distinguishes the two candidate tiebreak orders, so it is worth
# stating what it proves. Placement win% is compared BEFORE model rank (see the long comment
# above _sort_conference_teams): Oklahoma and Vanderbilt tie at .833 placement (post-bowl
# exclusion) while Texas sits at .75, so Texas sorts last despite ranking 11 to Vanderbilt's 14.
# A rank-first key would instead produce Oklahoma, Texas, Vanderbilt, promoting a 9-3 team over
# a 10-2 one on rating alone. Rank still decides Ole Miss over Texas A&M, who are tied on BOTH
# conference and placement record and never played -- the exhausted case rank exists for.
# ---------------------------------------------------------------------------
_SEC = "SEC"
_SEC_TEAMS = ["Georgia", "Alabama", "Ole Miss", "Texas A&M", "Oklahoma", "Texas", "Vanderbilt"]
_SEC_RANKS = {
    "Georgia": 5, "Ole Miss": 6, "Texas A&M": 8, "Oklahoma": 10, "Texas": 11, "Alabama": 12,
    "Vanderbilt": 14,
}


def _sec_single_row(game_id, team, opponent, status, conference_game, season_type, week_offset, neutral_site=False):
    """One team-oriented row against a unique filler opponent never added to teams_meta -- same
    trick as _row()/_played() above, extended with an explicit season_type so postseason rows
    can be built directly (the shared _row() helper always hardcodes season_type='regular')."""
    return dict(
        game_id=game_id,
        season=SEASON,
        season_type=season_type,
        team=team,
        opponent=opponent,
        conference=_SEC,
        conference_game=conference_game,
        status=status,
        start_date=(_SEASON_START + timedelta(days=7 * week_offset)).isoformat() + " 19:00:00",
        home_away="home",
        neutral_site=neutral_site,
    )


def _sec_conf_block(game_id_iter, team, wins, losses):
    """`wins` + `losses` conference games at weeks 0..(wins+losses-1) -- a normal regular slate
    with several teams' games sharing each early bucket, which identify_conference_championship_
    games never inspects (only the LATEST bucket matters)."""
    rows = []
    week = 0
    for w in range(wins):
        rows.append(_sec_single_row(next(game_id_iter), team, f"{team} Conf W{w}", "win", True, "regular", week))
        week += 1
    for l in range(losses):
        rows.append(_sec_single_row(next(game_id_iter), team, f"{team} Conf L{l}", "loss", True, "regular", week))
        week += 1
    return rows


def _sec_rest_block(game_id_iter, team, start_week, nonconf_w, nonconf_l, post_w, post_l):
    """Non-conference regular-season games, then postseason games, starting at `start_week`."""
    rows = []
    week = start_week
    for w in range(nonconf_w):
        rows.append(_sec_single_row(next(game_id_iter), team, f"{team} NonConf W{w}", "win", False, "regular", week))
        week += 1
    for l in range(nonconf_l):
        rows.append(_sec_single_row(next(game_id_iter), team, f"{team} NonConf L{l}", "loss", False, "regular", week))
        week += 1
    for w in range(post_w):
        rows.append(_sec_single_row(next(game_id_iter), team, f"{team} Bowl W{w}", "win", False, "postseason", week))
        week += 1
    for l in range(post_l):
        rows.append(_sec_single_row(next(game_id_iter), team, f"{team} Bowl L{l}", "loss", False, "postseason", week))
        week += 1
    return rows


def _sec_seven_team_rows():
    game_id_iter = itertools.count(1)
    rows = []

    # Conference slate: 8 games each, weeks 0-7.
    rows += _sec_conf_block(game_id_iter, "Georgia", 7, 1)
    rows += _sec_conf_block(game_id_iter, "Alabama", 7, 1)
    rows += _sec_conf_block(game_id_iter, "Ole Miss", 7, 1)
    rows += _sec_conf_block(game_id_iter, "Texas A&M", 7, 1)
    rows += _sec_conf_block(game_id_iter, "Oklahoma", 6, 2)
    rows += _sec_conf_block(game_id_iter, "Texas", 6, 2)
    rows += _sec_conf_block(game_id_iter, "Vanderbilt", 6, 2)

    # SEC Championship: Georgia over Alabama, week 8 -- alone in its bucket, strictly later than
    # every team's week 0-7 conference slate above, so it is correctly identified.
    ccg_id = next(game_id_iter)
    rows.append(_sec_single_row(ccg_id, "Georgia", "Alabama", "win", True, "regular", 8, neutral_site=True))
    rows.append(_sec_single_row(ccg_id, "Alabama", "Georgia", "loss", True, "regular", 8, neutral_site=True))

    # Non-conference regular season + postseason, per team -- see the ground-truth table in the
    # task brief. Georgia/Alabama start at week 9 (after their own week-8 CCG row); every other
    # team starts at week 8 (they have no CCG row to collide with).
    rows += _sec_rest_block(game_id_iter, "Georgia", 9, 4, 0, 0, 1)
    rows += _sec_rest_block(game_id_iter, "Alabama", 9, 3, 1, 1, 1)
    rows += _sec_rest_block(game_id_iter, "Ole Miss", 8, 4, 0, 2, 1)
    rows += _sec_rest_block(game_id_iter, "Texas A&M", 8, 4, 0, 0, 1)
    rows += _sec_rest_block(game_id_iter, "Oklahoma", 8, 4, 0, 0, 1)
    rows += _sec_rest_block(game_id_iter, "Texas", 8, 3, 1, 1, 0)
    rows += _sec_rest_block(game_id_iter, "Vanderbilt", 8, 4, 0, 0, 1)

    return rows


def _sec_teams_meta():
    return _teams_meta(_SEC_TEAMS, _SEC)


def test_2025_sec_seven_team_standings_order():
    rows = _sec_seven_team_rows()
    payload = build_schedule_payload(rows, _sec_teams_meta(), SEASON, team_ranks=_SEC_RANKS)
    entries = {e["team"]: e for e in _conf_entries(payload, "SEC")}

    # Displayed record/conf_record, verified against the live 2025 DB (task brief) -- unchanged
    # by the ordering work (K6 never touches these).
    assert entries["Georgia"]["record"] == {"wins": 12, "losses": 2}, entries["Georgia"]["record"]
    assert entries["Georgia"]["conf_record"] == {"wins": 7, "losses": 1}, entries["Georgia"]["conf_record"]
    assert entries["Alabama"]["record"] == {"wins": 11, "losses": 4}, entries["Alabama"]["record"]
    assert entries["Alabama"]["conf_record"] == {"wins": 7, "losses": 1}, entries["Alabama"]["conf_record"]
    assert entries["Ole Miss"]["record"] == {"wins": 13, "losses": 2}, entries["Ole Miss"]["record"]
    assert entries["Ole Miss"]["conf_record"] == {"wins": 7, "losses": 1}, entries["Ole Miss"]["conf_record"]
    assert entries["Texas A&M"]["record"] == {"wins": 11, "losses": 2}, entries["Texas A&M"]["record"]
    assert entries["Texas A&M"]["conf_record"] == {"wins": 7, "losses": 1}, entries["Texas A&M"]["conf_record"]
    assert entries["Oklahoma"]["record"] == {"wins": 10, "losses": 3}, entries["Oklahoma"]["record"]
    assert entries["Texas"]["record"] == {"wins": 10, "losses": 3}, entries["Texas"]["record"]
    assert entries["Vanderbilt"]["record"] == {"wins": 10, "losses": 3}, entries["Vanderbilt"]["record"]

    names = _conf_names(payload, "SEC")
    assert names == [
        "Georgia", "Alabama", "Ole Miss", "Texas A&M", "Oklahoma", "Vanderbilt", "Texas",
    ], (
        f"got {names}. Expected the champion (Georgia), then the CCG loser (Alabama), then the "
        "remaining teams by conference record with tiebreakers -- Ole Miss ahead of Texas A&M "
        "purely on model rank (6 vs 8, tied on BOTH conference and placement record, and they "
        "never played each other); then Oklahoma and Vanderbilt (both .833 placement) ahead of "
        "Texas (.75), because placement record is compared before rank, so Texas's better rank "
        "(11 vs Vanderbilt's 14) does not lift it over a whole game of record."
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
