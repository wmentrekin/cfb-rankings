"""Tests for T4: division-aware conference championship status
(artifacts/schedule_standings.py's DIVISIONAL_CHAMPIONSHIP_CONFERENCES,
compute_conference_championship_status's division pools, and the
cross-division conditional_opponent).

Every record in here is INVENTED. Real Sun Belt team names are used (matching
tests/test_division_grouping.py's convention) purely so the divisions are
recognizable -- none of these win/loss tallies describe an actual season.

Covers:
  1. The division race is ranked by OVERALL conference record (all conference
     games, divisional and non-divisional), NOT by a division-only record.
     Built so the two readings disagree: the East team that leads on overall
     conference record has the WORST division-only standing of the contenders,
     and vice versa. This is the finding most likely to be got wrong -- if
     anyone later "fixes" the code to count only intra-division games, this
     test flips both assertions and fails loudly.
  2. The two divisions are computed independently: a team can clinch the East
     while the West is still fully contested, and neither result contaminates
     the other.
  3. conditional_opponent for a still-possible EAST team resolves to the
     clinched WEST team -- the other division's champion, never one of its own
     division rivals (its own division's single slot is what it is competing
     for).
  4. Regression: a flat, non-divisional conference (Big Ten shape) produces
     byte-identical statuses to the pre-T4 top-2 algorithm, and injecting a
     division map cannot change them.
  5. Conservatism: an undecided division yields "possible" for the
     contenders -- including the boundary case where a trailing team's best
     case exactly TIES the leader's banked wins (it must NOT be eliminated,
     because a tiebreaker this module deliberately does not model could still
     hand it the division).
  6. Safety gate: an incomplete division map blanks the WHOLE conference
     rather than guessing a division race from partial data.

Run: python -m pytest tests/ -q   (or: python tests/test_division_championship_status.py)
"""
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifacts.schedule_standings import (  # noqa: E402
    DIVISIONAL_CHAMPIONSHIP_CONFERENCES,
    QUALIFYING_CHAMPIONSHIP_CONFERENCES,
    compute_conference_championship_status,
    compute_standings,
)

SEASON = 2026
SUN_BELT = "Sun Belt"

EAST = ["App State", "Coastal Carolina", "Georgia Southern", "Georgia State"]
WEST = ["Arkansas State", "Louisiana", "South Alabama", "Troy"]
SUN_BELT_DIVISIONS = {t: "East" for t in EAST}
SUN_BELT_DIVISIONS.update({t: "West" for t in WEST})


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _row(game_id, team, opponent, status, conference=SUN_BELT):
    """One team-oriented schedule_grid row -- only the columns
    compute_team_records actually reads are populated. Each game_id gets its
    own Saturday so build_schedule_payload can bucket them into distinct week
    columns (championship status itself is date-independent)."""
    return dict(
        game_id=game_id,
        season=SEASON,
        season_type="regular",
        team=team,
        opponent=opponent,
        conference=conference,
        conference_game=True,
        status=status,
        start_date=(date(2026, 9, 5) + timedelta(days=7 * (game_id - 1))).isoformat() + " 19:00:00",
        home_away="home",
        neutral_site=False,
    )


def _played(game_id, winner, loser):
    """Both team-oriented rows for one completed conference game."""
    return [_row(game_id, winner, loser, "win"), _row(game_id, loser, winner, "loss")]


def _record(conference, conf_wins, conf_losses, conf_games_remaining):
    """One compute_team_records()-shaped entry. The overall wins/losses are
    irrelevant to championship status (only the conf_* fields are read), so
    they simply mirror the conference record."""
    return {
        "conference": conference,
        "wins": conf_wins,
        "losses": conf_losses,
        "conf_wins": conf_wins,
        "conf_losses": conf_losses,
        "conf_games_remaining": conf_games_remaining,
    }


# ---------------------------------------------------------------------------
# Case 1: overall conference record, NOT division-only record
# ---------------------------------------------------------------------------
def _overall_vs_division_only_rows():
    """A completed conference season, deliberately built so the two possible
    readings of "division champion" pick DIFFERENT East teams.

    East, by OVERALL conference record (the real Sun Belt rule):
        App State 3-0, Coastal Carolina 2-2, Georgia Southern 1-1, Georgia State 0-3
    East, by DIVISION-ONLY record (the wrong-but-intuitive reading):
        Coastal Carolina 2-0, Georgia Southern 1-1, Georgia State 0-2, App State 0-0

    App State's three wins are ALL against West teams, so a division-only
    count credits it with nothing at all, while Coastal Carolina's two wins
    are both intra-division.
    """
    rows = []
    rows += _played(1, "App State", "Arkansas State")
    rows += _played(2, "App State", "Louisiana")
    rows += _played(3, "App State", "South Alabama")
    rows += _played(4, "Coastal Carolina", "Georgia Southern")
    rows += _played(5, "Coastal Carolina", "Georgia State")
    rows += _played(6, "Arkansas State", "Coastal Carolina")
    rows += _played(7, "Louisiana", "Coastal Carolina")
    rows += _played(8, "Georgia Southern", "Georgia State")
    rows += _played(9, "Troy", "Georgia State")
    return rows


def test_division_race_ranks_by_overall_conference_record():
    standings = compute_standings(_overall_vs_division_only_rows(), SEASON, divisions=SUN_BELT_DIVISIONS)

    # Sanity-check the fixture itself before trusting the assertions below.
    assert standings["App State"]["conf_record"] == {"wins": 3, "losses": 0}
    assert standings["Coastal Carolina"]["conf_record"] == {"wins": 2, "losses": 2}

    app_state = standings["App State"]["championship_status"]
    coastal = standings["Coastal Carolina"]["championship_status"]

    assert app_state == "clinched", (
        "App State leads the East on OVERALL conference record (3-0) and every other East "
        f"team is mathematically out, so it has clinched the division -- got {app_state!r}. "
        "If this now says 'eliminated', the division race is being ranked by DIVISION-ONLY "
        "record (App State is 0-0 in-division, all three of its wins are against West "
        "teams). That is the wrong rule: a Sun Belt division champion is decided by winning "
        "percentage across ALL conference games; divisional record is only a tiebreaker. "
        "Do not 'fix' this test -- fix the code back."
    )
    assert coastal == "eliminated", (
        "Coastal Carolina is 2-2 in conference play with no games left, behind App State's "
        f"3-0, so it cannot win the East -- got {coastal!r}. If this now says 'clinched', "
        "the code is ranking by division-only record (Coastal Carolina is 2-0 in-division). "
        "See the message above: that rule is wrong."
    )

    # The other two East teams are behind on both readings; they only confirm
    # the division is fully resolved.
    assert standings["Georgia Southern"]["championship_status"] == "eliminated"
    assert standings["Georgia State"]["championship_status"] == "eliminated"


# ---------------------------------------------------------------------------
# Case 2: divisions computed independently
# ---------------------------------------------------------------------------
def test_east_can_clinch_while_west_is_still_contested():
    """Same fixture as case 1. The East is fully resolved (App State clinched,
    everyone else eliminated) while the West is a three-way tie at 1-1/1-0
    that this module deliberately refuses to resolve -- so every West
    contender must still read 'possible'."""
    standings = compute_standings(_overall_vs_division_only_rows(), SEASON, divisions=SUN_BELT_DIVISIONS)
    status = {t: standings[t]["championship_status"] for t in EAST + WEST}

    assert status["App State"] == "clinched"
    east_clinched = [t for t in EAST if status[t] == "clinched"]
    assert east_clinched == ["App State"], f"exactly one East clinch expected, got {east_clinched}"

    # West: Arkansas State 1-1, Louisiana 1-1, Troy 1-0, South Alabama 0-1.
    # Three teams are tied on banked wins, so no one has clinched and no one
    # who can still reach the top is eliminated -- the East's resolved race
    # must not leak across.
    assert status["Arkansas State"] == "possible", status
    assert status["Louisiana"] == "possible", status
    assert status["Troy"] == "possible", status
    assert status["South Alabama"] == "eliminated", status
    assert [t for t in WEST if status[t] == "clinched"] == [], (
        f"no West team can have clinched a division decided by tiebreaker: {status}"
    )


# ---------------------------------------------------------------------------
# Case 3: cross-division conditional_opponent
# ---------------------------------------------------------------------------
def test_conditional_opponent_for_an_east_team_is_the_clinched_west_team():
    """West resolved (Troy clinched), East wide open. Every still-possible
    East team's conditional title-game opponent is Troy -- the OTHER
    division's champion."""
    records = {
        # East: all four 1-1 with two conference games left -- nobody near a
        # clinch or an elimination.
        "App State": _record(SUN_BELT, 1, 1, 2),
        "Coastal Carolina": _record(SUN_BELT, 1, 1, 2),
        "Georgia Southern": _record(SUN_BELT, 1, 1, 2),
        "Georgia State": _record(SUN_BELT, 1, 1, 2),
        # West: Troy has banked more wins than anyone else can still reach.
        "Troy": _record(SUN_BELT, 4, 0, 0),
        "Arkansas State": _record(SUN_BELT, 1, 3, 0),
        "Louisiana": _record(SUN_BELT, 1, 3, 0),
        "South Alabama": _record(SUN_BELT, 0, 4, 0),
    }
    result = compute_conference_championship_status(records, divisions=SUN_BELT_DIVISIONS)

    assert result["Troy"]["status"] == "clinched", result["Troy"]
    for team in EAST:
        assert result[team]["status"] == "possible", (team, result[team])
        opponent = result[team]["conditional_opponent"]
        assert opponent == "Troy", (
            f"{team} (East) should face the clinched WEST team in the title game, "
            f"got conditional_opponent={opponent!r}"
        )
        assert SUN_BELT_DIVISIONS[opponent] == "West", (
            f"{team}'s conditional opponent {opponent!r} is in its own division -- the "
            "title game is East champion vs. West champion, so the opponent must never "
            "come from the team's own division."
        )

    # The clinched team and the eliminated teams carry no conditional opponent,
    # matching the pre-existing convention for flat conferences.
    assert result["Troy"]["conditional_opponent"] is None
    for team in ("Arkansas State", "Louisiana", "South Alabama"):
        assert result[team]["status"] == "eliminated", (team, result[team])
        assert result[team]["conditional_opponent"] is None


def test_a_clinched_east_team_is_not_offered_to_its_own_division():
    """Mirror of the above with the divisions swapped, so the cross-division
    lookup is exercised in both directions rather than only East->West."""
    records = {
        "App State": _record(SUN_BELT, 4, 0, 0),
        "Coastal Carolina": _record(SUN_BELT, 1, 3, 0),
        "Georgia Southern": _record(SUN_BELT, 1, 3, 0),
        "Georgia State": _record(SUN_BELT, 0, 4, 0),
        "Troy": _record(SUN_BELT, 1, 1, 2),
        "Arkansas State": _record(SUN_BELT, 1, 1, 2),
        "Louisiana": _record(SUN_BELT, 1, 1, 2),
        "South Alabama": _record(SUN_BELT, 1, 1, 2),
    }
    result = compute_conference_championship_status(records, divisions=SUN_BELT_DIVISIONS)

    assert result["App State"]["status"] == "clinched"
    for team in WEST:
        assert result[team]["status"] == "possible", (team, result[team])
        assert result[team]["conditional_opponent"] == "App State", (team, result[team])
    # Nobody in the East is offered App State: its division rivals are all
    # eliminated, and App State itself is clinched.
    for team in EAST:
        assert result[team]["conditional_opponent"] is None, (team, result[team])


# ---------------------------------------------------------------------------
# Case 4: flat-conference regression
# ---------------------------------------------------------------------------
def _big_ten_records():
    """A flat (no-divisions) Big Ten shape exercising all three statuses plus
    the single-clinch conditional_opponent path."""
    return {
        "Ohio State": _record("Big Ten", 8, 0, 1),
        "Indiana": _record("Big Ten", 6, 2, 1),
        "Michigan": _record("Big Ten", 5, 3, 1),
        "Iowa": _record("Big Ten", 4, 4, 1),
        "Purdue": _record("Big Ten", 1, 7, 1),
        "Illinois": _record("Big Ten", 0, 8, 1),
    }


# Expected values are the PRE-T4 algorithm's output for _big_ten_records(),
# verified by running the same input through git's HEAD copy of
# artifacts/schedule_standings.py. Top 2 advance:
#   Ohio State  W=8 B=9 -- only Indiana (B=7) could reach its floor of 8? no
#                          one can, so <=1 others reach it -> clinched
#   Indiana     W=6 B=7 -- Ohio State (9) and Michigan (6) both reach its
#                          floor of 6 -> 2 others -> possible
#   Michigan    W=5 B=6 -- 2nd-highest other W is 6 (Indiana); 6 < 6 is false
#                          -> not eliminated -> possible
#   Iowa        W=4 B=5 -- 2nd-highest other W is 6; 5 < 6 -> eliminated
#   Purdue/Illinois     -- far below -> eliminated
_BIG_TEN_EXPECTED = {
    "Ohio State": {"status": "clinched", "conditional_opponent": None},
    "Indiana": {"status": "possible", "conditional_opponent": "Ohio State"},
    "Michigan": {"status": "possible", "conditional_opponent": "Ohio State"},
    "Iowa": {"status": "eliminated", "conditional_opponent": None},
    "Purdue": {"status": "eliminated", "conditional_opponent": None},
    "Illinois": {"status": "eliminated", "conditional_opponent": None},
}


def test_flat_conference_statuses_are_unchanged_by_the_division_support():
    records = _big_ten_records()

    # Called exactly as the pre-T4 code called it (no division arguments at all).
    assert compute_conference_championship_status(records) == _BIG_TEN_EXPECTED

    # And unchanged when a division map is injected -- including one that
    # names Sun Belt teams. A flat conference's teams are not in the map, and
    # nothing about another conference's divisions may leak in.
    for divisions in ({}, SUN_BELT_DIVISIONS, {t: None for t in records}):
        assert compute_conference_championship_status(records, divisions=divisions) == _BIG_TEN_EXPECTED, (
            f"flat Big Ten statuses changed when divisions={divisions!r} was injected"
        )


def test_sun_belt_is_the_only_divisional_conference_and_the_format_lists_are_disjoint():
    assert SUN_BELT in DIVISIONAL_CHAMPIONSHIP_CONFERENCES
    assert SUN_BELT not in QUALIFYING_CHAMPIONSHIP_CONFERENCES
    overlap = set(DIVISIONAL_CHAMPIONSHIP_CONFERENCES) & set(QUALIFYING_CHAMPIONSHIP_CONFERENCES)
    assert not overlap, f"a conference cannot have both formats: {overlap}"


# ---------------------------------------------------------------------------
# Case 5: conservatism
# ---------------------------------------------------------------------------
def test_an_undecided_division_yields_possible_not_a_premature_clinch():
    records = {
        # East, mid-season: App State leads at 4-0 but four conference games
        # remain for everyone, so nothing is settled.
        "App State": _record(SUN_BELT, 4, 0, 4),
        "Coastal Carolina": _record(SUN_BELT, 2, 2, 4),
        # Georgia Southern's best case (0 + 4 = 4) exactly TIES App State's
        # banked 4 wins. A tie is decided by tiebreakers this module
        # deliberately does not model, so it must NOT be called eliminated.
        "Georgia Southern": _record(SUN_BELT, 0, 4, 4),
        # Georgia State can reach at most 3 -- strictly below App State's
        # banked 4 -- so it genuinely cannot win the division.
        "Georgia State": _record(SUN_BELT, 0, 5, 3),
        "Troy": _record(SUN_BELT, 2, 2, 4),
        "Arkansas State": _record(SUN_BELT, 2, 2, 4),
        "Louisiana": _record(SUN_BELT, 2, 2, 4),
        "South Alabama": _record(SUN_BELT, 2, 2, 4),
    }
    result = compute_conference_championship_status(records, divisions=SUN_BELT_DIVISIONS)

    assert result["App State"]["status"] == "possible", (
        "App State leads the East but Coastal Carolina can still finish 6-2 -- calling this "
        f"a clinch would be a premature assertion. Got {result['App State']}"
    )
    assert result["Coastal Carolina"]["status"] == "possible"
    assert result["Georgia Southern"]["status"] == "possible", (
        "Georgia Southern's best case ties the leader's banked wins, and a tie is settled by "
        f"tiebreakers this module does not model -- it must not be eliminated. Got "
        f"{result['Georgia Southern']}"
    )
    assert result["Georgia State"]["status"] == "eliminated", (
        "Georgia State cannot reach the leader's banked win total even by winning out. Got "
        f"{result['Georgia State']}"
    )
    # Nothing has clinched anywhere, so no team gets a conditional opponent.
    assert all(v["conditional_opponent"] is None for v in result.values()), result


# ---------------------------------------------------------------------------
# Case 6: safety gate on an incomplete/unexpected division map
# ---------------------------------------------------------------------------
def test_incomplete_division_map_blanks_the_whole_conference():
    records = {t: _record(SUN_BELT, 4, 0, 0) if t == "App State" else _record(SUN_BELT, 0, 4, 0)
               for t in EAST + WEST}

    # Case A: no division map at all -- the pre-T4 behavior, blank for everyone.
    assert compute_conference_championship_status(records) == {}

    # Case B: one team missing from the map. App State would otherwise clinch
    # the East; a partial map must blank the entire conference instead, because
    # the missing team might belong to either division.
    partial = dict(SUN_BELT_DIVISIONS)
    del partial["Georgia State"]
    assert compute_conference_championship_status(records, divisions=partial) == {}

    # Case C: a single division (not the modeled two-division title game).
    one_division = {t: "East" for t in EAST + WEST}
    assert compute_conference_championship_status(records, divisions=one_division) == {}

    # Case D: a division too small to trust as complete data (min_members=4).
    lopsided = dict(SUN_BELT_DIVISIONS)
    lopsided["Troy"] = "East"
    lopsided["Louisiana"] = "East"
    lopsided["South Alabama"] = "East"
    assert compute_conference_championship_status(records, divisions=lopsided) == {}


# ---------------------------------------------------------------------------
# Wiring: artifacts/schedule.py must actually inject the division map
# ---------------------------------------------------------------------------
def test_schedule_payload_wires_divisions_into_the_championship_cell():
    """build_schedule_payload passes teams_meta's `division` into
    compute_standings. Without that one wiring line the Sun Belt's Conference
    Championship cells fall through to 'bye' -- the blank they used to be."""
    from artifacts.schedule import CONF_CHAMPIONSHIP_SLOT_ID, build_schedule_payload

    teams_meta = {t: {"conference": SUN_BELT, "division": SUN_BELT_DIVISIONS[t], "logos": None}
                  for t in EAST + WEST}
    payload = build_schedule_payload(_overall_vs_division_only_rows(), teams_meta, SEASON)

    cells = {}
    for conference in payload["conferences"]:
        for entry in conference["teams"]:
            for week in entry["weeks"]:
                if week["slot_id"] == CONF_CHAMPIONSHIP_SLOT_ID:
                    cells[entry["team"]] = week

    assert cells["App State"]["status"] == "clinched", cells["App State"]
    assert cells["Coastal Carolina"]["status"] == "eliminated", cells["Coastal Carolina"]
    # A West contender is still alive, and its conditional title-game opponent
    # is the clinched East team.
    assert cells["Troy"]["status"] == "possible", cells["Troy"]
    assert cells["Troy"]["conditional_opponent"] == "App State", cells["Troy"]
    assert all(c["status"] != "bye" for c in cells.values()), cells


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
