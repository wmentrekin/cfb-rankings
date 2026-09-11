"""Tests for T4b: identify_conference_championship_games's default
`qualifying_conferences` gate, widened from QUALIFYING_CHAMPIONSHIP_CONFERENCES
alone to the UNION of QUALIFYING_CHAMPIONSHIP_CONFERENCES and
DIVISIONAL_CHAMPIONSHIP_CONFERENCES (see the in-function comment in
artifacts/schedule.py for the full rationale).

The problem this closes: identify_conference_championship_games decides which
schedule_grid row is a conference's real title game so it gets DIVERTED out of
its week column into the dedicated Conference Championship column. It was
gated on QUALIFYING_CHAMPIONSHIP_CONFERENCES alone -- the flat top-2-of-one-
pool list that also gates the CLINCH/ELIMINATE status math. That list has no
entry for the Sun Belt (divisional) or, historically, the Pac-12, so their
real title games were never diverted: a real game sat in a week column AND
(for the Sun Belt, which does have a DIVISIONAL_CHAMPIONSHIP_CONFERENCES
entry) a computed status appeared in the Conference Championship column too,
plus the near-empty week column the un-diverted game leaves behind -- the same
phantom-column bug this project already fixed once for Army-Navy.

Covers:
  1. A Sun Belt-shaped fixture is now identified, where it previously was not.
  2. A Pac-12-shaped fixture likewise.
  3. A conference in NEITHER curated list (an invented FCS-looking name) is
     still NOT identified -- proving the union cannot admit arbitrary
     conferences, only conferences that are actually in one of the two
     hand-maintained FBS-only lists.
  4. Regression bar: real title games are still identified for the
     previously-covered (flat-format) conferences, and an in-progress season
     with a multi-game rivalry-week bucket still identifies nothing.
  5. The actual user-visible point: a diverted Sun Belt title game does not
     leave behind a near-empty week column (build_canonical_columns
     interaction).

Run: python -m pytest tests/ -q   (or: python tests/test_championship_diversion_scope.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifacts.schedule import (  # noqa: E402
    build_canonical_columns,
    identify_conference_championship_games,
)
from artifacts.schedule_standings import (  # noqa: E402
    DIVISIONAL_CHAMPIONSHIP_CONFERENCES,
    QUALIFYING_CHAMPIONSHIP_CONFERENCES,
)


def _game(game_id, season, home, away, start_date, conference, conference_game=True,
          season_type="regular", notes=None):
    """Expand one game into the two team-oriented rows schedule_grid emits."""
    common = dict(game_id=game_id, season=season, season_type=season_type,
                  conference=conference, conference_game=conference_game,
                  start_date=start_date, notes=notes)
    return [
        dict(team=home, opponent=away, home_away="home", **common),
        dict(team=away, opponent=home, home_away="away", **common),
    ]


def _rows(*games):
    out = []
    for g in games:
        out.extend(_game(*g))
    return out


# --- Real Sun Belt 2025: James Madison (East champion) vs Troy (West
# champion), 2025-12-06, the actual Sun Belt Championship Game -- sits alone
# in its bucket, one bucket later than the Nov 28-29 regular slate. ----------
SUN_BELT_2025 = _rows(
    (401756123, 2025, "Coastal Carolina", "Old Dominion", "2025-11-28 19:00:00", "Sun Belt"),
    (401756124, 2025, "James Madison", "Georgia Southern", "2025-11-29 16:00:00", "Sun Belt"),
    (401756125, 2025, "Troy", "South Alabama", "2025-11-29 19:00:00", "Sun Belt"),
    (401770123, 2025, "James Madison", "Troy", "2025-12-06 20:00:00", "Sun Belt"),
)

# --- Pac-12-shaped fixture: same structural pattern as the ACC/Sun Belt
# fixtures above (a regular slate bucket, then a single later game alone in
# its own bucket), using real 2026 Pac-12 member names. The Pac-12's actual
# 2026 title game has not been played (2026 has no championship games
# scheduled at all -- see the real-data check), so this is a shape-matching
# synthetic fixture, not a verbatim real game_id. -----------------------------
PAC12_SHAPED = _rows(
    (9001, 2026, "Boise State", "Fresno State", "2026-11-21 20:00:00", "Pac-12"),
    (9002, 2026, "San Diego State", "Colorado State", "2026-11-21 22:00:00", "Pac-12"),
    (9003, 2026, "Boise State", "San Diego State", "2026-12-04 21:00:00", "Pac-12"),
)

# --- A conference in NEITHER curated list. Same winning shape (lone game in a
# later bucket) as the fixtures above, but the conference name is not a member
# of QUALIFYING_CHAMPIONSHIP_CONFERENCES or DIVISIONAL_CHAMPIONSHIP_CONFERENCES
# -- an FCS-looking name, standing in for exactly what widening to "every
# conference in the data" would wrongly admit. --------------------------------
UNLISTED_CONFERENCE = "Big Sky"
UNLISTED_SHAPED = _rows(
    (9101, 2026, "Montana", "Montana State", "2026-11-14 20:00:00", UNLISTED_CONFERENCE),
    (9102, 2026, "Idaho", "Sacramento State", "2026-11-14 22:00:00", UNLISTED_CONFERENCE),
    (9103, 2026, "Montana", "Idaho", "2026-11-28 21:00:00", UNLISTED_CONFERENCE),
)

# --- Real ACC 2025 (flat, previously-covered format) -- regression fixture,
# same real data as tests/test_schedule_columns.py's ACC_2025. ---------------
ACC_2025 = _rows(
    (401754531, 2025, "Wake Forest", "NC State", "2025-09-11 23:30:00", "ACC"),
    (401754623, 2025, "Georgia Tech", "Clemson", "2025-09-13 16:00:00", "ACC"),
    (401754611, 2025, "Pittsburgh", "Miami", "2025-11-29 17:00:00", "ACC"),
    (401754610, 2025, "Duke", "Wake Forest", "2025-11-29 20:30:00", "ACC"),
    (401754613, 2025, "Virginia", "Virginia Tech", "2025-11-30 00:00:00", "ACC"),
    (401777328, 2025, "Virginia", "Duke", "2025-12-07 01:00:00", "ACC"),
)

# --- Real ACC 2026, in progress: six conference games share one rivalry-week
# start_date, and nothing later is scheduled -- must still identify nothing. -
ACC_2026 = _rows(
    (401858202, 2026, "Virginia", "NC State", "2026-08-29 19:30:00", "ACC"),
    (401858206, 2026, "Stanford", "Miami", "2026-09-05 01:00:00", "ACC"),
    (401858212, 2026, "Florida State", "SMU", "2026-09-07 23:30:00", "ACC"),
    (401858315, 2026, "California", "Pittsburgh", "2026-11-28 05:00:00", "ACC"),
    (401858313, 2026, "North Carolina", "NC State", "2026-11-28 05:00:00", "ACC"),
    (401858317, 2026, "Virginia Tech", "Virginia", "2026-11-28 05:00:00", "ACC"),
    (401858311, 2026, "Miami", "Boston College", "2026-11-28 05:00:00", "ACC"),
    (401858312, 2026, "Wake Forest", "Duke", "2026-11-28 05:00:00", "ACC"),
    (401858316, 2026, "Stanford", "SMU", "2026-11-28 05:00:00", "ACC"),
)


# ---------------------------------------------------------------------------
# 1. Sun Belt: now identified via the default (union) gate.
# ---------------------------------------------------------------------------
def test_sun_belt_title_game_is_now_identified_by_default():
    # Sanity: Sun Belt is NOT in the flat list, only the divisional one --
    # confirms this test actually exercises the widened gate, not the old one.
    assert "Sun Belt" not in QUALIFYING_CHAMPIONSHIP_CONFERENCES
    assert "Sun Belt" in DIVISIONAL_CHAMPIONSHIP_CONFERENCES

    found = identify_conference_championship_games(SUN_BELT_2025, 2025)
    assert found == {"Sun Belt": 401770123}, found

    # And explicitly confirm the OLD gate (flat list alone) missed it -- this
    # is the exact bug the widened default fixes.
    found_old_gate = identify_conference_championship_games(
        SUN_BELT_2025, 2025, QUALIFYING_CHAMPIONSHIP_CONFERENCES
    )
    assert found_old_gate == {}, (
        f"expected the flat-list-only gate to miss the Sun Belt (that's the bug being fixed), "
        f"got {found_old_gate}"
    )


# ---------------------------------------------------------------------------
# 2. Pac-12: now identified via the default (union) gate.
# ---------------------------------------------------------------------------
def test_pac12_title_game_is_now_identified_by_default():
    assert "Pac-12" in QUALIFYING_CHAMPIONSHIP_CONFERENCES  # Pac-12 IS in the flat list...

    found = identify_conference_championship_games(PAC12_SHAPED, 2026)
    assert found == {"Pac-12": 9003}, found


# ---------------------------------------------------------------------------
# 3. A conference in neither list is still not identified.
# ---------------------------------------------------------------------------
def test_conference_in_neither_list_is_not_identified():
    assert UNLISTED_CONFERENCE not in QUALIFYING_CHAMPIONSHIP_CONFERENCES
    assert UNLISTED_CONFERENCE not in DIVISIONAL_CHAMPIONSHIP_CONFERENCES

    found = identify_conference_championship_games(UNLISTED_SHAPED, 2026)
    assert found == {}, (
        f"the union gate must not admit a conference belonging to neither curated list "
        f"(this stands in for an FCS conference the widen-to-everything approach would have "
        f"wrongly admitted): got {found}"
    )


# ---------------------------------------------------------------------------
# 4. Regression bar: previously-covered conferences and the in-progress case.
# ---------------------------------------------------------------------------
def test_previously_covered_flat_conference_still_identified():
    found = identify_conference_championship_games(ACC_2025, 2025)
    assert found == {"ACC": 401777328}, found


def test_in_progress_season_still_identifies_nothing():
    found = identify_conference_championship_games(ACC_2026, 2026)
    assert found == {}, f"invented a championship matchup: {found}"


# ---------------------------------------------------------------------------
# 5. The actual user-visible point: no near-empty week column left behind.
# ---------------------------------------------------------------------------
def test_diverted_sun_belt_title_game_leaves_no_phantom_week_column():
    champ_ids = set(identify_conference_championship_games(SUN_BELT_2025, 2025).values())
    assert 401770123 in champ_ids, "the Sun Belt title game must be diverted for this test to be meaningful"

    columns = build_canonical_columns(SUN_BELT_2025, 2025, champ_ids, None)
    slot_ids = [slot_id for slot_id, _ in columns]

    # The Dec 6 bucket held ONLY the diverted title game, so no week column
    # for it may survive -- that would be the near-empty column this task
    # exists to prevent, the same phantom-column class as the Army-Navy fix.
    dec6_week = None
    from artifacts.schedule import _get_week_slot  # noqa: E402 -- internal helper, test-only import
    for row in SUN_BELT_2025:
        if row["game_id"] == 401770123:
            dec6_week = f"week-{_get_week_slot(row)}"
            break
    assert dec6_week is not None
    assert dec6_week not in slot_ids, (
        f"the diverted Sun Belt title game's bucket ({dec6_week}) must not mint an empty week "
        f"column: {slot_ids}"
    )
    assert "conf-championship" in slot_ids, "the dedicated championship slot must still be present"

    # The Nov 28-29 regular slate keeps its own column -- only the title
    # game's own singleton bucket disappears.
    nov_week = f"week-{_get_week_slot(SUN_BELT_2025[0])}"
    assert nov_week in slot_ids, f"the Sun Belt's regular slate column must survive: {slot_ids}"



# =============================================================================
# T1 (season-grid-standings-fixes, K1/K2/K3): the AUTHORITATIVE NOTES SIGNAL.
#
# CFBD's 2025 data model tags every real conference championship game with
# conference_game=FALSE and notes='<Conference> Championship' -- the structural
# rule above requires conference_game=TRUE to even nominate a candidate, so
# without this signal 2025 identifies NOTHING for any of its nine real title
# games. See the module comment above identify_conference_championship_games
# in artifacts/schedule.py for the full rationale.
# =============================================================================
import logging  # noqa: E402

import pytest  # noqa: E402


def _champ_notes_game(game_id, conference, notes, season=2025,
                       home="Home Team", away="Away Team", start_date="2025-12-06 12:00:00"):
    """One 2025-shaped championship-game fixture: conference_game=False, notes set, exactly
    the CFBD data shape T1 fixes identification for."""
    return _game(game_id, season, home, away, start_date, conference,
                 conference_game=False, season_type="regular", notes=notes)


# All nine real 2025 conference championship games, per the task brief's verified ground
# truth -- seven match their own `conference` value by plain equality after stripping
# " Championship"; American Athletic and Mid-American need the alias map (K3).
NINE_2025_CHAMPIONSHIPS = {
    "ACC": (601001, "ACC Championship"),
    "SEC": (601002, "SEC Championship"),
    "Big Ten": (601003, "Big Ten Championship"),
    "Big 12": (601004, "Big 12 Championship"),
    "American Athletic": (601005, "American Championship"),
    "Conference USA": (601006, "Conference USA Championship"),
    "Mid-American": (601007, "MAC Championship"),
    "Mountain West": (601008, "Mountain West Championship"),
    "Sun Belt": (601009, "Sun Belt Championship"),
}

NINE_2025_CHAMPIONSHIP_ROWS = []
for _conf, (_gid, _notes) in NINE_2025_CHAMPIONSHIPS.items():
    NINE_2025_CHAMPIONSHIP_ROWS.extend(_champ_notes_game(_gid, _conf, _notes))


def test_all_nine_2025_shaped_championships_are_identified_including_aliases():
    """T1 test 1: every one of the nine real 2025 championship games is identified from
    notes alone, including the two aliased conferences (American -> American Athletic,
    MAC -> Mid-American) -- despite conference_game=False, which the structural rule alone
    would treat as disqualifying."""
    found = identify_conference_championship_games(NINE_2025_CHAMPIONSHIP_ROWS, 2025)
    expected = {conf: gid for conf, (gid, _notes) in NINE_2025_CHAMPIONSHIPS.items()}
    assert found == expected, found
    # Explicitly confirm the two alias conferences specifically resolved correctly.
    assert found["American Athletic"] == 601005
    assert found["Mid-American"] == 601007


# --- 2024-shaped season: notes absent/None, conference_game=True -- the structural rule
# must behave EXACTLY as it did before T1 (real 2024 ACC title game shape). -----------
ACC_2024_STRUCTURAL = _rows(
    (401628356, 2024, "Wake Forest", "NC State", "2024-09-13 23:30:00", "ACC"),
    (401635536, 2024, "Georgia Tech", "Clemson", "2024-09-14 16:00:00", "ACC"),
    (401635525, 2024, "Pittsburgh", "Miami", "2024-11-29 17:00:00", "ACC"),
    (401635524, 2024, "Duke", "Wake Forest", "2024-11-29 20:30:00", "ACC"),
    (401645401, 2024, "SMU", "Clemson", "2024-12-07 20:00:00", "ACC"),
    # Decoy: conference_game=False, tagged conference='ACC', sitting alone in an EVEN LATER
    # bucket than the real title game -- if the structural rule's conference_game gate were
    # ever dropped, this would be wrongly identified instead of 401645401. Pins that gate.
    (401699999, 2024, "Notre Dame", "USC", "2024-12-14 20:00:00", "ACC", False),
)


def test_2024_shaped_season_identifies_exactly_what_the_structural_rule_did_before():
    """T1 test 2: notes is absent (None, the _game() default) on every row here, exactly the
    pre-2025 CFBD shape (confirmed live: notes is NULL on all ~23,000 rows for 2014-2024) --
    the notes signal must never fire, so the outcome is identical to the pure structural rule."""
    for row in ACC_2024_STRUCTURAL:
        assert row["notes"] is None  # pins the "2024-shaped" premise of this fixture
    found = identify_conference_championship_games(ACC_2024_STRUCTURAL, 2024)
    assert found == {"ACC": 401645401}, found


# --- Real 2025 Pac-12: two-team remnant, no notes row at all -- must still identify
# nothing (pins that the member-count gate is still live on the structural path when
# there is no authoritative notes match to bypass it). ---------------------------------
PAC12_2025_NO_NOTES = _rows(
    (401752900, 2025, "Oregon State", "Washington State", "2025-11-01 20:00:00", "Pac-12", True),
    (401752946, 2025, "Washington State", "Oregon State", "2025-11-29 20:00:00", "Pac-12", True),
)


def test_two_team_pac12_with_no_notes_rows_still_identifies_nothing():
    """T1 test 3: the notes signal has nothing to match here (every row's notes is None), so
    it never reaches the bypass -- the structural rule's member-count gate is still the thing
    stopping this two-team rematch from being crowned a championship."""
    for row in PAC12_2025_NO_NOTES:
        assert row["notes"] is None  # pins that this fixture carries no notes row to match
    found = identify_conference_championship_games(PAC12_2025_NO_NOTES, 2025)
    assert found == {}, f"invented a championship for a two-team conference: {found}"


# --- A notes value ending in "Championship" whose prefix does NOT match the row's own
# conference -- K2's mismatch path. -----------------------------------------------------
def test_notes_prefix_mismatch_identifies_nothing_and_logs_a_warning(caplog):
    """T1 test 4: 'Big 12 Championship' notes on a row whose OWN conference is 'Big Ten' (a
    deliberately wrong pairing, both real QUALIFYING_CHAMPIONSHIP_CONFERENCES members so this
    exercises the prefix-equality check itself, not the qualifying-conferences gate) must not
    match -- the prefix must resolve to THAT SAME row's conference, not just be a plausible-
    looking championship string. K2: logged, then falls through to (and is stopped by, single
    row / no other bucket) the structural rule."""
    rows = _champ_notes_game(602001, "Big Ten", "Big 12 Championship")
    with caplog.at_level(logging.WARNING):
        found = identify_conference_championship_games(rows, 2025)
    assert found == {}, found
    assert any("Big 12 Championship" in rec.getMessage() for rec in caplog.records), (
        f"expected a warning naming the notes mismatch, got: "
        f"{[rec.getMessage() for rec in caplog.records]}"
    )


@pytest.mark.parametrize("notes", [
    "SEC Championship Game",   # trailing word after the suffix
    "SEC championship",        # lowercase -- the match predicate is case-sensitive
], ids=["trailing-word", "lowercase"])
def test_notes_suffix_surprise_still_logs_even_though_it_does_not_match(caplog, notes):
    """PR #16 review finding. K2's promise is that a CFBD naming change surfaces as a log line
    rather than a silent miss, but the warning originally fired only on a PREFIX mismatch. A
    SUFFIX surprise -- a trailing word, or different casing -- fell through in silence, which
    would silently reintroduce the very defect the notes path exists to fix, for every
    conference at once and with nothing in the logs to say why.

    The MATCH predicate deliberately stays strict (these still identify nothing); only the LOG
    trigger is loose. Matching loosely instead would let an unrelated game carrying the word
    through, which is the worse error."""
    rows = _champ_notes_game(603001, "SEC", notes)
    with caplog.at_level(logging.WARNING):
        found = identify_conference_championship_games(rows, 2025)
    assert found == {}, found
    assert any("does not END" in rec.getMessage() for rec in caplog.records), (
        f"expected a warning that the notes value contains but does not end with the suffix, "
        f"got: {[rec.getMessage() for rec in caplog.records]}"
    )


# --- 2026 kickoff-classic notes values -- none end with "Championship". ---------------
KICKOFF_CLASSICS_2026 = _rows(
    (701001, 2026, "Kansas State", "Iowa State", "2026-08-27 23:00:00", "Big 12",
     True, "regular"),
)
KICKOFF_CLASSICS_2026[0]["notes"] = "Aer Lingus College Football Classic"
KICKOFF_CLASSICS_2026[1]["notes"] = "Aer Lingus College Football Classic"


def test_2026_kickoff_classic_notes_values_identify_nothing():
    """T1 test 5: none of the three real 2026 non-null notes values end with 'Championship'
    (confirmed live) -- the notes signal must not fire for them, and this single-row fixture
    also can't satisfy the structural rule (only one bucket), so the result is empty."""
    found = identify_conference_championship_games(KICKOFF_CLASSICS_2026, 2026)
    assert found == {}, found


# --- Postseason CFP National Championship -- must not be identified as a CONFERENCE
# championship despite its notes literally containing "Championship". -----------------
def test_postseason_national_championship_notes_identifies_nothing():
    """T1 test 6: season_type='postseason' gates the notes signal exactly like it already
    gates the structural one -- the CFP National Championship is not a conference title
    game no matter what its notes column says."""
    rows = _game(
        401764870, 2025, "Ohio State", "Notre Dame", "2026-01-19 20:00:00", "SEC",
        conference_game=False, season_type="postseason", notes="CFP National Championship",
    )
    found = identify_conference_championship_games(rows, 2025)
    assert found == {}, found


def test_postseason_row_with_a_conference_matching_notes_still_identifies_nothing():
    """T1 test 6b: unlike the CFP National Championship fixture above (whose notes prefix
    doesn't resolve to 'SEC' anyway), THIS row's notes prefix DOES resolve to its own
    conference -- it is excluded purely by season_type != 'regular', proving that gate is
    actually load-bearing on the notes path and not simply redundant with the prefix check."""
    rows = _game(
        605001, 2025, "Georgia", "Alabama", "2025-12-06 20:00:00", "SEC",
        conference_game=False, season_type="postseason", notes="SEC Championship",
    )
    found = identify_conference_championship_games(rows, 2025)
    assert found == {}, found


# --- A conference with BOTH an authoritative notes match and a competing structural
# candidate -- notes must win. -----------------------------------------------------------
def test_notes_match_wins_over_a_competing_structural_candidate():
    """T1 test 7: SEC has a real notes-tagged championship row (conference_game=False) AND a
    separate, self-sufficient structural candidate -- a regular conference_game=True slate
    plus a LATER lone conference_game=True game -- that would, ON ITS OWN, satisfy the
    structural rule and get identified as a DIFFERENT game_id (603003) if the structural rule
    ran for this conference. The notes match must win outright, proving the structural rule
    does not merely lose a tie-break but does not run at all once notes resolves the
    conference."""
    notes_game = _champ_notes_game(603001, "SEC", "SEC Championship",
                                    home="Georgia", away="Alabama",
                                    start_date="2025-12-06 20:00:00")
    regular_slate = _game(
        603002, 2025, "LSU", "Texas A&M", "2025-11-29 17:00:00", "SEC",
        conference_game=True, season_type="regular",
    )
    # Alone in its own later bucket -- structurally indistinguishable from a real title game,
    # so this WOULD be (wrongly) identified as SEC's championship if notes did not take
    # priority and suppress the structural rule for this conference entirely.
    competing_structural = _game(
        603003, 2025, "Missouri", "Vanderbilt", "2025-12-13 16:00:00", "SEC",
        conference_game=True, season_type="regular",
    )
    rows = notes_game + regular_slate + competing_structural
    found = identify_conference_championship_games(rows, 2025)
    assert found == {"SEC": 603001}, found


# --- Two distinct authoritative notes matches in the same conference -- ambiguity. -----
def test_two_distinct_notes_matches_in_one_conference_identify_nothing_and_warn(caplog):
    """T1 ambiguity handling: a conference plays at most one championship game, so two
    DISTINCT game_ids both producing an authoritative notes match in the same season is a
    'shouldn't happen' -- logged, and the whole conference falls through to the structural
    rule (which also finds nothing here, since neither row is conference_game=True)."""
    game_a = _champ_notes_game(604001, "Big 12", "Big 12 Championship",
                                home="Colorado", away="Iowa State",
                                start_date="2025-12-06 12:00:00")
    game_b = _champ_notes_game(604002, "Big 12", "Big 12 Championship",
                                home="Kansas State", away="BYU",
                                start_date="2025-12-06 15:30:00")
    rows = game_a + game_b
    with caplog.at_level(logging.WARNING):
        found = identify_conference_championship_games(rows, 2025)
    assert found == {}, found
    assert any("Big 12" in rec.getMessage() for rec in caplog.records), (
        f"expected a warning naming the ambiguous conference, got: "
        f"{[rec.getMessage() for rec in caplog.records]}"
    )


def test_two_team_conference_playing_twice_is_not_a_championship():
    """The real 2025 Pac-12: a two-team remnant that met twice.

    Oregon State and Washington State were the entire Pac-12 in 2024 and 2025, and played
    each other on Nov 1 and Nov 29 2025. The second meeting sits alone in a week bucket,
    strictly later than the first -- which satisfies every structural test the
    championship rule applies. Without a member-count gate the rule publishes a Pac-12
    championship game that never existed: the same fabrication as the "Wake Forest vs
    Duke" matchup, reached by a different route.

    This became reachable only when the Pac-12 joined the identification gate. Verified
    against production: the Pac-12 had exactly 2 members with conference games in both
    2024 and 2025, while every conference that really does hold a title game had 12-17.
    """
    rows = _rows(
        (401752900, 2025, "Oregon State", "Washington State", "2025-11-01 20:00:00", "Pac-12", True),
        (401752946, 2025, "Washington State", "Oregon State", "2025-11-29 20:00:00", "Pac-12", True),
    )
    found = identify_conference_championship_games(rows, 2025)
    assert found == {}, f"invented a championship for a two-team conference: {found}"


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
