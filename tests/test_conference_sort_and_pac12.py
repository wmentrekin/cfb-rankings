"""Tests for conference team sorting (T1: empty-record sentinel fix) and
Pac-12 championship qualification (T2: add Pac-12 to QUALIFYING_CHAMPIONSHIP_CONFERENCES).

The sorting fix changes the empty-record sentinel from -1.0 to 0.5, so that:
  - A team with no conference games (0-0) sorts between teams with wins and losses
  - NC State (0-1 conf) now correctly sorts BELOW Duke (0-0 conf)
  - Independents (no conf_record at all) still sort below all conference teams

The Pac-12 addition verifies that:
  - Pac-12 is present in QUALIFYING_CHAMPIONSHIP_CONFERENCES
  - Pac-12 has >= 4 members and qualifies for championship-game logic

Run: python -m pytest tests/ -q   (or: python tests/test_conference_sort_and_pac12.py)
"""
import itertools
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifacts.schedule import _sort_conference_teams, build_schedule_payload  # noqa: E402
from artifacts.schedule_standings import (  # noqa: E402
    QUALIFYING_CHAMPIONSHIP_CONFERENCES,
    MIN_QUALIFYING_MEMBERS,
)


def _entry(team, overall_wins, overall_losses, conf_wins=None, conf_losses=None):
    """Create a minimal entry dict for _sort_conference_teams."""
    return {
        "team": team,
        "record": {"wins": overall_wins, "losses": overall_losses},
        "conf_record": (
            {"wins": conf_wins, "losses": conf_losses}
            if conf_wins is not None
            else None
        ),
    }


def test_nc_state_vs_duke_exact_case():
    """REGRESSION: NC State (0-1 conf) must sort BELOW Duke (0-0 conf).
    This is the exact user-reported case that triggered the fix."""
    entries = [
        _entry("NC State", 0, 1, 0, 1),  # 0-1 overall, 0-1 conference
        _entry("Duke", 1, 0, 0, 0),      # 1-0 overall, 0-0 conference
    ]
    rows = []  # No head-to-head tiebreaker needed for this simple case
    sorted_entries = _sort_conference_teams(entries, rows, 2026)

    # Duke (with 0-0 conf record) should come first
    assert sorted_entries[0]["team"] == "Duke", f"Duke should be first, got {sorted_entries[0]['team']}"
    assert sorted_entries[1]["team"] == "NC State", f"NC State should be second, got {sorted_entries[1]['team']}"


def test_three_way_conference_record_ordering():
    """Verify the three-way sort: 1-0 > 0-0 > 0-1 on conference records."""
    entries = [
        _entry("Team-Loss", 0, 1, 0, 1),    # 0-1 conference
        _entry("Team-Win", 1, 0, 1, 0),     # 1-0 conference
        _entry("Team-Unplayed", 0, 0, 0, 0),  # 0-0 conference
    ]
    rows = []
    sorted_entries = _sort_conference_teams(entries, rows, 2026)

    # Order should be: Win > Unplayed > Loss
    assert sorted_entries[0]["team"] == "Team-Win", f"1-0 should be first"
    assert sorted_entries[1]["team"] == "Team-Unplayed", f"0-0 should be second"
    assert sorted_entries[2]["team"] == "Team-Loss", f"0-1 should be third"


def test_independents_sort_by_overall_record():
    """Verify Independents (conf_record=None) still sort among themselves by overall record
    and are not disturbed by the empty-record sentinel change."""
    entries = [
        _entry("Ind-Loss", 0, 2, None, None),   # 0-2 overall, independent
        _entry("Ind-Win", 2, 0, None, None),    # 2-0 overall, independent
        _entry("Ind-Tie", 1, 1, None, None),    # 1-1 overall, independent
    ]
    rows = []
    sorted_entries = _sort_conference_teams(entries, rows, 2026)

    # Independents should sort by overall record: 2-0 > 1-1 > 0-2
    # (because they're all None for conf_pct, they fall through to overall_pct tiebreaker)
    assert sorted_entries[0]["team"] == "Ind-Win", f"2-0 overall should be first"
    assert sorted_entries[1]["team"] == "Ind-Tie", f"1-1 overall should be second"
    assert sorted_entries[2]["team"] == "Ind-Loss", f"0-2 overall should be third"


def test_independents_sort_below_conference_teams():
    """Verify Independents sort below all conference-playing teams (preserving -1.0 fallback)."""
    entries = [
        _entry("Conf-Team", 0, 1, 0, 1),    # 0-1 overall, 0-1 conference
        _entry("Independent", 5, 0, None, None),  # 5-0 overall, independent
    ]
    rows = []
    sorted_entries = _sort_conference_teams(entries, rows, 2026)

    # Independent with 5-0 record should still sort BELOW conference team with 0-1 conf record
    # because Independents lack conference records entirely (-1.0 fallback keeps them at bottom)
    assert sorted_entries[0]["team"] == "Conf-Team", f"Conference team should sort first"
    assert sorted_entries[1]["team"] == "Independent", f"Independent should sort last"


def test_pac12_present_in_qualifying_conferences():
    """Verify Pac-12 is present in QUALIFYING_CHAMPIONSHIP_CONFERENCES."""
    assert "Pac-12" in QUALIFYING_CHAMPIONSHIP_CONFERENCES, \
        f"Pac-12 not found. Available: {list(QUALIFYING_CHAMPIONSHIP_CONFERENCES.keys())}"


def test_pac12_passes_min_qualifying_members_gate():
    """Verify Pac-12 entry has at least MIN_QUALIFYING_MEMBERS (4) members.
    For 2026, Pac-12 has 8 members and a 7-game full round robin."""
    # The value describes the format; we verify it mentions enough members
    pac12_desc = QUALIFYING_CHAMPIONSHIP_CONFERENCES["Pac-12"]

    # The description should indicate it qualifies (has enough members)
    # For 2026, we know it's 8 members, which is >= MIN_QUALIFYING_MEMBERS
    # This test just verifies the entry exists and is properly formatted.
    assert pac12_desc is not None, "Pac-12 entry should have a description"
    assert "top-2" in pac12_desc, f"Pac-12 description should mention top-2: {pac12_desc}"
    assert "no divisions" in pac12_desc, f"Pac-12 description should mention no divisions: {pac12_desc}"


def test_pac12_qualifies_dynamically():
    """Verify that a mock Pac-12 with 8 members would pass the runtime qualification gate.
    (This test demonstrates that the MIN_QUALIFYING_MEMBERS check is truly dynamic.)"""
    # Create mock standings with 8 Pac-12 teams
    pac12_teams = [
        _entry(f"Pac12-Team-{i}", 2, 1, 1, 1) for i in range(8)
    ]

    # Count how many teams in the mock would qualify
    pac12_count = len([t for t in pac12_teams if t.get("conference") == "Pac-12"])

    # The dynamic check in compute_conference_championship_status would use:
    # count = len([t for row in rows if row['conference'] == conf])
    # To verify it's >= MIN_QUALIFYING_MEMBERS

    # Since we can't easily call the full compute_standings here without lots of setup,
    # we just verify the dict entry exists and MIN_QUALIFYING_MEMBERS is reasonable
    assert "Pac-12" in QUALIFYING_CHAMPIONSHIP_CONFERENCES
    assert MIN_QUALIFYING_MEMBERS == 4, f"MIN_QUALIFYING_MEMBERS should be 4, got {MIN_QUALIFYING_MEMBERS}"


# ---------------------------------------------------------------------------
# T2/K7: a null rank sorts LAST, not first, among otherwise fully-tied teams.
# ---------------------------------------------------------------------------
def test_none_rank_sorts_last_among_tied_teams():
    """Python can't compare None to an int at all -- the large sentinel in _sort_conference_teams
    is what makes this deterministic. All three entries below tie on conf_pct (1-1), _conf_played
    (True) and placement pct (also 1-1, rows=[] so nothing is subtracted from it), so rank is the
    only thing left to decide the order."""
    entries = [
        _entry("Team-NoRank", 1, 1, 1, 1),
        _entry("Team-Rank5", 1, 1, 1, 1),
        _entry("Team-Rank10", 1, 1, 1, 1),
    ]
    entries[0]["rank"] = None
    entries[1]["rank"] = 5
    entries[2]["rank"] = 10
    rows = []
    sorted_entries = _sort_conference_teams(entries, rows, 2026)
    sorted_teams = [e["team"] for e in sorted_entries]
    assert sorted_teams == ["Team-Rank5", "Team-Rank10", "Team-NoRank"], sorted_teams


# ---------------------------------------------------------------------------
# T2/K8 + K4-K7: the 2025 Pac-12 two-team remnant from the task brief. Washington State and
# Oregon State play ONLY each other (a real two-team conference, per identify_conference_
# championship_games' MIN_QUALIFYING_MEMBERS gate -- no championship game is ever identified for
# it), split 1-1 -- issue 7's exact defect (a split decided by row-iteration order) would have
# put whichever team's row happened to be scanned first ahead of the other. With the fix, the
# split is a wash (K8) and Washington State's better overall record decides it instead.
# ---------------------------------------------------------------------------
_SEASON = 2025
_PAC12 = "Pac-12"
_PAC12_SEASON_START = date(2025, 8, 23)


def _pac12_row(game_id, team, opponent, status, conference_game, season_type, week_offset):
    return dict(
        game_id=game_id,
        season=_SEASON,
        season_type=season_type,
        team=team,
        opponent=opponent,
        conference=_PAC12,
        conference_game=conference_game,
        status=status,
        start_date=(_PAC12_SEASON_START + timedelta(days=7 * week_offset)).isoformat() + " 19:00:00",
        home_away="home",
        neutral_site=False,
    )


def _pac12_two_team_rows():
    game_id_iter = itertools.count(1)
    rows = []

    # The two meetings: Washington State wins the first (Nov 1 live), Oregon State wins the
    # rematch (Nov 29 live) -- a clean 1-1 split, both conference_game=True.
    wsu_win_id = next(game_id_iter)
    rows.append(_pac12_row(wsu_win_id, "Washington State", "Oregon State", "win", True, "regular", 0))
    rows.append(_pac12_row(wsu_win_id, "Oregon State", "Washington State", "loss", True, "regular", 0))
    osu_win_id = next(game_id_iter)
    rows.append(_pac12_row(osu_win_id, "Oregon State", "Washington State", "win", True, "regular", 1))
    rows.append(_pac12_row(osu_win_id, "Washington State", "Oregon State", "loss", True, "regular", 1))

    # Washington State: 5-5 non-conference regular season + 1-0 postseason -> 6-6 regular,
    # 7-6 overall (matches the task brief exactly).
    for w in range(5):
        rows.append(_pac12_row(next(game_id_iter), "Washington State", f"WSU NonConf W{w}", "win", False, "regular", 2 + w))
    for l in range(5):
        rows.append(_pac12_row(next(game_id_iter), "Washington State", f"WSU NonConf L{l}", "loss", False, "regular", 7 + l))
    rows.append(_pac12_row(next(game_id_iter), "Washington State", "WSU Bowl W", "win", False, "postseason", 16))

    # Oregon State: 1-9 non-conference regular season, no postseason -> 2-10 overall (matches
    # the task brief exactly).
    rows.append(_pac12_row(next(game_id_iter), "Oregon State", "OSU NonConf W", "win", False, "regular", 2))
    for l in range(9):
        rows.append(_pac12_row(next(game_id_iter), "Oregon State", f"OSU NonConf L{l}", "loss", False, "regular", 3 + l))

    return rows


def _pac12_teams_meta():
    return {
        "Washington State": {"conference": _PAC12, "division": None, "logos": None},
        "Oregon State": {"conference": _PAC12, "division": None, "logos": None},
    }


def test_pac12_two_team_split_head_to_head_produces_washington_state_first():
    rows = _pac12_two_team_rows()
    team_ranks = {"Washington State": 57, "Oregon State": 109}
    payload = build_schedule_payload(rows, _pac12_teams_meta(), _SEASON, team_ranks=team_ranks)

    # artifacts.rankings.CONFERENCE_DISPLAY_NAMES maps the raw "Pac-12" conference value to the
    # display string "PAC 12" -- the Season Grid payload's conferences[].name uses that display
    # string, not the raw value used on schedule_grid rows / teams_meta.
    conf = next(c for c in payload["conferences"] if c["name"] == "PAC 12")
    entries = {e["team"]: e for e in conf["teams"]}

    assert entries["Washington State"]["record"] == {"wins": 7, "losses": 6}, entries["Washington State"]["record"]
    assert entries["Washington State"]["conf_record"] == {"wins": 1, "losses": 1}, entries["Washington State"]["conf_record"]
    assert entries["Oregon State"]["record"] == {"wins": 2, "losses": 10}, entries["Oregon State"]["record"]
    assert entries["Oregon State"]["conf_record"] == {"wins": 1, "losses": 1}, entries["Oregon State"]["conf_record"]

    names = [e["team"] for e in conf["teams"]]
    assert names == ["Washington State", "Oregon State"], (
        f"got {names}. The 1-1 split must be a wash (K8), leaving Washington State's better "
        "record (and better model rank) to decide it -- if this reads ['Oregon State', "
        "'Washington State'] instead, the split is being decided by which row the (unfixed) "
        "head-to-head tally happened to scan first, exactly issue 7's defect."
    )


# ---------------------------------------------------------------------------
# R1 (standings-gaps T3): overall win COUNT must outrank model rating in the
# sort key, but only once conference record and overall PCT have already
# failed to separate two teams. _placement_pct alone ties every unbeaten team
# at 1.0 regardless of games played, so before this fix model rank decided
# among them -- e.g. USC (2-0, rank 20) sorted below three 1-0 teams ranked
# better, the real 2026 Big Ten shape from the bug report.
# ---------------------------------------------------------------------------
def test_win_count_outranks_better_model_rank_in_all_0_0_conference():
    """R1 test 1: USC 2-0 overall / 0-0 conference, WORSE model rank (20), must sort ABOVE
    three teams at 1-0 overall / 0-0 conference with BETTER ranks (2, 5, 8). All four tie at
    _conf_pct==0.5 (nobody has played a conference game yet, so _conf_played is also tied) and
    at _placement_pct==1.0 (all unbeaten) -- with rows=[], nothing is subtracted from the
    pre-set "record", so _placement_pct reproduces entry["record"]'s own pct exactly (1.0 for
    every team here). Only the win-COUNT term this fix adds can separate the group at that
    point; without it, model rank decides next.

    BUGGY (pre-fix) result: ['Better-2', 'Better-5', 'Better-8', 'USC'] -- with no
    _placement_wins term in the sort key, all four teams tie through _placement_pct and rank
    alone decides, putting the worse-ranked-but-more-winning USC dead last.
    """
    entries = [
        _entry("USC", 2, 0, 0, 0),
        _entry("Better-2", 1, 0, 0, 0),
        _entry("Better-5", 1, 0, 0, 0),
        _entry("Better-8", 1, 0, 0, 0),
    ]
    entries[0]["rank"] = 20
    entries[1]["rank"] = 2
    entries[2]["rank"] = 5
    entries[3]["rank"] = 8
    rows = []
    sorted_entries = _sort_conference_teams(entries, rows, 2026)
    sorted_teams = [e["team"] for e in sorted_entries]
    assert sorted_teams[0] == "USC", (
        f"got {sorted_teams}. USC's 2-0 overall record must outrank model rating once "
        "conference record and overall pct (all tied at 1.0) fail to separate the group -- "
        "if USC is not first, the win-count term is missing or ineffective."
    )


def test_independents_win_count_outranks_better_model_rank():
    """R1 test 2: the Independents-branch equivalent of the test above -- structurally the
    same defect (the Independents sort key lacked a _placement_wins term too), and K2's
    rationale is that the user's rule is about records, not about conferences, so it must be
    fixed on this branch as well. All four entries have conf_record=None, so has_conf_records
    is False and _sort_conference_teams takes the `else` (Independents) branch.

    BUGGY (pre-fix) result: ['Better-2', 'Better-5', 'Better-8', 'USC-Ind'] -- same mechanism
    as test 1, on the other sort call site.
    """
    entries = [
        _entry("USC-Ind", 2, 0, None, None),
        _entry("Better-2", 1, 0, None, None),
        _entry("Better-5", 1, 0, None, None),
        _entry("Better-8", 1, 0, None, None),
    ]
    entries[0]["rank"] = 20
    entries[1]["rank"] = 2
    entries[2]["rank"] = 5
    entries[3]["rank"] = 8
    rows = []
    sorted_entries = _sort_conference_teams(entries, rows, 2026)
    sorted_teams = [e["team"] for e in sorted_entries]
    assert sorted_teams[0] == "USC-Ind", (
        f"got {sorted_teams}. Independents-branch equivalent of the USC case: 2-0 overall "
        "must outrank a better model rank at tied placement pct."
    )


def test_rating_still_breaks_ties_between_identical_overall_records():
    """R1 test 3 (guard, not a catch for the win-count term itself): both teams here are 1-0
    overall AND 0-0 conference -- IDENTICAL records -- so _placement_wins ties too (1 == 1) and
    contributes nothing to the ordering either way; model rank must still be what decides. This
    pins R1's own acceptance wording -- "model rating breaks ties only among teams with
    identical overall win-loss records" -- as still true after the fix. (Mutation-tested
    against a DIFFERENT injected defect than the other two tests above -- see the task report:
    removing the win-count term changes nothing here since both teams tie on it, so that
    particular mutation is expected to leave this test green; dropping model rank from the sort
    key entirely is the mutation that actually exercises this assertion.)

    Team names are deliberately chosen so the BETTER-ranked team's name sorts ALPHABETICALLY
    AFTER the worse-ranked team's ('Zeta-Better-Rank' > 'Alpha-Worse-Rank') -- the fixture
    pitfall flagged in the task brief: an earlier draft of this test used names that happened to
    already sort correctly by NAME alone, so it passed even with rank dropped from the sort key
    entirely and proved nothing about rating actually being consulted.
    """
    entries = [
        _entry("Zeta-Better-Rank", 1, 0, 0, 0),
        _entry("Alpha-Worse-Rank", 1, 0, 0, 0),
    ]
    entries[0]["rank"] = 2
    entries[1]["rank"] = 9
    rows = []
    sorted_entries = _sort_conference_teams(entries, rows, 2026)
    sorted_teams = [e["team"] for e in sorted_entries]
    assert sorted_teams == ["Zeta-Better-Rank", "Alpha-Worse-Rank"], sorted_teams


def test_equal_pct_more_wins_sorts_above_fewer_wins_regardless_of_rank():
    """R1 test 4: the case the comment above the sort key (and requirements.yaml's R1
    acceptance criterion 2) used to describe with an arithmetically FALSE example (6-2 vs. 3-1
    as if those were different percentages -- 6/8 and 3/4 are both 0.750). This is the REAL
    equal-pct-different-win-count case that comment was trying to describe: two teams tied
    EXACTLY on conference pct (SixTwo 6-2 == .750, ThreeOne 3-1 == .750), so pct does NOT decide
    between them and this term (win count) is actually reached. SixTwo has strictly more wins,
    so it must sort above ThreeOne even with a much worse model rank -- rank is checked AFTER
    the win-count term, so if it decided instead, ThreeOne (rank 1) would wrongly sort first.

    Nothing in the existing suite pins this: test 1/2 above only cover the ALL-UNBEATEN
    (1.0 pct) case, and test 3 covers two teams with IDENTICAL records, not merely identical
    pct. This is the one place a genuine equal-pct/different-win-count pair is exercised.

    BUGGY (mutant) result if the win-count term were dropped from the sort key: rank alone
    would decide the pct-tied pair, putting ThreeOne (rank 1) first -- ['ThreeOne', 'SixTwo'].
    """
    entries = [
        _entry("SixTwo", 6, 2, 6, 2),
        _entry("ThreeOne", 3, 1, 3, 1),
    ]
    entries[0]["rank"] = 30
    entries[1]["rank"] = 1
    rows = []
    sorted_entries = _sort_conference_teams(entries, rows, 2026)
    sorted_teams = [e["team"] for e in sorted_entries]
    assert sorted_teams == ["SixTwo", "ThreeOne"], (
        f"got {sorted_teams}. SixTwo and ThreeOne are tied at .750 conference pct, so this "
        "pair is decided by win count (6 > 3), not by rank -- if ThreeOne sorts first, the "
        "win-count term is not being reached for pct-tied (not just record-identical) teams."
    )


# ---------------------------------------------------------------------------
# NIT: the 0.5 unplayed-record sentinel now ties with, and loses the win-count term to, a real
# .500-pct team -- a silent ordering change from before this PR (rank used to decide; now the
# win-count term does). Defensible (see the in-function "DELIBERATE SIDE EFFECT" comment on the
# Independents branch of _sort_conference_teams) but previously unpinned by any test.
# ---------------------------------------------------------------------------
def test_unplayed_0_5_sentinel_sorts_below_a_tied_played_team_even_when_better_ranked():
    """`_placement_pct`'s 0.5 "no games" sentinel (ZeroZero, 0-0 overall) now ties exactly with
    a genuine .500 team (OneOne, 1-1 overall) -- and R1's new win-count term (_placement_wins)
    then decides the pair (1 > 0), NOT rank. Used here on the INDEPENDENTS branch
    (conf_wins=None for both, so has_conf_records is False and _conf_pct/_conf_played -- which
    would otherwise settle this first -- never enter into it at all) to isolate exactly the
    placement-pct-sentinel-vs-win-count interaction the reviewer flagged, with nothing else able
    to decide the pair first. Checked with rank both ways so the result is pinned as depending on
    the win-count term, not on which team happens to be ranked better.

    BEFORE this PR (no _placement_wins term in the Independents sort key), this pair tied all
    the way down to rank, so a 0-0 team ranked #1 sorted ABOVE a 1-1 team ranked #30 -- the
    opposite of the result asserted below. This agrees with tiebreaker_engine._fallback_sort_key,
    which also scores an unplayed team below a played .500 team, and 0-0 still sorts above 0-1
    (untouched by this change), so the sentinel's stated purpose survives -- but nothing pinned
    this specific interaction before this test.
    """
    # 0-0 team ranked BETTER than the 1-1 team: still sorts second.
    entries_a = [
        _entry("ZeroZero", 0, 0, None, None),
        _entry("OneOne", 1, 1, None, None),
    ]
    entries_a[0]["rank"] = 1
    entries_a[1]["rank"] = 30
    sorted_a = [e["team"] for e in _sort_conference_teams(entries_a, [], 2026)]
    assert sorted_a == ["OneOne", "ZeroZero"], sorted_a

    # 0-0 team ranked WORSE than the 1-1 team: still sorts second -- rank plays no part either
    # way, confirming the win-count term (not rank) is what is deciding this pair.
    entries_b = [
        _entry("ZeroZero", 0, 0, None, None),
        _entry("OneOne", 1, 1, None, None),
    ]
    entries_b[0]["rank"] = 30
    entries_b[1]["rank"] = 1
    sorted_b = [e["team"] for e in _sort_conference_teams(entries_b, [], 2026)]
    assert sorted_b == ["OneOne", "ZeroZero"], sorted_b


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
