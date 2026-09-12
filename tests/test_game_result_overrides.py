"""Tests for the manual game-result override mechanism (database/game_overrides.py,
database/game_derivations.py, database/game_result_overrides.json).

No live database and no network: `apply_game_result_overrides` accepts an injectable
`engine`, so these tests exercise the real parameterized text() SQL against a private
in-memory SQLite database rather than mocking the SQL layer -- the same statements run
against Postgres in production run here unmodified.

The design and acceptance criteria this file covers are described in
database/game_overrides.py's own docstring, which also documents the override entry format.

Run: python -m pytest tests/ -q   (or: python tests/test_game_result_overrides.py)
"""
import ast
import contextlib
import json
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd  # noqa: E402
from pandas.testing import assert_series_equal  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import database.game_overrides as game_overrides  # noqa: E402
from database.game_derivations import compute_alpha, compute_margin, compute_winner  # noqa: E402
from database.game_overrides import apply_game_result_overrides  # noqa: E402

REQUIRED_ENTRY_FIELDS = {
    "game_id", "season", "week", "season_type", "home_team", "away_team",
    "home_score", "away_score", "reason", "declared_on",
}


# ---------------------------------------------------------------------------
# Fake `games` table: a private in-memory SQLite database, shared across the
# connections apply_game_result_overrides opens via engine.begin() (StaticPool
# keeps them on the one real connection -- plain "sqlite:///:memory:" hands
# out a fresh, empty database per connection). No live database, no network.
# ---------------------------------------------------------------------------
def _make_engine(rows):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE games ("
            "id INTEGER PRIMARY KEY, season INTEGER, week INTEGER, season_type TEXT, "
            "home_team TEXT, away_team TEXT, home_score INTEGER, away_score INTEGER, "
            "neutral_site INTEGER, winner TEXT, margin INTEGER, alpha REAL)"
        ))
        for row in rows:
            conn.execute(text(
                "INSERT INTO games (id, season, week, season_type, home_team, away_team, "
                "home_score, away_score, neutral_site, winner, margin, alpha) VALUES "
                "(:id, :season, :week, :season_type, :home_team, :away_team, :home_score, "
                ":away_score, :neutral_site, :winner, :margin, :alpha)"
            ), row)
    return engine


def _fetch(engine, game_id):
    with engine.begin() as conn:
        row = conn.execute(text("SELECT * FROM games WHERE id = :id"), {"id": game_id}).mappings().first()
    return dict(row)


@contextlib.contextmanager
def _with_overrides(overrides):
    """Temporarily replace game_overrides.load_overrides with a fixed list.

    Lets tests exercise apply_game_result_overrides against synthetic override
    entries without touching the shipped database/game_result_overrides.json
    (that file gets its own dedicated test below).
    """
    original = game_overrides.load_overrides
    game_overrides.load_overrides = lambda path=None: overrides
    try:
        yield
    finally:
        game_overrides.load_overrides = original


def _override_entry():
    """The seeded production entry (database/game_result_overrides.json), reproduced
    as a plain dict so tests can mutate it without touching the shipped file."""
    return {
        "game_id": 401858428, "season": 2026, "week": 1, "season_type": "regular",
        "home_team": "Michigan", "away_team": "Western Michigan",
        "home_score": 7, "away_score": 12,
        "reason": "test fixture", "declared_on": "2026-09-09",
    }


def _write_temp_overrides_file(entries):
    """A standalone database/game_result_overrides.json-shaped file for load_overrides
    tests that need to exercise real file/JSON parsing rather than the in-memory
    _with_overrides monkeypatch. Caller is responsible for unlinking the returned path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"version": 1, "overrides": entries}, f)
        return f.name


def _stored_row_matching(entry, **field_overrides):
    """The pre-override `games` row this entry is meant to correct -- the real CFBD
    result (Michigan 13, Western Michigan 12) before the operator's correction."""
    row = {
        "id": entry["game_id"], "season": entry["season"], "week": entry["week"],
        "season_type": entry["season_type"], "home_team": entry["home_team"],
        "away_team": entry["away_team"],
        "home_score": 13, "away_score": 12, "neutral_site": 0,
        "winner": "Michigan", "margin": 1, "alpha": 0.8,
    }
    row.update(field_overrides)
    return row


# ===========================================================================
# Derivation parity (AC4): the ingest DataFrame path and the extracted
# helpers must produce byte-identical winner/margin/alpha, dtype included.
# ===========================================================================
def _fixture_rows_df():
    """Real 2026 games plus the two cases the seeded override alone can't exercise:
    a neutral-site game (Florida State vs Alabama) and an unplayed 0-0 game
    (Notre Dame vs Purdue, still on the future schedule at fixture-authoring time)."""
    return pd.DataFrame({
        "homeTeam": ["Michigan", "Florida State", "Notre Dame"],
        "awayTeam": ["Western Michigan", "Alabama", "Purdue"],
        "homePoints": pd.array([13, 20, 0], dtype="Int64"),
        "awayPoints": pd.array([12, 24, 0], dtype="Int64"),
        "neutralSite": pd.array([False, True, False], dtype="bool"),
    })


def _old_inline_derivation(df):
    """Byte-for-byte the pre-T1 expressions this repo shipped in get_games.py."""
    out = df.copy()
    out["margin"] = abs(out["homePoints"] - out["awayPoints"])
    out["winner"] = out.apply(
        lambda row: None if (row["homePoints"] == 0 and row["awayPoints"] == 0)
        else (row["homeTeam"] if row["homePoints"] > row["awayPoints"] else row["awayTeam"]),
        axis=1,
    )
    out["alpha"] = out.apply(
        lambda row: 1 if row["neutralSite"] else (0.8 if row["homeTeam"] == row["winner"] else 1.2),
        axis=1,
    )
    out["margin"] = out["margin"].astype("Int64")
    out["alpha"] = out["alpha"].astype(float)
    return out


def _new_helper_derivation(df):
    """The T1 shape: margin/winner/alpha all via .apply(axis=1) calling the extracted
    helpers, with the same .astype recasts get_games.py keeps afterwards."""
    out = df.copy()
    out["margin"] = out.apply(lambda row: compute_margin(row["homePoints"], row["awayPoints"]), axis=1)
    out["winner"] = out.apply(
        lambda row: compute_winner(row["homeTeam"], row["awayTeam"], row["homePoints"], row["awayPoints"]),
        axis=1,
    )
    out["alpha"] = out.apply(
        lambda row: compute_alpha(row["homeTeam"], row["winner"], row["neutralSite"]), axis=1,
    )
    out["margin"] = out["margin"].astype("Int64")
    out["alpha"] = out["alpha"].astype(float)
    return out


def test_derivation_parity_between_ingest_path_and_extracted_helpers():
    df = _fixture_rows_df()
    old = _old_inline_derivation(df)
    new = _new_helper_derivation(df)

    assert_series_equal(old["margin"], new["margin"], check_dtype=True, check_names=False)
    assert_series_equal(old["winner"], new["winner"], check_dtype=True, check_names=False)
    assert_series_equal(old["alpha"], new["alpha"], check_dtype=True, check_names=False)


# ===========================================================================
# AC3: the seeded entry's numbers.
# ===========================================================================
def test_seeded_entry_yields_expected_winner_margin_alpha():
    entry = _override_entry()
    assert compute_winner(entry["home_team"], entry["away_team"], entry["home_score"], entry["away_score"]) == "Western Michigan"
    assert compute_margin(entry["home_score"], entry["away_score"]) == 5
    assert compute_alpha(entry["home_team"], "Western Michigan", False) == 1.2


def test_seeded_entry_applied_end_to_end_against_a_stored_row():
    entry = _override_entry()
    stored = _stored_row_matching(entry)  # pre-override: Michigan 13, Western Michigan 12
    engine = _make_engine([stored])

    with _with_overrides([entry]):
        result = apply_game_result_overrides(2026, engine=engine)

    assert result["applied"] == [401858428]
    assert result["failures"] == []
    row = _fetch(engine, 401858428)
    assert row["home_score"] == 7
    assert row["away_score"] == 12
    assert row["winner"] == "Western Michigan"
    assert row["margin"] == 5
    assert row["alpha"] == 1.2


# ===========================================================================
# AC5: an identity mismatch on any of the five fields writes nothing and is
# reported as a failure -- checked independently for each field.
# ===========================================================================
def _assert_mismatch_is_a_failure(field, wrong_value):
    entry = _override_entry()
    stored = _stored_row_matching(entry)
    stored[field] = wrong_value
    engine = _make_engine([stored])

    with _with_overrides([entry]):
        result = apply_game_result_overrides(entry["season"], engine=engine)

    assert result["applied"] == [], f"a {field} mismatch must not be applied"
    assert len(result["failures"]) == 1, f"a {field} mismatch must be reported exactly once"
    assert result["failures"][0]["game_id"] == entry["game_id"]
    row = _fetch(engine, entry["game_id"])
    assert row["home_score"] == stored["home_score"], f"a {field} mismatch must write nothing"
    assert row["away_score"] == stored["away_score"]
    assert row["winner"] == stored["winner"]


def test_identity_mismatch_on_season_writes_nothing_and_is_reported():
    _assert_mismatch_is_a_failure("season", 2025)


def test_identity_mismatch_on_week_writes_nothing_and_is_reported():
    _assert_mismatch_is_a_failure("week", 2)


def test_identity_mismatch_on_season_type_writes_nothing_and_is_reported():
    _assert_mismatch_is_a_failure("season_type", "postseason")


def test_identity_mismatch_on_home_team_writes_nothing_and_is_reported():
    _assert_mismatch_is_a_failure("home_team", "Ohio State")


def test_identity_mismatch_on_away_team_writes_nothing_and_is_reported():
    _assert_mismatch_is_a_failure("away_team", "Michigan State")


# ===========================================================================
# AC6: an override for a different season is a silent no-op, not a failure.
# ===========================================================================
def test_wrong_season_entry_is_a_silent_noop():
    entry = _override_entry()  # season 2026
    stored = _stored_row_matching(entry)
    engine = _make_engine([stored])

    with _with_overrides([entry]):
        result = apply_game_result_overrides(2025, engine=engine)  # a different season

    assert result["applied"] == []
    assert result["failures"] == [], "a wrong-season entry must be a no-op, not a failure"
    assert result["skipped_other_season"] == 1
    row = _fetch(engine, entry["game_id"])
    assert row["home_score"] == stored["home_score"], "wrong-season entry must not touch the row"


# ===========================================================================
# Malformed / missing override file: a failure, never a crash, never a
# silently empty result.
# ===========================================================================
def test_load_overrides_raises_on_a_missing_file():
    try:
        game_overrides.load_overrides(path="/nonexistent/does-not-exist.json")
        assert False, "load_overrides should have raised for a missing file"
    except (FileNotFoundError, OSError):
        pass


def test_load_overrides_raises_on_malformed_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{ this is not valid json")
        bad_path = f.name
    try:
        raised = False
        try:
            game_overrides.load_overrides(path=bad_path)
        except Exception:
            raised = True
        assert raised, "load_overrides should have raised for malformed JSON"
    finally:
        Path(bad_path).unlink(missing_ok=True)


def test_apply_reports_a_failure_for_a_missing_override_file_without_crashing():
    original_path = game_overrides.DEFAULT_OVERRIDES_PATH
    game_overrides.DEFAULT_OVERRIDES_PATH = Path("/nonexistent/does-not-exist.json")
    try:
        engine = _make_engine([])
        result = apply_game_result_overrides(2026, engine=engine)
    finally:
        game_overrides.DEFAULT_OVERRIDES_PATH = original_path

    assert result["applied"] == []
    assert result["failures"], "a missing override file must be reported as a failure, not swallowed silently"
    assert result["skipped_other_season"] == 0


# ===========================================================================
# AC7a: applying twice is idempotent -- the second pass derives from the
# entry's declared scores, not from the already-corrected row.
# ===========================================================================
def test_applying_twice_is_idempotent():
    entry = _override_entry()
    stored = _stored_row_matching(entry)  # pre-override: Michigan 13, Western Michigan 12
    engine = _make_engine([stored])

    with _with_overrides([entry]):
        first = apply_game_result_overrides(2026, engine=engine)
        row_after_first = _fetch(engine, entry["game_id"])
        second = apply_game_result_overrides(2026, engine=engine)
        row_after_second = _fetch(engine, entry["game_id"])

    assert first["applied"] == [entry["game_id"]]
    assert second["applied"] == [entry["game_id"]], "the second pass must also apply cleanly"
    assert second["failures"] == []
    assert row_after_first == row_after_second, "a second pass must leave the row identical"
    assert row_after_second["winner"] == "Western Michigan"
    assert row_after_second["margin"] == 5
    assert row_after_second["alpha"] == 1.2


# ===========================================================================
# AC7b (K7): neutral_site comes from the stored row, never the override
# entry -- a hardcoded False would give the wrong alpha for this fixture.
# ===========================================================================
def test_neutral_site_is_read_from_the_stored_row_not_hardcoded():
    entry = {
        "game_id": 999001, "season": 2026, "week": 3, "season_type": "regular",
        "home_team": "Florida State", "away_team": "Alabama",
        "home_score": 24, "away_score": 21,
        "reason": "test fixture -- neutral-site correction", "declared_on": "2026-09-09",
    }
    # Constructed fixture (not a real production row): a neutral-site game, so a
    # hardcoded False in the applier would produce alpha=0.8 instead of the correct 1.0.
    stored = {
        "id": 999001, "season": 2026, "week": 3, "season_type": "regular",
        "home_team": "Florida State", "away_team": "Alabama",
        "home_score": 20, "away_score": 24, "neutral_site": 1,
        "winner": "Alabama", "margin": 4, "alpha": 1.2,
    }
    engine = _make_engine([stored])

    with _with_overrides([entry]):
        result = apply_game_result_overrides(2026, engine=engine)

    assert result["applied"] == [999001]
    row = _fetch(engine, 999001)
    assert row["winner"] == "Florida State"
    assert row["alpha"] == 1.0, "neutral_site=True (from the stored row) must give alpha=1.0"


# ===========================================================================
# The shipped override file itself: parses, and every entry validates.
# ===========================================================================
def test_shipped_override_file_parses_and_every_entry_validates():
    overrides = game_overrides.load_overrides()
    assert isinstance(overrides, list)
    assert len(overrides) >= 1

    for entry in overrides:
        missing = REQUIRED_ENTRY_FIELDS - set(entry)
        assert not missing, f"override entry missing field(s): {missing}"
        assert isinstance(entry["game_id"], int)
        assert isinstance(entry["season"], int)
        assert isinstance(entry["week"], int)
        assert entry["season_type"] in ("regular", "postseason")
        assert isinstance(entry["home_team"], str) and entry["home_team"]
        assert isinstance(entry["away_team"], str) and entry["away_team"]
        assert isinstance(entry["home_score"], int) and entry["home_score"] >= 0
        assert isinstance(entry["away_score"], int) and entry["away_score"] >= 0
        assert isinstance(entry["reason"], str) and entry["reason"].strip()
        assert isinstance(entry["declared_on"], str)

    seeded = next((e for e in overrides if e["game_id"] == 401858428), None)
    assert seeded is not None, "the seeded Michigan / Western Michigan entry must be present"
    assert seeded["home_team"] == "Michigan"
    assert seeded["away_team"] == "Western Michigan"
    assert seeded["home_score"] == 7
    assert seeded["away_score"] == 12


def test_shipped_override_file_is_raw_valid_json():
    path = game_overrides.DEFAULT_OVERRIDES_PATH
    with open(path) as f:
        data = json.load(f)
    assert data["version"] == 1
    assert isinstance(data["overrides"], list)


# ===========================================================================
# Fix cycle 1, item 1: load_overrides validates field presence but must also
# validate type. A float score would apply and Postgres would silently ROUND
# it on the implicit cast to integer -- exactly the failure mode this feature
# exists to prevent. isinstance(True, int) is True in Python, so a JSON `true`
# must be rejected explicitly rather than passing an int check by accident.
# ===========================================================================
def test_load_overrides_rejects_a_float_score():
    entry = _override_entry()
    entry["home_score"] = 7.4
    entry["away_score"] = 12.6
    bad_path = _write_temp_overrides_file([entry])
    try:
        raised = False
        try:
            game_overrides.load_overrides(path=bad_path)
        except ValueError:
            raised = True
        assert raised, "a float score must be rejected, not silently applied and rounded by Postgres"
    finally:
        Path(bad_path).unlink(missing_ok=True)


def test_load_overrides_rejects_a_bool_for_an_int_field():
    entry = _override_entry()
    entry["week"] = True  # isinstance(True, int) is True -- must be rejected explicitly
    bad_path = _write_temp_overrides_file([entry])
    try:
        raised = False
        try:
            game_overrides.load_overrides(path=bad_path)
        except ValueError:
            raised = True
        assert raised, "a JSON boolean must be rejected for an int-typed field"
    finally:
        Path(bad_path).unlink(missing_ok=True)


# ===========================================================================
# Fix cycle 1, item 2: two entries for the same game_id must not silently
# last-write-win -- every other identity problem in this design refuses
# loudly, and a duplicate should too.
# ===========================================================================
def test_load_overrides_rejects_a_duplicate_game_id():
    entry = _override_entry()
    duplicate = dict(entry)
    duplicate["week"] = 2  # different week, same game_id -- still a duplicate identity
    bad_path = _write_temp_overrides_file([entry, duplicate])
    try:
        raised = False
        try:
            game_overrides.load_overrides(path=bad_path)
        except ValueError:
            raised = True
        assert raised, "two entries sharing a game_id must be rejected, not last-write-wins"
    finally:
        Path(bad_path).unlink(missing_ok=True)


# ===========================================================================
# Fix cycle 1, item 5: ordering IS the mechanism. All 16 tests above call
# apply_game_result_overrides directly, so none of them would notice a
# refactor that hoists the call above the postseason ingest (or below
# get_ratings) in main.py -- this test reads main.py itself via ast.
# ===========================================================================
TARGET_CALL_NAMES = ("load_games_to_db", "apply_game_result_overrides", "get_ratings")


def _main_call_order():
    main_py = Path(__file__).resolve().parent.parent / "main.py"
    tree = ast.parse(main_py.read_text())
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    order = []
    for node in ast.walk(main_fn):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name in TARGET_CALL_NAMES:
                order.append((node.lineno, name))
    order.sort()
    return [name for _, name in order]


def test_override_step_is_positioned_between_ingest_and_model():
    call_order = _main_call_order()
    load_games_positions = [i for i, name in enumerate(call_order) if name == "load_games_to_db"]
    override_positions = [i for i, name in enumerate(call_order) if name == "apply_game_result_overrides"]
    ratings_positions = [i for i, name in enumerate(call_order) if name == "get_ratings"]

    assert len(load_games_positions) == 2, f"expected 2 load_games_to_db calls in main(), found order={call_order}"
    assert len(override_positions) == 1, f"expected exactly 1 apply_game_result_overrides call, found order={call_order}"
    assert len(ratings_positions) == 1, f"expected exactly 1 get_ratings call, found order={call_order}"

    assert max(load_games_positions) < override_positions[0], (
        "apply_game_result_overrides must come after BOTH load_games_to_db calls (K3) -- "
        f"call order was {call_order}"
    )
    assert override_positions[0] < ratings_positions[0], (
        f"apply_game_result_overrides must come before get_ratings -- call order was {call_order}"
    )


# ===========================================================================
# Fix cycle 1, item 6: the OVERRIDE_FAILURE token main.py emits and the token
# the workflows grep for must be the literal same string. Drift here
# reproduces the exact silent-green publish test_entry_point_resolves.py
# exists to prevent, just one file over.
# ===========================================================================
OVERRIDE_FAILURE_TOKEN = "OVERRIDE_FAILURE"


def test_override_failure_token_matches_between_main_and_workflows():
    repo_root = Path(__file__).resolve().parent.parent
    main_src = (repo_root / "main.py").read_text()

    assert main_src.count(OVERRIDE_FAILURE_TOKEN) >= 2, (
        f"expected the literal {OVERRIDE_FAILURE_TOKEN!r} token in both main.py emitters "
        "(the per-failure log line, and the outer except's log line)"
    )

    for workflow_name in ("weekly-update.yml", "test-week-run.yml"):
        workflow_src = (repo_root / ".github" / "workflows" / workflow_name).read_text()
        match = re.search(r'grep -q "([^"]+)" output\.log', workflow_src)
        assert match is not None, f"{workflow_name} must grep output.log for a quoted token"
        assert match.group(1) == OVERRIDE_FAILURE_TOKEN, (
            f"{workflow_name} greps for {match.group(1)!r}, which does not match the literal "
            f"{OVERRIDE_FAILURE_TOKEN!r} token main.py actually emits"
        )


# ===========================================================================
# Fix cycle 1, item 7: an override naming a game_id absent from `games` (the
# R3 scenario -- CFBD retires/reassigns the id) must fail intelligibly, not
# with a raw TypeError from indexing a None row. Deleting the LookupError
# guard in game_overrides.py currently leaves every other test green.
# ===========================================================================
def test_absent_game_id_reports_an_intelligible_failure():
    entry = _override_entry()
    engine = _make_engine([])  # no games rows at all

    with _with_overrides([entry]):
        result = apply_game_result_overrides(2026, engine=engine)

    assert result["applied"] == []
    assert len(result["failures"]) == 1
    reason = result["failures"][0]["reason"]
    assert reason == f"no games row found for id={entry['game_id']}", reason


# ===========================================================================
# Fix cycle 1, item 8: a bad entry must not block a good one -- pins the
# per-entry-transaction design (one engine.begin() per override, not one for
# the whole season, which would roll back good overrides alongside bad ones).
# ===========================================================================
def test_multi_entry_partial_failure_does_not_block_a_good_entry():
    bad_entry = _override_entry()  # game_id 401858428
    bad_stored = _stored_row_matching(bad_entry)
    bad_stored["home_team"] = "Wrong Team"  # forces an identity-mismatch failure

    good_entry = {
        "game_id": 999002, "season": 2026, "week": 2, "season_type": "regular",
        "home_team": "Iowa", "away_team": "Iowa State",
        "home_score": 24, "away_score": 21,
        "reason": "test fixture", "declared_on": "2026-09-09",
    }
    good_stored = {
        "id": 999002, "season": 2026, "week": 2, "season_type": "regular",
        "home_team": "Iowa", "away_team": "Iowa State",
        "home_score": 10, "away_score": 21, "neutral_site": 0,
        "winner": "Iowa State", "margin": 11, "alpha": 1.2,
    }

    engine = _make_engine([bad_stored, good_stored])
    with _with_overrides([bad_entry, good_entry]):
        result = apply_game_result_overrides(2026, engine=engine)

    assert result["applied"] == [999002], "the good entry must still apply despite the bad one"
    assert len(result["failures"]) == 1
    assert result["failures"][0]["game_id"] == bad_entry["game_id"]

    good_row = _fetch(engine, 999002)
    assert good_row["home_score"] == 24
    assert good_row["winner"] == "Iowa"


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
