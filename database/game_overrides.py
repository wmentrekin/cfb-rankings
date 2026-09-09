"""Apply checked-in, operator-declared game-result overrides.

Overrides live in database/game_result_overrides.json (K1 in
docs/manual-result-overrides/plan.yaml) rather than in the database, so they
are reviewable in the PR and re-assert themselves on every pipeline run --
a re-ingest of the CFBD result can never win.

This module never raises for a per-entry problem, nor for a whole-file
problem (a missing or malformed override file). Every failure mode is
reported back in the returned `failures` list instead, so a bad override
entry -- or a bad override file -- cannot take down the pipeline run that
calls it. (main.py wraps the call in its own try/except as a second line of
defense per K5, but this module's own contract is: never raise.)

All database access goes through parameterized sqlalchemy text() executed via
engine.begin(), following scripts/repair_unplayed_game_winners.py:110 --
deliberately NOT get_games.py's Table(autoload_with=engine) reflection, which
needs a live connection to reflect and so cannot be exercised in CI.
"""
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv  # type: ignore
from sqlalchemy import create_engine, text  # type: ignore

from database.game_derivations import compute_alpha, compute_margin, compute_winner

DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent / "game_result_overrides.json"

# The five identity fields verified against the stored row before any write
# (K2/K7 -- season_type is included because a regular/postseason collision on
# one numeric CFBD id is exactly the case this check exists to prevent).
IDENTITY_FIELDS = ("season", "week", "season_type", "home_team", "away_team")

_REQUIRED_ENTRY_FIELDS = {
    "game_id", "season", "week", "season_type", "home_team", "away_team",
    "home_score", "away_score", "reason", "declared_on",
}

# Fields required to be a genuine int (not a float, and not a bool -- `isinstance(True, int)`
# is True in Python, so a JSON `true`/`false` would otherwise silently pass an int check).
_INT_FIELDS = ("game_id", "season", "week", "home_score", "away_score")
_STR_FIELDS = ("season_type", "home_team", "away_team", "reason", "declared_on")


def _is_strict_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)

SELECT_GAME = text(
    "SELECT season, week, season_type, home_team, away_team, neutral_site "
    "FROM games WHERE id = :game_id"
)

UPDATE_GAME = text(
    "UPDATE games "
    "SET home_score = :home_score, away_score = :away_score, "
    "winner = :winner, margin = :margin, alpha = :alpha "
    "WHERE id = :game_id"
)


def _build_engine():
    """Build an engine from the same DB_* env vars the rest of the repo uses."""
    load_dotenv()
    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )
    return create_engine(db_url)


def load_overrides(path: Optional[str] = None) -> list:
    """Load and validate database/game_result_overrides.json (or `path`).

    Raises (FileNotFoundError, json.JSONDecodeError, ValueError) on a missing,
    unreadable, or malformed file/schema -- including a wrong field type (a
    float or bool where an int is declared) and a duplicate game_id, both of
    which would otherwise apply silently with a last-write-wins result.
    apply_game_result_overrides is responsible for turning any of this into a
    reported failure rather than letting it crash the caller.
    """
    target = Path(path) if path is not None else DEFAULT_OVERRIDES_PATH
    with open(target, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "overrides" not in data:
        raise ValueError(f"{target}: missing top-level 'overrides' key")

    overrides = data["overrides"]
    if not isinstance(overrides, list):
        raise ValueError(f"{target}: 'overrides' must be a list")

    for i, entry in enumerate(overrides):
        if not isinstance(entry, dict):
            raise ValueError(f"{target}: overrides[{i}] is not an object")
        missing = _REQUIRED_ENTRY_FIELDS - set(entry)
        if missing:
            raise ValueError(f"{target}: overrides[{i}] missing field(s): {sorted(missing)}")

        bad_int_fields = [f for f in _INT_FIELDS if not _is_strict_int(entry[f])]
        if bad_int_fields:
            raise ValueError(
                f"{target}: overrides[{i}] field(s) must be int, not "
                f"{[type(entry[f]).__name__ for f in bad_int_fields]}: {bad_int_fields}"
            )

        bad_str_fields = [f for f in _STR_FIELDS if not isinstance(entry[f], str)]
        if bad_str_fields:
            raise ValueError(
                f"{target}: overrides[{i}] field(s) must be str, not "
                f"{[type(entry[f]).__name__ for f in bad_str_fields]}: {bad_str_fields}"
            )

    game_ids = [entry["game_id"] for entry in overrides]
    duplicates = sorted({g for g in game_ids if game_ids.count(g) > 1})
    if duplicates:
        raise ValueError(f"{target}: duplicate game_id(s) declared more than once: {duplicates}")

    return overrides


def overrides_for_season(overrides: list, season: int) -> list:
    """Overrides declared for exactly this season.

    A non-matching season is a silent no-op (AC6), not a failure -- callers
    filter with this *before* the identity check, so it never reaches
    `failures`.
    """
    return [entry for entry in overrides if entry["season"] == season]


def apply_game_result_overrides(year: int, engine=None) -> dict:
    """Apply every override declared for `year` against the stored `games` rows.

    Re-derives winner/margin/alpha from the override's declared scores (not
    from whatever the row currently holds) via database/game_derivations.py,
    so a second run is idempotent. `neutral_site` is read from the stored row,
    never from the override entry (K7).

    Never raises. Returns:
        {"applied": [game_id, ...], "skipped_other_season": n,
         "skipped_other_season_entries": [{"game_id": ..., "season": ...}, ...],
         "failures": [{"game_id": ..., "reason": ...}, ...]}

    `skipped_other_season_entries` carries the identity of each skipped entry
    (not just the count) -- a wrong-season entry can never turn the CI job red
    (AC6: it is filtered out before the identity check, by design), so this is
    the only signal that distinguishes a fat-fingered season from the correct
    steady state once the year rolls over. Additive alongside
    `skipped_other_season`, which existing callers keep reading unchanged.
    """
    try:
        all_overrides = load_overrides()
    except Exception as exc:
        return {
            "applied": [],
            "skipped_other_season": 0,
            "skipped_other_season_entries": [],
            "failures": [{"game_id": None, "reason": f"could not load override file: {exc}"}],
        }

    season_overrides = overrides_for_season(all_overrides, year)
    skipped_entries = [entry for entry in all_overrides if entry["season"] != year]
    skipped_other_season_entries = [
        {"game_id": entry.get("game_id"), "season": entry.get("season")} for entry in skipped_entries
    ]

    applied: list = []
    failures: list = []
    owns_engine = engine is None
    engine_built = False

    try:
        if engine is None:
            engine = _build_engine()
            engine_built = True

        for entry in season_overrides:
            game_id = entry.get("game_id")
            try:
                with engine.begin() as conn:
                    row = conn.execute(SELECT_GAME, {"game_id": game_id}).mappings().first()
                    if row is None:
                        raise LookupError(f"no games row found for id={game_id}")

                    mismatches = [f for f in IDENTITY_FIELDS if row[f] != entry[f]]
                    if mismatches:
                        detail = "; ".join(
                            f"{f}: stored={row[f]!r} override={entry[f]!r}" for f in mismatches
                        )
                        raise ValueError(f"identity mismatch on {mismatches}: {detail}")

                    home_score = entry["home_score"]
                    away_score = entry["away_score"]
                    neutral_site = row["neutral_site"]  # K7: from the stored row, never the entry
                    winner = compute_winner(entry["home_team"], entry["away_team"], home_score, away_score)
                    margin = compute_margin(home_score, away_score)
                    alpha = float(compute_alpha(entry["home_team"], winner, neutral_site))

                    conn.execute(UPDATE_GAME, {
                        "game_id": game_id,
                        "home_score": home_score,
                        "away_score": away_score,
                        "winner": winner,
                        "margin": margin,
                        "alpha": alpha,
                    })
                applied.append(game_id)
            except Exception as exc:
                failures.append({"game_id": game_id, "reason": str(exc)})
    except Exception as exc:
        # Anything that escapes the per-entry try above is a whole-run problem -- most
        # realistically _build_engine() itself failing (e.g. a DB_* env var missing or the
        # database unreachable). BEHAVIOR CHANGE (cycle-1 review item 3): this used to sit
        # outside the try entirely, so it raised straight out of this function -- violating
        # the "never raises" contract documented above, even though main.py's own try/except
        # happened to catch it anyway. Folded into `failures` here instead, so a transient
        # Postgres outage now reds the weekly job via OVERRIDE_FAILURE, where it previously
        # logged main.py's generic warning and returned green. That is a deliberate
        # improvement, not a side effect: an unreachable database silently leaving every
        # override unapplied is exactly the kind of failure this feature exists to surface.
        failures.append({"game_id": None, "reason": f"could not apply overrides: {exc}"})
    finally:
        if owns_engine and engine_built and engine is not None:
            engine.dispose()

    return {
        "applied": applied,
        "skipped_other_season": len(skipped_entries),
        "skipped_other_season_entries": skipped_other_season_entries,
        "failures": failures,
    }
