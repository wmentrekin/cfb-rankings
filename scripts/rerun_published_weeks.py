"""
Re-solve every published week with the model on this checkout, and republish.

BACKGROUND
----------
The 2026 retune changed what the model produces for the SAME games: the margin scale is taken
from the margins (it was taken from the home/away multiplier by mistake), and the 0.01 rating
floor is gone, with published ratings shifted so the lowest team reads 0.01 (model/params.py).
Every snapshot already in the `ratings` table was computed the old way, and the R2 files the
site reads are built from that table. Left alone, the next weekly run would publish a new-model
week on top of old-model history, and its week-over-week movement would compare two different
models.

There is no narrow way to redo history with main.py: it resolves the week from today's date,
re-ingests from CFBD, and handles one week per run. This script is that narrow way. It never
calls CFBD and never touches `games` or `teams`: it re-solves from the games already stored.

WHAT IT DOES
------------
Default (no --apply): solves every published (season, week), prints a summary, writes NOTHING.

--apply, in this order, stopping at the first failure:
  1. Solves every week, in memory. Nothing is written until all of them have solved.
  2. Copies the current `ratings` table to `ratings_pre_2026_retune`, once. If that table
     already exists it is left alone, so a second run cannot overwrite the original numbers
     with already-rewritten ones.
  3. Upserts every week's ratings into `ratings`, in ONE transaction: all weeks or none.
  4. Republishes the rankings file for every week, oldest first, so each week's movement is
     computed against the rewritten week before it and the "latest" keys end on the newest
     week. Each is read back from R2 and checked against what was solved.
  5. Republishes the Season Grid and Playoffs files for each season (both embed a
     rating-derived rank).

It does not ask the site to rebuild; the workflow that runs it does.

UNDO
----
    BEGIN;
    DELETE FROM ratings;
    INSERT INTO ratings SELECT * FROM ratings_pre_2026_retune;
    COMMIT;
then run this script's publish steps from a checkout of the old model, or let the next weekly
run republish the current week.

CREDENTIALS REQUIRED
--------------------
  default:   DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME (SELECTs only).
  --apply:   the same, plus R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY /
             R2_BUCKET_NAME.

Usage:
    python scripts/rerun_published_weeks.py
    python scripts/rerun_published_weeks.py --season 2026
    python scripts/rerun_published_weeks.py --apply
"""

import argparse
import contextlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # type: ignore  # noqa: E402
from sqlalchemy import MetaData, Table, create_engine, inspect, text  # type: ignore  # noqa: E402
from sqlalchemy.dialects.postgresql import insert  # type: ignore  # noqa: E402

import model.model as weekly_model  # noqa: E402
from artifacts.playoffs import publish_playoffs_artifact  # noqa: E402
from artifacts.r2 import get_r2_client, publish_rankings_artifact  # noqa: E402
from artifacts.schedule import publish_schedule_artifact  # noqa: E402
from database.model_to_db import ratings_to_df  # noqa: E402

BACKUP_TABLE = "ratings_pre_2026_retune"
KEY_COLUMNS = ["team", "season", "week"]


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def _engine():
    load_dotenv()
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )


@contextlib.contextmanager
def _quiet():
    """Silence the model's prints and the solver's own C-level output."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def published_weeks(engine, season=None) -> list:
    """Every (season, week) that has rows in `ratings`, oldest first."""
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT season, week FROM ratings ORDER BY season, week")).fetchall()
    weeks = [(int(s), int(w)) for s, w in rows]
    return [sw for sw in weeks if season is None or sw[0] == season]


def solve_all(weeks: list) -> dict:
    """{(season, week): ratings DataFrame}. The prior season is solved once per season, not once
    per week: it does not depend on the week."""
    prior_cache = {}
    real_prior = weekly_model.get_prior_ratings  # whatever the weekly model calls today

    def cached_prior(prior_year, *args, **kwargs):
        if prior_year not in prior_cache:
            prior_cache[prior_year] = real_prior(prior_year, *args, **kwargs)
        return dict(prior_cache[prior_year])  # get_ratings adds new teams to the dict it is given

    weekly_model.get_prior_ratings = cached_prior
    solved = {}
    try:
        for (season, week) in weeks:
            print(f"solving season={season} week={week} ...", flush=True)
            with _quiet():
                result = weekly_model.get_ratings(season, week)
            if not result:
                _fail(f"the model returned no ratings for season={season} week={week}; nothing was written.")
            ratings, records = result
            df = ratings_to_df(ratings, records, season, week)
            if df.empty or df["rating"].isnull().any():
                _fail(f"season={season} week={week} solved to an empty or incomplete table; nothing was written.")
            solved[(season, week)] = df
    finally:
        weekly_model.get_prior_ratings = real_prior
    return solved


def backup_ratings(engine) -> None:
    if inspect(engine).has_table(BACKUP_TABLE):
        print(f"Backup table {BACKUP_TABLE} already exists; leaving it as it is.")
        return
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE {BACKUP_TABLE} AS TABLE ratings"))
        count = conn.execute(text(f"SELECT count(*) FROM {BACKUP_TABLE}")).scalar()
    print(f"Copied {count} rows from ratings to {BACKUP_TABLE}.")


def write_all(engine, solved: dict) -> None:
    """One transaction for every week: either all of history is on the new model or none is."""
    table = Table("ratings", MetaData(), autoload_with=engine)
    total = 0
    with engine.begin() as conn:
        for key in sorted(solved):
            df = solved[key]
            for _, row in df.iterrows():
                stmt = insert(table).values(**row.to_dict())
                stmt = stmt.on_conflict_do_update(
                    index_elements=KEY_COLUMNS,
                    set_={c: getattr(stmt.excluded, c) for c in df.columns if c not in KEY_COLUMNS},
                )
                conn.execute(stmt)
                total += 1
    print(f"Upserted {total} rating rows across {len(solved)} weeks.")


def verify_rankings(client, bucket, season, week, df) -> None:
    """publish_rankings_artifact never raises, so read the file back and check it."""
    key = f"rankings/{season}/week-{week:02d}.json"
    try:
        published = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception as ex:
        _fail(f"could not read {key} back from R2 after publishing: {ex}")
    rows = published.get("rankings", [])
    expected = df.sort_values("rating", ascending=False)
    if len(rows) != len(expected):
        _fail(f"{key} has {len(rows)} teams; this run solved {len(expected)}.")
    top = expected.iloc[0]
    if rows[0]["team"] != top["team"] or abs(rows[0]["rating"] - round(float(top["rating"]), 2)) > 0.011:
        _fail(f"{key} leads with {rows[0]['team']} {rows[0]['rating']}; this run solved "
              f"{top['team']} {float(top['rating']):.2f}.")
    print(f"VERIFIED: {key} matches this run ({len(rows)} teams, No. 1 {rows[0]['team']}).")


def publish_all(solved: dict) -> None:
    client = get_r2_client()
    bucket = os.getenv("R2_BUCKET_NAME")
    if client is None or not bucket:
        _fail("R2 is not configured, so nothing was republished. The ratings table IS already rewritten; "
              "fix the R2 settings and run again with --apply (the rerun is idempotent).")
    for (season, week) in sorted(solved):
        publish_rankings_artifact(season, week)
        verify_rankings(client, bucket, season, week, solved[(season, week)])
    for season in sorted({s for s, _w in solved}):
        publish_schedule_artifact(season)
        publish_playoffs_artifact(season)
        print(f"Republished the Season Grid and Playoffs files for season={season}.")


def print_summary(solved: dict) -> None:
    print(f"\nsolved weeks: {len(solved)}")
    for season in sorted({s for s, _w in solved}):
        week = max(w for s, w in solved if s == season)
        df = solved[(season, week)].sort_values("rating", ascending=False)
        top = ", ".join(f"{r.team} {r.rating:.2f}" for r in df.head(5).itertuples())
        print(f"season {season} week {week}: {len(df)} teams, ratings {df['rating'].min():.2f} to "
              f"{df['rating'].max():.2f}. Top 5: {top}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=None, help="Only this season. Default: every published season.")
    parser.add_argument("--apply", action="store_true",
                        help="Rewrite the ratings table and republish. Without it nothing is written.")
    args = parser.parse_args()

    engine = _engine()
    try:
        weeks = published_weeks(engine, args.season)
        if not weeks:
            _fail("no published weeks found in ratings.")
        solved = solve_all(weeks)
        print_summary(solved)
        if not args.apply:
            print("\nDRY RUN -- every week solved; nothing was written to the database or to R2. "
                  "Pass --apply to rewrite and republish.")
            return
        backup_ratings(engine)
        write_all(engine, solved)
    finally:
        engine.dispose()

    publish_all(solved)
    print(f"\nDONE: {len(solved)} weeks rewritten and republished.")


if __name__ == "__main__":
    main()
