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
Default (no --apply): solves every published (season, week), prints a summary and how the
solved teams and records differ from the stored rows, and writes NOTHING.

--apply, in this order, stopping at the first failure:
  1. Solves every week, in memory. Nothing is written until all of them have solved.
  2. Copies the current `ratings` table to `ratings_pre_2026_retune`, once. If that table
     already exists it is left alone, so a second run cannot overwrite the original numbers
     with already-rewritten ones.
  3. In ONE transaction (all weeks or none): upserts every week's ratings into `ratings`, and
     deletes any stored row for a re-solved week whose team the model no longer rates, so no
     old-model row is left ranked beside new ones.
  4. Republishes the rankings file for every re-solved week. Each is read back from R2 and
     every team and rating is checked against what was solved. (Week-over-week movement is
     read from the database, which is fully rewritten by then.)
  5. Republishes the Season Grid and Playoffs files for each season (both embed a
     rating-derived rank), and checks each was rewritten during this run.
  6. ALWAYS, even if step 4 or 5 failed part-way: republishes the newest week in the
     database last and checks the "latest" keys point at it. Every rankings publish rewrites
     those keys, so without this a --season run of an old season, a failure part-way, or a
     weekly run that landed in the middle would leave the site on the wrong week.

Do not start it while a weekly run is in progress, and run it from `main`: a rerun from an
unmerged branch would be overwritten in style by the next weekly run from `main`.

It does not ask the site to rebuild; the workflow that runs it does.

UNDO
----
    BEGIN;
    DELETE FROM ratings r USING ratings_pre_2026_retune b
     WHERE r.season = b.season AND r.week = b.week;
    INSERT INTO ratings SELECT * FROM ratings_pre_2026_retune;
    COMMIT;
That restores only the weeks the backup holds and leaves any week published since alone. The
R2 files then need republishing from a checkout of the old model.

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
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # type: ignore  # noqa: E402
from sqlalchemy import MetaData, Table, create_engine, inspect, text  # type: ignore  # noqa: E402
from sqlalchemy.dialects.postgresql import insert  # type: ignore  # noqa: E402

import model.model as weekly_model  # noqa: E402
from artifacts.playoffs import SEASON_RULES, publish_playoffs_artifact  # noqa: E402
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
    # Week 0 is never published; get_ratings would read it as "the whole season".
    return [sw for sw in weeks if sw[1] >= 1 and (season is None or sw[0] == season)]


def newest_week(engine):
    """The newest (season, week) in `ratings` right now -- what the "latest" keys must show."""
    with engine.connect() as conn:
        row = conn.execute(text("SELECT season, week FROM ratings ORDER BY season DESC, week DESC LIMIT 1")).fetchone()
    return (int(row[0]), int(row[1])) if row else None


def stored_rows(engine) -> dict:
    """{(season, week): {team: (wins, losses)}} as stored today."""
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT season, week, team, wins, losses FROM ratings")).fetchall()
    stored = {}
    for season, week, team, wins, losses in rows:
        stored.setdefault((int(season), int(week)), {})[team] = (int(wins), int(losses))
    return stored


def print_drift(solved: dict, stored: dict) -> None:
    """How the re-solve differs from the stored rows for a reason OTHER than the model: teams
    added or dropped, or records changed by games corrected since the week was published."""
    print("\ndifferences from the stored rows that are not the model change:")
    clean = True
    for key in sorted(solved):
        df = solved[key]
        now = {r.team: (int(r.wins), int(r.losses)) for r in df.itertuples()}
        before = stored.get(key, {})
        added, dropped = sorted(set(now) - set(before)), sorted(set(before) - set(now))
        changed = sorted(t for t in set(now) & set(before) if now[t] != before[t])
        if added or dropped or changed:
            clean = False
            print(f"  season {key[0]} week {key[1]}: teams added {added or 'none'}; stored rows to delete "
                  f"{dropped or 'none'}; records changed {changed or 'none'}")
    if clean:
        print("  none: every week has the same teams and records as stored.")


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
    total = deleted = 0
    with engine.begin() as conn:
        for key in sorted(solved):
            df = solved[key]
            # A stored team this solve no longer rates would stay on the old model and be
            # ranked beside the new rows. Remove it in the same transaction.
            result = conn.execute(
                table.delete().where(table.c.season == key[0], table.c.week == key[1],
                                     table.c.team.notin_(list(df["team"])))
            )
            deleted += result.rowcount or 0
            for _, row in df.iterrows():
                stmt = insert(table).values(**row.to_dict())
                stmt = stmt.on_conflict_do_update(
                    index_elements=KEY_COLUMNS,
                    set_={c: getattr(stmt.excluded, c) for c in df.columns if c not in KEY_COLUMNS},
                )
                conn.execute(stmt)
                total += 1
    print(f"Upserted {total} rating rows across {len(solved)} weeks; deleted {deleted} stale rows.")


def _read_json(client, bucket, key) -> dict:
    try:
        return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception as ex:
        _fail(f"could not read {key} back from R2 after publishing: {ex}")


def verify_rankings(client, bucket, season, week, df) -> None:
    """publish_rankings_artifact never raises, so read the file back and check every team."""
    key = f"rankings/{season}/week-{week:02d}.json"
    rows = _read_json(client, bucket, key).get("rankings", [])
    published = {r["team"]: r["rating"] for r in rows}
    expected = {r.team: round(float(r.rating), 2) for r in df.itertuples()}
    if set(published) != set(expected):
        _fail(f"{key} has {len(published)} teams; this run solved {len(expected)} "
              f"(differing: {sorted(set(published) ^ set(expected))[:10]}).")
    wrong = [t for t in expected if abs(published[t] - expected[t]) > 0.011]
    if wrong:
        _fail(f"{key} does not carry this run's ratings for {len(wrong)} team(s), e.g. {wrong[0]}: "
              f"published {published[wrong[0]]}, solved {expected[wrong[0]]}.")
    print(f"VERIFIED: {key} matches this run ({len(rows)} teams, No. 1 {rows[0]['team']}).")


def verify_fresh(client, bucket, key, started: datetime) -> None:
    """The grid and playoffs publishers never raise either. Their payloads cannot be rebuilt
    here cheaply, so check the file was written during THIS run."""
    stamp = _read_json(client, bucket, key).get("generated_at_utc")
    try:
        generated = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
    except ValueError:
        _fail(f"{key} has no readable generated_at_utc ({stamp!r}); cannot tell whether it was republished.")
    if generated < started - timedelta(minutes=1):
        _fail(f"{key} was generated at {stamp}, before this run started: it was not republished.")
    print(f"VERIFIED: {key} was rewritten during this run.")


def verify_latest(client, bucket, season, week) -> None:
    for key in (f"rankings/{season}/latest.json", "rankings/latest.json"):
        payload = _read_json(client, bucket, key)
        if (payload.get("season"), payload.get("week")) != (season, week):
            _fail(f"{key} shows season {payload.get('season')} week {payload.get('week')}; the newest week "
                  f"in the database is season {season} week {week}. The site would show the wrong week.")
        print(f"VERIFIED: {key} is season {season} week {week}.")


def publish_all(engine, solved: dict, started: datetime) -> None:
    client = get_r2_client()
    bucket = os.getenv("R2_BUCKET_NAME")
    if client is None or not bucket:
        _fail("R2 is not configured, so nothing was republished. The ratings table IS already rewritten; "
              "fix the R2 settings and run again with --apply (the rerun is idempotent).")
    try:
        for (season, week) in sorted(solved):
            publish_rankings_artifact(season, week)
            verify_rankings(client, bucket, season, week, solved[(season, week)])
        for season in sorted({s for s, _w in solved}):
            publish_schedule_artifact(season)
            verify_fresh(client, bucket, f"schedule/{season}/latest.json", started)
            if season in SEASON_RULES:
                publish_playoffs_artifact(season)
                verify_fresh(client, bucket, f"playoffs/{season}/latest.json", started)
            print(f"Republished the Season Grid and Playoffs files for season={season}.")
    finally:
        # Runs on success, on a failed check above (sys.exit raises), and on any other error.
        newest = newest_week(engine)
        if newest is not None:
            publish_rankings_artifact(*newest)
            verify_latest(client, bucket, *newest)


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

    started = datetime.now(timezone.utc)
    engine = _engine()
    try:
        weeks = published_weeks(engine, args.season)
        if not weeks:
            _fail("no published weeks found in ratings.")
        solved = solve_all(weeks)
        print_summary(solved)
        print_drift(solved, stored_rows(engine))
        if not args.apply:
            print("\nDRY RUN -- every week solved; nothing was written to the database or to R2. "
                  "Pass --apply to rewrite and republish.")
            return
        backup_ratings(engine)
        write_all(engine, solved)
        publish_all(engine, solved, started)
    finally:
        engine.dispose()
    print(f"\nDONE: {len(solved)} weeks rewritten and republished.")


if __name__ == "__main__":
    main()
