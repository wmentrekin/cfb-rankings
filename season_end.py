"""When the weekly run should stop publishing rankings, and when it should stop altogether.

THE PROBLEM THIS REPLACES: the rankings end with conference championship week, but the weekly
job did not know that. Left alone after the title games it rated the Army-Navy game as a week of
its own and then republished that week every Sunday, so the schedule had to be switched off by
hand each December and back on each August.

THE RULE, read from the stored schedule so it needs no dates edited each year:

  final rankings week = the last regular-season week that has any game other than Army-Navy.

For 2025 that is week 15 and for 2026 week 14: championship week, with Army-Navy alone in the
week after. The weekly run then has three states, chosen fresh on every run:

  "rankings"   -- the season is in progress, or the final week has not been rated yet. The run
                  does everything it always did. A run that lands in the Army-Navy week before
                  the final week was rated is pulled back to the final week.
  "postseason" -- the final week is rated. No model run, no ratings write, no rankings file.
                  The run still ingests bowl and playoff results (and Army-Navy) and republishes
                  the Season Grid and the Playoffs bracket, which keep changing into January.
  finished     -- "postseason", and no game of the season has been played in the last
                  QUIET_DAYS days or is still to come. The run does nothing at all.

So the schedule can stay on all year. Nothing here knows a date except the December guard in
choose_mode, which exists so a gap in the stored schedule mid-season can never be mistaken for
the end of it.

A week passed explicitly (--week, the workflow's `week` input) always means "rankings" for that
week: this module only steers runs that choose their own week.
"""

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Set, Tuple

import pandas as pd  # type: ignore
from dotenv import load_dotenv  # type: ignore
from sqlalchemy import create_engine  # type: ignore

MODE_RANKINGS = "rankings"
MODE_POSTSEASON = "postseason"

ARMY_NAVY = frozenset({"Army", "Navy"})

# A season starts in late August and its last game is in late January. A run before July
# belongs to the season that started the previous calendar year.
SEASON_ROLLOVER_MONTH = 7

# Conference championship weekend is the first Saturday of December, so the Sunday after it is
# never earlier than December 2. Before December no run may enter postseason mode.
POSTSEASON_EARLIEST = (12, 1)

# How long after a season's last stored game the run keeps checking for results. Covers a
# weekly schedule with room to spare, and the gap between selection day and the first games.
QUIET_DAYS = 8


def default_season(today: date) -> int:
    """The season a run on `today` belongs to when none is given."""
    return today.year if today.month >= SEASON_ROLLOVER_MONTH else today.year - 1


def final_rankings_week(regular_games: Iterable[Tuple[int, str, str]]) -> Optional[int]:
    """The last regular-season week with a game other than Army-Navy, or None with no games.

    `regular_games` is (week, home_team, away_team) for every stored regular-season game.
    """
    weeks = [
        int(week) for week, home, away in regular_games
        if week is not None and {home, away} != ARMY_NAVY
    ]
    return max(weeks) if weeks else None


@dataclass(frozen=True)
class Decision:
    mode: str
    week: int
    reason: str


def choose_mode(
    today: date,
    season: int,
    resolved_week: int,
    final_week: Optional[int],
    final_week_rated: bool,
    explicit_week: bool,
) -> Decision:
    """Which state this run is in, and the week it should publish in "rankings" mode. Pure."""
    if explicit_week:
        return Decision(MODE_RANKINGS, resolved_week, "the week was given explicitly")
    if final_week is None:
        return Decision(MODE_RANKINGS, resolved_week, "no regular-season games are stored yet")
    if resolved_week < final_week:
        return Decision(MODE_RANKINGS, resolved_week, f"week {final_week} is the final rankings week and has not been reached")
    if today < date(season, *POSTSEASON_EARLIEST):
        return Decision(MODE_RANKINGS, resolved_week,
                        "it is before December, so the stored schedule is not trusted to say the season is over")
    if not final_week_rated:
        if resolved_week > final_week:
            return Decision(MODE_RANKINGS, final_week,
                            f"week {final_week} is the final rankings week and has not been rated yet; "
                            f"week {resolved_week} is not a rankings week")
        return Decision(MODE_RANKINGS, final_week, f"week {final_week} is the final rankings week")
    return Decision(MODE_POSTSEASON, final_week,
                    f"the final rankings week ({final_week}) is already rated; the rankings are finished for {season}")


def season_is_finished(now: datetime, start_dates: Iterable[datetime]) -> bool:
    """True when no stored game of the season is still to come or was played in the last
    QUIET_DAYS days. `now` and the start dates are naive UTC, as in games.start_date.

    False with no stored games at all: that is a season that has not loaded, not one that ended.
    """
    starts = [s for s in start_dates if s is not None and not pd.isnull(s)]
    if not starts:
        return False
    return max(starts) < now - timedelta(days=QUIET_DAYS)


@dataclass(frozen=True)
class SeasonState:
    final_week: Optional[int]
    rated_weeks: Set[int]
    start_dates: Tuple[datetime, ...]


def read_season_state(year: int) -> SeasonState:
    """What the database knows about `year`. SELECTs only. Raises on any database problem: the
    caller decides what a failed check means, because the answer differs in and out of season."""
    load_dotenv()
    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )
    try:
        games = pd.read_sql_query(
            f"SELECT week, home_team, away_team, season_type, start_date FROM games WHERE season = {int(year)};",
            engine,
        )
        rated = pd.read_sql_query(f"SELECT DISTINCT week FROM ratings WHERE season = {int(year)};", engine)
    finally:
        engine.dispose()
    regular = games[games["season_type"] == "regular"]
    return SeasonState(
        final_week=final_rankings_week(zip(regular["week"], regular["home_team"], regular["away_team"])),
        rated_weeks={int(w) for w in rated["week"]},
        start_dates=tuple(pd.to_datetime(games["start_date"]).dt.to_pydatetime()) if len(games) else (),
    )
