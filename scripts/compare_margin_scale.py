"""
Before/after report for the margin-scale fix. READ-ONLY: every database call is a SELECT.

BACKGROUND
----------
Both models used to take the "median margin" from the wrong field of a game tuple (alpha, the
0.8 / 1.0 / 1.2 home-away multiplier, instead of margin -- see model/margin_scale.py). Every
published snapshot in the `ratings` table was computed that way.

For each published (season, week) this script solves the real model twice on today's stored
games -- once with the old scale ("legacy"), once with the corrected one ("fixed") -- and
reports how far the two differ. It also compares the legacy re-solve with what is actually
stored, because a re-solve can differ from the published numbers for a reason that has nothing
to do with the fix: stored games corrected after that week was first published.

It writes nothing to the database or to R2, and never calls main.py.

CREDENTIALS REQUIRED
--------------------
DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME (read-only use).

Usage:
    python scripts/compare_margin_scale.py                      # every published week
    python scripts/compare_margin_scale.py --season 2026        # one season
    python scripts/compare_margin_scale.py --out report.md
"""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # type: ignore  # noqa: E402
import pandas as pd  # type: ignore  # noqa: E402
from dotenv import load_dotenv  # type: ignore  # noqa: E402
from sqlalchemy import create_engine, text  # type: ignore  # noqa: E402

import model.model as weekly_model  # noqa: E402
import model.prior_model as prior_model  # noqa: E402
import model.process_data as process_data  # noqa: E402
from model.margin_scale import (  # noqa: E402
    GAME_ALPHA_INDEX,
    GAME_MARGIN_INDEX,
    compute_k_margin,
    median_margin,
)
from model.params import DEFAULT_PARAMS, ModelParams  # noqa: E402

TARGET_GAP = 7.0  # TARGET_GAP_FOR_MEDIAN in both models
CAP = 20.0        # MAX_RATING_GAP_FROM_MARGIN in both models
FCS = "fcs"
FLOOR = 0.01    # r_min in both models
FLOOR_TOLERANCE = 1e-3  # a solver leaves a clamped rating a hair above the bound


def legacy_k_margin(games, target_gap):
    """The scale as it was before the fix: the median of position 5 (alpha), not margin."""
    values = [abs(g[GAME_ALPHA_INDEX]) for g in games
              if g[GAME_ALPHA_INDEX] is not None and g[GAME_ALPHA_INDEX] >= 0]
    median = float(np.median(values)) if values else 1.0
    return target_gap / max(np.sqrt(median), 1e-6)


K_FUNCS = {"legacy": legacy_k_margin, "fixed": compute_k_margin}

# Candidate retunes, all on the FIXED margin scale, aimed at the pile-up on the rating floor.
# (key, label, params). Nothing here is the production model: that is DEFAULT_PARAMS.
NO_FLOOR = -100.0
CANDIDATES = [
    ("no_floor", "1. No floor", ModelParams(r_min=NO_FLOOR)),
    ("gap4", "2a. Median gap 4", ModelParams(target_gap=4.0)),
    ("gap2_5", "2b. Median gap 2.5", ModelParams(target_gap=2.5)),
    ("loss0_1", "2c. Loss penalty 0.1", ModelParams(gamma_loss=0.1)),
    ("gap3_loss0_1", "2d. Gap 3, loss penalty 0.1", ModelParams(target_gap=3.0, gamma_loss=0.1)),
    ("two_sided", "3a. Two-sided margin", ModelParams(two_sided_margin=True)),
    ("two_sided_no_floor", "3b. Two-sided, no floor", ModelParams(two_sided_margin=True, r_min=NO_FLOOR)),
]


@contextlib.contextmanager
def _quiet():
    """Silence the models' prints and the solver's own C-level output."""
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


def _engine():
    load_dotenv()
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )


class Data:
    """One SELECT per season per table, then served from memory to the real model code."""

    def __init__(self, engine):
        self.engine = engine
        self._games = {}
        self._teams = {}

    def games(self, year, week=None):
        if year not in self._games:
            self._games[year] = pd.read_sql_query(
                text("SELECT * FROM rankings_games WHERE season = :s"), self.engine, params={"s": year})
        df = self._games[year]
        return df[df["week"] <= week].copy() if week else df.copy()

    def teams(self, year):
        if year not in self._teams:
            self._teams[year] = pd.read_sql_query(
                text("SELECT * FROM teams WHERE season = :s"), self.engine, params={"s": year})
        return self._teams[year].copy()


def _ranks(ratings: dict) -> dict:
    ordered = sorted((t for t in ratings if t != FCS), key=lambda t: ratings[t], reverse=True)
    return {team: i + 1 for i, team in enumerate(ordered)}


def _spearman(a: dict, b: dict) -> float:
    teams = sorted(set(a) & set(b))
    if len(teams) < 2:
        return float("nan")
    return float(np.corrcoef([a[t] for t in teams], [b[t] for t in teams])[0, 1])


def _top(ranks: dict, n: int) -> list:
    return [t for t, _ in sorted(ranks.items(), key=lambda kv: kv[1])[:n]]


def solve(mode, year, week, prior_cache, params=DEFAULT_PARAMS):
    """Run the real weekly model with the chosen scale in BOTH the weekly and the prior model."""
    k_func = K_FUNCS[mode]
    weekly_model.compute_k_margin = k_func
    prior_model.compute_k_margin = k_func

    def cached_prior(prior_year, prior_params=DEFAULT_PARAMS):
        key = (mode, prior_params, prior_year)
        if key not in prior_cache:
            prior_cache[key] = prior_model.get_prior_ratings(prior_year, prior_params)
        return dict(prior_cache[key])  # get_ratings adds new teams to the dict it is given

    weekly_model.get_prior_ratings = cached_prior
    with _quiet():
        ratings, records = weekly_model.get_ratings(year, week, params)
    return ratings, records


def _on_floor(ratings: dict, floor: float = FLOOR) -> int:
    """Teams the solver clamped to the minimum rating: tied with each other, in no real order."""
    return sum(1 for t, r in ratings.items() if t != FCS and r <= floor + FLOOR_TOLERANCE)


def _largest_tie(ratings: dict) -> int:
    """Size of the biggest group of teams within 0.001 of each other, wherever it sits."""
    values = sorted(r for t, r in ratings.items() if t != FCS)
    best = run = 1
    for a, b in zip(values, values[1:]):
        run = run + 1 if b - a <= FLOOR_TOLERANCE else 1
        best = max(best, run)
    return best if values else 0


def compare_week(data, stored, year, week, prior_cache):
    _teams, games, _fcs, _records, _conn = process_data.get_data_by_year_up_to_week(year, week)
    margins = [float(g[GAME_MARGIN_INDEX]) for g in games]
    k_legacy = legacy_k_margin(games, TARGET_GAP)
    k_fixed = compute_k_margin(games, TARGET_GAP)

    def capped(k):
        return sum(1 for m in margins if k * np.sqrt(max(m, 0.0)) >= CAP) / len(margins) if margins else 0.0

    legacy, records = solve("legacy", year, week, prior_cache)
    fixed, _ = solve("fixed", year, week, prior_cache)
    r_legacy, r_fixed = _ranks(legacy), _ranks(fixed)
    candidates = {}
    for key, _label, params in CANDIDATES:
        ratings, _ = solve("fixed", year, week, prior_cache, params)
        ranks = _ranks(ratings)
        values = [ratings[t] for t in ranks]
        candidates[key] = {
            "ratings": ratings,
            "floor": _on_floor(ratings, params.r_min),
            "largest_tie": _largest_tie(ratings),
            "spearman_vs_fixed": _spearman(ranks, r_fixed),
            "top25_shared_vs_fixed": len(set(_top(ranks, 25)) & set(_top(r_fixed, 25))),
            "top10_shared_vs_fixed": len(set(_top(ranks, 10)) & set(_top(r_fixed, 10))),
            "no1": _top(ranks, 1)[0], "min": min(values), "max": max(values),
        }
    moves = {t: r_legacy[t] - r_fixed[t] for t in r_legacy if t in r_fixed}  # + = rises after fix
    top25_moves = [abs(moves[t]) for t in set(_top(r_legacy, 25)) | set(_top(r_fixed, 25)) if t in moves]
    biggest = max(moves, key=lambda t: abs(moves[t])) if moves else None

    stored_ratings = stored.get((year, week), {})
    r_stored = _ranks(stored_ratings)
    common = [t for t in stored_ratings if t in legacy and t != FCS]
    return {
        "season": year, "week": week, "games": len(games),
        "median_margin": median_margin(games),
        "k_legacy": k_legacy, "k_fixed": k_fixed,
        "capped_legacy": capped(k_legacy), "capped_fixed": capped(k_fixed),
        "spearman": _spearman(r_legacy, r_fixed),
        "top10_shared": len(set(_top(r_legacy, 10)) & set(_top(r_fixed, 10))),
        "top25_shared": len(set(_top(r_legacy, 25)) & set(_top(r_fixed, 25))),
        "top25_same_spot": sum(1 for a, b in zip(_top(r_legacy, 25), _top(r_fixed, 25)) if a == b),
        "top25_max_move": max(top25_moves) if top25_moves else 0,
        "max_move": abs(moves[biggest]) if biggest else 0,
        "max_move_team": f"{biggest} ({r_legacy[biggest]} to {r_fixed[biggest]})" if biggest else "",
        "no1_legacy": _top(r_legacy, 1)[0], "no1_fixed": _top(r_fixed, 1)[0],
        "rating_range_legacy": max(legacy[t] for t in r_legacy) - min(legacy[t] for t in r_legacy),
        "rating_range_fixed": max(fixed[t] for t in r_fixed) - min(fixed[t] for t in r_fixed),
        # Legacy re-solve against what is published: anything but ~0 here is NOT the fix.
        "stored_teams": len(common),
        "stored_max_rating_diff": max((abs(stored_ratings[t] - legacy[t]) for t in common), default=float("nan")),
        "stored_rank_changes": sum(1 for t in common if r_stored.get(t) != r_legacy.get(t)),
        "floor_legacy": _on_floor(legacy), "floor_fixed": _on_floor(fixed),
        "floor_stored": _on_floor(stored_ratings),
        "tie_fixed": _largest_tie(fixed), "_candidates": candidates,
        "_legacy": legacy, "_fixed": fixed, "_records": records, "_stored": stored_ratings,
    }


def render(rows) -> str:
    out = ["# Margin-scale fix: before and after", ""]
    out += [
        "Legacy = the scale as published (median of the home/away multiplier). "
        "Fixed = median of the margins. Both solved today on the stored games.", "",
        "## Scale and cap", "",
        "| Season | Week | Games | Median margin | k legacy | k fixed | Capped legacy | Capped fixed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        out.append(
            f"| {r['season']} | {r['week']} | {r['games']} | {r['median_margin']:.1f} | {r['k_legacy']:.2f} "
            f"| {r['k_fixed']:.2f} | {r['capped_legacy']:.0%} | {r['capped_fixed']:.0%} |")
    out += [
        "", "## Ordering", "",
        "| Season | Week | Rank corr. | Top 10 shared | Top 25 shared | Top 25 same spot | "
        "Biggest top-25 move | Biggest move | No. 1 legacy | No. 1 fixed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        out.append(
            f"| {r['season']} | {r['week']} | {r['spearman']:.3f} | {r['top10_shared']} | {r['top25_shared']} "
            f"| {r['top25_same_spot']} | {r['top25_max_move']} | {r['max_move_team']} "
            f"| {r['no1_legacy']} | {r['no1_fixed']} |")
    out += [
        "", "## Rating spread and drift from the published numbers", "",
        "Drift compares the legacy re-solve with the stored `ratings` rows. It is not caused by the "
        "fix: it is games corrected since that week was published, or solver noise.", "",
        "| Season | Week | Spread legacy | Spread fixed | Stored teams | Max rating drift | Ranks that differ |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "SPREAD_ROWS",
        "", "## Teams on the rating floor", "",
        "A team on the floor (0.01) is tied with every other team there, so its rank among them "
        "is arbitrary.", "",
        "| Season | Week | Teams | Published | Legacy re-solve | Fixed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        out.append(
            f"| {r['season']} | {r['week']} | {r['rating_range_legacy']:.1f} | {r['rating_range_fixed']:.1f} "
            f"| {r['stored_teams']} | {r['stored_max_rating_diff']:.3f} | {r['stored_rank_changes']} |")
    spread_at = out.index("SPREAD_ROWS")
    floor_header = out[spread_at + 1:spread_at + 8]
    spread_rows = out[spread_at + 8:]
    out = out[:spread_at] + spread_rows + floor_header
    for r in rows:
        out.append(
            f"| {r['season']} | {r['week']} | {len(r['_fixed']) - (1 if FCS in r['_fixed'] else 0)} "
            f"| {r['floor_stored']} | {r['floor_legacy']} | {r['floor_fixed']} |")

    out += ["", "## Candidate retunes: teams tied in the largest group", "",
            "All on the fixed margin scale. \"Fixed\" is the fix with today's constants.", "",
            "| Season | Week | Fixed | " + " | ".join(label for _k, label, _p in CANDIDATES) + " |",
            "| --- | --- | --- | " + " | ".join("---" for _ in CANDIDATES) + " |"]
    for r in rows:
        out.append(f"| {r['season']} | {r['week']} | {r['tie_fixed']} | "
                   + " | ".join(str(r["_candidates"][k]["largest_tie"]) for k, _l, _p in CANDIDATES) + " |")
    for season in sorted({r["season"] for r in rows}):
        last = max((r for r in rows if r["season"] == season), key=lambda r: r["week"])
        out += ["", f"## Candidate retunes at {season} week {last['week']}, against the fix alone", "",
                "| Candidate | On its floor | Largest tie | Lowest | Highest | Rank corr. | Top 10 shared | Top 25 shared | No. 1 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key, label, _p in CANDIDATES:
            c = last["_candidates"][key]
            out.append(f"| {label} | {c['floor']} | {c['largest_tie']} | {c['min']:.2f} | {c['max']:.2f} "
                       f"| {c['spearman_vs_fixed']:.3f} | {c['top10_shared_vs_fixed']} | {c['top25_shared_vs_fixed']} | {c['no1']} |")
    for season in sorted({r["season"] for r in rows}):
        last = max((r for r in rows if r["season"] == season), key=lambda r: r["week"])
        legacy_ranks, fixed_ranks = _ranks(last["_legacy"]), _ranks(last["_fixed"])
        out += ["", f"## {season} week {last['week']}: top 25, fixed against legacy", "",
                "| Fixed rank | Team | Fixed rating | Legacy rank | Legacy rating |",
                "| --- | --- | --- | --- | --- |"]
        for team in _top(fixed_ranks, 25):
            out.append(f"| {fixed_ranks[team]} | {team} | {last['_fixed'][team]:.2f} "
                       f"| {legacy_ranks.get(team, '')} | {last['_legacy'].get(team, float('nan')):.2f} |")
        dropped = [t for t in _top(legacy_ranks, 25) if fixed_ranks.get(t, 99) > 25]
        if dropped:
            out += ["", "Out of the top 25 after the fix: "
                    + ", ".join(f"{t} ({legacy_ranks[t]} to {fixed_ranks[t]})" for t in dropped) + "."]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=None, help="Only this season. Default: every published season.")
    parser.add_argument("--out", type=str, default=None, help="Also write the report to this path.")
    parser.add_argument("--dump-json", type=str, default=None,
                        help="Also write every week's ratings (published, legacy, fixed) and records as JSON.")
    args = parser.parse_args()

    engine = _engine()
    data = Data(engine)
    try:
        stored_df = pd.read_sql_query(text("SELECT team, rating, season, week FROM ratings"), engine)
        if args.season is not None:
            stored_df = stored_df[stored_df["season"] == args.season]
        stored = {
            (int(s), int(w)): dict(zip(g["team"], g["rating"].astype(float)))
            for (s, w), g in stored_df.groupby(["season", "week"])
        }
        if not stored:
            print("FAIL: no published weeks found in ratings.", file=sys.stderr)
            sys.exit(1)

        # The real model code, fed from memory instead of one connection per call.
        process_data.get_games_by_year_week = data.games
        process_data.get_fbs_teams_by_year = data.teams

        prior_cache, rows = {}, []
        for (year, week) in sorted(stored):
            print(f"solving season={year} week={week} ...", file=sys.stderr, flush=True)
            rows.append(compare_week(data, stored, year, week, prior_cache))
    finally:
        engine.dispose()

    if args.dump_json:
        dump = [{
            "season": r["season"], "week": r["week"],
            "teams": [
                {"team": t, "fixed": round(r["_fixed"][t], 3), "legacy": round(r["_legacy"].get(t, float("nan")), 3),
                 "published": (round(r["_stored"][t], 3) if t in r["_stored"] else None),
                 "w": r["_records"][t][0], "l": r["_records"][t][1],
                 "c": {k: round(r["_candidates"][k]["ratings"][t], 3) for k, _l, _p in CANDIDATES}}
                for t in r["_fixed"] if t != FCS
            ],
        } for r in rows]
        dump = {"candidates": [{"key": k, "label": l} for k, l, _p in CANDIDATES], "weeks": dump}
        Path(args.dump_json).write_text(json.dumps(dump, separators=(",", ":")))

    report = render(rows)
    if args.out:
        Path(args.out).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
