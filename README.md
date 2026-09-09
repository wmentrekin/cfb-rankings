# **College Football Ranking Optimization**

📊 The public rankings app now lives in the sibling [`personal-site`](../personal-site) repo, which reads published ranking artifacts from Cloudflare R2 and renders them on the live site. This repo no longer ships a public frontend of its own.

## **What This Project Does**

This project builds an automated, data-driven **college football team ranking system** using convex quadratic programming (QP). Each week, the model ingests updated game results, optimizes team rating values that best explain on-field outcomes, and pushes the computed rankings to a cloud database for storage and publication.

The model is designed to:
- Rank all **FBS teams** based on results from the season up to the current week.  
- Account for **margin of victory, game location, and opponent strength.**  
- Handle **FCS losses** with a dummy “FCS” rating variable. 
- Smoothly transition from **prior-season ratings** to **current-season rankings** as the season progresses. 

---

## **Architecture Summary**

| Component | Purpose | Technology |
|------------|----------|-------------|
| **Data Source** | `requests` → `Supabase` | [College Football Data API](https://collegefootballdata.com/) |
| **Database** | Stores team, game, and model results tables | **[Supabase](https://supabase.com/) PostgreSQL** |
| **Data Processing** | Loads teams/games and prepares features | `process_data.py`, `get_games.py`, `get_teams.py` |
| **Model** | Solves convex QP for team ratings | `cvxpy`, `numpy`, `pandas` |
| **Artifact Publishing** | Publishes ranking artifacts (JSON) for the public site to consume | `artifacts/r2.py` → Cloudflare R2 |
| **Automation** | Weekly GitHub Action scheduled via cron (Sundays 3AM ET) | `.github/workflows/weekly-update.yml` |
| **Public Frontend** | Rankings display, methodology, and project context | sibling [`personal-site`](../personal-site) repo (reads R2 artifacts, deployed on Cloudflare Pages) |

**Data Flow Summary:**
1. GitHub Action triggers `main.py` every Sunday morning.  
2. New games are pulled from the API and inserted into Supabase.  
3. The model runs using all games up to the current week.  
4. Ratings are written back to the database.  
5. A ranking artifact (JSON) is published to Cloudflare R2, then `personal-site` is triggered to rebuild and pick it up.
6. Logs are saved for diagnostics.

### **Manual Result Overrides**

CFBD's own result for a game is not always the one the rankings should use -- an officiating
decision the repo owner disputes, for example. `database/game_result_overrides.json` is a
checked-in, version-controlled list of such corrections. On every run, `main.py` applies every
override declared for that season immediately after games are ingested and before the model
runs (`database/game_overrides.py`), so the override always wins over whatever CFBD reports on
the next re-ingest -- a manual database edit alone would just get overwritten. `winner`,
`margin`, and `alpha` are re-derived from the override's declared scores using the same helpers
ingest itself uses (`database/game_derivations.py`), so the two can never drift apart.

Each override must match the game it names on `season`, `week`, `season_type`, `home_team`, and
`away_team` -- a mismatch is skipped and logged as a failure rather than silently corrupting the
wrong game (surfaced as `OVERRIDE_FAILURE` in the run log and failed in CI by
`.github/workflows/weekly-update.yml`).

To add an override, append an entry to the `overrides` array in
`database/game_result_overrides.json`:

```json
{
  "game_id": 401858428,
  "season": 2026,
  "week": 1,
  "season_type": "regular",
  "home_team": "Michigan",
  "away_team": "Western Michigan",
  "home_score": 7,
  "away_score": 12,
  "reason": "Why this correction was made.",
  "declared_on": "2026-09-09"
}
```

`game_id` is the CFBD game id (find it via `games.id` for the game in question); `home_score`
and `away_score` are the corrected final score. `neutral_site` is intentionally not part of an
entry -- it is read from the stored `games` row, since it isn't something the operator is
declaring.

---

## **Local Setup (uv + Python 3.12)**

This repo now uses `uv` for dependency and environment management.

```bash
uv sync --extra pipeline
uv run python main.py
```

For custom runs:

```bash
uv run python main.py --year 2025 --week 1
```
