# **College Football Ranking Optimization**

An automated college football team ranking system. Each week it ingests game results, solves a
convex quadratic program for the team ratings that best explain those results — accounting for
margin of victory, game location and opponent strength — and publishes the rankings as JSON
artifacts to Cloudflare R2.

📊 **The public site is not here.** It lives in the sibling
[`personal-site`](../personal-site) repo, which reads the published artifacts from R2 and renders
them. This repo is the pipeline and the model only.

## Local setup

```bash
uv sync --extra pipeline
uv run python main.py              # current season/week, computed from the date
uv run python main.py --year 2025 --week 1
```
