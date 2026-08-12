# AGENTS.md

This file is the high-level entry point for local agent context in `cfb-rankings`.

It should help future work stay aligned with the repo's actual role and the relationship to `personal-site`, which is now the live public presentation layer.

## Purpose

`cfb-rankings` is the pipeline and modeling repo for the college football rankings project.

At a high level, this repo should:

- ingest source data from CollegeFootballData
- maintain the ranking model and supporting data-processing logic
- write durable ranking outputs for publication
- support repeatable scheduled and manual runs
- remain the source of truth for modeling behavior and pipeline operations

This repo is not intended to become the long-term public website surface.

## Current State

- the Streamlit app has been retired: `frontend/streamlit_app.py` is deleted and the deployed Streamlit Cloud app has been taken down
- the public-facing rankings experience now lives in `personal-site`, which reads published ranking artifacts from Cloudflare R2 (it does not query Supabase or this repo's database directly)
- scheduled and manual GitHub Actions runs exist under `.github/workflows/`
- the model and pipeline are already cleanly separated from the website repo
- Supabase remains this repo's internal operational database (ingestion + model execution); it is no longer read by any public-facing surface

## Long-Term Vision

The primary long-term product direction is:

- ~~migrate the public rankings experience from the Streamlit app to `personal-site`~~ -- **done**: the Streamlit app is retired and `personal-site` is live, reading real ranking artifacts from R2
- keep `cfb-rankings` focused on pipeline, model, artifact generation, and publishing
- let `personal-site` become the stable public presentation layer for rankings, methodology, and project context

The intended architecture direction is static-first:

- `cfb-rankings` computes rankings
- `cfb-rankings` publishes stable output artifacts
- `personal-site` consumes those artifacts and renders the public rankings page

This is preferred over tightly coupling the website to the database at request time.

## Relationship To `personal-site`

`personal-site` should be treated as the permanent public home for this project.

Repo boundary:

- `cfb-rankings`: data ingestion, modeling, pipeline execution, artifact publishing, operational workflows
- `personal-site`: page design, copy, methodology presentation, rankings display, public routing

Migration work should strengthen this boundary rather than blur it.

Do not move model logic or heavy pipeline behavior into `personal-site` just to speed up the migration.

## Preferred Publishing Direction

Public rankings should ultimately be delivered as durable artifacts rather than by having the website directly query the operational database.

Likely options to evaluate:

- static JSON artifacts committed or published for site consumption
- Supabase-backed artifact storage if that remains operationally simplest
- Cloudflare-friendly artifact storage aligned with the eventual site deployment path

**Resolved**: Cloudflare R2 was chosen. `artifacts/r2.py` publishes a JSON ranking artifact per season/week after each pipeline run, and `personal-site` consumes it directly -- Supabase is not in the public read path.

Current bias:

- prefer static artifact delivery for the website
- avoid making the public page dependent on live DB availability
- keep the website resilient even when pipeline infrastructure is idle between season runs

## Near-Term Priorities

1. ~~define the migration architecture from Streamlit to `personal-site`~~ -- **done**
2. ~~decide the ranking artifact format and publishing location~~ -- **done**: R2, see `artifacts/r2.py`
3. simplify or restructure GitHub Actions around artifact production and publishing
4. ~~replace the current Streamlit-centric public experience with a stronger page in `personal-site`~~ -- **done**: `personal-site` is live
5. keep the methodology accurate as model behavior evolves

## Constraints And Preferences

- preserve this repo as the source of truth for the model and pipeline
- prefer stable static publication over dynamic runtime coupling
- keep operational complexity proportional to the scale of the project
- make in-season updates reliable and easy to inspect
- favor incremental migration over a one-shot rewrite
- keep design-heavy UI work in `personal-site`, not here

## Known Open Questions

- ~~what exact artifact shape should be published for the site~~ -- **resolved**: `{season, week, generated_at_utc, rankings:[{rank, team, conference, record, rating, delta}]}`, see `artifacts/r2.py`
- ~~where should artifacts live long-term~~ -- **resolved**: Cloudflare R2
- should the pipeline still write to database tables once the site no longer depends on them -- still yes for now: Supabase remains the operational DB the pipeline/model run against; only the public read path moved off it
- how much historical season/week browsing should the public site support at launch -- open, see `docs/cfb-site-migration/requirements.yaml`
- ~~whether GitHub Actions should publish directly into `personal-site`, to shared storage, or to both~~ -- **resolved**: shared storage (R2), not committed into `personal-site`

## Guidance For Future Agents

- the public rankings page is served from `personal-site`, reading published R2 artifacts -- this repo has no frontend of its own anymore
- when making pipeline changes, consider whether they improve artifact publishing and operational clarity
- when making changes that affect published artifacts, prefer decisions that reduce coupling to the public site (schema changes are a cross-repo concern)
- if planning further site-facing work, coordinate with the local context already defined in `../personal-site/AGENTS.md` and `../personal-site/agent-context/pages/cfb-rankings.md`

## Suggested Workflow

For substantial migration work, prefer the shared workflow in `.agents/` (git subtree). Read
`.agents/AGENTS.md` first, then invoke it with `/work` — a single orchestrator that handles
discovery, planning, execution, and verification internally, not separate phase commands.

Durable planning artifacts should live under:

- `docs/<feature>/requirements.yaml`
- `docs/<feature>/plan.yaml`
- `docs/<feature>/implementation-report.yaml`

Use this file for stable repo-level direction, not for detailed feature plans.
