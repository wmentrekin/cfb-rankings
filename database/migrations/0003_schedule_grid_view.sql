-- Migration: 0003_schedule_grid_view
-- Feature: schedule-grid (T3)
--
-- Purpose: introduce `schedule_grid` as the second silver-layer view over
-- the bronze `games` table (the first, rankings_games, landed in T1). Where
-- rankings_games is a plain single-predicate filter, schedule_grid is a
-- RESHAPE: `games` has one row per game (home_team/away_team/home_score/
-- away_score columns); the Season Grid feature needs "for team X, what
-- happened in week N", i.e. one row per team per game. This view is a
-- UNION ALL of the home-team perspective and the away-team perspective of
-- every row in `games` (all season_types -- no season_type filter here,
-- unlike rankings_games), so every games row becomes exactly two
-- schedule_grid rows.
--
-- IDENTIFIER: `games` stores teams as TEXT (home_team/away_team, matching
-- teams.school) -- there is no team_id integer column anywhere on `games`.
-- `team`/`opponent` below are therefore team-name TEXT, which is the
-- correct, actually-available key, not a fabricated numeric id.
--
-- GRAIN: (season, team, season_type, week) -- season_type is included in
-- the logical key (not just week) because postseason week numbers are
-- expected to collide with regular-season week numbers once postseason
-- rows exist (see T1/T2). A given (season, team, season_type, week) should
-- return at most one row in practice.
--
-- STATUS DERIVATION (load-bearing): played vs. upcoming is derived from
-- start_date vs. now(), NEVER from score nullness -- confirmed finding:
-- CFBD stores unplayed/future games with home_score=away_score=0, not
-- NULL, so a naive "score present" check would misclassify every future
-- game as a 0-0 tie/loss. status = 'upcoming' when start_date is in the
-- future; otherwise 'win'/'loss' by comparing team_score to opp_score.
-- This view never emits 'bye' -- a bye means NO games row exists for that
-- team/slot at all, which this view structurally cannot produce (it only
-- ever has rows for games that exist); synthesizing bye rows into the full
-- grid is a later, Python-side task (T4b), not a SQL concern.
--
-- 'tbd' STATUS EDGE CASE -- investigated, not guessed (per T3 handoff):
-- the handoff asked whether a postseason row might exist with a real
-- start_date but an undetermined opponent (e.g. an early CFP bracket slot
-- before both participants are known), which would warrant a distinct
-- 'tbd' status rather than 'upcoming'. Two things were checked against
-- live data in this sandbox:
--   1. No postseason rows exist at all yet (`SELECT count(*) FROM games
--      WHERE season_type = 'postseason'` = 0) -- T2's real backfill
--      (scripts/backfill_2025_postseason.py) has not been run here, so
--      there is no live data to exhibit or rule out this pattern.
--   2. The handoff's own suggested structural signal for it -- a
--      home_team/away_team value that doesn't match any row in
--      teams.school for that season -- was tested against the *existing*
--      24,873 regular-season rows and found to be UNRELIABLE: real,
--      perfectly legitimate games routinely have a non-FBS opponent that
--      is correctly absent from teams.school (teams.school only lists FBS
--      teams; get_games_by_year_week keeps a game if EITHER side is FBS),
--      e.g. real rows like Cincinnati vs. UT Martin or Eastern Illinois
--      vs. Western Illinois. Using "opponent not in teams.school" as a
--      'tbd' signal would misclassify these as 'tbd' instead of a normal
--      'upcoming'/'win'/'loss' game.
-- Given no live postseason evidence either way, and the one concretely
-- proposed signal disproven against real data, this view does NOT emit
-- 'tbd' and defaults every future row to 'upcoming', per the handoff's
-- explicit fallback ("if you don't find evidence of this pattern... default
-- to 'upcoming'... do not guess at a specific CFBD placeholder convention
-- without evidence"). Revisit once the real postseason backfill runs and
-- actual CFP-bracket rows can be inspected.
--
-- TIE / STALE-SCORE EDGE CASE: a row with start_date in the past and
-- team_score == opp_score is treated as 'upcoming' rather than a genuine
-- tie result. Two things can produce this: (a) an actual tied final score,
-- which is effectively impossible in modern CFB -- every FBS game since
-- 1996 goes to mandatory overtime until decided -- or (b) a game whose
-- start_date has passed but whose score CFBD hasn't posted/synced yet
-- (still 0-0, or otherwise equal, shortly after kickoff). (b) is the
-- realistic case, and treating it as 'upcoming' is the safe choice: it
-- never asserts a false 'win' or 'loss' for a game that isn't actually
-- final.
--
-- PASS THROUGH AS-IS (no transformation): neutral_site, conference_game,
-- notes, playoff_round_name, playoff_round_order, playoff_bracket_slot,
-- playoff_bowl_name, start_date, week, season_type. `game_id` (games.id)
-- is additionally carried through so a consumer can recognize both rows of
-- the same underlying game (e.g. T4b's Army-Navy carve-out needs to find
-- "the game between these two teams" specifically) -- not in the handoff's
-- explicit pass-through list, but a minimal, necessary addition for the
-- view to be usable at this grain.

CREATE VIEW schedule_grid AS
SELECT
  game_id,
  season,
  week,
  season_type,
  team,
  opponent,
  conference,
  home_away,
  team_score,
  opp_score,
  CASE
    WHEN start_date > now() THEN 'upcoming'
    WHEN team_score > opp_score THEN 'win'
    WHEN team_score < opp_score THEN 'loss'
    ELSE 'upcoming'
  END AS status,
  neutral_site,
  conference_game,
  start_date,
  notes,
  playoff_round_name,
  playoff_round_order,
  playoff_bracket_slot,
  playoff_bowl_name
FROM (
  SELECT
    id AS game_id, season, week, season_type, start_date,
    home_team AS team, away_team AS opponent,
    home_score AS team_score, away_score AS opp_score,
    home_conference AS conference,
    'home'::text AS home_away,
    neutral_site, conference_game, notes,
    playoff_round_name, playoff_round_order, playoff_bracket_slot, playoff_bowl_name
  FROM games

  UNION ALL

  SELECT
    id AS game_id, season, week, season_type, start_date,
    away_team AS team, home_team AS opponent,
    away_score AS team_score, home_score AS opp_score,
    away_conference AS conference,
    'away'::text AS home_away,
    neutral_site, conference_game, notes,
    playoff_round_name, playoff_round_order, playoff_bracket_slot, playoff_bowl_name
  FROM games
) team_games;
