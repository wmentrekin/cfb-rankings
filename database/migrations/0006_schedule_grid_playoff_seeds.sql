-- 0006: expose the CFP seed on schedule_grid, per team.
--
-- WHY THIS EXISTS
-- ---------------
-- Migration 0005 adds games.playoff_home_seed / games.playoff_away_seed so the Season Grid can
-- label a first-round CFP bye as "Bye (No. 1)" rather than a bare "CFP Bye". But adding the
-- columns to `games` is not enough on its own, and the reason is easy to miss:
-- artifacts/schedule.py reads `SELECT * FROM schedule_grid`, and schedule_grid is NOT
-- `SELECT * FROM games` -- it is a view with an explicit column list, repeated in both halves of
-- a UNION ALL. A column absent from that list simply does not exist downstream, however well
-- populated it is in the base table. Without this migration the seed could never reach the
-- artifact, and every bye would degrade to the plain label forever -- looking exactly like the
-- intended "CFBD did not populate a seed" fallback while actually being a plumbing gap.
--
-- WHY team_seed/opp_seed RATHER THAN THE RAW COLUMNS
-- --------------------------------------------------
-- schedule_grid's whole idiom is one row per team per game, from that team's perspective: it
-- already flips home_score/away_score into team_score/opp_score, and home_conference/
-- away_conference into conference. Passing playoff_home_seed/playoff_away_seed through
-- unflipped would break that idiom and force every consumer to re-derive "which side am I?"
-- from home_away -- duplicating, in Python, the exact flip this view exists to perform once, in
-- one place. So the seed is flipped here like everything else.
--
-- IDEMPOTENCY / ORDERING
-- ----------------------
-- CREATE OR REPLACE VIEW requires the existing column list to be preserved in order and type;
-- the two new columns are appended at the end, which satisfies that. Run 0005 first -- this
-- migration references columns 0005 creates.

CREATE OR REPLACE VIEW schedule_grid AS
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
  playoff_bowl_name,
  team_seed,
  opp_seed
FROM (
  SELECT
    id AS game_id, season, week, season_type, start_date,
    home_team AS team, away_team AS opponent,
    home_score AS team_score, away_score AS opp_score,
    home_conference AS conference,
    'home'::text AS home_away,
    neutral_site, conference_game, notes,
    playoff_round_name, playoff_round_order, playoff_bracket_slot, playoff_bowl_name,
    playoff_home_seed AS team_seed, playoff_away_seed AS opp_seed
  FROM games

  UNION ALL

  SELECT
    id AS game_id, season, week, season_type, start_date,
    away_team AS team, home_team AS opponent,
    away_score AS team_score, home_score AS opp_score,
    away_conference AS conference,
    'away'::text AS home_away,
    neutral_site, conference_game, notes,
    playoff_round_name, playoff_round_order, playoff_bracket_slot, playoff_bowl_name,
    playoff_away_seed AS team_seed, playoff_home_seed AS opp_seed
  FROM games
) team_games;
