-- Migration: 0001_rankings_games_view
-- Feature: schedule-grid (T1)
--
-- Purpose: introduce `rankings_games` as the silver-layer view the ranking
-- model reads from, guarding it against non-regular-season rows before any
-- postseason backfill (a later task) adds them to the shared `games` table.
--
-- This is a plain, single-predicate filter -- conference championship games
-- are already included under season_type='regular' today, not a separate
-- category needing detection logic. Bronze stays `games`; this is one of
-- two disjoint-predicate silver views over it (the other, schedule_grid,
-- lands in a later task).

CREATE VIEW rankings_games AS
SELECT *
FROM games
WHERE season_type = 'regular';
