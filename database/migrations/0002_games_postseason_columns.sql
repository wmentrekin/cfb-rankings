-- Migration: 0002_games_postseason_columns
-- Feature: schedule-grid (T2)
--
-- Purpose: widen the bronze `games` table to carry postseason/CFP identity
-- fields returned by CFBD's /games endpoint (top-level `notes`, plus a
-- nested `playoff` object present only on CFP-affiliated games). These are
-- prerequisites for T3's schedule_grid view, which buckets postseason rows
-- into Conference Championship / Army-Navy / CFP-round columns.
--
-- All five columns are nullable with no default: additive and safe for the
-- 24,873 existing rows regardless of their current season_type. Most rows
-- (regular season, non-CFP postseason) will have some or all of these as
-- null -- that's correct, not a gap.
--
-- Column mapping from CFBD's /games response (see T2 handoff for the
-- defensive-access rationale -- the nested `playoff` object's exact field
-- names were not live-verified from this sandbox):
--   notes                 <- Game.notes (top-level)
--   playoff_round_name    <- Game.playoff.round_name
--   playoff_round_order   <- Game.playoff.round_order
--   playoff_bracket_slot  <- Game.playoff.bracket_slot (CFBD documents this
--                            as a STRING, not numeric -- deliberately TEXT
--                            here, not INTEGER, to avoid silently nulling
--                            out any real value that isn't purely digits)
--   playoff_bowl_name     <- Game.playoff.bowl_name

ALTER TABLE games
  ADD COLUMN notes TEXT,
  ADD COLUMN playoff_round_name TEXT,
  ADD COLUMN playoff_round_order INTEGER,
  ADD COLUMN playoff_bracket_slot TEXT,
  ADD COLUMN playoff_bowl_name TEXT;
