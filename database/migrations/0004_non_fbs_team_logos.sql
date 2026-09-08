-- Migration: 0004_non_fbs_team_logos
-- Feature: schedule-grid (T7a)
--
-- Purpose: give the Season Grid a place to look up logos for non-FBS
-- Division-I opponents (FCS today; ii/ii-iii/iii classifications are stored
-- too, in case they ever show up as an FBS opponent's opponent). Today those
-- opponents render as plain text because `teams` has no row for them.
--
-- WHY A SEPARATE TABLE, NOT MORE ROWS IN `teams`: `teams` is the rating
-- model's definition of the FBS universe, read by four different
-- consumers that all assume every row is FBS --
-- model/process_data.py:143 (`SELECT * FROM teams WHERE season = {year}`
-- becomes the model's team list -- an FCS row here would make the model
-- rate FCS teams and treat FBS-vs-FCS games as FBS-vs-FBS, corrupting every
-- rating), database/get_games.py:135 (`SELECT school FROM teams WHERE
-- season = {year}` filters which games are ingested),
-- artifacts/schedule.py:796 (`_fetch_teams_meta`; its keys become the set of
-- teams that get Season Grid rows), and main.py:31 (the "season has teams"
-- guard). Adding FCS rows to `teams` would corrupt all four. `non_fbs_teams`
-- is a disjoint, parallel table: same (season, school) grain as `teams`,
-- populated by the same CFBD `/teams` payload, but scoped in the ingest
-- code (T7a) to non-FBS classifications only, so it can never overlap with
-- or be mistaken for the FBS universe. Looking up logos by joining
-- COALESCE(teams, non_fbs_teams) is a later, separate task (T7b) -- this
-- migration and its ingest only make the data available.
--
-- GRAIN: (season, school) -- matches `teams`' own (id, season) grain in
-- spirit (one row per team per season) but keys on school rather than
-- CFBD's numeric id, since `school` is the join key every other table in
-- this schema already uses (games.home_team/away_team,
-- schedule_grid.team/opponent, ratings.team). `id` is still stored
-- (CFBD's own team id) for reference/debugging, but is not part of the key
-- -- CFBD ids are not currently relied on anywhere else in this schema, and
-- (season, school) is what T7b's lookup will actually join on.
--
-- COLUMNS: mirrors the subset of `teams`' columns relevant to rendering a
-- logo/label (school, mascot, abbreviation, conference, classification,
-- color, alternatecolor, logos), not the full stadium/location column set
-- `teams` carries -- that data has no known consumer for a non-FBS
-- opponent in this feature.
--
-- classification is NOT constrained to a fixed set here: CFBD's own
-- documented values today are fbs, fcs, ii, ii/iii, iii, but this table's
-- only real contract is "not fbs" (enforced by the ingest, not the schema)
-- -- a CHECK constraint would just be one more thing to update if CFBD adds
-- or renames a classification.

CREATE TABLE non_fbs_teams (
  id INTEGER,
  season INTEGER NOT NULL,
  school TEXT NOT NULL,
  mascot TEXT,
  abbreviation TEXT,
  conference TEXT,
  classification TEXT,
  color TEXT,
  alternatecolor TEXT,
  logos TEXT[],
  PRIMARY KEY (season, school)
);

ALTER TABLE non_fbs_teams ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public can read non_fbs_teams"
  ON non_fbs_teams
  FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Service can write non_fbs_teams"
  ON non_fbs_teams
  FOR ALL
  TO public
  USING (auth.role() = 'service_role'::text)
  WITH CHECK (auth.role() = 'service_role'::text);
