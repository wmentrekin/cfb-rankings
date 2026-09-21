# Postseason assignment — discovery notes (2026-09-21)

Ephemeral `$work` scratch. Durable facts move into code/config with provenance at build time.
External facts below are **search-snippet corroborated only**: outbound page fetches are blocked
by this sandbox's network policy (curl to wikipedia/cfp.com/CFBD returns no connection). Anything
marked *contested* or *provisional* needs a primary source supplied by the repo owner or a fetch-
capable session before it is encoded.

## What the repo already provides (repo-researcher-1)
- `artifacts/schedule_standings.py` is pure. `compute_standings(rows, season, ...)` returns per team:
  conference, overall record, conference record (None for independents), championship_status,
  bowl_status (`eligible` at 6 wins / `ineligible` at 7 losses / else `possible`; no FCS-win cap, no APR).
- `artifacts/schedule.py::_resolve_conference_champions(champ_games_by_conf, rows, season)` ->
  `{raw_conference: team}`, populated only once the CCG is PLAYED. Raw conference strings
  ("Big Ten") differ from the rankings artifact's display strings ("BIG 10"); map via
  `artifacts/rankings.py::CONFERENCE_DISPLAY_NAMES`. Independents are raw `"FBS Independents"`.
- Rankings come only from the DB (`get_ratings_with_conference`) or the published artifact
  `rankings/{season}/latest.json` (`{rank, team, conference, record, rating, delta, logo}`).
- `games` rows store `season_type`, `notes` (bowl full name), `playoff_round_name`,
  `playoff_bowl_name`, `playoff_bracket_slot`, seeds (schema only, never confirmed populated).
  `main.py` re-ingests the whole postseason slate every run. Seasons actually present in the DB
  beyond 2025/2026 are UNVERIFIED (live-DB question; out of subagent scope).
- `artifacts/bowl_names.py` curated roots; 43-name 2025-26 pinned table in
  `tests/test_postseason_slots.py`. Display-only; carries no tie-in information.
- `artifacts/conference_tiebreakers.json` + `tiebreaker_rules.py` = the config pattern to copy:
  schema_version, season_min/season_max entries, provenance/source/notes per entry, strict
  unknown-key rejection, season-overlap check, one typed error class, loader tests.
- Publishing: `rankings/{y}/week-NN.json`+`latest.json`+`rankings/latest.json`+self-healing
  `index.json` from bucket listing; publish functions never raise. Weekly run Sunday 07:00 UTC;
  hook point after `publish_schedule_artifact` in `main.py` (own try/except, `if not args.staging`).
- Tests: `_row(...)` builder, invented records, module constants, MUTATION-CHECK LOG docstrings.

## personal-site (repo-researcher-2)
- `scripts/fetch-rankings-data.mjs` hard-codes two artifact kinds; a third kind = new R2 prefix,
  parallel fetch loop, local fallback check. Page is build-time filesystem reads + client fetch of
  `/data/cfb/...` on season change. Third tab vs separate route both viable; route is cleaner for a
  distinct bracket UX. Reuse `src/lib/conferences.ts`; reuse `playoff_round`/`cfp_seed` field names.

## CFP 2026-27 (platform-researcher-1)
- 12 teams for 2026 (CFP extension announced 2026-01-23). 2027+ undecided -> field size is config.
- AQ: champions of ACC, Big Ten, Big 12, SEC (named) + one Group of Six slot.
  Follow-up pass (5 outlets, 2026-01 .. 2026-08-11 NCAA.com explainer): the 2026 G6 slot goes to the
  highest-ranked G6 TEAM, champion or not, per the 2024 MOU; the only dissent is an unadopted
  reversal push by G6 commissioners. High confidence, primary text unverified. Engine carries
  `g6_slot_rule: champion | any_team`; recommended default `any_team`; owner decides at checkpoint.
- AQ first, then at-large from the remaining highest-ranked teams (2024: Clemson #16 in, Alabama
  #11 out).
- Notre Dame: MOU signed 2024-03, effective 2026 season: final rank <= 12 => guaranteed berth
  (<= 13 in a 14-team field; lapses above 14). Did not apply in 2025. Owner's encoding = default.
- Straight seeding by rank; top 4 byes regardless of champion status; no re-seeding; no
  conference-mate or rematch avoidance; no special seed for the 5th champion.
- 2026-27 bracket: R1 Dec 18-19 at higher seed (5v12, 6v11, 7v10, 8v9). QF: Fiesta Dec 30; Cotton,
  Rose, Peach Jan 1 2027 (1 vs 8/9, 2 vs 7/10, 3 vs 6/11, 4 vs 5/12). SF: Orange Jan 14, Sugar Jan 15.
  NCG: Mon Jan 25 2027, Allegiant Stadium, Las Vegas. Seeds 1-3 CHOOSE their QF bowl in rank order;
  engine needs a deterministic stand-in.
- All six NY6 bowls host CFP games every year -> none is in the ordinary tie-in pool.

## Bowl slate + tie-ins 2026-27 (platform-researcher-2)
- Key unlock: Bowl Season announced that all non-CFP tie-ins for 2026-27 CARRY OVER from 2025-26
  (re-alignment postponed until the CFP format settles). Deltas: + Puerto Rico Bowl (MAC vs G5/
  at-large, Dec 22, Bayamón), + Poinsettia Bowl revived (Pac-12 champion if not in CFP vs a
  "Pac-12 legacy team" pool: Arizona, Arizona State, Cal, Colorado, Oregon, Stanford, UCLA, USC, Utah,
  Washington), - LA Bowl, - GameAbove/Detroit. Xbox Bowl (Frisco; CUSA vs Sun Belt) replaced the
  Bahamas Bowl already in 2025-26. A second Frisco game on Dec 15 is *provisional*.
- Total non-CFP count unconfirmed (secondary sources disagree; ~40-44).
- Partial tie-in table (provisional): Citrus SEC/B1G; ReliaQuest SEC-pool/B1G-or-ACC; Duke's Mayo
  ACC/SEC-or-B1G; Music City SEC-pool/B1G; Liberty SEC-pool/B12; Texas SEC-pool/B12; Gator
  SEC-pool/ACC; Pinstripe ACC/B1G; Cactus B12/B1G; Alamo B12/Pac-12-legacy; Pop-Tarts B12/ACC;
  Guaranteed Rate B1G/B12; Independence B12?/AAC (LOW confidence); Xbox CUSA/SBC; New Orleans
  SBC #2/CUSA? (conflict); 68 Ventures SBC #5/MAC; Arizona MAC/MWC; Idaho Potato MAC/MWC-or-at-large;
  Hawai'i CUSA/MWC; New Mexico MWC/CUSA-or-at-large; Armed Forces B12-or-AAC/pool; Boca Raton, Cure,
  Frisco, Salute to Veterans, Myrtle Beach = Sun Belt picks 1-6 vs G5/at-large.
- Selection procedures: SEC = Citrus first, then a "Pool of Six" (Gator, Music City, Liberty, Texas,
  ReliaQuest, Las Vegas) placed by the SEC office in consultation (geography, matchups, avoiding
  repeats). ACC = three tiers; published criteria "geographic proximity, avoiding repeat
  appearances and matchups, regular-season won-loss records". Big Ten = Citrus first, remaining
  order not officially published. Big 12 = published "Bowl Selection Central" page (unfetchable);
  projected Alamo -> Pop-Tarts -> Texas -> Liberty -> Guaranteed Rate -> Independence. Sun Belt =
  numbered draft (ESPN-owned bowls picks 1/3/4, New Orleans 2, 68 Ventures 5). AAC/MAC/CUSA/MWC:
  no published procedure found.
- No currently published hard rules found for "one fewer win" or "no consecutive-year repeat";
  only the ACC's soft "avoid repeat appearances and matchups". Treat as soft constraints, infer
  empirically.
- Rebuilt Pac-12 / Mountain West split for 2026-27 is the most unsettled area (Poinsettia is the
  only confirmed bespoke tie-in; ~12 combined non-CFP slots reported for ~28 eligible teams).
- Prestige priors: CBS Sports annual bowl rankings; payout figures (businessofcollegesports.com).
- History source: CFBD `/games?seasonType=postseason` `notes` (already ingested) primary;
  Wikipedia per-season bowl pages as cross-check.

## Primary sources the owner can supply (fetch-blocked here)
- https://en.wikipedia.org/wiki/2026%E2%80%9327_NCAA_football_bowl_games (slate table)
- https://bowlseason.com/sports/bowl/schedule/2026-27
- https://collegefootballplayoff.com/news/2026/2/3/2627-2728-bowls and .../2026/1/23/2627-format
- SEC "Bowl Selection Process" page; theacc.com bowl selection page; bigten.org bowl selection;
  https://big12sports.com/news/2025/5/2/football-big-12-2025-bowl-selection-page.aspx
- pressdemocrat.com 2026-08-16 "Explaining 2026 tie-ins for legacy teams, newcomers"; mwcconnection
  tie-in breakdown
- Notre Dame / CFP MOU coverage (ESPN 2024-03-15; On3 2026 Bevacqua statement)

## CFP quarterfinal/semifinal placement (platform-researcher-4, 2026-09-21)
- Since 2025-26 (CFP release 2025-05-22): seed 1 selects its QF bowl AND the SF bowl it advances
  to; seeds 2 and 3 pick from the remaining QF sites in order (each pick fixes that half's SF
  path); seed 4 is assigned the remaining QF. Done on Selection Sunday (2026-12-06 reported).
  Stated factors: rank order, "current contract bowl relationships", geographic preference for
  the 1 seed's semifinal. No documented opponent-proximity or TV constraint found.
- Geometry: QF(1 vs 8/9) winner meets QF(4 vs 5/12) winner; QF(2 vs 7/10) meets QF(3 vs 6/11).
  Which SF bowl each half uses is fixed by the 1 seed's pick, not by slot.
- 2025-26 actual: 1 Indiana -> Rose (SF path Peach); 2 Ohio State -> Cotton (Fiesta); 3 Georgia ->
  Sugar (Fiesta); 4 Texas Tech -> Orange (Peach). Semis: Peach (Indiana v Oregon), Fiesta (Miami v
  Ole Miss).
- 2024-25 actual (committee placed by contract relationships, no team choice): 1 Oregon Rose, 2
  Georgia Sugar, 3 Boise State Fiesta, 4 Arizona State Peach. Semis: Cotton (Rose/Peach winners),
  Orange (Fiesta/Sugar winners).
- First round 2026-27: Fri Dec 18 one game 8pm ET; Sat Dec 19 three games; higher seed hosts;
  day/window per pairing set on Selection Sunday (not pre-set by seed).
- 2027-28 (CFP release 2026-07-09): QF Sugar Fri Dec 31 2027 + Fiesta/Peach/Rose Sat Jan 1 2028;
  SF Orange Jan 13 + Cotton Jan 14 2028 (SF date pairing from a truncated snippet); NCG New
  Orleans. 2028-29 NCG Tampa. Rotation: 25-26 QF Cotton/Orange/Rose/Sugar, SF Fiesta/Peach;
  26-27 QF Fiesta/Cotton/Rose/Peach, SF Orange/Sugar; 27-28 QF Sugar/Fiesta/Peach/Rose, SF
  Orange/Cotton.
- Design consequence: QF stand-in = per-conference ordered NY6 preference list (legacy
  relationships: B1G Rose; SEC Sugar; ACC Orange; B12 Fiesta/Cotton; others nearest/fallback),
  seeds 1-3 take the first available in order, seed 4 gets the remainder; SF bowl per half follows
  from the 1 seed's pick + config `semifinal_hosts`. Provenance "heuristic"; tagged stand_in.

## Mountain West / rebuilt Pac-12 / legacy pool (platform-researcher-11, partial: search quota hit)
- Legacy Pac-12 group (12 schools incl. Cal, Stanford, Oregon St, Wash St) tied to Alamo, Las
  Vegas, Holiday, Sun, Poinsettia, Independence; ordered by overall record; "won't be affiliated
  with the selection process for their current conferences" (pressdemocrat.com 2026-08-16; sunbowl.org
  release). Overflow -> ESPN bowl pool by availability. Poinsettia second slot = 10-team sub-pool
  excluding Oregon St / Wash St. Alamo: Big 12 12-team pool vs legacy 12-team pool, overall record.
- Holiday 2026-27: ACC vs legacy (mwcconnection); Las Vegas: Big Ten vs legacy; Sun: ACC vs legacy;
  Independence: legacy vs Big 12 (returns to Pac-12 rotation).
- Mountain West 2026-27 bowls (medium): Boca Raton, Idaho Potato, Puerto Rico, Armed Forces, Frisco,
  New Mexico, Hawai'i; Cactus/Rate = MWC backup when Big Ten/Big 12 cannot fill.
- Actuals captured: 2023-24 LA UCLA v Boise, Holiday USC v Louisville, Alamo Arizona v Oklahoma, Sun
  Oregon St v Notre Dame; 2024-25 Holiday Wash St v Syracuse, LA Boise v Washington; 2025-26 LA Boise
  v Washington, New Mexico SDSU v North Texas, Idaho Potato Utah St v Wash St, Arizona Fresno St v
  Miami (OH). Unknown: full MWC slates 2023-25, Oregon St 2024-26, residual MWC membership, counts.
- BLOCKER: session-wide WebSearch quota exhausted (200/200) mid-task; other deep dives will be
  partial. Options: raise CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION (owner), fresh sessions, or
  owner-supplied primary texts (preferred).
