# 2026-27 bowl tie-ins and selection models — DRAFT v2 (2026-09-21)

Seed data for `postseason/bowl_tie_ins.json` (task T3). Sources: platform-researcher-3 (first
table) + eight deep dives (R7-*/R8; all search-snippet only, all cut short by the session search
quota). Confidence per row. Anything below "high" is encoded `provisional` until the owner supplies
a primary text. Governing fact (multiple sources, 2026): **non-CFP tie-ins for 2026-27 carry over
unchanged from 2025-26**; the announced deltas are + Puerto Rico, + Poinsettia, - LA Bowl,
- GameAbove/Detroit (cancelled 2026-02-10, no replacement). Count: 35 non-CFP FBS bowls (best
supported; one source says 36).

## 0. Structural findings that change the engine model (see selection-logic-spec.md §4.0, §4.10-4.12)
1. **Affiliation groups.** The 12 legacy Pac-12 schools are a bowl group of their own for 2026-27
   (Alamo, Las Vegas, Holiday, Sun, Poinsettia, Independence), ordered by overall record, and are
   not in their current conference's selection. (pressdemocrat 2026-08-16; sunbowl.org)
2. **Operator pools.** ESPN Events owns ~17 bowls and places Group of Six teams across them as a
   flex pool. The Sun Belt's "picks 1/3/4" ARE the ESPN Events pool; the AAC fills "4 of 8" pool
   bowls; the MAC has no pecking order and is placed by geography; CUSA is "guaranteed seven".
   So G6 selection is not a per-conference numbered draft: it is a few fixed conference slots plus
   one operator-level pool fill. (sunbeltsports.org 2020-05-28; theamerican.org bowls page;
   getsomemaction.com; conferenceusa.com 2025-12-07; espnpressroom.com 2025-12)
3. **Declines and the 5-7 pool are real inputs.** 2024-25: Marshall withdrew from the Independence
   Bowl, replaced by 5-7 Louisiana Tech. 2025-26: Kansas State and Iowa State declined (fined
   $500k each), replacements drawn by APR rank; the Birmingham Bowl became Georgia Southern vs
   Appalachian State (both Sun Belt) after SEC/ACC/AAC schools declined. Rate Bowl and
   Independence Bowl had no Big 12 team as a result. The engine needs `declined` as an input and
   a replacement rule.
4. **SEC alternation is a published parity rule**: Las Vegas in even seasons, Duke's Mayo in odd
   (secsports.com selection-process page; confirmed by 2024 Texas A&M -> Las Vegas, 2025
   Mississippi State -> Duke's Mayo). 2026 is even -> Las Vegas. Duke's Mayo 2026 side A is
   therefore NOT SEC (likely Big Ten; unconfirmed).

## 1. Per-conference selection models (what the config `selection.conferences[C]` encodes)
| Group | Mode | Order / structure | Criteria quoted | Confidence |
|---|---|---|---|---|
| SEC | `citrus_then_office_pool` | CFP -> Citrus (best available) -> Pool of Six {ReliaQuest, Gator, Music City, Texas, Liberty, + Las Vegas (even) / Duke's Mayo (odd)} placed by the SEC office -> ESPN Events fills Birmingham / Gasparilla only if teams remain | "preferences expressed by the SEC's bowl eligible schools, input from the SEC's affiliated bowls, travel considerations, attention to previous matchups and additional relevant factors" (secsports.com) | high (structure) |
| Big 12 | `numbered` | Alamo -> Pop-Tarts -> Texas -> Liberty -> Rate/Cactus -> Independence ("rough pecking order... Alamo first, Independence last"); bowls may skip to avoid regular-season or prior-bowl rematches; a bowl may take any eligible team regardless of record | big12sports.com Bowl Selection Central (2025-05-02, re-served for 2026); aggregator quotes | medium (order secondary-sourced) |
| ACC | `tiered` (2 or 3 tiers — sources conflict; membership unpublished) | 17 possible destinations for 2026-27 (theacc.com 2026-06-03). Tier 1 reportedly Gator, Duke's Mayo, Sun (blog, low reliability); Tier 2 Gasparilla, Birmingham, First Responder; a backfill band (Independence, Armed Forces, Frisco Football Classic listed in the 2026 release). Criteria: "geographic proximity, avoiding repeat appearances and matchups, regular-season won-loss records" (theacc.com 2019-07-11; echoed 2026). Notre Dame: may take a non-CFP ACC slot if within one win of, equal to, or ranked above the ACC team; never over an ACC team with 2+ more wins (Wikipedia "Bids to college bowl games"; BCS-era text, current applicability unconfirmed). Cal/Stanford: dual eligibility (ACC slate AND legacy pool) in 2026-27, the legacy arrangement's final season. | medium |
| Big Ten | `numbered (inferred)` | No official order beyond Citrus. Observed 2025-26 (clean sample): Citrus > ReliaQuest > Music City > Pinstripe > Las Vegas > Rate/Cactus. Sixth bowl alternates: Las Vegas in ODD seasons, Duke's Mayo in EVEN (mirror of the SEC's rule; 2023 NW->LV, 2024 Minnesota->Mayo, 2025 Nebraska->LV). Norms (soft, sometimes broken): variety across years; avoid non-conference rematches; "don't pass over a team with 2 more wins" (2021 counter-example). | medium (order inferred from actuals) |
| Sun Belt | `numbered + operator_pool` | 1/3/4 = ESPN Events flex pool (Boca Raton, Cure, Frisco, Myrtle Beach, Salute to Veterans; ESPN also places into Idaho Potato, First Responder, New Mexico); 2 = New Orleans; 5 = 68 Ventures; beyond = G6 free-for-all | sunbeltsports.org 2020-05-28 lineup (rolled forward) | medium-high |
| AAC | `fixed_annual + operator_pool` | Annual: Armed Forces, Hawai'i, Military (vs ACC), Fenway (vs ACC). Then 4 of 8: Birmingham, Cure, Gasparilla, Boca Raton, Frisco, First Responder, New Mexico, Myrtle Beach. Secondary: Liberty (fired 2025: Navy), Quick Lane (defunct). No order. | theamerican.org "primary"/"secondary agreements" | medium-high |
| CUSA | `guaranteed_count + operator_pool` | "guaranteed seven bowl appearances"; New Orleans (vs Sun Belt) standing; 2025 slate: Myrtle Beach, First Responder, Independence, Salute to Veterans, New Orleans, Xbox, 68 Ventures | conferenceusa.com 2025-12-07 | medium |
| MAC | `operator_pool` (geography) | "does not have a pecking order... place them in games depending on geography"; bowls: Arizona, Idaho Potato, 68 Ventures, Puerto Rico (new); Detroit gone; NIU leaves for MWC 2026-07-01; UMass in since 2025 | secondary snippet (source page not isolated) | medium |
| Mountain West | `operator_pool` | ~4-7 slots: Arizona, Idaho Potato, New Mexico, Hawai'i (home), + Boca Raton/Puerto Rico/Armed Forces/Frisco access; Rate/Cactus is the MWC BACKUP when Big Ten/Big 12 cannot fill | InForum 2026; mwcconnection | medium |
| Pac-12 (rebuilt) | `champion_slot` | Champion (if not CFP) -> Poinsettia; others via ESPN pool by availability | pressdemocrat 2026-08-16; SI 2026-08/09 | high |
| Legacy Pac-12 group (12) | `pool_by_record` | Alamo, Las Vegas, Holiday, Sun, Poinsettia (10-team sub-pool excl. OSU/WSU), Independence; ordered by overall record; overflow -> ESPN pool | pressdemocrat 2026-08-16; sunbowl.org; sanantonioreport | high |

## 2. Bowl table (35 non-CFP; slot A / slot B are the best-supported 2026-27 sources)
| Root | 2026 name | City | Date | Slot A | Slot B | Notes | Conf. |
|---|---|---|---|---|---|---|---|
| Citrus | Cheez-It Citrus | Orlando | Jan 1 | SEC #1 | Big Ten #1 | — | high |
| ReliaQuest | ReliaQuest | Tampa | Dec 31 | SEC pool | Big Ten | — | high |
| Gator | TaxSlayer Gator | Jacksonville | Dec 30 | SEC pool | ACC | vs ACC all 3 years | high |
| Music City | Liberty Mutual Music City | Nashville | Dec 30 | SEC pool | Big Ten | vs B1G all 3 years | high |
| Texas | Kinder's Texas | Houston | Dec 31 | SEC pool | Big 12 #3 | LSU here 2024 AND 2025 (repeat) | high |
| Liberty | AutoZone Liberty | Memphis | Dec/Jan | Big 12 #4 | SEC pool, AAC secondary | actual: AAC 2 of 3 (Memphis 2023, Navy 2025), SEC 1 of 3 | high |
| Las Vegas | Las Vegas | Las Vegas | Dec 31 | SEC (even seasons) / Big Ten (odd seasons) | Legacy Pac-12 group | 2023 Northwestern v Utah; 2024 Texas A&M v USC (legacy); 2025 Nebraska v Utah. 2026 even -> SEC v legacy | high |
| Duke's Mayo | Duke's Mayo | Charlotte | Dec 26 | ACC | SEC (odd) / Big Ten (even) | 2023 UNC v West Virginia (SEC short of teams; Big 12 backfill); 2024 Virginia Tech v Minnesota; 2025 Wake Forest v Mississippi State. 2026 even -> Big Ten | high |
| Pop-Tarts | Pop-Tarts | Orlando | late Dec | Big 12 #2 | ACC | — | high |
| Alamo | Valero Alamo | San Antonio | Dec | Big 12 #1 (12-team pool) | Legacy Pac-12 group | 2024 BYU v Colorado (both Big 12; Colorado via legacy slot) | high |
| Rate / Cactus | Guaranteed Rate (name unresolved) | Phoenix | Dec 26 | Big 12 #5 | Big Ten | MWC backup; 2025 Minnesota v New Mexico (no Big 12: opt-outs) | medium |
| Independence | Radiance Technologies Independence | Shreveport | Dec | CUSA anchor | Big 12 #6 / Legacy Pac-12 / Sun Belt fallback | actual: 2023 Texas Tech v Cal; 2024 Army v LA Tech (5-7 repl.); 2025 LA Tech v Coastal | low-medium |
| Pinstripe | Bad Boy Mowers Pinstripe | Bronx | Dec 26 | Big Ten | ACC | — | high |
| Holiday | Trust & Will Holiday | San Diego | Dec | ACC | Legacy Pac-12 group | Big Ten tie ended 2019; 2024 Syracuse v Wash St; 2025 SMU v Arizona | high (A) |
| Sun | Tony the Tiger Sun | El Paso | Dec 31 | ACC | Legacy Pac-12 group | — | high |
| Poinsettia | SDCCU Poinsettia (revived) | San Diego | Dec 23 | Pac-12 champion (if not CFP) | Legacy 10-team sub-pool | new 2026-27 | high |
| Military | Go Bowling Military | Annapolis | Dec | ACC | AAC (annual) | — | high |
| Fenway | Wasabi Fenway | Boston | Dec 26 | ACC (or Notre Dame) | AAC (annual) | 2025 Army v UConn (independent) -> ACC side can be filled by independents/pool | medium-high |
| Gasparilla | Union Home Mortgage Gasparilla | Tampa | Dec 18 | AAC (pool) | rotating ACC/SEC/Big 12; SEC residual via ESPN Events | 2023 GT v UCF; 2024 Florida v Tulane; 2025 NC St v Memphis | high |
| Birmingham | JLab Birmingham | Birmingham | Dec | SEC residual (ESPN Events) | ACC / AAC pool | 2023 Duke v Troy; 2024 Vandy v GT; 2025 GaSo v App St (opt-out backfill, 5-7 pool) | medium |
| Armed Forces | Lockheed Martin Armed Forces | Fort Worth | Dec 23 | AAC (annual; service academies when available) | open pool (MWC/Air Force, else Sun Belt/SEC) | 2023 Air Force v JMU; 2024 Oklahoma v Navy; 2025 Texas St v Rice | low |
| First Responder | SERVPRO First Responder | Dallas | Dec/Jan | AAC (pool) | CUSA / Sun Belt (pool) | published ACC/Big 12/CUSA framing contradicted by all 3 actual games (AAC v CUSA/SBC) | medium |
| Frisco | Scooter's Coffee Frisco | Frisco | Dec 23 | G6 operator pool | G6 operator pool (ACC access) | 2023 UTSA v Marshall; 2024 Memphis v West Virginia; 2025 UNLV v Ohio. The ACC's 2026 release also lists a "Frisco Football Classic" on Dec 15 — same date as the Xbox Bowl; likely the same game under another name (conflict) | medium |
| Xbox | Xbox | Frisco | ~Dec 15 | CUSA | Sun Belt | 2025 Missouri St v Arkansas St | high |
| New Orleans | R+L Carriers New Orleans | New Orleans | Dec | Sun Belt #2 | CUSA (standing) | 2023 Louisiana v Jax St; 2024 GaSo v Sam Houston; 2025 Southern Miss v WKU | high |
| 68 Ventures | 68 Ventures | Mobile | Dec 26 | Sun Belt #5 | MAC / CUSA | 2023 S Alabama v E Michigan; 2024 Ark St v Bowling Green; 2025 Louisiana v Delaware (CUSA) | high |
| Myrtle Beach | Myrtle Beach | Myrtle Beach | Dec | G6 operator pool (SBC/CUSA/MAC/AAC) | same | 2023 GaSo v Ohio; 2024 Coastal v UTSA; 2025 Kennesaw St (CUSA) v W Michigan | medium |
| Cure | StaffDNA Cure | Orlando | Dec | G6 operator pool | same | 2023 App St v Miami OH; 2024 Ohio v Jax St; 2025 ODU v USF | medium |
| Boca Raton | Bush's Boca Raton | Boca Raton | Dec 18 | G6 operator pool (AAC/SBC/MAC) | same, ACC access | 2025 Toledo v Louisville (ACC) | medium |
| Salute to Veterans | IS4S Salute to Veterans | Montgomery | Dec 15 | Sun Belt | MAC / CUSA | 2024 S Alabama v W Michigan; 2025 Troy v Jax St | medium-high |
| New Mexico | Isleta New Mexico | Albuquerque | Dec 24 | Mountain West | CUSA / AAC (pool) | 2025 North Texas (AAC) v SDSU | medium-high |
| Hawai'i | Sheraton Hawai'i | Honolulu | Dec 24 | Mountain West (Hawai'i home) | AAC annual / CUSA / other | 2025 Hawai'i v Cal (ACC) | low-medium |
| Idaho Potato | Famous Idaho Potato | Boise | Dec 21 | MAC | Mountain West (+ Pac-12 overflow) | 2025 Utah St v Wash St | high |
| Arizona | Snoop Dogg Arizona | Tucson | Dec | MAC | Mountain West | 2023 Toledo v Wyoming; 2025 Fresno St v Miami OH | high |
| Puerto Rico | Puerto Rico (new) | Bayamón | Dec 22 | MAC | G6 operator pool | new 2026-27 | medium-high |

## 3. Actual assignments captured (evidence for tier inference; incomplete)
- SEC 2023: Rose Alabama; Orange Georgia; Cotton Missouri; Peach Ole Miss; Citrus Tennessee; Texas
  Texas A&M; Gator Kentucky; Music City Auburn; ReliaQuest LSU (9 teams).
  2024: CFP Georgia, Texas, Tennessee; Citrus South Carolina; ReliaQuest Alabama; Music City Missouri;
  Las Vegas Texas A&M; Liberty Arkansas; Texas LSU; Gator Ole Miss; Birmingham Vanderbilt; Gasparilla
  Florida; Armed Forces Oklahoma (v Navy; outside the SEC chain) (13).
  2025: CFP Georgia, Ole Miss, Texas A&M, Oklahoma, Alabama; Citrus Texas; Texas LSU; Gator Missouri;
  Music City Tennessee; ReliaQuest Vanderbilt; Duke's Mayo Mississippi State (11).
  Observed: Citrus always the best non-CFP team; Pool of Six shows no stable hierarchy (ReliaQuest got
  LSU 10-3 / Alabama / Vanderbilt in successive years).
- Big 12 2023: Alamo Oklahoma; Pop-Tarts Kansas State; Texas Oklahoma State; Liberty Iowa State (v
  Memphis); Rate Kansas (v UNLV); Independence Texas Tech (v Cal).
  2024: Alamo BYU v Colorado; Pop-Tarts Iowa State (#18) v Miami; Texas Baylor v LSU; Liberty Texas Tech
  v Arkansas; Rate Kansas State v Rutgers; Independence none (Army v LA Tech).
  2025: Alamo TCU v USC; Pop-Tarts BYU v Georgia Tech; Texas Houston (#21) v LSU; Liberty Cincinnati v
  Navy; Rate none; Independence none (K-State/Iowa State declined).
  Observed: consistent with the published order; no repeat bowls within 3 years.
- AAC 2025 (record 9): CFP Tulane; Cure USF; Fenway Army v UConn; New Mexico North Texas; Armed Forces
  Rice (5-7) v Texas St; Liberty Navy; Gasparilla Memphis; First Responder UTSA v FIU; 9th unknown.
- CUSA 2025 (7): Myrtle Beach Kennesaw St; First Responder FIU; Independence LA Tech; Salute to
  Veterans Jax St; New Orleans WKU; Xbox Missouri St; 68 Ventures Delaware. 5 of 7 opponents Sun Belt.
- Sun Belt 2023 (12): Myrtle Beach GaSo; New Orleans Louisiana; Cure App St; Frisco Marshall; Birmingham
  Troy; Camellia Ark St; Armed Forces JMU; 68 Ventures S Alabama; Hawai'i Coastal; Idaho Potato Georgia
  St; ODU, Texas St (bowls uncertain).
  2024 (8): Salute to Veterans S Alabama; New Orleans GaSo; Myrtle Beach Coastal; 68 Ventures Ark St; 4
  not captured. 2025 (10): CFP JMU; Salute to Veterans Troy; Cure ODU; 68 Ventures Louisiana; Xbox Ark
  St; Myrtle Beach Kennesaw? (CUSA — conflict; likely Sun Belt side was a different team); New Orleans
  Southern Miss; Birmingham GaSo (v App St); 2 not captured.
- MAC 2023 (6): Myrtle Beach Ohio; Cure Miami OH; Camellia NIU; 68 Ventures E Michigan; Quick Lane
  Bowling Green; Arizona Toledo. 2024 (7): Salute to Veterans W Michigan; Cure Ohio; Idaho Potato NIU;
  GameAbove Toledo; 68 Ventures Bowling Green; Arizona Miami OH; Bahamas Buffalo. 2025 (5): Myrtle
  Beach W Michigan; Boca Raton Toledo; Frisco Ohio; GameAbove Central Michigan; Arizona Miami OH.
- Big Ten 2023: CFP Michigan; Cotton Ohio State; Peach Penn State; Citrus Iowa; ReliaQuest Wisconsin;
  Music City Maryland; Pinstripe Rutgers; Las Vegas Northwestern (no B1G in Duke's Mayo/Holiday/Rate).
  2024: CFP Ohio State, Oregon, Penn State, Indiana; Citrus Illinois; Duke's Mayo Minnesota; Pinstripe
  Nebraska; Las Vegas USC (legacy side); Rate Rutgers; ReliaQuest/Music City teams unresolved.
  2025: CFP Indiana, Oregon, Ohio State; Citrus Michigan (#18); ReliaQuest Iowa (#23); Music City
  Illinois; Pinstripe Penn State; Las Vegas Nebraska; Rate Minnesota. Observed order tracks strength.
- ACC 2023 (11): Orange Florida State; Pop-Tarts NC State; Gator Clemson; Duke's Mayo North Carolina
  (v West Virginia); Holiday Louisville (v USC); Fenway Boston College (v SMU); Pinstripe Miami;
  Gasparilla Georgia Tech; Boca Raton Syracuse; Birmingham Duke (v Troy); Virginia Tech Military? (v
  Tulane — bowl name per ACC agent conflicts; Military Bowl 2023 was VT v Tulane).
  2024 (13): CFP Clemson, SMU; Pop-Tarts Miami; Pinstripe Boston College; Birmingham Georgia Tech;
  Military NC State; Fenway North Carolina (v UConn); Holiday Syracuse; Sun Louisville (v Washington);
  LA Cal (v UNLV); GameAbove Pitt; Duke's Mayo Virginia Tech; Duke, Virginia unresolved.
  2025 (11 + CFP): CFP Miami (QF), Duke (ACC champion, 5 losses); Gasparilla NC State; Boca Raton
  Louisville; Hawai'i Cal; Military Pitt; Gator Virginia (#19); Duke's Mayo Wake Forest; Pop-Tarts
  Georgia Tech (#22); Pinstripe Clemson (v Penn State); Holiday SMU (v Arizona). Repeat: Virginia Tech
  in Duke's Mayo 2023 and 2024 (consecutive).
- MWC / Pac-12 / legacy: 2023 LA UCLA v Boise; Holiday USC v Louisville; Alamo Arizona v Oklahoma; Sun
  Oregon St v Notre Dame. 2024 Holiday Wash St v Syracuse; LA Boise v Washington; Alamo Colorado
  (legacy slot). 2025 LA Boise v Washington; New Mexico SDSU; Idaho Potato Utah St v Wash St; Arizona
  Fresno St; Alamo USC (legacy) v TCU; Holiday Arizona (legacy) v SMU; Hawai'i Hawai'i v Cal.

## 4. Corrections applied vs v1
Cactus = Rate (one bowl). Duke's Mayo 2026 is not SEC (parity). Las Vegas has an SEC side in even
seasons. Liberty's SEC side is weaker than AAC in practice. Independence's anchor is CUSA, not Big
12. First Responder is AAC vs CUSA/SBC in practice. Holiday side A is ACC (not Big Ten/ACC).
Fenway's ACC side can be an independent. Myrtle Beach / Cure / Boca Raton / Frisco are ESPN
operator-pool bowls, not Sun Belt numbered picks. 68 Ventures' second side can be CUSA. New Mexico's
second side can be AAC. GameAbove/Detroit cancelled; NIU -> MWC; UMass -> MAC.

## 5. Still unresolved (needs a primary text or a fresh search session)
ACC tier count (2 vs 3) and membership; Big Ten official order (inferred only); legacy-pool
carve-out vs Cal/Stanford "dual eligibility" (two sources disagree; encode as primary group =
legacy, secondary = ACC, stand-in); legacy pool sunsets after the 2026 season (pressdemocrat) ->
season_max 2026; Notre Dame's
ACC access rule; Hawai'i second side; Armed Forces pool rule; Sun Belt pick 6; MAC Arizona vs Idaho
Potato order; legacy-pool procedural carve-out (second source); Rate/Cactus 2026 name; AAC 9th
team 2025; 2023-24 and 2024-25 AAC/CUSA/MWC full slates; records/ranks for most rows; NCAA
FCS-win counting text; APR replacement order text; 35 vs 36.
