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

## 2. Bowl table — 2026-27, THIS YEAR'S evidence (v3, 2026-09-21)
Status: `2026` = confirmed by a 2026-dated source (bowl/conference release, Bowl Season list,
2026-dated Wikipedia page, 2026 local coverage); `carry` = 2025 structure carried over with no
2026-specific statement beyond the blanket carry-over; `open` = a 2026 fact still missing.
Every non-CFP tie-in contract ends after this season (Arizona Bowl executive, tucson.com 2026)
-> all entries `season_max: 2026`.

| Root | 2026 name | City / venue | 2026 date | Slot A | Slot B | 2026 evidence | Status | Note |
|---|---|---|---|---|---|---|---|---|
| Citrus | Cheez-It Citrus | Orlando, Camping World | Jan 1 2027 (Jan 1 vs 2 unpinned) | SEC #1 | Big Ten #1 | Florida Citrus Sports 2026 page; Wikipedia | 2026 | — |
| ReliaQuest | ReliaQuest | Tampa, Raymond James | Dec 31 2026 noon ET ESPN | SEC pool | Big Ten | Wikipedia 2026; Tampa Free Press | 2026 | — |
| Gator | TaxSlayer Gator | Jacksonville, EverBank | Dec 30 2026 | SEC pool | ACC | Wikipedia 2026 (82nd) | 2026 | — |
| Music City | Liberty Mutual Music City | Nashville, Nissan | Dec 30 2026 3pm ET ESPN | SEC pool | Big Ten | Wikipedia 2026 | 2026 | — |
| Texas | Kinder's Texas | Houston, NRG | Dec 31 2026 7:30pm ET ESPN | SEC pool | Big 12 #3 | Wikipedia 2026 (20th) | 2026 | — |
| Liberty | AutoZone Liberty | Memphis, Simmons Bank Liberty | Dec 2026 (date open) | Big 12 #4 | SEC pool primary / AAC secondary | CFN 2026-27 page | 2026 (structure) | which side fills B in 2026 unknowable until December |
| Las Vegas | Las Vegas | Las Vegas, Allegiant | Dec 31 2026 12:45pm PT ESPN | SEC (even seasons) | Legacy Pac-12 group | Wikipedia 2026; news3lv; pressdemocrat 2026-08-16 | 2026 | Big Ten side in odd seasons |
| Duke's Mayo | Duke's Mayo | Charlotte, Bank of America | Mon Dec 28 2026 3:30pm ET ABC (ACC release; Wikipedia's Dec 26 is stale) | ACC | Big Ten (even seasons) | theacc.com 2026-06-03; franchise page rule (B1G even / SEC odd; the other goes to Las Vegas) | 2026 | "2026 Duke's Mayo Bowl" Wikipedia page = the Jan 2 2026 game (2025 season), hence its "SEC" |
| Pop-Tarts | Pop-Tarts | Orlando, Camping World | Dec 29 2026 5:30pm ET ESPN | Big 12 #2 | ACC #1 non-CFP (incl. Notre Dame) | Wikipedia 2026; poptartsbowl.com | 2026 | explicit "top ACC selection vs second Big 12 selection" |
| Alamo | Valero Alamo | San Antonio, Alamodome | Dec 29 2026 8pm CT ESPN | Big 12 #1 (12-team pool) | Legacy Pac-12 12-team pool | Wikipedia 2026; alamobowl.com | 2026 | first-available from each pool by overall record |
| Cactus | Cactus Bowl (rebranded from Rate, 2026-06-03) | Tempe, Mountain America Stadium | Dec 26 2026 5:30pm ABC | Big 12 #5 | Big Ten | azfamily 2026-06-03; statepress 2026-06 | 2026 | MWC backup if B1G/B12 cannot fill |
| Independence | Radiance Technologies Independence | Shreveport | Dec 22 2026 8:30pm ET ESPN | Big 12 #6 (legacy Pac-12 backup) | CUSA anchor | Wikipedia 2026; SI/spokesman Pac-12 2026 | 2026 | primary/backup, not a conflict |
| Pinstripe | Bad Boy Mowers Pinstripe | Bronx, Yankee Stadium | Dec 26 2026 noon ET ABC | Big Ten | ACC | Wikipedia 2026; theacc 2026-06-03 | 2026 | — |
| Holiday | Trust & Will Holiday (47th) | San Diego, Snapdragon | Dec 28 2026 2pm PT FOX | ACC | Legacy Pac-12 group | holidaybowl.com 2026; kvia 2026-04-20 | 2026 | no Big Ten access |
| Sun | Tony the Tiger Sun (Old El Paso sponsor per bowlseason.com 2026-09-17) | El Paso | Dec 31 2026 noon MT CBS | ACC | Legacy Pac-12 group | sunbowl.org; El Paso Times 2026-04-29; aol 2026 | 2026 | "traditionally the last Pac-12 bowl to select" |
| Poinsettia | SDCCU Poinsettia (revived) | San Diego, Snapdragon | Dec 23 2026 5:30pm MT | Pac-12 champion (if not CFP) | Legacy 10-team sub-pool (excl. OSU/WSU) | SI 2026; on3; mwcconnection 2026 | 2026 | — |
| Military | Freedom Mortgage Military | Annapolis | Dec 28 2026 2pm ET ESPN | ACC | AAC annual | theacc 2026-06-03 | 2026 | — |
| Fenway | Wasabi Fenway | Boston, Fenway Park | Dec 26 2026 2pm ET ESPN | ACC (or Notre Dame, "close in record") | AAC annual | theacc 2026-06-03; Wikipedia 2026 | 2026 | ND numeric threshold only in older text |
| Gasparilla | Union Home Mortgage Gasparilla | Tampa, Raymond James | Dec 18 2026 | AAC pool | rotating ACC/SEC (SEC residual via ESPN Events) | Wikipedia 2026; FanSided 2026 | 2026 | — |
| Birmingham | JLab Birmingham (20th) | Birmingham, Protective | Dec 29 2026 | ACC/SEC/American three-way pool | same | Wikipedia 2026; wbrc 2026-06-04; FanSided | 2026 | SEC only if a team remains after Pool of Six |
| Armed Forces | Lockheed Martin Armed Forces | Fort Worth, Amon G. Carter (venue carry) | Dec 23 2026 | American annual | CUSA (2026 list) | Wikipedia 2026; FanSided 2026 | 2026 | actual partners varied 2023-25 |
| First Responder | SERVPRO First Responder | Dallas, Ford Stadium | Sat Jan 2 2027 | American pool | ACC / Big 12 / CUSA pool | firstresponderbowl.com 2026-06-03; FanSided | 2026 | actuals 2023-25 all AAC v CUSA/SBC |
| Frisco Bowl | Frisco Bowl (9th) | Frisco, Ford Center | Dec 23 2026 8pm CT | G6 operator pool | G6 pool (ACC access) | thefriscobowl.com 2026 | 2026 | — |
| Frisco Football Classic | Frisco Football Classic (= 2025 Xbox Bowl, renamed) | Frisco, Ford Center | Dec 15 2026 9pm ET | CUSA / Sun Belt pool (inherited) | same; ACC lists it as a destination | InForum 2026; theacc 2026-06-03; Wikipedia 2026-27 | 2026 | not a third Frisco game; bowl_names root must map both names |
| New Orleans | R+L Carriers New Orleans (26th) | New Orleans, Caesars Superdome | Dec 23 2026 1pm CT | Sun Belt #2 | CUSA | crescentcitysports 2026 | 2026 | — |
| 68 Ventures | 68 Ventures (28th) | Mobile, Hancock Whitney | Dec 26 2026 5:30pm ET | Sun Belt #5 | MAC | Wikipedia 2026 | 2026 | 2025 CUSA opponent was a one-off |
| Myrtle Beach | Myrtle Beach (7th) | Conway SC, Brooks Stadium | Dec 21 2026 11am ET | G6 pool: any two of CUSA/MAC/Sun Belt | same | Wikipedia 2026; coastal.edu 2026-06-03 | 2026 | — |
| Cure | Cure Bowl | Orlando, Exploria | Dec 22 2026 | G6 operator pool | ACC access | Wikipedia 2026 | 2026 (carry structure) | — |
| Boca Raton | Bush's Boca Raton | Boca Raton, FAU Flagler CU Stadium | Dec 18 2026 11am ET ESPN | G6 pool (AAC/CUSA/MAC/SBC) | + MWC, select independents | bocaratontribune / tapinto 2026-06 | 2026 | — |
| Salute to Veterans | IS4S Salute to Veterans | Montgomery, Cramton Bowl | Dec 15 2026 5:30pm ET ESPN | flat SBC/MAC/CUSA pool | same | wsfa etc 2026-06-03 | 2026 (date); anchor open | draft's "Sun Belt anchor" unconfirmed for 2026 |
| New Mexico | Isleta New Mexico | Albuquerque | Dec 24 2026 11:30am MT ESPN | Mountain West | CUSA (2026 sources name CUSA; wider pool unconfirmed) | Wikipedia 2026; CFN 2026-27 | 2026 | 2025 actual was AAC |
| Hawai'i | Sheraton Hawai'i | Honolulu, Ching Athletics Complex | Dec 24 2026 1pm HST ESPN | Mountain West (Hawai'i home; full MWC member from 2026-07-01) | AAC / CUSA pool | Wikipedia 2026; hawaii.edu 2026-06-03 | 2026 | — |
| Idaho Potato | Famous Idaho Potato | Boise, Albertsons | Dec 21 2026 2:30pm MT ESPN | MAC | Mountain West | Wikipedia 2026 | 2026 | — |
| Arizona | Arizona Bowl | Tucson, Arizona Stadium | date TBD | MAC | Mountain West | Wikipedia 2026; tucson.com 2026 | 2026 (tie-ins); date open | contract ends after this season |
| Puerto Rico | Puerto Rico Bowl (new) | Bayamón, Juan Ramón Loubriel | Dec 22 2026 1:30pm ET ESPN | MAC anchor | G6 at-large — second side NOT YET ANNOUNCED | getsomemaction 2026-05-08; CFN | 2026 (MAC side); B open | provisional at_large candidate |

Removed vs v2: "Xbox" (renamed Frisco Football Classic), "Rate" (renamed Cactus). Count stays 35.
Summary: 35 rows; 32 have both sides 2026-confirmed; 3 have an open side (Liberty B, Salute to
Veterans anchor, Puerto Rico B); 3 have an open date (Liberty, Arizona, Citrus day).

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
(after the 2026-confirmation pass) Liberty 2026 slot B and date; Salute to Veterans anchor; Puerto
Rico opponent side (not yet announced by anyone); Arizona Bowl date; Citrus Jan 1 vs 2; New Mexico
wider pool; ND numeric threshold in a 2026 text;
ACC tier count (2 vs 3) and membership; Big Ten official order (inferred only); legacy-pool
carve-out vs Cal/Stanford "dual eligibility" (two sources disagree; encode as primary group =
legacy, secondary = ACC, stand-in); legacy pool sunsets after the 2026 season (pressdemocrat) ->
season_max 2026; Notre Dame's
ACC access rule; Hawai'i second side; Armed Forces pool rule; Sun Belt pick 6; MAC Arizona vs Idaho
Potato order; legacy-pool procedural carve-out (second source); Rate/Cactus 2026 name; AAC 9th
team 2025; 2023-24 and 2024-25 AAC/CUSA/MWC full slates; records/ranks for most rows; NCAA
FCS-win counting text; APR replacement order text; 35 vs 36.
