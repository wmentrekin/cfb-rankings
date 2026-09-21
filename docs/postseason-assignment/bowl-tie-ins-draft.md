# 2026-27 non-CFP bowl tie-ins — DRAFT seed data for `postseason/bowl_tie_ins.json` (T3)

Source: platform-researcher-3 (search snippets only; no page read in full). Every row is
snippet-corroborated. Confidence: high = 2+ agreeing sources incl. a conference/bowl announcement;
medium = one solid source; low = conflicting or carry-over inference. Rows marked low MUST be
encoded as `provisional` until the owner supplies a primary text. Correction vs earlier notes:
Cactus Bowl and Guaranteed Rate / Rate Bowl are ONE bowl (Phoenix/Tempe), not two.
Count: 35 non-CFP FBS bowls (best supported; one snippet says 36).

| Root | 2026 name | City / venue | Date | Slot A | Slot B | Pick / tier | Either-or / notes | Conf. |
|---|---|---|---|---|---|---|---|---|
| Gasparilla | Union Home Mortgage Gasparilla | Tampa, Raymond James | Dec 18 | AAC | ACC or SEC (3-way pool) | — | soft rotation AAC/ACC/SEC | high |
| Frisco | Scooter's Coffee Frisco | Frisco, Ford Center | Dec 23 | G5 pool (AAC-anchored) | G5 pool | — | — | medium |
| Independence | Radiance Technologies Independence | Shreveport | Dec 22 | Big 12 | CUSA | — | CONFLICT: AAC lineup calls it a guaranteed AAC tie-in | low |
| Pop-Tarts | Pop-Tarts | Orlando | late Dec | ACC | Big 12 | Big 12 #2 (projected) | — | high |
| Xbox | Xbox | Frisco, Ford Center | ~Dec 15 | CUSA | Sun Belt | — | replaced Bahamas in 2025-26; this is the "second Frisco game" | high |
| Rate / Cactus | Guaranteed Rate (2026 branding unresolved) | Phoenix/Tempe | Dec | Big 12 | Big Ten | Big 12 #5 (projected) | name/venue for 2026 unresolved | medium (tie-in) / low (name) |
| Duke's Mayo | Duke's Mayo | Charlotte | Dec 26 | ACC | SEC | — | historically alternates SEC/Big Ten; 2026 = SEC | high |
| Myrtle Beach | Myrtle Beach | Myrtle Beach | Dec | Sun Belt | AAC / G5 pool | SBC pick # unknown | — | medium |
| Music City | Music City | Nashville, Nissan | Dec 30 | Big Ten | SEC (pool of six) | — | — | high |
| New Mexico | New Mexico | Albuquerque | Dec 24 | CUSA | Mountain West | — | — | high |
| Armed Forces | Lockheed Martin Armed Forces | Fort Worth | Dec | AAC / MWC (contractual, service-academy anchor) | pool fill | — | 2025-26 actual was Sun Belt vs AAC; contested | low |
| First Responder | SERVPRO First Responder | Dallas, Ford Stadium | Dec | rotates AAC/ACC/Big 12/CUSA | same pool | — | one snippet says "Group of Six" for this cycle | low |
| Las Vegas | Las Vegas | Las Vegas | Dec | Big Ten | Pac-12 legacy pool | — | also named in SEC "pool of six" (contested) | high |
| New Orleans | R+L Carriers New Orleans | New Orleans | Dec | CUSA | Sun Belt | Sun Belt #2 | — | high |
| Boca Raton | Bush's Boca Raton | Boca Raton, FAU | Dec 18 | AAC | G5 pool | SBC pick # unknown | — | medium |
| 68 Ventures | 68 Ventures | Mobile | Dec 26 | Sun Belt | MAC | Sun Belt #5 | — | high |
| Liberty | AutoZone Liberty | Memphis | Dec | Big 12 | SEC pool (AAC alternate) | Big 12 #4 | — | high |
| Pinstripe | Bad Boy Mowers Pinstripe | Bronx, Yankee Stadium | Dec 26 | Big Ten | ACC | — | — | high |
| Citrus | Cheez-It Citrus | Orlando | Jan 1 2027 | SEC | Big Ten | SEC #1, Big Ten #1 | — | high |
| Idaho Potato | Famous Idaho Potato | Boise | Dec 21 | MAC | Mountain West | order vs Arizona unresolved | — | high / low (order) |
| Military | Go Bowling Military | Annapolis | Dec | ACC | AAC | — | — | high |
| Salute to Veterans | IS4S Salute to Veterans | Montgomery | Dec 15 | Sun Belt | MAC or CUSA pool | — | — | medium |
| Birmingham | JLab Birmingham | Birmingham | Dec | SEC | AAC (ACC named as 3rd possible) | — | 3-way ambiguity | medium |
| Texas | Kinder's Texas | Houston, NRG | Dec 31 | Big 12 | SEC pool | Big 12 #3 (projected) | — | high |
| ReliaQuest | ReliaQuest | Tampa, Raymond James | Dec 31 | Big Ten | SEC pool | — | — | high |
| Hawai'i | Sheraton Hawai'i | Honolulu | Dec 24 | rotates AAC/CUSA/MWC | same pool | — | — | medium-high |
| Arizona | Snoop Dogg Arizona | Tucson | Dec | MAC | Mountain West | — | — | high |
| Cure | StaffDNA Cure | Orlando | Dec | AAC | Sun Belt | — | — | medium |
| Gator | TaxSlayer Gator | Jacksonville | Dec 30 | ACC | SEC pool | — | — | high |
| Sun | Tony the Tiger Sun | El Paso | Dec | ACC | Pac-12 legacy pool | — | — | high |
| Holiday | Trust & Will Holiday | San Diego | Dec | Big Ten or ACC (alternation, 2026 side unconfirmed) | Pac-12 legacy pool | — | — | medium / low (side A) |
| Alamo | Valero Alamo | San Antonio, Alamodome | Dec | Big 12 | Pac-12 / legacy pool (12-team list incl. Cal, Oregon St, Wash St) | Big 12 #1 | — | high |
| Fenway | Fenway | Boston, Fenway Park | Dec 26 | ACC | AAC (or Notre Dame) | — | — | high |
| Puerto Rico (NEW) | Puerto Rico | Bayamón | Dec 22 | MAC | G5/at-large pool (AAC/CUSA/SBC) | — | new 2026-27 | medium-high |
| Poinsettia (REVIVED) | SDCCU Poinsettia | San Diego, Snapdragon | Dec 23 | Pac-12 champion (if not in CFP) | Pac-12 legacy pool (10-team list: Ariz, ASU, Cal, Colo, Ore, Stan, UCLA, USC, Utah, Wash) | champion vs pool | new 2026-27 | high |

## Per-conference selection order (as far as published)
- SEC: Citrus first; then "Pool of Six" (Gator, Music City, Liberty, Texas, ReliaQuest, Las Vegas —
  Las Vegas contested) placed by the SEC office (geography, matchups, repeat avoidance). No numbered
  order. -> mode office_pool; stand-in = declaration order.
- Big Ten: Citrus first; remainder (ReliaQuest, Music City, Pinstripe, Duke's Mayo, Rate/Cactus, Las
  Vegas/Holiday) unpublished; aggregator guess Citrus -> ReliaQuest -> Duke's Mayo/Las Vegas ->
  Music City -> Pinstripe -> Cactus. -> mode tiered with inferred order (T5), provisional.
- Big 12: Alamo -> Pop-Tarts -> Texas -> Liberty (#4 confirmed) -> Rate/Cactus -> Independence, per
  the (unfetchable) Big 12 Bowl Selection Central page + aggregators. -> mode numbered, medium.
- ACC: three tiers; criteria "geographic proximity, avoiding repeat appearances and matchups,
  regular-season won-loss records" (theacc.com 2026-06-03). Tier membership unpublished. -> tiered.
- Sun Belt: numbered draft; New Orleans #2, 68 Ventures #5 confirmed; Boca Raton, Cure, Salute to
  Veterans, Myrtle Beach, Frisco/Xbox fill the rest (numbers unknown). -> numbered, partial.
- AAC: no order; access to Independence (contested), Fenway, Military, Armed Forces, Hawai'i +
  Birmingham, Boca Raton, Cure, Frisco, Gasparilla, New Mexico, Myrtle Beach by availability.
- MAC: Arizona, Idaho Potato (order unknown), 68 Ventures, Puerto Rico.
- CUSA: New Mexico, Hawai'i (rotation), New Orleans, Xbox, Independence, Salute to Veterans.
- Mountain West: ~4 slots (Arizona, Idaho Potato, New Mexico, Hawai'i rotation). Least settled.
- Rebuilt Pac-12 / legacy pool: Poinsettia, Alamo, Las Vegas, Holiday, Sun (+ Independence per one
  source). Pool membership differs by bowl (10-team vs 12-team lists). Least settled.

## Unresolved (need a primary text)
Independence pairing; Armed Forces pairing; First Responder pairing; Birmingham third side; Holiday
side A for 2026; legacy-pool membership per bowl; Big Ten order; Sun Belt picks 1/3/4/6; AAC/MAC/
CUSA/MWC orders; 35 vs 36 count.
