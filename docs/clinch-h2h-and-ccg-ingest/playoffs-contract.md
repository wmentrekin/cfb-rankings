# Playoffs artifact + Season Grid contract (v1, 2026-11-23)

Shared by both repos for this pass. cfb-rankings PRODUCES it; personal-site CONSUMES it. Ephemeral
`$work` scratch: the durable statement lives in code/tests on each side.

## Owner rules (verbatim intent)
- Applies to seasons 2025 and 2026 ONLY (2027 moves to a 24-team format). Any other season: no
  playoffs artifact is built or published, and the site offers only the seasons the index lists.
- 2026 = PROJECTION from our rankings, "if the season ended today":
  1. champions of ACC, Big Ten, Big 12, SEC, regardless of rank;
  2. the highest-ranked Group of Six TEAM (American Athletic, Conference USA, Mid-American,
     Mountain West, Pac-12, Sun Belt), champion or not;
  3. Notre Dame, if ranked in the top 12;
  4. remaining spots to the next highest-ranked teams, to 12 total;
  5. seeds 1-12 strictly by rank; seeds 1-4 get byes.
  Undecided champion: an actual champion (title game played) wins outright. Otherwise the team
  first in that conference's standings; when several teams are tied on conference record at the
  top, the one ranked highest by the model. Once a title game is set (identified in the stored games, played or not, else two teams clinched), the
  same rule applies among those two (ranked ones only).
  After selection day (stored CFP rows identify all 12 seeds) the 2026 file switches to `kind: "actual"`.
- 2025 = the ACTUAL bracket and results from stored postseason games (seeds, bowl names, scores
  are populated in `games`; verified 2026-11-23: 11 games, FR1-4, QF1-4, SF1-2, CH).
- Season Grid: no CFP-specific columns. The postseason column is "Bowls"; a playoff team's Bowls
  cell says just "CFP".

## R2 keys
`playoffs/{season}/latest.json`, `playoffs/index.json` = `{generated_at_utc, latest_season, seasons: [2026, 2025]}`
(same shape as `schedule/index.json`). Site copies to `public/data/cfb/playoffs/`.

## `playoffs/{season}/latest.json`
```jsonc
{
  "season": 2026,
  "kind": "projected",            // "projected" | "actual"
  "as_of_week": 12,               // rankings week the projection uses; null for "actual"
  "generated_at_utc": "…",
  "teams": [                      // exactly 12, ordered by seed
    {
      "seed": 1,
      "team": "Utah",
      "rank": 1,                  // our model rank; for "actual" the final model rank or null
      "record": "11-0",
      "conference": "BIG 12",     // DISPLAY name, same vocabulary as the rankings artifact
      "logo_url": "https://…",   // same resolution as the schedule artifact; may be null
      "bid": "champion",          // "champion" | "g6" | "notre_dame" | "at_large"
      "bid_label": "Projected Big 12 Champion",  // human text, Title Case; e.g. "ACC Champion",
                                  // "Highest-Ranked Group of Six", "Top-12 Guarantee", "At-Large"
      "bye": true
    }
  ],
  "rounds": [                     // always these four, in this order
    {"round": "first_round",  "label": "First Round",            "dates_label": "Dec 18-19", "note": "Campus Sites"},
    {"round": "quarterfinal", "label": "Quarterfinals",          "dates_label": "Dec 30 - Jan 1", "note": null},
    {"round": "semifinal",    "label": "Semifinals",             "dates_label": "Jan 14-15", "note": null},
    {"round": "championship", "label": "National Championship",  "dates_label": "Jan 25", "note": "Las Vegas"}
  ],
  "games": [                      // exactly 11
    {
      "slot": "FR1",              // unique id within the payload
      "round": "first_round",
      "order": 0,                 // vertical position within its round, top to bottom, so that
                                  // game k of round r+1 is fed by games 2k and 2k+1 of round r
                                  // (first round: order 0..3 feeds QF order 0..3 one-to-one)
      "name": null,               // bowl/game display name, e.g. "Rose Bowl"; null when unknown
      "date": null,               // "YYYY-MM-DD" or null
      "status": "projected",      // "projected" | "scheduled" | "final"
      "top":    {"seed": 8,  "team": "…", "score": null, "from": null},
      "bottom": {"seed": 9,  "team": "…", "score": null, "from": null},
      "winner": null              // "top" | "bottom" | null
    }
    // A side whose team is not yet known has team: null, seed: null and "from": "<slot>" naming
    // the game whose winner fills it. A side that IS known may still carry "from" (actual brackets).
    // Quarterfinals: "top" is the bye seed, "bottom" comes from the first-round game.
  ],
  "first_out": [                  // projected only; next teams by rank outside the field (0-4); [] for actual
    {"team": "…", "rank": 13, "record": "9-2", "conference": "SEC", "logo_url": "…"}
  ]
}
```
Bracket geometry (both seasons; matches the real 2025 bracket): first round 8v9, 5v12, 7v10, 6v11
at the higher seed. Quarterfinal order 0: seed 1 v winner 8/9; order 1: seed 4 v winner 5/12;
order 2: seed 2 v winner 7/10; order 3: seed 3 v winner 6/11. Semifinal order 0 = QF 0 + QF 1;
order 1 = QF 2 + QF 3. First-round `order` matches the quarterfinal it feeds.
Team display data for a game side is looked up in `teams` by `team`.
Side placement: first round and quarterfinals put the HIGHER seed on top. Semifinals and the
championship put the side fed by the earlier feeder game (order 2k) on top and the later one
(2k+1) on the bottom, regardless of seed, so connector lines never cross. The site renders `top`
and `bottom` exactly as given and must not re-sort sides by seed.

## Season Grid payload change (`schedule/{season}/latest.json`)
- `weeks` no longer contains `cfp-quarterfinals`, `cfp-semifinals`, `cfp-national-championship`.
  The last slot stays `cfp-r1-bowls` with label "Bowls".
- A playoff team (any CFP-bracket game, or a bye) gets ONE Bowls cell:
  `{"slot_id": "cfp-r1-bowls", "label": "Bowls", "status": "cfp", "playoff_round": "CFP",
    "game_name": "College Football Playoff", "game_name_short": "CFP", "cfp_seed": <int|null>}`
  with no opponent, score, or win/loss status. Non-playoff teams are unchanged (bowl game cell or
  the eligible/ineligible/possible placeholder).
- Site renders `status == "cfp"` as the plain text "CFP" (muted placeholder style, same height as
  the other one-line placeholder cells). The site must also render sensibly against an OLD payload
  (extra columns simply appear) and the old site against a NEW payload (unknown status falls back
  to its generic text) -- deploy order is pipeline first, then site.
