# Postseason assignment engine — selection logic specification (draft v0.2, 2026-09-21)

Purpose: a rule set precise enough that two implementers produce the same output from the same
input. Every rule below is either a published rule (cited in research-notes.md /
bowl-tie-ins-draft.md), a config value, or a NAMED STAND-IN for a human decision. Stand-ins are
tagged in the output. Open decisions for the owner are collected in §9.

## 0. Conventions
- Deterministic: output = f(SeasonEndState, config). No randomness; no dependence on dict or file
  iteration order except where "declaration order" is explicitly the rule.
- Every ordering ends with `team name (ASCII ascending)` as the final tiebreak, so no two teams
  ever compare equal.
- `rank(T)` = position of T in `state.ranking` (strict, 1..N over all FBS teams).
- `standing(C)` = `state.conference_order[C]` = the Season Grid's order for conference C
  (artifacts/schedule.py::_sort_conference_teams; champions/CCG losers pinned first).
- `season` = the season year (2026 for the 2026-27 postseason). Parity rules use `season % 2`.
- `H` = history window in seasons (config `penalties.history_seasons`, default 3: the three
  seasons before `season`).
- `wins(T)`, `losses(T)` = overall record; `eligible(T)` = `state.teams[T].bowl_status ==
  "eligible"` (6 wins under the existing rule; see §8 for the eligibility hook).

## 1. Inputs (SeasonEndState)
| field | type | source |
|---|---|---|
| season, as_of_week | int, int|None | pipeline |
| teams | {team: {conference_raw, is_independent, wins, losses, conf_wins, conf_losses, bowl_status}} | adapter over compute_standings + teams table |
| ranking | [{rank, team}] | rankings artifact / DB query; strict order |
| champions | {conference_raw: {team, status: actual|projected}} | _resolve_conference_champions; else standing(C)[0] tagged projected |
| conference_order | {conference_raw: [team...]} | _sort_conference_teams per conference |
| history | {team: [{season, bowl_root, opponent}]} | games rows, season_type=postseason, last H seasons |
| regular_season_opponents | {team: set(team)} | games rows this season incl. CCG |
| declined | set(team) | manual input / overrides file (teams that opted out of a bowl) |

Validation on entry: every team in `ranking` exists in `teams`; every champion is a member of its
conference this season; `conference_order[C]` is a permutation of C's members.

## 2. CFP selection — `select_cfp_field(state, cfp_rules)`
`cfp_rules` is the entry whose `season_min <= season <= season_max`.

```
field = []                      # ordered list of {team, path, reasons}
def admit(team, path, reason): if team not in field: field.append(...)

# Step 1 — automatic qualifiers
if rules.aq_mode == "top_n_champions":                       # 2024, 2025
    champs = [c.team for c in state.champions.values()]
    for t in sorted(champs, key=rank)[:rules.aq_count]:
        admit(t, "aq_champion", f"among the {rules.aq_count} highest-ranked champions")
elif rules.aq_mode == "named_plus_g6":                       # 2026+
    for conf in rules.aq_conferences:                        # ACC, Big Ten, Big 12, SEC
        c = state.champions.get(conf)                         # missing => error (config/data mismatch)
        admit(c.team, "aq_p4", f"{conf} champion" + (" (projected)" if c.status=="projected" else ""))
    g6_pool = [t for t in teams if conference(t) in rules.g6_conferences]
    if rules.g6_slot_rule == "any_team":
        g6 = min(g6_pool, key=rank)
    else:  # "champion"
        g6 = min([state.champions[c].team for c in rules.g6_conferences if c in state.champions], key=rank)
    admit(g6, "aq_g6", f"highest-ranked Group of Six {'team' if rules.g6_slot_rule=='any_team' else 'champion'}")
    # a G6 team admitted in the P4 loop cannot happen (disjoint sets); a team can never be admitted twice.

# Step 2 — independent guarantee (2026+: Notre Dame, max_rank 12; 13 if field_size == 14; none if > 14)
for g in rules.independent_guarantees:
    if rank(g.team) <= g.max_rank: admit(g.team, "independent_guarantee", f"{g.team} ranked {rank} <= {g.max_rank}")

# Step 3 — at-large
remaining = rules.field_size - len(field)
for t in sorted(all_teams, key=rank):
    if remaining == 0: break
    if t not in field: admit(t, "at_large", f"next highest-ranked, rank {rank(t)}"); remaining -= 1

# Step 4 — bumped list: every team ranked above the LOWEST-ranked admitted team that is not in the field
cutoff = max(rank(t) for t in field)
bumped = [{team, rank, reason} for t not in field if rank(t) < cutoff]
#   reason = "displaced by AQ ranked below it" if any AQ has rank > rank(t) else "displaced by independent guarantee"

# Step 5 — seeding
if rules.seeding == "straight":                              # 2025+
    seeds = sorted(field, key=rank)                          # seed i = i-th by rank
elif rules.seeding == "champions_top4":                      # 2024
    aq = sorted([t for t in field if path(t) in ("aq_champion","aq_p4","aq_g6")], key=rank)
    seeds = aq[:4] + sorted([t for t in field if t not in aq[:4]], key=rank)
byes = seeds[:rules.bye_count]
```
Invariants: `len(field) == field_size`; `field` has no duplicates; every AQ conference has exactly
one representative via its AQ path; seeds are a permutation of field.

Worked cases (from research-notes.md; final ranks as reported):
- 2024 (top_n_champions=5, champions_top4): Clemson (ACC champion, #16) admitted as the 5th
  champion; Alabama (#11, non-champion) is bumped. Seeds 1-4 = Oregon, Georgia, Boise State,
  Arizona State (the 4 highest-ranked champions); Clemson seeded 12th among the rest by rank.
- 2025 (top_n_champions=5, straight): two champions ranked outside the top 12 (reported: Duke and
  the G6 champion) consume AQ slots, so only 7 non-champions get in and Notre Dame (#11) is bumped.
  Byes go to seeds 1-4 by rank regardless of champion status.
- 2026 (named_plus_g6, straight, guarantee ND<=12): the same 2025 state would admit Notre Dame via
  the guarantee and bump the 7th-ranked non-champion instead. Fixture must assert both outcomes
  under their own rule entries.

## 3. Bracket — `build_bracket(cfp_field, cfp_rules)`
- First round: pairs from `rules.first_round_pairings` (default [(5,12),(6,11),(7,10),(8,9)]);
  `home = higher seed`, `site = campus`, dates from `rules.first_round.dates`; day/window per game
  is `null` until Selection Sunday (config may set `first_round_windows` later).
- Quarterfinals: feeds from `rules.quarterfinal_feeds` (default {1: (8,9), 2: (7,10), 3: (6,11),
  4: (5,12)}).
- QF bowl placement (STAND-IN for "seeds 1-3 choose", cfp release 2025-05-22):
```
available = list(rules.quarterfinal_hosts)        # declaration order = fallback order
for seed in 1..3:
    prefs = rules.qf_preferences.get(conference(seed), [])   # e.g. Big Ten: [Rose, Cotton, Fiesta, Peach]
    pick = first(b for b in prefs if b in available) or available[0]
    tag = "qf_conference_preference" if pick in prefs else "qf_hosts_order"
    assign(seed, pick, stand_in=tag); available.remove(pick)
assign(4, available[0], stand_in="qf_remainder")
```
- Semifinals: halves are {QF(seed 1), QF(seed 4)} and {QF(seed 2), QF(seed 3)}. The 1 seed's half
  plays in `rules.semifinal_choice_by_top_seed[QF bowl of seed 1]` if that map exists for the
  season, else `rules.semifinal_hosts[0]`; the other half gets the other semifinal host.
  Tag `sf_by_top_seed_choice` / `sf_hosts_order`.
- Championship: `rules.championship` (site, city, date).

## 4. Bowl assignment — `assign_bowls(state, cfp_field, tie_ins, policy)`

### 4.0 Bowl affiliation is not conference membership
For bowl purposes a team belongs to an AFFILIATION GROUP, which defaults to its conference but
can be overridden per season in `tie_ins.affiliations: {team: group_id}`. 2026-27 evidence
(pressdemocrat.com 2026-08-16; sunbowl.org release): the 12 legacy Pac-12 schools (Arizona,
Arizona State, Cal, Colorado, Oregon, Oregon State, Stanford, UCLA, USC, Utah, Washington,
Washington State) are tied as a group to the Alamo, Las Vegas, Holiday, Sun, Poinsettia, and
Independence bowls, ordered by OVERALL record, and are "not affiliated with the selection
process for their current conferences". So:
- `group(T)` = `affiliations.get(T, conference(T))`; every rule in §4.3-4.7 that says
  "conference C" means "group C". `affiliations[T]` may be `{primary, secondary}`: T is placed by
  its primary group's walk; if unplaced there it becomes available to its secondary group's
  fallback fill (§4.6) — the stand-in for the conflicting "carve-out" vs "dual eligibility" reports
  on Cal/Stanford (tag `secondary_affiliation`).
- A group's `selection.conferences[group]` entry may use `mode: pool_by_record` whose standing
  order is `(wins desc, losses asc, rank asc, name)` instead of a conference standing order.
- CFP selection (§2) is unaffected: it uses real conference membership and champions.
- Overflow from a group with more eligible teams than obligations follows §4.6/§4.8 (the 2026-27
  legacy group's overflow goes to the ESPN pool by availability, per the same source); the
  Cactus/Rate Bowl names the Mountain West as its backup source when Big Ten/Big 12 cannot fill.
- Poinsettia's second slot uses a 10-team sub-pool that EXCLUDES Oregon State and Washington
  State (they are the current-Pac-12 side); encode as a separate `pools` entry.
Owner prerequisite P2 should include a second procedural source confirming the carve-out.

### 4.1 Slate for the season
`bowls = [b for b in tie_ins.bowls if b.season_min <= season <= b.season_max and not b.cfp_host]`,
kept in DECLARATION ORDER (this is the only place file order is semantic; see §4.4).
Validation: no two bowls share a `bowl_root` for the season; every referenced conference is a
known raw conference for the season; every pool id exists.

### 4.2 Static slot resolution (before any team is placed)
Each slot has `candidates: [{source_type, conference|pool_id, pick, tier, parity, priority}]`.
```
applicable = [c for c in slot.candidates if c.parity is None or c.parity == ("even" if season%2==0 else "odd")]
slot.resolved = sorted(applicable, key=priority)        # ordered fallback chain, never empty
```
`slot.resolved[0]` is the PRIMARY source; later entries are used only in §4.6. If `candidates`
is empty or the row is provenance `provisional` with unknown sides, the slot is
`[{source_type: at_large, priority: 1}]` and the bowl is flagged `provisional: true` in output.

### 4.3 Obligation lists per conference (pick index)
For each conference C, `obligations(C)` = every (bowl, slot) whose PRIMARY source is
`conference == C`, ordered by:
1. `selection.conferences[C].mode`:
   - `numbered`: the slot's `pick` (published pick numbers; a missing pick sorts after all
     numbered picks, then by declaration order) — tag `published_pick`.
   - `tiered`: `tier` ascending, then declaration order within a tier — tag `tier_then_declaration`
     (STAND-IN: real tiered conferences place within a tier by judgement).
   - `office_pool`: the conference's `selection.conferences[C].order` list if given, else
     declaration order — tag `office_pool_declaration_order` (STAND-IN for the SEC office's
     placement).
   - `inferred`: the order written by the tier-inference task, provenance `inferred`.
2. The resulting position is `pick_index(C, slot)` = 1..k.

### 4.4 Team pools per conference
`pool(C)` = `[t for t in standing(C) if eligible(t) and t not in cfp_field]`, in standing order.
CFP teams are removed first, so a conference's pick 1 goes to its best remaining team.
Independents form no conference pool; they enter only via §4.6 (pool/at_large slots, or a pool
that names them, e.g. Notre Dame in an ACC-access pool).

### 4.5 The walk (rounds by pick index)
```
K = max pick_index over all conferences
for k in 1..K:
    for C in selection.conference_order:                 # fixed list in config; default: raw names ASCII ascending
        slot = the obligation of C with pick_index == k (skip if none)
        if slot.team is not None: continue               # already filled (cannot happen for primary slots; defensive)
        T = choose(pool(C), slot)                         # §4.7
        if T is None: mark slot.unfilled_by_primary = True; continue   # falls to §4.6
        place(T, slot, reasons=[f"{C} pick {k} ({tag})", *penalty_reasons])
        remove T from every pool
```
Within one round, if both sides of a bowl are primary slots of two conferences, the conference
earlier in `conference_order` fills first and the later one sees it for the rematch check. This
asymmetry is deliberate and documented; it mirrors sequential real drafts.

### 4.6 Pool / at-large / fallback fill (after all rounds)
Walk `bowls` in declaration order; for each slot still empty, walk `slot.resolved` in order:
- `conference`: candidates = `pool(C)` (whatever is left).
- `pool`: candidates = all unplaced eligible teams whose conference (or team name) is in
  `pools[pool_id]`, ordered by `pools[pool_id].order_rule` (`rank` default; `standing` also
  allowed), then name.
- `legacy_pool`: same as pool; membership is a team list; a team is available only if its
  current conference has NOT placed it (it was skipped or unplaced), never pulled out of a
  placement already made — tag `legacy_pool_after_current_conference`.
- `at_large`: candidates = all unplaced eligible teams, ordered by rank then name.
First candidate source with a non-`None` `choose()` result wins; reasons record which fallback
level was used.

### 4.7 `choose(candidates, slot)` — the selection function
```
other = team already placed in the bowl's other slot (or None)
def penalty(T):   # lexicographic tuple, smaller is better; order configurable via policy.penalty_order
    rematch        = 1 if other and other in regular_season_opponents[T] else 0
    repeat_bowl    = 1 if any(h.bowl_root == bowl.root for h in history[T]) else 0
    repeat_opp     = 1 if other and any(h.opponent == other for h in history[T]) else 0
    return (rematch, repeat_bowl, repeat_opp)          # default order
best = None
for T in candidates:                                    # candidates already in the conference's standing order (or pool order)
    if best is not None and wins(best_so_far_top) - wins(T) > policy.win_window: break   # hard rule
    if best is None or penalty(T) < penalty(best): best = T
return best
```
`policy.win_window` (default 1): a slot may skip past better-standing teams only while the skipped
team's win total exceeds the candidate's by at most `win_window` (STAND-IN for the "within one
game" conventions several conferences publish; owner decision §9). With `win_window = 0` the walk is
pure standing order and penalties only break exact win ties.
A penalty of all zeros always stops the scan (the best-standing clean team wins).

### 4.8 Shortfall and surplus
- After §4.6, slots still empty are filled from 5-7 teams (`losses == 7 and wins == 5`) in rank
  order, tag `apr_fill_by_rank` (STAND-IN for the APR-order rule; owner decision OQ7). Still
  empty → `team: None`, reason `no_eligible_team`.
- Eligible teams never placed are listed in `unplaced_eligible` with reason `no_slot`; the count is
  a headline diagnostic (real seasons leave 0-3 out).
- A conference with more obligations than eligible teams leaves its later slots to §4.6.

### 4.10 Operator pools (ESPN Events) — Group of Six placement
Evidence (bowl-tie-ins-draft.md §0.2): G6 placement is a few FIXED conference slots plus an
operator-level flex pool, not per-conference numbered drafts. Encode `pools[operator_espn]` with
`bowls: [...]` (declaration order = prestige/date order chosen by the owner or inferred by T5) and
`conferences: [AAC, CUSA, MAC, Sun Belt, Mountain West, Pac-12]`. Walk: after §4.5 fills every
fixed slot (SEC/Big Ten/ACC/Big 12 picks; Sun Belt #2 and #5; AAC annual bowls; CUSA New Orleans;
Poinsettia champion; legacy group), the operator pool fills its bowls' empty slots in declaration
order from unplaced eligible G6 teams ordered by `(rank, name)`, applying §4.7 penalties and a
per-conference cap = that conference's `guaranteed_count` where published (CUSA 7; AAC "4 of 8" +
4 annual). STAND-IN tag `operator_pool_by_rank` (real placement weighs geography and travel).

### 4.11 Declines (opt-outs) and replacements
Input `state.declined: set(team)` (2025-26: Kansas State, Iowa State). Declined teams are removed
from every pool before §4.5 and listed in output `declined`. A slot that would have been theirs is
filled by the normal walk. If, after §4.6, slots remain empty, §4.8 fills from 5-7 teams; the real
rule draws replacements by APR rank (2024 Louisiana Tech; 2025 Birmingham Bowl backfill) — our
stand-in is model rank, tag `apr_fill_by_rank`.

### 4.12 Parity rules
A candidate with `parity` applies only in matching seasons (SEC: Las Vegas even, Duke's Mayo odd,
per secsports.com; Big Ten mirrors it: Duke's Mayo even, Las Vegas odd, inferred from 2023-25
actuals). Because the SEC pool of six therefore has a different sixth bowl each year, the
SEC's obligations list is built from the resolved slots (§4.2), never hard-coded.

### 4.9 Output
```
{ bowls: [{bowl_root, name, city, date, provisional, slots: [{source_used, team, conference, record, rank, reasons: [...]}]}],
  unplaced_eligible: [...], filled_from_5_7: [...],
  stand_ins_used: {tag: count}, provisional_bowls: [...] }
```

## 5. Explanation chain
Every placement's `reasons` is an ordered list of short strings; the first is always the rule
that created the slot (e.g. "SEC pick 3 (office_pool_declaration_order)"), followed by any penalty
avoided ("skipped Ole Miss: repeat_bowl (Gator 2024)"), the win_window check if it bound, and any
fallback level used. The page renders the first reason inline and the rest on hover/expand.

## 6. Validation (acceptance for the build passes)
1. Determinism: identical state+config twice → byte-identical output; reordering bowls with
   different pick numbers → identical output; reordering same-pick bowls → only those two may swap.
2. Backtest 2025-26 and 2024-25 with the COMMITTEE's final ranking substituted for ours and the
   real end-state: CFP field must match exactly (12/12) under that season's rule entry; bracket
   pairings must match; QF bowl placement may differ (stand-in) and is reported, not asserted.
3. Backtest bowls the same way: report exact-match rate per conference and overall; no threshold
   asserted in v1, but the rate is a tracked number and drives tier inference (T5).
4. Config flips: `g6_slot_rule`, `independent_guarantees`, `seeding`, `aq_mode`, `win_window` each
   have a fixture whose expected output changes when flipped (mutation-check log convention).
5. Loader: unknown keys, overlapping seasons, unknown conference/pool references, a `cfp_host`
   bowl not present in that season's cfp_rules hosts → load error naming the path.

## 7. Config parameters (all season-scoped unless noted)
cfp_rules: field_size, bye_count, aq_mode, aq_count, aq_conferences, g6_conferences,
g6_slot_rule, seeding, independent_guarantees[{team,max_rank}], first_round_pairings,
quarterfinal_feeds, first_round{dates}, quarterfinal_hosts[{bowl_root,date}],
semifinal_hosts[...], semifinal_choice_by_top_seed{}, qf_preferences{conf:[...]},
championship{site,city,date}.
bowl_tie_ins: bowls[...] (§4.1-4.2 shapes), pools{id:{conferences|teams, order_rule}}, affiliations{team: group_id} (§4.0),
selection{conference_order, conferences{C:{mode, order, criteria, provenance, source}}}.
policy (global, versioned): penalty_order, history_seasons (H), win_window, apr_fill: bool.

## 8. Eligibility hook (not v1)
`eligible()` is the existing 6-win rule. The NCAA counts at most one FCS win toward six (and only
against an FCS opponent meeting the scholarship threshold); the pipeline stores FCS losses but not
FCS-win metadata, so this cannot be modelled today. Config `eligibility_rule: wins_6` now;
`ncaa_fcs_counting` reserved.

## 9. Owner decisions (this spec)
- D1 = OQ1 g6_slot_rule default (any_team as reported vs champion).
- D2 = OQ2 pick order basis (standing order vs model rank).
- D3 = OQ3 QF stand-in (preference lists vs host order).
- D4 win_window default (1 recommended; 0 = pure standing order).
- D5 penalty_order default ((rematch, repeat_bowl, repeat_opp) recommended).
- D6 history window H (3 recommended, matches "last 2-3 years").
- D7 = OQ7 5-7 fill as APR stand-in.
- D8 conference_order default (ASCII ascending) — only matters for same-round rematch visibility.
- D9 operator-pool bowl order (declaration order supplied by owner vs inferred by T5).
- D10 whether `declined` is a manual overrides file (like database/game_result_overrides.json) — recommended.
