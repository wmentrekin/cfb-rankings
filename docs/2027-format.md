# Planning for the 2027 Season Format

Status: planning only. Nothing in this document is built, and nothing here changes what the
pipeline or the site publishes today. Written 2026-11-27; several facts below were announced in
the last three weeks and some are still undecided, so each is marked.

## 1. What Is Changing

### Confirmed

| Fact | Detail | Source |
| --- | --- | --- |
| Field size | 24 teams from the 2027 season through 2031-32. | CFP release, Nov 9, 2026 |
| Automatic bids | The ACC, Big Ten, Big 12 and SEC champions, plus one Group of Six place. The other 19 are the highest-ranked teams left. | CFP, ESPN |
| Seeding | Straight by committee rank, 1 to 24. A champion gets no seeding boost. | Athlon, CFP explainer |
| Byes | Seeds 1 to 8 skip the first round. | CFP explainer |
| First round | Seeds 9 to 16 host seeds 17 to 24 on campus. Eight games, Dec 9-11, 2027 (Thursday to Saturday). | ESPN |
| Second round | Seeds 1 to 8 host the first-round winners on campus. Eight games, Dec 16-18, 2027. | ESPN |
| Quarterfinals | At bowls: Sugar Dec 31, 2027; Fiesta, Peach and Rose Jan 1, 2028. | CBS, Yahoo |
| Semifinals | Jan 13 and 14, 2028. | CBS, Yahoo |
| Final | Jan 24, 2028, New Orleans. | ESPN |
| Total games | 23 (8 + 8 + 4 + 2 + 1). Today it is 11. | derived |
| Power Four title games | Gone from 2027. Champions come from regular-season standings and tiebreakers. | CFP, several outlets |
| Regular season | First Saturday Aug 28, 2027; last Saturday Nov 27, 2027. Fourteen weeks, two open dates per team. | CFP, fbschedules |
| Army-Navy | Moves to the first Saturday in December (Dec 4, 2027, Philadelphia) as a 13th game by NCAA waiver. The committee will not consider it in its final ranking. Reviewable each year. | CFP release, Nov 25, 2026; ESPN |
| Scheduling standards | Power Four teams need 10 qualifying games. Only one FCS win earns selection credit. | fbschedules |

### Reported but not settled

| Question | What is known |
| --- | --- |
| Is the Group of Six place for a champion or for any team? | Sources disagree. ESPN and NBC say "highest-ranked team"; Front Office Sports says "highest-ranked Group of 6 champion". The CFP's own explainer needs reading directly; its site refuses automated requests. Our 2026 rule already uses the highest-ranked team. |
| Do the Group of Six conferences keep their title games? | None has decided. Coaches want them; the calendar may not allow it. The Mountain West has said it could give up one open week to fit one in. |
| Selection day | Not announced. If it is Sunday Dec 5, any Group of Six title game on Dec 4 is the same weekend as Army-Navy. |
| How the second-round winners are paired for the quarterfinals | Not found. Assume a fixed bracket (no reseeding) until the CFP says otherwise. |
| Notre Dame | The 12-team format guarantees a place if Notre Dame is in the top 12. Whether any guarantee survives is not reported; with 19 at-large places it is unlikely to matter. |
| Power Four tiebreakers for naming a champion | Not published. Today's rules pick two teams for a title game; picking one champion may use different steps. |
| Non-playoff bowls | Bowl Season says there will be fewer. No count, no revised eligibility rule, no revised tie-ins. |
| How CFBD will label the new rounds | Unknown until the data exists. The actual-bracket builder reads round names from CFBD's game notes. |

### Positions already taken for this project

- Army-Navy counts toward our ratings even though the committee ignores it. The teams play it
  as a real game.
- The Group of Six place goes to the highest-ranked team, champion or not.

## 2. What It Touches

Today the season's shape is spread across the code as constants that are true for 2025 and 2026.
Almost none of them are keyed by season. That is the core problem: 2027 cannot be added beside
2026 until they are.

### Pipeline (cfb-rankings)

| Area | Where | Today | 2027 |
| --- | --- | --- | --- |
| Week from a date | `utils.get_cfb_week` | Anchored on the Tuesday on or after Aug 24's Monday. The last Saturday of August is "Week 0", a partial slate. Capped at week 16. | Aug 28, 2027 is a full opening weekend. CFBD will most likely call it Week 1, which puts our date-derived week one behind CFBD's all season. The anchor has to be per season, or read from the stored games. |
| Week 0 guard | `main.py` | Refuses to run for week 0. | Fine if the week numbering is fixed; wrong (skips opening weekend) if it is not. |
| Prior fade | `model/model.py` | The prior's weight reaches 0 at week 7. | Two open weeks mean teams have played fewer games by week 7 than they have today. Decide whether the fade should follow weeks, games played, or how connected the schedule is. |
| Which games are rated | `rankings_games` view | Regular season only. | Army-Navy stays in as long as CFBD tags it regular season. Worth an explicit check rather than an assumption. Playoff games stay out, as now. |
| Title-game table | `QUALIFYING_CHAMPIONSHIP_CONFERENCES`, `DIVISIONAL_CHAMPIONSHIP_CONFERENCES` | One table for every season: these conferences send their top two to a title game. | Needs a season. In 2027 the Power Four send nobody anywhere: the race is for first place. Each Group of Six conference is one or the other, to be announced. |
| Clinch and elimination | `compute_conference_championship_status` | Race for a top-two berth (`top_n = 2`), or first in a division (`top_n = 1`). | The `top_n = 1` path already exists and is tested, so "clinched the title" and "eliminated from the title" fall out of it. The head-to-head shortcut is written for a two-team tie for the last berth and needs rechecking for first place. |
| Naming a champion | `tiebreaker_engine`, `conference_tiebreakers.json` | Orders tied teams as far as the steps we can compute allow. A title game then settles the champion. | No game settles it. The engine has to produce a champion or say plainly that it cannot (steps that use outside ratings or a draw). The projection needs a fallback: best conference record, then our rank, as now. |
| Title-game detection | `identify_conference_championship_games` | Finds games whose CFBD note ends in "Championship". | Unchanged for conferences that keep a game; must find nothing for the Power Four and not be fooled by a regular final-week game. |
| Grid columns | `artifacts/schedule.py` | Week columns, a conference-championship column, an Army-Navy week column, one Bowls column. | Fourteen week columns. Dec 4 becomes one shared column: Army-Navy, plus any Group of Six title games. The separate Army-Navy column goes away. The Bowls column becomes "Postseason". |
| Flex weeks | `FLEX_WEEK_TBD_CONFIG` | Pac-12 week 13 pairings announced late. | Review once 2027 schedules exist. |
| Bowl eligibility | `compute_bowl_eligibility` | Eligible at 6 wins, out at 7 losses. | Unknown. If the bowls shrink or the rule changes, this status may stop meaning anything and should be droppable per season. |
| Playoff rules | `artifacts/playoffs.py` `SEASON_RULES` | Keyed by season already. But the bracket shape is fixed in code: four rounds, 12 seeds, byes for 1 to 4, 11 slots named FR/QF/SF/CH. | Five rounds, 24 seeds, byes for 1 to 8, 23 slots. The bracket has to be built from a description of the rounds, not from constants. |
| Projected field | `select_projected_field` | Four champions (title-game winner, or a projection), best Group of Six team, Notre Dame if top 12, then at-large by rank; seeds follow our rank. | Same idea with 19 at-large places. Champions come from standings, not a title game. First teams out: keep four. |
| Projection to actual | `_complete_actual_bracket` | Switches when stored rows identify all 12 seeds. | Same test with 24, once we know how CFBD stores seeds and round names. The hand-entered first-teams-out rule carries over. |
| Schedule | `weekly-update.yml` | Sundays, plus two dated runs for 2026 selection day. | First run is a week earlier (Aug 29). Playoff games start on a Thursday (Dec 9), so the look-ahead ingest and any dated runs need new dates. |

### Site (personal-site)

| Area | Where | Today | 2027 |
| --- | --- | --- | --- |
| Bracket | `PlayoffBracket.astro` and its client copy | Four columns with fixed connector geometry. | Five columns and 23 games. The geometry is written for 4-2-1 feeders; first round into second round is one feeder plus a bye, like today's first round into quarterfinals, so the pattern extends, but the width does not: five columns at desktop width leave about 230px each. Needs a design pass, not only more columns. |
| The Field | same | 12 rows beside First Out. | 24 rows. |
| Projection note | same | One line with the selection date. | Same, once the date is announced. |
| Grid status wording | `ScheduleGrid.astro` | "Clinched", "Eliminated", "would play X" for a title game. | For a conference with no game: clinched or eliminated from the title, and no opponent line. |
| Grid columns | same | Mirrors the pipeline's slots. | Follows the pipeline. Fixed column widths already handle a different column count. |
| Bowl logos | `src/lib/bowlLogos.ts` | 40 bowls. | Prune to whatever bowls remain. |
| Methodology | methodology page | Says the prior fades at week 7 in football. | Update if the fade changes. |

## 3. Proposed Shape

One description of a season, used by everything that currently hard-codes one.

```
SeasonFormat(season)
  first_saturday              -> week numbering
  last_regular_saturday
  extra_regular_dates         -> Army-Navy on Dec 4
  title_game_conferences      -> who still plays one; everyone else races for first
  playoff:
    field_size, bye_seeds
    rounds: name, label, games, site (campus or bowl), dates
    automatic_bids: conferences, group_of_six_rule, guarantees
    first_out_count, selection_date
  bowls: eligibility rule, or none
```

The 2025 and 2026 entries would restate exactly what the constants say now, so introducing it
changes no output. 2027 is then a new entry, not a rewrite.

The playoffs payload the site reads is already close to general: it lists rounds, and each game
names its round, its order in the round and which game feeds each side. A 24-team payload fits
that contract with one more round. The work is in the two things that assume four rounds: the
pipeline's bracket builder and the site's layout.

## 4. Order of Work

Safe now, with no public change:

1. Introduce the season description with 2025 and 2026 values and move the scattered constants
   behind it. Prove the published files are byte-for-byte the same.
2. Rebuild the bracket builder on a description of rounds. Prove 2025 and 2026 payloads are the
   same, then add a 24-team description with tests on an invented field.
3. Make the site's bracket draw however many rounds the payload has, checked against a 24-team
   fixture. The live site shows nothing new, because no 2027 file is published.

Blocked on announcements:

4. Title-game table for 2027: needs each Group of Six conference's decision.
5. Champion tiebreakers: needs the Power Four's 2027 rules.
6. Grid columns for Dec 4 and the postseason: needs selection day and the bowl picture.
7. Actual-bracket reading: needs real CFBD data for the new rounds.
8. Week numbering: can be designed now, confirmed when CFBD publishes the 2027 schedule.

## 5. Decisions Needed

1. Should the prior fade by week, as now, or by games played? Two open weeks make "week 7" an
   earlier point in most teams' seasons than it is today.
2. Should playoff games count toward the ratings? They do not today. With 23 games starting
   Dec 9, the final ratings would otherwise ignore the most informative games of the year.
3. If Group of Six title games survive, should they count in our ratings the week of Dec 4
   alongside Army-Navy, and should the projected field wait for them?
4. Should the grid keep a bowl column and bowl-eligibility status in 2027 if the bowls shrink?
5. How should a conference title that our tiebreak steps cannot settle be shown: as undecided,
   or as our best guess labelled as a projection?

## 6. Things to Watch

- CFP: selection day date; quarterfinal pairing rule; the exact Group of Six wording.
- Each Group of Six conference: title game yes or no, and its date.
- ACC, Big Ten, Big 12, SEC: 2027 champion tiebreakers.
- Bowl Season: surviving bowls, eligibility rule, tie-ins.
- CFBD: 2027 schedule release (week numbers, Army-Navy's season type), then round labels for
  the new playoff rounds in December 2027.
- Conference membership for 2027, which the standings tables depend on.
- The Army-Navy policy, which the CFP can revisit each year.

## Sources

- CFP, 24-team expansion: https://collegefootballplayoff.com/news/2026/11/9/24-team-expansion
- CFP, about the 24-team format: https://collegefootballplayoff.com/sports/2026/11/9/24-team-format
- CFP, Army-Navy update: https://collegefootballplayoff.com/news/2026/11/25/army-navy-update
- ESPN, 24-team format FAQ: https://www.espn.com/college-football/story/_/id/50433738/cfp-playoff-24-team-format-faq
- ESPN, Army-Navy waiver: https://www.espn.com/college-football/story/_/id/50535192/waiver-approved-army-navy-13th-game-2027
- fbschedules, scheduling standards: https://fbschedules.com/college-football-playoff-sets-scheduling-standards-for-24-team-era/
- NCAA.com, 24-team bracket: https://www.ncaa.com/news/football/article/2026-11-11/everything-know-about-24-team-college-football-playoff-bracket
- Athlon, format Q&A: https://athlonsports.com/college-football/playoff-format-expansion-24-teams-cfp-explained
- CBS Sports, 2026 and 2027 playoff dates: https://www.cbssports.com/college-football/news/college-football-playoff-schedule-dates-2026-2027/
- Front Office Sports, Group of Six scheduling: https://frontofficesports.com/article/group-of-6-leaders-fear-scheduling-squeeze-24-team-playoff/
- Yahoo Sports, future of bowl games: https://sports.yahoo.com/articles/24-team-cfp-coming-happens-104040964.html

These were read as search summaries, not as full pages. Dates and rules should be rechecked
against the CFP's own pages before anything is built on them.
