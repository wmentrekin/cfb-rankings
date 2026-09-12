# Primary-source tiebreaker text

Supplied by the repo owner on 2026-09-11, pasted from the conferences' own published
policies. **This directory is the authoritative input for `conference_tiebreakers.yaml`.**

It exists because the build environment cannot reach any conference website: the egress
gateway returns `403` to CONNECT for every one of them (verified with curl, not only through
the WebFetch tool). Research-derived step orders proved unreliable — two of ten were wrong —
so rules are transcribed from text a human supplied, and each conference's config entry
records which of these files it came from.

## Provenance levels used in the config

| level | meaning |
|---|---|
| `primary_source` | transcribed from a file here |
| `verified_against_real_tie` | `primary_source`, and it reproduces a known real outcome |
| `search_derived` | reconstructed from secondary reporting; **not** to be trusted for step order |

## Received

All ten FBS conferences. Supplied 2026-09-11 (Power 4) and 2026-09-12 (Group of 6).

| conference | file | era covered | notes |
|---|---|---|---|
| ACC | `acc.txt` | **amended 2026-07-01** | Also contains a secondary summary describing the PRE-amendment chain. The two differ materially — see below. |
| SEC | `sec.txt` | 2024 onward (divisionless) | Includes Appendix A, the full capped-relative-scoring-margin formula. Appendix B (~25 worked examples) is referenced but its text was not supplied. |
| Big 12 | `big12.txt` | 2024 onward (16 teams) | |
| Big Ten | `bigten.txt` | 2024 onward (18 teams, divisionless) | |
| American | `american.txt` | undated | Chain dominated by the CFP-ranking cascade; two gates no other conference has. |
| Mountain West | `mountainwest.txt` | **dated 8/1/2023** | The only Group of 6 document with an explicit date. Explains the real 2025 four-way tie. |
| MAC | `mac.txt` | undated | Fully expressible today; peculiar only in step ORDER. |
| Sun Belt | `sunbelt.txt` | undated | The only conference here that still plays DIVISIONS. Most demanding of the ten. |
| Conference USA | `cusa.txt` | undated | |
| Pac-12 | `pac12.txt` | **2026 only** | For 2024–25 the Pac-12 was Oregon State and Washington State alone. Must be season-scoped. |

Seven of the ten are undated, so each needs a defensible `season_min` in the config rather than
an open-ended range. Earlier seasons resolve to no rules and fall back, which is honest about
what we have not read — see the ACC's 2023 floor for the pattern.

## Two findings that shaped the design

**The ACC has two distinct policies and they are not interchangeable.** The amended
2026-07-01 text runs head-to-head, then SportSource Team Success Ranking, then a draw —
it DELETED the common-opponent and strength-of-schedule steps. The 2025 chain still had
them, and strength of schedule is what actually put Duke through a five-way tie (Duke's
opponents went 32-32, .500, best of the five). A single rule set per conference would
therefore be wrong for one era or the other, which is why config entries are season-scoped.

**"Per SportSource Analytics" does not always mean unobtainable.** The SEC's step E publishes
its entire formula in Appendix A, so it is computable from scores we already hold. The ACC's
Team Success Ranking publishes nothing and is genuinely opaque. The distinction is per
conference and must not be generalised from the name.
