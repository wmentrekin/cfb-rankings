"""artifacts/bowl_names.py -- short, one-line display names for postseason games (K6, see
docs/season-grid-postseason-format/plan.yaml).

WHY A RULE, NOT JUST A MAP (the mapping REVERSED into this repo, but the shape of it is new):
CONFERENCE_DISPLAY_NAMES (artifacts/rankings.py) is a flat curated dict, and that precedent is
fine there because it covers ~11 conference tokens that are structurally stable -- realignment is
rare and each old/new name is a deliberate, memorable event. Ordinary bowl names are a much worse
fit for the same shape: CFBD's real 2025 vocabulary is roughly 35 sponsor-prefixed bowls (e.g.
"Union Home Mortgage Gasparilla Bowl", "Bucked Up LA Bowl"), and title sponsors change most
offseasons -- a pure ~40-entry map would silently stop shortening the moment a sponsor changes,
reintroducing the long-label row-height bug this whole feature exists to fix, with no warning and
no failing test until someone notices a row wrap on the live site.

So the durable part is a STRUCTURAL RULE: an ordinary bowl name is [sponsor prefix] + [root name]
+ "Bowl", and the root name is reliably the single token immediately before the trailing "Bowl" --
true for every sponsor-prefixed bowl AND for the sponsor-eponymous ones with no separate root
("Pop-Tarts Bowl", "Xbox Bowl", "Rate Bowl", already <= 2 words, so the rule is a no-op on them).
That rule needs no maintenance as sponsors rotate.

_OVERRIDES exists ONLY for names the structural rule cannot reach at all -- CFP round names that
don't end in the word "Bowl" ("...First Round Game", "...National Championship Presented by
AT&T"). It is deliberately small and is not an attempt to re-enumerate the ~40 bowl names the rule
already handles. An override going stale (a presenting sponsor changes) or a genuinely new,
unmapped shape degrades to the FULL name unchanged, never to something wrong -- the personal-site
CSS clamp (K6) is what actually guarantees a single line for that case, not this module.
"""
import re
from typing import Optional

# Exact full names (as CFBD's playoff_bowl_name/notes carries them) that the structural rule
# below cannot shorten, because they don't end in the word "Bowl". Keep this SMALL: it is not a
# home for ordinary sponsor-prefixed bowls (the rule already covers those durably) -- only for
# shapes the rule structurally cannot parse.
_OVERRIDES = {
    "College Football Playoff First Round Game": "CFP First Round",
    "College Football Playoff National Championship Presented by AT&T": "CFP National Championship",
}

# The root bowl name is the single token immediately before a trailing "Bowl" (case-insensitive,
# tolerant of extra whitespace). Matches "Gasparilla Bowl" out of "Union Home Mortgage Gasparilla
# Bowl", and is a no-op on already-short names like "Rose Bowl" or "Pop-Tarts Bowl".
_TRAILING_BOWL_RE = re.compile(r"(\S+\s+Bowl)\s*$", re.IGNORECASE)


def short_bowl_name(name: Optional[str]) -> Optional[str]:
    """Short, single-line form of a postseason game's full display name.

    None/empty input passes through unchanged (mirrors _game_name_for_row's own null
    discipline -- this function is never the reason a null becomes a string or vice versa).
    An unmapped name that doesn't end in "Bowl" and isn't in _OVERRIDES degrades to the FULL
    name, not a guess -- see the module docstring for why that's the safe direction.
    """
    if not name:
        return name
    if name in _OVERRIDES:
        return _OVERRIDES[name]
    match = _TRAILING_BOWL_RE.search(name.strip())
    if match:
        return match.group(1)
    return name
