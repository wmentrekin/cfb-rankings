"""artifacts/bowl_names.py -- short, one-line display names for postseason games (K6, see
docs/season-grid-postseason-format/plan.yaml).

REVISED after fix-cycle-1 review: the first version of this module used a STRUCTURAL rule ("the
root name is the single token before a trailing 'Bowl'") instead of a curated map, reasoning that
sponsor rotation would rot a map but not a rule. That premise is false for real bowl names -- many
real 2025 roots are two or three tokens, not one, and the rule silently produced a WRONG name
rather than an unshortened one:

    TransPerfect Music City Bowl       -> "City Bowl"        (root is "Music City Bowl")
    Isleta New Mexico Bowl             -> "Mexico Bowl"       (root is "New Mexico Bowl")
    Lockheed Martin Armed Forces Bowl  -> "Forces Bowl"       (root is "Armed Forces Bowl")
    SERVPRO First Responder Bowl       -> "Responder Bowl"    (root is "First Responder Bowl")
    SRS Distribution Las Vegas Bowl    -> "Vegas Bowl"        (root is "Las Vegas Bowl")
    R+L Carriers New Orleans Bowl      -> "Orleans Bowl"      (root is "New Orleans Bowl")
    RoofClaim.com Boca Raton Bowl      -> "Raton Bowl"        (root is "Boca Raton Bowl")

A wrong short name is worse than a long one: it's what a touch or screen-reader user identifies
the bowl by (see K7's role="img"/aria-label pattern, which reads whichever name this module hands
it), and K8's own principle -- a wrong value on a public page is worse than an absent one --
applies here exactly as it does to CFP seeds.

THE FIX: a curated list of canonical bowl ROOT names (_ROOT_NAMES), matched against the full name
by LONGEST case-insensitive suffix, on a whitespace boundary. This is still sponsor-agnostic by
construction -- a new presenting sponsor changes only the prefix in front of a root that's already
in the list, so a sponsor rotation never requires touching this module, which was the actual
durability concern that motivated (but did not justify) the rejected structural rule. What curation
buys back is correctness for multi-token roots, which no prefix-stripping rule can distinguish from
a single-token root without already knowing where the root starts.

_ROOT_NAMES is NOT claimed exhaustive -- CFBD's real slate is roughly 35 bowls a season, and only
the ones verified against real 2025 names (either in the original task fixtures or surfaced by
review) are curated here. A bowl not in this list -- whether genuinely new or simply not yet added
-- passes through WHOLE, never guessed at by truncation. That is the safe direction: the
personal-site CSS clamp (K6) is what actually guarantees a single line for that case, not this
module. Extending coverage later means adding a verified root name, never inventing one.
"""
from typing import Optional

# Exact full names (as CFBD's playoff_bowl_name/notes carries them) that don't end in the word
# "Bowl" at all, so no suffix match against _ROOT_NAMES below could ever reach them. Kept
# separate from _ROOT_NAMES (which is itself suffix-matched) because these are matched by exact
# full-string equality, not suffix -- there is no sponsor-prefixed variant of a CFP round name.
#
# NO "CFP" PREFIX (fix-cycle-1, round 2): the frontend badges a CFP cell separately, driven by
# playoff_round being non-null (K5) -- a quarterfinal's short label is already just "Rose Bowl",
# with the CFP identity carried entirely by the badge, never repeated in the label. Prefixing
# these two with "CFP" duplicated that identity inside the very label the badge sits next to
# ("[CFP] CFP First R...") and cost the characters that forced the truncation in the first
# place. playoff_round still carries the full round name as data for any consumer that wants it
# (K5) -- nothing here is lost, only no longer said twice.
_OVERRIDES = {
    "College Football Playoff First Round Game": "First Round",
    "College Football Playoff National Championship Presented by AT&T": "National Championship",
}

# Canonical bowl root names -- the part of a bowl's official name that survives a title-sponsor
# change, in the CASING each bowl actually uses. Verified against real names only (see module
# docstring); NOT an attempt to enumerate CFBD's full ~35-bowl slate. Matched against the full
# name by longest case-insensitive suffix on a whitespace boundary (see short_bowl_name), so a
# root that is itself the WHOLE name (no separate sponsor at all, e.g. "Myrtle Beach Bowl", or a
# sponsor-eponymous rebrand with no separate root, e.g. "Pop-Tarts Bowl") matches too -- there is
# nothing in front of it to strip.
_ROOT_NAMES = [
    # Single-token roots (original 2025 fixture set).
    "Gasparilla Bowl",
    "LA Bowl",
    "Frisco Bowl",
    "Independence Bowl",
    # Sponsor-eponymous rebrands: the sponsor IS the name, no separate root to strip.
    "Pop-Tarts Bowl",
    "Xbox Bowl",
    "Rate Bowl",
    "Duke's Mayo Bowl",
    # No title sponsor at all today -- the full name already is the root.
    "Myrtle Beach Bowl",
    # Multi-token roots (fix-cycle-1: the cases a single-trailing-token rule got wrong).
    "Music City Bowl",
    "New Mexico Bowl",
    "Armed Forces Bowl",
    "First Responder Bowl",
    "Las Vegas Bowl",
    "New Orleans Bowl",
    "Boca Raton Bowl",
    # New Year's Six / CFP quarterfinal-and-semifinal hosts -- already exactly their root name,
    # included so an accidental future sponsor prefix (e.g. a "XYZ Rose Bowl") still shortens
    # correctly instead of silently falling through to the full name.
    "Rose Bowl",
    "Sugar Bowl",
    "Cotton Bowl",
    "Orange Bowl",
    "Fiesta Bowl",
    "Peach Bowl",
]
# Longest-first so a suffix match against a shorter root can never shadow a longer, more specific
# one that also matches (no case in the current list actually collides, but the ordering makes
# that guarantee structural rather than incidental).
_ROOT_NAMES_BY_LENGTH_DESC = sorted(_ROOT_NAMES, key=len, reverse=True)


def short_bowl_name(name: Optional[str]) -> Optional[str]:
    """Short, single-line form of a postseason game's full display name.

    None/empty input passes through unchanged (mirrors _game_name_for_row's own null discipline
    -- this function is never the reason a null becomes a string or vice versa). A name that
    doesn't match _OVERRIDES and doesn't end (on a whitespace boundary) with any curated entry in
    _ROOT_NAMES degrades to the FULL name, not a guess -- see the module docstring for why that's
    the only safe direction for an unrecognized name.
    """
    if not name:
        return name
    if name in _OVERRIDES:
        return _OVERRIDES[name]
    stripped = name.strip()
    lowered = stripped.lower()
    for root in _ROOT_NAMES_BY_LENGTH_DESC:
        root_lower = root.lower()
        if lowered == root_lower or lowered.endswith(" " + root_lower):
            return root
    return name
