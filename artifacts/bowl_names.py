"""artifacts/bowl_names.py -- short, one-line display names for postseason games (K6, see
the season-grid-postseason-format plan).

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

_ROOT_NAMES is NOT claimed exhaustive beyond the verified 2025 slate -- CFBD's real slate is
roughly 35 bowls a season, and as of this pass all 35 ordinary 2025 bowls plus the 6 CFP-hosted
New Year's Six/quarterfinal/semifinal roots are curated (see tests/test_postseason_slots.py's
complete 43-name pinned table). A bowl not in this list -- whether genuinely new next season or
simply not yet added -- passes through WHOLE, never guessed at by truncation. That is the safe
direction: the personal-site CSS clamp is what actually guarantees a single line for that case,
not this module. Extending coverage later means adding a verified root name, never inventing one.

The displayed short name carries
neither the sponsor NOR the word "Bowl" itself -- "Gasparilla", "Music City", "Sugar", not
"Gasparilla Bowl". So once a curated root is resolved (never for an unmatched pass-through, and
never for an _OVERRIDES result -- those are already exactly the short form the user wants, e.g.
"First Round"), _strip_trailing_bowl removes a trailing " Bowl Classic" (checked first, since it
also, trivially, ends in the word "Bowl" but the whole "Classic" qualifier must leave with it) or
otherwise a trailing " Bowl", refusing any strip that would leave an empty or whitespace-only
result -- an empty short name must never happen, so a hypothetically degenerate root falls back to
its unstripped form rather than vanish.
"""
from typing import Optional

# Exact full names (as CFBD's playoff_bowl_name/notes carries them) that don't end in the word
# "Bowl" at all, so no suffix match against _ROOT_NAMES below could ever reach them. Kept
# separate from _ROOT_NAMES (which is itself suffix-matched) because these are matched by exact
# full-string equality, not suffix -- there is no sponsor-prefixed variant of a CFP round name.
# Never passed through _strip_trailing_bowl: these values are already exactly the short form the
# user wants ("First Round", not "First Round Game"; "National Championship", not "National").
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
# change, in the CASING each bowl actually uses. Matched against the full name by longest
# case-insensitive suffix on a whitespace boundary (see short_bowl_name), so a root that is
# itself the WHOLE name (no separate sponsor at all, e.g. "Myrtle Beach Bowl", or a
# sponsor-eponymous rebrand with no separate root, e.g. "Pop-Tarts Bowl") matches too -- there is
# nothing in front of it to strip. The trailing "Bowl" on every entry below is stripped by
# short_bowl_name AFTER the match (K9) -- it stays part of the curated string here because
# matching still operates against the bowl's real official name, only the *displayed* result is
# shortened further.
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
    # The 19 ordinary 2025 bowls that were verified against the live DB but not yet curated.
    # Each right-hand root below is the
    # canonical part of the CFBD `notes` string on the left (kept here only as a comment, never
    # matched against):
    #   68 Ventures Bowl                  -> 68 Ventures Bowl   (sponsor IS the name)
    #   AutoZone Liberty Bowl             -> Liberty Bowl
    #   Bad Boy Mowers Pinstripe Bowl     -> Pinstripe Bowl
    #   Cheez-It Citrus Bowl              -> Citrus Bowl
    #   Famous Idaho Potato Bowl          -> Idaho Potato Bowl
    #   GameAbove Sports Bowl             -> GameAbove Sports Bowl (sponsor IS the name)
    #   Go Bowling Military Bowl          -> Military Bowl        (sponsor "Go Bowling" contains
    #                                                               "Bowl" but is never inspected --
    #                                                               only the curated root is matched)
    #   IS4S Salute to Veterans Bowl      -> Salute to Veterans Bowl
    #   JLab Birmingham Bowl              -> Birmingham Bowl
    #   Kinder's Texas Bowl               -> Texas Bowl
    #   ReliaQuest Bowl                   -> ReliaQuest Bowl      (sponsor IS the name)
    #   Sheraton Hawaiʻi Bowl             -> Hawaiʻi Bowl         (U+02BB ʻOKINA, not an apostrophe)
    #   Snoop Dogg Arizona Bowl           -> Arizona Bowl
    #   StaffDNA Cure Bowl                -> Cure Bowl
    #   TaxSlayer Gator Bowl              -> Gator Bowl
    #   Tony the Tiger Sun Bowl           -> Sun Bowl
    #   Trust & Will Holiday Bowl         -> Holiday Bowl
    #   Valero Alamo Bowl                 -> Alamo Bowl
    #   Wasabi Fenway Bowl                -> Fenway Bowl
    "68 Ventures Bowl",
    "Liberty Bowl",
    "Pinstripe Bowl",
    "Citrus Bowl",
    "Idaho Potato Bowl",
    "GameAbove Sports Bowl",
    "Military Bowl",
    "Salute to Veterans Bowl",
    "Birmingham Bowl",
    "Texas Bowl",
    "ReliaQuest Bowl",
    "Hawaiʻi Bowl",
    "Arizona Bowl",
    "Cure Bowl",
    "Gator Bowl",
    "Sun Bowl",
    "Holiday Bowl",
    "Alamo Bowl",
    "Fenway Bowl",
]
# Longest-first so a suffix match against a shorter root can never shadow a longer, more specific
# one that also matches. Verified (K9): none of the 19 newly-added roots is a whitespace-boundary
# suffix of, or has as a whitespace-boundary suffix, any other entry in this list -- each ends in
# a distinct token immediately before "Bowl" ("Ventures", "Liberty", "Pinstripe", "Citrus",
# "Potato", "Sports", "Military", "Veterans", "Birmingham", "Texas", "ReliaQuest", "Hawaiʻi",
# "Arizona", "Cure", "Gator", "Sun", "Holiday", "Alamo", "Fenway"), so the longest-first ordering
# remains a structural guarantee, not an incidental one, even with the expanded list.
_ROOT_NAMES_BY_LENGTH_DESC = sorted(_ROOT_NAMES, key=len, reverse=True)


def _strip_trailing_bowl(root: str) -> str:
    """K9: strip a resolved curated root's trailing sponsor-inert suffix so the displayed short
    name carries neither the sponsor nor the word "Bowl" ("Gasparilla", not "Gasparilla Bowl").

    Only ever called on a root that has ALREADY been resolved by exact or suffix match against
    _ROOT_NAMES -- never on a raw, unmatched input name and never on an _OVERRIDES result, both of
    which are returned as-is by short_bowl_name before this function is reached.

    " Bowl Classic" is checked before " Bowl" -- a root ending in the former also, trivially, ends
    in the word "Bowl", but the whole "Classic" qualifier must be removed with it rather than
    surviving as an orphaned trailing word.

    Refuses any strip that would leave an empty or whitespace-only result, falling back to the
    unstripped root instead -- no input may ever produce an empty short name.
    """
    for suffix in (" Bowl Classic", " Bowl"):
        if root.endswith(suffix):
            candidate = root[: -len(suffix)]
            if candidate.strip():
                return candidate
            return root
    return root


def short_bowl_name(name: Optional[str]) -> Optional[str]:
    """Short, single-line form of a postseason game's full display name.

    None/empty input passes through unchanged (mirrors _game_name_for_row's own null discipline
    -- this function is never the reason a null becomes a string or vice versa). A name that
    doesn't match _OVERRIDES and doesn't end (on a whitespace boundary) with any curated entry in
    _ROOT_NAMES degrades to the FULL name, not a guess -- see the module docstring for why that's
    the only safe direction for an unrecognized name. A name that DOES resolve to a curated root
    has its trailing "Bowl"/"Bowl Classic" stripped (K9, _strip_trailing_bowl) -- an _OVERRIDES
    result and an unmatched pass-through are both returned before that stripping is ever reached,
    so neither is ever altered by it.
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
            return _strip_trailing_bowl(root)
    return name
