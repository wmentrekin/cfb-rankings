"""
Schema, loader and load-time validator for the per-conference tiebreaker configuration
(docs/conference-tiebreakers/, task T1).

WHY JSON, NOT YAML (deviation from plan K1)
--------------------------------------------
Plan K1 says "a YAML file of per-conference ordered step lists". PyYAML is not a declared
dependency in pyproject.toml, and all five GitHub Actions workflows run `uv sync --frozen`,
which refuses to install anything not already pinned in uv.lock. Adding PyYAML is out of this
task's scope (forbidden_scope: pyproject.toml, uv.lock). The config is therefore JSON, parsed
with the stdlib `json` module -- no new dependency. Because JSON has no comments, every place a
human would otherwise leave one gets an explicit string field instead: `notes` on a rule set,
`cites` on a step.

WHAT THIS MODULE OWNS
----------------------
- The config schema (Step / MultiTeamRules / RuleSet / TiebreakerConfig dataclasses).
- `load_conference_rules` / `validate_config`: strict, load-time validation. A malformed or
  unknown-step config raises `TiebreakerConfigError` naming the offending thing and its path
  in the document -- never a silent pass-through to be discovered later at use.
- `rules_for(config, conference, season)`: the single lookup callers (T4) need. No config for a
  conference, or no rule set covering the season, returns None -- callers fall back to current
  behaviour (plan R4 / requirements AC7). This module never decides what the fallback is.

THE REGISTRY-COUPLING PROBLEM (dependency injection)
------------------------------------------------------
A parallel agent is building `artifacts/tiebreaker_steps.py`, exporting
`STEP_REGISTRY: Dict[str, Callable]`. This module must validate config step names against that
registry's keys, but must not import, depend on, or create that module. `validate_config` (and
`load_conference_rules`) therefore accept `known_step_names` as an explicit parameter. When the
caller omits it, `_resolve_known_steps()` tries `from artifacts.tiebreaker_steps import
STEP_REGISTRY` lazily (inside the function body, not at module import time) and falls back to
the module-level `KNOWN_STEPS` constant below -- logging a warning -- if that import fails.
Tests in tests/test_tiebreaker_rules.py always inject an explicit name set and never depend on
the sibling module's existence.

MIN_CONFERENCE_GAMES HAS NO SILENT DEFAULT (K6)
-------------------------------------------------
`external_ranking` substitutes this project's own continuous rating for every conference step
that cites an outside ranking service (SportSource Analytics Team Success Ranking / Rating
Score, CFP ranking, etc. -- see docs/conference-tiebreakers/source-rules/README.md). Because
this project's rating effectively never ties, an ungated `external_ranking` would resolve every
tie it reaches and pre-empt the overall-record fallback the product owner asked for (K6, AC5) --
most visibly in the thin-information early season, e.g. a 2-0 team and a 1-0 team with no
common opponents yet. So every `external_ranking` step MUST state `params.min_conference_games`
explicitly; the loader raises rather than silently defaulting it (this is what "make it
explicit in every config entry rather than implicit" means in practice). The recommended value,
used throughout the four Power 4 configs, is **4**: a defensible midpoint for an 8-9 game
conference slate -- enough games that a rating differential reflects the season rather than
small-sample noise (K5 reasons similarly, at half this threshold, for the much weaker
common-opponents signal), while still resolving ties well before the final week, when a
conference tie is most likely to matter.

TWO PARAMETERS THE ORIGINAL BRIEF DID NOT ANTICIPATE
--------------------------------------------------------
Reading all four primary-source files directly (not just the transcription table in the task
brief) surfaced two more per-step nuances that vary by conference and so must be config
parameters, not hard-coded step behaviour:

- `opponents_cumulative_conf_pct.ignore_opponent_count_mismatch` (bool): the Big Ten's text
  (Section A.5(a) / B.4(a)) explicitly says that on an unbalanced schedule (fewer than nine
  conference games), cumulative opponent conference win percentage is compared "REGARDLESS of
  how many conference opponents each team played". The SEC and Big 12 texts say nothing about
  this at all. Absent evidence a conference actually grants this exception, the default here is
  **False** for every conference except where a source file states it -- the Big Ten is the only
  one currently set to True. Silence in the other three texts is not evidence of the same
  behaviour; it is evidence we don't know, so the safe default is the stricter reading.

- `vs_placed_opponents.tied_opponent_handling` ("head_to_head_then_combine" | "combine") and
  `vs_placed_opponents.exhaust_all_opponents` (bool): the SEC (step C) explicitly resolves a tie
  among the best-placed common opponents by head-to-head first, falling back to combining their
  records only if head-to-head fails to separate them, and explicitly continues to the next
  common opponent by order of finish until all are exhausted. The Big 12 (two-team step c /
  multi-team step c) skips head-to-head entirely and goes straight to a combined "AS A GROUP"
  record. The Big Ten (A.4 / B.3) states neither. Given the Big Ten's phrasing ("proceeding
  through the common conference opponents based on their order of finish") most closely mirrors
  the Big 12's unqualified "proceeding through the standings" language -- and gives no textual
  basis for inventing a head-to-head detour the Big Ten never mentions -- this module's Big Ten
  config entries default `tied_opponent_handling` to "combine", matching the Big 12. This is
  documented here as a call, not a fact: T2's `vs_placed_opponents` implementation should accept
  both parameter names (`tied_opponent_handling`, `exhaust_all_opponents`) regardless of which
  KNOWN_STEPS entries currently exercise them.

THE `when` PREDICATE (ACC and SEC round-robin branch)
--------------------------------------------------------
The ACC's amended (2026-onward) three-or-more-team procedure branches on a condition -- whether
all Tied Teams are mutual common opponents (a complete round robin among the tied group) -- into
two genuinely different first comparisons: best intra-group record (`sub_group_record`) if it IS
a complete round robin, or a sweep-in/sweep-out test (`sweep_in_out`) if it is NOT. A flat
`steps[]` list cannot express a branch. This module adds an optional `when` key to the step
schema (`"round_robin_among_tied"` | `"not_round_robin_among_tied"`, absent = always applies),
validated against the same closed-set pattern as every other enum here. The driver (T3) skips a
step whose `when` does not match the group's actual shape, exactly as it already skips a step
that returns "no opinion". The SEC's step A (2.a "IS a complete round robin" / 2.b "is NOT") has
the identical shape and uses the same mechanism.

TWO SEMANTICS EXPRESSED VIA `multi_team.restart_at` / `multi_team.eliminated_teams_locked`
-----------------------------------------------------------------------------------------------
- Peel-off restart target: the SEC, Big 12 and Big Ten all restart a surviving sub-group at the
  beginning of the procedure appropriate to ITS size -- a pair restarts at `two_team` regardless
  of whether it is now contending for 2nd, 3rd, or any other place; a group of 3+ restarts at
  `multi_team`. `restart_at = "size_appropriate_restart"` names this single policy, which is
  sufficient because none of the four source texts make the restart target depend on anything
  BUT group size. The ACC's amended text is different in kind: it restarts "including the
  definition of tied teams" -- the tied group itself is re-derived, not merely re-entered at a
  step. `restart_at = "redefine_tied_teams"` names that.
- No re-entry: the Big Ten states outright that an eliminated team "SHALL NOT be pulled back
  into the tiebreaker for any future step" (Section B). The SEC and Big 12 describe teams being
  "removed from the tiebreaker" / "eliminated" without the Big Ten's explicit never-return
  language, but nothing in either text describes a removed team returning, so this module treats
  the invariant as universal and records it explicitly per rule set as
  `multi_team.eliminated_teams_locked` rather than leaving it implicit in the driver. The one
  exception is the ACC's amended entry: because its restart explicitly REDEFINES the tied-teams
  group each time, a team that was excluded from one iteration is not obviously barred from
  qualifying again under a redefinition, and the source text does not say either way (unlike the
  Big Ten's explicit lock). That entry sets `eliminated_teams_locked = False` and says so in
  `notes` -- flagged for T3/T4 review rather than guessed at.

`vs_placed_opponents`'s recursive sub-rule -- if the tied group's best-placed common opponents
are themselves tied, break that inner tie by head-to-head only, and failing that, COMBINE their
records -- is the same rule in all three sources that have it (SEC, explicit; the schema's
`tied_opponent_handling`/`exhaust_all_opponents` are what varies BETWEEN conferences, but no
source shows within-conference, tie-by-tie variation of this inner recursion), so it is fixed
behaviour inside the `vs_placed_opponents` primitive (T2's responsibility), not a config knob.

ACC TIE DEFINITION DIFFERS FROM PLAN K9
------------------------------------------
Plan K9 assumes the engine is only ever invoked on teams already equal on conference win
percentage. The ACC's own text (Section 1) contradicts this for the ACC specifically: its Tied
Teams are the team(s) with the best conference win percentage PLUS any team that played an
alternate number of conference games and has either the same number of conference WINS or the
same number of LOSSES as them -- so an ACC 6-2 team and a 6-3 team can be Tied Teams, broken
"starting with the highest win percentage, working downward" (Section 2.a). This module records
that as `RuleSet.tie_definition` ("win_pct", the default matching every other conference here,
or "acc_alternate_games" for both ACC entries). Building the actual tied-group membership from
this definition is explicitly OUT of scope here -- that is where `_sort_conference_teams` builds
groups (T4). This is a correction to K9 for the caller to carry forward, not implemented here.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger("cfb_lp")

# Path this module reads by default; callers (T4) may pass a different path, e.g. for tests.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "conference_tiebreakers.json"

SCHEMA_VERSION = 1

# --- Closed vocabularies, validated by exact membership everywhere they appear. ---------------
PROVENANCE_LEVELS: FrozenSet[str] = frozenset(
    {"primary_source", "verified_against_real_tie", "search_derived"}
)
RESTART_POLICIES: FrozenSet[str] = frozenset(
    {"size_appropriate_restart", "redefine_tied_teams"}
)
TIE_DEFINITIONS: FrozenSet[str] = frozenset(
    {"win_pct", "acc_alternate_games", "cusa_within_one_win"}
)
WHEN_PREDICATES: FrozenSet[str] = frozenset(
    {"round_robin_among_tied", "not_round_robin_among_tied"}
)
TIED_OPPONENT_HANDLINGS: FrozenSet[str] = frozenset(
    {"head_to_head_then_combine", "combine"}
)

# Per-step accepted parameter names (K2's fixed, small vocabulary, as handed down by the task
# brief -- see the module docstring for the two params added after reading the source files
# directly: opponents_cumulative_conf_pct.ignore_opponent_count_mismatch and
# vs_placed_opponents.{tied_opponent_handling,exhaust_all_opponents}).
STEP_PARAMS: Dict[str, FrozenSet[str]] = {
    "head_to_head": frozenset(),
    "sub_group_record": frozenset(),
    "sweep_in_out": frozenset({"sides"}),
    "common_opponents_record": frozenset({"min_sample", "scope"}),
    "vs_placed_opponents": frozenset(
        {"direction", "tied_opponent_handling", "exhaust_all_opponents",
         "advance_on_unequal_games", "standings_scope"}
    ),
    "opponents_cumulative_conf_pct": frozenset({"ignore_opponent_count_mismatch"}),
    "capped_relative_scoring_margin": frozenset({"offense_cap", "defense_floor"}),
    "total_wins_capped": frozenset({"max_games", "cap_fcs_wins"}),
    "external_ranking": frozenset({"min_conference_games"}),
    "random_draw": frozenset(),
    # The Mountain West / Sun Belt / American cascade: an outside ranking conditioned on the
    # final weekend's result. See artifacts/tiebreaker_steps.conditional_external_ranking for
    # what the K6 rating substitution costs here, which is more than it costs elsewhere.
    "conditional_external_ranking": frozenset(
        {"min_conference_games", "ranked_cutoff", "condition"}
    ),
    # Overall winning percentage in the three variants the American, Mountain West and Sun Belt
    # each ask for. Distinct from total_wins_capped, which is the Big 12's 12-game win COUNT.
    "overall_win_pct": frozenset({"fcs_win_cap", "fbs_only"}),
    # Sun Belt only -- the one conference of the ten that still plays divisions.
    "divisional_record": frozenset(),
}

# The set of step NAMES, used both for validation fallback and for the drift-detection test
# against artifacts.tiebreaker_steps.STEP_REGISTRY (set(STEP_REGISTRY) == KNOWN_STEPS).
KNOWN_STEPS: FrozenSet[str] = frozenset(STEP_PARAMS)

# Allowed keys at each level of the schema -- unknown keys are rejected (strict), not ignored.
TOP_LEVEL_KEYS: FrozenSet[str] = frozenset({"schema_version", "conferences"})
RULESET_KEYS: FrozenSet[str] = frozenset(
    {
        "season_min", "season_max", "provenance", "source_file", "notes",
        "two_team", "multi_team", "tie_definition",
    }
)
MULTI_TEAM_KEYS: FrozenSet[str] = frozenset(
    {"restart_at", "steps", "eliminated_teams_locked"}
)
STEP_KEYS: FrozenSet[str] = frozenset({"step", "params", "cites", "when"})


class TiebreakerConfigError(ValueError):
    """Raised for any malformed or invalid conference-tiebreaker config, at load time.

    The message always names the offending thing (a key, a step name, a conference, a season
    range) and its path in the document, so a reviewer or CI failure can find the bad line
    without re-deriving it from a generic "validation failed" message.
    """


@dataclass(frozen=True)
class Step:
    """One entry in a `two_team` or `multi_team.steps` list.

    Attributes:
        step: One of KNOWN_STEPS (or the registry's keys, when injected) -- the primitive to
            apply.
        params: Step-specific parameters, validated against STEP_PARAMS[step] at load time.
        cites: A short quotation or label pointing at the source-rules line that justifies this
            step. Required and non-empty when the owning rule set's provenance is
            "primary_source" or "verified_against_real_tie".
        when: Optional predicate gating this step to one branch of a conditional procedure (see
            WHEN_PREDICATES). None means "always applies".
    """

    step: str
    params: Dict[str, Any] = field(default_factory=dict)
    cites: str = ""
    when: Optional[str] = None


@dataclass(frozen=True)
class MultiTeamRules:
    """The multi-team (3+) tiebreaker procedure for one rule set.

    Attributes:
        restart_at: One of RESTART_POLICIES -- where a surviving sub-group re-enters the
            procedure after a peel-off.
        steps: Ordered steps applied to the tied group; see Step.when for conditional branches.
        eliminated_teams_locked: True if a team removed from the tie during peel-off can never
            re-enter on a later restart (see module docstring for the one conference where this
            is not asserted true).
    """

    restart_at: str
    steps: Tuple[Step, ...]
    eliminated_teams_locked: bool


@dataclass(frozen=True)
class RuleSet:
    """One conference's tiebreaker procedure for a season range.

    Attributes:
        season_min: First season this rule set governs, inclusive. None = open-ended (no known
            lower bound / applies to every season on record before season_max).
        season_max: Last season this rule set governs, inclusive. None = open-ended (still in
            effect).
        provenance: One of PROVENANCE_LEVELS.
        source_file: Repo-relative path to the source-rules file this was transcribed from.
        notes: Free-text human context -- what this era is, any caveat a reviewer needs. This is
            JSON's substitute for a YAML comment (see module docstring).
        two_team: The two-team tiebreaker procedure, applied top to bottom, no restart.
        multi_team: The multi-team (3+) tiebreaker procedure, including its restart policy.
        tie_definition: One of TIE_DEFINITIONS -- how the tied GROUP is defined in the first
            place. "win_pct" (the default, and correct for every conference here except the ACC)
            means "equal on conference win percentage", matching plan K9. "acc_alternate_games"
            flags that this conference's own text defines a wider group (see module docstring);
            building that group is out of this module's scope.
    """

    season_min: Optional[int]
    season_max: Optional[int]
    provenance: str
    source_file: str
    notes: str
    two_team: Tuple[Step, ...]
    multi_team: MultiTeamRules
    tie_definition: str = "win_pct"

    def covers(self, season: int) -> bool:
        """True if this rule set's season range includes `season` (both bounds inclusive)."""
        if self.season_min is not None and season < self.season_min:
            return False
        if self.season_max is not None and season > self.season_max:
            return False
        return True


@dataclass(frozen=True)
class TiebreakerConfig:
    """The fully validated config: schema_version plus every conference's rule sets."""

    schema_version: int
    conferences: Dict[str, Tuple[RuleSet, ...]]

    def rules_for(self, conference: str, season: int) -> Optional[RuleSet]:
        """Return the single RuleSet governing `conference` in `season`, or None.

        None means either the conference has no config at all, or none of its rule sets' season
        ranges cover this season -- both cases the caller (T4) should treat identically: fall
        back to current behaviour (AC7), never raise.
        """
        for rule_set in self.conferences.get(conference, ()):
            if rule_set.covers(season):
                return rule_set
        return None


def rules_for(config: TiebreakerConfig, conference: str, season: int) -> Optional[RuleSet]:
    """Module-level convenience wrapper around TiebreakerConfig.rules_for -- see there."""
    return config.rules_for(conference, season)


def _resolve_known_steps() -> FrozenSet[str]:
    """Lazily import artifacts.tiebreaker_steps.STEP_REGISTRY; fall back to KNOWN_STEPS.

    Imported inside the function body (never at module import time) so this module never fails
    to import, and never depends on, the sibling module a parallel agent owns. Logs a warning
    when the registry is unavailable so the fallback is visible in normal operation, not just in
    tests.
    """
    try:
        from artifacts.tiebreaker_steps import STEP_REGISTRY  # type: ignore
    except ImportError:
        logger.warning(
            "artifacts.tiebreaker_steps.STEP_REGISTRY unavailable; falling back to the "
            "built-in KNOWN_STEPS vocabulary (%d steps).", len(KNOWN_STEPS),
        )
        return KNOWN_STEPS
    return frozenset(STEP_REGISTRY)


def _reject_unknown_keys(item: Dict[str, Any], allowed: FrozenSet[str], path: str) -> None:
    """Raise if `item` has any key outside `allowed`, naming the first (sorted) offender."""
    unknown = sorted(set(item) - allowed)
    if unknown:
        raise TiebreakerConfigError(
            f"{path}: unknown key {unknown[0]!r} (allowed keys: {sorted(allowed)})"
        )


def _get_optional_int(item: Dict[str, Any], key: str, path: str) -> Optional[int]:
    value = item.get(key)
    if value is not None and not isinstance(value, int):
        raise TiebreakerConfigError(f"{path}.{key}: must be an integer or null, got {value!r}")
    return value


def _get_str(item: Dict[str, Any], key: str, path: str, default: str = "") -> str:
    value = item.get(key, default)
    if not isinstance(value, str):
        raise TiebreakerConfigError(f"{path}.{key}: must be a string, got {value!r}")
    return value


def _build_step(path: str, item: Any, known_step_names: FrozenSet[str]) -> Step:
    if not isinstance(item, dict):
        raise TiebreakerConfigError(f"{path}: expected an object, got {type(item).__name__}")
    _reject_unknown_keys(item, STEP_KEYS, path)

    step_name = item.get("step")
    if step_name not in known_step_names:
        raise TiebreakerConfigError(
            f"{path}.step: unknown step {step_name!r}; valid step names are "
            f"{sorted(known_step_names)}"
        )

    params = item.get("params", {})
    if not isinstance(params, dict):
        raise TiebreakerConfigError(f"{path}.params: must be an object, got {params!r}")
    allowed_params = STEP_PARAMS.get(step_name, frozenset())
    for key in sorted(params):
        if key not in allowed_params:
            raise TiebreakerConfigError(
                f"{path}.params.{key}: step {step_name!r} does not accept a {key!r} "
                f"parameter; accepted parameters are {sorted(allowed_params)}"
            )

    # K6: external_ranking's gate must be explicit, never silently defaulted (see module
    # docstring's MIN_CONFERENCE_GAMES section).
    if step_name == "external_ranking" and "min_conference_games" not in params:
        raise TiebreakerConfigError(
            f"{path}.params.min_conference_games: required for step 'external_ranking' -- "
            "must be stated explicitly (K6), not left to a silent default"
        )

    if step_name == "vs_placed_opponents":
        direction = params.get("direction")
        if direction is not None and direction != "descending":
            raise TiebreakerConfigError(
                f"{path}.params.direction: only 'descending' is supported, got {direction!r}"
            )
        tied_opponent_handling = params.get("tied_opponent_handling")
        if (
            tied_opponent_handling is not None
            and tied_opponent_handling not in TIED_OPPONENT_HANDLINGS
        ):
            raise TiebreakerConfigError(
                f"{path}.params.tied_opponent_handling: {tied_opponent_handling!r} is not "
                f"one of {sorted(TIED_OPPONENT_HANDLINGS)}"
            )

    cites = _get_str(item, "cites", path)

    when = item.get("when")
    if when is not None and when not in WHEN_PREDICATES:
        raise TiebreakerConfigError(
            f"{path}.when: {when!r} is not one of {sorted(WHEN_PREDICATES)} (or absent)"
        )

    return Step(step=step_name, params=dict(params), cites=cites, when=when)


def _build_multi_team(
    path: str, item: Any, known_step_names: FrozenSet[str]
) -> MultiTeamRules:
    if not isinstance(item, dict):
        raise TiebreakerConfigError(f"{path}: expected an object, got {type(item).__name__}")
    _reject_unknown_keys(item, MULTI_TEAM_KEYS, path)

    restart_at = item.get("restart_at")
    if restart_at not in RESTART_POLICIES:
        raise TiebreakerConfigError(
            f"{path}.restart_at: {restart_at!r} is not one of {sorted(RESTART_POLICIES)}"
        )

    steps_raw = item.get("steps")
    if not isinstance(steps_raw, list) or len(steps_raw) == 0:
        raise TiebreakerConfigError(f"{path}.steps: must be a non-empty list of steps")
    steps = tuple(
        _build_step(f"{path}.steps[{i}]", s, known_step_names)
        for i, s in enumerate(steps_raw)
    )

    eliminated_teams_locked = item.get("eliminated_teams_locked")
    if not isinstance(eliminated_teams_locked, bool):
        raise TiebreakerConfigError(
            f"{path}.eliminated_teams_locked: required and must be a boolean, "
            f"got {eliminated_teams_locked!r}"
        )

    return MultiTeamRules(
        restart_at=restart_at, steps=steps, eliminated_teams_locked=eliminated_teams_locked
    )


def _build_rule_set(
    conference: str, index: int, item: Any, known_step_names: FrozenSet[str]
) -> RuleSet:
    path = f"conferences.{conference}[{index}]"
    if not isinstance(item, dict):
        raise TiebreakerConfigError(f"{path}: expected an object, got {type(item).__name__}")
    _reject_unknown_keys(item, RULESET_KEYS, path)

    season_min = _get_optional_int(item, "season_min", path)
    season_max = _get_optional_int(item, "season_max", path)
    if season_min is not None and season_max is not None and season_min > season_max:
        raise TiebreakerConfigError(
            f"{path}: season_min ({season_min}) > season_max ({season_max})"
        )

    provenance = item.get("provenance")
    if provenance not in PROVENANCE_LEVELS:
        raise TiebreakerConfigError(
            f"{path}.provenance: {provenance!r} is not one of {sorted(PROVENANCE_LEVELS)}"
        )

    source_file = _get_str(item, "source_file", path)
    notes = _get_str(item, "notes", path)

    tie_definition = item.get("tie_definition", "win_pct")
    if tie_definition not in TIE_DEFINITIONS:
        raise TiebreakerConfigError(
            f"{path}.tie_definition: {tie_definition!r} is not one of "
            f"{sorted(TIE_DEFINITIONS)}"
        )

    two_team_raw = item.get("two_team")
    if not isinstance(two_team_raw, list) or len(two_team_raw) == 0:
        raise TiebreakerConfigError(f"{path}.two_team: must be a non-empty list of steps")
    two_team = tuple(
        _build_step(f"{path}.two_team[{i}]", s, known_step_names)
        for i, s in enumerate(two_team_raw)
    )

    multi_team_raw = item.get("multi_team")
    multi_team = _build_multi_team(f"{path}.multi_team", multi_team_raw, known_step_names)

    require_cites = provenance in {"primary_source", "verified_against_real_tie"}
    if require_cites:
        for i, step in enumerate(two_team):
            if not step.cites:
                raise TiebreakerConfigError(
                    f"{path}.two_team[{i}].cites: required and non-empty for "
                    f"provenance={provenance!r}"
                )
        for i, step in enumerate(multi_team.steps):
            if not step.cites:
                raise TiebreakerConfigError(
                    f"{path}.multi_team.steps[{i}].cites: required and non-empty for "
                    f"provenance={provenance!r}"
                )

    return RuleSet(
        season_min=season_min,
        season_max=season_max,
        provenance=provenance,
        source_file=source_file,
        notes=notes,
        two_team=two_team,
        multi_team=multi_team,
        tie_definition=tie_definition,
    )


def _fmt_range(rule_set: RuleSet) -> str:
    lo = "-inf" if rule_set.season_min is None else str(rule_set.season_min)
    hi = "+inf" if rule_set.season_max is None else str(rule_set.season_max)
    return f"[{lo}, {hi}]"


def _check_no_overlap(conference: str, rule_sets: List[RuleSet]) -> None:
    """Raise if any two of this conference's rule sets cover a common season.

    This is the bug that would silently apply the wrong ACC era (or any conference's wrong era)
    -- it gets its own explicit check, independent of every other validation, per the task brief.
    """
    neg_inf, pos_inf = float("-inf"), float("inf")
    intervals = [
        (
            rule_set.season_min if rule_set.season_min is not None else neg_inf,
            rule_set.season_max if rule_set.season_max is not None else pos_inf,
            i,
        )
        for i, rule_set in enumerate(rule_sets)
    ]
    intervals.sort()
    for (lo1, hi1, i1), (lo2, hi2, i2) in zip(intervals, intervals[1:]):
        if lo2 <= hi1:
            raise TiebreakerConfigError(
                f"conferences.{conference}: rule sets {i1} and {i2} have overlapping season "
                f"ranges ({_fmt_range(rule_sets[i1])} overlaps {_fmt_range(rule_sets[i2])})"
            )


def validate_config(
    raw: Any, known_step_names: Optional[FrozenSet[str]] = None
) -> TiebreakerConfig:
    """Strictly validate an already-parsed config document and build a TiebreakerConfig.

    Args:
        raw: The parsed JSON document (a dict), e.g. from `json.loads`.
        known_step_names: The set of valid step names to validate `step` fields against. Tests
            should always pass this explicitly. Production callers may omit it, in which case
            `_resolve_known_steps()` supplies it (see there).
    Returns:
        TiebreakerConfig: the validated, immutable config.
    Raises:
        TiebreakerConfigError: for any schema violation, naming the offending key/value and its
            path in the document.
    """
    if known_step_names is None:
        known_step_names = _resolve_known_steps()

    if not isinstance(raw, dict):
        raise TiebreakerConfigError(f"<root>: expected an object, got {type(raw).__name__}")
    _reject_unknown_keys(raw, TOP_LEVEL_KEYS, "<root>")

    schema_version = raw.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise TiebreakerConfigError(
            f"<root>.schema_version: expected {SCHEMA_VERSION}, got {schema_version!r}"
        )

    conferences_raw = raw.get("conferences")
    if not isinstance(conferences_raw, dict):
        raise TiebreakerConfigError(
            f"<root>.conferences: must be an object, got {type(conferences_raw).__name__}"
        )

    conferences: Dict[str, Tuple[RuleSet, ...]] = {}
    for conference, rule_sets_raw in conferences_raw.items():
        if not isinstance(rule_sets_raw, list) or len(rule_sets_raw) == 0:
            raise TiebreakerConfigError(
                f"conferences.{conference}: must be a non-empty list of rule sets"
            )
        rule_sets = [
            _build_rule_set(conference, i, item, known_step_names)
            for i, item in enumerate(rule_sets_raw)
        ]
        _check_no_overlap(conference, rule_sets)
        conferences[conference] = tuple(rule_sets)

    return TiebreakerConfig(schema_version=schema_version, conferences=conferences)


def load_conference_rules(
    path: "str | Path" = DEFAULT_CONFIG_PATH,
    known_step_names: Optional[FrozenSet[str]] = None,
) -> TiebreakerConfig:
    """Load, parse and validate the conference tiebreaker config from `path`.

    Args:
        path: Path to the JSON config file. Defaults to
            artifacts/conference_tiebreakers.json next to this module.
        known_step_names: Passed through to `validate_config` -- see there.
    Returns:
        TiebreakerConfig: the validated, immutable config.
    Raises:
        TiebreakerConfigError: the file is missing, is not valid JSON, or fails schema
            validation -- always naming the offending thing.
    """
    path = Path(path)
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise TiebreakerConfigError(f"{path}: config file not found") from None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TiebreakerConfigError(f"{path}: invalid JSON ({exc})") from exc
    return validate_config(raw, known_step_names)
