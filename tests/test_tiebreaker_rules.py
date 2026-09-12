"""Tests for artifacts/tiebreaker_rules.py: the conference-tiebreaker config schema, loader and
load-time validator (docs/conference-tiebreakers/plan.yaml task T1).

Every validation failure the loader is meant to catch gets its own test asserting the error
message names the offending thing -- a key, a step, a conference, a season range -- not just
that *some* error was raised. `validate_config` is exercised directly against small, hand-built
dicts with an explicit `known_step_names` set, per the task's dependency-injection requirement:
these tests never import or depend on artifacts.tiebreaker_steps (a parallel agent's file).

Also covers:
  - Every config conference key is a raw `teams.conference` value (CONFERENCE_DISPLAY_NAMES
    membership), so a typo like "Big10" fails the suite.
  - The real artifacts/conference_tiebreakers.json loads cleanly and the ACC resolves to its
    pre-2026 vs. 2026-amended rule set correctly (AC9).
  - The registry-drift test: IF artifacts.tiebreaker_steps is importable, its STEP_REGISTRY keys
    must equal KNOWN_STEPS exactly -- skipped via pytest.importorskip when that module does not
    exist yet.

No network, no database. Run: python -m pytest tests/ -q
"""
import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from artifacts.rankings import CONFERENCE_DISPLAY_NAMES  # noqa: E402
from artifacts.tiebreaker_rules import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    KNOWN_STEPS,
    TiebreakerConfigError,
    load_conference_rules,
    rules_for,
    validate_config,
)

# A small, self-contained known-step set for tests that don't care about the full vocabulary.
_TEST_KNOWN_STEPS = frozenset({"head_to_head", "external_ranking", "random_draw"})


def _step(step="head_to_head", params=None, cites="cited line", when=None):
    d = {"step": step, "params": params if params is not None else {}, "cites": cites}
    if when is not None:
        d["when"] = when
    return d


def _rule_set(**overrides):
    """A minimal, valid rule set dict. Callers mutate the returned dict for negative tests."""
    rs = {
        "season_min": 2024,
        "season_max": None,
        "provenance": "primary_source",
        "source_file": "docs/conference-tiebreakers/source-rules/sec.txt",
        "notes": "test fixture",
        "tie_definition": "win_pct",
        "two_team": [_step(cites="two-team step A")],
        "multi_team": {
            "restart_at": "size_appropriate_restart",
            "eliminated_teams_locked": True,
            "steps": [_step(cites="multi-team step A")],
        },
    }
    rs.update(overrides)
    return rs


def _config(conferences=None):
    return {
        "schema_version": 1,
        "conferences": conferences if conferences is not None else {"SEC": [_rule_set()]},
    }


def _validate(cfg, known=_TEST_KNOWN_STEPS):
    return validate_config(cfg, known_step_names=known)


# --- Conference keys must be raw teams.conference values --------------------------------------

def test_all_config_conference_keys_are_raw_conference_values():
    """A typo like 'Big10' (instead of 'Big 12'/'Big Ten') must fail this test, per the task
    brief -- config keys are the raw `teams.conference` strings (CONFERENCE_DISPLAY_NAMES keys),
    not the display abbreviations."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    for conference in cfg.conferences:
        assert conference in CONFERENCE_DISPLAY_NAMES, (
            f"{conference!r} is not a raw conference value from CONFERENCE_DISPLAY_NAMES"
        )


def test_real_config_covers_the_power_four_and_never_the_independents():
    """Written to stay true as Group of 6 conferences are transcribed, rather than pinning an
    exact set that every addition would have to edit. Two things must hold permanently: the four
    Power 4 conferences are configured, and FBS Independents never is -- it is not a conference,
    has no championship game, and compute_team_records sets its conf_wins to None, so a rule set
    for it could never be reached."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    configured = set(cfg.conferences)
    assert {"SEC", "Big 12", "Big Ten", "ACC"} <= configured
    assert "FBS Independents" not in configured


# --- rules_for / ACC era lookup (AC9) ----------------------------------------------------------

def test_acc_resolves_to_pre_amendment_set_for_2025():
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    rs = rules_for(cfg, "ACC", 2025)
    assert rs is not None
    assert rs.provenance == "search_derived"
    assert rs.season_max == 2025


def test_acc_resolves_to_amended_set_for_2026():
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    rs = rules_for(cfg, "ACC", 2026)
    assert rs is not None
    assert rs.provenance == "primary_source"
    assert rs.season_min == 2026
    assert rs.tie_definition == "acc_alternate_games"


def test_rules_for_unknown_conference_returns_none():
    """No config for a conference -> None, so the caller falls back to current behaviour
    (AC7) -- never raises."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    # A name that is not a conference at all, so this stays true no matter how many real
    # conferences get transcribed. Using a real-but-unconfigured conference here made the test
    # silently stop testing anything the moment that conference gained a rule set.
    assert rules_for(cfg, "Not A Conference", 2025) is None
    # And a real conference asked about a season outside every era it has.
    assert rules_for(cfg, "SEC", 1999) is None


def test_rules_for_season_outside_every_range_returns_none():
    cfg = _config({"SEC": [_rule_set(season_min=2024, season_max=2025)]})
    validated = _validate(cfg)
    assert validated.rules_for("SEC", 2026) is None
    assert validated.rules_for("SEC", 2023) is None


# --- Unknown step name --------------------------------------------------------------------------

def test_unknown_step_name_fails_naming_the_step_and_valid_names():
    cfg = _config({"SEC": [_rule_set(two_team=[_step(step="frobnicate")])]})
    with pytest.raises(TiebreakerConfigError) as exc:
        _validate(cfg)
    msg = str(exc.value)
    assert "frobnicate" in msg
    for name in sorted(_TEST_KNOWN_STEPS):
        assert name in msg


# --- Unknown key anywhere (strict) ---------------------------------------------------------------

def test_unknown_top_level_key_is_rejected():
    cfg = _config()
    cfg["bogus_top_level"] = True
    with pytest.raises(TiebreakerConfigError, match=r"bogus_top_level"):
        _validate(cfg)


def test_unknown_ruleset_key_is_rejected():
    cfg = _config({"SEC": [_rule_set(extra_field="nope")]})
    with pytest.raises(TiebreakerConfigError, match=r"extra_field"):
        _validate(cfg)


def test_unknown_multi_team_key_is_rejected():
    rs = _rule_set()
    rs["multi_team"]["bogus"] = 1
    cfg = _config({"SEC": [rs]})
    with pytest.raises(TiebreakerConfigError, match=r"bogus"):
        _validate(cfg)


def test_unknown_step_level_key_is_rejected():
    cfg = _config({"SEC": [_rule_set(two_team=[{"step": "head_to_head", "unexpected": 1}])]})
    with pytest.raises(TiebreakerConfigError, match=r"unexpected"):
        _validate(cfg)


# --- provenance enum -----------------------------------------------------------------------------

def test_invalid_provenance_is_rejected():
    cfg = _config({"SEC": [_rule_set(provenance="vibes")]})
    with pytest.raises(TiebreakerConfigError, match=r"vibes"):
        _validate(cfg)


# --- season_min > season_max ----------------------------------------------------------------------

def test_season_min_greater_than_season_max_is_rejected():
    cfg = _config({"SEC": [_rule_set(season_min=2026, season_max=2024)]})
    with pytest.raises(TiebreakerConfigError, match=r"2026.*2024|season_min"):
        _validate(cfg)


# --- Overlapping season ranges (its own explicit check) --------------------------------------------

def test_overlapping_season_ranges_same_conference_rejected():
    cfg = _config(
        {
            "SEC": [
                _rule_set(season_min=2020, season_max=2025),
                _rule_set(season_min=2024, season_max=None),
            ]
        }
    )
    with pytest.raises(TiebreakerConfigError) as exc:
        _validate(cfg)
    msg = str(exc.value)
    assert "SEC" in msg
    assert "overlap" in msg.lower()


def test_adjacent_non_overlapping_season_ranges_are_accepted():
    """2025/2026 boundary case: season_max=2025 then season_min=2026 must NOT be flagged as an
    overlap -- this is exactly the real ACC boundary."""
    cfg = _config(
        {
            "SEC": [
                _rule_set(season_min=None, season_max=2025),
                _rule_set(season_min=2026, season_max=None),
            ]
        }
    )
    validated = _validate(cfg)
    assert len(validated.conferences["SEC"]) == 2


# --- empty two_team / multi_team.steps ---------------------------------------------------------

def test_empty_two_team_is_rejected():
    cfg = _config({"SEC": [_rule_set(two_team=[])]})
    with pytest.raises(TiebreakerConfigError, match=r"two_team"):
        _validate(cfg)


def test_empty_multi_team_steps_is_rejected():
    rs = _rule_set()
    rs["multi_team"]["steps"] = []
    cfg = _config({"SEC": [rs]})
    with pytest.raises(TiebreakerConfigError, match=r"steps"):
        _validate(cfg)


# --- restart_at enum -----------------------------------------------------------------------------

def test_invalid_restart_at_is_rejected():
    rs = _rule_set()
    rs["multi_team"]["restart_at"] = "just_wing_it"
    cfg = _config({"SEC": [rs]})
    with pytest.raises(TiebreakerConfigError, match=r"just_wing_it"):
        _validate(cfg)


# --- missing/empty cites on primary-source rule set -----------------------------------------------

def test_missing_cites_on_primary_source_ruleset_rejected():
    cfg = _config(
        {"SEC": [_rule_set(two_team=[_step(cites="")], provenance="primary_source")]}
    )
    with pytest.raises(TiebreakerConfigError, match=r"cites"):
        _validate(cfg)


def test_missing_cites_on_search_derived_ruleset_is_allowed():
    """search_derived is weaker evidence by design -- cites is not mandatory there."""
    cfg = _config(
        {"SEC": [_rule_set(two_team=[_step(cites="")], provenance="search_derived")]}
    )
    validated = _validate(cfg)
    assert validated.conferences["SEC"][0].provenance == "search_derived"


# --- params key the named step does not accept ------------------------------------------------

def test_unaccepted_param_name_is_rejected():
    cfg = _config(
        {"SEC": [_rule_set(two_team=[_step(step="head_to_head", params={"min_sample": 2})])]}
    )
    with pytest.raises(TiebreakerConfigError, match=r"min_sample"):
        _validate(cfg)


def test_accepted_param_name_passes():
    known = _TEST_KNOWN_STEPS | {"common_opponents_record"}
    cfg = _config(
        {
            "SEC": [
                _rule_set(
                    two_team=[
                        _step(step="common_opponents_record", params={"min_sample": 2})
                    ]
                )
            ]
        }
    )
    validated = _validate(cfg, known=known)
    assert validated.conferences["SEC"][0].two_team[0].params == {"min_sample": 2}


# --- external_ranking requires an explicit min_conference_games (K6) ----------------------------

def test_external_ranking_without_min_conference_games_is_rejected():
    cfg = _config({"SEC": [_rule_set(two_team=[_step(step="external_ranking", params={})])]})
    with pytest.raises(TiebreakerConfigError, match=r"min_conference_games"):
        _validate(cfg)


def test_external_ranking_with_min_conference_games_passes():
    cfg = _config(
        {
            "SEC": [
                _rule_set(
                    two_team=[
                        _step(step="external_ranking", params={"min_conference_games": 4})
                    ]
                )
            ]
        }
    )
    validated = _validate(cfg)
    assert validated.conferences["SEC"][0].two_team[0].params["min_conference_games"] == 4


# --- when predicate (round-robin branch) ---------------------------------------------------------

def test_invalid_when_predicate_is_rejected():
    cfg = _config({"SEC": [_rule_set(two_team=[_step(when="on_a_full_moon")])]})
    with pytest.raises(TiebreakerConfigError, match=r"on_a_full_moon"):
        _validate(cfg)


def test_valid_when_predicate_passes():
    cfg = _config({"SEC": [_rule_set(two_team=[_step(when="round_robin_among_tied")])]})
    validated = _validate(cfg)
    assert validated.conferences["SEC"][0].two_team[0].when == "round_robin_among_tied"


def test_absent_when_means_always():
    cfg = _config({"SEC": [_rule_set(two_team=[_step()])]})
    validated = _validate(cfg)
    assert validated.conferences["SEC"][0].two_team[0].when is None


# --- tie_definition enum --------------------------------------------------------------------------

def test_invalid_tie_definition_is_rejected():
    cfg = _config({"SEC": [_rule_set(tie_definition="whatever_feels_right")]})
    with pytest.raises(TiebreakerConfigError, match=r"whatever_feels_right"):
        _validate(cfg)


# A non-default tie definition is a claim that the conference's own text defines "tied" as
# something other than equal conference win percentage. Three conferences do, and they disagree
# with each other, so each is listed explicitly here: adding a row is a deliberate act, and
# anything NOT listed must stay at the K9-compatible default.
EXPECTED_NON_DEFAULT_TIE_DEFINITIONS = {
    # acc.txt section 1: best win percentage, PLUS any team on an alternate number of conference
    # games with the same number of wins OR the same number of losses. 2026 entry only.
    ("ACC", 2026): "acc_alternate_games",
    # cusa.txt section C: unequal conference games played, WITHIN ONE conference win of the
    # leader, AND an equal number of losses.
    ("Conference USA", 2024): "cusa_within_one_win",
}


def test_non_default_tie_definitions_appear_only_where_the_source_states_one():
    """K9 correction, generalised. Plan K9 assumed the engine is only ever handed teams equal on
    conference win percentage; three conferences define "tied" differently, and each definition
    is its own rule. This pins the non-default values to exactly the eras whose text supports
    them, so one cannot spread to a conference by copy-paste."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    for conference, rule_sets in cfg.conferences.items():
        for rs in rule_sets:
            expected = EXPECTED_NON_DEFAULT_TIE_DEFINITIONS.get(
                (conference, rs.season_min), "win_pct"
            )
            assert rs.tie_definition == expected, (
                f"{conference} {rs.season_min}-{rs.season_max}: expected {expected!r}, "
                f"got {rs.tie_definition!r}"
            )


def test_the_acc_and_cusa_definitions_are_actually_distinct_values():
    """They are different rules and must not collapse to one value -- the ACC's is wins-or-losses,
    CUSA's is within-one-win-and-equal-losses."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    acc = rules_for(cfg, "ACC", 2026).tie_definition
    cusa = rules_for(cfg, "Conference USA", 2025).tie_definition
    assert acc == "acc_alternate_games"
    assert cusa == "cusa_within_one_win"
    assert acc != cusa


def test_tie_definition_defaults_to_win_pct_when_absent():
    rs = _rule_set()
    del rs["tie_definition"]
    cfg = _config({"SEC": [rs]})
    validated = _validate(cfg)
    assert validated.conferences["SEC"][0].tie_definition == "win_pct"


# --- eliminated_teams_locked required boolean -----------------------------------------------------

def test_missing_eliminated_teams_locked_is_rejected():
    rs = _rule_set()
    del rs["multi_team"]["eliminated_teams_locked"]
    cfg = _config({"SEC": [rs]})
    with pytest.raises(TiebreakerConfigError, match=r"eliminated_teams_locked"):
        _validate(cfg)


def test_non_boolean_eliminated_teams_locked_is_rejected():
    rs = _rule_set()
    rs["multi_team"]["eliminated_teams_locked"] = "true"
    cfg = _config({"SEC": [rs]})
    with pytest.raises(TiebreakerConfigError, match=r"eliminated_teams_locked"):
        _validate(cfg)


# --- vs_placed_opponents's two conference-varying params -------------------------------------------

def test_vs_placed_opponents_invalid_direction_is_rejected():
    known = _TEST_KNOWN_STEPS | {"vs_placed_opponents"}
    cfg = _config(
        {
            "SEC": [
                _rule_set(
                    two_team=[
                        _step(step="vs_placed_opponents", params={"direction": "ascending"})
                    ]
                )
            ]
        }
    )
    with pytest.raises(TiebreakerConfigError, match=r"ascending"):
        _validate(cfg, known=known)


def test_vs_placed_opponents_invalid_tied_opponent_handling_is_rejected():
    known = _TEST_KNOWN_STEPS | {"vs_placed_opponents"}
    cfg = _config(
        {
            "SEC": [
                _rule_set(
                    two_team=[
                        _step(
                            step="vs_placed_opponents",
                            params={"tied_opponent_handling": "flip_a_coin"},
                        )
                    ]
                )
            ]
        }
    )
    with pytest.raises(TiebreakerConfigError, match=r"flip_a_coin"):
        _validate(cfg, known=known)


def test_vs_placed_opponents_valid_params_pass():
    known = _TEST_KNOWN_STEPS | {"vs_placed_opponents"}
    cfg = _config(
        {
            "SEC": [
                _rule_set(
                    two_team=[
                        _step(
                            step="vs_placed_opponents",
                            params={
                                "direction": "descending",
                                "tied_opponent_handling": "head_to_head_then_combine",
                                "exhaust_all_opponents": True,
                            },
                        )
                    ]
                )
            ]
        }
    )
    validated = _validate(cfg, known=known)
    assert validated.conferences["SEC"][0].two_team[0].params["tied_opponent_handling"] == (
        "head_to_head_then_combine"
    )


# --- root / structural shape errors ------------------------------------------------------------

def test_wrong_schema_version_is_rejected():
    cfg = _config()
    cfg["schema_version"] = 2
    with pytest.raises(TiebreakerConfigError, match=r"schema_version"):
        _validate(cfg)


def test_non_dict_root_is_rejected():
    with pytest.raises(TiebreakerConfigError, match=r"<root>"):
        _validate(["not", "a", "dict"])


def test_missing_file_raises_naming_the_path():
    missing = REPO_ROOT / "artifacts" / "does_not_exist_tiebreakers.json"
    with pytest.raises(TiebreakerConfigError, match=r"does_not_exist_tiebreakers\.json"):
        load_conference_rules(path=missing, known_step_names=KNOWN_STEPS)


def test_invalid_json_raises_naming_the_path(tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text("{not valid json")
    with pytest.raises(TiebreakerConfigError, match=r"broken\.json"):
        load_conference_rules(path=bad, known_step_names=KNOWN_STEPS)


# --- Fixture isolation sanity: mutating one rule set dict must not bleed into another -----------

def test_rule_set_helper_returns_independent_copies():
    a = _rule_set()
    b = _rule_set()
    a["multi_team"]["steps"][0]["cites"] = "mutated"
    assert b["multi_team"]["steps"][0]["cites"] == "multi-team step A"


def test_deepcopy_of_real_config_still_validates():
    """Sanity check that the real config isn't accidentally relying on shared mutable state
    between rule sets (e.g. two conferences pointing at the same dict)."""
    import json

    raw = json.loads(DEFAULT_CONFIG_PATH.read_text())
    raw_copy = copy.deepcopy(raw)
    validate_config(raw_copy, known_step_names=KNOWN_STEPS)


# --- Registry-drift test: catches KNOWN_STEPS and STEP_REGISTRY drifting apart ------------------

def test_known_steps_matches_step_registry_if_present():
    """If artifacts.tiebreaker_steps exists (a parallel agent's file, T2), its STEP_REGISTRY
    keys must equal this module's KNOWN_STEPS exactly. Skipped while that module doesn't exist
    yet -- this is the test that catches the two halves drifting apart once both land."""
    tiebreaker_steps = pytest.importorskip("artifacts.tiebreaker_steps")
    assert set(tiebreaker_steps.STEP_REGISTRY) == KNOWN_STEPS


# --- The round-robin / sweep pair, per source text ---------------------------------------------

def test_every_multi_team_chain_opens_with_a_round_robin_gated_intra_group_record():
    """Universal across all ten documents: the first multi-team step is an intra-group record
    measure, and it applies only when the tied teams all played one another.

    This exists because the first transcription collapsed the Big Ten's and Big 12's opening step
    to `sweep_in_out` alone. Both texts head that step "Winning percentage in games among the
    tied teams" and then qualify their sub-clauses with "If all teams involved did not play each
    other" (bigten.txt:43-48, big12.txt:38-43). Encoding only the sub-clauses silently deletes the
    complete-round-robin case, where the rule is to rank by record within the group -- a tie
    resolving differently from what the conference published, with nothing to show it happened.

    The gate matters independently: mountainwest.txt's multi step 1 says a partially-played group
    "SHALL REMAIN TIED", which is why the real 2025 four-way tie was not decided on UNLV's 0-2
    intra-group record. An ungated intra-group ranking would have placed UNLV last.
    """
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    for conference, rule_sets in cfg.conferences.items():
        for rs in rule_sets:
            steps = rs.multi_team.steps
            era = f"{conference} {rs.season_min}-{rs.season_max}"
            assert steps[0].step == "sub_group_record", era
            assert steps[0].when == "round_robin_among_tied", era


# Whether a conference publishes a sweep clause at all is a claim about its source text, so the
# answer is listed rather than inferred from the config it is meant to check. Five documents
# describe a lone sweeper advancing; sec.txt and acc.txt also demote a total-loser; pac12.txt and
# cusa.txt state no sweep clause of any kind.
CONFERENCES_WITH_A_SWEEP_STEP = {
    "SEC", "ACC", "Big 12", "Big Ten", "Mid-American",
    "Mountain West",     # mountainwest.txt multi 1: "unless one team defeated all other tied teams"
    "American Athletic", # american.txt 10.6.3: the same escape, identical in force
}
CONFERENCES_WITHOUT_A_SWEEP_STEP = {
    "Pac-12",            # pac12.txt multi 1: "the process moves to the next criterion"
    "Conference USA",    # cusa.txt states one chain and no sweep clause anywhere
    "Sun Belt",          # sunbelt.txt multi 1 states the measure with no sweep qualification
}


def test_a_sweep_step_is_present_exactly_where_the_source_publishes_one():
    """Guards both directions. Omitting a sweep a conference does publish silently deletes a rule;
    ADDING one a conference does not publish invents a rule that can eliminate a real team from a
    tie. The shape assertions in the next test cannot catch the second case, because an invented
    step would sit in exactly the right position."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    for conference, rule_sets in cfg.conferences.items():
        for rs in rule_sets:
            era = f"{conference} {rs.season_min}-{rs.season_max}"
            has_sweep = any(st.step == "sweep_in_out" for st in rs.multi_team.steps)
            if conference in CONFERENCES_WITH_A_SWEEP_STEP:
                assert has_sweep, f"{era}: source publishes a sweep clause but the config omits it"
            elif conference in CONFERENCES_WITHOUT_A_SWEEP_STEP:
                assert not has_sweep, (
                    f"{era}: source states NO sweep clause, but the config has one -- that "
                    f"invents a rule which can eliminate a team from a tie"
                )
            else:
                raise AssertionError(
                    f"{conference} is in neither sweep list; add it to one after reading its "
                    f"source file, rather than leaving the question unasked"
                )


def test_a_sweep_step_when_present_is_second_and_gated_to_the_other_branch():
    """Whether a sweep step exists at all is per-conference, because the texts give THREE
    readings: the SEC and ACC promote a sweeper and demote a total-loser; the Big 12, Big Ten and
    MAC publish only the promote half; the Pac-12 and CUSA state no sweep clause whatsoever, so
    their configs correctly omit the step rather than inventing it.

    What is invariant is the SHAPE when one is present: immediately after the round-robin step,
    gated to the complementary branch. Otherwise the two steps could both fire, or neither."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    seen_with, seen_without = [], []
    for conference, rule_sets in cfg.conferences.items():
        for rs in rule_sets:
            steps = rs.multi_team.steps
            era = f"{conference} {rs.season_min}-{rs.season_max}"
            sweeps = [i for i, st in enumerate(steps) if st.step == "sweep_in_out"]
            if not sweeps:
                seen_without.append(era)
                continue
            assert len(sweeps) == 1, f"{era}: more than one sweep step"
            assert sweeps[0] == 1, f"{era}: sweep step is at index {sweeps[0]}, expected 1"
            assert steps[1].when == "not_round_robin_among_tied", era
            seen_with.append(era)
    # Both readings must actually be represented, or this test is vacuous in one direction.
    assert seen_with, "no conference configures a sweep step; the assertions above never ran"
    assert seen_without, "no conference omits the sweep step; the omission path is untested"


# Which wording each conference uses for its final-weekend condition is a claim about its source
# text, so the answer is listed rather than inferred from the config it checks. The distinction is
# not cosmetic: a team IDLE in the final weekend satisfies "does not lose" and fails "wins", so
# the two settings order such a team differently.
EXPECTED_FINAL_WEEK_CONDITIONS = {
    # mountainwest.txt 2(a)/(b): "and WINS on the final weekend of the regular season".
    "Mountain West": "wins",
    # american.txt 10.5.3 / 10.5.5 / 10.6.4: "and DOESN'T LOSE in the final weekend".
    "American Athletic": "does_not_lose",
    # sunbelt.txt 5-8: "and WINS in the final weekend of the Conference regular season".
    "Sun Belt": "wins",
}


def test_final_week_condition_matches_each_conference_s_own_wording():
    """The Mountain West says "wins" and the American says "doesn't lose", and both are direct
    quotations. Defaulting either to the other's wording would order a team that was idle in the
    final weekend by a rule its conference did not write."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    checked = []
    for conference, rule_sets in cfg.conferences.items():
        expected = EXPECTED_FINAL_WEEK_CONDITIONS.get(conference)
        for rs in rule_sets:
            era = f"{conference} {rs.season_min}-{rs.season_max}"
            for st in list(rs.two_team) + list(rs.multi_team.steps):
                if st.step != "conditional_external_ranking":
                    continue
                assert expected is not None, (
                    f"{era} uses conditional_external_ranking but is in neither wording list; "
                    f"add it after reading its source file"
                )
                assert st.params.get("condition") == expected, (
                    f"{era}: expected condition {expected!r}, got "
                    f"{st.params.get('condition')!r}"
                )
                checked.append(era)
    assert checked, "no conference configures the cascade; this test never ran"
    # Both wordings must actually appear, or one of them is untested.
    conditions = {EXPECTED_FINAL_WEEK_CONDITIONS[c] for c in EXPECTED_FINAL_WEEK_CONDITIONS}
    assert conditions == {"wins", "does_not_lose"}


# The Sun Belt is the only one of the ten conferences that still plays divisions, so it is the
# only one whose steps may be division-scoped. Any other conference acquiring a division-scoped
# parameter means a copy-paste, not a reading.
DIVISIONAL_CONFERENCE = "Sun Belt"


def test_division_scoped_parameters_appear_only_in_the_divisional_conference():
    """Guards both directions. Dropping the Sun Belt's scopes silently answers a WIDER question
    than sunbelt.txt asks -- step 4 is "common NON-DIVISIONAL opponents" precisely because step 2
    already compared divisional records, so unscoping it double-counts divisional games. Adding a
    scope anywhere else would restrict a conference that has no divisions to decline forever."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    found_in_divisional = []
    for conference, rule_sets in cfg.conferences.items():
        for rs in rule_sets:
            era = f"{conference} {rs.season_min}-{rs.season_max}"
            for st in list(rs.two_team) + list(rs.multi_team.steps):
                scoped = (
                    st.params.get("scope") in ("divisional", "non_divisional")
                    or st.params.get("standings_scope") == "divisional"
                    or st.step == "divisional_record"
                )
                if conference == DIVISIONAL_CONFERENCE:
                    if scoped:
                        found_in_divisional.append(st.step)
                else:
                    assert not scoped, (
                        f"{era} step {st.step!r} is division-scoped, but only "
                        f"{DIVISIONAL_CONFERENCE} plays divisions"
                    )
    # All three of the Sun Belt's division-scoped measures must actually be there.
    assert "divisional_record" in found_in_divisional
    assert "common_opponents_record" in found_in_divisional, (
        "sunbelt.txt step 4 is 'common NON-DIVISIONAL opponents'; the scope is missing"
    )
    assert "vs_placed_opponents" in found_in_divisional, (
        "sunbelt.txt step 3 walks the DIVISIONAL standings; standings_scope is missing"
    )


def test_sweep_sides_is_promote_only_wherever_the_text_omits_the_demote_half():
    """Only sec.txt and acc.txt state both halves. Five other documents describe a lone sweeper
    advancing and say nothing about a team that lost to everyone, so demoting one would order
    those conferences by a rule they never wrote down."""
    both_sides_conferences = {"SEC", "ACC"}
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    for conference, rule_sets in cfg.conferences.items():
        for rs in rule_sets:
            for st in rs.multi_team.steps:
                if st.step != "sweep_in_out":
                    continue
                sides = st.params.get("sides", "both")
                era = f"{conference} {rs.season_min}-{rs.season_max}"
                if conference in both_sides_conferences:
                    assert sides == "both", era
                else:
                    assert sides == "promote_only", era


def test_pac12_rules_do_not_reach_the_two_team_pac12_seasons():
    """SEASON FLOOR 2026, and it is load-bearing rather than cosmetic. pac12.txt is explicitly the
    2026 policy for a rebuilt eight-team conference on a seven-game schedule. In 2024 and 2025 the
    Pac-12 was Oregon State and Washington State ALONE -- two members, no championship game -- so
    applying this chain to those seasons would run a conference procedure against a conference
    that did not exist in that form."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    assert rules_for(cfg, "Pac-12", 2026) is not None
    assert rules_for(cfg, "Pac-12", 2025) is None
    assert rules_for(cfg, "Pac-12", 2024) is None


def test_no_rule_set_is_open_ended_backwards():
    """Every conference needs a defensible earliest season. Seven of the ten source documents are
    undated, and the grid renders seasons back to 2014, when most of these conferences had
    divisions and different procedures that nobody has supplied. An open-ended floor would apply
    today's chain to an era we have not read; resolving to no rules and falling back is the
    honest alternative."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    for conference, rule_sets in cfg.conferences.items():
        earliest = min(
            (rs.season_min if rs.season_min is not None else -1) for rs in rule_sets
        )
        assert earliest > 0, (
            f"{conference} has a rule set with no season_min; give it a floor and record the "
            f"reasoning in that entry's notes"
        )


def test_acc_pre_amendment_era_starts_at_the_first_divisionless_season():
    """The pre-2026 ACC chain is NOT open-ended backwards. The 2014-2022 ACC had Atlantic/Coastal
    divisions and a different procedure that nobody has supplied, so those seasons must resolve to
    no config and fall back to the pre-engine ordering rather than silently borrowing a
    divisionless chain."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    pre = rules_for(cfg, "ACC", 2025)
    assert pre.season_min == 2023
    assert rules_for(cfg, "ACC", 2022) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
