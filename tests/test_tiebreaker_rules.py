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


def test_real_config_loads_and_covers_all_four_power_four_conferences():
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    assert set(cfg.conferences) == {"SEC", "Big 12", "Big Ten", "ACC"}


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
    assert rules_for(cfg, "Mountain West", 2025) is None


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


def test_acc_entries_carry_the_non_default_tie_definition():
    """K9 correction: the ACC's 2026-onward tie definition is NOT plain win_pct equality --
    this must be recorded, and the real config must actually carry it."""
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    acc_2026 = rules_for(cfg, "ACC", 2026)
    assert acc_2026.tie_definition == "acc_alternate_games"
    # Every non-ACC conference (and the ACC's own pre-2026 entry) stays at the K9-compatible
    # default, so the correction is scoped to exactly where the source text supports it.
    for conference, rule_sets in cfg.conferences.items():
        for rs in rule_sets:
            if conference == "ACC" and rs.season_min == 2026:
                continue
            assert rs.tie_definition == "win_pct"


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

def test_every_multi_team_chain_opens_with_the_round_robin_sweep_pair():
    """Each conference's multi-team chain opens with an intra-group record measure, and all four
    source texts split it into two conditional branches rather than one step.

    This exists because the first transcription collapsed the Big Ten's and Big 12's opening step
    to `sweep_in_out` alone. Both texts head that step "Winning percentage in games among the
    tied teams" and then qualify their sub-clauses with "If all teams involved did not play each
    other" (bigten.txt:43-48, big12.txt:38-43). Encoding only the sub-clauses silently deletes the
    complete-round-robin case, where the rule is to rank by record within the group -- a tie
    resolving differently from what the conference published, with nothing to show it happened.
    """
    cfg = load_conference_rules(known_step_names=KNOWN_STEPS)
    for conference, rule_sets in cfg.conferences.items():
        for rs in rule_sets:
            steps = rs.multi_team.steps
            era = f"{conference} {rs.season_min}-{rs.season_max}"
            assert steps[0].step == "sub_group_record", era
            assert steps[0].when == "round_robin_among_tied", era
            assert steps[1].step == "sweep_in_out", era
            assert steps[1].when == "not_round_robin_among_tied", era


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
