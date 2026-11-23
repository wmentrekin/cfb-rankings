"""Pure selection rules of the 2026 projected playoff field (artifacts/playoffs.py).

MUTATION CHECK (run by hand against artifacts/playoffs.py; each mutant was confirmed to make at
least one test below fail, then reverted):
  - Notre Dame threshold `<=` -> `<`                       -> test_notre_dame_at_12_in
  - drop the Notre Dame guarantee block                    -> test_notre_dame_at_12_in
  - G6 slot picks the WORST-ranked G6 team (stand-in for "G6 champion only") -> test_g6_slot_goes_to_best_g6_team_not_champion
  - fill loop overshoots the field size (13 teams)          -> test_champion_ranked_20th_is_in_and_seeded_by_rank / test_no_duplicates_and_fills_to_twelve
  - bid precedence reversed (at-large overwrites champion) -> test_actual_champion_gets_unprefixed_label
  - seeds by bid class instead of rank                     -> test_champion_ranked_20th_is_in_and_seeded_by_rank
  - resolver: first-listed leader instead of model rank    -> test_resolver_tie_at_top_goes_to_higher_model_rank
  - resolver: ignore played champion                       -> test_resolver_actual_champion_beats_better_ranked_leader
  - resolver: skip two-clinched restriction                -> test_resolver_two_clinched_teams_restrict_candidates
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifacts.playoffs import (  # noqa: E402
    SEASON_RULES,
    resolve_projected_champions,
    select_first_out,
    select_projected_field,
)

RULES = SEASON_RULES[2026]
AQ = RULES["aq_conferences"]


def _league():
    """30 teams ranked 1..30. Conferences: ACC a1.., SEC s1.., Big Ten b1.., Big 12 x1.., a few
    Group of Six teams and Notre Dame placed by the individual tests."""
    ranks, conf = {}, {}

    def add(name, rank, conference):
        ranks[name], conf[name] = rank, conference

    return ranks, conf, add


def _champs(**by_conf):
    return {conf: {"team": team, "actual": False} for conf, team in by_conf.items()}


def _base():
    ranks, conf, add = _league()
    add("Utah", 1, "Big 12")
    add("Georgia", 2, "SEC")
    add("Ohio State", 3, "Big Ten")
    add("Miami", 4, "ACC")
    for i, rank in enumerate([5, 6, 7, 8, 9, 10, 11, 12, 14, 15]):
        add(f"AtLarge{i}", rank, "SEC" if i % 2 else "Big Ten")
    add("Tulane", 16, "American Athletic")
    add("Boise State", 17, "Mountain West")
    add("Notre Dame", 13, "FBS Independents")
    add("Wake Forest", 20, "ACC")
    return ranks, conf, add


def _place(ranks, team, rank):
    """Move `team` to `rank`, shifting everyone else so ranks stay unique and contiguous order holds."""
    others = sorted((t for t in ranks if t != team), key=lambda t: ranks[t])
    for i, t in enumerate(others, start=1):
        ranks[t] = i if i < rank else i + 1
    ranks[team] = rank


def _field(ranks, conf, champs):
    return select_projected_field(ranks, conf, champs, RULES)


def _by_team(field):
    return {e["team"]: e for e in field}


def test_champion_ranked_20th_is_in_and_seeded_by_rank():
    ranks, conf, _ = _base()
    champs = _champs(**{"Big 12": "Utah", "SEC": "Georgia", "Big Ten": "Ohio State", "ACC": "Wake Forest"})
    field = _field(ranks, conf, champs)
    teams = _by_team(field)
    assert teams["Wake Forest"]["bid"] == "champion"
    assert teams["Wake Forest"]["bid_label"] == "Projected ACC Champion"
    # Strictly by rank: the 20th-ranked champion sits at the bottom of the field, not above a
    # better-ranked at-large team.
    assert [e["rank"] for e in field] == sorted(e["rank"] for e in field)
    assert field[-1]["team"] == "Wake Forest" and field[-1]["seed"] == 12
    assert [e["seed"] for e in field] == list(range(1, 13))


def test_actual_champion_gets_unprefixed_label():
    ranks, conf, _ = _base()
    champs = {"ACC": {"team": "Miami", "actual": True}}
    assert _by_team(_field(ranks, conf, champs))["Miami"]["bid_label"] == "ACC Champion"


def test_byes_are_seeds_one_through_four_and_first_out():
    ranks, conf, _ = _base()
    field = _field(ranks, conf, _champs(ACC="Wake Forest"))
    assert [e["bye"] for e in field] == [True] * 4 + [False] * 8
    out = select_first_out(ranks, field, 4)
    in_field = {e["team"] for e in field}
    assert len(out) == 4 and not (set(out) & in_field)
    outside = sorted((r, t) for t, r in ranks.items() if t not in in_field)
    assert out == [t for _, t in outside[:4]]


def test_g6_slot_goes_to_best_g6_team_not_champion():
    ranks, conf, _ = _base()
    # Boise State (17) is the Mountain West champion, but Tulane (16) is the higher-ranked G6 team.
    field = _field(ranks, conf, _champs(ACC="Wake Forest"))
    teams = _by_team(field)
    assert teams["Tulane"]["bid"] == "g6" and teams["Tulane"]["bid_label"] == "Highest-Ranked Group of Six"
    assert "Boise State" not in teams


def test_notre_dame_at_12_in():
    ranks, conf, _ = _base()
    _place(ranks, "Notre Dame", 12)
    field = _field(ranks, conf, _champs(ACC="Wake Forest"))
    nd = _by_team(field)["Notre Dame"]
    assert nd["bid"] == "notre_dame" and nd["bid_label"] == "Top-12 Guarantee"


def test_notre_dame_at_13_out_when_field_is_full_without_it():
    ranks, conf, _ = _base()
    field = _field(ranks, conf, _champs(ACC="Wake Forest"))
    assert "Notre Dame" not in _by_team(field)  # rank 13: no guarantee, and the field fills by rank


def test_notre_dame_at_5_consumes_no_extra_slot():
    ranks, conf, _ = _base()
    _place(ranks, "Notre Dame", 5)
    field = _field(ranks, conf, _champs(ACC="Wake Forest"))
    teams = _by_team(field)
    assert len(field) == 12 and len(teams) == 12
    # In on rank alone, labelled by the guarantee precedence (champion > g6 > notre_dame > at_large).
    assert teams["Notre Dame"]["bid"] == "notre_dame" and teams["Notre Dame"]["seed"] == 5


def test_no_duplicates_and_fills_to_twelve():
    ranks, conf, _ = _base()
    # Every guaranteed class is also top-ranked; a champion who is also G6-best cannot happen, but
    # a champion ranked 1 must appear exactly once.
    field = _field(ranks, conf, _champs(**{"Big 12": "Utah", "SEC": "Georgia", "Big Ten": "Ohio State", "ACC": "Miami"}))
    names = [e["team"] for e in field]
    assert len(names) == len(set(names)) == 12
    assert _by_team(field)["Utah"]["bid"] == "champion"


def test_unranked_teams_ignored_and_short_field_is_none():
    ranks, conf, _ = _base()
    conf["Unranked U"] = "SEC"  # no rank -> never selected
    assert "Unranked U" not in _by_team(_field(ranks, conf, _champs(ACC="Wake Forest")))
    short = {f"T{i}": i for i in range(1, 12)}
    assert select_projected_field(short, {}, {}, RULES) is None


# ---------------------------------------------------------------------------
# Champion resolver
# ---------------------------------------------------------------------------
def _st(conference, w, l, status="possible"):
    return {"conference": conference, "conf_record": {"wins": w, "losses": l}, "championship_status": status}


def test_resolver_actual_champion_beats_better_ranked_leader():
    ranks = {"A": 1, "B": 9}
    standings = {"A": _st("ACC", 8, 0), "B": _st("ACC", 5, 3)}
    out = resolve_projected_champions(ranks, standings, {"ACC": "B"}, ["ACC"])
    assert out["ACC"] == {"team": "B", "actual": True}


def test_resolver_tie_at_top_goes_to_higher_model_rank():
    ranks = {"A": 7, "B": 3, "C": 2}
    # A and B tie at 7-1; C (better ranked) is a game back. A beat B head-to-head, which must not matter.
    standings = {"A": _st("ACC", 7, 1), "B": _st("ACC", 7, 1), "C": _st("ACC", 6, 2)}
    out = resolve_projected_champions(ranks, standings, {}, ["ACC"])
    assert out["ACC"] == {"team": "B", "actual": False}


def test_resolver_two_clinched_teams_restrict_candidates():
    ranks = {"A": 10, "B": 11, "C": 1}
    # C leads the standings on rank and record, but only A and B hold title-game berths.
    standings = {
        "A": _st("ACC", 6, 2, "clinched"), "B": _st("ACC", 7, 1, "clinched"), "C": _st("ACC", 7, 1, "possible"),
    }
    out = resolve_projected_champions(ranks, standings, {}, ["ACC"])
    assert out["ACC"]["team"] == "B"  # best record of the two, not the overall leader C


def test_resolver_single_clinched_team_does_not_restrict():
    ranks = {"A": 10, "C": 1}
    standings = {"A": _st("ACC", 7, 1, "clinched"), "C": _st("ACC", 7, 1, "possible")}
    assert resolve_projected_champions(ranks, standings, {}, ["ACC"])["ACC"]["team"] == "C"


def test_resolver_ignores_unranked_and_missing_conferences():
    ranks = {"A": 4}
    standings = {"A": _st("ACC", 5, 3), "Z": _st("ACC", 8, 0), "S": _st("SEC", 8, 0)}
    out = resolve_projected_champions(ranks, standings, {}, ["ACC", "SEC"])
    assert out == {"ACC": {"team": "A", "actual": False}}


# ---------------------------------------------------------------------------
# Champion resolver: the IDENTIFIED title game narrows candidates (review F2)
# ---------------------------------------------------------------------------
def test_resolver_identified_pair_never_lets_a_third_tied_team_win():
    ranks = {"A": 8, "B": 6, "C": 1}
    standings = {"A": _st("ACC", 7, 2), "B": _st("ACC", 7, 2), "C": _st("ACC", 7, 2)}
    out = resolve_projected_champions(ranks, standings, {}, ["ACC"], title_game_pairs={"ACC": {"A", "B"}})
    assert out["ACC"] == {"team": "B", "actual": False}
    # without the identified game the best-ranked of the three would have been picked
    assert resolve_projected_champions(ranks, standings, {}, ["ACC"])["ACC"]["team"] == "C"


def test_resolver_identified_pair_beats_a_different_clinched_pair():
    ranks = {"A": 5, "B": 6, "C": 1, "D": 2}
    standings = {
        "A": _st("ACC", 6, 2), "B": _st("ACC", 6, 2), "C": _st("ACC", 7, 1, "clinched"), "D": _st("ACC", 7, 1, "clinched"),
    }
    out = resolve_projected_champions(ranks, standings, {}, ["ACC"], title_game_pairs={"ACC": {"A", "B"}})
    assert out["ACC"]["team"] == "A"


def test_resolver_actual_champion_still_beats_the_identified_pair():
    ranks = {"A": 1, "B": 2, "C": 3}
    standings = {t: _st("ACC", 7, 2) for t in ranks}
    out = resolve_projected_champions(ranks, standings, {"ACC": "C"}, ["ACC"], title_game_pairs={"ACC": {"A", "B"}})
    assert out["ACC"] == {"team": "C", "actual": True}


def test_resolver_pair_with_one_unranked_team_picks_the_ranked_one_not_an_eliminated_third():
    standings = {
        "A": _st("ACC", 6, 2, "clinched"), "B": _st("ACC", 7, 1, "clinched"), "C": _st("ACC", 8, 0, "eliminated"),
    }
    ranks = {"A": 10, "C": 1}  # B (the better record of the clinched pair) is unranked
    out = resolve_projected_champions(ranks, standings, {}, ["ACC"])
    assert out["ACC"] == {"team": "A", "actual": False}
    out = resolve_projected_champions(ranks, standings, {}, ["ACC"], title_game_pairs={"ACC": {"A", "B"}})
    assert out["ACC"] == {"team": "A", "actual": False}


def test_resolver_pair_with_neither_team_ranked_gives_no_champion_and_warns(caplog):
    standings = {
        "A": _st("ACC", 6, 2, "clinched"), "B": _st("ACC", 7, 1, "clinched"), "C": _st("ACC", 8, 0),
    }
    with caplog.at_level("WARNING", logger="cfb_lp"):
        out = resolve_projected_champions({"C": 1}, standings, {}, ["ACC"])
        assert "ACC" not in out
        out = resolve_projected_champions({"C": 1}, standings, {}, ["ACC"], title_game_pairs={"ACC": {"A", "B"}})
        assert "ACC" not in out
    assert caplog.text.count("neither of the") == 2 and "ACC" in caplog.text


def test_resolver_falls_back_to_the_whole_conference_without_a_usable_pair():
    ranks = {"A": 9, "B": 8, "C": 1}
    standings = {t: _st("ACC", 7, 2) for t in ranks}
    assert resolve_projected_champions(ranks, standings, {}, ["ACC"])["ACC"]["team"] == "C"
    assert resolve_projected_champions(ranks, standings, {}, ["ACC"], title_game_pairs={"ACC": set()})["ACC"]["team"] == "C"
    # a pair naming a team outside the conference's standings is not trusted
    out = resolve_projected_champions(ranks, standings, {}, ["ACC"], title_game_pairs={"ACC": {"A", "Zed"}})
    assert out["ACC"]["team"] == "C"
    # nor one containing an already-eliminated team (a false-positive identified game)
    standings["B"] = _st("ACC", 2, 7, "eliminated")
    out = resolve_projected_champions(ranks, standings, {}, ["ACC"], title_game_pairs={"ACC": {"A", "B"}})
    assert out["ACC"]["team"] == "C"


def test_title_game_pairs_reads_the_teams_of_the_identified_game_ids():
    from artifacts.playoffs import _title_game_pairs

    rows = [
        {"game_id": 7, "team": "A", "opponent": "B"}, {"game_id": 7, "team": "B", "opponent": "A"},
        {"game_id": 8, "team": "A", "opponent": "X"},
        {"game_id": 9, "team": "P", "opponent": "Q"}, {"game_id": 9, "team": "R", "opponent": "Q"},
    ]
    assert _title_game_pairs(rows, {"ACC": 7, "SEC": 9, "Big Ten": 10}) == {"ACC": {"A", "B"}}


def test_build_playoffs_payload_uses_one_shared_standings_prelude(monkeypatch):
    from artifacts import playoffs, schedule

    assert playoffs.compute_season_standings is schedule.compute_season_standings
    assert not hasattr(playoffs, "_season_standings")
