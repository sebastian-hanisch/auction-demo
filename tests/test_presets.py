"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert)."""

import pytest

import cn_constants as C
from cn_scenario import generate_instance
from auction_bids import bundle_bid_table
from auction_evaluation import compare_instance, mechanism_table, verdict
from auction_mechanism import misreport, utility


def _measure(preset):
    instance = generate_instance(
        preset["n_jobs"], preset["n_agents"], preset["duration_variability"], preset["travel_time_per_unit"], preset["seed"],
    )
    language, b = preset["language"], (preset["b"] if preset["language"] == C.LANG_SIZE else None)
    cmp = compare_instance(instance, language, b, cp_time_limit=5.0)
    gap = cmp["gap_pct"]
    rules = {r["rule"]: r for r in mechanism_table(instance)["rows"]}
    true_bids = bundle_bid_table(instance)
    honest = misreport(true_bids, preset["bidder"], "A", 1.0)
    lied = misreport(true_bids, preset["bidder"], "A", preset["lam"])
    return {
        "verdict": verdict(cmp)[1], "cnp_gap": gap["cnp"], "swap_gap": gap["swap"], "language_gap": gap.get("language"),
        "bids_language": cmp["cells"]["language"]["bids"] if cmp["cells"]["language"] else None,
        "vcg_makespan_gap": rules["B"]["makespan_gap"], "vcg_ratio": rules["B"]["ratio"],
        "clarke_min_margin": rules["E"]["min_margin"],
        "lie_gain": utility(true_bids, lied, preset["bidder"]) - utility(true_bids, honest, preset["bidder"]),
        "lie_same_alloc": lied.masks == honest.masks,
    }


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 8


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if isinstance(expected, str):
            assert value == expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"


def test_preset_settings_are_within_slider_bounds():
    for preset in C.PRESETS.values():
        assert C.N_JOBS_MIN <= preset["n_jobs"] <= C.N_JOBS_MAX
        assert C.N_AGENTS_MIN <= preset["n_agents"] <= C.N_AGENTS_MAX
        assert preset["language"] in C.LANGUAGE_LABELS
        assert 0 <= preset["bidder"] < preset["n_agents"]
        assert C.LAMBDA_MIN <= preset["lam"] <= C.LAMBDA_MAX
