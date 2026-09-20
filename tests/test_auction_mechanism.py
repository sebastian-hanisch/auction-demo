import numpy as np

import cn_constants as C
from cn_scenario import generate_instance
from auction_bids import apply_language, bundle_bid_table, language_mask, mask_of
from auction_mechanism import (
    RULES, misreport, misreport_curve, payment_stats, real_makespan, regret_of_agent, run_mechanism, utility,
)
from helpers_auction import ex1, ex3


def _payments(outcome):
    return tuple(round(p, 9) for p in outcome.payments)


# --- Handbeispiele ------------------------------------------------------------------------------------

def test_ex3_rules_by_hand():
    bids = bundle_bid_table(ex3())
    a = run_mechanism(bids, "A")                       # Pay-as-bid: jeder bekommt sein eigenes Gebot
    assert a.value == 22 and _payments(a) == (22.0, 22.0)
    b = run_mechanism(bids, "B")                       # Summen-Zuteilung + VCG
    assert b.value == 42 and b.masks == (mask_of((0, 1, 2)), 0) and _payments(b) == (47.0, 0.0)
    assert _payments(run_mechanism(bids, "C")) == (42.0, 0.0)
    e = run_mechanism(bids, "E")                       # Clarke-Formel auf Makespan-Zuteilung
    assert _payments(e) == (25.0, 20.0)
    d = run_mechanism(bids, "D")
    assert _payments(d) == (26.0, 26.0)


def test_ex3_clarke_formula_on_makespan_violates_individual_rationality():
    bids = bundle_bid_table(ex3())
    stats = payment_stats(bids, run_mechanism(bids, "E"))
    assert stats["defined"] and stats["min_margin"] < 0             # A1 bekommt 20, seine Kosten sind 22
    assert payment_stats(bids, run_mechanism(bids, "B"))["min_margin"] >= 0


def test_ex1_vcg_and_pay_as_bid_lie():
    bids = bundle_bid_table(ex1())
    vcg = run_mechanism(bids, "B")
    assert _payments(vcg) == (32.0, 14.0)
    honest = misreport(bids, 1, "A", 1.0)
    lie = misreport(bids, 1, "A", 4.0)
    assert honest.masks == lie.masks                                # gleiche Zuteilung ...
    assert utility(bids, honest, 1) == 0.0
    assert abs(utility(bids, lie, 1) - 18.0) < 1e-9                 # ... aber 18 Minuten Gewinn durch Aufblähen


# --- VCG-Eigenschaften -----------------------------------------------------------------------------

def test_vcg_on_sum_objective_is_truthful_on_the_lambda_grid_and_random_multipliers():
    rng = np.random.default_rng(0)
    violations = 0
    for seed in range(12):
        instance = generate_instance(6, 3, 0.4, 1.0, seed)
        bids = bundle_bid_table(instance)
        for agent in range(3):
            honest = utility(bids, misreport(bids, agent, "B", 1.0), agent)
            if honest is None:
                continue
            assert honest >= -1e-9                                    # individuelle Rationalität
            for factor in list(C.LAMBDA_GRID) + list(rng.uniform(0.3, 4.0, 8)):
                outcome = misreport(bids, agent, "B", float(factor))
                u = utility(bids, outcome, agent) if outcome.feasible else None
                if u is not None and u > honest + 1e-9:
                    violations += 1
    assert violations == 0


def test_losers_pay_nothing():
    for seed in range(6):
        bids = bundle_bid_table(generate_instance(6, 3, 0.4, 1.0, seed))
        for rule in RULES:
            outcome = run_mechanism(bids, rule)
            for a, mask in enumerate(outcome.masks):
                if mask == 0:
                    assert outcome.payments[a] == 0.0


def test_vcg_is_undefined_for_a_monopolist_and_never_a_made_up_number():
    instance = generate_instance(6, 3, 0.4, 1.0, 1)
    bids = apply_language(bundle_bid_table(instance), language_mask(instance, C.LANG_SIZE, 2))      # b = ceil(6/3): alle gleich nötig
    outcome = run_mechanism(bids, "B")
    assert outcome.feasible and outcome.undefined_agents
    for a in outcome.undefined_agents:
        assert np.isnan(outcome.payments[a]) and utility(bids, outcome, a) is None
    assert payment_stats(bids, outcome)["defined"] is False


def test_infeasible_language_reports_infeasible_outcome():
    instance = generate_instance(6, 3, 0.4, 1.0, 1)
    bids = apply_language(bundle_bid_table(instance), language_mask(instance, C.LANG_SIZE, 1))
    outcome = run_mechanism(bids, "A")
    assert not outcome.feasible


# --- Fehlmeldung / Regret ------------------------------------------------------------------------------

def test_real_makespan_uses_true_bids_of_the_awarded_bundles():
    bids = bundle_bid_table(ex3())
    lied = misreport(bids, 0, "A", 3.0)
    assert real_makespan(bids, lied.masks) <= 22 + 3 * 22
    honest = misreport(bids, 0, "A", 1.0)
    assert real_makespan(bids, honest.masks) == honest.value == 22


def test_misreport_curve_matches_single_calls_and_regret_is_gain_over_honesty():
    bids = bundle_bid_table(ex1())
    curve = misreport_curve(bids, 1, "A", C.LAMBDA_GRID)
    by_factor = {row[0]: row for row in curve}
    assert by_factor[1.0][1] == 0.0
    gain, factor = regret_of_agent(bids, 1, "A", C.LAMBDA_GRID)
    assert gain == max(0.0, max(row[1] for row in curve if row[1] is not None))
    assert factor in C.LAMBDA_GRID
    # unter VCG (Summenziel) gibt es nichts zu gewinnen
    assert regret_of_agent(bids, 1, "B", C.LAMBDA_GRID)[0] == 0.0
