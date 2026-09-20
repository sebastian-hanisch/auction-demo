from math import ceil

import numpy as np
import pytest

import cn_constants as C
from cn_bidding import AgentState, compute_bid
from cn_bruteforce import solve_bruteforce
from cn_ortools_reference import SCALE, solve_with_ortools
from cn_scenario import generate_instance
from cn_schedule import schedule_from_assignment
from auction_bids import (
    apply_language, bundle_bid, bundle_bid_table, bundle_route, jobs_of, language_mask, mask_of, min_bundle_size, n_bids,
)
from auction_wd import blocks_wd_poly, wd_bruteforce, wd_cpsat, winner_determination
from helpers_auction import ex1, ex3, held_karp_bid


# --- Gebote -----------------------------------------------------------------------------------------

def test_closed_form_bids_equal_held_karp_over_all_bundles():
    for seed in range(12):
        instance = generate_instance(3 + seed % 4, 2 + seed % 3, 0.5, 0.2 + (seed % 4) * 0.5, seed)
        table = bundle_bid_table(instance)
        for agent in range(instance.n_agents):
            for mask in range(1 << instance.n_jobs):
                expected = held_karp_bid(instance, agent, jobs_of(mask, instance.n_jobs))
                assert abs(table[agent][mask] - expected) < 1e-9, f"seed={seed} agent={agent} mask={mask}"


def test_bundle_bid_equals_route_completion_time_in_the_vehicle():
    for seed in range(15):
        instance = generate_instance(4 + seed % 4, 3, 0.4, 1.0, seed)
        table = bundle_bid_table(instance)
        for agent in range(instance.n_agents):
            for mask in (1, 5, 7, (1 << instance.n_jobs) - 1):
                route = bundle_route(instance, agent, mask)
                finish = schedule_from_assignment(instance, {agent: route})[0][agent]
                assert abs(finish - table[agent][mask]) < 1e-9
                assert abs(bundle_bid(instance, agent, jobs_of(mask, instance.n_jobs)) - table[agent][mask]) < 1e-12


def test_single_job_bundles_are_the_first_round_contract_net_bids():
    instance = generate_instance(6, 3, 0.4, 1.0, 4)
    table = bundle_bid_table(instance)
    for j, job in enumerate(instance.jobs):
        for agent in range(3):
            state = AgentState(agent_id=agent, position=instance.agent_start_positions[agent], free_time=0.0)
            assert abs(table[agent][1 << j] - compute_bid(instance, state, job).finish_time) < 1e-12


def test_empty_bundle_costs_nothing_and_table_shape():
    instance = generate_instance(5, 3, 0.3, 1.0, 1)
    table = bundle_bid_table(instance)
    assert table.shape == (3, 32) and np.all(table[:, 0] == 0.0)


# --- Sprachen und Zählformeln -----------------------------------------------------------------------

@pytest.mark.parametrize("n_jobs", [3, 5, 8, 12])
def test_bid_count_formulas_equal_counted_language_masks(n_jobs):
    instance = generate_instance(n_jobs, 3, 0.3, 1.0, 2)
    for language, b in ((C.LANG_ALL, None), (C.LANG_SIZE, 2), (C.LANG_SIZE, min(4, n_jobs)),
                        (C.LANG_BLOCK1, None), (C.LANG_BLOCK2, None)):
        counted = 3 * (int(language_mask(instance, language, b).sum()) - 1)      # ohne die leere Menge
        assert counted == n_bids(n_jobs, 3, language, b), f"{language} b={b}"


def test_documented_bid_counts_for_twelve_jobs_three_agents():
    assert n_bids(12, 3, C.LANG_ALL) == 12285
    assert n_bids(12, 3, C.LANG_SIZE, 4) == 2379 == n_bids(12, 3, C.LANG_BLOCK2)
    assert n_bids(12, 3, C.LANG_BLOCK1) == 234


def test_blocks_are_contiguous_in_position_order():
    instance = generate_instance(6, 2, 0.3, 1.0, 3)
    order = sorted(range(6), key=lambda j: (instance.jobs[j].position, j))
    mask = language_mask(instance, C.LANG_BLOCK1)
    for m in range(1, 64):
        ranks = sorted(order.index(j) for j in jobs_of(m, 6))
        contiguous = ranks == list(range(ranks[0], ranks[-1] + 1))
        assert bool(mask[m]) == contiguous, m


def test_size_limit_below_ceil_n_over_k_is_infeasible_at_ceil_it_is_feasible():
    instance = generate_instance(8, 3, 0.3, 1.0, 5)
    bids = bundle_bid_table(instance)
    b0 = min_bundle_size(8, 3)
    assert b0 == ceil(8 / 3) == 3
    assert not winner_determination(apply_language(bids, language_mask(instance, C.LANG_SIZE, b0 - 1)), "max").feasible
    assert winner_determination(apply_language(bids, language_mask(instance, C.LANG_SIZE, b0)), "max").feasible


# --- Gewinnerermittlung -----------------------------------------------------------------------------

@pytest.mark.parametrize("objective", ["max", "sum"])
def test_dp_equals_enumeration_oracle(objective):
    for seed in range(12):
        instance = generate_instance(3 + seed % 4, 2 + seed % 2, 0.4, 1.0, seed)
        bids = bundle_bid_table(instance)
        assert abs(winner_determination(bids, objective).value - wd_bruteforce(bids, objective)) < 1e-9, f"seed={seed}"
        restricted = apply_language(bids, language_mask(instance, C.LANG_BLOCK1))
        assert abs(winner_determination(restricted, objective).value - wd_bruteforce(restricted, objective)) < 1e-9


def test_dp_equals_cpsat_set_partition_and_reports_pair_count():
    for seed in range(3):
        instance = generate_instance(7, 3, 0.4, 1.0, seed)
        bids = bundle_bid_table(instance)
        dp = winner_determination(bids, "max")
        value, optimal, _ = wd_cpsat(bids, 20.0)
        assert optimal and abs(value - dp.value) < 2e-3
        assert dp.pairs == 3 * 3 ** 7


def test_wd_returns_a_partition():
    instance = generate_instance(8, 3, 0.4, 1.0, 6)
    result = winner_determination(bundle_bid_table(instance), "max")
    union = 0
    for m in result.masks:
        assert union & m == 0
        union |= m
    assert union == (1 << 8) - 1


def test_all_bundle_auction_equals_bruteforce_and_cpsat_optimum():
    tolerance_per_job = 2.0 / SCALE
    for seed in range(8):
        for n_jobs in (3, 4, 5, 6):
            instance = generate_instance(n_jobs, 2 + seed % 2, 0.4, 1.0, seed)
            auction = winner_determination(bundle_bid_table(instance), "max").value
            assert abs(auction - solve_bruteforce(instance)[0]) < 1e-9, f"seed={seed} n={n_jobs}"
            cp = solve_with_ortools(instance, time_limit_seconds=5.0)
            assert cp.makespan >= auction - 1e-6 and cp.makespan - auction < tolerance_per_job * n_jobs


def test_one_block_poly_equals_masked_dp():
    for seed in range(16):
        instance = generate_instance(4 + seed % 7, 2 + seed % 3, 0.4, 1.0, seed)
        masked = winner_determination(
            apply_language(bundle_bid_table(instance), language_mask(instance, C.LANG_BLOCK1)), "max").value
        value, masks = blocks_wd_poly(instance)
        assert abs(value - masked) < 1e-9, f"seed={seed}"
        union = 0
        for m in masks:
            assert union & m == 0
            union |= m
        assert union == (1 << instance.n_jobs) - 1


def test_sweep_route_makespan_equals_wd_value():
    for seed in range(8):
        instance = generate_instance(9, 3, 0.4, 1.0, seed)
        result = winner_determination(bundle_bid_table(instance), "max")
        schedules = {a: bundle_route(instance, a, m) for a, m in enumerate(result.masks)}
        assert abs(schedule_from_assignment(instance, schedules)[1] - result.value) < 1e-9


def test_wd_is_deterministic():
    instance = generate_instance(7, 3, 0.0, 1.0, 2)
    bids = bundle_bid_table(instance)
    assert winner_determination(bids, "max").masks == winner_determination(bids, "max").masks


# --- Handbeispiele ------------------------------------------------------------------------------------

def test_ex3_bid_table_by_hand():
    table = bundle_bid_table(ex3())
    expected = {  # Auftragsmenge: (A0, A1)
        (0,): (18, 22), (1,): (8, 18), (0, 1): (26, 33), (2,): (14, 24), (0, 2): (34, 39), (1, 2): (22, 32),
        (0, 1, 2): (42, 47),
    }
    for jobs, (a0, a1) in expected.items():
        assert table[0][mask_of(jobs)] == a0 and table[1][mask_of(jobs)] == a1, jobs


def test_ex3_winner_determination_by_hand():
    bids = bundle_bid_table(ex3())
    wd = winner_determination(bids, "max")
    assert wd.value == 22 and wd.masks == (mask_of((1, 2)), mask_of((0,)))
    wd_sum = winner_determination(bids, "sum")
    assert wd_sum.value == 42 and wd_sum.masks == (mask_of((0, 1, 2)), 0)


def test_ex1_winner_determination_by_hand():
    instance = ex1()
    table = bundle_bid_table(instance)
    assert table[0][mask_of((0, 1))] == 24 and table[1][mask_of((2,))] == 6 and table[0][mask_of((0,))] == 11
    wd = winner_determination(table, "max")
    assert wd.value == 24 and wd.masks == (mask_of((0, 1)), mask_of((2,)))
