"""Auswertung: Contract Net / Task-Swap / Auktion in der gewählten Gebotssprache / Auktion mit ALLEN Bündeln (= das exakte
Optimum dieses Vehikels) / CP-SAT auf DERSELBEN Instanz; Sprach-, Skalierungs- und Mechanismus-Sweeps auf festen Instanzen.

Die Auktion mit allen Bündeln ist mathematisch trivial das Optimum (Mengenpartition mit allen Spalten, Bündel-Gebot = beste
Route) - sie ist der Ankerpunkt, nicht die Empirie. Gemessen werden die KOSTEN: Gebote, Gewinnerermittlung, Sprache, Anreize.

Prozentangaben "vs. CNP": (Verfahren - CNP) / CNP * 100 - negativ heißt besser als Contract Net."""

import time

import numpy as np

import cn_constants as C
from cn_negotiation import negotiate
from cn_ortools_reference import solve_with_ortools
from cn_protocol import run_protocol
from cn_scenario import generate_instance
from cn_schedule import schedule_from_assignment
from auction_bids import (
    apply_language, bundle_bid_table, bundle_route, language_mask, min_bundle_size, n_bids, n_bids_cnp,
)
from auction_mechanism import RULES, misreport_curve, payment_stats, real_makespan, regret_of_agent, run_mechanism
from auction_wd import blocks_wd_poly, winner_determination


def pct_vs(value, reference):
    return (value - reference) / reference * 100.0 if reference > 0 else 0.0


def masks_to_schedules(instance, masks):
    """Bündel-Masken -> {agent: Besuchsreihenfolge} (Sweep-Reihenfolge, für Gantt und Makespan)."""
    return {a: bundle_route(instance, a, m) for a, m in enumerate(masks)}


def schedule_makespan(instance, schedules):
    return schedule_from_assignment(instance, schedules)[1]


def _language_result(instance, bids, language, b):
    """(Wert, Masken, Sekunden) der Gewinnerermittlung in der gewählten Sprache. Ein-Block-Sprache: polynomiell."""
    started = time.perf_counter()
    if language == C.LANG_BLOCK1:
        value, masks = blocks_wd_poly(instance)
        return value, masks, time.perf_counter() - started
    result = winner_determination(apply_language(bids, language_mask(instance, language, b)), "max")
    return result.value, result.masks, time.perf_counter() - started


def _cpsat_schedules(instance, result):
    by_agent = {a: [] for a in range(instance.n_agents)}
    for job, agent in result.assignment.items():
        by_agent[agent].append((result.starts[job], job))
    return {a: tuple(j for _, j in sorted(v)) for a, v in by_agent.items()}


def compare_instance(instance, language, b=None, cp_time_limit=C.ORTOOLS_TIME_LIMIT_SECONDS, include_cpsat=True):
    """Alle Spalten auf einer Instanz."""
    n, k = instance.n_jobs, instance.n_agents
    bids = bundle_bid_table(instance)
    cnp = run_protocol(instance)
    swap = negotiate(instance, cnp)
    started = time.perf_counter()
    all_wd = winner_determination(bids, "max")
    all_seconds = time.perf_counter() - started
    if language == C.LANG_ALL:
        lang_value, lang_masks, lang_seconds = all_wd.value, all_wd.masks, all_seconds
    else:
        lang_value, lang_masks, lang_seconds = _language_result(instance, bids, language, b)
    feasible = bool(np.isfinite(lang_value))

    cells = {
        "cnp": {"makespan": cnp.makespan, "schedules": dict(cnp.schedules), "bids": n_bids_cnp(n, k)},
        "swap": {"makespan": swap.final_makespan, "schedules": dict(swap.final_schedules), "bids": n_bids_cnp(n, k),
                 "swaps": len(swap.swaps)},
        "all": {"makespan": all_wd.value, "schedules": masks_to_schedules(instance, all_wd.masks),
                "bids": n_bids(n, k, C.LANG_ALL), "seconds": all_seconds},
        "language": None,
        "cpsat": None,
    }
    if feasible:
        cells["language"] = {
            "makespan": schedule_makespan(instance, masks_to_schedules(instance, lang_masks)),
            "schedules": masks_to_schedules(instance, lang_masks), "bids": n_bids(n, k, language, b),
            "seconds": lang_seconds, "wd_value": lang_value,
        }
    if include_cpsat:
        result = solve_with_ortools(instance, time_limit_seconds=cp_time_limit)
        if result.feasible:
            cells["cpsat"] = {
                "makespan": result.makespan, "schedules": _cpsat_schedules(instance, result), "optimal": result.optimal,
                "seconds": result.wall_time_ms / 1000.0,
            }
    reference = all_wd.value                       # exakter Optimalwert (alle Bündel, beste Route)
    return {
        "n_jobs": n, "n_agents": k, "language": language, "b": b, "reference": reference, "cells": cells,
        "vs_cnp": {name: pct_vs(c["makespan"], cnp.makespan) for name, c in cells.items() if c is not None},
        "gap_pct": {name: pct_vs(c["makespan"], reference) for name, c in cells.items() if c is not None},
        "min_bundle_size": min_bundle_size(n, k),
    }


def verdict(cmp):
    """Verdict-Kaskade (Warnungen zuerst) -> (Stufe, Code, Daten); die App setzt den Text."""
    cells, gap, vs = cmp["cells"], cmp["gap_pct"], cmp["vs_cnp"]
    data = {
        "language_vs_cnp": vs.get("language"), "language_gap": gap.get("language"), "cnp_gap": gap.get("cnp"),
        "swap_gap": gap.get("swap"), "bids_language": None if cells["language"] is None else cells["language"]["bids"],
        "bids_all": cells["all"]["bids"], "bids_cnp": cells["cnp"]["bids"],
    }
    cp = cells["cpsat"]
    if cp is None or not cp.get("optimal", False):
        return "info", "cpsat_unproven", data
    if cells["language"] is None or data["language_vs_cnp"] > C.LANGUAGE_WORSE_THAN_CNP_PCT:
        return "warning", "language_worse_than_cnp", data
    if gap["language"] - gap["swap"] > C.LANGUAGE_WORSE_THAN_SWAP_PCT:
        return "warning", "language_worse_than_swap", data
    if data["cnp_gap"] <= C.CNP_ALREADY_OPTIMAL_PCT:
        return "info", "cnp_already_optimal", data
    if data["language_gap"] <= C.LANGUAGE_IS_OPTIMAL_PCT:
        return "success", "language_is_optimal", data
    if data["language_vs_cnp"] <= -C.BUNDLE_GAIN_SUCCESS_PCT:
        return "success", "bundle_gain", data
    return "info", "neutral", data


def _sweep_instances(n_jobs, n_agents, var, travel, n_instances):
    return [generate_instance(n_jobs, n_agents, var, travel, C.HELDOUT_SEED_BASE + i) for i in range(n_instances)]


def sweep_size(n_jobs):
    return C.N_SWEEP_INSTANCES if n_jobs <= 10 else C.N_SWEEP_INSTANCES_MID if n_jobs <= 12 else 12


def language_sweep(n_jobs, n_agents, var, travel, n_instances=None):
    """Feste Instanzen (unabhängig vom Demo-Seed): Lücke zum exakten Optimum je Sprache, dazu CNP und Task-Swap; Gewinn- und
    Verlustanteile gegen CNP und Swap; Gebotszahl."""
    n_instances = n_instances or sweep_size(n_jobs)
    b0 = min_bundle_size(n_jobs, n_agents)
    languages = [("cnp", "Contract Net (roh)", None, None), ("swap", "Task-Swap", None, None)]
    languages += [
        ("size0", f"Größenlimit b = {b0} (kleinstmöglich)", C.LANG_SIZE, b0),
        ("size1", f"Größenlimit b = {b0 + 1}", C.LANG_SIZE, b0 + 1),
        ("block1", "1 Block", C.LANG_BLOCK1, None),
        ("block2", "2 Blöcke", C.LANG_BLOCK2, None),
        ("all", "Alle Bündel (= Optimum)", C.LANG_ALL, None),
    ]
    values = {key: [] for key, *_ in languages}
    for instance in _sweep_instances(n_jobs, n_agents, var, travel, n_instances):
        bids = bundle_bid_table(instance)
        cnp = run_protocol(instance)
        values["cnp"].append(cnp.makespan)
        values["swap"].append(negotiate(instance, cnp).final_makespan)
        for key, _, language, b in languages[2:]:
            if language == C.LANG_BLOCK1:
                values[key].append(blocks_wd_poly(instance)[0])
            else:
                mask = language_mask(instance, language, b)
                values[key].append(winner_determination(apply_language(bids, mask), "max").value)
    optimum = np.array(values["all"])
    rows = []
    for key, label, language, b in languages:
        ms = np.array(values[key])
        gap = (ms - optimum) / optimum * 100.0
        rows.append({
            "key": key, "label": label, "language": language, "b": b,
            "mean_gap": float(gap.mean()), "median_gap": float(np.median(gap)), "max_gap": float(gap.max()),
            "optimal_frac": float((ms <= optimum * (1 + 1e-9)).mean()),
            "lose_vs_cnp": float((ms > np.array(values["cnp"]) + 1e-9).mean()),
            "lose_vs_swap": float((ms > np.array(values["swap"]) + 1e-9).mean()),
            "bids": (n_bids_cnp(n_jobs, n_agents) if language is None else n_bids(n_jobs, n_agents, language, b)),
        })
    return {"n_instances": n_instances, "rows": rows, "b0": b0}


def scaling_sweep(n_agents, ns, var, travel, cp_time_limit=C.ORTOOLS_SWEEP_TIME_LIMIT_SECONDS):
    """Je n: Gebotszahl (alle Bündel, Formel), Zeit der exakten Gewinnerermittlung (nur bis EXACT_DP_MAX_N), Zeit und Lücke der
    polynomiellen Ein-Block-Auktion. Referenz: exakter DP bis n=16, darüber das CP-SAT-Incumbent (UNBEWIESEN - es kann am
    einfachen CP-SAT-Modell liegen, wenn die Ein-Block-Auktion es schlägt)."""
    rows = []
    for n in ns:
        count = C.SCALING_INSTANCES if n <= C.EXACT_DP_MAX_N else max(2, C.SCALING_INSTANCES - 2)
        times_dp, times_block, gaps_block, gaps_cnp, cp_proven, refs = [], [], [], [], 0, []
        for instance in _sweep_instances(n, n_agents, var, travel, count):
            block_start = time.perf_counter()
            block_value, _ = blocks_wd_poly(instance)
            times_block.append(time.perf_counter() - block_start)
            cnp_value = run_protocol(instance).makespan
            if n <= C.EXACT_DP_MAX_N:
                started = time.perf_counter()
                reference = winner_determination(bundle_bid_table(instance), "max").value
                times_dp.append(time.perf_counter() - started)
                kind = "exakt (DP)"
            else:
                result = solve_with_ortools(instance, time_limit_seconds=cp_time_limit)
                reference = result.makespan if result.feasible else block_value
                cp_proven += 1 if result.optimal else 0
                kind = "CP-SAT-Incumbent (unbewiesen)" if cp_proven == 0 else "CP-SAT (teils bewiesen)"
            refs.append(kind)
            gaps_block.append(pct_vs(block_value, reference))
            gaps_cnp.append(pct_vs(cnp_value, reference))
        rows.append({
            "n_jobs": n, "bids_all": n_bids(n, n_agents, C.LANG_ALL), "bids_block1": n_bids(n, n_agents, C.LANG_BLOCK1),
            "dp_seconds": float(np.mean(times_dp)) if times_dp else None, "block_seconds": float(np.mean(times_block)),
            "block_gap": float(np.mean(gaps_block)), "cnp_gap": float(np.mean(gaps_cnp)), "reference": refs[0],
            "n_instances": count,
        })
    return rows


def _language_bids(instance, language, b):
    """Gebotstabelle, eingeschränkt auf die gewählte Sprache (die Agenten bieten nur diese Bündel)."""
    bids = bundle_bid_table(instance)
    if language is None or language == C.LANG_ALL:
        return bids
    return apply_language(bids, language_mask(instance, language, b))


def mechanism_table(instance, rules=("A", "B", "C", "E"), language=None, b=None):
    """Ziel x Zahlung auf EINER Instanz mit ehrlichen (wahren) Geboten in der gewählten Gebotssprache: echter Makespan vs.
    exaktem Optimum (alle Bündel), Zahlung/Kosten, kleinste Rendite (Zahlung - Kosten), Monopol-Agenten. Bei einer Sprache
    ohne zulässige Zuteilung ist `infeasible` gesetzt und `rows` leer."""
    optimum = winner_determination(bundle_bid_table(instance), "max").value
    true_bids = _language_bids(instance, language, b)
    if not winner_determination(true_bids, "max").feasible:
        return {"optimum": optimum, "rows": [], "true_bids": true_bids, "infeasible": True}
    rows = []
    for rule in rules:
        outcome = run_mechanism(true_bids, rule)
        stats = payment_stats(true_bids, outcome)
        rows.append({
            "rule": rule, "objective": RULES[rule][0], "makespan": real_makespan(true_bids, outcome.masks),
            "makespan_gap": pct_vs(real_makespan(true_bids, outcome.masks), optimum), "masks": outcome.masks, **stats,
        })
    return {"optimum": optimum, "rows": rows, "true_bids": true_bids, "infeasible": False}


def lying_lab(instance, agent, rules=("A", "B", "C", "E"), factors=C.LAMBDA_GRID, language=None, b=None):
    """Ein strategischer Bieter, alle anderen ehrlich: Nutzen je Faktor und Regel (Untergrenze des Regrets)."""
    true_bids = _language_bids(instance, language, b)
    return {rule: misreport_curve(true_bids, agent, rule, factors) for rule in rules}


def mechanism_sweep(n_jobs, n_agents, var, travel, n_instances=15, factors=(0.7, 1.3, 1.6, 2.0, 3.0)):
    """Gepoolt über feste Instanzen: je Regel mittlere Makespan-Lücke, Zahlung/Kosten, Anteil IR-Verletzungen, Anteil Gewinner
    mit profitabler Lüge, mittlerer Regret (in % der Kosten) und Anteil Instanzen mit Monopol (VCG nicht definiert)."""
    acc = {rule: {"gap": [], "ratio": [], "ir": 0, "undefined": 0, "winners": 0, "lie_winners": 0, "regret": []}
           for rule in ("A", "B", "C", "E")}
    for instance in _sweep_instances(n_jobs, n_agents, var, travel, n_instances):
        true_bids = bundle_bid_table(instance)
        optimum = winner_determination(true_bids, "max").value
        for rule in acc:
            outcome = run_mechanism(true_bids, rule)
            stats = payment_stats(true_bids, outcome)
            acc[rule]["gap"].append(pct_vs(real_makespan(true_bids, outcome.masks), optimum))
            if not stats["defined"]:
                acc[rule]["undefined"] += 1
                continue
            if stats["ratio"] is not None:
                acc[rule]["ratio"].append(stats["ratio"])
            acc[rule]["ir"] += 1 if stats["min_margin"] < -1e-9 else 0
            for agent in range(n_agents):
                if outcome.masks[agent] == 0:
                    continue
                acc[rule]["winners"] += 1
                gain, _ = regret_of_agent(true_bids, agent, rule, factors)
                gain = gain or 0.0
                acc[rule]["lie_winners"] += 1 if gain > 1e-9 else 0
                acc[rule]["regret"].append(gain / true_bids[agent][outcome.masks[agent]] * 100.0)
    rows = {}
    for rule, a in acc.items():
        rows[rule] = {
            "mean_gap": float(np.mean(a["gap"])), "ratio": float(np.mean(a["ratio"])) if a["ratio"] else None,
            "ir_share": a["ir"] / n_instances, "undefined_share": a["undefined"] / n_instances,
            "lie_share": (a["lie_winners"] / a["winners"]) if a["winners"] else 0.0,
            "mean_regret_pct": float(np.mean(a["regret"])) if a["regret"] else 0.0,
        }
    return {"n_instances": n_instances, "rows": rows}
