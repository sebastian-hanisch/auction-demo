import cn_constants as C
from cn_ortools_reference import SCALE
from cn_scenario import generate_instance
from auction_evaluation import (
    compare_instance, language_sweep, lying_lab, masks_to_schedules, mechanism_sweep, mechanism_table, schedule_makespan,
    scaling_sweep, verdict,
)


def test_compare_instance_anchors_and_orderings():
    for seed in range(6):
        instance = generate_instance(7, 3, 0.3, 1.0, seed)
        cmp = compare_instance(instance, C.LANG_BLOCK2, cp_time_limit=5.0)
        cells, reference = cmp["cells"], cmp["reference"]
        assert cells["all"]["makespan"] == reference
        for name in ("cnp", "swap", "language"):
            assert cells[name]["makespan"] >= reference - 1e-9
        assert cells["swap"]["makespan"] <= cells["cnp"]["makespan"] + 1e-9
        assert cells["cpsat"]["optimal"]
        assert abs(cells["cpsat"]["makespan"] - reference) < (2.0 / SCALE) * 7
        assert cmp["gap_pct"]["all"] == 0.0
        # Gantt-Pläne und ausgewiesener Makespan stimmen überein
        assert abs(schedule_makespan(instance, cells["all"]["schedules"]) - reference) < 1e-9


def test_language_cell_is_none_when_size_limit_is_infeasible():
    instance = generate_instance(9, 3, 0.3, 1.0, 3)
    cmp = compare_instance(instance, C.LANG_SIZE, b=2, include_cpsat=False)      # b < ceil(9/3)
    assert cmp["cells"]["language"] is None
    assert verdict(cmp)[1] == "cpsat_unproven"                                   # ohne CP-SAT nur der Info-Fall


def test_verdict_cascade_codes():
    def make(cnp, swap, language, cpsat_optimal=True, reference=100.0):
        cell = lambda v: {"makespan": v, "bids": 1}
        cells = {"cnp": cell(cnp), "swap": cell(swap), "all": cell(reference), "language": cell(language),
                 "cpsat": {"makespan": reference, "optimal": cpsat_optimal}}
        gap = {k: (v["makespan"] - reference) / reference * 100 for k, v in cells.items()}
        vs = {k: (v["makespan"] - cnp) / cnp * 100 for k, v in cells.items()}
        return {"cells": cells, "gap_pct": gap, "vs_cnp": vs}
    assert verdict(make(130, 110, 100, cpsat_optimal=False))[1] == "cpsat_unproven"
    assert verdict(make(110, 105, 120))[1] == "language_worse_than_cnp"
    assert verdict(make(130, 105, 112))[1] == "language_worse_than_swap"
    assert verdict(make(100.2, 100.2, 100.2))[1] == "cnp_already_optimal"
    assert verdict(make(130, 120, 100.1))[1] == "language_is_optimal"
    assert verdict(make(130, 120, 115))[1] == "bundle_gain"
    assert verdict(make(130, 127, 127.5))[1] == "neutral"


def test_language_sweep_structure_and_monotonicity():
    result = language_sweep(6, 3, 0.3, 1.0, n_instances=8)
    rows = {r["key"]: r for r in result["rows"]}
    assert rows["all"]["mean_gap"] == 0.0 and rows["all"]["optimal_frac"] == 1.0
    assert rows["block2"]["mean_gap"] <= rows["block1"]["mean_gap"] + 1e-9        # mehr Bündel, nie schlechter
    assert rows["size1"]["mean_gap"] <= rows["size0"]["mean_gap"] + 1e-9
    assert rows["swap"]["mean_gap"] <= rows["cnp"]["mean_gap"] + 1e-9
    assert rows["all"]["bids"] > rows["block2"]["bids"] > rows["block1"]["bids"] > rows["cnp"]["bids"]


def test_sweeps_are_deterministic_and_use_fixed_seeds():
    first = language_sweep(6, 3, 0.3, 1.0, n_instances=5)
    second = language_sweep(6, 3, 0.3, 1.0, n_instances=5)
    assert first == second
    assert C.HELDOUT_SEED_BASE > 10_000                                           # nie ein Demo-Seed


def test_scaling_sweep_rows_small():
    rows = scaling_sweep(3, (6, 8), 0.3, 1.0)
    assert [r["n_jobs"] for r in rows] == [6, 8]
    assert rows[0]["bids_all"] == 3 * 63 and rows[1]["bids_block1"] == 3 * 36
    assert all(r["reference"] == "exakt (DP)" and r["block_gap"] >= -1e-9 for r in rows)
    assert all(r["dp_seconds"] is not None for r in rows)


def test_mechanism_table_and_lying_lab_and_sweep_smoke():
    instance = generate_instance(6, 3, 0.3, 1.0, 11)
    table = mechanism_table(instance)
    by_rule = {r["rule"]: r for r in table["rows"]}
    assert set(by_rule) == {"A", "B", "C", "E"}
    assert by_rule["A"]["makespan_gap"] < 1e-9                                    # Makespan-Zuteilung ist optimal
    assert by_rule["B"]["makespan_gap"] >= -1e-9                                  # Summenziel kann nie besser sein
    curves = lying_lab(instance, 1)
    assert set(curves) == {"A", "B", "C", "E"} and len(curves["A"]) == len(C.LAMBDA_GRID)
    sweep = mechanism_sweep(6, 3, 0.3, 1.0, n_instances=4)
    assert sweep["rows"]["B"]["lie_share"] == 0.0                                 # VCG-Wahrheit auf dem Faktoren-Gitter
    assert sweep["rows"]["B"]["ir_share"] == 0.0


def test_masks_to_schedules_cover_all_jobs_once():
    instance = generate_instance(8, 3, 0.3, 1.0, 2)
    cmp = compare_instance(instance, C.LANG_ALL, include_cpsat=False)
    plans = cmp["cells"]["all"]["schedules"]
    jobs = sorted(j for route in plans.values() for j in route)
    assert jobs == list(range(8))
    assert sorted(masks_to_schedules(instance, (255, 0, 0))[0]) == list(range(8))


def test_mechanism_table_on_the_minimal_size_limit_language_makes_vcg_undefined():
    instance = generate_instance(8, 3, 0.3, 1.0, 11)
    table = mechanism_table(instance, language=C.LANG_SIZE, b=3)                  # b = ceil(8/3): jeder Agent ist unverzichtbar
    by_rule = {r["rule"]: r for r in table["rows"]}
    assert not table["infeasible"]
    assert by_rule["B"]["defined"] is False and by_rule["B"]["undefined"]
    assert by_rule["A"]["defined"] is True                                          # Pay-as-bid braucht keine Ersatzlösung
    assert by_rule["A"]["makespan_gap"] >= -1e-9                                    # eingeschränkte Sprache: nie besser als das Optimum


def test_mechanism_table_and_lab_for_infeasible_language():
    instance = generate_instance(8, 3, 0.3, 1.0, 11)
    table = mechanism_table(instance, language=C.LANG_SIZE, b=2)
    assert table["infeasible"] and table["rows"] == []


def test_every_figure_locks_its_axes_for_touch_scrolling():
    import auction_visualization as V
    from auction_evaluation import compare_instance, language_sweep, mechanism_sweep
    from auction_bids import bundle_bid_table, language_mask
    instance = generate_instance(6, 3, 0.3, 1.0, 1)
    cmp = compare_instance(instance, C.LANG_BLOCK1, include_cpsat=False)
    table = mechanism_table(instance)
    figures = [
        V.build_schedule_figure(instance, cmp["cells"]["all"]["schedules"], cmp["reference"]),
        V.build_comparison_bars(cmp),
        V.build_bid_scatter(instance, bundle_bid_table(instance), language_mask(instance, C.LANG_ALL), (0, 0, 0)),
        V.build_scaling_charts(scaling_sweep(3, (6, 8), 0.3, 1.0)),
        V.build_language_sweep_chart(language_sweep(6, 3, 0.3, 1.0, n_instances=3)),
        V.build_payment_charts(table),
        V.build_misreport_chart(lying_lab(instance, 0), 0),
        V.build_mechanism_sweep_chart(mechanism_sweep(6, 3, 0.3, 1.0, n_instances=2)),
    ]
    for fig in figures:
        axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
        assert axes and all(a.get("fixedrange") is True for a in axes)
