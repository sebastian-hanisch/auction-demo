"""Visualisierungen der Auktions-Demo: Gantt einer Zuteilung, Vergleichsbalken, Gebots-Streudiagramm, Skalierungs- und
Sprach-Sweep, Zahlungs-/Anreiz-Grafiken."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cn_constants as C
from cn_visualization import AGENT_COLORS
from auction_bids import jobs_of

COLUMN_LABELS = {
    "cnp": "Contract Net", "swap": "Task-Swap", "language": "Auktion (gewählte Sprache)",
    "all": "Auktion (alle Bündel)", "cpsat": "CP-SAT",
}
COLUMN_COLORS = {"cnp": "#B0B0B0", "swap": "#56B4E9", "language": "#D55E00", "all": "#009E73", "cpsat": "#0072B2"}
RULE_COLORS = {"A": "#D55E00", "B": "#0072B2", "C": "#CC79A7", "E": "#E69F00"}
BLOCK_COLOR = "#0072B2"
ALL_COLOR = "#009E73"
CNP_COLOR = "#B0B0B0"


def _lock(fig):
    """Achsen sperren: kein Pinch-Zoom/Ziehen, damit Touch-Geräte die Seite scrollen können (Hover-Tooltips bleiben)."""
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _intervals(instance, schedules):
    intervals = {a: [] for a in range(instance.n_agents)}
    for agent_id in range(instance.n_agents):
        position = instance.agent_start_positions[agent_id]
        free_time = 0.0
        for job_index in schedules.get(agent_id, ()):
            job = instance.jobs[job_index]
            start = free_time + instance.travel_time(position, job.position)
            end = start + job.duration
            intervals[agent_id].append((job_index, start, end))
            free_time, position = end, job.position
    return intervals


def build_schedule_figure(instance, schedules, reference=None):
    """Gantt einer Zuteilung; die gestrichelte Linie ist das exakte Optimum (Auktion mit allen Bündeln)."""
    fig = go.Figure()
    for agent_id, items in _intervals(instance, schedules).items():
        color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
        for job_index, start, end in items:
            fig.add_trace(go.Bar(
                x=[end - start], y=[f"Agent {agent_id + 1}"], base=start, orientation="h", marker=dict(color=color),
                showlegend=False, hovertext=f"Auftrag {job_index + 1}: {start:.1f} - {end:.1f} min", hoverinfo="text",
                text=f"A{job_index + 1}", textposition="inside",
            ))
    if reference is not None:
        fig.add_vline(x=reference, line_dash="dash", line_color="gray", annotation_text="Optimum", annotation_position="top")
    fig.update_layout(barmode="overlay", xaxis_title="Zeit (min)", yaxis_title=None, height=120 + 60 * instance.n_agents,
                      margin=dict(l=10, r=10, t=30, b=10))
    fig.update_yaxes(categoryorder="array", categoryarray=[f"Agent {a + 1}" for a in range(instance.n_agents)],
                     autorange="reversed")
    return _lock(fig)


def build_comparison_bars(cmp, show_language=True):
    """Makespan je Verfahren (kleiner ist besser) mit dem Optimum als Linie. Bei "Alle Bündel" entfällt die Sprach-Spalte
    (sie wäre eine Kopie der Optimum-Spalte)."""
    names = [n for n in COLUMN_LABELS if cmp["cells"].get(n) is not None and (show_language or n != "language")]
    values = [cmp["cells"][n]["makespan"] for n in names]
    fig = go.Figure(go.Bar(
        x=[COLUMN_LABELS[n] for n in names], y=values, marker_color=[COLUMN_COLORS[n] for n in names],
        text=[f"{v:.1f}" for v in values], textposition="outside",
    ))
    fig.add_hline(y=cmp["reference"], line_dash="dash", line_color="gray", annotation_text="exaktes Optimum",
                  annotation_position="bottom right")
    fig.update_yaxes(title="Makespan (min)", rangemode="tozero")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
    return _lock(fig)


def build_bid_scatter(instance, bids, mask, winner_masks):
    """Alle angebotenen Bündel: Bündelgröße gegen das BESTE (niedrigste) Gebot aller Agenten. Die Zuschlagsbündel des Optimums
    der gewählten Sprache sind groß markiert."""
    n = instance.n_jobs
    sizes = np.array([bin(m).count("1") for m in range(1 << n)])
    best = bids.min(axis=0)
    offered = mask.copy()
    offered[0] = False
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.25, 0.25, size=len(sizes))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=(sizes + jitter)[offered], y=best[offered], mode="markers", name="angebotene Bündel",
        marker=dict(size=4, color=CNP_COLOR, opacity=0.55), hoverinfo="skip",
    ))
    winners = [(a, m) for a, m in enumerate(winner_masks) if m]
    if winners:
        fig.add_trace(go.Scatter(
            x=[bin(m).count("1") for _, m in winners], y=[bids[a][m] for a, m in winners], mode="markers+text",
            name="Zuschlag", text=[f"A{a + 1}" for a, _ in winners], textposition="top center",
            marker=dict(size=13, color="#D55E00", line=dict(color="black", width=1)),
            hovertext=[f"Agent {a + 1}: Aufträge {', '.join(str(j + 1) for j in jobs_of(m, n))}" for a, m in winners],
            hoverinfo="text",
        ))
    fig.update_xaxes(title="Bündelgröße (Aufträge)", dtick=1)
    fig.update_yaxes(title="niedrigstes Gebot (min)")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    return _lock(fig)


def build_scaling_charts(rows):
    """Zwei Tafeln: Gebotszahl (alle Bündel exponentiell, 1 Block quadratisch) und Zeit der Gewinnerermittlung (log)."""
    ns = [r["n_jobs"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Anzahl Gebote (log)", "Zeit der Gewinnerermittlung (log)"))
    fig.add_trace(go.Scatter(x=ns, y=[r["bids_all"] for r in rows], mode="lines+markers", name="alle Bündel",
                             line=dict(color=ALL_COLOR, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ns, y=[r["bids_block1"] for r in rows], mode="lines+markers", name="1 Block",
                             line=dict(color=BLOCK_COLOR, width=3)), row=1, col=1)
    dp = [(r["n_jobs"], r["dp_seconds"]) for r in rows if r["dp_seconds"] is not None]
    fig.add_trace(go.Scatter(x=[a for a, _ in dp], y=[b for _, b in dp], mode="lines+markers", name="exakt, alle Bündel",
                             line=dict(color=ALL_COLOR, width=3, dash="dot"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=ns, y=[r["block_seconds"] for r in rows], mode="lines+markers", name="1 Block (polynomiell)",
                             line=dict(color=BLOCK_COLOR, width=3, dash="dot"), showlegend=False), row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Sekunden", row=1, col=2)
    fig.update_xaxes(title_text="Aufträge n", dtick=2)
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return _lock(fig)


def build_language_sweep_chart(sweep):
    """Pareto: Gebotszahl (log) gegen mittlere Lücke zum exakten Optimum je Gebotssprache."""
    fig = go.Figure()
    colors = {"cnp": CNP_COLOR, "swap": COLUMN_COLORS["swap"], "size0": "#CC79A7", "size1": "#E69F00", "block1": BLOCK_COLOR,
              "block2": "#56B4E9", "all": ALL_COLOR}
    positions = {"cnp": "top center", "swap": "top center", "size0": "top center", "size1": "bottom left", "block1": "top center",
                 "block2": "top center", "all": "bottom right"}          # gestaffelt: size1/block2/all liegen fast aufeinander
    for row in sweep["rows"]:
        fig.add_trace(go.Scatter(
            x=[row["bids"]], y=[row["mean_gap"]], mode="markers+text", text=[row["label"]], textposition=positions[row["key"]],
            marker=dict(size=14, color=colors[row["key"]], line=dict(color="black", width=1)), showlegend=False,
            hovertext=f"{row['label']}: {row['bids']} Gebote, mittlere Lücke {row['mean_gap']:.1f} %, schlechtester Fall "
                      f"{row['max_gap']:.0f} %", hoverinfo="text",
        ))
    bids = [row["bids"] for row in sweep["rows"]]
    fig.update_xaxes(title="Anzahl Gebote (log)", type="log", range=[np.log10(min(bids)) - 0.3, np.log10(max(bids)) + 0.45])
    fig.update_yaxes(title="mittlere Lücke zum Optimum (%)", rangemode="tozero")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
    return _lock(fig)


def build_payment_charts(table):
    """Zwei Tafeln je Regel: Makespan-Lücke zum Optimum und Zahlung/Kosten (fehlt bei nicht definierter Zahlung)."""
    rows = table["rows"]
    labels = [f"Regel {r['rule']}" for r in rows]
    colors = [RULE_COLORS[r["rule"]] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Makespan über dem Optimum (%)", "Zahlung / Kosten"))
    fig.add_trace(go.Bar(x=labels, y=[r["makespan_gap"] for r in rows], marker_color=colors, showlegend=False,
                         text=[f"{r['makespan_gap']:+.1f}" for r in rows], textposition="outside"), row=1, col=1)
    ratios = [r["ratio"] if r["defined"] and r["ratio"] is not None else None for r in rows]
    fig.add_trace(go.Bar(x=labels, y=ratios, marker_color=colors, showlegend=False,
                         text=["n. def." if v is None else f"{v:.2f}" for v in ratios], textposition="outside"), row=1, col=2)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return _lock(fig)


def build_misreport_chart(curves, agent):
    """Nutzen des strategischen Bieters (Zahlung - wahre Kosten) je Faktor λ, eine Linie je Regel; ehrlich = λ 1."""
    fig = go.Figure()
    for rule, rows in curves.items():
        pts = [(f, u) for f, u, _, _ in rows if u is not None]
        fig.add_trace(go.Scatter(
            x=[f for f, _ in pts], y=[u for _, u in pts], mode="lines+markers", name=f"Regel {rule}",
            line=dict(color=RULE_COLORS[rule], width=3),
        ))
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="ehrlich", annotation_position="top")
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_xaxes(title=f"Faktor λ, mit dem Agent {agent + 1} alle seine Gebote skaliert")
    fig.update_yaxes(title="Nutzen = Zahlung − wahre Kosten (min)")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.3))
    return _lock(fig)


def build_mechanism_sweep_chart(sweep):
    """Gepoolt: Anteil Gewinner mit profitabler Lüge, Anteil Instanzen mit Zahlung unter Kosten, mittlere Makespan-Lücke."""
    rules = list(sweep["rows"])
    labels = [f"Regel {r}" for r in rules]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Gewinner mit profitabler Lüge (%)", "Zahlung < Kosten (% Inst.)",
                                                        "Makespan über Optimum (%)"))
    colors = [RULE_COLORS[r] for r in rules]
    series = [[sweep["rows"][r]["lie_share"] * 100 for r in rules], [sweep["rows"][r]["ir_share"] * 100 for r in rules],
              [sweep["rows"][r]["mean_gap"] for r in rules]]
    for col, values in enumerate(series, start=1):
        fig.add_trace(go.Bar(x=labels, y=values, marker_color=colors, showlegend=False,
                             text=[f"{v:.0f}" for v in values], textposition="outside"), row=1, col=col)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return _lock(fig)
