"""
Kombinatorische Auktionen an der Kran-Auftragsvergabe – interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Siebtes Stück der "Konzepte"-Reihe, Multi-Agenten-Koordinations-Linie - ein unabhängiger Zweig vom Contract-Net-Wurzelstück.
Contract Net vergibt Aufträge einzeln und ist deshalb kurzsichtig; hier bieten die Agenten auf ganze Auftrags-BÜNDEL. Die
Demo zeigt, was das bringt (die Auktion mit allen Bündeln ist das exakte Optimum) und vor allem, was es KOSTET: exponentiell
viele Gebote, eine NP-harte Gewinnerermittlung und Zahlungsregeln, die nicht alles gleichzeitig können.
"""

import numpy as np
import streamlit as st

import cn_constants as C
from cn_presets import (
    apply_preset,
    bounds,
    clamp_dependent_state,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from cn_scenario import generate_instance
from auction_bids import (
    apply_language, bundle_bid_table, jobs_of, language_mask, min_bundle_size, n_bids,
)
from auction_evaluation import (
    compare_instance, language_sweep, lying_lab, mechanism_sweep, mechanism_table, scaling_sweep, sweep_size, verdict,
)
from auction_mechanism import RULE_LABELS, misreport, real_makespan, utility
from auction_visualization import (
    COLUMN_LABELS,
    build_bid_scatter,
    build_comparison_bars,
    build_language_sweep_chart,
    build_mechanism_sweep_chart,
    build_misreport_chart,
    build_payment_charts,
    build_scaling_charts,
    build_schedule_figure,
)
from auction_wd import wd_cpsat, winner_determination

st.set_page_config(page_title="Kombinatorische Auktionen – Sebastian Hanisch", layout="wide")

CPSAT_BUTTON_MAX_N = 10
MECHANISM_RULES = ("A", "B", "C", "E")


def _instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed):
    return generate_instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed)


@st.cache_data(show_spinner=False)
def _compute_comparison(scenario_key, language, b):
    instance = _instance(*scenario_key)
    return compare_instance(instance, language, b, cp_time_limit=C.ORTOOLS_TIME_LIMIT_SECONDS)


@st.cache_data(show_spinner=False)
def _compute_mechanism(scenario_key, language, b):
    return mechanism_table(_instance(*scenario_key), MECHANISM_RULES, language, b)


@st.cache_data(show_spinner=False)
def _compute_lab(scenario_key, bidder, language, b):
    return lying_lab(_instance(*scenario_key), bidder, MECHANISM_RULES, language=language, b=b)


@st.cache_data(show_spinner=False)
def _compute_language_sweep(n_jobs, n_agents, var, travel):
    return language_sweep(n_jobs, n_agents, var, travel)


@st.cache_data(show_spinner=False)
def _compute_scaling(n_agents, ns, var, travel):
    return scaling_sweep(n_agents, ns, var, travel)


@st.cache_data(show_spinner=False)
def _compute_mechanism_sweep(n_jobs, n_agents, var, travel):
    return mechanism_sweep(n_jobs, n_agents, var, travel, n_instances=C.N_MECHANISM_SWEEP_INSTANCES // 2)


def _fmt_int(n):
    """Tausendertrennung deutsch (Punkt)."""
    return f"{n:,}".replace(",", ".")


def _fmt_seconds(seconds):
    return f"{seconds:.1f} s" if seconds >= 1 else f"{seconds * 1000:.0f} ms"


def _start_owner(session_key, key):
    st.session_state[session_key] = key


def _delta_metric(column, label, value, reference, reference_label, help_text=None):
    """Delta-Regel des Portfolios: Wert DIESER Karte minus Referenz, niedriger ist besser."""
    delta = value - reference
    if abs(delta) < 1e-6:
        column.metric(label, f"{value:.1f} min", delta="±0.0 min", delta_color="off", help=help_text)
    else:
        column.metric(label, f"{value:.1f} min", delta=f"{delta:+.1f} min ggü. {reference_label}",
                      delta_color="inverse", help=help_text)


st.title("🔨 Kombinatorische Auktionen an der Kran-Auftragsvergabe")
st.info(
    "**Wurzel dieses Zweigs:** Contract Net (contract-net-demo). Diese Demo behebt seine Kurzsichtigkeit, indem Agenten auf "
    "**Bündel** von Aufträgen bieten - und misst, was das kostet. Die Frage \"Kann ein *gelernter* Mechanismus die dabei "
    "offen bleibenden Anreizprobleme lösen?\" (RegretNet, Dütting et al., ICML 2019) ist ausdrücklich **nicht** Teil dieses "
    "Stücks, sondern nur die Fortsetzung, die die gemessene Lücke motiviert."
)
st.markdown(
    """
Contract Net vergibt einen Auftrag nach dem anderen an den gerade billigsten Agenten - und kann später nicht mehr
korrigieren, was nur *gemeinsam* günstig gewesen wäre. Hier bieten die Agenten auf **Bündel**. Mit *allen* Bündeln ist die
Auktion trivial das exakte Optimum (das ist der Ankerpunkt, keine Überraschung). Interessant sind die **Kosten**:
Gebotszahl, Gewinnerermittlung, eingeschränkte Gebotssprachen und die Frage, ob die Agenten ehrlich bieten.
"""
)
st.caption(
    "Gemessene Kernaussage: eine Sprache mit nur wenigen Bündeln (zwei zusammenhängende Blöcke) reicht fast immer für das "
    "Optimum; ein zu kleines Bündellimit verliert dagegen auch gegen Contract Net. Die Gewinnerermittlung hat ihre Mauer bei "
    "etwa 16 Aufträgen. Und keine handentworfene Zahlungsregel erfüllt gleichzeitig Makespan-Optimum, Kostendeckung und Ehrlichkeit."
)

with st.expander("Wie funktioniert diese Demo?", expanded=True):
    st.markdown(
        r"""
**Gebot auf ein Bündel.** Das Gebot eines Agenten auf ein Bündel $S$ ist die **Fertigstellungszeit seiner besten Route**
durch $S$ ab seiner Startposition. Auf der 1-D-Schiene ist sie geschlossen bekannt (erst zum näheren Ende der Positions-Spanne
fahren, dann in einem Zug bis zum anderen): $\text{bid}(S) = \sum d_j + \tau\big((hi - lo) + \min(|s-lo|, |s-hi|)\big)$. Das
ist gegen eine Held-Karp-Teilmengen-DP geprüft. *Nur in 1-D:* in 2-D wäre schon das Gebot ein TSP-Pfad und NP-hart -
dass sich die Ergebnisse dorthin übertragen, ist vermutet, nicht gemessen.

**XOR-Gebote, Makespan.** Jeder Agent gewinnt **höchstens ein** Bündel (das leere kostet 0). Dann ist der Makespan genau das
**Maximum der Zuschlagsgebote** - und die **Gewinnerermittlung** ist eine Min-Max-Mengenpartition: jeder Auftrag genau
einmal, jeder Agent höchstens ein Bündel. Sie ist NP-hart; hier exakt per Teilmengen-DP ($k\cdot 3^n$ Schritte).

**Gebotssprachen.** *Alle Bündel*: $k(2^n-1)$ Gebote. *Größenlimit $b$*: nur Bündel bis Größe $b$ (bei XOR muss
$b \ge \lceil n/k \rceil$ sein, sonst gibt es gar keine Zuteilung). *1 Block*: zusammenhängende Abschnitte des nach Position
sortierten Auftragsstroms ($k\,n(n+1)/2$ Gebote, Gewinnerermittlung sogar polynomiell). *2 Blöcke*: Vereinigungen von
höchstens zwei Blöcken.

**Zahlungen und Anreize.** Der Auktionator *kauft* die Bearbeitung (Beschaffungsauktion). **Modellannahme:** die Kosten eines
Agenten sind seine eigene Fertigstellungszeit, 1 Geldeinheit = 1 Minute - im Fahrzeug selbst sind die Kosten öffentlich, dass
ein Agent "lügen" könnte, ist reine Modellierung. Verglichen werden vier handentworfene Regeln (Ziel der Zuteilung × Zahlung).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row_start in range(0, len(preset_names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, preset_names[row_start:row_start + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name))

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_jobs = st.slider(
        "Anzahl Aufträge", *bounds("n_jobs_slider"), key="n_jobs_slider",
        help="Bis 12 läuft alles live (Gewinnerermittlung über alle Bündel ~0.1 s). Die Skalierungs-Messung darunter geht weiter.",
    )
    n_agents = st.slider("Anzahl Agenten (Kräne)", *bounds("n_agents_slider"), key="n_agents_slider")
    duration_variability = st.slider(
        "Streuung der Auftragsdauer", *bounds("duration_variability_slider"), key="duration_variability_slider",
    )
    travel_time_per_unit = st.slider(
        "Anfahrtszeit pro Positionseinheit", *bounds("travel_time_per_unit_slider"), key="travel_time_per_unit_slider",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button(
        "🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für Auftragspositionen und -dauern.",
    )

    st.markdown("**Gebotssprache**")
    clamp_dependent_state(int(n_jobs), int(n_agents))
    language = st.selectbox(
        "Welche Bündel dürfen geboten werden?", options=list(C.LANGUAGE_LABELS), format_func=lambda k: C.LANGUAGE_LABELS[k],
        key="language_select",
        help="Weniger Bündel = weniger Gebote und eine leichtere Gewinnerermittlung - aber eventuell ein schlechteres Ergebnis.",
    )
    b_lo, b_hi = min_bundle_size(int(n_jobs), int(n_agents)), int(n_jobs)
    if language == C.LANG_SIZE and b_lo < b_hi:
        b = st.slider(
            "Bündelgrenze b", b_lo, b_hi, key="b_slider",
            help=f"Kleinstmöglich ⌈n/k⌉ = {b_lo}: darunter kann bei XOR-Geboten nicht jeder Auftrag vergeben werden.",
        )
    else:
        # Nur bei "Größenlimit" wirksam: verborgen, der Wert bleibt erhalten (und ist per clamp_dependent_state gültig).
        st.session_state["b_slider"] = st.session_state["b_slider"]
        b = st.session_state["b_slider"]

sync_query_params(
    n_jobs, n_agents, duration_variability, travel_time_per_unit, seed, language, b,
    st.session_state["bidder_select"], st.session_state["lambda_slider"],
)

scenario_key = (int(n_jobs), int(n_agents), duration_variability, travel_time_per_unit, int(seed))
lang_b = int(b) if language == C.LANG_SIZE else None
instance = _instance(*scenario_key)
bids = bundle_bid_table(instance)
mask = language_mask(instance, language, lang_b)
lang_bids = apply_language(bids, mask)

with st.spinner("Löse alle Verfahren (CP-SAT läuft bei größeren Instanzen bis zum Zeitlimit)..."):
    cmp = _compute_comparison(scenario_key, language, lang_b)
cells = cmp["cells"]
cnp_ms = cells["cnp"]["makespan"]
optimum = cmp["reference"]

# --- 📐 Vergleich ----------------------------------------------------------------

st.subheader("📐 Vergleich: Contract Net, Task-Swap, Bündelauktion, exaktes Optimum")

cp = cells["cpsat"]
if cp is not None:
    cp_note = ("beweist Optimalität" if cp.get("optimal") else "Zeitlimit erreicht: beste gefundene Lösung, nicht bewiesen") + (
        " - CP-SAT rechnet auf einem 0,1-Minuten-Raster (aufgerundet) und liegt deshalb bis etwa 1 % über dem exakten Wert"
    )
else:
    cp_note = "kein Ergebnis im Zeitlimit"

show_language = language != C.LANG_ALL       # bei "Alle Bündel" wäre die Sprach-Karte eine Kopie der Optimum-Karte
cols = st.columns(5 if show_language else 4)
cols[0].metric("Contract Net (roh)", f"{cnp_ms:.1f} min", help="Wurzelstück: Aufträge einzeln, gierig.")
_delta_metric(cols[1], "Task-Swap", cells["swap"]["makespan"], cnp_ms, "CNP",
              help_text=f"Contract Net plus paarweises Nachverhandeln ({cells['swap']['swaps']} Tausch(e)).")
next_col = 2
if show_language:
    if cells["language"] is None:
        cols[next_col].metric(
            f"Auktion: {C.LANGUAGE_SHORT[language]}", "nicht möglich",
            help=f"Mit Größenlimit b = {lang_b} < ⌈n/k⌉ = {cmp['min_bundle_size']} kann bei XOR-Geboten nicht jeder Auftrag vergeben werden.",
        )
    else:
        _delta_metric(cols[next_col], f"Auktion: {C.LANGUAGE_SHORT[language]}", cells["language"]["makespan"], cnp_ms, "CNP",
                      help_text=f"Gewinnerermittlung in {_fmt_seconds(cells['language']['seconds'])}.")
    next_col += 1
_delta_metric(cols[next_col], "Alle Bündel (exakt)", cells["all"]["makespan"], cnp_ms, "CNP",
              help_text="Exaktes Optimum dieses Fahrzeugs - Mengenpartition mit allen Bündeln, jedes Gebot = beste Route.")
if cp is not None:
    _delta_metric(cols[next_col + 1], "CP-SAT (zentral)", cp["makespan"], cnp_ms, "CNP", help_text=cp_note)
else:
    cols[next_col + 1].metric("CP-SAT (zentral)", "kein Ergebnis", help=cp_note)

bid_cols = st.columns(5 if show_language else 4)
bid_cols[0].metric("Gebote: Contract Net", _fmt_int(cells["cnp"]["bids"]), help="n · k Einzelgebote.")
if show_language:
    bid_cols[2].metric(
        f"Gebote: {C.LANGUAGE_SHORT[language]}", _fmt_int(n_bids(instance.n_jobs, instance.n_agents, language, lang_b)),
        help="Nichtleere Bündel je Agent, mal Agenten (geschlossene Formel, gegen die gezählte Tabelle geprüft).",
    )
bid_cols[next_col].metric("Gebote: alle Bündel", _fmt_int(cells["all"]["bids"]), help="k · (2^n − 1), nichtleere Bündel je Agent.")

level, code, data = verdict(cmp)
bids_lang, bids_all, bids_cnp = data["bids_language"], data["bids_all"], data["bids_cnp"]
if code == "language_worse_than_cnp":
    if cells["language"] is None:
        st.warning(
            f"⚠️ **Mit dieser Sprache gibt es keine Zuteilung**: Bündel bis Größe {lang_b} reichen bei {instance.n_agents} "
            f"Agenten nicht für {instance.n_jobs} Aufträge (kleinstmöglich b = {cmp['min_bundle_size']})."
        )
    else:
        st.warning(
            f"⚠️ **Diese Gebotssprache ist schlechter als Contract Net**: {data['language_vs_cnp']:+.1f} % (Lücke zum Optimum "
            f"{data['language_gap']:.1f} %). Die Sprache lässt genau die guten Bündel nicht zu - "
            f"{_fmt_int(bids_lang)} statt {_fmt_int(bids_all)} Gebote sparen hier zu viel."
        )
elif code == "language_worse_than_swap":
    st.warning(
        f"⚠️ **Task-Swap schlägt diese Sprache**: Lücke zum Optimum {data['swap_gap']:.1f} % (Swap) gegen "
        f"{data['language_gap']:.1f} % (Auktion)."
    )
elif code == "cnp_already_optimal":
    st.info(
        f"ℹ️ **Contract Net ist hier bereits (fast) optimal** ({data['cnp_gap']:.1f} % über dem Optimum). Bündel bringen auf "
        f"dieser Instanz nichts - kosten aber {_fmt_int(bids_all)} statt {_fmt_int(bids_cnp)} Gebote."
    )
elif language == C.LANG_ALL and code in ("language_is_optimal", "bundle_gain"):
    st.success(
        f"✅ **Mit allen Bündeln ist die Auktion exakt** - Contract Net liegt {data['cnp_gap']:.1f} % über dem Optimum, "
        f"Task-Swap {data['swap_gap']:.1f} %. Der Preis: {_fmt_int(bids_all)} statt {_fmt_int(bids_cnp)} Gebote."
    )
elif code == "language_is_optimal":
    st.success(
        f"✅ **Diese Sprache findet das Optimum** (Lücke {data['language_gap']:.1f} %) mit {_fmt_int(bids_lang)} statt "
        f"{_fmt_int(bids_all)} Geboten - Contract Net liegt {data['cnp_gap']:.1f} % darüber."
    )
elif code == "bundle_gain":
    st.success(
        f"✅ **Bündel helfen**: {-data['language_vs_cnp']:.1f} % besser als Contract Net (Lücke zum Optimum "
        f"{data['language_gap']:.1f} %) mit {_fmt_int(bids_lang)} Geboten."
    )
elif code == "cpsat_unproven":
    st.info(
        "ℹ️ **CP-SAT hat im Zeitlimit kein bewiesenes Optimum geliefert** - die Referenz bleibt trotzdem exakt (Auktion mit allen "
        "Bündeln, per DP bewiesen); nur der CP-SAT-Vergleich ist hier nicht belastbar."
    )
else:
    st.info(f"Kaum Unterschied: Auktion {data['language_vs_cnp']:+.1f} % gegenüber Contract Net.")
if code != "cpsat_unproven" and cp is not None and cp["makespan"] < optimum - 0.1:
    st.info("Hinweis: CP-SAT liegt rechnerisch unter dem Auktions-Optimum - das kann nur Rundung sein (CP-SAT rechnet ganzzahlig).")

st.plotly_chart(build_comparison_bars(cmp, show_language), width="stretch", key="comparison_bars")

available = [n for n in COLUMN_LABELS if cells.get(n) is not None and (show_language or n != "language")]
if st.session_state.get("gantt_column") not in available:
    st.session_state["gantt_column"] = "all"
gantt_choice = st.selectbox("Zeitplan anzeigen", options=available, format_func=lambda n: COLUMN_LABELS[n],
                            key="gantt_column")
st.plotly_chart(
    build_schedule_figure(instance, cells[gantt_choice]["schedules"], optimum), width="stretch",
    key=f"gantt_{gantt_choice}",
)

st.markdown("---")

# --- Gebotstabelle ---------------------------------------------------------------

st.markdown("## 📦 Gebote auf Bündel")
k, n = instance.n_agents, instance.n_jobs
wd_all = winner_determination(bids, "max")
wd_lang = winner_determination(lang_bids, "max")
st.caption(
    f"Jeder der {k} Agenten gibt für jedes angebotene Bündel eine Fertigstellungszeit ab: bei {n} Aufträgen sind das "
    f"$2^{{{n}}} - 1$ = {_fmt_int(2 ** n - 1)} nichtleere Bündel je Agent, also {_fmt_int(cells['all']['bids'])} Gebote - "
    f"gegen nur {_fmt_int(cells['cnp']['bids'])} Einzelgebote bei Contract Net."
    + ("" if language == C.LANG_ALL else
       f" In der gewählten Sprache werden davon {_fmt_int(n_bids(n, k, language, lang_b))} angeboten.")
)
scatter_masks = wd_lang.masks if wd_lang.feasible else wd_all.masks
st.plotly_chart(build_bid_scatter(instance, bids, mask, scatter_masks), width="stretch", key="bid_scatter")
if not wd_lang.feasible:
    st.caption("Die gewählte Sprache hat keine Zuteilung - markiert sind die Zuschlagsbündel des Optimums (alle Bündel).")
else:
    st.caption("Grau: das niedrigste Gebot je angebotenem Bündel. Orange: die Zuschlagsbündel der gewählten Sprache.")

synergy_rows = []
for agent, m in enumerate(scatter_masks):
    jobs = jobs_of(m, n)
    if len(jobs) >= 2:
        singles = sum(bids[agent][1 << j] for j in jobs)
        synergy_rows.append(
            f"Agent {agent + 1} übernimmt Aufträge {', '.join(str(j + 1) for j in jobs)}: Bündelgebot "
            f"**{bids[agent][m]:.1f} min** statt der Summe seiner Einzelgebote **{singles:.1f} min** "
            f"(**{singles - bids[agent][m]:.1f} min Synergie** - die Anfahrt wird geteilt)."
        )
if synergy_rows:
    st.markdown("**Wo die Synergie steckt.** " + " ".join(synergy_rows))

st.markdown("---")

# --- Gewinnerermittlung ----------------------------------------------------------

st.markdown("## 🧮 Gewinnerermittlung: die Mauer")
st.caption(
    "Welche Bündel gewinnen? Eine Min-Max-Mengenpartition - NP-hart. Exakt per Teilmengen-DP: $k \\cdot 3^n$ Schritte, aber die "
    "Gebotstabelle wächst mit $2^n$. Die Ein-Block-Sprache dagegen hat eine polynomielle Gewinnerermittlung (DP über "
    "Präfix × Agentenmenge)."
)
w1, w2, w3 = st.columns(3)
w1.metric("Zeit, alle Bündel (exakt)", _fmt_seconds(wd_all.seconds))
w2.metric("Teilmengen-Paare", _fmt_int(wd_all.pairs), help="k · 3^n ausgewertete Paare (Bündel, Rest).")
w3.metric("Zeit, gewählte Sprache", _fmt_seconds(cells["language"]["seconds"]) if cells["language"] else "–")

cpsat_key = (scenario_key, language, lang_b)
if n > CPSAT_BUTTON_MAX_N:
    st.caption(
        f"CP-SAT als Mengenpartitions-Löser auf der Gebotstabelle gibt es nur bis {CPSAT_BUTTON_MAX_N} Aufträge - darüber "
        "scheitert diese naive Formulierung schon bei 11-12 Aufträgen."
    )
elif st.session_state.get("wd_cpsat_owner") != cpsat_key:
    st.button(
        "⚔️ Gewinnerermittlung: Teilmengen-DP gegen CP-SAT-Mengenpartition", on_click=_start_owner,
        args=("wd_cpsat_owner", cpsat_key), key="wd_cpsat_start",
        help="CP-SAT löst dieselbe Mengenpartition auf der (gewählten) Gebotstabelle (Zeitlimit 20 s).",
    )
else:
    with st.spinner("CP-SAT löst die Mengenpartition..."):
        cp_value, cp_optimal, cp_seconds = wd_cpsat(lang_bids, 20.0)
    d1, d2 = st.columns(2)
    d1.metric("Teilmengen-DP", f"{wd_lang.value:.1f} min", help=f"{_fmt_seconds(wd_lang.seconds)}, bewiesen exakt.")
    d2.metric("CP-SAT auf der Gebotstabelle", f"{cp_value:.1f} min" if np.isfinite(cp_value) else "kein Ergebnis",
              help=f"{_fmt_seconds(cp_seconds)}, {'bewiesen optimal' if cp_optimal else 'Zeitlimit erreicht'}.")

scaling_choice = st.selectbox(
    "Skalierungs-Messung bis n =", options=[12, 16, 20, 30], index=1, key="scaling_limit",
    help="Alle Bündel exakt bis n = 16, darüber nur die Ein-Block-Auktion gegen ein unbewiesenes CP-SAT-Ergebnis.",
)
scaling_ns = tuple(x for x in C.SCALING_NS if x <= scaling_choice)
scaling_key = (int(n_agents), scaling_ns, duration_variability, travel_time_per_unit)
if st.session_state.get("scaling_owner") != scaling_key:
    est = (1 if scaling_choice <= 12 else 11) + 32 * sum(1 for x in scaling_ns if x > C.EXACT_DP_MAX_N)
    st.button(
        f"📈 Skalierungs-Messung starten (ca. {est} s)", on_click=_start_owner, args=("scaling_owner", scaling_key),
        key="scaling_start",
        help="Feste Instanzen (Seeds unabhängig vom Demo-Seed): Gebotszahl und Zeit der Gewinnerermittlung je n.",
    )
else:
    with st.spinner("Messe die Skalierung..."):
        rows = _compute_scaling(*scaling_key)
    st.plotly_chart(build_scaling_charts(rows), width="stretch", key="scaling_chart")
    table = {
        "n": [r["n_jobs"] for r in rows], "Gebote (alle Bündel)": [_fmt_int(r["bids_all"]) for r in rows],
        "Gebote (1 Block)": [_fmt_int(r["bids_block1"]) for r in rows],
        "Zeit exakt": ["–" if r["dp_seconds"] is None else _fmt_seconds(r["dp_seconds"]) for r in rows],
        "Zeit 1 Block": [_fmt_seconds(r["block_seconds"]) for r in rows],
        "Lücke 1 Block": [f"{r['block_gap']:+.1f} %" for r in rows], "Lücke CNP": [f"{r['cnp_gap']:+.1f} %" for r in rows],
        "Referenz": [r["reference"] for r in rows],
    }
    st.table(table)
    st.caption(
        "**Exakt bis n ≈ 14 (bewiesen bis 16), darüber beste bekannte Lösung.** Bei n > 16 ist die Referenz das Ergebnis von "
        "CP-SAT im Zeitlimit - meist *nicht bewiesen optimal*. Dass die polynomielle Ein-Block-Auktion dort besser (negative "
        "Lücke) als CP-SAT abschneidet, kann ein Artefakt des einfachen CP-SAT-Modells sein und ist nur eine Größenordnung "
        "(wenige Instanzen)."
    )

st.markdown("---")

# --- Sprach-Sweep ---------------------------------------------------------------

st.markdown("## ⚖️ Gebote gegen Lücke: welche Sprache reicht?")
sweep_key = (int(n_jobs), int(n_agents), duration_variability, travel_time_per_unit)
if st.session_state.get("lang_sweep_owner") != sweep_key:
    est = max(1, round(0.5 * 2.0 ** (int(n_jobs) - 8) * (int(n_agents) / 3) ** 0.8))
    st.button(
        f"📊 Sprach-Sweep starten ({sweep_size(int(n_jobs))} feste Instanzen, ca. {est} s)", on_click=_start_owner,
        args=("lang_sweep_owner", sweep_key), key="lang_sweep_start",
        help="Feste Instanzen (Seeds unabhängig vom Demo-Seed): mittlere Lücke zum exakten Optimum je Gebotssprache.",
    )
else:
    with st.spinner("Rechne den Sprach-Sweep..."):
        sweep = _compute_language_sweep(*sweep_key)
    st.plotly_chart(build_language_sweep_chart(sweep), width="stretch", key="language_sweep_chart")
    st.table({
        "Verfahren": [r["label"] for r in sweep["rows"]], "Gebote": [_fmt_int(r["bids"]) for r in sweep["rows"]],
        "mittlere Lücke": [f"{r['mean_gap']:.1f} %" for r in sweep["rows"]],
        "Median": [f"{r['median_gap']:.1f} %" for r in sweep["rows"]],
        "schlechtester Fall": [f"{r['max_gap']:.0f} %" for r in sweep["rows"]],
        "exakt optimal": [f"{r['optimal_frac'] * 100:.0f} %" for r in sweep["rows"]],
        "schlechter als CNP": [f"{r['lose_vs_cnp'] * 100:.0f} %" for r in sweep["rows"]],
    })
    size0 = next(r for r in sweep["rows"] if r["key"] == "size0")
    st.caption(
        f"{sweep['n_instances']} feste Instanzen. Das kleinstmögliche Größenlimit b = ⌈n/k⌉ = {sweep['b0']} spart die "
        f"meisten Gebote ({_fmt_int(size0['bids'])}), liegt im schlechtesten Fall aber {size0['max_gap']:.0f} % über dem Optimum "
        f"und ist auf {size0['lose_vs_cnp'] * 100:.0f} % der Instanzen schlechter als Contract Net."
    )

st.markdown("---")

# --- Anreize ---------------------------------------------------------------------

st.markdown("## 🎭 Anreize und Zahlungen")
st.warning(
    "**Modellannahme (nicht Teil des Fahrzeugs):** Der Auktionator *kauft* die Bearbeitung (Beschaffungsauktion). Die Kosten eines "
    "Agenten sind seine eigene Fertigstellungszeit, 1 Geldeinheit = 1 Minute. Dass ein Agent seine Gebote verfälschen könnte, ist "
    "Modellierung - im Fahrzeug selbst sind die Kosten öffentlich. Gemessen wird nur der **Ein-Bieter-Regret bei ehrlichen "
    "anderen** - eine *untere Schranke* des echten Regrets, nicht der volle Gleichgewichtsbegriff. Die Regeln laufen auf den "
    "Bündeln, die die gewählte Gebotssprache erlaubt (der Vergleichswert 'Optimum' bleibt das exakte Optimum aller Bündel)."
)
mech = _compute_mechanism(scenario_key, language, lang_b)
if mech["infeasible"]:
    st.warning(
        f"⚠️ **Mit dieser Gebotssprache gibt es keine Zuteilung** - also auch keine Zahlungen zu vergleichen (Größenlimit "
        f"b = {lang_b} < ⌈n/k⌉ = {cmp['min_bundle_size']}). Regler b erhöhen oder eine andere Sprache wählen."
    )
else:
    rows_by_rule = {r["rule"]: r for r in mech["rows"]}
    table_rows = {
        "Regel": [RULE_LABELS[r] for r in MECHANISM_RULES],
        "Makespan": [f"{rows_by_rule[r]['makespan']:.1f} min" for r in MECHANISM_RULES],
        "über Optimum": [f"{rows_by_rule[r]['makespan_gap']:+.1f} %" for r in MECHANISM_RULES],
        "Zahlung / Kosten": [
            f"{rows_by_rule[r]['ratio']:.2f}" if rows_by_rule[r]["defined"] and rows_by_rule[r]["ratio"] is not None else "nicht definiert"
            for r in MECHANISM_RULES
        ],
        "kleinste Rendite": [
            f"{rows_by_rule[r]['min_margin']:+.1f} min" if rows_by_rule[r]["defined"] else "–" for r in MECHANISM_RULES
        ],
    }
    st.table(table_rows)
    st.plotly_chart(build_payment_charts(mech), width="stretch", key="payment_charts")
    monopoly = rows_by_rule["B"]
    if not monopoly["defined"]:
        names = ", ".join(f"Agent {a + 1}" for a in monopoly["undefined"])
        st.warning(
            f"⚠️ **VCG ist hier nicht definiert:** ohne {names} gibt es keine zulässige Zuteilung (Monopol) - die Zahlung wäre "
            "unbeschränkt. Es wird keine Ersatzzahl erfunden."
        )
    neg = [r for r in MECHANISM_RULES if rows_by_rule[r]["defined"] and rows_by_rule[r]["min_margin"] is not None
           and rows_by_rule[r]["min_margin"] < -1e-9]
    if neg:
        st.warning(
            "⚠️ **Zahlung unter den Kosten (Individuelle Rationalität verletzt)** bei " + ", ".join(f"Regel {r}" for r in neg)
            + ": ein Agent würde draufzahlen und die Teilnahme ablehnen."
        )
    st.caption(
        "Regel B (Summenziel + VCG) ist wahrheitsgetreu - aber Summenziel ist nicht Makespan-Ziel: sie minimiert die *Gesamtzeit* aller "
        "Agenten und lädt oft alles auf einen. Regel E klebt die VCG-Formel auf die Makespan-Zuteilung - dann stimmt der Anreiz "
        "nicht mehr. Regel A zahlt einfach das Gebot - dann lohnt es sich aufzublähen."
    )

    st.markdown("**Lügen-Labor: ein strategischer Bieter, alle anderen ehrlich**")
    lab_a, lab_b = st.columns(2)
    with lab_a:
        bidder = st.selectbox("Bieter", options=list(range(int(n_agents))), format_func=lambda a: f"Agent {a + 1}",
                              key="bidder_select")
    with lab_b:
        lam = st.slider(
            "Faktor λ (alle eigenen Gebote werden mit λ multipliziert)", *bounds("lambda_slider"), key="lambda_slider", step=0.05,
            help="λ < 1: unterbieten, λ > 1: aufblähen. Der Nutzen wird gegen die WAHREN Kosten bewertet.",
        )
    true_bids = mech["true_bids"]
    lab_cols = st.columns(len(MECHANISM_RULES))
    for col, rule in zip(lab_cols, MECHANISM_RULES):
        honest = misreport(true_bids, bidder, rule, 1.0)
        lied = misreport(true_bids, bidder, rule, float(lam))
        honest_u = utility(true_bids, honest, bidder) if honest.feasible else None
        lied_u = utility(true_bids, lied, bidder) if lied.feasible else None
        if honest_u is None or lied_u is None:
            col.metric(RULE_LABELS[rule], "nicht definiert", help="Zahlung nicht definiert (Monopol) oder keine Zuteilung.")
            continue
        changed = lied.masks != honest.masks
        col.metric(
            RULE_LABELS[rule], f"{lied_u:+.1f} min", delta=f"{lied_u - honest_u:+.1f} min ggü. ehrlich",
            delta_color="normal", help=(
                f"Ehrlich: {honest_u:+.1f} min. Zuteilung {'ändert sich' if changed else 'bleibt gleich'}; echter Makespan "
                f"{real_makespan(true_bids, lied.masks):.1f} min (ehrlich {real_makespan(true_bids, honest.masks):.1f})."
            ),
        )
    lab = _compute_lab(scenario_key, int(bidder), language, lang_b)
    st.plotly_chart(build_misreport_chart(lab, int(bidder)), width="stretch", key=f"misreport_chart_{bidder}")
    st.caption(
        "Liegt eine Linie bei einem anderen λ höher als bei 'ehrlich' (λ = 1), lohnt sich Lügen. Bei Regel B (VCG) liegt kein "
        "λ über dem ehrlichen Nutzen. Das Gitter ist eine Untergrenze: der wahre Regret ist mindestens so groß."
    )


regret_key = (int(n_jobs), int(n_agents), duration_variability, travel_time_per_unit)
if int(n_jobs) > C.MECHANISM_SWEEP_MAX_N:
    st.caption(f"Regret-Sweep nur bis {C.MECHANISM_SWEEP_MAX_N} Aufträge (die Entfernungs-DPs der Zahlungen kosten sonst zu viel).")
elif st.session_state.get("regret_sweep_owner") != regret_key:
    est = max(2, round(1.6 * 2.15 ** (int(n_jobs) - 6) * (int(n_agents) / 3) ** 2.4))
    st.button(
        f"📊 Regret-Sweep starten ({C.N_MECHANISM_SWEEP_INSTANCES // 2} feste Instanzen, ca. {est} s)", on_click=_start_owner,
        args=("regret_sweep_owner", regret_key), key="regret_sweep_start",
        help="Feste Instanzen: je Regel Makespan-Lücke, IR-Verletzungen, Gewinner mit profitabler Lüge, mittlerer Regret.",
    )
else:
    with st.spinner("Rechne den Regret-Sweep..."):
        msweep = _compute_mechanism_sweep(*regret_key)
    st.plotly_chart(build_mechanism_sweep_chart(msweep), width="stretch", key="mechanism_sweep_chart")
    st.table({
        "Regel": [RULE_LABELS[r] for r in MECHANISM_RULES],
        "Makespan über Optimum": [f"{msweep['rows'][r]['mean_gap']:+.1f} %" for r in MECHANISM_RULES],
        "Zahlung / Kosten": [f"{msweep['rows'][r]['ratio']:.2f}" if msweep["rows"][r]["ratio"] else "–" for r in MECHANISM_RULES],
        "Zahlung < Kosten (Inst.)": [f"{msweep['rows'][r]['ir_share'] * 100:.0f} %" for r in MECHANISM_RULES],
        "VCG nicht definiert (Inst.)": [f"{msweep['rows'][r]['undefined_share'] * 100:.0f} %" for r in MECHANISM_RULES],
        "Gewinner mit profitabler Lüge": [f"{msweep['rows'][r]['lie_share'] * 100:.0f} %" for r in MECHANISM_RULES],
        "mittlerer Regret (% der Kosten)": [f"{msweep['rows'][r]['mean_regret_pct']:.1f} %" for r in MECHANISM_RULES],
    })
    good = [
        r for r in MECHANISM_RULES
        if msweep["rows"][r]["mean_gap"] < 2.0 and msweep["rows"][r]["ir_share"] == 0.0 and msweep["rows"][r]["lie_share"] < 0.05
    ]
    verdict_text = (
        f"Regel {', '.join(good)} liegt hier gleichzeitig nahe am Makespan-Optimum, deckt die Kosten und belohnt kein Lügen."
        if good else
        "Keine der vier Regeln liegt gleichzeitig nahe am Makespan-Optimum, deckt die Kosten und belohnt ehrliches Bieten."
    )
    st.caption(
        f"{msweep['n_instances']} feste Instanzen, alle Bündel, Faktoren-Gitter λ ∈ {{0.7, 1.3, 1.6, 2.0, 3.0}}. " + verdict_text
    )

with st.expander("🔭 Ausblick: warum das eine gelernte Fortsetzung motiviert (nicht gezeigt)"):
    st.markdown(
        f"""
Die Messungen oben zeigen eine **Lücke**, keine Lösung: keine Handregel erfüllt gleichzeitig ein Makespan-Optimum,
Kostendeckung und Ehrlichkeit. **RegretNet** (Dütting et al., ICML 2019) lernt Zuteilung und Zahlung als neuronales Netz, das
den Regret als Nebenbedingung minimiert. Ausdrücklich **nicht gezeigt** ist, dass ein gelernter Mechanismus hier besser wäre.
Zwei ehrliche Lücken: RegretNet ist ursprünglich *verkaufsseitig* (Erlösmaximierung), hier ist es *Beschaffung* mit
Makespan-Ziel; und die Gebotstabelle hat $k \\cdot 2^n$ Zahlen ({_fmt_int(cells['all']['bids'])} bei diesen Einstellungen) - ein
gelernter Mechanismus müsste damit umgehen oder die Sprache einschränken.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gebot.** $b_a(S) = \sum_{j \in S} d_j + \tau\big((hi_S - lo_S) + \min(|s_a - lo_S|,\,|s_a - hi_S|)\big)$, $b_a(\emptyset)=0$.
$lo_S, hi_S$: kleinste und größte Position im Bündel, $s_a$: Startposition, $\tau$: Fahrzeit pro Einheit.

**Gewinnerermittlung (Makespan, XOR).**
$$\min_{S_1,\dots,S_k}\ \max_a\, b_a(S_a) \quad \text{s.t. } S_a \text{ disjunkt},\ \textstyle\bigcup_a S_a = \{1,\dots,n\}.$$
DP: $F_a[T] = \min_{S \subseteq T} \max\big(F_{a-1}[T \setminus S],\, b_a(S)\big)$, Aufwand $k \cdot 3^n$. Für das Summenziel steht
$+$ statt $\max$.

**Ein-Block-Sprache.** Bündel = zusammenhängende Abschnitte des nach Position sortierten Auftragsstroms: $k\,n(n+1)/2$ Gebote,
DP über (Präfix, benutzte Agentenmenge), polynomiell.

**Gebotszahlen.** Alle Bündel $k(2^n-1)$; Größenlimit $k\sum_{s\le b}\binom{n}{s}$; zwei Blöcke $k\sum_{i \le 4}\binom{n}{i}$.

**VCG (Clarke).** $p_a = W_{-a} - \big(W - b_a(S_a)\big)$ mit $W$ = Wert der Summen-Zuteilung, $W_{-a}$ = Wert ohne Agent $a$;
nicht definiert, wenn ohne $a$ keine zulässige Zuteilung existiert. Wahrheitsgetreu nur für das Summenziel.

**Regret.** $\text{regret}_a = \max_{\lambda}\big(u_a(\lambda b_a) - u_a(b_a)\big)$ bei ehrlichen anderen, $u_a$ = Zahlung − wahre Kosten
des Zuschlagsbündels. Das Gitter über $\lambda$ macht ihn zu einer unteren Schranke.

Implementiert in `auction_bids.py`, `auction_wd.py`, `auction_mechanism.py` und `auction_evaluation.py`.
        """
    )

with st.expander("🧪 Was nicht funktioniert hat (einmalige Messung im Prototyp, kein App-Abschnitt)"):
    st.markdown(
        """
Zwei naheliegende Näherungen der Gewinnerermittlung wurden getestet und verworfen (Lücke zum exakten Optimum, gepoolt über
feste Instanzen; einmalige Messung, Größenordnung, keine Garantie):

- **Je Agent nur die *m* billigsten Bündel** anbieten lassen: **+98 % bis +267 %** über dem Optimum.
- **Dichte-Greedy als Gewinnerermittlung** (Bündel nach Gebot je Auftrag ordnen und vergeben): **+136 % bis +220 %**.

Beides ist weit schlechter als Contract Net - die Struktur des Problems (zusammenhängende Bündel im 1-D-Strom) gehört in die
Gebotssprache, nicht in eine Heuristik oben drauf.
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
