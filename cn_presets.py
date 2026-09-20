"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem OR-Demo-Portfolio, siehe
constraint-programming-demo/csp_presets.py). Zusätzlich zu den fünf Szenario-Reglern: Gebotssprache, Bündelgrenze b sowie
strategischer Bieter und Fehlmeldungs-Faktor λ der Anreiz-Sektion."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import cn_constants as C
from auction_bids import min_bundle_size


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "n_jobs_slider": SettingSpec("jobs", int, C.DEFAULT_N_JOBS, C.N_JOBS_MIN, C.N_JOBS_MAX),
    "n_agents_slider": SettingSpec("agents", int, C.DEFAULT_N_AGENTS, C.N_AGENTS_MIN, C.N_AGENTS_MAX),
    "duration_variability_slider": SettingSpec(
        "var", float, C.DEFAULT_DURATION_VARIABILITY, C.DURATION_VARIABILITY_MIN, C.DURATION_VARIABILITY_MAX
    ),
    "travel_time_per_unit_slider": SettingSpec(
        "travel", float, C.DEFAULT_TRAVEL_TIME_PER_UNIT, C.TRAVEL_TIME_PER_UNIT_MIN, C.TRAVEL_TIME_PER_UNIT_MAX
    ),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
    "language_select": SettingSpec("lang", str, C.DEFAULT_LANGUAGE),
    "b_slider": SettingSpec("b", int, C.DEFAULT_B, 1, C.N_JOBS_MAX),
    "bidder_select": SettingSpec("bidder", int, 0, 0, C.N_AGENTS_MAX - 1),
    "lambda_slider": SettingSpec("lam", float, C.DEFAULT_LAMBDA, C.LAMBDA_MIN, C.LAMBDA_MAX),
}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default
    if "force_regen" not in st.session_state:
        st.session_state["force_regen"] = False


def clamp_dependent_state(n_jobs, n_agents):
    """Bündelgrenze b und Bieter hängen von n und k ab: vor dem Instanziieren der Widgets auf die aktuellen Grenzen klemmen
    (auch wenn der b-Regler gerade verborgen ist - sein Wert bleibt erhalten, wird aber nie ungültig)."""
    lo, hi = min_bundle_size(n_jobs, n_agents), n_jobs
    st.session_state["b_slider"] = min(max(int(st.session_state["b_slider"]), lo), hi)
    st.session_state["bidder_select"] = min(max(int(st.session_state["bidder_select"]), 0), n_agents - 1)


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    if st.session_state.get("language_select") not in C.LANGUAGE_LABELS:
        st.session_state["language_select"] = C.DEFAULT_LANGUAGE
    st.session_state["permalink_loaded"] = True


def sync_query_params(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed, language, b, bidder, lam):
    try:
        st.query_params["jobs"] = str(int(n_jobs))
        st.query_params["agents"] = str(int(n_agents))
        st.query_params["var"] = str(duration_variability)
        st.query_params["travel"] = str(travel_time_per_unit)
        st.query_params["seed"] = str(int(seed))
        st.query_params["lang"] = str(language)
        st.query_params["b"] = str(int(b))
        st.query_params["bidder"] = str(int(bidder))
        st.query_params["lam"] = str(lam)
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    st.session_state["n_jobs_slider"] = p["n_jobs"]
    st.session_state["n_agents_slider"] = p["n_agents"]
    st.session_state["duration_variability_slider"] = p["duration_variability"]
    st.session_state["travel_time_per_unit_slider"] = p["travel_time_per_unit"]
    st.session_state["seed_input"] = p["seed"]
    st.session_state["language_select"] = p["language"]
    st.session_state["b_slider"] = p["b"]
    st.session_state["bidder_select"] = p["bidder"]
    st.session_state["lambda_slider"] = p["lam"]
    st.session_state["force_regen"] = True


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
    st.session_state["force_regen"] = True
