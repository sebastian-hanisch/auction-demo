"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, verborgener b-Regler, Randgrößen."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import cn_constants as C

APP = Path(__file__).resolve().parent.parent / "app.py"


def _run(setup=None):
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def test_default_renders_without_exception():
    at = _run()
    assert any("Vergleich" in h.value for h in at.subheader)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    def setup(at):
        p = C.PRESETS[name]
        at.session_state["n_jobs_slider"] = p["n_jobs"]
        at.session_state["n_agents_slider"] = p["n_agents"]
        at.session_state["seed_input"] = p["seed"]
        at.session_state["language_select"] = p["language"]
        at.session_state["b_slider"] = p["b"]
        at.session_state["bidder_select"] = p["bidder"]
        at.session_state["lambda_slider"] = p["lam"]
    _run(setup)


def test_size_slider_hidden_but_state_preserved_and_clamped():
    def setup(at):
        at.session_state["language_select"] = C.LANG_SIZE
        at.session_state["b_slider"] = 6
    at = _run(setup)
    assert any(s.key == "b_slider" for s in at.slider)
    at.session_state["language_select"] = C.LANG_ALL
    at.run()
    assert not at.exception
    assert not any(s.key == "b_slider" for s in at.slider)          # kein toter Regler
    assert at.session_state["b_slider"] == 6                          # Wert bleibt erhalten
    at.session_state["n_jobs_slider"] = 4
    at.run()
    assert not at.exception
    assert at.session_state["b_slider"] <= 4                          # auf die neuen Grenzen geklemmt


def test_extreme_sizes_render():
    def small(at):
        at.session_state["n_jobs_slider"] = C.N_JOBS_MIN
        at.session_state["n_agents_slider"] = C.N_AGENTS_MAX
    _run(small)

    def large(at):
        at.session_state["n_jobs_slider"] = C.N_JOBS_MAX
        at.session_state["n_agents_slider"] = C.N_AGENTS_MAX
        at.session_state["language_select"] = C.LANG_BLOCK2
    _run(large)


def test_infeasible_size_limit_never_reachable_via_slider_state():
    def setup(at):
        at.session_state["n_jobs_slider"] = 12
        at.session_state["n_agents_slider"] = 2
        at.session_state["language_select"] = C.LANG_SIZE
        at.session_state["b_slider"] = 1               # < ceil(12/2) = 6 -> wird geklemmt
    at = _run(setup)
    assert at.session_state["b_slider"] == 6
