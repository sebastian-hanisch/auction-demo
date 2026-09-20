"""Defaults, Slider-Grenzen und Presets für die Kombinatorische-Auktions-Demo.
Die Szenario-Konstanten (POSITION_RANGE_MAX ... SPIKE_MULTIPLIER) sind wortgleich aus dcop-demo/ladcop-demo übernommen -
dasselbe Vehikel -, da `cn_scenario.py` sie importiert; alles Übrige ist neu."""

DEFAULT_N_JOBS = 8
DEFAULT_N_AGENTS = 3
DEFAULT_DURATION_VARIABILITY = 0.3
DEFAULT_TRAVEL_TIME_PER_UNIT = 1.0
DEFAULT_SEED = 9

# Bis 12 Aufträge läuft alles live (Gewinnerermittlung über alle Bündel ~0.07 s, CP-SAT ~0.5 s). Der exakte
# Teilmengen-DP hat seine Mauer bei n ~ 16 (der Skalierungs-Sweep zeigt sie per Knopf).
N_JOBS_MIN, N_JOBS_MAX = 4, 12
N_AGENTS_MIN, N_AGENTS_MAX = 2, 4
DURATION_VARIABILITY_MIN, DURATION_VARIABILITY_MAX = 0.0, 1.0
TRAVEL_TIME_PER_UNIT_MIN, TRAVEL_TIME_PER_UNIT_MAX = 0.2, 2.0

POSITION_RANGE_MAX = 20.0
DURATION_BASE_RANGE = (5, 15)
SPIKE_PROBABILITY_SCALE = 0.4
SPIKE_MULTIPLIER = 4.0

# CP-SAT-Referenzlauf: harte Zeitgrenze (ab n ~ 14 oft unbewiesen -> "beste gefundene Lösung").
ORTOOLS_TIME_LIMIT_SECONDS = 5.0
ORTOOLS_SWEEP_TIME_LIMIT_SECONDS = 10.0

# --- Gebotssprachen -------------------------------------------------------------------------------
LANG_ALL = "all"          # jedes Bündel (2^n - 1 Gebote je Agent)
LANG_SIZE = "size"        # Bündel bis zu einer Größe b
LANG_BLOCK1 = "block1"    # zusammenhängende Blöcke im nach Position sortierten Auftragsstrom
LANG_BLOCK2 = "block2"    # Vereinigungen von höchstens zwei solchen Blöcken
LANGUAGE_LABELS = {
    LANG_ALL: "Alle Bündel",
    LANG_SIZE: "Größenlimit b",
    LANG_BLOCK1: "1 Block (zusammenhängend)",
    LANG_BLOCK2: "2 Blöcke",
}
LANGUAGE_SHORT = {LANG_ALL: "Alle Bündel", LANG_SIZE: "Größenlimit b", LANG_BLOCK1: "1 Block", LANG_BLOCK2: "2 Blöcke"}
DEFAULT_LANGUAGE = LANG_ALL
DEFAULT_B = 4                        # Startwert des Bündelgrenzen-Reglers (wird auf ceil(n/k)..n geklemmt)

# --- Sweeps ---------------------------------------------------------------------------------------
HELDOUT_SEED_BASE = 100_000          # feste Sweep-Instanzen, unabhängig vom Demo-Seed
N_SWEEP_INSTANCES = 30               # n <= 10
N_SWEEP_INSTANCES_MID = 20           # n <= 12
SCALING_NS = (8, 10, 12, 14, 16, 20, 24, 30)
SCALING_INSTANCES = 5
EXACT_DP_MAX_N = 16                  # bis hierher beweist der Teilmengen-DP das Optimum in Sekunden
MECHANISM_SWEEP_MAX_N = 10
N_MECHANISM_SWEEP_INSTANCES = 30

# --- Strategisches Bieten (Ein-Bieter-Regret) -----------------------------------------------------
# Strategieraum: gleichmäßiges Skalieren aller eigenen Gebote; das Gitter ist eine Untergrenze des Regrets.
LAMBDA_MIN, LAMBDA_MAX, DEFAULT_LAMBDA = 0.5, 3.0, 1.5
LAMBDA_GRID = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0, 3.0)

# --- Verdict-Kaskade ------------------------------------------------------------------------------
LANGUAGE_WORSE_THAN_CNP_PCT = 1.0       # Sprache > +1 % schlechter als Contract Net
LANGUAGE_WORSE_THAN_SWAP_PCT = 1.0      # ... als Task-Swap
CNP_ALREADY_OPTIMAL_PCT = 0.5           # Contract Net höchstens 0.5 % über der Referenz
LANGUAGE_IS_OPTIMAL_PCT = 0.5           # Sprache höchstens 0.5 % über der Referenz
BUNDLE_GAIN_SUCCESS_PCT = 3.0           # Sprache mindestens 3 % besser als Contract Net

_BASE = {"duration_variability": DEFAULT_DURATION_VARIABILITY, "travel_time_per_unit": DEFAULT_TRAVEL_TIME_PER_UNIT,
         "language": LANG_ALL, "b": DEFAULT_B, "bidder": 0, "lam": DEFAULT_LAMBDA}

PRESETS = {
    "Bündel fangen die Synergie": {**_BASE, "n_jobs": 10, "n_agents": 3, "seed": 9},
    "Alle Bündel = Optimum": {**_BASE, "n_jobs": 12, "n_agents": 3, "seed": 9},
    "Kleines Bündellimit verliert": {**_BASE, "n_jobs": 12, "n_agents": 3, "seed": 58, "language": LANG_SIZE, "b": 4},
    "Zusammenhängende Blöcke reichen": {**_BASE, "n_jobs": 12, "n_agents": 3, "seed": 11, "language": LANG_BLOCK1},
    "Summenziel ≠ Makespan": {**_BASE, "n_jobs": 10, "n_agents": 3, "seed": 14},
    "VCG zahlt drauf": {**_BASE, "n_jobs": 8, "n_agents": 3, "seed": 11},
    "Pay-as-bid: Bieter gewinnt": {**_BASE, "n_jobs": 8, "n_agents": 3, "seed": 7, "bidder": 0, "lam": 1.5},
    "Contract Net reicht schon": {**_BASE, "n_jobs": 6, "n_agents": 4, "seed": 0},
}
PRESET_HELP = {
    "Bündel fangen die Synergie": "Contract Net liegt 60 % über dem Optimum, Task-Swap noch 25 %. Mit Bündelgeboten findet die "
        "Auktion das Optimum - weil Aufträge, die nebeneinander liegen, gemeinsam billiger sind als einzeln.",
    "Alle Bündel = Optimum": "12 Aufträge, 3 Agenten: 12 285 Gebote und trotzdem eine Gewinnerermittlung in Bruchteilen einer "
        "Sekunde. Die Auktion mit allen Bündeln ist das exakte Optimum - der Preis sind die exponentiell vielen Gebote.",
    "Kleines Bündellimit verliert": "Bündel bis Größe 4 (dem kleinstmöglichen Limit): 2 379 statt 12 285 Gebote - aber das Ergebnis "
        "ist 43 % über dem Optimum und schlechter als Contract Net und Task-Swap.",
    "Zusammenhängende Blöcke reichen": "Nur zusammenhängende Abschnitte des nach Position sortierten Auftragsstroms: 234 statt "
        "12 285 Gebote - und trotzdem das Optimum. Die Gewinnerermittlung ist hier sogar polynomiell.",
    "Summenziel ≠ Makespan": "VCG ist wahrheitsgetreu, weil es die Gesamtzeit minimiert - aber die Gesamtzeit ist nicht der "
        "Makespan: die Summen-Zuteilung lädt fast alles auf einen Agenten und liegt 143 % über dem Optimum, "
        "schlechter als rohes Contract Net.",
    "VCG zahlt drauf": "Ehrlich bieten lohnt sich bei VCG, aber der Auktionator zahlt 22 % mehr als die Kosten. Die "
        "Clarke-Formel auf der Makespan-Zuteilung dagegen zahlt einem Agenten weniger als seine Kosten.",
    "Pay-as-bid: Bieter gewinnt": "Agent 1 bläht alle seine Gebote um 50 % auf - die Zuteilung bleibt gleich, er bekommt aber "
        "17 Minuten mehr. Pay-as-bid belohnt das Lügen (Lügen-Labor, Regel A).",
    "Contract Net reicht schon": "Auf dieser Instanz ist Contract Net bereits optimal: Bündel bringen nichts, kosten aber "
        "252 statt 24 Gebote. Nicht jede Instanz braucht eine Auktion.",
}


def _band(value, tolerance):
    return (value - tolerance, value + tolerance)


# Erwartete Messwerte je Preset (mit dem ausgelieferten Code kalibriert); tests/test_presets.py prüft sie.
# Prozentwerte +-4 Punkte, Zahlung/Kosten +-0.05, Minuten +-2.
PRESET_EXPECTED_BANDS = {
    "Bündel fangen die Synergie": {"verdict": "language_is_optimal", "cnp_gap": _band(60.0, 4), "swap_gap": _band(25.0, 4),
                                   "language_gap": _band(0.0, 1)},
    "Alle Bündel = Optimum": {"verdict": "language_is_optimal", "cnp_gap": _band(42.0, 4), "language_gap": _band(0.0, 1),
                              "bids_language": (12285, 12285)},
    "Kleines Bündellimit verliert": {"verdict": "language_worse_than_cnp", "cnp_gap": _band(15.5, 4),
                                     "swap_gap": _band(15.5, 4), "language_gap": _band(42.9, 4)},
    "Zusammenhängende Blöcke reichen": {"verdict": "language_is_optimal", "cnp_gap": _band(33.3, 4), "language_gap": _band(0.0, 1),
                                        "bids_language": (234, 234)},
    "Summenziel ≠ Makespan": {"cnp_gap": _band(27.0, 4), "vcg_makespan_gap": _band(143.5, 4)},
    "VCG zahlt drauf": {"vcg_ratio": _band(1.22, 0.05), "clarke_min_margin": _band(-15.3, 2)},
    "Pay-as-bid: Bieter gewinnt": {"lie_gain": _band(17.0, 2), "lie_same_alloc": (True, True)},
    "Contract Net reicht schon": {"verdict": "cnp_already_optimal", "cnp_gap": _band(0.0, 1), "bids_language": (252, 252)},
}
