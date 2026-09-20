# Kombinatorische Auktionen an der Kran-Auftragsvergabe – Streamlit-Demo

Siebtes Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations
Research und Machine Learning", **Multi-Agenten-Koordinations-Linie** - ein **unabhängiger Zweig** vom Wurzelstück
[contract-net-demo](../contract-net-demo) (wie [dcop-demo](../dcop-demo) und [marl-demo](../marl-demo)). Das Vehikel (Kran-Aufträge
auf einer 1-D-Schiene, Makespan-Ziel) ist wortgleich aus den Geschwistern übernommen.

Idee: Contract Net vergibt Aufträge **einzeln** und kann später nicht korrigieren, was nur *gemeinsam* günstig gewesen wäre.
Hier bieten die Agenten auf **Bündel** von Aufträgen. Mit *allen* Bündeln ist die Auktion **trivial das exakte Optimum** (das ist
der Ankerpunkt, keine Überraschung). Gemessen werden die **Kosten**: die Zahl der Gebote, die Gewinnerermittlung, eingeschränkte
Gebotssprachen und die Anreize. Als Einsatzbeleg für Auktionen in der Multi-Roboter-Aufgabenverteilung: Gerkey & Matarić (2004),
*A formal analysis and taxonomy of task allocation in multi-robot systems*.

## Was dieses Stück tut

**Gebot.** Das Gebot eines Agenten auf ein Bündel *S* ist die Fertigstellungszeit seiner besten Route durch *S*. Auf der 1-D-Schiene
ist sie geschlossen bekannt: `bid(S) = Σ dauer + τ·((hi − lo) + min(|s − lo|, |s − hi|))` - erst zum näheren Ende der Positions-Spanne,
dann in einem Zug zum anderen (gegen eine Held-Karp-Teilmengen-DP geprüft, Abweichung < 1e-9). **Nur in 1-D:** in 2-D wäre schon das
Gebot ein TSP-Pfad und NP-hart; dass sich die Ergebnisse dorthin übertragen, ist vermutet, **nicht gemessen**.

**XOR-Gebote → Makespan = Maximum.** Jeder Agent gewinnt höchstens ein Bündel (das leere kostet 0). Dann ist der Makespan exakt das
Maximum der Zuschlagsgebote, und die **Gewinnerermittlung** ist eine Min-Max-Mengenpartition (NP-hart). Exakt per Teilmengen-DP,
`k·3^n` Schritte, numpy-vektorisiert: n = 12 in ~0.07 s, n = 14 in ~0.3 s, n = 16 in ~1.7 s. Die Reihenfolge *innerhalb* eines Bündels
ist der Sweep, nicht die aufsteigende Auftragsnummer - sonst wäre die Auktion mit allen Bündeln nicht das Optimum
(aufsteigend: +1.3 … +8.6 % darüber, einmalige Messung).

**Vier Gebotssprachen** (Gebote je Instanz mit n = 12, k = 3):

| Sprache | Gebote | Idee |
|---|---|---|
| Alle Bündel | 12 285 (`k(2ⁿ−1)`) | exaktes Optimum |
| Größenlimit *b* | 2 379 bei *b* = 4 | nur Bündel bis Größe *b*; bei XOR muss *b* ≥ ⌈n/k⌉ sein, sonst gibt es keine Zuteilung |
| 2 Blöcke | 2 379 (`k·Σ_{i≤4} C(n,i)`) | Vereinigung von höchstens zwei zusammenhängenden Abschnitten des nach Position sortierten Auftragsstroms |
| 1 Block | 234 (`k·n(n+1)/2`) | ein zusammenhängender Abschnitt; Gewinnerermittlung **polynomiell** (DP über Präfix × Agentenmenge, n = 100 in ~0.04 s) |
| *(Contract Net)* | 36 (`n·k`) | Einzelgebote |

## Was gemessen wurde

Lücke zum exakten Optimum, gepoolt über 480 feste Instanzen (16 Einstellungen × 30 Seeds; Prototyp-Messung, die App wiederholt sie
auf Knopfdruck für die jeweils gewählte Größe):

| Verfahren | mittlere Lücke |
|---|---|
| Contract Net | 25.2 % (Median 22.7, P90 50, Max 86) |
| Task-Swap | 11.9 % |
| Größenlimit *b* = ⌈n/k⌉ | 5.2 % (Median 0, aber Max 44 - **verliert im Einzelfall auch gegen Contract Net**) |
| Größenlimit *b* = ⌈n/k⌉ + 1 | 0.4 % |
| 1 Block | 5.6 % |
| **2 Blöcke** | **0.2 %** (exakt optimal in 83 %, verliert nie gegen Contract Net) |
| Summen- statt Makespan-Zuteilung | 45.9 % |

Kernaussage: **zwei zusammenhängende Blöcke** (2 379 statt 12 285 Gebote) reichen fast immer für das Optimum; ein zu kleines
Größenlimit spart ähnlich viele Gebote, verliert aber im Einzelfall deutlich (bis +44 %) - die *Struktur* der Sprache zählt, nicht nur ihre Größe.

### Die Mauer der Gewinnerermittlung

Die Gebotstabelle wächst mit 2ⁿ, der exakte Teilmengen-DP mit 3ⁿ; die Mauer liegt bei n ≈ 16. CP-SAT als Mengenpartitions-Löser auf der
Gebotstabelle scheitert schon bei n = 11-12. Die polynomielle Ein-Block-Auktion läuft dagegen bis n = 100 in Bruchteilen einer Sekunde
und liegt bei n ≥ 20 sogar **unter** dem 20-s-Incumbent von CP-SAT (−6.0 % bei n = 30, 0 von 8 Läufen bewiesen). Das ist die Größenordnung
einer kleinen Stichprobe gegen **unbewiesene** Referenzwerte - "exakt bis n ≈ 14, darüber beste bekannte Lösung" - und **kann ein Artefakt des
einfachen CP-SAT-Modells sein**.

### Anreize und Zahlungen (Beschaffungsauktion)

**Modellannahme, ausdrücklich nicht Teil des Fahrzeugs:** der Auktionator *kauft* die Bearbeitung; die Kosten eines Agenten sind seine
eigene Fertigstellungszeit, 1 Geldeinheit = 1 Minute. Dass ein Agent seine Gebote verfälschen könnte, ist Modellierung - im Fahrzeug sind die
Kosten öffentlich. Gemessen wird nur der **Ein-Bieter-Regret bei ehrlichen anderen** (Strategieraum: alle eigenen Gebote mit λ ∈ [0.5, 3]
multiplizieren) - eine **untere Schranke** des echten Regrets. Der Wettlauf, in dem alle gleichzeitig aufblähen (Pay-as-bid), wird nur als
Text erwähnt, nicht simuliert.

| Regel (Ziel × Zahlung) | Ergebnis (Prototyp, feste Instanzen) |
|---|---|
| A: Makespan + Pay-as-bid | Makespan optimal; **100 % der Gewinner** können mit einer Lüge profitieren |
| B: Summe + VCG | wahrheitsgetreu (0 Verstöße bei 6 000 Fehlmeldungen), Zahlung/Kosten 1.09 (max 1.22) - aber Makespan **+43.9 %**, schlechter als rohes Contract Net: Summenziel ≠ Makespan |
| C: Summe + Pay-as-bid | gleiche Zuteilung wie B (ebenso schlechter Makespan); nur wenige Gewinner können mit einer Lüge profitieren (App-Messung n = 10: 11 %) |
| E: Makespan + Clarke-Formel | Makespan optimal, aber **Zahlung < Kosten in 93.5 %** (Individuelle Rationalität verletzt) |

Bei *b* = ⌈n/k⌉ ist VCG in 30/30 Instanzen **nicht definiert** (jeder Agent ist unverzichtbar - ein Monopol); die App meldet das, statt eine
Zahl zu erfinden. Ergebnis: **keine Handregel** erfüllt gleichzeitig Makespan-Optimum, Kostendeckung und Ehrlichkeit.

### Ausblick: warum das eine gelernte Fortsetzung motiviert (nicht gezeigt)

Diese Lücke motiviert **RegretNet** (Dütting et al., ICML 2019: Zuteilung und Zahlung als neuronales Netz, Regret als Nebenbedingung) als
mögliche Fortsetzung. **Nicht** Teil dieses Stücks und **nicht gezeigt**, dass ein gelernter Mechanismus hier besser wäre. Zwei offene Lücken:
RegretNet ist ursprünglich *verkaufsseitig* (Erlösmaximierung), hier ist es Beschaffung mit Makespan-Ziel; und die Gebotstabelle hat
`k·2ⁿ` Zahlen (bei n = 12, k = 3: 12 285), ein gelernter Mechanismus müsste damit umgehen oder die Sprache einschränken.

### Was nicht funktioniert hat (einmalige Messung, kein App-Abschnitt)

- **Je Agent nur die *m* billigsten Bündel** anbieten: **+98 … +267 %** über dem Optimum.
- **Dichte-Greedy als Gewinnerermittlung**: **+136 … +220 %**.
- **Batch-Variante** (Aufträge in Runden zu je *m* angekündigt, je Runde eine Bündelauktion, danach fest): kein Modul, keine App - nur
  diese Tabelle (n = 12, k = 3, 100 Instanzen, Seeds 0-99, Ankündigung in Indexreihenfolge; Prototyp):

| Batchgröße *m* | Gebote | mittlere Lücke |
|---|---|---|
| 1 | 36 | 28.8 % |
| 3 | 84 | 23.2 % |
| 6 | 378 | 10.8 % |
| 9 | 1 554 | 13.6 % |
| 11 | 6 144 | 13.4 % |
| 12 | 12 285 | 0.0 % |

Die Lücke ist **nicht monoton** in *m* (m = 8: 15.0 %, m = 5: 21.0 %); erst der vollständige Batch (*m* = n) ist optimal. Ankündigung nach
Position statt Index ist schlechter (m = 6: 18.3 % statt 10.8 %).

## Verifikation

- Bündelgebote = Held-Karp-Routen (alle Bündel, n ≤ 6); Einzelauftrags-Gebote = Contract-Net-Gebote der ersten Runde.
- Gewinnerermittlung: Teilmengen-DP = Aufzählungs-Orakel (n ≤ 6, Max- und Summenziel) = CP-SAT-Mengenpartition (n = 7); Ein-Block-Polynom-DP =
  maskierter Teilmengen-DP (n ≤ 10).
- Auktion mit allen Bündeln = `cn_bruteforce` (1e-9) und CP-SAT im Rundungsband (CP-SAT rechnet auf einem 0,1-Minuten-Raster und liegt deshalb
  bis etwa 1 % über dem exakten Wert - die App weist darauf hin).
- Zählformeln der Gebotszahlen gegen gezählte Tabelleneinträge (n = 3 … 12), Größenlimit unter ⌈n/k⌉ infeasibel.
- Handbeispiele (3 Aufträge, 2 Agenten) für Gebotstabelle, Gewinnerermittlung, VCG, Clarke-Formel auf Makespan (Zahlung < Kosten) und die profitable
  Pay-as-bid-Lüge; VCG-Wahrheit auf dem λ-Gitter und bei Zufalls-Multiplikatoren; Monopol ⇒ `nicht definiert` statt Fantasiezahl.
- Alle 8 Presets liegen in ihren kalibrierten Bändern (`tests/test_presets.py`); AppTest-Rauchtests für Standard, jedes Preset, verborgenen
  b-Regler (Wert bleibt erhalten, wird geklemmt) und Randgrößen.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Vergleich, Gebote, Gewinnerermittlung, Sprach-Sweep, Anreize |
| `auction_bids.py` | geschlossene Bündelgebote, Sprachen, Zählformeln |
| `auction_wd.py` | Gewinnerermittlung (Teilmengen-DP, Ein-Block-Polynom-DP, CP-SAT-Mengenpartition, Orakel) |
| `auction_mechanism.py` | Regeln A/B/C/D/E, Zahlungen, Fehlmeldung, Regret |
| `auction_evaluation.py` | Vergleich, Verdict-Kaskade, Sprach-, Skalierungs-, Mechanismus-Sweeps |
| `auction_visualization.py` | Plotly-Grafiken |
| `cn_presets.py`, `cn_constants.py` | Regler, Permalink, Presets, Grenzen |
| `cn_*.py` (übrige) | Vehikel (wortgleich aus den Geschwister-Demos), `cn_negotiation.py` = Task-Swap |
| `tests/` | Gebote, Gewinnerermittlung, Mechanismen, Evaluation, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
