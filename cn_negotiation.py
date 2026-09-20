"""Task-Swap-Verhandlung: die Fortsetzung des Contract Net Protocol (siehe
`cn_protocol.py`). Nachdem alle Aufträge vergeben sind - der Manager kennt
jetzt (anders als während der Vergabe selbst) die VOLLSTÄNDIGE Zuteilung -
tauschen je zwei Agenten wechselseitig einen eigenen Auftrag, wenn das den
Makespan senkt. Das ändert NUR, wem ein Auftrag gehört, nie die Reihenfolge
innerhalb der Warteschlange eines Agenten (die bleibt aufsteigend nach
Auftrags-Index) - einseitige "Geschenke" (ein Auftrag wandert ohne Gegenzug)
sind bewusst NICHT Teil dieses Mechanismus.

Terminiert, sobald ein vollständiger Durchlauf über alle Agentenpaare und
Auftragspaare keinen verbessernden Tausch mehr findet (lokales Optimum -
die ehrliche Schwäche dieses Stücks: eine auf paarweise, wechselseitige
Tausche beschränkte lokale Suche muss nicht das globale Optimum erreichen,
das der zentrale CP-SAT-Referenzlöser findet).

Ein Agentenpaar wird nur betrachtet, wenn die ROHE physische Distanz ihrer
STARTPOSITIONEN die `communication_range` nicht überschreitet - bewusst NICHT
ihrer Endpositionen nach Phase 1: die Verhandlung findet statt, sobald der volle
Plan steht, aber BEVOR irgendein Agent auch nur einen Meter gefahren ist. Zu
diesem Zeitpunkt steht jeder Agent noch an seiner Startposition - "wer kann mit
wem reden" ist also eine Frage der Startpositionen, nicht einer hypothetischen
Position, die ein Agent erst nach Abarbeitung seiner gesamten (noch gar nicht
begonnenen) Warteschlange erreichen würde. Das ist der einzige Punkt in dieser
Demo, an dem "mehrere Agenten" tatsächlich etwas anderes bedeutet als "eine
zentrale lokale Suche": Phase 1 (Bieten) braucht ohnehin nie Agent-zu-Agent-
Kommunikation, nur diese Verhandlung tut es.
Bei `communication_range=0` findet sich kein einziges gültiges Paar - das
Ergebnis ist dann exakt das unveränderte Contract-Net-Ergebnis, nie ein Absturz
oder ein ungültiger Zustand (siehe `test_zero_communication_range_reduces_to_
raw_cnp_result`) - der Ausfall degradiert graduell, statt total zu versagen,
wie es ein zentraler Solver ohne vollständige Information tun würde."""

from dataclasses import dataclass

from cn_schedule import schedule_from_assignment

EPSILON = 1e-9


@dataclass(frozen=True)
class SwapStep:
    round: int
    agent_a: int
    agent_b: int
    job_from_a: int  # Auftrag, den Agent a abgibt (geht an Agent b)
    job_from_b: int  # Auftrag, den Agent b abgibt (geht an Agent a)
    schedules_after: dict
    agent_finish_times_after: tuple
    makespan_after: float


@dataclass(frozen=True)
class NegotiationResult:
    protocol_result: object  # cn_protocol.ProtocolResult, unverändert
    swaps: tuple  # Tupel von SwapStep, in Akzeptanzreihenfolge (kann leer sein)
    final_schedules: dict
    final_assignment: dict  # job_index -> agent_id, nach der Verhandlung
    final_agent_finish_times: tuple
    final_makespan: float
    reached_local_optimum: bool  # False nur, wenn max_rounds erreicht wurde (Sicherheitsnetz)


def _apply_swap(schedules, agent_a, agent_b, job_from_a, job_from_b):
    """Baut neue schedules: job_from_a wandert von agent_a zu agent_b und
    umgekehrt job_from_b von agent_b zu agent_a - je Agent aufsteigend nach
    Auftrags-Index sortiert (Reihenfolge bleibt sonst unverändert)."""
    remaining_a = [j for j in schedules[agent_a] if j != job_from_a]
    remaining_b = [j for j in schedules[agent_b] if j != job_from_b]
    new_schedules = dict(schedules)
    new_schedules[agent_a] = tuple(sorted(remaining_a + [job_from_b]))
    new_schedules[agent_b] = tuple(sorted(remaining_b + [job_from_a]))
    return new_schedules


def negotiate(instance, protocol_result, max_rounds=50, epsilon=EPSILON, communication_range=float("inf")):
    schedules = {a: tuple(jobs) for a, jobs in protocol_result.schedules.items()}
    agent_finish_times, makespan = schedule_from_assignment(instance, schedules)

    # Startpositionen, nicht Endpositionen: die Verhandlung passiert, bevor irgendein
    # Agent losgefahren ist (siehe Modul-Docstring).
    start_positions = instance.agent_start_positions

    swaps = []
    round_number = 0
    reached_local_optimum = False

    while round_number < max_rounds:
        round_number += 1
        best_delta = -epsilon
        best_candidate = None  # (agent_a, agent_b, job_from_a, job_from_b, new_schedules, new_finish_times, new_makespan)

        agent_ids = sorted(a for a in schedules if len(schedules[a]) > 0)
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent_a, agent_b = agent_ids[i], agent_ids[j]
                if abs(start_positions[agent_a] - start_positions[agent_b]) > communication_range:
                    continue
                for job_from_a in schedules[agent_a]:
                    for job_from_b in schedules[agent_b]:
                        candidate_schedules = _apply_swap(schedules, agent_a, agent_b, job_from_a, job_from_b)
                        candidate_finish_times, candidate_makespan = schedule_from_assignment(
                            instance, candidate_schedules
                        )
                        delta = candidate_makespan - makespan
                        if delta < best_delta:
                            best_delta = delta
                            best_candidate = (
                                agent_a, agent_b, job_from_a, job_from_b,
                                candidate_schedules, candidate_finish_times, candidate_makespan,
                            )

        if best_candidate is None:
            reached_local_optimum = True
            break

        agent_a, agent_b, job_from_a, job_from_b, schedules, agent_finish_times, makespan = best_candidate
        swaps.append(
            SwapStep(
                round=round_number,
                agent_a=agent_a,
                agent_b=agent_b,
                job_from_a=job_from_a,
                job_from_b=job_from_b,
                schedules_after=schedules,
                agent_finish_times_after=agent_finish_times,
                makespan_after=makespan,
            )
        )

    final_assignment = {}
    for agent_id, jobs in schedules.items():
        for job_index in jobs:
            final_assignment[job_index] = agent_id

    return NegotiationResult(
        protocol_result=protocol_result,
        swaps=tuple(swaps),
        final_schedules=schedules,
        final_assignment=final_assignment,
        final_agent_finish_times=agent_finish_times,
        final_makespan=makespan,
        reached_local_optimum=reached_local_optimum,
    )
