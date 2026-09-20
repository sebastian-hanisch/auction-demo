"""Handbeispiele der Auktions-Demo (EX3 und EX1) und ein Held-Karp-Orakel für die Bündelgebote."""

from itertools import combinations

from cn_scenario import Instance, Job


def ex3():
    """3 Aufträge (Pos, Dauer): J0=(8, 15), J1=(5, 8), J2=(3, 12); Agenten bei 5 und 15; τ=1."""
    return Instance(
        n_jobs=3, n_agents=2, jobs=(Job(0, 8.0, 15.0), Job(1, 5.0, 8.0), Job(2, 3.0, 12.0)),
        agent_start_positions=(5.0, 15.0), travel_time_per_unit=1.0,
    )


def ex1():
    """3 Aufträge: (4, 10), (7, 10), (16, 5); Agenten bei 5 und 15; τ=1."""
    return Instance(
        n_jobs=3, n_agents=2, jobs=(Job(0, 4.0, 10.0), Job(1, 7.0, 10.0), Job(2, 16.0, 5.0)),
        agent_start_positions=(5.0, 15.0), travel_time_per_unit=1.0,
    )


def held_karp_bid(instance, agent, jobs):
    """Beste Route durch `jobs` per Teilmengen-DP über alle Reihenfolgen - unabhängig von der geschlossenen Formel."""
    jobs = list(jobs)
    if not jobs:
        return 0.0
    tau = instance.travel_time_per_unit
    start = instance.agent_start_positions[agent]
    size = len(jobs)
    best = {}
    for i, j in enumerate(jobs):
        best[(1 << i, i)] = abs(start - instance.jobs[j].position) * tau + instance.jobs[j].duration
    for count in range(2, size + 1):
        for subset in combinations(range(size), count):
            mask = sum(1 << i for i in subset)
            for last in subset:
                previous = mask ^ (1 << last)
                candidates = [
                    best[(previous, p)] + abs(instance.jobs[jobs[p]].position - instance.jobs[jobs[last]].position) * tau
                    for p in subset if p != last
                ]
                best[(mask, last)] = min(candidates) + instance.jobs[jobs[last]].duration
    full = (1 << size) - 1
    return min(best[(full, i)] for i in range(size))
