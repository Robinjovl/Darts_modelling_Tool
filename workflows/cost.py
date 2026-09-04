"""Cost model: simulation counts per method, worker sizing and pre-flight estimates (design section 6.2).

Formulas follow the design document: Saltelli ``N x (D + 2)`` without second-order indices and
``N x (2D + 2)`` with them (``N`` a power of two); central finite differences ``2 x dim`` per
gradient plus line-search evaluations as a range; ES-MDA ``(n_steps + 1) x ne`` including the
posterior prediction; robust evaluation ``candidates x ne``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from workflows.spec import StudySpec

LINE_SEARCH_RUNS_DEFAULT = (
    10,
    45,
)  # extra objective evaluations per L-BFGS-B iteration (section 11)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def saltelli_runs(n_base: int, n_dims: int, second_order: bool = False) -> int:
    if not is_power_of_two(n_base):
        raise ValueError(f"Sobol base sample size must be a power of two, got {n_base}")
    return n_base * ((2 * n_dims + 2) if second_order else (n_dims + 2))


def morris_runs(n_trajectories: int, n_dims: int) -> int:
    return n_trajectories * (n_dims + 1)


def fd_gradient_runs(n_controls: int, central: bool = True) -> int:
    return (2 if central else 1) * n_controls


def esmda_runs(ne: int, n_steps: int) -> int:
    return (n_steps + 1) * ne


def robust_runs(n_candidates: int, ne: int) -> int:
    return n_candidates * ne


def planned_simulation_range(spec: StudySpec) -> tuple:
    """(low, high) simulation count implied by ``spec.design`` for the documented methods.

    Only finite-difference control optimization has a genuine range (line-search evaluations per
    iteration, ``design["line_search_runs"]`` as ``[low, high]``); other methods are exact.
    """
    d = spec.design
    if spec.workflow == "optimize" and d.get("driver") == "fd_controls":
        low, high = d.get("line_search_runs", LINE_SEARCH_RUNS_DEFAULT)
        gradient = fd_gradient_runs(int(d["n_controls"]))
        iterations = int(d["max_iterations"])
        return iterations * (gradient + int(low)), iterations * (gradient + int(high))
    count = planned_simulations(spec)
    return count, count


def planned_simulations(spec: StudySpec) -> int:
    """Simulation count implied by ``spec.design``; the upper bound where a range applies."""
    d = spec.design
    if spec.workflow == "ensemble":
        method = d.get("method", "lhs")
        n_dims = int(d.get("n_dims", len(spec.parameters)))
        if method in ("lhs", "sobol_sequence"):
            return int(d["n"])
        if method == "morris":
            return morris_runs(int(d["n_trajectories"]), n_dims)
        if method == "saltelli":
            return saltelli_runs(
                int(d["n_base"]), n_dims, bool(d.get("second_order", False))
            )
        raise ValueError(f"unknown ensemble design method {method!r}")
    if spec.workflow == "hm-esmda":
        return esmda_runs(int(d["ne"]), int(d["n_steps"]))
    if spec.workflow == "hm-adjoint":
        # one forward plus one adjoint gradient (about one forward-equivalent) per iteration
        return 2 * int(d["max_iterations"]) * int(d.get("n_starts", 1))
    if spec.workflow == "optimize":
        driver = d.get("driver", "exhaustive")
        if driver in ("exhaustive", "robust"):
            n_candidates = d.get("n_candidates", d.get("max_candidates"))
            if n_candidates is None:
                raise ValueError(
                    "exhaustive estimate needs design.max_candidates or design.n_candidates "
                    "(the feasible set is only known after the adapter geometry is built)"
                )
            ne = int(d.get("ne", 1))
            # the driver always simulates the baseline configuration as well
            return robust_runs(int(n_candidates), ne) + ne
        if driver == "fd_controls":
            _, high = d.get("line_search_runs", LINE_SEARCH_RUNS_DEFAULT)
            per_iteration = fd_gradient_runs(int(d["n_controls"])) + int(high)
            return int(d["max_iterations"]) * per_iteration
        if driver in ("derivative_free", "surrogate"):
            return int(d["n_evaluations"]) * int(d.get("ne", 1))
        raise ValueError(f"unknown optimize driver {driver!r}")
    raise ValueError(f"unknown workflow {spec.workflow!r}")


def _cgroup_cpu_limit() -> float | None:
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as handle:
            quota, period = handle.read().split()
    except (OSError, ValueError):
        return None
    if quota == "max":
        return None
    return float(quota) / float(period)


def _available_memory_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1e6
    except OSError:
        return None
    return None


def usable_workers(
    threads_per_member: int = 1,
    memory_per_member_gb: float | None = None,
    max_workers: int | None = None,
    reserve_cores: int = 2,
) -> int:
    """Worker count bounded by CPU affinity, cgroup quota and memory; load average only throttles.

    Single implementation of the sizing rule; the executor (implementation step 4) imports it.
    """
    cores = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    quota = _cgroup_cpu_limit()
    if quota:
        cores = min(cores, int(quota))
    try:
        load = os.getloadavg()[0]
    except (OSError, AttributeError):
        load = 0.0
    budget = max(1, int(cores - reserve_cores - load)) // max(1, threads_per_member)
    if memory_per_member_gb:
        available = _available_memory_gb()
        if available is not None:
            budget = min(budget, max(1, int(available / memory_per_member_gb)))
    if max_workers is not None:
        budget = min(budget, max_workers)
    return max(1, budget)


@dataclass
class Estimate:
    simulations: int
    workers: int
    wall_s: tuple
    cpu_hours: tuple
    disk_gb: tuple
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "simulations": self.simulations,
            "workers": self.workers,
            "wall_s": list(self.wall_s),
            "cpu_hours": list(self.cpu_hours),
            "disk_gb": list(self.disk_gb),
            "notes": list(self.notes),
        }


def estimate(
    simulations: int,
    member_seconds: tuple,
    workers: int,
    startup_seconds: tuple = (0.5, 2.0),
    failure_rate: float = 0.05,
    extra_runs: int = 0,
    member_disk_mb: float = 3.0,
) -> Estimate:
    """Pre-flight estimate as ranges: per-member wall (init + run), expected retries, extra runs.

    ``extra_runs`` covers post-validation and reference re-runs a driver plans beyond the design's
    own count (design section 6.2).
    """
    if simulations < 0 or extra_runs < 0:
        raise ValueError("simulation counts must be non-negative")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not 0.0 <= failure_rate < 1.0:
        raise ValueError("failure_rate must be in [0, 1)")
    if member_seconds[0] < 0 or member_seconds[1] < member_seconds[0]:
        raise ValueError("member_seconds must be a non-negative (low, high) pair")
    total = simulations + extra_runs
    expected_attempts = total * (1.0 + failure_rate)
    lo = (member_seconds[0] + startup_seconds[0]) * expected_attempts
    hi = (member_seconds[1] + startup_seconds[1]) * expected_attempts
    cpu_hours = (lo / 3600.0, hi / 3600.0)
    wall = (lo / workers, hi / workers)
    disk = (
        total * member_disk_mb / 1024.0,
        expected_attempts * member_disk_mb / 1024.0,
    )
    notes = [
        f"{total} planned simulations, {failure_rate:.0%} expected retries, {workers} workers"
    ]
    return Estimate(
        simulations=total,
        workers=workers,
        wall_s=wall,
        cpu_hours=cpu_hours,
        disk_gb=disk,
        notes=notes,
    )
