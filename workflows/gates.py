"""Versioned, pure validity gates (design section 6.2). Scientific tolerances live here, separate
from the deterministic replay tolerances of ``darts.tools.cicd_tools``."""

from __future__ import annotations

import numpy as np

GATES_VERSION = "0.1.0"


def normalized_residuals(
    predicted: np.ndarray, observed: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    ) / np.asarray(sigma, dtype=float)


def chi2(
    predicted_ensemble: np.ndarray, observed: np.ndarray, sigma: np.ndarray
) -> float:
    """Ensemble mean of the reduced chi-square with a diagonal covariance (degrees of freedom = nd)."""
    r = normalized_residuals(np.atleast_2d(predicted_ensemble), observed, sigma)
    return float(np.mean(np.mean(r**2, axis=1)))


def chi2_band(value: float, low: float = 0.5, high: float = 2.0) -> bool:
    return low <= value <= high


def held_out_rmse(
    predicted_ensemble: np.ndarray, observed: np.ndarray, sigma: np.ndarray
) -> float:
    """Normalized RMSE of the ensemble mean on held-out data."""
    mean = np.mean(np.atleast_2d(predicted_ensemble), axis=0)
    return float(np.sqrt(np.mean(normalized_residuals(mean, observed, sigma) ** 2)))


def coverage(
    predicted_ensemble: np.ndarray, observed: np.ndarray, level: float = 0.8
) -> float:
    """Fraction of observations inside the central ``level`` interval of the ensemble."""
    ens = np.atleast_2d(predicted_ensemble)
    lo = np.percentile(ens, 100 * (1 - level) / 2, axis=0)
    hi = np.percentile(ens, 100 * (1 + level) / 2, axis=0)
    obs = np.asarray(observed, dtype=float)
    return float(np.mean((obs >= lo) & (obs <= hi)))


def spread_ratio(posterior: np.ndarray, prior: np.ndarray) -> float:
    """Mean posterior standard deviation over mean prior standard deviation (collapse diagnostic)."""
    return float(
        np.mean(np.std(posterior, axis=0, ddof=1))
        / max(np.mean(np.std(prior, axis=0, ddof=1)), 1e-300)
    )


def gradient_angle(g1: np.ndarray, g2: np.ndarray) -> float:
    g1, g2 = np.asarray(g1, dtype=float), np.asarray(g2, dtype=float)
    c = g1 @ g2 / max(np.linalg.norm(g1) * np.linalg.norm(g2), 1e-300)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def regret(
    objective: float,
    reference: float,
    baseline: float,
    eps_abs: float = 1e-9,
    eps_rel: float = 1e-3,
    maximize: bool = True,
) -> float | None:
    """Normalized regret; ``None`` when the benchmark is uninformative (reference too close to baseline)."""
    sign = 1.0 if maximize else -1.0
    gap = sign * (reference - baseline)
    floor = max(eps_abs, eps_rel * abs(reference))
    if gap < floor:
        return None
    return float(sign * (reference - objective) / gap)


def budget_exceeded(simulations: int, budget: int) -> bool:
    return simulations > budget
