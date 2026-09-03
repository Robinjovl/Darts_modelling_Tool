"""Parameter families and the unit-hypercube mapping to adapter realizations (design section 6.2).

Every family maps a slice of a point ``u`` in ``[0, 1]^D`` to physical values through its inverse
CDF, so that Latin-hypercube, Sobol, Morris and Saltelli designs all live in the same unit cube.
``Parameterization.to_realization(u)`` returns the dict an adapter's ``apply``/``build`` consume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from workflows.spec import ParameterSpec

DENSE_FIELD_LIMIT = 4000  # cells; larger fields need gstools (not yet wired)


@dataclass
class Family:
    """Base class: ``dim`` unit coordinates -> a contribution to the realization dict."""

    name: str
    target: str

    @property
    def dim(self) -> int:
        return 1

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def contribute(self, realization: dict, values: np.ndarray) -> None:
        realization[self.target] = float(values[0]) if values.size == 1 else values

    def labels(self) -> list:
        return (
            [self.name]
            if self.dim == 1
            else [f"{self.name}[{i}]" for i in range(self.dim)]
        )


@dataclass
class ScalarParam(Family):
    low: float = 0.0
    high: float = 1.0

    def from_unit(self, u):
        return self.low + (self.high - self.low) * np.asarray(u, dtype=float)


@dataclass
class LogScalarParam(Family):
    """Log-uniform between ``low`` and ``high`` (both positive)."""

    low: float = 0.1
    high: float = 10.0

    def __post_init__(self):
        if self.low <= 0 or self.high <= self.low:
            raise ValueError("LogScalarParam needs 0 < low < high")

    def from_unit(self, u):
        return np.exp(
            math.log(self.low)
            + (math.log(self.high) - math.log(self.low)) * np.asarray(u, dtype=float)
        )


@dataclass
class MultiplierField(Family):
    """Independent log-normal multipliers on ``n`` connections or regions (``sigma_log10``)."""

    n: int = 1
    sigma_log10: float = 0.3
    groups: list | None = None  # optional mapping connection -> group index (len n)

    @property
    def dim(self) -> int:
        return (max(self.groups) + 1) if self.groups is not None else self.n

    def from_unit(self, u):
        z = stats.norm.ppf(np.clip(np.asarray(u, dtype=float), 1e-12, 1 - 1e-12))
        group_values = 10.0 ** (self.sigma_log10 * z)
        if self.groups is not None:
            return group_values[np.asarray(self.groups, dtype=int)]
        return group_values


@dataclass
class LogPermField(Family):
    """Correlated log10-permeability on cell centroids, truncated Karhunen-Loeve expansion.

    Exponential covariance with correlation length ``range_m`` on Euclidean centroid distances;
    ``n_components`` leading modes are kept. Realization target is a per-cell permeability array
    (rebuild scope in the adapter). Positions are set with ``set_centroids`` before sampling.
    """

    mean_log10: float = math.log10(500.0)
    sigma_log10: float = 0.3
    range_m: float = 1000.0
    n_components: int = 10
    centroids: np.ndarray | None = field(default=None, repr=False)
    _basis: np.ndarray | None = field(default=None, repr=False)
    _scales: np.ndarray | None = field(default=None, repr=False)

    @property
    def dim(self) -> int:
        return self.n_components

    def set_centroids(self, centroids) -> None:
        xy = np.asarray(centroids, dtype=float)
        if xy.shape[0] > DENSE_FIELD_LIMIT:
            raise ValueError(
                f"{xy.shape[0]} cells exceed the dense KL limit {DENSE_FIELD_LIMIT}; gstools needed"
            )
        d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
        cov = (self.sigma_log10**2) * np.exp(-d / self.range_m)
        w, v = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1][: self.n_components]
        self.centroids = xy
        self._scales = np.sqrt(np.clip(w[order], 0.0, None))
        self._basis = v[:, order]

    @property
    def energy_fraction(self) -> float:
        if self._scales is None:
            raise RuntimeError("call set_centroids first")
        total = self.sigma_log10**2 * self.centroids.shape[0]
        return float((self._scales**2).sum() / total)

    def from_unit(self, u):
        if self._basis is None:
            raise RuntimeError("call set_centroids first")
        z = stats.norm.ppf(np.clip(np.asarray(u, dtype=float), 1e-12, 1 - 1e-12))
        log10k = self.mean_log10 + self._basis @ (self._scales * z)
        return 10.0**log10k

    def contribute(self, realization, values):
        realization[self.target] = np.asarray(values, dtype=float)


FAMILIES = {
    "ScalarParam": ScalarParam,
    "LogScalarParam": LogScalarParam,
    "MultiplierField": MultiplierField,
    "LogPermField": LogPermField,
}


@dataclass
class Parameterization:
    families: list

    @classmethod
    def from_specs(cls, specs: list, geometry: dict | None = None) -> Parameterization:
        families = []
        for spec in specs:
            spec = spec if isinstance(spec, ParameterSpec) else ParameterSpec(**spec)
            if spec.family not in FAMILIES:
                raise KeyError(
                    f"unknown parameter family {spec.family!r}; known: {sorted(FAMILIES)}"
                )
            args = dict(spec.args)
            target = args.pop("target", DEFAULT_TARGETS.get(spec.family, spec.name))
            family = FAMILIES[spec.family](name=spec.name, target=target, **args)
            if isinstance(family, LogPermField):
                if geometry is None:
                    raise ValueError(
                        "LogPermField needs adapter geometry (cell centroids)"
                    )
                family.set_centroids(geometry["centroids"])
            families.append(family)
        return cls(families)

    @property
    def dim(self) -> int:
        return sum(f.dim for f in self.families)

    def labels(self) -> list:
        return [label for f in self.families for label in f.labels()]

    def to_realization(self, u: np.ndarray) -> dict:
        u = np.asarray(u, dtype=float)
        if u.shape != (self.dim,):
            raise ValueError(
                f"expected a unit point of dimension {self.dim}, got shape {u.shape}"
            )
        realization, offset = {}, 0
        for family in self.families:
            values = np.atleast_1d(family.from_unit(u[offset : offset + family.dim]))
            family.contribute(realization, values)
            offset += family.dim
        return realization


DEFAULT_TARGETS = {
    "ScalarParam": "value",
    "LogScalarParam": "tran_multiplier",
    "MultiplierField": "tran_multiplier",
    "LogPermField": "permx",
}


def jsonable(realization: dict) -> dict:
    """Realization with numpy arrays converted to lists (for tasks and result files)."""
    return {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in realization.items()
    }
