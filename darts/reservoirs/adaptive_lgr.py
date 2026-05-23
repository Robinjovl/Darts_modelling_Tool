from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from darts.reservoirs.struct_reservoir_with_lgr import LGRPatch, StructReservoirWithLGR


@dataclass(frozen=True)
class AdaptiveLGRConfig:
    """
    Configuration for report-step adaptive local grid refinement.

    :param refine: Refinement ratio per parent cell in I/J/K directions.
    :type refine: tuple[int, int, int]
    :param buffer_cells: Number of parent-cell layers added around marked cells.
    :type buffer_cells: int
    :param gradient_threshold: Normalized neighbor-gradient threshold used to mark cells.
    :type gradient_threshold: float
    :param indicator_variables: Primary variables used by the refinement indicator. If None, all variables except pressure are used.
    :type indicator_variables: tuple[str, ...] | None
    :param seed_parent_cells: Parent I/J/K cells that are always refined.
    :type seed_parent_cells: tuple[tuple[int, int, int], ...]
    :param preserve_existing: Keep already refined parent cells refined; this disables coarsening.
    :type preserve_existing: bool
    :param max_refined_parent_cells: Optional cap on gradient-marked cells that can grow the refined region before buffering and mandatory cells are added.
    :type max_refined_parent_cells: int | None
    :param patch_name_prefix: Prefix for generated LGR patch names.
    :type patch_name_prefix: str
    :param min_value_range: Minimum range used for normalized gradients.
    :type min_value_range: float
    """

    refine: tuple[int, int, int] = (3, 3, 1)
    buffer_cells: int = 1
    gradient_threshold: float = 0.15
    indicator_variables: tuple[str, ...] | None = None
    seed_parent_cells: tuple[tuple[int, int, int], ...] = ()
    preserve_existing: bool = True
    max_refined_parent_cells: int | None = None
    patch_name_prefix: str = "amr"
    min_value_range: float = 1e-12

    def __post_init__(self) -> None:
        refine = tuple(int(v) for v in self.refine)
        if len(refine) != 3 or any(v < 1 for v in refine):
            raise ValueError("refine must contain three positive entries.")
        object.__setattr__(self, "refine", refine)

        if self.buffer_cells < 0:
            raise ValueError("buffer_cells must be non-negative.")
        if self.gradient_threshold < 0.0:
            raise ValueError("gradient_threshold must be non-negative.")
        if self.max_refined_parent_cells is not None:
            if self.max_refined_parent_cells < 1:
                raise ValueError("max_refined_parent_cells must be positive.")
        if self.min_value_range <= 0.0:
            raise ValueError("min_value_range must be positive.")


@dataclass(frozen=True)
class AdaptiveLGRPlan:
    """
    Refinement plan produced from a current LGR reservoir state.

    :param lgrs: Generated LGR patches.
    :type lgrs: list[LGRPatch]
    :param selected_parent_cells: Parent-global cells selected before rectangular patch conversion.
    :type selected_parent_cells: set[int]
    :param changed: Whether the generated LGR layout differs from the current one.
    :type changed: bool
    """

    lgrs: list[LGRPatch]
    selected_parent_cells: set[int]
    changed: bool


def initial_lgrs_from_config(
    grid_shape: tuple[int, int, int],
    config: AdaptiveLGRConfig,
) -> list[LGRPatch]:
    """
    Build initial LGR patches from mandatory seed cells.

    :param grid_shape: Parent-grid shape as nx, ny, nz.
    :type grid_shape: tuple[int, int, int]
    :param config: AMR configuration.
    :type config: AdaptiveLGRConfig
    :return: Initial LGR patches.
    :rtype: list[LGRPatch]
    """
    selected = {
        _parent_global_from_ijk(cell, grid_shape) for cell in config.seed_parent_cells
    }
    selected = _expand_parent_cells(selected, grid_shape, config.buffer_cells)
    return lgrs_from_parent_cells(selected, grid_shape, config)


def plan_adaptive_lgr(
    reservoir: StructReservoirWithLGR,
    state: np.ndarray,
    variable_names: list[str],
    config: AdaptiveLGRConfig,
) -> AdaptiveLGRPlan:
    """
    Plan a refine-only LGR layout from the current reservoir state.

    :param reservoir: Current structured LGR reservoir.
    :type reservoir: StructReservoirWithLGR
    :param state: Reservoir state with shape (n_res_blocks, n_vars).
    :type state: np.ndarray
    :param variable_names: State variable names corresponding to state columns.
    :type variable_names: list[str]
    :param config: AMR configuration.
    :type config: AdaptiveLGRConfig
    :return: Adaptive LGR plan.
    :rtype: AdaptiveLGRPlan
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (len(reservoir.cells), len(variable_names)):
        raise ValueError(
            "state shape must be "
            f"({len(reservoir.cells)}, {len(variable_names)}), got {state.shape}."
        )

    grid_shape = (reservoir.nx, reservoir.ny, reservoir.nz)
    parent_state, active_parent_cells = _aggregate_parent_state(reservoir, state)
    selected, scores = _select_gradient_cells(
        parent_state,
        active_parent_cells,
        grid_shape,
        variable_names,
        config,
    )

    seed_cells = {
        _parent_global_from_ijk(cell, grid_shape) for cell in config.seed_parent_cells
    }
    seed_buffer = _expand_parent_cells(seed_cells, grid_shape, config.buffer_cells)

    preserved_parent_cells = set(seed_buffer)
    if config.preserve_existing:
        preserved_parent_cells.update(reservoir.refined_parent_cells.keys())
    preserved_parent_cells.intersection_update(active_parent_cells)

    selected = _cap_growth_parent_cells(
        selected,
        scores,
        active_parent_cells,
        preserved_parent_cells,
        grid_shape,
        config,
    )
    selected = _expand_parent_cells(selected, grid_shape, config.buffer_cells)
    selected.update(seed_buffer)
    if config.preserve_existing:
        selected.update(reservoir.refined_parent_cells.keys())
    selected.intersection_update(active_parent_cells)

    lgrs = lgrs_from_parent_cells(selected, grid_shape, config)
    changed = lgr_layout_key(lgrs) != lgr_layout_key(reservoir.lgrs)
    return AdaptiveLGRPlan(lgrs=lgrs, selected_parent_cells=selected, changed=changed)


def lgrs_from_parent_cells(
    parent_cells: Iterable[int],
    grid_shape: tuple[int, int, int],
    config: AdaptiveLGRConfig,
) -> list[LGRPatch]:
    """
    Convert selected parent cells into rectangular LGR patches.

    :param parent_cells: Parent-global cells to refine.
    :type parent_cells: Iterable[int]
    :param grid_shape: Parent-grid shape as nx, ny, nz.
    :type grid_shape: tuple[int, int, int]
    :param config: AMR configuration.
    :type config: AdaptiveLGRConfig
    :return: Non-overlapping rectangular LGR patches.
    :rtype: list[LGRPatch]
    """
    selected = set(parent_cells)
    patches = []
    for patch_id, ijk_run in enumerate(_selected_cell_i_runs(selected, grid_shape)):
        i_min, i_max, j, k = ijk_run
        patches.append(
            LGRPatch(
                f"{config.patch_name_prefix}_{patch_id}",
                (i_min + 1, i_max + 1),
                (j + 1, j + 1),
                (k + 1, k + 1),
                config.refine,
            )
        )
    return patches


def lgr_layout_key(lgrs: Iterable[LGRPatch]) -> tuple:
    """
    Return a hashable representation of an LGR layout.

    :param lgrs: LGR patches.
    :type lgrs: Iterable[LGRPatch]
    :return: Layout key.
    :rtype: tuple
    """
    return tuple(
        sorted(
            (
                patch.i_range,
                patch.j_range,
                patch.k_range,
                patch.refine,
            )
            for patch in lgrs
        )
    )


def project_reservoir_state(
    source_reservoir: StructReservoirWithLGR,
    source_state: np.ndarray,
    target_reservoir: StructReservoirWithLGR,
) -> np.ndarray:
    """
    Project primary variables from one LGR reservoir grid to another.

    The first AMR implementation is refine-only, so projecting by source cell
    containment is sufficient and avoids introducing artificial gradients inside
    newly refined parent cells.

    :param source_reservoir: Reservoir holding the current state.
    :type source_reservoir: StructReservoirWithLGR
    :param source_state: Current state with shape (source_n_res_blocks, n_vars).
    :type source_state: np.ndarray
    :param target_reservoir: New reservoir grid.
    :type target_reservoir: StructReservoirWithLGR
    :return: Projected state with shape (target_n_res_blocks, n_vars).
    :rtype: np.ndarray
    """
    source_state = np.asarray(source_state, dtype=float)
    if source_state.shape[0] != len(source_reservoir.cells):
        raise ValueError(
            f"source_state has {source_state.shape[0]} cells, "
            f"expected {len(source_reservoir.cells)}."
        )

    source_by_parent: dict[int, list] = {}
    for cell in source_reservoir.cells:
        source_by_parent.setdefault(cell.parent_global, []).append(cell)

    source_centers = np.asarray(
        [cell.center for cell in source_reservoir.cells], dtype=float
    )
    projected = np.empty(
        (len(target_reservoir.cells), source_state.shape[1]), dtype=float
    )
    for target_cell in target_reservoir.cells:
        source_idx = _source_cell_for_target(
            target_cell,
            source_by_parent.get(target_cell.parent_global, []),
            source_centers,
        )
        projected[target_cell.idx] = source_state[source_idx]
    return projected


def reservoir_state_from_engine(model) -> np.ndarray:
    """
    Return the current reservoir-state matrix from a DARTS model engine.

    :param model: Initialized DARTS model.
    :type model: darts.models.darts_model.DartsModel
    :return: Reservoir state with shape (n_res_blocks, n_vars).
    :rtype: np.ndarray
    """
    nb = model.reservoir.mesh.n_res_blocks
    nv = model.physics.n_vars
    return np.array(model.physics.engine.X[: nb * nv], copy=True).reshape((nb, nv))


def count_lgr_parent_cells(lgrs: Iterable[LGRPatch]) -> int:
    """
    Count parent cells covered by LGR patches.

    :param lgrs: LGR patches.
    :type lgrs: Iterable[LGRPatch]
    :return: Number of parent-grid cells covered by the patches.
    :rtype: int
    """
    return sum(
        (patch.i_range[1] - patch.i_range[0] + 1)
        * (patch.j_range[1] - patch.j_range[0] + 1)
        * (patch.k_range[1] - patch.k_range[0] + 1)
        for patch in lgrs
    )


def _aggregate_parent_state(
    reservoir: StructReservoirWithLGR,
    state: np.ndarray,
) -> tuple[dict[int, np.ndarray], set[int]]:
    totals: dict[int, np.ndarray] = {}
    weights: dict[int, float] = {}
    for cell in reservoir.cells:
        weight = max(cell.volume * cell.poro, 0.0)
        if weight == 0.0:
            weight = max(cell.volume, 1.0)
        totals[cell.parent_global] = (
            totals.get(cell.parent_global, np.zeros(state.shape[1], dtype=float))
            + weight * state[cell.idx]
        )
        weights[cell.parent_global] = weights.get(cell.parent_global, 0.0) + weight

    parent_state = {
        parent_global: total / weights[parent_global]
        for parent_global, total in totals.items()
    }
    return parent_state, set(parent_state.keys())


def _select_gradient_cells(
    parent_state: dict[int, np.ndarray],
    active_parent_cells: set[int],
    grid_shape: tuple[int, int, int],
    variable_names: list[str],
    config: AdaptiveLGRConfig,
) -> tuple[set[int], dict[int, float]]:
    indicator_names = (
        tuple(variable_names[1:])
        if config.indicator_variables is None
        else config.indicator_variables
    )
    indicator_indexes = [
        variable_names.index(name) for name in indicator_names if name in variable_names
    ]
    if not indicator_indexes:
        return set(), {}

    values = np.asarray([parent_state[cell] for cell in sorted(active_parent_cells)])
    scales = {
        idx: max(float(np.ptp(values[:, idx])), config.min_value_range)
        for idx in indicator_indexes
    }

    selected: set[int] = set()
    scores: dict[int, float] = {}
    for cell in active_parent_cells:
        for neighbor in _neighbor_parent_cells(cell, grid_shape):
            if neighbor not in active_parent_cells or neighbor < cell:
                continue
            score = max(
                abs(parent_state[cell][idx] - parent_state[neighbor][idx]) / scales[idx]
                for idx in indicator_indexes
            )
            if score >= config.gradient_threshold:
                selected.add(cell)
                selected.add(neighbor)
                scores[cell] = max(scores.get(cell, 0.0), score)
                scores[neighbor] = max(scores.get(neighbor, 0.0), score)
    return selected, scores


def _cap_growth_parent_cells(
    selected: set[int],
    scores: dict[int, float],
    active_parent_cells: set[int],
    preserved_parent_cells: set[int],
    grid_shape: tuple[int, int, int],
    config: AdaptiveLGRConfig,
) -> set[int]:
    if config.max_refined_parent_cells is None or len(selected) == 0:
        return selected

    growth_candidates = [
        cell
        for cell in selected
        if _candidate_adds_parent_cells(
            cell,
            active_parent_cells,
            preserved_parent_cells,
            grid_shape,
            config.buffer_cells,
        )
    ]
    sorted_cells = sorted(
        growth_candidates,
        key=lambda cell: scores.get(cell, 0.0),
        reverse=True,
    )
    return set(sorted_cells[: config.max_refined_parent_cells])


def _candidate_adds_parent_cells(
    parent_global: int,
    active_parent_cells: set[int],
    preserved_parent_cells: set[int],
    grid_shape: tuple[int, int, int],
    buffer_cells: int,
) -> bool:
    candidate_cells = _expand_parent_cells({parent_global}, grid_shape, buffer_cells)
    candidate_cells.intersection_update(active_parent_cells)
    return not candidate_cells.issubset(preserved_parent_cells)


def _selected_cell_i_runs(
    selected: set[int],
    grid_shape: tuple[int, int, int],
) -> list[tuple[int, int, int, int]]:
    grouped: dict[tuple[int, int], list[int]] = {}
    for parent_global in selected:
        i, j, k = _ijk0_from_parent_global(parent_global, grid_shape)
        grouped.setdefault((k, j), []).append(i)

    runs = []
    for (k, j), i_values in sorted(grouped.items()):
        start = None
        previous = None
        for i in sorted(i_values):
            if start is None:
                start = i
            elif previous is not None and i != previous + 1:
                runs.append((start, previous, j, k))
                start = i
            previous = i
        if start is not None and previous is not None:
            runs.append((start, previous, j, k))
    return runs


def _expand_parent_cells(
    selected: set[int],
    grid_shape: tuple[int, int, int],
    layers: int,
) -> set[int]:
    expanded = set(selected)
    frontier = set(selected)
    for _ in range(layers):
        next_frontier = set()
        for cell in frontier:
            next_frontier.update(_neighbor_parent_cells(cell, grid_shape))
        next_frontier.difference_update(expanded)
        expanded.update(next_frontier)
        frontier = next_frontier
    return expanded


def _neighbor_parent_cells(
    parent_global: int,
    grid_shape: tuple[int, int, int],
) -> list[int]:
    i, j, k = _ijk0_from_parent_global(parent_global, grid_shape)
    nx, ny, nz = grid_shape
    neighbors = []
    for di, dj, dk in (
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ):
        ii, jj, kk = i + di, j + dj, k + dk
        if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
            neighbors.append(_parent_global_from_ijk0((ii, jj, kk), grid_shape))
    return neighbors


def _source_cell_for_target(target_cell, candidates, source_centers: np.ndarray) -> int:
    center = target_cell.center
    for cell in candidates:
        if _cell_contains(cell, center):
            return cell.idx

    if candidates:
        centers = np.asarray([cell.center for cell in candidates], dtype=float)
        return candidates[int(np.argmin(np.linalg.norm(centers - center, axis=1)))].idx

    return int(np.argmin(np.linalg.norm(source_centers - center, axis=1)))


def _cell_contains(cell, point: tuple[float, float, float]) -> bool:
    tol = 1e-10
    return (
        cell.x_min - tol <= point[0] <= cell.x_max + tol
        and cell.y_min - tol <= point[1] <= cell.y_max + tol
        and cell.z_min - tol <= point[2] <= cell.z_max + tol
    )


def _parent_global_from_ijk(
    ijk: tuple[int, int, int],
    grid_shape: tuple[int, int, int],
) -> int:
    i, j, k = ijk
    nx, ny, nz = grid_shape
    if i < 1 or j < 1 or k < 1 or i > nx or j > ny or k > nz:
        raise IndexError(f"Parent cell {ijk} is outside parent grid {grid_shape}.")
    return _parent_global_from_ijk0((i - 1, j - 1, k - 1), grid_shape)


def _parent_global_from_ijk0(
    ijk0: tuple[int, int, int],
    grid_shape: tuple[int, int, int],
) -> int:
    i, j, k = ijk0
    nx, ny, _ = grid_shape
    return i + nx * (j + ny * k)


def _ijk0_from_parent_global(
    parent_global: int,
    grid_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    nx, ny, _ = grid_shape
    k, rem = divmod(int(parent_global), nx * ny)
    j, i = divmod(rem, nx)
    return i, j, k
