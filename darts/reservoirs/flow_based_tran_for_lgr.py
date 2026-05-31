"""
Flow-based transmissibility correction for LGR coarse-fine interfaces.

This module is used by :class:`StructReservoirWithLGR` when
``lgr_coarse_fine_tran_mode="flow_based"``. In the normal mode, each
coarse-fine connection gets a transmissibility from the overlap between the
coarse face and the fine faces. In this mode, the code solves a small local
flow problem around each side of the LGR patch, then uses that result to
correct those coarse-fine transmissibilities.

It does not require a dynamic reservoir simulation. The calculation
uses only the grid geometry and rock properties.

For each ``(patch, axis, side)``, the code does the following.

1. Make a temporary fine grid around the selected LGR side.

   The local region contains the LGR patch and a few neighboring parent cells,
   controlled by ``padding``. Parent cells in this local region are split using
   the same refinement ratio as the LGR. Cells inside the patch are the real
   LGR cells. Cells outside the patch are temporary virtual fine cells copied
   from the parent coarse cells. These virtual cells are used only for this
   local calculation and are not added to the simulation mesh.

2. Choose the planes used by the local problem.

   The sketch below shows the setup only in the normal direction of the LGR
   side. Padding in the two tangential directions is not shown.

   ::

       side = +1

       LGR patch                         outside parent cells, split virtually
       +-------------+-------------------+--------------------+---------+
       | LGR inside  | LGR face cells    | first outside slab | far_ids |
       |             | interface_ids     | support_ids        | fixed   |
       |             | fixed p = 1       | solved p           | p = 0   |
       +-------------+-------------------+--------------------+---------+
                         |<-- q_cross is summed across these links -->|

       side = -1

       outside parent cells, split virtually                         LGR patch
       +---------+--------------------+-------------------+-------------+
       | far_ids | first outside slab | LGR face cells    | LGR inside  |
       | fixed   | support_ids        | interface_ids     |             |
       | p = 0   | solved p           | fixed p = 1       |             |
       +---------+--------------------+-------------------+-------------+
                   |<-- q_cross is summed across these links -->|

   ``interface_ids`` are the real LGR cells on the selected patch face. They
   are fixed to ``p = 1``.

   ``far_ids`` are cells on the far outer side of the local region. They are
   fixed to ``p = 0``.

   ``support_ids`` are virtual fine cells in the first coarse-cell slab outside
   the LGR. Their average pressure is used as the coarse-side pressure in the
   final transmissibility formula.

3. Connect neighboring local cells.

   The local grid uses the same TPFA connection formula as the global LGR
   grid. For hydraulic flow, each connection uses face area, normal
   permeability, half-cell distances, and the DARTS Darcy conversion constant.

   Thermal flow is solved in the same way, but with physical rock conductive
   connections. A geometric thermal coefficient ``tranD`` is multiplied by the
   average rock conductive factor of the two cells:

       ``0.5 * ((1 - phi_a) * rcond_a + (1 - phi_b) * rcond_b)``

4. Solve the local steady linear problem.

   For each unknown local cell ``i``, the equation is:

       ``sum_j T_ij * (p_i - p_j) = 0``

   This is a steady unit-pressure-drop problem. Since it is linear, the chosen
   values ``p = 1`` and ``p = 0`` only set the scale.

   After solving, the code sums the flux through the LGR face:

       ``q_cross = sum_crossing T_ij * (p_lgr - p_virtual)``

   The total transmissibility for this LGR side is:

       ``T_eff_total = q_cross / (1 - mean(p_support))``

   The denominator uses the first outside slab, not the far boundary, because
   the corrected global connection should represent the drop between the LGR
   face and the neighboring coarse-side region.

5. Put the total back on the real global connections.

   The local solve gives one total transmissibility for one LGR side. The
   global mesh may have several actual coarse-fine links on that side:

   ::

       one coarse cell face can overlap several LGR fine faces

             coarse cell C
            +---------------+
            |               |
            +---------------+
            +---+---+---+---+
            |f1 |f2 |f3 |f4 |   LGR fine cells
            +---+---+---+---+

       local solve
            |
            v
       one T_eff_total for (patch, axis, side)
            |
            v
       actual coarse-fine links on that side

          link 1: raw T_1  ->  T_1 * T_eff_total / sum(raw T)
          link 2: raw T_2  ->  T_2 * T_eff_total / sum(raw T)
          ...
          link n: raw T_n  ->  T_n * T_eff_total / sum(raw T)

   Hydraulic transmissibilities are distributed in proportion to the raw
   face-overlap values. If all raw values are zero, the total is split evenly.

   For thermal flow, the local solve returns a physical conductive total, but
   the engine stores ``tran_thermal`` as the geometric ``tranD`` coefficient.
   The code therefore scales the raw ``tran_thermal`` values so that, after
   applying each link's rock conductive factor, the physical total matches the
   local solve result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

ConnectionTransmissibility = Callable[[Any, Any, int, float], tuple[float, float]]


@dataclass(frozen=True)
class CoarseFineConnection:
    """
    One real coarse-fine connection that can be rescaled.
    """

    index: int
    patch_name: str
    axis: int
    side: int
    lgr_cell_idx: int
    coarse_cell_idx: int
    raw_tran: float
    raw_tran_thermal: float


@dataclass(frozen=True)
class UpscalingResult:
    """
    The total values from one local solve.
    """

    hydraulic_total: float
    thermal_physical_total: float
    n_interface_links: int


@dataclass(frozen=True)
class _VirtualCell:
    idx: int
    name: str
    dx: float
    dy: float
    dz: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    poro: float
    permx: float
    permy: float
    permz: float
    rcond: float
    hcap: float
    op_num: int
    volume: float
    parent_global: int | None = None


class LGRCoarseFineFlowBasedUpscaler:
    """
    Run local solves to correct LGR coarse-fine transmissibilities.
    """

    def __init__(
        self,
        reservoir: Any,
        parent_arrays: dict[str, np.ndarray],
        connection_transmissibility: ConnectionTransmissibility,
        padding: int,
    ):
        self.reservoir = reservoir
        self.parent_arrays = parent_arrays
        self.connection_transmissibility = connection_transmissibility
        self.padding = int(padding)

    def effective_transmissibility(
        self, patch: Any, axis: int, side: int
    ) -> UpscalingResult:
        """
        Return hydraulic and thermal totals for one side of one LGR patch.

        ``axis`` says which direction is normal to the side: 0 is x/i, 1 is
        y/j, and 2 is z/k. ``side`` is ``-1`` for the lower side and ``+1`` for
        the upper side.

        This builds the temporary local fine grid, solves the local hydraulic
        and thermal problems, and returns one total for each. It does not
        change the reservoir or the global connection arrays.
        """

        cells, grid, is_lgr, interface_ids, support_ids, far_ids = (
            self._build_local_grid(patch, axis, side)
        )
        hydraulic_connections, thermal_connections, crossing = (
            self._build_local_connections(cells, grid, is_lgr, interface_ids, axis)
        )
        hydraulic_total = self._solve_conductance(
            len(cells),
            hydraulic_connections,
            crossing,
            interface_ids,
            support_ids,
            far_ids,
        )
        thermal_physical_total = self._solve_conductance(
            len(cells),
            thermal_connections,
            crossing,
            interface_ids,
            support_ids,
            far_ids,
        )
        return UpscalingResult(
            hydraulic_total=hydraulic_total,
            thermal_physical_total=thermal_physical_total,
            n_interface_links=len(crossing),
        )

    def _build_local_grid(
        self, patch: Any, axis: int, side: int
    ) -> tuple[list[Any], np.ndarray, list[bool], set[int], set[int], set[int]]:
        parent_lo, parent_hi = self._local_parent_bounds(patch, axis, side)
        patch_lo = np.asarray(
            [patch.i_range[0] - 1, patch.j_range[0] - 1, patch.k_range[0] - 1],
            dtype=int,
        )
        patch_hi = np.asarray(
            [patch.i_range[1] - 1, patch.j_range[1] - 1, patch.k_range[1] - 1],
            dtype=int,
        )
        refine = np.asarray(patch.refine, dtype=int)
        parent_counts = parent_hi - parent_lo + 1
        grid_shape = tuple((parent_counts * refine).tolist())
        grid = np.full(grid_shape, -1, dtype=np.int32)
        cells: list[Any] = []
        is_lgr: list[bool] = []

        for parent_k in range(parent_lo[2], parent_hi[2] + 1):
            for parent_j in range(parent_lo[1], parent_hi[1] + 1):
                for parent_i in range(parent_lo[0], parent_hi[0] + 1):
                    parent_ijk = np.asarray([parent_i, parent_j, parent_k], dtype=int)
                    inside_patch = bool(
                        np.all(parent_ijk >= patch_lo)
                        and np.all(parent_ijk <= patch_hi)
                    )
                    self._append_virtual_parent_cells(
                        patch,
                        parent_ijk,
                        parent_lo,
                        patch_lo,
                        refine,
                        inside_patch,
                        grid,
                        cells,
                        is_lgr,
                    )

        far_coord = 0 if side < 0 else grid.shape[axis] - 1
        interface_ids = _ids_on_lgr_interface(
            grid, axis, side, parent_lo, patch_lo, patch_hi, refine
        )
        support_ids = _ids_in_adjacent_parent_support(
            grid, axis, side, parent_lo, patch_lo, patch_hi, refine
        )
        far_ids = _ids_on_fine_plane(grid, axis, far_coord)
        if (
            not interface_ids
            or not support_ids
            or not far_ids
            or interface_ids & far_ids
        ):
            raise ValueError(
                f"Cannot build flow-upscaling domain for patch {patch.name!r}, "
                f"axis={axis}, side={side}."
            )
        return cells, grid, is_lgr, interface_ids, support_ids, far_ids

    def _local_parent_bounds(
        self, patch: Any, axis: int, side: int
    ) -> tuple[np.ndarray, np.ndarray]:
        patch_lo = np.asarray(
            [patch.i_range[0] - 1, patch.j_range[0] - 1, patch.k_range[0] - 1],
            dtype=int,
        )
        patch_hi = np.asarray(
            [patch.i_range[1] - 1, patch.j_range[1] - 1, patch.k_range[1] - 1],
            dtype=int,
        )
        parent_max = np.asarray(
            [self.reservoir.nx - 1, self.reservoir.ny - 1, self.reservoir.nz - 1],
            dtype=int,
        )

        parent_lo = np.maximum(0, patch_lo - self.padding)
        parent_hi = np.minimum(parent_max, patch_hi + self.padding)
        if side < 0:
            parent_hi[axis] = patch_hi[axis]
        else:
            parent_lo[axis] = patch_lo[axis]

        has_support = (
            parent_lo[axis] < patch_lo[axis]
            if side < 0
            else parent_hi[axis] > patch_hi[axis]
        )
        if not has_support:
            raise ValueError(
                f"Patch {patch.name!r} has no parent-cell support for "
                f"flow upscaling on axis={axis}, side={side}."
            )
        return parent_lo, parent_hi

    def _append_virtual_parent_cells(
        self,
        patch: Any,
        parent_ijk: np.ndarray,
        parent_lo: np.ndarray,
        patch_lo: np.ndarray,
        refine: np.ndarray,
        inside_patch: bool,
        grid: np.ndarray,
        cells: list[Any],
        is_lgr: list[bool],
    ) -> None:
        parent_global = _ijk0_to_global(
            int(parent_ijk[0]),
            int(parent_ijk[1]),
            int(parent_ijk[2]),
            self.reservoir.nx,
            self.reservoir.ny,
        )
        parent_center = self.parent_arrays["centers"][parent_global]
        parent_dx = float(self.parent_arrays["dx"][parent_global])
        parent_dy = float(self.parent_arrays["dy"][parent_global])
        parent_dz = float(self.parent_arrays["dz"][parent_global])
        x0 = float(parent_center[0] - 0.5 * parent_dx)
        y0 = float(parent_center[1] - 0.5 * parent_dy)
        z0 = float(parent_center[2] - 0.5 * parent_dz)
        dx = parent_dx / refine[0]
        dy = parent_dy / refine[1]
        dz = parent_dz / refine[2]
        volume = _scaled_fine_volume(
            dx,
            dy,
            dz,
            parent_dx,
            parent_dy,
            parent_dz,
            float(self.parent_arrays["volume"][parent_global]),
        )

        for fk in range(refine[2]):
            for fj in range(refine[1]):
                for fi in range(refine[0]):
                    local_fine_ijk = (parent_ijk - parent_lo) * refine + np.asarray(
                        [fi, fj, fk], dtype=int
                    )
                    if inside_patch:
                        patch_fine_ijk = (parent_ijk - patch_lo) * refine + np.asarray(
                            [fi, fj, fk], dtype=int
                        )
                        cell_idx = int(
                            self.reservoir.lgr_cell_maps[patch.name][
                                tuple(patch_fine_ijk)
                            ]
                        )
                        cell = self.reservoir.cells[cell_idx]
                    else:
                        cell = _VirtualCell(
                            idx=-1,
                            name=(
                                f"virtual:{parent_ijk[0] + 1},"
                                f"{parent_ijk[1] + 1},{parent_ijk[2] + 1}:"
                                f"{fi + 1},{fj + 1},{fk + 1}"
                            ),
                            dx=dx,
                            dy=dy,
                            dz=dz,
                            x_min=x0 + fi * dx,
                            x_max=x0 + (fi + 1) * dx,
                            y_min=y0 + fj * dy,
                            y_max=y0 + (fj + 1) * dy,
                            z_min=z0 + fk * dz,
                            z_max=z0 + (fk + 1) * dz,
                            poro=float(self.parent_arrays["poro"][parent_global]),
                            permx=float(self.parent_arrays["permx"][parent_global]),
                            permy=float(self.parent_arrays["permy"][parent_global]),
                            permz=float(self.parent_arrays["permz"][parent_global]),
                            rcond=float(self.parent_arrays["rcond"][parent_global]),
                            hcap=float(self.parent_arrays["hcap"][parent_global]),
                            op_num=int(self.parent_arrays["op_num"][parent_global]),
                            volume=volume,
                            parent_global=parent_global,
                        )

                    grid[tuple(local_fine_ijk)] = len(cells)
                    cells.append(cell)
                    is_lgr.append(inside_patch)

    def _build_local_connections(
        self,
        cells: list[Any],
        grid: np.ndarray,
        is_lgr: list[bool],
        interface_ids: set[int],
        upscaling_axis: int,
    ) -> tuple[
        list[tuple[int, int, float]],
        list[tuple[int, int, float]],
        list[tuple[int, int]],
    ]:
        hydraulic_connections = []
        thermal_connections = []
        crossing = []

        for axis in range(3):
            for index in np.ndindex(grid.shape):
                if index[axis] + 1 >= grid.shape[axis]:
                    continue
                neighbor = list(index)
                neighbor[axis] += 1
                cell_idx = int(grid[index])
                neighbor_idx = int(grid[tuple(neighbor)])
                if cell_idx < 0 or neighbor_idx < 0:
                    continue

                cell = cells[cell_idx]
                neighbor_cell = cells[neighbor_idx]
                area = _cell_face_overlap_area(cell, neighbor_cell, axis)
                if area <= 0.0:
                    continue
                hydraulic, tran_thermal = self.connection_transmissibility(
                    cell, neighbor_cell, axis, area
                )
                thermal = tran_thermal * thermal_link_factor(cell, neighbor_cell)
                hydraulic_connections.append((cell_idx, neighbor_idx, hydraulic))
                thermal_connections.append((cell_idx, neighbor_idx, thermal))

                crosses_lgr_interface = axis == upscaling_axis and (
                    (cell_idx in interface_ids and not is_lgr[neighbor_idx])
                    or (neighbor_idx in interface_ids and not is_lgr[cell_idx])
                )
                if crosses_lgr_interface:
                    crossing.append((cell_idx, neighbor_idx))

        if not crossing:
            raise ValueError("No local LGR interface links found for flow upscaling.")
        return hydraulic_connections, thermal_connections, crossing

    @staticmethod
    def _solve_conductance(
        n_cells: int,
        connections: list[tuple[int, int, float]],
        crossing: list[tuple[int, int]],
        interface_ids: set[int],
        support_ids: set[int],
        far_ids: set[int],
    ) -> float:
        """
        Solve the local problem and return the equivalent conductance.

        The same code is used for hydraulic conductance and for physical
        thermal conductance. The interface is fixed to ``p = 1``, the far side
        is fixed to ``p = 0``, and the flux through the LGR face is converted
        to one equivalent value.
        """

        fixed_values = {idx: 1.0 for idx in interface_ids}
        fixed_values.update({idx: 0.0 for idx in far_ids})
        conductance_by_pair = {
            (min(cell_idx, neighbor_idx), max(cell_idx, neighbor_idx)): conductance
            for cell_idx, neighbor_idx, conductance in connections
        }
        crossing_conductance = sum(
            conductance_by_pair[
                (min(cell_idx, neighbor_idx), max(cell_idx, neighbor_idx))
            ]
            for cell_idx, neighbor_idx in crossing
        )
        if crossing_conductance <= 0.0:
            return 0.0

        active_nodes = _nodes_reachable_from_fixed(
            n_cells, connections, set(fixed_values)
        )
        unknown_ids = [
            idx
            for idx in range(n_cells)
            if idx not in fixed_values and idx in active_nodes
        ]
        unknown_pos = {idx: pos for pos, idx in enumerate(unknown_ids)}

        pressure = np.zeros(n_cells, dtype=float)
        for idx, value in fixed_values.items():
            pressure[idx] = value

        if unknown_ids:
            matrix_rows = []
            matrix_cols = []
            matrix_data = []
            rhs = np.zeros(len(unknown_ids), dtype=float)
            for cell_idx, neighbor_idx, conductance in connections:
                if conductance <= 0.0:
                    continue
                _add_conductance_to_system(
                    matrix_rows,
                    matrix_cols,
                    matrix_data,
                    rhs,
                    unknown_pos,
                    fixed_values,
                    cell_idx,
                    neighbor_idx,
                    conductance,
                )
            matrix = coo_matrix(
                (matrix_data, (matrix_rows, matrix_cols)),
                shape=(len(unknown_ids), len(unknown_ids)),
            ).tocsr()
            pressure[unknown_ids] = spsolve(matrix, rhs)

        total_flux = 0.0
        for cell_idx, neighbor_idx in crossing:
            conductance = conductance_by_pair[
                (min(cell_idx, neighbor_idx), max(cell_idx, neighbor_idx))
            ]
            if cell_idx in interface_ids:
                total_flux += conductance * (
                    pressure[cell_idx] - pressure[neighbor_idx]
                )
            else:
                total_flux += conductance * (
                    pressure[neighbor_idx] - pressure[cell_idx]
                )

        dp_macro = 1.0 - float(np.mean(pressure[list(support_ids)]))
        if abs(dp_macro) <= 1e-14:
            raise ZeroDivisionError("Local flow-upscaling pressure drop is too small.")
        return max(float(total_flux / dp_macro), 0.0)


def scale_by_raw_distribution(
    raw_values: np.ndarray, target_total: float
) -> np.ndarray:
    """
    Spread one side total over real links using raw values as weights.
    """

    raw_total = float(np.sum(raw_values))
    if raw_total > 0.0:
        return raw_values * (target_total / raw_total)
    if raw_values.size == 0:
        return raw_values
    return np.full(raw_values.shape, target_total / raw_values.size, dtype=float)


def thermal_scale_from_physical_conductance(
    cells: list[Any],
    connections: list[CoarseFineConnection],
    raw_tran_thermal: np.ndarray,
    target_physical_total: float,
) -> float:
    """
    Return the scale factor for the stored thermal ``tranD`` values.

    The local solve gives a physical conductive total. The global connection
    list stores geometric ``tranD`` values, so the code compares them after
    applying each link's rock conductive factor.
    """

    raw_physical_total = 0.0
    for conn, raw_value in zip(connections, raw_tran_thermal, strict=True):
        lgr_cell = cells[conn.lgr_cell_idx]
        coarse_cell = cells[conn.coarse_cell_idx]
        raw_physical_total += raw_value * thermal_link_factor(lgr_cell, coarse_cell)
    if raw_physical_total <= 0.0:
        return 0.0
    return target_physical_total / raw_physical_total


def thermal_link_factor(cell_a: Any, cell_b: Any) -> float:
    """
    Return the average rock conductive factor for two connected cells.
    """

    return 0.5 * (
        (1.0 - cell_a.poro) * cell_a.rcond + (1.0 - cell_b.poro) * cell_b.rcond
    )


def _ids_on_fine_plane(grid: np.ndarray, axis: int, coordinate: int) -> set[int]:
    ids = set()
    slicer = [slice(None)] * grid.ndim
    slicer[axis] = coordinate
    for value in np.asarray(grid[tuple(slicer)]).reshape(-1):
        if value >= 0:
            ids.add(int(value))
    return ids


def _ids_on_lgr_interface(
    grid: np.ndarray,
    axis: int,
    side: int,
    parent_lo: np.ndarray,
    patch_lo: np.ndarray,
    patch_hi: np.ndarray,
    refine: np.ndarray,
) -> set[int]:
    normal_coord = (
        int((patch_lo[axis] - parent_lo[axis]) * refine[axis])
        if side < 0
        else int((patch_hi[axis] - parent_lo[axis] + 1) * refine[axis] - 1)
    )
    interface_ids = set()

    for index in np.ndindex(grid.shape):
        if index[axis] != normal_coord:
            continue
        in_patch_tangent = True
        for dim in range(3):
            if dim == axis:
                continue
            tangent_start = int((patch_lo[dim] - parent_lo[dim]) * refine[dim])
            tangent_stop = int((patch_hi[dim] - parent_lo[dim] + 1) * refine[dim])
            if not tangent_start <= index[dim] < tangent_stop:
                in_patch_tangent = False
                break
        if in_patch_tangent:
            cell_idx = int(grid[index])
            if cell_idx >= 0:
                interface_ids.add(cell_idx)
    return interface_ids


def _ids_in_adjacent_parent_support(
    grid: np.ndarray,
    axis: int,
    side: int,
    parent_lo: np.ndarray,
    patch_lo: np.ndarray,
    patch_hi: np.ndarray,
    refine: np.ndarray,
) -> set[int]:
    normal_parent = patch_lo[axis] - 1 if side < 0 else patch_hi[axis] + 1
    normal_start = int((normal_parent - parent_lo[axis]) * refine[axis])
    normal_stop = normal_start + int(refine[axis])
    support_ids = set()

    for index in np.ndindex(grid.shape):
        if not normal_start <= index[axis] < normal_stop:
            continue
        in_patch_tangent = True
        for dim in range(3):
            if dim == axis:
                continue
            tangent_start = int((patch_lo[dim] - parent_lo[dim]) * refine[dim])
            tangent_stop = int((patch_hi[dim] - parent_lo[dim] + 1) * refine[dim])
            if not tangent_start <= index[dim] < tangent_stop:
                in_patch_tangent = False
                break
        if in_patch_tangent:
            cell_idx = int(grid[index])
            if cell_idx >= 0:
                support_ids.add(cell_idx)
    return support_ids


def _cell_face_overlap_area(cell_a: Any, cell_b: Any, axis: int) -> float:
    if axis == 0:
        overlap0 = max(
            0.0, min(cell_a.y_max, cell_b.y_max) - max(cell_a.y_min, cell_b.y_min)
        )
        overlap1 = max(
            0.0, min(cell_a.z_max, cell_b.z_max) - max(cell_a.z_min, cell_b.z_min)
        )
    elif axis == 1:
        overlap0 = max(
            0.0, min(cell_a.x_max, cell_b.x_max) - max(cell_a.x_min, cell_b.x_min)
        )
        overlap1 = max(
            0.0, min(cell_a.z_max, cell_b.z_max) - max(cell_a.z_min, cell_b.z_min)
        )
    else:
        overlap0 = max(
            0.0, min(cell_a.x_max, cell_b.x_max) - max(cell_a.x_min, cell_b.x_min)
        )
        overlap1 = max(
            0.0, min(cell_a.y_max, cell_b.y_max) - max(cell_a.y_min, cell_b.y_min)
        )
    return overlap0 * overlap1


def _add_conductance_to_system(
    matrix_rows: list[int],
    matrix_cols: list[int],
    matrix_data: list[float],
    rhs: np.ndarray,
    unknown_pos: dict[int, int],
    fixed_values: dict[int, float],
    cell_idx: int,
    neighbor_idx: int,
    conductance: float,
) -> None:
    cell_unknown = cell_idx in unknown_pos
    neighbor_unknown = neighbor_idx in unknown_pos
    if cell_unknown:
        row = unknown_pos[cell_idx]
        _add_matrix_value(matrix_rows, matrix_cols, matrix_data, row, row, conductance)
        if neighbor_unknown:
            _add_matrix_value(
                matrix_rows,
                matrix_cols,
                matrix_data,
                row,
                unknown_pos[neighbor_idx],
                -conductance,
            )
        else:
            rhs[row] += conductance * fixed_values[neighbor_idx]
    if neighbor_unknown:
        row = unknown_pos[neighbor_idx]
        _add_matrix_value(matrix_rows, matrix_cols, matrix_data, row, row, conductance)
        if cell_unknown:
            _add_matrix_value(
                matrix_rows,
                matrix_cols,
                matrix_data,
                row,
                unknown_pos[cell_idx],
                -conductance,
            )
        else:
            rhs[row] += conductance * fixed_values[cell_idx]


def _add_matrix_value(
    matrix_rows: list[int],
    matrix_cols: list[int],
    matrix_data: list[float],
    row: int,
    col: int,
    value: float,
) -> None:
    matrix_rows.append(row)
    matrix_cols.append(col)
    matrix_data.append(value)


def _nodes_reachable_from_fixed(
    n_cells: int,
    connections: list[tuple[int, int, float]],
    fixed_ids: set[int],
) -> set[int]:
    adjacency: list[list[int]] = [[] for _ in range(n_cells)]
    for cell_idx, neighbor_idx, conductance in connections:
        if conductance <= 0.0:
            continue
        adjacency[cell_idx].append(neighbor_idx)
        adjacency[neighbor_idx].append(cell_idx)

    reachable = set(fixed_ids)
    stack = list(fixed_ids)
    while stack:
        cell_idx = stack.pop()
        for neighbor_idx in adjacency[cell_idx]:
            if neighbor_idx not in reachable:
                reachable.add(neighbor_idx)
                stack.append(neighbor_idx)
    return reachable


def _scaled_fine_volume(
    dx: float,
    dy: float,
    dz: float,
    parent_dx: float,
    parent_dy: float,
    parent_dz: float,
    parent_volume: float,
) -> float:
    fine_volume = dx * dy * dz
    parent_geometric_volume = parent_dx * parent_dy * parent_dz
    if parent_geometric_volume <= 0.0:
        return fine_volume
    return fine_volume * parent_volume / parent_geometric_volume


def _ijk0_to_global(i: int, j: int, k: int, nx: int, ny: int) -> int:
    return i + nx * (j + ny * k)
