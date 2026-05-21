from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np

from darts.engines import conn_mesh, index_vector, ms_well, timer_node, value_vector
from darts.reservoirs.mesh.struct_discretizer import StructDiscretizer
from darts.reservoirs.reservoir_base import ReservoirBase
from darts.reservoirs.struct_reservoir import StructReservoir

_MIN_TRAN = 1e-5


@dataclass
class LGRPatch:
    """
    Local structured refinement patch inside a parent structured reservoir.

    Ranges are one-based and inclusive, matching the I/J/K convention used by
    :class:`StructReservoir.add_perforation`.

    :param name: Unique patch name.
    :type name: str
    :param i_range: Parent-grid I range, one-based and inclusive.
    :type i_range: tuple[int, int]
    :param j_range: Parent-grid J range, one-based and inclusive.
    :type j_range: tuple[int, int]
    :param k_range: Parent-grid K range, one-based and inclusive.
    :type k_range: tuple[int, int]
    :param refine: Refinement ratio per parent cell in I/J/K directions.
    :type refine: tuple[int, int, int]
    :param poro: Optional fine-cell porosity override.
    :type poro: Any
    :param permx: Optional fine-cell x-permeability override.
    :type permx: Any
    :param permy: Optional fine-cell y-permeability override.
    :type permy: Any
    :param permz: Optional fine-cell z-permeability override.
    :type permz: Any
    :param rcond: Optional fine-cell rock conductivity override.
    :type rcond: Any
    :param hcap: Optional fine-cell heat-capacity override.
    :type hcap: Any
    :param op_num: Optional fine-cell operator-region override.
    :type op_num: Any
    """

    name: str
    i_range: tuple[int, int]
    j_range: tuple[int, int]
    k_range: tuple[int, int]
    refine: tuple[int, int, int]
    poro: Any = None
    permx: Any = None
    permy: Any = None
    permz: Any = None
    rcond: Any = None
    hcap: Any = None
    op_num: Any = None

    def __post_init__(self) -> None:
        self.i_range = _as_range(self.i_range, "i_range")
        self.j_range = _as_range(self.j_range, "j_range")
        self.k_range = _as_range(self.k_range, "k_range")
        self.refine = _as_refine(self.refine)

    @property
    def parent_shape(self) -> tuple[int, int, int]:
        return (
            self.i_range[1] - self.i_range[0] + 1,
            self.j_range[1] - self.j_range[0] + 1,
            self.k_range[1] - self.k_range[0] + 1,
        )

    @property
    def fine_shape(self) -> tuple[int, int, int]:
        pi, pj, pk = self.parent_shape
        ri, rj, rk = self.refine
        return pi * ri, pj * rj, pk * rk


@dataclass(frozen=True)
class _Cell:
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
    lgr_name: str | None = None
    lgr_ijk: tuple[int, int, int] | None = None

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            0.5 * (self.x_min + self.x_max),
            0.5 * (self.y_min + self.y_max),
            0.5 * (self.z_min + self.z_max),
        )


@dataclass(frozen=True)
class _Face:
    cell: _Cell
    lo0: float
    hi0: float
    lo1: float
    hi1: float


class LGRStructReservoir(ReservoirBase):
    """
    Structured reservoir assembled from a parent grid and local refinements.

    The parent cells covered by an LGR patch are replaced by fine cells. The
    resulting mesh is assembled as a connection list, which means EPM and DFM
    wells can perforate either parent cells or fine LGR cells.

    :param timer: Timer node.
    :type timer: timer_node
    :param parent: Parent structured reservoir.
    :type parent: StructReservoir
    :param lgrs: Local refinement patches.
    :type lgrs: list[LGRPatch]
    :param cache: Reservoir cache flag.
    :type cache: bool
    """

    def __init__(
        self,
        timer: timer_node,
        parent: StructReservoir,
        lgrs: list[LGRPatch],
        cache: bool = False,
    ):
        super().__init__(timer, cache)
        if parent.is_cpg:
            raise NotImplementedError(
                "LGRStructReservoir currently supports non-CPG structured grids only."
            )

        self.parent = parent
        self.lgrs = list(lgrs)
        self.nx = parent.nx
        self.ny = parent.ny
        self.nz = parent.nz
        self.ndims = self._infer_ndims()
        self.boundary_volumes = dict(parent.boundary_volumes)
        self.connected_well_segments = {}

        self.cells: list[_Cell] = []
        self.parent_cell_map: dict[int, int] = {}
        self.refined_parent_cells: dict[int, str] = {}
        self.lgr_cell_maps: dict[str, np.ndarray] = {}
        self.lgr_offsets: dict[str, int] = {}
        self.centroids_all_cells: np.ndarray | None = None
        self.centroids: np.ndarray | None = None

        self.global_data: dict[str, Any] = {}
        self.actnum = None

    def discretize(self, cache: bool = False, verbose: bool = False) -> conn_mesh:
        self.parent.boundary_volumes = dict(self.boundary_volumes)
        self.parent.discretize(verbose=verbose)

        self._build_cells()
        cell_m, cell_p, tran, tran_thermal = self._build_connections()

        mesh = conn_mesh()
        mesh.init(
            index_vector(cell_m),
            index_vector(cell_p),
            value_vector(tran),
            value_vector(tran_thermal),
        )

        self.cell_m = cell_m
        self.cell_p = cell_p
        self.tran = tran
        self.tran_thermal = tran_thermal

        poro = np.asarray([cell.poro for cell in self.cells], dtype=float)
        rcond = np.asarray([cell.rcond for cell in self.cells], dtype=float)
        hcap = np.asarray([cell.hcap for cell in self.cells], dtype=float)
        depth = np.asarray([cell.center[2] for cell in self.cells], dtype=float)
        volume = np.asarray([cell.volume for cell in self.cells], dtype=float)
        op_num = np.asarray([cell.op_num for cell in self.cells], dtype=np.int32)

        np.array(mesh.poro, copy=False)[:] = poro
        np.array(mesh.rock_cond, copy=False)[:] = rcond
        np.array(mesh.heat_capacity, copy=False)[:] = hcap
        np.array(mesh.depth, copy=False)[:] = depth
        np.array(mesh.volume, copy=False)[:] = volume
        np.array(mesh.op_num, copy=False)[:] = op_num

        self.mesh = mesh
        self.n = len(self.cells)
        self.actnum = np.ones(self.n, dtype=np.int32)
        self.local_to_global = np.arange(self.n, dtype=np.int32)
        self.global_to_local = np.arange(self.n, dtype=np.int32)

        self.poro = poro
        self.rcond = rcond
        self.hcap = hcap
        self.depth = depth
        self.volume = np.array(mesh.volume, copy=False)
        self.op_num = op_num
        self.dx = np.asarray([cell.dx for cell in self.cells], dtype=float)
        self.dy = np.asarray([cell.dy for cell in self.cells], dtype=float)
        self.dz = np.asarray([cell.dz for cell in self.cells], dtype=float)
        self.permx = np.asarray([cell.permx for cell in self.cells], dtype=float)
        self.permy = np.asarray([cell.permy for cell in self.cells], dtype=float)
        self.permz = np.asarray([cell.permz for cell in self.cells], dtype=float)
        self.centroids_all_cells = np.asarray(
            [cell.center for cell in self.cells], dtype=float
        )
        self.centroids = self.centroids_all_cells

        self.global_data = {
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "poro": self.poro,
            "permx": self.permx,
            "permy": self.permy,
            "permz": self.permz,
            "rcond": self.rcond,
            "hcap": self.hcap,
            "depth": self.depth,
            "volume": np.array(self.volume, copy=True),
            "op_num": self.op_num,
            "actnum": self.actnum,
            "global_to_local": self.global_to_local,
        }
        return mesh

    def set_boundary_volume(self, boundary_volumes: dict):
        self.boundary_volumes = dict(boundary_volumes)

    def get_lgr_cell_index(
        self, lgr_name: str, lgr_cell_idx: tuple[int, int, int]
    ) -> int:
        """
        Return the assembled cell index for a fine LGR cell.

        :param lgr_name: LGR patch name.
        :type lgr_name: str
        :param lgr_cell_idx: Fine-cell I/J/K index inside the LGR, one-based.
        :type lgr_cell_idx: tuple[int, int, int]
        :return: Assembled reservoir block index.
        :rtype: int
        """
        if lgr_name not in self.lgr_cell_maps:
            raise KeyError(f"Unknown LGR patch {lgr_name!r}.")
        i, j, k = lgr_cell_idx
        cell_map = self.lgr_cell_maps[lgr_name]
        if (
            i < 1
            or j < 1
            or k < 1
            or i > cell_map.shape[0]
            or j > cell_map.shape[1]
            or k > cell_map.shape[2]
        ):
            raise IndexError(
                f"LGR cell index {lgr_cell_idx} is outside patch {lgr_name!r} with shape {cell_map.shape}."
            )
        cell_index = int(cell_map[i - 1, j - 1, k - 1])
        if cell_index < 0:
            raise ValueError(
                f"LGR cell {lgr_cell_idx} in patch {lgr_name!r} is inactive."
            )
        return cell_index

    def get_parent_cell_index(self, res_cell_idx: tuple[int, int, int]) -> int:
        """
        Return the assembled cell index for an unrefined parent cell.

        :param res_cell_idx: Parent-grid I/J/K index, one-based.
        :type res_cell_idx: tuple[int, int, int]
        :return: Assembled reservoir block index.
        :rtype: int
        """
        parent_global = self._parent_global_from_ijk(res_cell_idx)
        if parent_global in self.refined_parent_cells:
            lgr_name = self.refined_parent_cells[parent_global]
            raise ValueError(
                f"Parent cell {res_cell_idx} is replaced by LGR patch {lgr_name!r}; "
                "use lgr_name and lgr_cell_idx to select a fine cell."
            )
        if parent_global not in self.parent_cell_map:
            raise ValueError(
                f"Parent cell {res_cell_idx} is inactive or outside the assembled reservoir."
            )
        return self.parent_cell_map[parent_global]

    def resolve_cell_index(
        self,
        res_cell_idx: int | tuple[int, int, int] | None = None,
        lgr_name: str | None = None,
        lgr_cell_idx: tuple[int, int, int] | None = None,
    ) -> int:
        """
        Resolve a parent, LGR, or assembled cell reference to a reservoir block.

        :param res_cell_idx: Assembled integer cell id or parent I/J/K tuple.
        :type res_cell_idx: int | tuple[int, int, int] | None
        :param lgr_name: Optional LGR patch name.
        :type lgr_name: str | None
        :param lgr_cell_idx: Fine-cell I/J/K tuple inside the LGR, one-based.
        :type lgr_cell_idx: tuple[int, int, int] | None
        :return: Assembled reservoir block index.
        :rtype: int
        """
        if lgr_name is not None:
            fine_idx = lgr_cell_idx if lgr_cell_idx is not None else res_cell_idx
            if not isinstance(fine_idx, tuple):
                raise TypeError(
                    "An LGR perforation requires lgr_cell_idx or a tuple res_cell_idx."
                )
            return self.get_lgr_cell_index(lgr_name, fine_idx)
        if isinstance(res_cell_idx, int):
            if res_cell_idx < 0 or res_cell_idx >= len(self.cells):
                raise IndexError(
                    f"Assembled cell index {res_cell_idx} is outside [0, {len(self.cells)})."
                )
            return int(res_cell_idx)
        if isinstance(res_cell_idx, tuple):
            return self.get_parent_cell_index(res_cell_idx)
        raise TypeError("A perforation requires res_cell_idx or lgr_name/lgr_cell_idx.")

    def add_perforation(
        self,
        well_name: str,
        res_cell_idx: int | tuple[int, int, int] | None = None,
        well_seg_idx: int = None,
        well_diameter: float = 0.1524,
        well_index: float = None,
        well_indexD: float = 0.0,
        segment_direction: str = "z_axis",
        skin: float = 0.0,
        ms_epm: bool = None,
        with_peaceman_for_coupled_well_reservoir: bool = False,
        verbose: bool = False,
        lgr_name: str | None = None,
        lgr_cell_idx: tuple[int, int, int] | None = None,
    ):
        """
        Add an EPM or DFM well perforation to an assembled parent or LGR cell.

        :param well_name: Well name.
        :type well_name: str
        :param res_cell_idx: Parent I/J/K tuple or assembled integer cell index.
        :type res_cell_idx: int | tuple[int, int, int] | None
        :param well_seg_idx: DFM well segment index, one-based from the wellhead.
        :type well_seg_idx: int
        :param well_diameter: Well diameter.
        :type well_diameter: float
        :param well_index: Optional flow well index override.
        :type well_index: float
        :param well_indexD: Optional thermal well index override.
        :type well_indexD: float
        :param segment_direction: Segment direction.
        :type segment_direction: str
        :param skin: Skin factor.
        :type skin: float
        :param ms_epm: Whether an EPM well uses one segment per perforation.
        :type ms_epm: bool
        :param with_peaceman_for_coupled_well_reservoir: Use Peaceman DFM perforation transmissibility.
        :type with_peaceman_for_coupled_well_reservoir: bool
        :param verbose: Print perforation information.
        :type verbose: bool
        :param lgr_name: Optional LGR patch name.
        :type lgr_name: str | None
        :param lgr_cell_idx: Fine-cell I/J/K tuple inside the LGR, one-based.
        :type lgr_cell_idx: tuple[int, int, int] | None
        """
        well = self.get_well(well_name)
        if well is None:
            raise ValueError(
                f"Unknown well {well_name!r}. Add it before adding perforations."
            )

        cell_index = self.resolve_cell_index(res_cell_idx, lgr_name, lgr_cell_idx)
        cell = self.cells[cell_index]

        if well.ms_type == ms_well.MS_Type.EPM:
            if well_seg_idx is not None:
                raise AssertionError(
                    "If the well is of the EPM type, well_seg_idx must not be specified!"
                )
            if with_peaceman_for_coupled_well_reservoir:
                raise AssertionError(
                    "Coupled well-reservoir can be used only if the well type is DFM!"
                )
            wi, wid = self._calc_well_index(
                cell, well_diameter, segment_direction, skin
            )
            well_block = len(well.perforations) if ms_epm else 0
        elif well.ms_type == ms_well.MS_Type.DFM:
            if well_seg_idx is None:
                raise AssertionError(
                    "If the well is of the DFM type, well_seg_idx must be specified!"
                )
            if ms_epm is not None:
                raise AssertionError(
                    "If the well is of the DFM type, ms_epm must not be specified!"
                )
            wi, wid = self._calc_dfm_well_index(
                cell,
                well_diameter,
                segment_direction,
                with_peaceman_for_coupled_well_reservoir,
                skin,
            )
            well_block = well_seg_idx - 2
        else:
            raise ValueError(f"Unsupported well type for well {well.name!r}.")

        if well_index is None:
            well_index = wi
        if well_indexD is None:
            well_indexD = wid

        if well.ms_type == ms_well.MS_Type.EPM:
            self._update_epm_well_geometry(well, cell, segment_direction)

        for perf in well.perforations:
            if perf[0] == well_block and perf[1] == cell_index:
                if verbose:
                    print(
                        f"Neglected duplicate perforation for well {well.name} to block {cell_index}"
                    )
                return

        assert well_index >= 0
        assert well_indexD >= 0
        well.perforations = well.perforations + [
            (well_block, cell_index, well_index, well_indexD)
        ]
        if verbose:
            print(
                f"Added perforation for well {well.name} to block {cell_index:d} "
                f"with WI={well_index:f} and WID={well_indexD:f}"
            )

    def find_cell_index(self, coord: list | np.ndarray) -> int:
        """
        Find the assembled cell closest to an XYZ coordinate.

        :param coord: XYZ coordinate.
        :type coord: list | np.ndarray
        :return: Assembled cell index.
        :rtype: int
        """
        centers = np.asarray(self.centroids_all_cells)
        distances = np.linalg.norm(centers - np.asarray(coord, dtype=float), axis=1)
        return int(np.argmin(distances))

    def init_vtk(self, output_directory: str, export_grid_data: bool = True):
        raise NotImplementedError(
            "VTK output for LGRStructReservoir is not implemented yet."
        )

    def output_to_vtk(
        self,
        ith_step: int,
        t: float,
        output_directory: str,
        prop_names: list,
        data: dict,
    ):
        raise NotImplementedError(
            "VTK output for LGRStructReservoir is not implemented yet."
        )

    def _build_cells(self) -> None:
        self._validate_lgrs()
        parent_arrays = self._parent_arrays()
        active_parent_cells = set(
            np.asarray(self.parent.discretizer.local_to_global, dtype=int)
        )

        self.cells = []
        self.parent_cell_map = {}
        self.refined_parent_cells = {}
        self.lgr_cell_maps = {}
        self.lgr_offsets = {}

        patch_by_parent = self._patch_by_parent_cell()
        for parent_global in sorted(active_parent_cells):
            if parent_global in patch_by_parent:
                self.refined_parent_cells[parent_global] = patch_by_parent[
                    parent_global
                ].name
                continue
            self._append_parent_cell(parent_global, parent_arrays)

        for patch in self.lgrs:
            self.lgr_offsets[patch.name] = len(self.cells)
            self._append_lgr_cells(patch, parent_arrays, active_parent_cells)

    def _validate_lgrs(self) -> None:
        names = set()
        covered: set[int] = set()
        for patch in self.lgrs:
            if patch.name in names:
                raise ValueError(f"Duplicate LGR patch name {patch.name!r}.")
            names.add(patch.name)
            self._validate_patch_range(patch)
            for parent_global in self._iter_patch_parent_globals(patch):
                if parent_global in covered:
                    raise ValueError("Overlapping LGR patches are not supported.")
                covered.add(parent_global)

    def _validate_patch_range(self, patch: LGRPatch) -> None:
        if patch.i_range[0] < 1 or patch.i_range[1] > self.nx:
            raise ValueError(
                f"LGR patch {patch.name!r} i_range is outside parent grid."
            )
        if patch.j_range[0] < 1 or patch.j_range[1] > self.ny:
            raise ValueError(
                f"LGR patch {patch.name!r} j_range is outside parent grid."
            )
        if patch.k_range[0] < 1 or patch.k_range[1] > self.nz:
            raise ValueError(
                f"LGR patch {patch.name!r} k_range is outside parent grid."
            )

    def _patch_by_parent_cell(self) -> dict[int, LGRPatch]:
        mapping = {}
        for patch in self.lgrs:
            for parent_global in self._iter_patch_parent_globals(patch):
                mapping[parent_global] = patch
        return mapping

    def _iter_patch_parent_globals(self, patch: LGRPatch):
        for k in range(patch.k_range[0] - 1, patch.k_range[1]):
            for j in range(patch.j_range[0] - 1, patch.j_range[1]):
                for i in range(patch.i_range[0] - 1, patch.i_range[1]):
                    yield _ijk0_to_global(i, j, k, self.nx, self.ny)

    def _parent_arrays(self) -> dict[str, np.ndarray]:
        disc = self.parent.discretizer
        if self.parent.global_data["depth"] is None:
            self.parent.global_data["depth"] = disc.centroids_all_cells[:, 2].flatten(
                order="F"
            )
        names = ["poro", "permx", "permy", "permz", "rcond", "hcap", "depth", "op_num"]
        arrays = {
            name: disc.convert_to_flat_array(self.parent.global_data[name], name)
            for name in names
        }
        arrays["dx"] = disc.convert_to_flat_array(self.parent.global_data["dx"], "dx")
        arrays["dy"] = disc.convert_to_flat_array(self.parent.global_data["dy"], "dy")
        arrays["dz"] = disc.convert_to_flat_array(self.parent.global_data["dz"], "dz")
        arrays["volume"] = arrays["dx"] * arrays["dy"] * arrays["dz"]
        parent_volume = np.asarray(self.parent.volume, dtype=float)
        for local_idx, parent_global in enumerate(disc.local_to_global):
            arrays["volume"][parent_global] = parent_volume[local_idx]
        centers = np.array(disc.centroids_all_cells, dtype=float, copy=True)
        centers[:, 2] = arrays["depth"]
        arrays["centers"] = centers
        return arrays

    def _append_parent_cell(
        self, parent_global: int, parent_arrays: dict[str, np.ndarray]
    ) -> None:
        idx = len(self.cells)
        center = parent_arrays["centers"][parent_global]
        dx = float(parent_arrays["dx"][parent_global])
        dy = float(parent_arrays["dy"][parent_global])
        dz = float(parent_arrays["dz"][parent_global])
        cell = _Cell(
            idx=idx,
            name=f"parent:{parent_global}",
            dx=dx,
            dy=dy,
            dz=dz,
            x_min=float(center[0] - 0.5 * dx),
            x_max=float(center[0] + 0.5 * dx),
            y_min=float(center[1] - 0.5 * dy),
            y_max=float(center[1] + 0.5 * dy),
            z_min=float(center[2] - 0.5 * dz),
            z_max=float(center[2] + 0.5 * dz),
            poro=float(parent_arrays["poro"][parent_global]),
            permx=float(parent_arrays["permx"][parent_global]),
            permy=float(parent_arrays["permy"][parent_global]),
            permz=float(parent_arrays["permz"][parent_global]),
            rcond=float(parent_arrays["rcond"][parent_global]),
            hcap=float(parent_arrays["hcap"][parent_global]),
            op_num=int(parent_arrays["op_num"][parent_global]),
            volume=float(parent_arrays["volume"][parent_global]),
            parent_global=parent_global,
        )
        self.cells.append(cell)
        self.parent_cell_map[parent_global] = idx

    def _append_lgr_cells(
        self,
        patch: LGRPatch,
        parent_arrays: dict[str, np.ndarray],
        active_parent_cells: set[int],
    ) -> None:
        patch_arrays = {
            "poro": _prepare_patch_array(patch.poro, patch, "poro", float),
            "permx": _prepare_patch_array(patch.permx, patch, "permx", float),
            "permy": _prepare_patch_array(patch.permy, patch, "permy", float),
            "permz": _prepare_patch_array(patch.permz, patch, "permz", float),
            "rcond": _prepare_patch_array(patch.rcond, patch, "rcond", float),
            "hcap": _prepare_patch_array(patch.hcap, patch, "hcap", float),
            "op_num": _prepare_patch_array(patch.op_num, patch, "op_num", int),
        }
        cell_map = np.full(patch.fine_shape, -1, dtype=np.int32)
        ri, rj, rk = patch.refine

        for pk, parent_k in enumerate(range(patch.k_range[0] - 1, patch.k_range[1])):
            for pj, parent_j in enumerate(
                range(patch.j_range[0] - 1, patch.j_range[1])
            ):
                for pi_, parent_i in enumerate(
                    range(patch.i_range[0] - 1, patch.i_range[1])
                ):
                    parent_global = _ijk0_to_global(
                        parent_i, parent_j, parent_k, self.nx, self.ny
                    )
                    if parent_global not in active_parent_cells:
                        raise ValueError(
                            f"LGR patch {patch.name!r} covers inactive parent cell "
                            f"{(parent_i + 1, parent_j + 1, parent_k + 1)}."
                        )
                    parent_center = parent_arrays["centers"][parent_global]
                    parent_dx = float(parent_arrays["dx"][parent_global])
                    parent_dy = float(parent_arrays["dy"][parent_global])
                    parent_dz = float(parent_arrays["dz"][parent_global])
                    x0 = float(parent_center[0] - 0.5 * parent_dx)
                    y0 = float(parent_center[1] - 0.5 * parent_dy)
                    z0 = float(parent_center[2] - 0.5 * parent_dz)
                    dx = parent_dx / ri
                    dy = parent_dy / rj
                    dz = parent_dz / rk
                    volume = _scaled_fine_volume(
                        dx,
                        dy,
                        dz,
                        parent_dx,
                        parent_dy,
                        parent_dz,
                        float(parent_arrays["volume"][parent_global]),
                    )

                    for fk in range(rk):
                        for fj in range(rj):
                            for fi in range(ri):
                                lgr_i = pi_ * ri + fi
                                lgr_j = pj * rj + fj
                                lgr_k = pk * rk + fk
                                idx = len(self.cells)
                                cell_map[lgr_i, lgr_j, lgr_k] = idx
                                self.cells.append(
                                    _Cell(
                                        idx=idx,
                                        name=f"{patch.name}:{lgr_i + 1},{lgr_j + 1},{lgr_k + 1}",
                                        dx=dx,
                                        dy=dy,
                                        dz=dz,
                                        x_min=x0 + fi * dx,
                                        x_max=x0 + (fi + 1) * dx,
                                        y_min=y0 + fj * dy,
                                        y_max=y0 + (fj + 1) * dy,
                                        z_min=z0 + fk * dz,
                                        z_max=z0 + (fk + 1) * dz,
                                        poro=self._cell_prop(
                                            patch_arrays["poro"],
                                            parent_arrays["poro"],
                                            parent_global,
                                            lgr_i,
                                            lgr_j,
                                            lgr_k,
                                        ),
                                        permx=self._cell_prop(
                                            patch_arrays["permx"],
                                            parent_arrays["permx"],
                                            parent_global,
                                            lgr_i,
                                            lgr_j,
                                            lgr_k,
                                        ),
                                        permy=self._cell_prop(
                                            patch_arrays["permy"],
                                            parent_arrays["permy"],
                                            parent_global,
                                            lgr_i,
                                            lgr_j,
                                            lgr_k,
                                        ),
                                        permz=self._cell_prop(
                                            patch_arrays["permz"],
                                            parent_arrays["permz"],
                                            parent_global,
                                            lgr_i,
                                            lgr_j,
                                            lgr_k,
                                        ),
                                        rcond=self._cell_prop(
                                            patch_arrays["rcond"],
                                            parent_arrays["rcond"],
                                            parent_global,
                                            lgr_i,
                                            lgr_j,
                                            lgr_k,
                                        ),
                                        hcap=self._cell_prop(
                                            patch_arrays["hcap"],
                                            parent_arrays["hcap"],
                                            parent_global,
                                            lgr_i,
                                            lgr_j,
                                            lgr_k,
                                        ),
                                        op_num=int(
                                            self._cell_prop(
                                                patch_arrays["op_num"],
                                                parent_arrays["op_num"],
                                                parent_global,
                                                lgr_i,
                                                lgr_j,
                                                lgr_k,
                                            )
                                        ),
                                        volume=volume,
                                        parent_global=parent_global,
                                        lgr_name=patch.name,
                                        lgr_ijk=(lgr_i + 1, lgr_j + 1, lgr_k + 1),
                                    )
                                )
        self.lgr_cell_maps[patch.name] = cell_map

    @staticmethod
    def _cell_prop(
        patch_array: np.ndarray | None,
        parent_array: np.ndarray,
        parent_global: int,
        i: int,
        j: int,
        k: int,
    ) -> float:
        if patch_array is None:
            return float(parent_array[parent_global])
        return float(patch_array[i, j, k])

    def _build_connections(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cell_m: list[int] = []
        cell_p: list[int] = []
        tran: list[float] = []
        tran_thermal: list[float] = []
        for axis in range(3):
            minus_faces, plus_faces = self._faces_by_axis(axis)
            for plane, faces_m in minus_faces.items():
                faces_p = plus_faces.get(plane, [])
                for face_m in faces_m:
                    for face_p in faces_p:
                        area = _overlap_area(face_m, face_p)
                        if area <= 0.0:
                            continue
                        tm, tt = self._connection_transmissibility(
                            face_m.cell, face_p.cell, axis, area
                        )
                        if tm > _MIN_TRAN or tt > _MIN_TRAN:
                            cell_m.append(face_m.cell.idx)
                            cell_p.append(face_p.cell.idx)
                            tran.append(tm)
                            tran_thermal.append(tt)
        return (
            np.asarray(cell_m, dtype=np.int32),
            np.asarray(cell_p, dtype=np.int32),
            np.asarray(tran, dtype=float),
            np.asarray(tran_thermal, dtype=float),
        )

    def _faces_by_axis(
        self, axis: int
    ) -> tuple[dict[float, list[_Face]], dict[float, list[_Face]]]:
        minus_faces: dict[float, list[_Face]] = {}
        plus_faces: dict[float, list[_Face]] = {}
        for cell in self.cells:
            if axis == 0:
                minus_key = _plane_key(cell.x_max)
                plus_key = _plane_key(cell.x_min)
                minus_face = _Face(cell, cell.y_min, cell.y_max, cell.z_min, cell.z_max)
                plus_face = _Face(cell, cell.y_min, cell.y_max, cell.z_min, cell.z_max)
            elif axis == 1:
                minus_key = _plane_key(cell.y_max)
                plus_key = _plane_key(cell.y_min)
                minus_face = _Face(cell, cell.x_min, cell.x_max, cell.z_min, cell.z_max)
                plus_face = _Face(cell, cell.x_min, cell.x_max, cell.z_min, cell.z_max)
            else:
                minus_key = _plane_key(cell.z_max)
                plus_key = _plane_key(cell.z_min)
                minus_face = _Face(cell, cell.x_min, cell.x_max, cell.y_min, cell.y_max)
                plus_face = _Face(cell, cell.x_min, cell.x_max, cell.y_min, cell.y_max)
            minus_faces.setdefault(minus_key, []).append(minus_face)
            plus_faces.setdefault(plus_key, []).append(plus_face)
        return minus_faces, plus_faces

    @staticmethod
    def _connection_transmissibility(
        cell_m: _Cell, cell_p: _Cell, axis: int, area: float
    ) -> tuple[float, float]:
        widths = (
            (cell_m.dx, cell_p.dx),
            (cell_m.dy, cell_p.dy),
            (cell_m.dz, cell_p.dz),
        )
        perms = (
            (cell_m.permx, cell_p.permx),
            (cell_m.permy, cell_p.permy),
            (cell_m.permz, cell_p.permz),
        )
        d_m = 0.5 * widths[axis][0]
        d_p = 0.5 * widths[axis][1]
        k_m, k_p = perms[axis]
        if k_m > 0.0 and k_p > 0.0:
            tran = area / (d_m / k_m + d_p / k_p) * StructDiscretizer.darcy_constant
        else:
            tran = 0.0
        tran_thermal = area / (d_m + d_p)
        return tran, tran_thermal

    def _parent_global_from_ijk(self, res_cell_idx: tuple[int, int, int]) -> int:
        i, j, k = res_cell_idx
        if i < 1 or j < 1 or k < 1 or i > self.nx or j > self.ny or k > self.nz:
            raise IndexError(
                f"Parent cell index {res_cell_idx} is outside {(self.nx, self.ny, self.nz)}."
            )
        return _ijk0_to_global(i - 1, j - 1, k - 1, self.nx, self.ny)

    @staticmethod
    def _calc_well_index(
        cell: _Cell,
        well_diameter: float,
        segment_direction: str,
        skin: float,
    ) -> tuple[float, float]:
        well_radius = well_diameter / 2
        return _peaceman_well_index(cell, well_radius, segment_direction, skin)

    @staticmethod
    def _calc_dfm_well_index(
        cell: _Cell,
        well_diameter: float,
        segment_direction: str,
        with_peaceman: bool,
        skin: float,
    ) -> tuple[float, float]:
        if segment_direction != "z_axis":
            raise ValueError(
                "Coupled well-reservoir model does not support non-z-axis segments!"
            )
        if with_peaceman:
            return _peaceman_well_index(
                cell, well_diameter / 2, segment_direction, skin
            )
        if skin != 0:
            raise AssertionError(
                "Skin factor can be applied only when the Peaceman model is used!"
            )
        well_radius = well_diameter / 2
        geom_coef = 2 * pi * cell.dz / np.log((cell.dx / 2 + well_radius) / well_radius)
        return cell.permx * geom_coef * StructDiscretizer.darcy_constant, geom_coef

    @staticmethod
    def _update_epm_well_geometry(
        well: ms_well, cell: _Cell, segment_direction: str
    ) -> None:
        segment_length = {"x_axis": cell.dx, "y_axis": cell.dy, "z_axis": cell.dz}[
            segment_direction
        ]
        if len(well.perforations) == 0:
            well.well_head_depth = cell.center[2]
            well.well_body_depth = well.well_head_depth
            well.segment_depth_increment = segment_length
            well.segment_volume *= well.segment_depth_increment
        else:
            well.well_head_depth = min(well.well_head_depth, cell.center[2])
            well.well_body_depth = well.well_head_depth

    def _infer_ndims(self) -> int:
        return (
            int(self.nx > 1 or any(patch.fine_shape[0] > 1 for patch in self.lgrs))
            + int(self.ny > 1 or any(patch.fine_shape[1] > 1 for patch in self.lgrs))
            + int(self.nz > 1 or any(patch.fine_shape[2] > 1 for patch in self.lgrs))
        )


def _peaceman_well_index(
    cell: _Cell,
    well_radius: float,
    segment_direction: str,
    skin: float,
) -> tuple[float, float]:
    well_index = 0.0
    well_indexD = 0.0
    if segment_direction == "z_axis":
        if cell.permx * cell.permy != 0:
            peaceman_rad = (
                0.28
                * np.sqrt(
                    np.sqrt(cell.permy / cell.permx) * cell.dx**2
                    + np.sqrt(cell.permx / cell.permy) * cell.dy**2
                )
                / (
                    (cell.permy / cell.permx) ** 0.25
                    + (cell.permx / cell.permy) ** 0.25
                )
            )
            well_index = (
                2
                * pi
                * cell.dz
                * np.sqrt(cell.permx * cell.permy)
                / (np.log(peaceman_rad / well_radius) + skin)
            )
            conduction_rad = 0.28 * np.sqrt(cell.dx**2 + cell.dy**2) / 2.0
            well_indexD = (
                2 * pi * cell.dz / (np.log(conduction_rad / well_radius) + skin)
            )
    elif segment_direction == "x_axis":
        if cell.permz * cell.permy != 0:
            peaceman_rad = (
                0.28
                * np.sqrt(
                    np.sqrt(cell.permy / cell.permz) * cell.dz**2
                    + np.sqrt(cell.permz / cell.permy) * cell.dy**2
                )
                / (
                    (cell.permy / cell.permz) ** 0.25
                    + (cell.permz / cell.permy) ** 0.25
                )
            )
            well_index = (
                2
                * pi
                * cell.dx
                * np.sqrt(cell.permz * cell.permy)
                / (np.log(peaceman_rad / well_radius) + skin)
            )
            conduction_rad = 0.28 * np.sqrt(cell.dz**2 + cell.dy**2) / 2.0
            well_indexD = (
                2 * pi * cell.dx / (np.log(conduction_rad / well_radius) + skin)
            )
    elif segment_direction == "y_axis":
        if cell.permx * cell.permz != 0:
            peaceman_rad = (
                0.28
                * np.sqrt(
                    np.sqrt(cell.permz / cell.permx) * cell.dx**2
                    + np.sqrt(cell.permx / cell.permz) * cell.dz**2
                )
                / (
                    (cell.permz / cell.permx) ** 0.25
                    + (cell.permx / cell.permz) ** 0.25
                )
            )
            well_index = (
                2
                * pi
                * cell.dy
                * np.sqrt(cell.permx * cell.permz)
                / (np.log(peaceman_rad / well_radius) + skin)
            )
            conduction_rad = 0.28 * np.sqrt(cell.dz**2 + cell.dx**2) / 2.0
            well_indexD = (
                2 * pi * cell.dy / (np.log(conduction_rad / well_radius) + skin)
            )
    else:
        raise ValueError(f"Unknown segment_direction {segment_direction!r}.")
    return well_index * StructDiscretizer.darcy_constant, well_indexD


def _as_range(value: tuple[int, int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two entries.")
    result = int(value[0]), int(value[1])
    if result[0] > result[1]:
        raise ValueError(f"{name} must be ordered as (first, last).")
    return result


def _as_refine(value: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError("refine must contain three entries.")
    result = int(value[0]), int(value[1]), int(value[2])
    if any(v < 1 for v in result):
        raise ValueError("refine entries must be positive.")
    return result


def _ijk0_to_global(i: int, j: int, k: int, nx: int, ny: int) -> int:
    return i + nx * (j + ny * k)


def _prepare_patch_array(data, patch: LGRPatch, name: str, dtype):
    if data is None:
        return None
    fine_shape = patch.fine_shape
    parent_shape = patch.parent_shape
    arr = np.asarray(data, dtype=dtype)
    if arr.ndim == 0:
        return np.full(fine_shape, arr.item(), dtype=dtype)
    if arr.shape == fine_shape:
        return arr
    if arr.shape == parent_shape:
        ri, rj, rk = patch.refine
        return np.repeat(np.repeat(np.repeat(arr, ri, axis=0), rj, axis=1), rk, axis=2)
    if arr.size == np.prod(fine_shape):
        return np.reshape(arr, fine_shape, order="F")
    raise ValueError(
        f"LGR patch {patch.name!r} property {name!r} has shape {arr.shape}; "
        f"expected scalar, parent shape {parent_shape}, or fine shape {fine_shape}."
    )


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


def _plane_key(value: float) -> float:
    return round(float(value), 12)


def _overlap_area(face_a: _Face, face_b: _Face) -> float:
    overlap0 = max(0.0, min(face_a.hi0, face_b.hi0) - max(face_a.lo0, face_b.lo0))
    overlap1 = max(0.0, min(face_a.hi1, face_b.hi1) - max(face_a.lo1, face_b.lo1))
    return overlap0 * overlap1
