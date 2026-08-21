from dataclasses import dataclass
from enum import Enum

import numpy as np

from darts.engines import ms_well, value_vector
from darts.models.conditions import BlockCSRView


class PI_Type(Enum):
    MOLAR = "molar"
    MASS = "mass"
    VOLUMETRIC = "volumetric"


# --------------------------------------------------------------------------
# Region-aware property resolution (shared by the darts.pipes hooks)
#
# A hook that evaluates fluid properties for a block must use the SAME
# property container the engine uses for that block, otherwise a multi-region
# model silently gets the wrong fluid. The mapping is:
#
#   * reservoir block b  ->  region ``physics.regions[mesh.op_num[b]]``
#     ``DartsModel.set_op_list`` builds
#     ``op_list = [acc_flux_itor[r] for r in physics.regions] + [acc_flux_w_itor]``
#     and the engine selects the operator set of block ``b`` with
#     ``block_idxs[mesh->op_num[b]]`` (``engine_base.h``), so ``op_num`` is an
#     INDEX INTO ``op_list``, i.e. slot ``i`` belongs to region
#     ``physics.regions[i]``. ``DartsModel.reconstruct_velocities`` fills the
#     engine's per-region ``molar_weights`` under exactly that convention.
#     For the usual registration order (regions 0, 1, ... n-1) the index and
#     the region tag coincide.
#   * well block  ->  region ``physics.regions[0]``
#     ``PhysicsBase.set_operators`` builds ``WellOperators`` /
#     ``WellCtrlOperators`` from ``property_containers[self.regions[0]]``, and
#     ``DartsModel.set_op_list`` overwrites ``mesh.op_num`` for the well blocks
#     with the well-operator slot (``len(op_list) - 1``), so ``op_num`` is NOT
#     a region tag there and must not be used.
#
# TODO(M4): move these helpers into a shared ``darts.pipes`` utility module
# once the conditions layer owns the block -> region mapping.
# --------------------------------------------------------------------------


def registered_regions(physics) -> list:
    """Region tags registered on ``physics``, in the engine's operator order.

    :param physics: physics object owning ``property_containers``
    :returns: ordered list of region tags (possibly empty)
    :rtype: list
    """
    regions = getattr(physics, "regions", None)
    if regions:
        return list(regions)
    # Lightweight test doubles may expose only property_containers; then the
    # container mapping's own ordering defines the regions.
    containers = getattr(physics, "property_containers", None)
    if isinstance(containers, dict):
        return list(containers)
    if containers is not None:
        return list(range(len(containers)))
    return []


def _no_regions_error(block_idx) -> KeyError:
    where = "" if block_idx is None else f"block {block_idx}: "
    return KeyError(
        f"{where}the physics has no registered property container "
        "(property_containers is empty). Register one with "
        "physics.add_property_region(property_container, region)."
    )


def well_cell_region(physics, block_idx: int = None):
    """Region tag whose property container the engine uses for WELL cells.

    :param physics: physics object owning ``property_containers``
    :param block_idx: well block index, used only in the error message
    :returns: region tag
    :raises KeyError: if no property container is registered at all
    """
    regions = registered_regions(physics)
    if not regions:
        raise _no_regions_error(block_idx)
    return regions[0]


def reservoir_cell_region(model, block_idx: int):
    """Region tag of a RESERVOIR block, from ``mesh.op_num``.

    :param model: model owning ``physics`` and ``reservoir``
    :param block_idx: reservoir block index
    :returns: region tag
    :raises KeyError: if the block's operator index has no registered region
    """
    physics = model.physics
    regions = registered_regions(physics)
    if not regions:
        raise _no_regions_error(block_idx)
    op_num = int(np.asarray(model.reservoir.mesh.op_num)[block_idx])
    if not 0 <= op_num < len(regions):
        raise KeyError(
            f"Block {block_idx} has operator index op_num={op_num}, which is "
            f"not one of the {len(regions)} registered operator sets; the "
            f"registered regions are {regions}. A reservoir block must point "
            "at a region with a property container -- resolving it to region "
            f"{regions[0]!r} would silently evaluate the wrong fluid."
        )
    return regions[op_num]


def block_region(model, block_idx: int):
    """Region tag of any block, reservoir or well.

    :param model: model owning ``physics`` and ``reservoir``
    :param block_idx: global block index
    :returns: region tag
    """
    n_res_blocks = int(model.reservoir.mesh.n_res_blocks)
    if block_idx >= n_res_blocks:
        return well_cell_region(model.physics, block_idx)
    return reservoir_cell_region(model, block_idx)


def property_container_of_region(physics, region, block_idx: int = None):
    """Property container registered for ``region``.

    :param physics: physics object owning ``property_containers``
    :param region: region tag
    :param block_idx: block index the region was resolved from (error message)
    :returns: the property container
    :raises KeyError: if ``region`` has no registered property container
    """
    containers = getattr(physics, "property_containers", None)
    if containers is None:
        raise _no_regions_error(block_idx)
    try:
        return containers[region]
    except (KeyError, IndexError, TypeError):
        where = "" if block_idx is None else f"Block {block_idx} resolves to "
        raise KeyError(
            f"{where}region {region!r}, which has no registered property "
            f"container; the registered regions are "
            f"{registered_regions(physics)}. Register one with "
            "physics.add_property_region(property_container, region) -- "
            "falling back to the first region would silently evaluate the "
            "wrong fluid."
        ) from None


def property_container_for_block(model, block_idx: int):
    """Property container the engine uses for ``block_idx``.

    Resolves the region from the block itself (``mesh.op_num`` for reservoir
    blocks, the well-operator region for well blocks) instead of defaulting to
    region 0, and raises when that region has no registered container.

    :param model: model owning ``physics`` and ``reservoir``
    :param block_idx: global block index
    :returns: the property container of the block's region
    """
    region = block_region(model, block_idx)
    return property_container_of_region(model.physics, region, block_idx)


def well_cell_property_container(physics, block_idx: int = None):
    """Property container the engine uses for WELL cells.

    :param physics: physics object owning ``property_containers``
    :param block_idx: well block index, used only in error messages
    :returns: the property container of the well-operator region
    """
    region = well_cell_region(physics, block_idx)
    return property_container_of_region(physics, region, block_idx)


@dataclass(frozen=True)
class LinearDFMWellIPRConnection:
    well_name: str
    perforation_index: int
    pi: float
    pi_type: PI_Type
    ipr_pressure_offset: float = 0.0
    ipr_intercept: float = 0.0


class LinearDFMWellIPRHook:
    """
    Apply a linear total-rate IPR for DFM well perforations.

    The used linear IPR is
        q_total = A + B * (p_well - p_reservoir - dp_offset)

    where B is the productivity index ``pi`` (per bar of drawdown), A is
    ``ipr_intercept`` and q_total is a total RATE interpreted according to
    pi_type:
      - PI_Type.MASS: q_total in kg/day (pi in kg/day/bar)
      - PI_Type.MOLAR: q_total in kmol/day (pi in kmol/day/bar)
      - PI_Type.VOLUMETRIC: q_total in m3/day at the UPSTREAM in-situ
        conditions (pi in m3/day/bar)

    The total rate is converted to component molar rates using the upstream
    state and added directly to the engine RHS/Jacobian.

    Fluid properties (Mw, molar enthalpy, molar density) are taken from the
    property container of the region of the block being evaluated. The well
    side and the reservoir side are resolved INDEPENDENTLY, once at bind time
    (:meth:`_get_resolved_connections`): the perforated reservoir cell may
    belong to a different operator region than the well cell, and the upstream
    side of the connection selects which of the two containers is used. See
    the region helpers at the top of this module; a block whose region has no
    registered property container raises instead of falling back to region 0.

    :param allow_nonzero_well_indexD: by default a perforation with a non-zero
        thermal well index (WID) is rejected because the hook cannot verify
        the intent. Set True to allow it: WID drives the conductive/diffusive
        heat term the engine assembles independently of the advective IPR flux
        managed here, so a non-zero WID does not double-count the IPR flux.
    """

    def __init__(
        self,
        model,
        connections: list[LinearDFMWellIPRConnection],
        pressure_eps_bar: float = 1e-7,
        composition_eps: float = 1e-8,
        thermal_eps: float = 1e-6,
        allow_nonzero_well_indexD: bool = False,
    ):
        self.model = model
        self.connections = tuple(connections)
        self.pressure_eps_bar = float(pressure_eps_bar)
        self.composition_eps = float(composition_eps)
        self.thermal_eps = float(thermal_eps)
        self.allow_nonzero_well_indexD = bool(allow_nonzero_well_indexD)
        self._resolved_connections = None

    def apply(self, dt: float, t: float = None):
        del t
        if not self.connections:
            return

        n_vars = self.model.physics.n_vars
        # the view is rebuilt per call (cheap: numpy views only) so a re-init
        # of the engine (which reallocates jac_vals) cannot leave it stale
        jac = BlockCSRView(self.model.physics.engine, n_vars)
        resolved_connections = self._get_resolved_connections(jac)
        rhs = np.asarray(self.model.physics.engine.RHS)
        X = np.asarray(self.model.physics.engine.X)

        for resolved in resolved_connections:
            wb_idx = resolved["well_block_idx"]
            rb_idx = resolved["res_block_idx"]

            well_state = X[wb_idx * n_vars : (wb_idx + 1) * n_vars].copy()
            res_state = X[rb_idx * n_vars : (rb_idx + 1) * n_vars].copy()

            base_flux = self._evaluate_connection_flux(
                resolved=resolved,
                well_state=well_state,
                res_state=res_state,
            )

            well_base = wb_idx * n_vars
            res_base = rb_idx * n_vars
            rhs[well_base : well_base + n_vars] += base_flux["well_residual"] * dt
            rhs[res_base : res_base + n_vars] += base_flux["res_residual"] * dt

            jac_well = self._differentiate_flux(
                resolved=resolved,
                base_flux=base_flux,
                well_state=well_state,
                res_state=res_state,
                target="well",
            )
            jac_res = self._differentiate_flux(
                resolved=resolved,
                base_flux=base_flux,
                well_state=well_state,
                res_state=res_state,
                target="res",
            )

            jac.add_block(resolved["diag_well"], jac_well[:n_vars, :] * dt)
            jac.add_block(resolved["off_well_res"], jac_res[:n_vars, :] * dt)
            jac.add_block(resolved["off_res_well"], jac_well[n_vars:, :] * dt)
            jac.add_block(resolved["diag_res"], jac_res[n_vars:, :] * dt)

    def _get_resolved_connections(self, jac: BlockCSRView) -> tuple[dict, ...]:
        if self._resolved_connections is not None:
            return self._resolved_connections

        resolved = []
        seen = set()
        for connection in self.connections:
            key = (connection.well_name, connection.perforation_index)
            if key in seen:
                raise ValueError(
                    f"Duplicate LinearDFMWellIPRConnection for well "
                    f"{connection.well_name!r}, perforation "
                    f"{connection.perforation_index}: the IPR flux would be "
                    "applied twice."
                )
            seen.add(key)

            well = self.model.reservoir.get_well(connection.well_name)
            if well is None:
                available = ", ".join(w.name for w in self.model.reservoir.wells)
                raise KeyError(
                    f"Well {connection.well_name!r} not found; available wells: "
                    f"[{available}]."
                )
            if well.ms_type != ms_well.MS_Type.DFM:
                raise NotImplementedError(
                    "LinearDFMWellIPRHook currently supports only DFM wells."
                )
            if not 0 <= connection.perforation_index < len(well.perforations):
                raise IndexError(
                    f"Perforation index {connection.perforation_index} is out of bounds for well {connection.well_name!r}."
                )
            if connection.pi < 0.0:
                raise ValueError(
                    f"Connection to well {connection.well_name!r}, perforation "
                    f"{connection.perforation_index} has a negative productivity "
                    f"index (pi={connection.pi}). A negative PI is an "
                    "unconditionally unstable anti-physical feedback (flux grows "
                    "with the pressure difference it opposes)."
                )

            perf_segment_local, res_block_idx, well_index, well_indexD = (
                well.perforations[connection.perforation_index]
            )
            if well_index != 0.0:
                raise ValueError(
                    f"Perforation {connection.perforation_index} of well {connection.well_name!r} "
                    f"has a non-zero well index (WI={well_index}). LinearDFMWellIPRHook manages "
                    "the well-reservoir flux directly; a non-zero well_index would cause double-counting. "
                    "Set well_index=0.0 when calling add_perforation()."
                )
            if well_indexD != 0.0 and not self.allow_nonzero_well_indexD:
                raise ValueError(
                    f"Perforation {connection.perforation_index} of well {connection.well_name!r} "
                    f"has a non-zero thermal well index (WID={well_indexD}). WID drives the "
                    "conductive/diffusive heat term the engine assembles independently, so it "
                    "cannot double-count the advective IPR flux; pass "
                    "allow_nonzero_well_indexD=True to LinearDFMWellIPRHook if this is intended, "
                    "or set well_indexD=0.0 when calling add_perforation()."
                )
            well_block_idx = well.well_body_idx + perf_segment_local
            # Resolve the fluid of EACH SIDE from the region of ITS OWN block
            # (see the region helpers at the top of this module): the
            # perforated reservoir cell may live in a different operator region
            # than the well cell, and using region 0 for both would silently
            # evaluate the wrong fluid in a multi-region model. Resolved once
            # here, not per Newton iteration.
            resolved.append(
                {
                    "spec": connection,
                    "well_block_idx": well_block_idx,
                    "res_block_idx": res_block_idx,
                    "well_region": block_region(self.model, well_block_idx),
                    "res_region": block_region(self.model, res_block_idx),
                    "well_pc": property_container_for_block(self.model, well_block_idx),
                    "res_pc": property_container_for_block(self.model, res_block_idx),
                    "diag_well": jac.diag_pos(well_block_idx),
                    "diag_res": jac.diag_pos(res_block_idx),
                    "off_well_res": jac.block_pos(well_block_idx, res_block_idx),
                    "off_res_well": jac.block_pos(res_block_idx, well_block_idx),
                }
            )

        self._resolved_connections = tuple(resolved)
        return self._resolved_connections

    def _differentiate_flux(
        self,
        resolved: dict,
        base_flux: dict,
        well_state: np.ndarray,
        res_state: np.ndarray,
        target: str,
    ) -> np.ndarray:
        n_vars = self.model.physics.n_vars
        jac = np.zeros((2 * n_vars, n_vars), dtype=float)
        for var_idx in range(n_vars):
            if target == "well":
                perturbed_state, delta = self._perturb_state(well_state, var_idx)
                if delta == 0.0:
                    continue
                flux = self._evaluate_connection_flux(
                    resolved=resolved,
                    well_state=perturbed_state,
                    res_state=res_state,
                )
            else:
                perturbed_state, delta = self._perturb_state(res_state, var_idx)
                if delta == 0.0:
                    continue
                flux = self._evaluate_connection_flux(
                    resolved=resolved,
                    well_state=well_state,
                    res_state=perturbed_state,
                )

            flux_vector = np.concatenate((flux["well_residual"], flux["res_residual"]))
            base_vector = np.concatenate(
                (base_flux["well_residual"], base_flux["res_residual"])
            )
            jac[:, var_idx] = (flux_vector - base_vector) / delta
        return jac

    def _perturb_state(
        self, state: np.ndarray, var_idx: int
    ) -> tuple[np.ndarray, float]:
        perturbed = state.copy()
        n_vars = self.model.physics.n_vars
        nc = self.model.physics.nc

        if var_idx == 0:
            delta = max(abs(state[var_idx]) * 1e-7, self.pressure_eps_bar)
            perturbed[var_idx] += delta
            return perturbed, delta

        if 1 <= var_idx < nc:
            eps_z = getattr(self.model.physics, "sim_eps", 1e-12)
            sum_other = float(np.sum(state[1:nc]) - state[var_idx])
            lower = eps_z
            upper = 1.0 - eps_z - sum_other
            trial = max(abs(state[var_idx]) * 1e-7, self.composition_eps)

            up_room = upper - state[var_idx]
            down_room = state[var_idx] - lower
            # perturb into the direction with more room to the composition bound
            if up_room >= down_room:
                delta = min(trial, max(up_room, 0.0))
            else:
                delta = -min(trial, max(down_room, 0.0))

            # essentially no room on either side: keep a signed floor so the
            # finite difference in _differentiate_flux does not divide flash
            # noise by a near-zero delta
            min_delta = trial * 1e-3
            if abs(delta) < min_delta:
                delta = min_delta if up_room >= down_room else -min_delta

            perturbed[var_idx] += delta
            return perturbed, delta

        if self.model.physics.thermal and var_idx == n_vars - 1:
            delta = max(abs(state[var_idx]) * 1e-7, self.thermal_eps)
            perturbed[var_idx] += delta
            return perturbed, delta

        return perturbed, 0.0

    def _evaluate_connection_flux(
        self,
        resolved: dict,
        well_state: np.ndarray,
        res_state: np.ndarray,
    ) -> dict:
        spec = resolved["spec"]
        n_vars = self.model.physics.n_vars
        nc = self.model.physics.nc
        thermal = bool(self.model.physics.thermal)
        energy_eq_idx = nc

        total_rate = spec.ipr_intercept + spec.pi * (
            well_state[0] - res_state[0] - spec.ipr_pressure_offset
        )

        # The upstream side sets the fluid: take its property container, which
        # was resolved from the region of that very block at bind time.
        if total_rate >= 0.0:
            upstream_state = well_state
            upstream_block_idx = resolved["well_block_idx"]
            upstream_pc = resolved["well_pc"]
        else:
            upstream_state = res_state
            upstream_block_idx = resolved["res_block_idx"]
            upstream_pc = resolved["res_pc"]

        overall_composition = self._state_overall_composition(upstream_state)
        mw_avg = self._mean_molecular_weight(upstream_pc, overall_composition)
        molar_rate = self._convert_total_rate_to_molar_rate(
            total_rate=total_rate,
            pi_type=spec.pi_type,
            upstream_state=upstream_state,
            overall_composition=overall_composition,
            mw_avg=mw_avg,
            property_container=upstream_pc,
        )

        component_rate = molar_rate * overall_composition[:nc]

        well_residual = np.zeros(n_vars, dtype=float)
        res_residual = np.zeros(n_vars, dtype=float)
        well_residual[:nc] += component_rate
        res_residual[:nc] -= component_rate

        if thermal:
            molar_enthalpy = self._state_molar_enthalpy(upstream_pc, upstream_state)
            specific_potential_energy = self.model.reservoir.mesh.cell_spe[
                upstream_block_idx
            ]
            energy_rate = molar_rate * (
                molar_enthalpy + specific_potential_energy * mw_avg
            )
            well_residual[energy_eq_idx] += energy_rate
            res_residual[energy_eq_idx] -= energy_rate

        return {
            "well_residual": well_residual,
            "res_residual": res_residual,
        }

    def _convert_total_rate_to_molar_rate(
        self,
        total_rate: float,
        pi_type,
        upstream_state: np.ndarray,
        overall_composition: np.ndarray,
        mw_avg: float,
        property_container,
    ) -> float:
        pi_type = self._normalize_pi_type(pi_type)
        if pi_type == PI_Type.MOLAR:
            return total_rate
        if pi_type == PI_Type.MASS:
            return total_rate / mw_avg
        if pi_type == PI_Type.VOLUMETRIC:
            return total_rate * self._total_molar_density(
                property_container, upstream_state
            )
        raise NotImplementedError(f"Unsupported PI type: {pi_type!r}")

    @staticmethod
    def _normalize_pi_type(pi_type):
        if isinstance(pi_type, PI_Type):
            return pi_type
        raise ValueError(f"Unsupported PI type: {pi_type!r}")

    def _state_overall_composition(self, state: np.ndarray) -> np.ndarray:
        nc = self.model.physics.nc
        if nc == 1:
            return np.array([1.0], dtype=float)

        zc = np.empty(nc, dtype=float)
        zc[:-1] = state[1:nc]
        zc[-1] = 1.0 - np.sum(zc[:-1])

        eps_z = getattr(self.model.physics, "sim_eps", 1e-12)
        zc = np.maximum(zc, eps_z)
        zc /= np.sum(zc)
        return zc

    def _mean_molecular_weight(
        self, property_container, overall_composition: np.ndarray
    ) -> float:
        nc = self.model.physics.nc
        return float(
            np.sum(np.asarray(property_container.Mw[:nc]) * overall_composition[:nc])
        )

    def _state_molar_enthalpy(self, property_container, state: np.ndarray) -> float:
        if not self.model.physics.thermal:
            return 0.0
        if self.model.physics.state_spec == self.model.physics.StateSpecification.PH:
            return float(state[-1])
        state_vector = value_vector(state.tolist())
        return float(property_container.compute_total_enthalpy(state_vector))

    def _total_molar_density(self, property_container, state: np.ndarray) -> float:
        pc = property_container
        state_vector = value_vector(state.tolist())
        pc.evaluate(state_vector)
        return float(np.sum(pc.sat[pc.ph] * pc.dens_m[pc.ph]))
