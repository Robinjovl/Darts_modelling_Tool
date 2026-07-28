from dataclasses import dataclass
from enum import Enum

import numpy as np

from darts.engines import ms_well, value_vector
from darts.physics.base.physics_base import PhysicsBase


class PI_Type(Enum):
    MOLAR = "molar"
    MASS = "mass"
    VOLUMETRIC = "volumetric"


@dataclass(frozen=True)
class LinearDFMWellIPRConnection:
    well_name: str
    perforation_index: int
    pi: float
    pi_type: PI_Type
    ipr_pressure_offset: float = 0.0
    ipr_intercept: float = 0.0


class LinearDFMWellIPR:
    """
    Apply a linear total-rate IPR for DFM well perforations.

    The used linear IPR is
        q_total = A + B * (p_well - p_reservoir - dp_offset)

    where q_total is interpreted according to pi_type:
      - PI_Type.MASS: kg/day/bar
      - PI_Type.MOLAR: kmol/day/bar
      - PI_Type.VOLUMETRIC: m3/day/bar

    The total rate is converted to component molar rates using the upstream
    state and added directly to the engine RHS/Jacobian.
    """

    def __init__(
        self,
        model,
        connections: list[LinearDFMWellIPRConnection],
        pressure_eps_bar: float = 1e-7,
        composition_eps: float = 1e-8,
        thermal_eps: float = 1e-6,
    ):
        self.model = model
        self.connections = tuple(connections)
        self.pressure_eps_bar = float(pressure_eps_bar)
        self.composition_eps = float(composition_eps)
        self.thermal_eps = float(thermal_eps)
        self._resolved_connections = None

    def apply(self, dt: float, t: float = None):
        del t
        if not self.connections:
            return

        resolved_connections = self._get_resolved_connections()
        rhs = np.asarray(self.model.physics.engine.RHS)
        jac_vals = np.asarray(self.model.physics.engine.jac_vals)
        X = np.asarray(self.model.physics.engine.X)

        n_vars = self.model.physics.n_vars
        n_jac_block_size = n_vars * n_vars

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

            self._accumulate_dense_block(
                jac_vals,
                resolved["diag_well"],
                jac_well[:n_vars, :] * dt,
                n_jac_block_size,
            )
            self._accumulate_dense_block(
                jac_vals,
                resolved["off_well_res"],
                jac_res[:n_vars, :] * dt,
                n_jac_block_size,
            )
            self._accumulate_dense_block(
                jac_vals,
                resolved["off_res_well"],
                jac_well[n_vars:, :] * dt,
                n_jac_block_size,
            )
            self._accumulate_dense_block(
                jac_vals,
                resolved["diag_res"],
                jac_res[n_vars:, :] * dt,
                n_jac_block_size,
            )

    def _get_resolved_connections(self) -> tuple[dict, ...]:
        if self._resolved_connections is not None:
            return self._resolved_connections

        jac_diags = np.asarray(self.model.physics.engine.jac_diags)
        resolved = []
        for connection in self.connections:
            well = self.model.reservoir.get_well(connection.well_name)
            if well.ms_type != ms_well.MS_Type.DFM:
                raise NotImplementedError(
                    "LinearDFMWellIPR currently supports only DFM wells."
                )
            if not 0 <= connection.perforation_index < len(well.perforations):
                raise IndexError(
                    f"Perforation index {connection.perforation_index} is out of bounds for well {connection.well_name!r}."
                )

            perf_segment_local, res_block_idx, _, _ = well.perforations[
                connection.perforation_index
            ]
            well_block_idx = well.well_body_idx + perf_segment_local
            resolved.append(
                {
                    "spec": connection,
                    "well_block_idx": well_block_idx,
                    "res_block_idx": res_block_idx,
                    "diag_well": int(jac_diags[well_block_idx]),
                    "diag_res": int(jac_diags[res_block_idx]),
                    "off_well_res": self._find_csr_block_position(
                        well_block_idx, res_block_idx
                    ),
                    "off_res_well": self._find_csr_block_position(
                        res_block_idx, well_block_idx
                    ),
                }
            )

        self._resolved_connections = tuple(resolved)
        return self._resolved_connections

    def _find_csr_block_position(self, row_block: int, col_block: int) -> int:
        jac_rows = np.asarray(self.model.physics.engine.jac_rows)
        jac_cols = np.asarray(self.model.physics.engine.jac_cols)
        row_start = jac_rows[row_block]
        row_end = jac_rows[row_block + 1]
        off_pos = np.where(jac_cols[row_start:row_end] == col_block)[0]
        if len(off_pos) == 0:
            raise RuntimeError(
                f"CSR block ({row_block}, {col_block}) was not found in the Jacobian pattern."
            )
        return int(row_start + off_pos[0])

    @staticmethod
    def _accumulate_dense_block(
        jac_vals: np.ndarray,
        block_pos: int,
        dense_block: np.ndarray,
        n_jac_block_size: int,
    ):
        start = block_pos * n_jac_block_size
        jac_vals[start : start + n_jac_block_size] += dense_block.reshape(-1)

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
            if up_room >= min(trial, max(up_room, 0.0)):
                delta = min(trial, max(up_room, 0.0))
            else:
                delta = -min(trial, max(down_room, 0.0))

            if abs(delta) <= 0.0:
                return perturbed, 0.0

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

        if total_rate >= 0.0:
            upstream_state = well_state
            upstream_block_idx = resolved["well_block_idx"]
        else:
            upstream_state = res_state
            upstream_block_idx = resolved["res_block_idx"]

        overall_composition = self._state_overall_composition(upstream_state)
        mw_avg = self._mean_molecular_weight(overall_composition)
        molar_rate = self._convert_total_rate_to_molar_rate(
            total_rate=total_rate,
            pi_type=spec.pi_type,
            upstream_state=upstream_state,
            overall_composition=overall_composition,
            mw_avg=mw_avg,
        )

        component_rate = molar_rate * overall_composition[:nc]

        well_residual = np.zeros(n_vars, dtype=float)
        res_residual = np.zeros(n_vars, dtype=float)
        well_residual[:nc] += component_rate
        res_residual[:nc] -= component_rate

        if thermal:
            molar_enthalpy = self._state_molar_enthalpy(upstream_state)
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
    ) -> float:
        pi_type = self._normalize_pi_type(pi_type)
        if pi_type == PI_Type.MOLAR:
            return total_rate
        if pi_type == PI_Type.MASS:
            return total_rate / mw_avg
        if pi_type == PI_Type.VOLUMETRIC:
            return total_rate * self._total_molar_density(upstream_state)
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

    def _mean_molecular_weight(self, overall_composition: np.ndarray) -> float:
        return float(
            np.sum(
                np.asarray(
                    self.model.physics.property_containers[0].Mw[
                        : self.model.physics.nc
                    ]
                )
                * overall_composition[: self.model.physics.nc]
            )
        )

    def _state_molar_enthalpy(self, state: np.ndarray) -> float:
        if not self.model.physics.thermal:
            return 0.0
        if self.model.physics.state_spec == PhysicsBase.StateSpecification.PH:
            return float(state[-1])
        state_vector = value_vector(state.tolist())
        return float(
            self.model.physics.property_containers[0].compute_total_enthalpy(
                state_vector
            )
        )

    def _total_molar_density(self, state: np.ndarray) -> float:
        pc = self.model.physics.property_containers[0]
        state_vector = value_vector(state.tolist())
        pc.evaluate(state_vector)
        return float(np.sum(pc.sat[pc.ph] * pc.dens_m[pc.ph]))
