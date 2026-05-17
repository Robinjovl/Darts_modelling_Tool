import numpy as np

from darts.interpolators import operator_set_evaluator_iface, value_vector
from darts.physics.base.property_base import PropertyBase


class OperatorsBase(operator_set_evaluator_iface):
    n_ops: int

    def __init__(
        self,
        property_container: PropertyBase,
        thermal: bool,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        Constructor of OperatorsBase base class

        :param property_container: Property container of type PropertyBase
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition interval along OBL composition axes to obtain consistent points for extrapolation
                    (must be equal along all composition axes in current setup)
        """
        super().__init__()

        self.property = property_container

        self.thermal = thermal

        self.nc = property_container.nc
        self.ne = self.nc + self.thermal
        self.nph = property_container.nph
        self.eps_z = (
            property_container.eps_z if hasattr(property_container, 'eps_z') else 1e-13
        )

        self.extrapolation_flag = extrapolation_flag
        self.dz = dz
        assert self.nc <= 2 or not extrapolation_flag or dz is not None, (
            "Please provide dz for extrapolation"
        )

    def apply_extrapolation(self, state, values):
        """
        Method that determines whether or not extrapolation should be applied to current state (z[-1] < 0).
        If so, it will call extrapolate() and return True, such that evaluate() skips further evaluation of operators

        :param state: Vector with state [P, z, (T/H)]
        :param values: Vector with operator values
        :return: Whether or not extrapolation has been applied to this state
        """
        # Find composition, if last composition is negative, apply extrapolation
        zc = np.append(state[1 : self.nc], 1 - np.sum(state[1 : self.nc]))

        if len(zc) > 2 and zc[-1] < 0.99 * self.eps_z and self.extrapolation_flag:
            # TODO: Fix second condition, this is problematic for small eps_z values (~1e-14)
            self.extrapolate(state, values)
            return True
        else:
            return False

    def extrapolate(self, state, values):
        """
        When composition lies outside the simplex (∑ z_i ≠ 1 or some z_i < 0), perform exact hyperplane extrapolation:
        Fit each operator value via val = a·z + c through exactly d+1 valid reference points ,
        then evaluate at the out‑of‑bounds composition. Pressure (and temperature) remain constant.
        State layout: [ p, z₁, …, z_d, (T) ]
        """
        # Unpack state
        vec = state.to_numpy()
        if self.thermal:
            p, T = vec[0], vec[-1]
            z = vec[1:-1].copy()
        else:
            p = vec[0]
            z = vec[1:].copy()

        zero_comps = [i for i in range(self.nc - 1) if z[i] <= 2 * self.eps_z]
        nonzero_comps = [1 if z[i] > 2 * self.eps_z else 0 for i in range(self.nc - 1)]
        nonzero_comp_idxs = [
            i for i, is_nonzero in enumerate(nonzero_comps) if is_nonzero
        ]
        dims = np.sum(nonzero_comps)

        # Build supporting points from the downward hypercube (excluding incoming).
        # Filter nodes by the shared plane constraint, then take the furthest node and the
        # nc_nonzero-1 closest nodes from the filtered set.
        candidates = []
        for mask in range(1, 1 << dims):  # 1 << n_axes multiplies 1 by 2^d
            zp = z.copy()
            for i, axis in enumerate(nonzero_comp_idxs):
                # binary operator & compares binary notation of 'mask' and 2^axis
                if mask & (1 << i):
                    zp[axis] -= self.dz
            dist2 = np.sum((zp - z) ** 2)
            last_z = 1.0 - np.sum(zp)
            candidates.append((dist2, mask, zp, last_z >= 0.0))

        filtered = [c for c in candidates if c[3]]
        if not filtered:
            filtered = candidates
        filtered = sorted(filtered, key=lambda c: c[0])

        furthest = filtered[-1]
        closest = [c for c in filtered if c[1] != furthest[1]][:dims]
        selected = [furthest] + closest

        n_supporting_points = dims + 1
        if len(selected) < n_supporting_points:
            remaining = [c for c in candidates if c[1] not in {s[1] for s in selected}]
            remaining = sorted(remaining, key=lambda c: c[0])
            selected.extend(remaining[: n_supporting_points - len(selected)])

        supporting_points = [c[2] for c in selected]

        # Gather valid reference points
        zps_list = []
        vals_list = []
        for zp in supporting_points:
            if self.thermal:
                ref_state = value_vector(np.concatenate(([p], zp, [T])))
            else:
                ref_state = value_vector(np.concatenate(([p], zp)))
            ref_vals = value_vector(np.zeros(self.n_ops))
            self.evaluate(ref_state, ref_vals)
            zps_list.append(zp)
            vals_list.append(ref_vals.to_numpy())

        # Use the first d+1 valid points to define hyperplane implicitly via val = a·z + c
        zps = np.stack(zps_list[: dims + 1])  # shape (dims+1, dims)
        vals = np.stack(vals_list[: dims + 1])  # shape (dims+1, n_ops)

        # Build and solve B · X = vals, where B = [zps | 1]
        B = np.hstack((zps, np.ones((dims + 1, 1))))  # shape (dims+1, dims+1)
        B = np.delete(B, zero_comps, axis=1)
        X = np.linalg.solve(B, vals)  # shape (dims+1, n_ops)

        # Separate coefficients
        a = X[:-1, :]  # shape (dims, n_ops)
        c = X[-1, :]  # shape (n_ops,)

        # Extrapolate values
        z_nonzero = z[nonzero_comp_idxs]
        ext = a.T.dot(z_nonzero) + c

        # Write back into values array
        out = np.array(values, copy=False)
        out[: self.n_ops] = ext
        return out


class WellCtrlOperators(OperatorsBase):
    """
    Set of operators for well controls of EPM and DFM wells.

    Operator layout:
    NP EPM molar-rate, NP EPM mass-rate, NP EPM volumetric-rate,
    NP EPM advective-heat-rate, pressure, temperature,
    NP DFM molar-rate, NP DFM mass-rate, NP DFM volumetric-rate,
    NP DFM advective-heat-rate.
    """

    def __init__(
        self,
        property_container: PropertyBase,
        thermal: bool,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        Constructor of WellCtrlOperators class

        :param property_container: Property container of type PropertyBase
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition interval along OBL composition axes to obtain consistent points for extrapolation
                    (must be equal along all composition axes in current setup)
        """
        super().__init__(
            property_container, thermal, extrapolation_flag=extrapolation_flag, dz=dz
        )

        self.n_rate_ctrl_types = 4  # molar, mass, volumetric, and advective heat rates
        self.n_state_ctrl_ops = 2  # pressure and temperature
        self.epm_rate_ctrl_ops_offset = 0
        self.state_ctrl_ops_offset = self.n_rate_ctrl_types * self.nph
        self.dfm_rate_ctrl_ops_offset = (
            self.state_ctrl_ops_offset + self.n_state_ctrl_ops
        )
        self.n_ops = self.n_state_ctrl_ops + 2 * self.n_rate_ctrl_types * self.nph

    def calc_rate_ctrl_ops(self, values, offset, rate_factor):
        # Molar rate ctrl operator
        idx = offset
        values[idx + self.property.ph] = (
            self.property.dens_m[self.property.ph] * rate_factor
        )

        # Mass rate ctrl operator
        idx += self.nph
        values[idx + self.property.ph] = (
            self.property.dens[self.property.ph] * rate_factor
        )

        # Volumetric rate ctrl operator
        idx += self.nph
        values[idx + self.property.ph] = rate_factor

        # Advective heat rate ctrl operator
        idx += self.nph
        if self.thermal:
            values[idx + self.property.ph] = (
                self.property.enthalpy[self.property.ph]
                * self.property.dens_m[self.property.ph]
                * rate_factor
            )

    def evaluate(self, state, values):
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state, values):
            return 0

        state_np = state.to_numpy()
        values_np = values.to_numpy()
        values_np[:] = 0

        self.property.evaluate(state_np)
        if self.thermal:
            self.property.evaluate_thermal(state_np)

        epm_rate_factor = (
            self.property.kr[self.property.ph] / self.property.mu[self.property.ph]
        )
        self._fill_rate_ctrl_ops(
            values_np, self.epm_rate_ctrl_ops_offset, epm_rate_factor
        )

        # Store pressure (P) and temperature (T) of the current state for a generic state specification.
        # This is needed when pressure or temperature is not part of the state variables
        # (e.g., volume instead of pressure, or enthalpy instead of temperature).
        idx = self.state_ctrl_ops_offset
        values_np[idx + 0] = state[0]
        values_np[idx + 1] = self.property.temperature

        dfm_rate_factor = self.property.sat[self.property.ph]
        self._fill_rate_ctrl_ops(
            values_np, self.dfm_rate_ctrl_ops_offset, dfm_rate_factor
        )

        return 0


class ThermalVarOperator(OperatorsBase):
    """
    ThermalVarOperator gives the thermal variable for generic state specification
    """

    def __init__(
        self,
        property_container: PropertyBase,
        thermal: bool,
        is_pt: bool = True,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        Constructor of ThermalVarOperator class

        :param property_container: Property container of type PropertyBase
        :param thermal: Switch to indicate if energy conservation equation is there
        :param is_pt: Switch to indicate if state specification is P, PT, or PH
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition interval along OBL composition axes to obtain consistent points for extrapolation
                    (must be equal along all composition axes in current setup)
        """
        super().__init__(property_container, thermal, extrapolation_flag, dz)

        self.n_ops = 1
        self.is_pt = is_pt

    def evaluate(self, state_pt, values):
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state_pt, values):
            return 0

        values_np = values.to_numpy()
        values_np[:] = 0

        if not self.thermal:
            values_np[0] = self.property.temperature
        elif self.is_pt:
            values_np[0] = state_pt[-1]
        else:
            values_np[0] = self.property.compute_total_enthalpy(state_pt=state_pt)

        return 0


class PropertyOperators(OperatorsBase):
    """
    This class contains a set of operators for evaluation of output properties.
    A set of interpolators is created in the :class:`Physics` object to rapidly obtain properties after simulation.
    """

    def __init__(
        self,
        property_container: PropertyBase,
        thermal: bool,
        props: dict = None,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        This is the constructor for PropertyOperator.
        The properties to be obtained from the PropertyOperators are passed to PropertyContainer as a dictionary.

        :param property_container: PropertyBase object to evaluate properties at given state
        :param thermal: Bool for thermal
        :param props: Optional dictionary of properties, default is taken from PropertyContainer
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition interval along OBL composition axes to obtain consistent points for extrapolation
                    (must be equal along all composition axes in current setup)
        """
        super().__init__(property_container, thermal, extrapolation_flag, dz)

        self.props = property_container.output_props if props is None else props
        self.props_name = [key for key in self.props.keys()]
        self.props_idx = {prop: j for j, prop in enumerate(self.props_name)}
        self.n_ops = len(self.props_name)

    def evaluate(self, state: value_vector, values: value_vector):
        """
        This function evaluates the properties at given `state` (P,z) or (P,T,z) from the :class:`PropertyContainer` object.
        The user-specified properties are stored in the `values` object.

        :param state: Vector of state variables [pres, comp_0, ..., comp_N-1, (temp)]
        :type state: darts.interpolators.value_vector
        :param values: Vector for storage of operator values
        :type values: darts.interpolators.value_vector
        """
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state, values):
            return 0

        state_np = state.to_numpy()
        values_np = values.to_numpy()
        _ = self.property.evaluate(state_np)
        if self.thermal:
            _ = self.property.evaluate_thermal(state_np)

        for i, prop in enumerate(self.props_name):
            output = self.props[prop]()
            values_np[i] = output if not np.isnan(output) else 0.0

        return 0
