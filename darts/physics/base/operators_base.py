import numpy as np

from darts.engines import operator_set_evaluator_iface, value_vector
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
        d = np.sum(nonzero_comps)
        last_z = 1.0 - np.sum(z)

        # Build candidate points by subtracting dz along each axis and uniformly
        # Axes where z[i] = zero degenerate for extrapolation (e.g., a 3D extrapolation reduces to 2D)
        # There exist two cases of supporting points: one where points are on the same side, one where they are on opposite side
        # (see hydrate paper, https://doi.org/10.1016/j.ecmx.2026.101616)
        # TODO: check if this setup is general for any number of components. It certainly works for NC = 3 and NC = 4
        supporting_points = []
        # z[-1] = -dz (points on same side of hypercube)
        if last_z >= -1.1 * self.dz:
            for i in range(self.nc - 1):
                if nonzero_comps[i]:
                    zp = z.copy()
                    # subtract dz from z[i] to obtain the ith supporting point
                    zp[i] -= self.dz
                    supporting_points.append(zp)
        # z[-1] < -dz (opposite points in hypercube)
        else:
            for i in range(self.nc - 1):
                if nonzero_comps[i]:  # only for nonzero compositions,
                    for j in range(i + 1, self.nc - 1):
                        zp = z.copy()
                        # subtract dz from z[i] and z[j] to obtain the ith supporting point
                        zp[i] -= self.dz
                        zp[j] -= self.dz
                        supporting_points.append(zp)
        # Finally, append the point directly opposite to the extrapolated point
        supporting_points.append(
            np.array(
                [
                    z[i] - self.dz if nonzero_comps[i] else z[i]
                    for i in range(self.nc - 1)
                ]
            )
        )

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
        zps = np.stack(zps_list[: d + 1])  # shape (d+1, d)
        vals = np.stack(vals_list[: d + 1])  # shape (d+1, n_ops)

        # Build and solve B · X = vals, where B = [zps | 1]
        B = np.hstack((zps, np.ones((d + 1, 1))))  # shape (d+1, d+1)
        B = np.delete(B, zero_comps, axis=1)
        X = np.linalg.solve(B, vals)  # shape (d+1, n_ops)

        # Separate coefficients
        a = X[:-1, :]  # shape (d, n_ops)
        c = X[-1, :]  # shape (n_ops,)

        # Extrapolate values
        z_nonzero = z[z > 2 * self.eps_z]
        ext = a.T.dot(z_nonzero) + c

        # Write back into values array
        out = np.array(values, copy=False)
        out[: self.n_ops] = ext
        return out


class WellControlOperators(OperatorsBase):
    """
    Set of operators for well controls. It contains the pressure, composition and temperature of the wellhead,
    plus a set of rate-control operators for different types of rates: molar-, mass-, volumetric- or advective
    heat rate controls
    """

    def __init__(
        self,
        property_container: PropertyBase,
        thermal: bool,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        Constructor of WellControlOperators class

        :param property_container: Property container of type PropertyBase
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition interval along OBL composition axes to obtain consistent points for extrapolation
                    (must be equal along all composition axes in current setup)
        """
        super().__init__(
            property_container, thermal, extrapolation_flag=extrapolation_flag, dz=dz
        )

        self.n_ops = 2 + self.nph * 4

    def evaluate(self, state, values):
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state, values):
            return 0

        state_np = state.to_numpy()
        values_np = values.to_numpy()
        values_np[:] = 0

        self.property.evaluate(state_np)

        # Store rate controls
        mobility = (
            self.property.kr[self.property.ph] / self.property.mu[self.property.ph]
        )

        # Molar rate
        idx = 0
        values_np[idx + self.property.ph] = (
            self.property.dens_m[self.property.ph] * mobility
        )

        # Mass rate
        idx += self.nph
        values_np[idx + self.property.ph] = (
            self.property.dens[self.property.ph] * mobility
        )

        # Volumetric rate
        idx += self.nph
        values_np[idx + self.property.ph] = mobility

        # Advective heat rate
        idx += self.nph
        if self.thermal:
            self.property.evaluate_thermal(state_np)
            values_np[idx + self.property.ph] = (
                self.property.enthalpy[self.property.ph]
                * self.property.dens_m[self.property.ph]
                * mobility
            )

        # Store P, T and composition of current state
        idx += self.nph
        values_np[idx + 0] = state[0]
        values_np[idx + 1] = self.property.temperature

        return 0


class WellInitOperators(OperatorsBase):
    """
    WellInitOperators initialize the well BHP/BHT for generic state specification
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
        Constructor of WellInitOperators class

        :param property_container: Property container of type PropertyBase
        :param thermal: Switch to indicate if energy conservation equation is there
        :param is_pt: Switch to indicate if state specification is P/PT or PH
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
        :type state: darts.engines.value_vector
        :param values: Vector for storage of operator values
        :type values: darts.engines.value_vector
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
