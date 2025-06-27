import numpy as np
from darts.engines import operator_set_evaluator_iface, value_vector
from darts.physics.base.property_base import PropertyBase


class OperatorsBase(operator_set_evaluator_iface):
    n_ops: int

    def __init__(self, property_container: PropertyBase, thermal: bool, extrapolation_flag: bool = True):
        super().__init__()

        self.property = property_container

        self.thermal = thermal

        self.nc = property_container.nc
        self.ne = self.nc + self.thermal
        self.nph = property_container.nph

        self.extrapolation_flag = extrapolation_flag

    def apply_extrapolation(self, state, values):
        # Find composition, if last composition is negative, apply extrapolation
        zc = np.append(state[1:self.nc], 1 - np.sum(state[1:self.nc]))

        if zc[-1] < -1e-12 / 10 and self.extrapolation_flag:
            self.extrapolate(state, values)
            return 1
        else:
            return 0

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

        d = z.size
        dz = abs(1.0 - z.sum())

        # Build candidate points by subtracting dz along each axis and uniformly
        candidates = []
        for i in range(d):
            zp = z.copy()
            zp[i] -= dz
            candidates.append(zp)
        candidates.append(z - dz)

        # Gather valid reference points
        zps_list = []
        vals_list = []
        for zp in candidates:
            if (zp >= 0).all() and zp.sum() <= 1.0:
                if self.thermal:
                    ref_state = value_vector(np.concatenate(([p], zp, [T])))
                else:
                    ref_state = value_vector(np.concatenate(([p], zp)))
                ref_vals = value_vector(np.zeros(self.n_ops))
                self.evaluate(ref_state, ref_vals)
                zps_list.append(zp)
                vals_list.append(ref_vals.to_numpy())

        # Use the first d+1 valid points to define hyperplane implicitly via val = a·z + c
        zps = np.stack(zps_list[:d + 1])  # shape (d+1, d)
        vals = np.stack(vals_list[:d + 1])  # shape (d+1, n_ops)

        # Build and solve B · X = vals, where B = [zps | 1]
        B = np.hstack((zps, np.ones((d + 1, 1))))  # shape (d+1, d+1)
        X = np.linalg.solve(B, vals)  # shape (d+1, n_ops)

        # Separate coefficients
        a = X[:-1, :]  # shape (d, n_ops)
        c = X[-1, :]  # shape (n_ops,)

        # Extrapolate values
        ext = a.T.dot(z) + c

        # Write back into values array
        out = np.array(values, copy=False)
        out[:self.n_ops] = ext
        return out


class WellControlOperators(OperatorsBase):
    """
    Set of operators for well controls. It contains the pressure, composition and temperature of the wellhead,
    plus a set of rate-control operators for different types of rates: molar-, mass-, volumetric- or advective
    heat rate controls
    """
    def __init__(self, property_container: PropertyBase, thermal: bool, extrapolation_flag: bool = True):
        super().__init__(property_container, thermal, extrapolation_flag=extrapolation_flag)

        self.n_ops = 2 + self.nph * 4

    def evaluate(self, state, values):
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state, values):
            return 0

        vec_state_as_np = state.to_numpy()
        vec_values_as_np = values.to_numpy()
        vec_values_as_np[:] = 0

        self.property.evaluate(vec_state_as_np)

        # Store rate controls
        mobility = self.property.kr[self.property.ph] / self.property.mu[self.property.ph]

        # Molar rate
        idx = 0
        vec_values_as_np[idx + self.property.ph] = self.property.dens_m[self.property.ph] * mobility

        # Mass rate
        idx += self.nph
        vec_values_as_np[idx + self.property.ph] = self.property.dens[self.property.ph] * mobility

        # Volumetric rate
        idx += self.nph
        vec_values_as_np[idx + self.property.ph] = mobility

        # Advective heat rate
        idx += self.nph
        if self.thermal:
            self.property.evaluate_thermal(vec_state_as_np)
            vec_values_as_np[idx + self.property.ph] = \
                    self.property.enthalpy[self.property.ph] * self.property.dens_m[self.property.ph] * mobility

        # Store P, T and composition of current state
        idx += self.nph
        vec_values_as_np[idx + 0] = state[0]
        vec_values_as_np[idx + 1] = self.property.temperature

        return 0


class WellInitOperators(OperatorsBase):
    def __init__(self, property_container: PropertyBase, thermal: bool, is_pt: bool = True, extrapolation_flag: bool = True):
        super().__init__(property_container, thermal, extrapolation_flag=extrapolation_flag)

        self.n_ops = 1
        self.is_pt = is_pt

    def evaluate(self, state_pt, values):
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state_pt, values):
            return 0

        vec_values_as_np = values.to_numpy()
        vec_values_as_np[:] = 0

        if self.is_pt:
            vec_values_as_np[0] = state_pt[-1]
        else:
            state_pt = np.array(list(state_pt[:self.nc]) + [state_pt[-1] if self.thermal else self.temperature])
            vec_values_as_np[0] = self.property.compute_total_enthalpy(state_pt=state_pt)

        return 0


class PropertyOperators(OperatorsBase):
    """
    This class contains a set of operators for evaluation of output properties.
    A set of interpolators is created in the :class:`Physics` object to rapidly obtain properties after simulation.
    """
    def __init__(self, property_container: PropertyBase, thermal: bool, props: dict = None, extrapolation_flag: bool = True):
        """
        This is the constructor for PropertyOperator.
        The properties to be obtained from the PropertyOperators are passed to PropertyContainer as a dictionary.

        :param property_container: PropertyBase object to evaluate properties at given state
        :param thermal: Bool for thermal
        :param props: Optional dictionary of properties, default is taken from PropertyContainer
        """
        super().__init__(property_container, thermal, extrapolation_flag=extrapolation_flag)

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

        _ = self.property.evaluate(state)
        if self.thermal:
            _ = self.property.evaluate_thermal(state)

        for i, prop in enumerate(self.props_name):
            output = self.props[prop]()
            values[i] = output if not np.isnan(output) else 0.

        return 0
