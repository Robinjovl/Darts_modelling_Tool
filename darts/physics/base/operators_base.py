import numpy as np
import abc
from darts.engines import operator_set_evaluator_iface, value_vector
from darts.physics.base.property_base import PropertyBase


class OperatorsBase(operator_set_evaluator_iface):
    n_ops: int

    def __init__(self, property_container: PropertyBase, thermal: bool):
        super().__init__()

        self.property = property_container

        self.thermal = thermal

        self.nc = property_container.nc
        self.ne = self.nc + self.thermal
        self.nph = property_container.nph

    @abc.abstractmethod
    def evaluate(self, state, values):
        pass

    def apply_extrapolation(self, state, values):
        # Find composition, if last composition is negative, apply extrapolation
        zc = np.append(state[1:self.nc], 1 - np.sum(state[1:self.nc]))
        if zc[-1] < -self.min_z / 10:
            self.extrapolate(state, values)
            return 1
        else:
            return 0

    def extrapolate(self, state, values):
        """
        Extrapolates the operator value at (z1, z2) using known valid ref points.
        Should be called when z3 < self.min_z (i.e., unphysical composition).

        Parameters:
            z1, z2    : coordinates (compositions) at current point
            op_index  : which operator to extrapolate (e.g., DELTA index)
        """
        vec_state = state.to_numpy()
        z1 = vec_state[1]
        z2 = vec_state[2]
        p = vec_state[0]
        T = vec_state[3]
        # dz = self.min_z
        # dz = (1.+self.min_z*10)/200
        dz = abs(1 - z1 - z2)

        # vec_values = values.to_numpy()
        vec_values = np.array(values, copy=False)
        ref_points = [
            (z1 - dz, z2),
            (z1 - dz, z2 - dz),
            (z1, z2 - dz),
            (z1 - dz / 2, z2 - dz / 2)
        ]

        A = []
        B = []

        for ref_z1, ref_z2 in ref_points:
            # for rz1, rz2 in ref_points:
            #     rz3 = 1.0 - rz1 - rz2
            #     if rz1 < 0 or rz2 < 0 or rz3 < 0:
            #         continue  #  skip bad ref point
            rz3 = 1.0 - ref_z1 - ref_z2
            if ref_z1 < 0 or ref_z2 < 0 or rz3 < 0:
                continue
            ref_state_np = np.array([p, ref_z1, ref_z2, T])
            ref_state = value_vector(ref_state_np)
            ref_values = value_vector(np.zeros(self.n_ops))

            self.evaluate(ref_state, ref_values)
            # comp = np.array([ref_z1, ref_z2, 1-ref_z1-ref_z2+self.min_z])

            # comp = np.array([ref_z1, ref_z2, 1])
            # A.append(list(comp/np.sum(comp)))
            A.append([ref_z1, ref_z2, 1])
            B.append(ref_values.to_numpy())

            # A.append([ref_z1, ref_z2, 1.0])
            # b_matrix.append(ref_val.to_numpy())
            # acc_flux_itor = self.property.acc_flux_itor
            # acc_flux_itor[0].evaluate_with_derivatives(ref_state, indices, ref_val, ref_dval)
            # # acc_flux_itor[0].evaluate_with_derivatives(ref_state, indices, ref_val, ref_dval)
            # ref_operator_vectors.append(ref_val.to_numpy())
        # # Fit plane per operator
        # A = np.array(A)
        # b_matrix = np.array(b_matrix)
        # Fit linear plane to each operator across z1/z2
        A = np.array(A)
        B = np.array(B)  # shape: (3, n_ops)
        coeffs = np.linalg.lstsq(A, B, rcond=None)[0]  # shape: (3, n_ops)

        # # Fit plane for each operator
        # coeffs = np.linalg.lstsq(A, b_matrix, rcond=None)[0]
        # extrapolated = coeffs[0] * z1 + coeffs[1] * z2 + coeffs[2]
        # Evaluate extrapolated operator values
        extrapolated = coeffs[0] * z1 + coeffs[1] * z2 + coeffs[2]  # shape: (n_ops,)
        # 1) Get the raw numpy array backing your DARTS `values`
        # out = values.to_numpy()
        #
        # # 2) Copy your extrapolated operators directly into it
        # out[:] = extrapolated
        # print("extrapolated", extrapolated)

        # vec_extrapolated = extrapolated.to_numpy()

        # values[:] = extrapolated
        # values.copy_from(value_vector(extrapolated))
        for i, value in enumerate(extrapolated):
            vec_values[i] = np.float64(value)

        # values.copy_from(value_vector(extrapolated.tolist()))
        # values_np = values.to_numpy()
        # values_np[:] = extrapolated
        # # A = np.column_stack([
        #     [pt[0] for pt in ref_points],
        #     [pt[1] for pt in ref_points],
        #     np.ones(len(ref_points))
        # ])
        # v_np = values.to_numpy()
        # for op in range(self.n_ops):
        #     b = np.array([rv[op] for rv in ref_values])
        #     coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        #     v_np[op] = coeffs[0] * z1 + coeffs[1] * z2 + coeffs[2]

        return vec_values


class WellControlOperators(OperatorsBase):
    """
    Set of operators for well controls. It contains the pressure, composition and temperature of the wellhead,
    plus a set of rate-control operators for different types of rates: molar-, mass-, volumetric- or advective
    heat rate controls
    """
    def __init__(self, property_container: PropertyBase, thermal: bool):
        super().__init__(property_container, thermal)

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
            vec_values_as_np[idx + self.property.ph] = \
                    self.property.enthalpy[self.property.ph] * self.property.dens_m[self.property.ph] * mobility

        # Store P, T and composition of current state
        idx += self.nph
        vec_values_as_np[idx + 0] = state[0]
        vec_values_as_np[idx + 1] = self.property.temperature

        return 0


class WellInitOperators(OperatorsBase):
    def __init__(self, property_container: PropertyBase, thermal: bool, is_pt: bool = True):
        super().__init__(property_container, thermal)

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
    def __init__(self, property_container: PropertyBase, thermal: bool, props: dict = None):
        """
        This is the constructor for PropertyOperator.
        The properties to be obtained from the PropertyOperators are passed to PropertyContainer as a dictionary.

        :param property_container: PropertyBase object to evaluate properties at given state
        :param thermal: Bool for thermal
        :param props: Optional dictionary of properties, default is taken from PropertyContainer
        """
        super().__init__(property_container, thermal)

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
