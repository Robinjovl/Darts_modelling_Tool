import numpy as np

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
        vec_state_as_np = state.to_numpy()
        vec_values_as_np = values.to_numpy()
        vec_values_as_np[:] = 0

        self.property.evaluate(vec_state_as_np)

        # Store rate controls
        mobility = (
            self.property.kr[self.property.ph] / self.property.mu[self.property.ph]
        )

        # Molar rate
        idx = 0
        vec_values_as_np[idx + self.property.ph] = (
            self.property.dens_m[self.property.ph] * mobility
        )

        # Mass rate
        idx += self.nph
        vec_values_as_np[idx + self.property.ph] = (
            self.property.dens[self.property.ph] * mobility
        )

        # Volumetric rate
        idx += self.nph
        vec_values_as_np[idx + self.property.ph] = mobility

        # Advective heat rate
        idx += self.nph
        if self.thermal:
            self.property.evaluate_thermal(vec_state_as_np)
            vec_values_as_np[idx + self.property.ph] = (
                self.property.enthalpy[self.property.ph]
                * self.property.dens_m[self.property.ph]
                * mobility
            )

        # Store P, T and composition of current state
        idx += self.nph
        vec_values_as_np[idx + 0] = state[0]
        vec_values_as_np[idx + 1] = self.property.temperature

        return 0


class WellInitOperators(OperatorsBase):
    def __init__(
        self, property_container: PropertyBase, thermal: bool, is_pt: bool = True
    ):
        super().__init__(property_container, thermal)

        self.n_ops = 1
        self.is_pt = is_pt

    def evaluate(self, state_pt, values):
        vec_values_as_np = values.to_numpy()
        vec_values_as_np[:] = 0

        if self.is_pt:
            vec_values_as_np[0] = state_pt[-1]
        else:
            state_pt = np.array(
                list(state_pt[: self.nc])
                + [state_pt[-1] if self.thermal else self.temperature]
            )
            vec_values_as_np[0] = self.property.compute_total_enthalpy(
                state_pt=state_pt
            )

        return 0


class PropertyOperators(OperatorsBase):
    """
    This class contains a set of operators for evaluation of output properties.
    A set of interpolators is created in the :class:`Physics` object to rapidly obtain properties after simulation.
    """

    def __init__(
        self, property_container: PropertyBase, thermal: bool, props: dict = None
    ):
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
        _ = self.property.evaluate(state)
        if self.thermal:
            _ = self.property.evaluate_thermal(state)

        for i, prop in enumerate(self.props_name):
            output = self.props[prop]()
            values[i] = output if not np.isnan(output) else 0.0

        return 0
