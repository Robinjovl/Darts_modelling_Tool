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
        _ = self.property.evaluate(state)
        if self.thermal:
            _ = self.property.evaluate_thermal(state)

        for i, prop in enumerate(self.props_name):
            output = self.props[prop]()
            values[i] = output if not np.isnan(output) else 0.

        return 0
