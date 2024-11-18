import numpy as np
from darts.engines import value_vector


class PropertyBase:
    nc: int
    nph: int
    output_props = {}

    def get_state(self, state: value_vector):
        pass

    def evaluate(self, state: value_vector):
        pass

    def evaluate_thermal(self, state: value_vector):
        pass

    def evaluate_at_cond(self, state: value_vector):
        pass

    def set_output_props(self, props: dict):
        """
        :param props: Dictionary of lambdas with output properties to be evaluated
        :type props: dict[str, lambda]
        """
        self.output_props = props
        return
