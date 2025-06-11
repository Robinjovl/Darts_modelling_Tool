from darts.engines import *
from darts.physics.base.physics_base import PhysicsBase
from darts.physics.super.physics import Compositional
from darts.physics.base.operators_base import WellControlOperators, WellInitOperators

from phreeqc_dissolution.operator_evaluator import my_own_acc_flux_etor, my_own_comp_etor, my_own_property_evaluator
import numpy as np

# Define our own operator evaluator class
class PhreeqcDissolution(Compositional):
    def __init__(self, timer, elements, n_points, axes_min, axes_max, input_data_struct, properties,
                 platform='cpu', itor_type='multilinear', itor_mode='adaptive', itor_precision='d', cache=True):
        # Obtain properties from user input during initialization:
        self.input_data_struct = input_data_struct
        nc = len(elements)
        NE = nc
        vars = ["p"] + elements[:-1]
        phases = ['vapor', 'liquid']
        self.initial_operators = {}

        super().__init__(components=elements, phases=phases, n_points=n_points,
                         min_p=axes_min[0], max_p=axes_max[0], min_z=axes_min[1], max_z=1-axes_min[1],
                         axes_min=axes_min, axes_max=axes_max, n_axes_points=n_points,
                         timer=timer, cache=cache)
        self.vars = vars

    def set_operators(self):
        """
        Function to set operator objects: :class:`ReservoirOperators` for each of the reservoir regions,
        :class:`WellOperators` for the well segments, :class:`WellControlOperators` for well control
        and a :class:`PropertyOperator` for the evaluation of properties.
        """
        for region in self.regions:
            self.reservoir_operators[region] = my_own_acc_flux_etor(self.input_data_struct, self.property_containers[region])
            self.initial_operators[region] = my_own_comp_etor(self.input_data_struct, self.property_containers[region])
            self.property_operators[region] = my_own_property_evaluator(self.input_data_struct, self.property_containers[region])

        self.well_ctrl_operators = WellControlOperators(self.property_containers[self.regions[0]], self.thermal)
        self.well_init_operators = WellInitOperators(self.property_containers[self.regions[0]], self.thermal,
                                                     is_pt=(self.state_spec <= PhysicsBase.StateSpecification.PT))

    def set_interpolators(self, platform='cpu', itor_type='multilinear', itor_mode='adaptive',
                          itor_precision='d', is_barycentric: bool = False):

        # Create actual accumulation and flux interpolator:
        self.acc_flux_itor = {}
        self.comp_itor = {}
        self.property_itor = {}
        for region in self.regions:
            self.acc_flux_itor[region] = self.create_interpolator(evaluator=self.reservoir_operators[region],
                                                          timer_name='reservoir interpolation',
                                                          n_ops=self.n_ops,
                                                          axes_min=self.axes_min,
                                                          axes_max=self.axes_max,
                                                          platform=platform,
                                                          algorithm=itor_type,
                                                          mode=itor_mode,
                                                          precision=itor_precision,
                                                          is_barycentric=is_barycentric)

            # ==============================================================================================================
            # Create initialization & porosity evaluator
            self.comp_itor[region] = self.create_interpolator(evaluator=self.initial_operators[region],
                                                      timer_name='comp %d interpolation' % region,
                                                      n_ops=self.input_data_struct.n_init_ops,
                                                      axes_min=self.axes_min,
                                                      axes_max=self.axes_max,
                                                      platform=platform,
                                                      algorithm=itor_type,
                                                      mode=itor_mode,
                                                      precision=itor_precision,
                                                      is_barycentric=is_barycentric)

            # ==============================================================================================================
            # Create property interpolator:
            self.property_itor[region] = self.create_interpolator(evaluator=self.property_operators[region],
                                                      timer_name='property %d interpolation' % region,
                                                      n_ops=self.input_data_struct.n_prop_ops,
                                                      axes_min=self.axes_min,
                                                      axes_max=self.axes_max,
                                                      platform=platform,
                                                      algorithm=itor_type,
                                                      mode=itor_mode,
                                                      precision=itor_precision,
                                                      is_barycentric=is_barycentric)

        self.acc_flux_w_itor = self.acc_flux_itor[0]

        self.well_ctrl_itor = self.create_interpolator(self.well_ctrl_operators, n_ops=self.well_ctrl_operators.n_ops,
                                                       axes_min=self.axes_min, axes_max=self.axes_max,
                                                       timer_name='well controls interpolation',
                                                       platform=platform, algorithm=itor_type, mode=itor_mode,
                                                       precision=itor_precision)
        self.well_init_itor = self.create_interpolator(self.well_init_operators, n_ops=self.well_init_operators.n_ops,
                                                       axes_min=value_vector(self.PT_axes_min),
                                                       axes_max=value_vector(self.PT_axes_max),
                                                       timer_name='well initialization',
                                                       platform=platform, algorithm=itor_type, mode=itor_mode,
                                                       precision=itor_precision)
