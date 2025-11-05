from darts.engines import *
from darts.physics.base.operators_base import WellControlOperators, WellInitOperators
from darts.physics.base.physics_base import PhysicsBase
from darts.physics.chemistry.operator_evaluator import (
    CoversionOperators,
    PropertyOperators,
    ReservoirOperators,
)
from darts.physics.super.physics import Compositional


# Define our own operator evaluator class
class ElementBasedReactiveFlow(Compositional):
    def __init__(
        self,
        timer,
        elements,
        n_points,
        axes_min,
        axes_max,
        properties,
        platform='cpu',
        itor_type='multilinear',
        itor_mode='adaptive',
        itor_precision='d',
        cache=True,
    ):
        vars = ["p"] + elements[:-1]
        phases = ['vapor', 'liquid']
        self.initial_operators = {}

        super().__init__(
            components=elements,
            phases=phases,
            n_points=n_points,
            min_p=axes_min[0],
            max_p=axes_max[0],
            min_z=axes_min[1],
            max_z=1 - axes_min[1],
            axes_min=axes_min,
            axes_max=axes_max,
            n_axes_points=n_points,
            timer=timer,
            cache=cache,
        )
        self.vars = vars

    def set_operators(self):
        """
        Function to set operator objects: :class:`ReservoirOperators` for each of the reservoir regions,
        :class:`WellOperators` for the well segments, :class:`WellControlOperators` for well control
        and a :class:`PropertyOperator` for the evaluation of properties.
        """
        for region in self.regions:
            self.reservoir_operators[region] = ReservoirOperators(
                self.property_containers[region]
            )
            self.initial_operators[region] = CoversionOperators(
                self.property_containers[region]
            )
            self.property_operators[region] = PropertyOperators(
                self.property_containers[region]
            )

        self.well_ctrl_operators = WellControlOperators(
            self.property_containers[self.regions[0]], self.thermal
        )
        self.well_init_operators = WellInitOperators(
            self.property_containers[self.regions[0]],
            self.thermal,
            is_pt=(self.state_spec <= PhysicsBase.StateSpecification.PT),
        )

    def set_interpolators(
        self,
        platform='cpu',
        itor_type='multilinear',
        itor_mode='adaptive',
        itor_precision='d',
        is_barycentric: bool = False,
    ):
        # Create actual accumulation and flux interpolator:
        self.acc_flux_itor = {}
        self.comp_itor = {}
        self.property_itor = {}
        for region in self.regions:
            self.acc_flux_itor[region], _ = self.create_interpolator(
                evaluator=self.reservoir_operators[region],
                timer_name='reservoir interpolation',
                n_ops=self.n_ops,
                axes_min=self.axes_min,
                axes_max=self.axes_max,
                platform=platform,
                algorithm=itor_type,
                mode=itor_mode,
                precision=itor_precision,
                is_barycentric=is_barycentric,
            )

            # ==============================================================================================================
            # Create initialization & porosity evaluator
            self.comp_itor[region], n_comp_ops = self.create_interpolator(
                evaluator=self.initial_operators[region],
                timer_name=f'comp {region} interpolation',
                n_ops=len(self.initial_operators[region].props_name),
                axes_min=self.axes_min,
                axes_max=self.axes_max,
                platform=platform,
                algorithm=itor_type,
                mode=itor_mode,
                precision=itor_precision,
                is_barycentric=is_barycentric,
            )
            self.n_comp_itor_ops = n_comp_ops

            # ==============================================================================================================
            # Create property interpolator:
            self.property_itor[region], n_property_ops = self.create_interpolator(
                evaluator=self.property_operators[region],
                timer_name=f'property {region} interpolation',
                n_ops=len(self.property_operators[region].props_name),
                axes_min=self.axes_min,
                axes_max=self.axes_max,
                platform=platform,
                algorithm=itor_type,
                mode=itor_mode,
                precision=itor_precision,
                is_barycentric=is_barycentric,
            )
            self.n_property_itor_ops = n_property_ops
        self.acc_flux_w_itor = self.acc_flux_itor[0]

        self.well_ctrl_itor, n_well_ctrl_ops = self.create_interpolator(
            self.well_ctrl_operators,
            n_ops=self.well_ctrl_operators.n_ops,
            axes_min=self.axes_min,
            axes_max=self.axes_max,
            timer_name='well controls interpolation',
            platform=platform,
            algorithm=itor_type,
            mode=itor_mode,
            precision=itor_precision,
        )
        self.n_well_ctrl_itor_ops = n_well_ctrl_ops
        self.well_init_itor, n_well_init_ops = self.create_interpolator(
            self.well_init_operators,
            n_ops=self.well_init_operators.n_ops,
            axes_min=value_vector(self.PT_axes_min),
            axes_max=value_vector(self.PT_axes_max),
            timer_name='well initialization',
            platform=platform,
            algorithm=itor_type,
            mode=itor_mode,
            precision=itor_precision,
        )
        self.n_well_init_itor_ops = n_well_init_ops
