from darts.engines import timer_node, value_vector
from darts.physics.base.operators_base import (
    PropertyOperators as BasePropertyOperators,
)
from darts.physics.base.operators_base import (
    ThermalVarOperator,
    WellControlOperators,
)
from darts.physics.base.physics_base import PhysicsBase
from darts.physics.chemistry.operator_evaluator import (
    ConversionOperators,
    ReservoirOperators,
)
from darts.physics.super.physics import Compositional


# Define our own operator evaluator class
class ElementBasedReactiveFlow(Compositional):
    """
    This is the Physics class for element-based reactive flow.
    """

    def __init__(
        self,
        timer: timer_node,
        elements: list[str],
        phases: list[str],
        n_points: int | list[int],
        axes_min: list[float],
        axes_max: list[float],
        epsilon_z: float,
        sim_eps_multiplier: float = 10,
        extrapolation_flag: bool = True,
        cache: bool = True,
    ):
        """
        Constructor for ElementBasedReactiveFlow class.
        :param timer: Timer object
        :type timer: timer_node
        :param elements: List of elements
        :type elements: list
        :param phases: List of phases
        :type phases: List
        :param n_points: Number of points
        :type n_points: int
        :param axes_min: Minimum axes values
        :type axes_min: list
        :param axes_max: Maximum axes values
        :type axes_max: list
        :param epsilon_z: Epsilon value for composition OBL axes (min_axis_z, max_axis_z)
        :type epsilon_z: float
        :param sim_eps_multiplier: Multiplier to epsilon_z to obtain sim_eps (minimum offset of solution state from
                                    OBL bounds, calculated as min_sim_z/max_sim_z in engine), default is 10
        :type sim_eps_multiplier: float
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :type extrapolation_flag: bool
        :param cache: Cache flag
        :type cache: bool
        """
        vars = ["p"] + elements[:-1]
        self.initial_operators = {}
        self.output_property_containers = {}

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
            epsilon_z=epsilon_z,
            sim_eps_multiplier=sim_eps_multiplier,
            extrapolation_flag=extrapolation_flag,
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
                self.property_containers[region],
                self.thermal,
                extrapolation_flag=self.extrapolation_flag,
                dz=self.dz,
            )
            self.initial_operators[region] = ConversionOperators(
                self.property_containers[region],
                self.thermal,
                extrapolation_flag=self.extrapolation_flag,
                dz=self.dz,
            )
            self.property_operators[region] = BasePropertyOperators(
                self.output_property_containers[region],
                self.thermal,
                extrapolation_flag=self.extrapolation_flag,
                dz=self.dz,
            )

        self.well_ctrl_operators = WellControlOperators(
            self.property_containers[self.regions[0]],
            self.thermal,
            extrapolation_flag=self.extrapolation_flag,
            dz=self.dz,
        )
        self.thermal_var_operator = ThermalVarOperator(
            self.property_containers[self.regions[0]],
            self.thermal,
            is_pt=(self.state_spec <= PhysicsBase.StateSpecification.PT),
            extrapolation_flag=self.extrapolation_flag,
            dz=self.dz,
        )

    def add_property_region(
        self, property_container, output_property_container, region: int = 0
    ):
        super().add_property_region(property_container, region)
        self.output_property_containers[region] = output_property_container

    def set_interpolators(
        self,
        platform='cpu',
        itor_type='multilinear',
        itor_mode='adaptive',
        itor_precision='d',
        is_barycentric: bool = False,
    ):
        """
        Function to set interpolator objects:
        - :class:`acc_flux_itor` main interpolator
        - :class:`comp_itor` initialization and porosity interpolator
        - :class:`property_itor` output property interpolator
        - :class:`well_ctrl_itor` well control interpolator
        - :class:`thermal_var_itor` well initialization interpolator
        :param platform: Platform to run the simulation
        :type platform: str (cpu or gpu)
        :param itor_type: Interpolator type
        :type itor_type: str (multilinear or linear)
        :param itor_mode: Interpolator mode
        :type itor_mode: str (adaptive or static)
        :param itor_precision: Interpolator precision
        :type itor_precision: str
        :param is_barycentric: Flag which turn on barycentric interpolation on Delaunay simplices
        :type is_barycentric: bool
        """
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
        self.thermal_var_itor, n_thermal_var_ops = self.create_interpolator(
            self.thermal_var_operator,
            n_ops=self.thermal_var_operator.n_ops,
            axes_min=value_vector(self.PT_axes_min),
            axes_max=value_vector(self.PT_axes_max),
            timer_name='well initialization',
            platform=platform,
            algorithm=itor_type,
            mode=itor_mode,
            precision=itor_precision,
        )
        self.n_thermal_var_ops = n_thermal_var_ops
