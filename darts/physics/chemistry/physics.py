from darts.engines import timer_node
from darts.physics.base.operator_evaluator import (
    PropertyOperators as BasePropertyOperators,
)
from darts.physics.base.operator_evaluator import (
    ThermalVarOperator,
    WellCtrlOperators,
)
from darts.physics.base.physics import PhysicsBase
from darts.physics.chemistry.operator_evaluator import (
    ConversionOperators,
    ReservoirOperators,
)


# Define our own operator evaluator class
class ElementBasedReactiveFlow(PhysicsBase):
    """
    This is the Physics class for element-based reactive flow.
    """

    def __init__(
        self,
        timer: timer_node,
        elements: list[str],
        phases: list[str],
        axes_step: list[float],
        axes_origin: list[float] = None,
        epsilon_z: float = 1e-9,
        sim_eps_multiplier: float = 10,
        extrapolation_flag: bool = True,
        cache: bool = True,
    ):
        """
        Constructor for ElementBasedReactiveFlow class.

        :param timer: Timer object.
        :param elements: List of elements.
        :param phases: List of phases.
        :param axes_step: Per-axis cell size [p_step, z_step_1, ..., z_step_{n_el-1}].
        :param axes_origin: Per-axis grid origin (defaults via Compositional).
        :param epsilon_z: Composition axis offset (default 1e-9).
        :param sim_eps_multiplier: Multiplier on epsilon_z to obtain sim_eps.
        :param extrapolation_flag: Enable extrapolation logic for z[last] < 0 (n_el >= 3).
        :param cache: Cache supporting points to disk between runs.
        """
        vars = ["p"] + elements[:-1]
        self.initial_operators = {}
        self.output_property_containers = {}

        super().__init__(
            components=elements,
            phases=phases,
            axes_step=axes_step,
            axes_origin=axes_origin,
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
        :class:`WellOperators` for the well segments, :class:`WellCtrlOperators` for well controls
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

        self.well_ctrl_operators = WellCtrlOperators(
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

    def _parallel_wrap_targets(self):
        """
        Chemistry has reservoir/initial/property per region, plus singular
        well_ctrl_operators and thermal_var_operator. There is no separate
        well_operators (acc_flux_w_itor aliases acc_flux_itor[0]).
        """
        targets = []
        for region in self.regions:
            targets.append(('reservoir_operators', region))
            targets.append(('initial_operators', region))
            targets.append(('property_operators', region))
        targets.append(('well_ctrl_operators', None))
        targets.append(('thermal_var_operator', None))
        return targets

    def set_interpolators(
        self,
        platform='cpu',
        itor_type='multilinear',
        itor_precision='d',
        is_barycentric: bool = False,
        parallel_evaluation: bool = False,
        n_workers: int = None,
        evaluator_factory_hook=None,
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
        :param itor_precision: Interpolator precision
        :type itor_precision: str
        :param is_barycentric: Flag which turn on barycentric interpolation on Delaunay simplices
        :type is_barycentric: bool
        :param parallel_evaluation: Enable parallel batch evaluation via multiprocessing
        :type parallel_evaluation: bool
        :param n_workers: Number of worker processes (default: os.cpu_count())
        :type n_workers: int
        :param evaluator_factory_hook: Callable ``(attribute, region) -> factory`` for parallel evaluation
        :type evaluator_factory_hook: callable
        """
        # Optionally wrap every chemistry evaluator with ParallelEvaluator via a
        # single shared pool. Chemistry has no separate well_operators (well uses
        # acc_flux_itor[0]) but does have initial_operators per region.
        if parallel_evaluation:
            self._wrap_evaluators_parallel(
                self._parallel_wrap_targets(),
                evaluator_factory_hook,
                n_workers,
            )

        # Create actual accumulation and flux interpolator:
        # NOTE: every create_interpolator call passes region=str(region), exactly as
        # Compositional.set_interpolators does. The on-disk supporting-point cache
        # name is built from the evaluator CLASS name plus the shape parameters and
        # this tag, so without it the regions of a multi-region model -- whose
        # evaluators are all ReservoirOperators/ConversionOperators/PropertyOperators
        # of identical shape -- hash to ONE cache file and silently serve each other's
        # points. Regions differ precisely in their property container (kinetics,
        # rel perm, ...), so those points are not interchangeable.
        self.acc_flux_itor = {}
        self.comp_itor = {}
        self.property_itor = {}
        for region in self.regions:
            self.acc_flux_itor[region], _ = self.create_interpolator(
                evaluator=self.reservoir_operators[region],
                timer_name=f'reservoir {region} interpolation',
                n_ops=self.n_ops,
                platform=platform,
                algorithm=itor_type,
                precision=itor_precision,
                region=str(region),
                is_barycentric=is_barycentric,
            )

            # ==============================================================================================================
            # Create initialization & porosity evaluator
            self.comp_itor[region], n_comp_ops = self.create_interpolator(
                evaluator=self.initial_operators[region],
                timer_name=f'comp {region} interpolation',
                n_ops=len(self.initial_operators[region].props_name),
                platform=platform,
                algorithm=itor_type,
                precision=itor_precision,
                region=str(region),
                is_barycentric=is_barycentric,
            )
            self.n_comp_itor_ops = n_comp_ops

            # ==============================================================================================================
            # Create property interpolator:
            self.property_itor[region], n_property_ops = self.create_interpolator(
                evaluator=self.property_operators[region],
                timer_name=f'property {region} interpolation',
                n_ops=len(self.property_operators[region].props_name),
                platform=platform,
                algorithm=itor_type,
                precision=itor_precision,
                region=str(region),
                is_barycentric=is_barycentric,
            )
            self.n_property_itor_ops = n_property_ops
        self.acc_flux_w_itor = self.acc_flux_itor[0]

        self.well_ctrl_itor, n_well_ctrl_ops = self.create_interpolator(
            self.well_ctrl_operators,
            n_ops=self.well_ctrl_operators.n_ops,
            timer_name='well controls interpolation',
            platform=platform,
            algorithm=itor_type,
            precision=itor_precision,
        )
        self.n_well_ctrl_itor_ops = n_well_ctrl_ops
        self.thermal_var_itor, n_thermal_var_ops = self.create_interpolator(
            self.thermal_var_operator,
            n_ops=self.thermal_var_operator.n_ops,
            timer_name='well initialization',
            platform=platform,
            algorithm=itor_type,
            precision=itor_precision,
        )
        self.n_thermal_var_ops = n_thermal_var_ops
