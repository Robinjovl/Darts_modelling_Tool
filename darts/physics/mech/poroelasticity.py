import numpy as np

from darts.engines import *
from darts.physics.base.operator_evaluator import *
from darts.physics.base.operator_evaluator import (
    PropertyOperators,
    ThermalVarOperator,
    WellCtrlOperators,
)
from darts.physics.base.physics import PhysicsBase


class Poroelasticity(PhysicsBase):
    """
    This is the Physics class for compositional poroelastic simulation.

    It includes:
    - Creating Reservoir, Well, Rate and Property operators and interpolators for P-z or P-T-z compositional simulation
    - Initializing the :class:`super_engine`
    - Setting well controls (rate, bhp)
    - Defining initial and boundary conditions
    """

    def __init__(
        self,
        components: list,
        phases: list,
        timer: timer_node,
        axes_step: list[float],
        axes_origin: list[float] = None,
        epsilon_z: float = 1e-9,
        sim_eps_multiplier: float = 10,
        extrapolation_flag: bool = True,
        state_spec: PhysicsBase.StateSpecification = PhysicsBase.StateSpecification.P,
        cache: bool = False,
        discretizer: str = 'mech_discretizer',
    ):
        """
        Constructor of the Poroelasticity Physics class. Defines the OBL grid for P-z
        or P-T-z compositional simulation via PhysicsBase with mechanics added.

        :param components: List of components.
        :param phases: List of phases.
        :param timer: Timer object.
        :param axes_step: Per-axis cell size (forwarded to PhysicsBase).
        :param axes_origin: Per-axis grid origin (defaults via PhysicsBase).
        :param epsilon_z: Composition axis offset (default 1e-9).
        :param sim_eps_multiplier: Multiplier on epsilon_z to obtain sim_eps.
        :param extrapolation_flag: Extrapolation logic for z[last] < 0 (nc >= 3).
        :param state_spec: P (default), PT, or PH.
        :param cache: Cache supporting points to disk between runs.
        :param discretizer: 'mech_discretizer' (default) or 'pm_discretizer'.
        """
        super().__init__(
            components=components,
            phases=phases,
            timer=timer,
            axes_step=axes_step,
            axes_origin=axes_origin,
            epsilon_z=epsilon_z,
            sim_eps_multiplier=sim_eps_multiplier,
            extrapolation_flag=extrapolation_flag,
            state_spec=state_spec,
            cache=cache,
        )

        self.n_dim = 3
        self.discretizer_name = discretizer

        if self.discretizer_name == 'mech_discretizer':
            # Number of operators = NE /*acc*/ + NE * NP /*flux*/ + NP * /*density*/ + NP /*UPSAT*/ + NE * NP /*gradient*/ + NE /*kinetic*/
            # + 2 * NP /*gravpc*/ + 1 /*poro*/ + NP /*LAMBDA*/ + NP /*SAT*/ + NP /*enthalpy*/
            # + 2 /*temperature and pressure*/ + 1 /*rock density*/
            # n_ops = NE * (2 * nph + 2) + 7 * nph + 4
            self.n_ops = self.n_vars * (2 * self.nph + 2) + 7 * self.nph + 4
        else:  # if self.discretizer_name == 'pm_discretizer':
            self.n_ops = 2 * self.n_vars
            assert not self.thermal

    def get_engine_interpolator_state(self, n_blocks: int = None) -> np.ndarray:
        """
        Return the OBL interpolator state for the mechanics engine.

        Unlike the flow engines, the poroelastic ``engine.X`` stores ``n_dim`` displacement
        degrees of freedom per cell in addition to the flow unknowns, so the per-cell stride is
        returned by ``engine.get_n_vars()`` (not the physics' flow ``n_vars`` that the base
        implementation assumes -- that would mix displacements into the state and fail to
        reshape). Rather than hard-code the field order per discretizer, we read the layout
        straight from the engine: the flow unknowns form a contiguous block of ``n_vars``
        columns starting at ``engine.P_VAR``
        (``mech_discretizer``: ``P_VAR == 0`` -> flow first; ``pm_discretizer``:
        ``P_VAR == n_dim`` -> flow last, matching ``UnstructReservoirMech.cell_property``).

        We reshape to ``(-1, engine.get_n_vars())``, slice the
        ``[P_VAR : P_VAR + n_vars)`` flow columns, and append any history fields before
        flattening (so the layout matches the reservoir/well interpolators that consume
        ``[X | Xhistory]``).

        :param n_blocks: Number of reservoir blocks. When ``None``, inferred from
                         ``engine.X.size // engine.get_n_vars()``
        :type n_blocks: int, optional
        :returns: One-dimensional array of length ``n_blocks * n_state`` with primary flow vars
                  and history values interleaved per cell (displacements stripped)
        :rtype: numpy.ndarray
        """
        # Per-cell width in engine.X = flow unknowns + displacement degrees of freedom.
        stride = self.engine.get_n_vars()
        if n_blocks is None:
            n_blocks = self.engine.X.size // stride
        # flow unknowns are a contiguous block of n_vars columns starting at P_VAR
        flow_cols = slice(self.engine.P_VAR, self.engine.P_VAR + self.n_vars)
        X = np.asarray(self.engine.X, copy=False).reshape(-1, stride)[
            :n_blocks, flow_cols
        ]
        if not self.history_fields:
            return X.flatten()
        Xhistory = np.asarray(self.engine.Xhistory, copy=False).reshape(
            -1, self.n_history
        )[:n_blocks]
        return np.concatenate([X, Xhistory], axis=1).flatten()

    def set_engine(self, discretizer: str = 'mech_discretizer', platform: str = 'cpu'):
        """
        Function to set :class:`engine_super` object.

        :param discretizer: Which discretizer in use (affect the choice of engine):
        'mech_discretizer' (default) or 'pm_discretizer'
        :type discretizer: str
        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        """
        if discretizer == 'mech_discretizer':
            if self.thermal:
                return eval(
                    f"engine_super_elastic_{platform}{self.nc:d}_{self.nph:d}_t"
                )()
            else:
                return eval(
                    f"engine_super_elastic_{platform}{self.nc:d}_{self.nph:d}"
                )()
        else:  # discretizer == 'pm_discretizer':
            return eval(f"engine_pm_{platform}")()

    def set_operators(self):
        """
        Function to set operator objects: :class:`ReservoirOperators` for each of the reservoir regions,
        :class:`WellOperators` for the well segments, :class:`WellCtrlOperators` for well controls
        and a :class:`PropertyOperator` for the evaluation of properties.
        """
        if self.discretizer_name == "pm_discretizer":
            for region, prop_container in self.property_containers.items():
                self.reservoir_operators[region] = SinglePhaseGeomechanicsOperators(
                    prop_container,
                    self.thermal,
                    extrapolation_flag=self.extrapolation_flag,
                    dz=self.dz,
                )
                self.property_operators[region] = PropertyOperators(
                    prop_container,
                    self.thermal,
                    extrapolation_flag=self.extrapolation_flag,
                    dz=self.dz,
                )
            self.well_operators = SinglePhaseGeomechanicsOperators(
                self.property_containers[self.regions[0]],
                self.thermal,
                extrapolation_flag=self.extrapolation_flag,
                dz=self.dz,
            )
        else:
            for region, prop_container in self.property_containers.items():
                self.reservoir_operators[region] = GeomechanicsReservoirOperators(
                    prop_container,
                    self.thermal,
                    extrapolation_flag=self.extrapolation_flag,
                    dz=self.dz,
                )
                self.property_operators[region] = PropertyOperators(
                    prop_container,
                    self.thermal,
                    extrapolation_flag=self.extrapolation_flag,
                    dz=self.dz,
                )
            self.well_operators = GeomechanicsReservoirOperators(
                self.property_containers[self.regions[0]],
                thermal=False,
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

        return

    def init_wells(self, wells):
        """ ""
        Function to initialize physics of wells for poromechanics

        :param wells: List of :class:`ms_well` objects
        """
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_mech_physics(
                self.engine.get_n_vars(),
                self.engine.P_VAR,
                self.n_vars,
                self.n_ops,
                self.phases,
                self.well_ctrl_itor,
                self.thermal_var_itor,
                self.thermal,
            )

    def set_initial_conditions_from_depth_table(
        self,
        mesh: conn_mesh,
        input_distribution: dict,
        input_depth: list | np.ndarray,
        input_displacement: list,
    ):
        """
        Function to set initial conditions from given distribution of properties over depth.

        :param mesh: conn_mesh object
        :param input_distribution: Initial distributions of unknowns over depth, must have keys equal to self.vars
                                   and each entry is scalar or array of length equal to depths
        :param input_depth: Array of depths over which depth table has been specified
        :param input_displacement: Displacement [], array
        """
        super().set_initial_conditions_from_depth_table(
            mesh, input_depth=input_depth, input_distribution=input_distribution
        )

        # set initial displacements
        for i in range(self.n_dim):
            np.asarray(mesh.displacement)[i :: self.n_dim] = input_displacement[i]

    def set_initial_conditions_from_array(
        self, mesh: conn_mesh, input_distribution: dict, input_displacement: list
    ):
        """
        Method to set initial conditions by arrays or uniformly for all cells

        :param mesh: conn_mesh object
        :param input_distribution: Initial distributions of unknowns over grid, must have keys equal to self.vars
                                   and each entry is scalar or array of length equal to number of cells
        :param input_displacement: Displacement [], array
        """
        super().set_initial_conditions_from_array(
            mesh, input_distribution=input_distribution
        )

        # set initial displacements
        for i in range(self.n_dim):
            np.asarray(mesh.displacement)[i :: self.n_dim] = input_displacement[i]
