import numpy as np
from typing import Union
from darts.engines import *
from darts.physics.base.physics_base import PhysicsBase
from darts.physics.base.operators_base import PropertyOperators
from darts.physics.geothermal.operator_evaluator import *


class Geothermal(PhysicsBase):
    """
    This is the Physics class for Geothermal simulation.

    It includes:
    - Creating Reservoir, Well, Rate and Property operators and interpolators for P-H simulation
    - Initializing the :class:`nce_g_engine`
    - Setting well controls (rate, bhp)
    - Defining initial and boundary conditions
    """

    def __init__(self, timer: timer_node, n_points: int, min_p: float, max_p: float, min_e: float, max_e: float,
                 mass_rate: bool = False, cache: bool = False):
        """
        This is the constructor of the Geothermal Physics class.

        It defines the OBL grid for P-H simulation.

        :param timer: Timer object
        :type timer: :class:`darts.engines.timer_node`
        :param n_points: Number of OBL points along axes
        :type n_points: int
        :param min_p, max_p: Minimum, maximum pressure
        :type min_p, max_p: float
        :param min_e, max_e: Minimum, maximum enthalpy
        :type min_e, max_e: float
        :param mass_rate: Switch for mass rate/volume rate?
        :type mass_rate: bool
        :param cache: Switch to cache operator values
        :type cache: bool
        """
        # Set nc=1, thermal=True
        nc = 1

        # Define phases and variables
        self.mass_rate = mass_rate
        if self.mass_rate:
            phases = ['water_mass', 'steam_mass', 'temperature', 'energy']
        else:
            phases = ['water', 'steam', 'temperature', 'energy']
        variables = ['pressure', 'enthalpy']
        state_spec = PhysicsBase.StateSpecification.PH

        # Define OBL axes
        axes_min = value_vector([min_p, min_e])
        axes_max = value_vector([max_p, max_e])
        n_axes_points = index_vector([n_points] * len(variables))

        # Define number of operators:
        # N_OPS = NC /*acc*/ + NC * NP /*flux*/ + 2 + NP /*energy acc, flux, cond*/ + NP /*density*/ + 1 /*temperature*/
        # = nc + nc*NP + 2 + NP + NP + 1 = 10
        n_ops = 10

        # Call PhysicsBase constructor
        super().__init__(state_spec=state_spec, variables=variables, nc=nc, phases=phases, n_ops=n_ops,
                         axes_min=axes_min, axes_max=axes_max, n_axes_points=n_axes_points, timer=timer, cache=cache)

    def set_operators(self):
        """
        Function to set operator objects: :class:`acc_flux_gravity_evaluator` for each of the reservoir regions,
        :class:`acc_flux_gravity_evaluator_python_well` for the well cells
        and :class:`geothermal_rate_custom_evaluator_python` for evaluation of rates.
        """
        for region in self.regions:
            self.reservoir_operators[region] = acc_flux_gravity_evaluator_python(self.property_containers[region])
            self.property_operators[region] = PropertyOperators(self.property_containers[region], thermal=True)
            self.mass_flux_operators[region] = MassFluxOperators(self.property_containers[region])
        self.wellbore_operators = acc_flux_gravity_evaluator_python_well(self.property_containers[self.regions[0]])

        # create rate operators evaluator
        if self.mass_rate:
            self.rate_operators = geothermal_mass_rate_custom_evaluator_python(self.property_containers[self.regions[0]])
        else:
            self.rate_operators = geothermal_rate_custom_evaluator_python(self.property_containers[self.regions[0]])

        return

    def set_engine(self, discr_type: str = 'tpfa', platform: str = 'cpu'):
        """
        Function to set :class:`engine_nce_g` object.

        :param discr_type: Type of discretization, 'tpfa' (default) or 'mpfa'
        :type discr_type: str
        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        """
        return eval("engine_nce_g_%s%d_%d" % (platform, self.nc, self.nph - 2))()

    def define_well_controls(self):
        # create well controls
        # water stream
        # pure water injection at constant temperature

        self.water_inj_stream = value_vector([1.0])
        # water injection at constant temperature with bhp control
        self.new_bhp_water_inj = lambda bhp, temp: gt_bhp_temp_inj_well_control(self.phases, self.n_vars, bhp, temp,
                                                                                self.water_inj_stream, self.rate_itor)
        # water injection at constant temperature with volumetric rate control
        self.new_rate_water_inj = lambda rate, temp: gt_rate_temp_inj_well_control(self.phases, 0, self.n_vars, rate,
                                                                                   temp, self.water_inj_stream,
                                                                                   self.rate_itor)
        # water production with bhp control
        self.new_bhp_prod = lambda bhp: gt_bhp_prod_well_control(bhp)
        # water production with volumetric rate control
        self.new_rate_water_prod = lambda rate: gt_rate_prod_well_control(self.phases, 0, self.n_vars,
                                                                          rate, self.rate_itor)
        # water injection of constant enthalpy with mass rate control
        self.new_mass_rate_water_inj = lambda rate, enth: \
            gt_mass_rate_enthalpy_inj_well_control(self.phases, 0, self.n_vars,
                                                   self.water_inj_stream,
                                                   rate, enth,
                                                   self.rate_itor)
        # water production with mass rate control
        self.new_mass_rate_water_prod = lambda rate: gt_mass_rate_prod_well_control(self.phases, 0, self.n_vars,
                                                                                    rate, self.rate_itor)
        return

    def set_uniform_initial_conditions(self, mesh: conn_mesh,
                                       pressure_input: Union[float, list, np.ndarray],
                                       composition_input: Union[list, np.ndarray] = None,
                                       temperature_input: Union[float, list, np.ndarray] = None):
        """""
        Function to set uniform initial reservoir condition

        :param mesh: conn_mesh object
        :param pressure_input: Pressure [bar], uniform or array
        :param composition_input: List of compositions [z_0, ..., z_{nc-1}], set of scalars or arrays, not used in Geothermal physics
        :param temperature_input: Temperature [K], only required for thermal models, uniform or array
        """
        assert isinstance(mesh, conn_mesh)
        # nb = mesh.n_blocks

        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = pressure_input

        enthalpy = np.array(mesh.enthalpy, copy=False)
        if hasattr(pressure_input, '__len__'):
            # Pressure specified as an array
            for j in range(mesh.n_blocks):
                state = value_vector([pressure_input[j], 0])
                temp = temperature_input[j] if hasattr(temperature_input, "__len__") else temperature_input
                enthalpy[j] = self.property_containers[0].compute_total_enthalpy(state, temp)
        else:
            state = value_vector([pressure_input, 0])
            enth = self.property_containers[0].compute_total_enthalpy(state, temperature_input)
            enthalpy[:] = enth

    def set_nonuniform_initial_conditions(self, mesh: conn_mesh, pressure_grad: float = 0., temperature_grad: float = 0.,
                                          ref_depth_p: float = 0., p_at_ref_depth: float = 1.,
                                          ref_depth_T: float = 0., T_at_ref_depth: float = 293.15,
                                          composition_input: Union[list, np.ndarray] = None):
        """
        Method to set initial conditions with gradients

        :param mesh: conn_mesh object
        :param pressure_grad: Pressure gradient [bar/km], calculates pressure based on depth [1/km], default is 0
        :param temperature_grad: Temperature gradient [K/km], calculates temperature based on depth [1/km], default is 0
        :param ref_depth_p: Reference depth for pressure [km], default is 0
        :param p_at_ref_depth: Pressure at reference depth [bar], default is 1
        :param ref_depth_T: Reference depth for temperature [K], default is 0
        :param T_at_ref_depth: Temperature at reference depth [K], default is 293.15
        :param composition_input: Unused variable in Geothermal physics
        """
        assert isinstance(mesh, conn_mesh)

        depth = np.array(mesh.depth, copy=True)
        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = (depth[:pressure.size] / 1000 - ref_depth_p) * pressure_grad + p_at_ref_depth

        # set initial enthalpy through given temperature and pressure
        enthalpy = np.array(mesh.enthalpy, copy=False)
        temperature = (depth[:pressure.size] / 1000 - ref_depth_T) * temperature_grad + T_at_ref_depth

        for j in range(mesh.n_blocks):
            state = value_vector([pressure[j], 0])
            enthalpy[j] = self.property_containers[0].compute_total_enthalpy(state, temperature[j])
