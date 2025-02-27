import numpy as np
from typing import Union
import warnings
from scipy.interpolate import interp1d
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
        components = ["H2O"]

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
        super().__init__(state_spec=state_spec, variables=variables, components=components, phases=phases, n_ops=n_ops,
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

    def set_initial_conditions_from_depth_table(self, mesh: conn_mesh, input_distribution: dict,
                                                input_depth: Union[list, np.ndarray]):
        """
        Function to set initial conditions from given distribution of properties over depth.

        :param mesh: conn_mesh object
        :param input_distribution: Initial distributions of unknowns over depth, must have keys equal to self.vars
                                   and each entry is scalar or array of length equal to depths
        :param input_depth: Array of depths over which depth table has been specified
        """
        # Assertions of consistent depth table specification
        assert 'pressure' in input_distribution.keys() and ('temperature' in input_distribution.keys() or
                                                            'enthalpy' in input_distribution.keys())
        input_depth = input_depth if not np.isscalar(input_depth) else np.array([input_depth])
        for key, input_values in input_distribution.values():
            input_values = input_values if not np.isscalar(input_values) else np.ones(len(input_depth)) * input_values
            assert len(input_values) == len(input_depth)

        # Get depths and primary variable arrays from mesh object
        depths = np.asarray(mesh.depth)

        # adjust the size of initial_state array in c++
        mesh.initial_state.resize(mesh.n_blocks * self.n_vars)

        # Loop over variables to fill initial_state vector in c++
        for ith_var, variable in enumerate(self.vars):
            if variable == "enthalpy" and "enthalpy" not in input_distribution.keys():
                # If temperature has been provided, interpolate pressure and temperature to compute enthalpies
                p_itor = interp1d(input_depth, input_distribution['pressure'], kind='linear', fill_value='extrapolate')
                pressure = p_itor(depths)

                t_itor = interp1d(input_depth, input_distribution['temperature'], kind='linear', fill_value='extrapolate')
                temperature = t_itor(depths)

                values = np.empty(mesh.n_blocks)
                for j in range(mesh.n_blocks):
                    state = np.array([pressure[j], temperature[j]])
                    values[j] = self.property_containers[0].compute_total_enthalpy(state, temperature[j])
            else:
                # Else, interpolate primary variable
                itor = interp1d(input_depth, input_distribution[variable], kind='linear', fill_value='extrapolate')
                values = itor(depths)

            np.asarray(mesh.initial_state)[ith_var::self.n_vars] = values

    def set_initial_conditions_from_array(self, mesh: conn_mesh, input_distribution: dict):
        """""
        Function to set uniform initial reservoir condition

        :param mesh: conn_mesh object
        :param input_distribution: Initial distributions of unknowns over grid, must have keys equal to self.vars
                                   and each entry is scalar or array of length equal to number of cells
        """
        for variable, values in input_distribution.items():
            if not np.isscalar(values) and not len(values) == mesh.n_blocks:
                warnings.warn('Initial condition for variable {} has different length, resizing {} to {}'.
                              format(variable, len(values), mesh.n_blocks))
                input_distribution[variable] = np.resize(np.asarray(values), mesh.n_blocks)

        # adjust the size of initial_state array in c++
        mesh.initial_state.resize(mesh.n_blocks * self.n_vars)

        # set initial pressure
        np.asarray(mesh.initial_state)[0::self.n_vars] = input_distribution['pressure']

        # interpolate pressure and temperature to compute enthalpies
        enthalpy = np.empty(mesh.n_blocks)
        if 'enthalpy' in input_distribution.keys():
            enth = np.ones(mesh.n_blocks) * input_distribution['enthalpy'] if not np.isscalar(input_distribution['enthalpy']) else input_distribution['enthalpy']
            enthalpy[:] = enth
        elif not np.isscalar(input_distribution['pressure']):
            # Pressure specified as an array
            for j in range(mesh.n_blocks):
                state = value_vector([input_distribution['pressure'][j], 0])
                temp = input_distribution['temperature'][j] if not np.isscalar(input_distribution['temperature']) else input_distribution['temperature']
                enthalpy[j] = self.property_containers[0].compute_total_enthalpy(state, temp)
        else:
            state = value_vector([input_distribution['pressure'], 0])  # enthalpy is dummy variable
            enth = self.property_containers[0].compute_total_enthalpy(state, input_distribution['temperature'])
            enthalpy[:] = enth

        np.asarray(mesh.initial_state)[(self.n_vars - 1)::self.n_vars] = enthalpy
