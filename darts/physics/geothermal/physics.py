import numpy as np
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

        # Define OBL axes
        axes_min = value_vector([min_p, min_e])
        axes_max = value_vector([max_p, max_e])
        n_axes_points = index_vector([n_points] * len(variables))

        # Define number of operators:
        n_ops = 12

        # Call PhysicsBase constructor
        super().__init__(variables=variables, nc=nc, phases=phases, n_ops=n_ops,
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

    def determine_obl_bounds(self, state_min, state_max):
        """
        Function to compute minimum and maximum enthalpy (kJ/kmol)

        :param state_min: (P,T,z) state corresponding to minimum enthalpy value
        :param state_max: (P,T,z) state corresponding to maximum enthalpy value
        """
        self.axes_min[1] = self.property_containers[0].compute_total_enthalpy(state_min, state_min[1])
        self.axes_max[1] = self.property_containers[0].compute_total_enthalpy(state_max, state_max[1])

        return

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

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_temperature):
        """""
        Function to set uniform initial reservoir condition

        :param mesh: :class:`Mesh` object
        :param uniform_pressure: Uniform pressure setting
        :param uniform_temperature: Uniform temperature setting
        """
        assert isinstance(mesh, conn_mesh)
        # nb = mesh.n_blocks

        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        state = value_vector([uniform_pressure, 0])
        enth = self.property_containers[0].compute_total_enthalpy(state, uniform_temperature)

        enthalpy = np.array(mesh.enthalpy, copy=False)
        enthalpy.fill(enth)

    def set_nonuniform_initial_conditions(self, mesh, pressure_grad, temperature_grad, ref_depth_p=0, p_at_ref_depth=1,
                                          ref_depth_T=0, T_at_ref_depth=293.15):
        """
        Function to set nonuniform initial reservoir condition

        :param mesh: :class:`Mesh` object
        :param pressure_grad: Pressure gradient, calculates pressure based on depth [1/km]
        :param temperature_grad: Temperature gradient, calculates temperature based on depth [1/km]
        :param ref_depth_p: the reference depth for the pressure, km
        :param p_at_ref_depth: the value of the pressure at the reference depth, bars
        :param ref_depth_T: the reference depth for the temperature, km
        :param T_at_ref_depth: the value of the temperature at the reference depth, K
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
