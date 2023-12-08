from darts.engines import *
from darts.physics.properties.iapws.iapws_property import *
from darts.physics.physics_base import PhysicsBase
from darts.physics.geothermal.operator_evaluator import *

import numpy as np


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
                 mass_rate: bool = False, cache: bool = True):
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

        # Define number of operators:
        n_ops = 12

        # Call PhysicsBase constructor
        super().__init__(variables=variables, nc=nc, phases=phases, n_ops=n_ops,
                         axes_min=axes_min, axes_max=axes_max, n_points=n_points, timer=timer, cache=cache)

    def set_operators(self, regions):
        """
        Function to set operator objects: :class:`acc_flux_gravity_evaluator` for each of the reservoir regions,
        :class:`acc_flux_gravity_evaluator_python_well` for the well cells
        and :class:`geothermal_rate_custom_evaluator_python` for evaluation of rates.

        :param regions: List of regions. It contains the keys of the `property_containers` and `reservoir_operators` dict
        :type regions: list
        :param output_properties: Output property operators object, default is None
        """
        for region, prop_container in self.property_containers.items():
            self.reservoir_operators[region] = acc_flux_gravity_evaluator_python(prop_container)
        self.wellbore_operators = acc_flux_gravity_evaluator_python_well(self.property_containers[regions[0]])

        # create rate operators evaluator
        if self.mass_rate:
            self.rate_operators = geothermal_mass_rate_custom_evaluator_python(self.property_containers[regions[0]])
        else:
            self.rate_operators = geothermal_rate_custom_evaluator_python(self.property_containers[regions[0]])

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
        E = self.property_containers[0].total_enthalpy(uniform_temperature)
        enth = E.evaluate(state)

        enthalpy = np.array(mesh.enthalpy, copy=False)
        enthalpy.fill(enth)

    def set_nonuniform_initial_conditions(self, mesh, pressure_grad, temperature_grad):
        """
        Function to set nonuniform initial reservoir condition

        :param mesh: :class:`Mesh` object
        :param pressure_grad: Pressure gradient, calculates pressure based on depth [1/km]
        :param temperature_grad: Temperature gradient, calculates temperature based on depth [1/km]
        """
        assert isinstance(mesh, conn_mesh)

        depth = np.array(mesh.depth, copy=True)
        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = depth[:pressure.size] / 1000 * pressure_grad + 1

        enthalpy = np.array(mesh.enthalpy, copy=False)
        temperature = depth[:pressure.size] / 1000 * temperature_grad + 293.15

        for j in range(mesh.n_blocks):
            state = value_vector([pressure[j], 0])
            E = iapws_total_enthalpy_evalutor(temperature[j])
            enthalpy[j] = E.evaluate(state)
