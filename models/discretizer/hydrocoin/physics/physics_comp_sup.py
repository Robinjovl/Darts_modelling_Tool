import numpy as np
from darts.engines import *
from darts.models.physics.physics_base import PhysicsBase

from physics.operator_evaluator_sup import *


# Define our own operator evaluator class
class Compositional(PhysicsBase):
    def __init__(self, property_container: list, timer, n_points, min_p, max_p, min_z, max_z, min_t=-1, max_t=-1, thermal=0,
                 platform='cpu', itor_type='multilinear', itor_mode='adaptive', itor_precision='d', cache=False, out_props=None,
                 discr_type='mpfa'):
        super().__init__(cache)
        # Obtain properties from user input during initialization:
        self.timer = timer.node["simulation"]
        self.components = property_container[0].components_name
        self.nc = property_container[0].nc
        self.phases = property_container[0].phases_name
        self.nph = property_container[0].nph
        self.n_vars = self.nc + thermal
        self.n_props = 4
        NE = self.n_vars
        self.vars = ['pressure', 'Temp']

        self.n_axes_points = index_vector([n_points] * self.n_vars)

        """ Name of interpolation method and engine used for this physics: """
        # engine including gravity term

        self.n_ops = NE + self.nph * NE + self.nph + self.nph * NE + NE + 3 + 2 * self.nph + 1

        if thermal:
            self.vars = ['pressure'] + self.components[:-1] + ['temperature']
            self.n_axes_min = value_vector([min_p] + [min_z] * (self.nc - 1) + [min_t])
            self.n_axes_max = value_vector([max_p] + [max_z] * (self.nc - 1) + [max_t])
            self.acc_flux_etor_verycoarse = ReservoirThermalOperators(property_container[0])
            self.acc_flux_etor_coarse = ReservoirThermalOperators(property_container[1])
            #self.acc_flux_etor_fine = ReservoirThermalOperators(property_container[2])
            #self.acc_flux_etor_veryfine = ReservoirThermalOperators(property_container[3])
            self.acc_flux_w_etor = WellOperators(property_container[0])  # assume isothermal flow in wells
            if discr_type == 'mpfa':
                self.engine = eval("engine_super_mp_%s%d_%d_t" % (platform, self.nc, self.nph))()
            else:
                self.engine = eval("engine_super_%s%d_%d_t" % (platform, self.nc, self.nph))()
        else:
            self.vars = ['pressure'] + self.components[:-1]
            self.n_axes_min = value_vector([min_p] + [min_z] * (self.nc - 1))
            self.n_axes_max = value_vector([max_p] + [max_z] * (self.nc - 1))
            self.acc_flux_etor_verycoarse = ReservoirOperators(property_container[0])
            self.acc_flux_etor_coarse = ReservoirOperators(property_container[1])
            #self.acc_flux_etor_fine = ReservoirOperators(property_container[2])
            #self.acc_flux_etor_veryfine = ReservoirOperators(property_container[3])
            self.acc_flux_w_etor = WellOperators(property_container[0])
            if discr_type == 'mpfa':
                self.engine = eval("engine_super_mp_%s%d_%d" % (platform, self.nc, self.nph))()
            else:
                self.engine = eval("engine_super_%s%d_%d" % (platform, self.nc, self.nph))()

        self.rate_etor = RateOperators(property_container[0])

        if out_props is not None:
            self.property_etor = out_props
        else:
            self.property_etor = DefaultPropertyEvaluator(self.vars, property_container)

        # try first to create interpolator with 4-byte index type
        self.acc_flux_itor_verycoarse = self.create_interpolator(self.acc_flux_etor_verycoarse, self.n_vars, self.n_ops, self.n_axes_points,
                                                        self.n_axes_min, self.n_axes_max, platform=platform)
        self.acc_flux_itor_coarse = self.create_interpolator(self.acc_flux_etor_coarse, self.n_vars, self.n_ops, self.n_axes_points,
                                                        self.n_axes_min, self.n_axes_max, platform=platform)
        # self.acc_flux_itor_fine = self.create_interpolator(self.acc_flux_etor_fine, self.n_vars, self.n_ops, self.n_axes_points,
        #                                                 self.n_axes_min, self.n_axes_max, platform=platform)
        # self.acc_flux_itor_veryfine = self.create_interpolator(self.acc_flux_etor_veryfine, self.n_vars, self.n_ops, self.n_axes_points,
        #                                                   self.n_axes_min, self.n_axes_max, platform=platform)

        self.acc_flux_w_itor = self.create_interpolator(self.acc_flux_w_etor, self.n_vars, self.n_ops, self.n_axes_points,
                                                        self.n_axes_min, self.n_axes_max, platform=platform)

        self.property_itor = self.create_interpolator(self.property_etor, self.n_vars, self.n_props, self.n_axes_points,
                                                      self.n_axes_min, self.n_axes_max, platform=platform)

        self.rate_itor = self.create_interpolator(self.rate_etor, self.n_vars, self.nph, self.n_axes_points,
                                                  self.n_axes_min, self.n_axes_max, platform='cpu')

        self.create_itor_timers(self.acc_flux_itor_verycoarse, 'reservoir interpolation 0')
        self.create_itor_timers(self.acc_flux_itor_coarse, 'reservoir interpolation C')
        # self.create_itor_timers(self.acc_flux_itor_fine, 'reservoir interpolation D')
        # self.create_itor_timers(self.acc_flux_itor_veryfine, 'reservoir interpolation ESF')
        self.create_itor_timers(self.acc_flux_w_itor, 'well interpolation')

        self.create_itor_timers(self.property_itor, 'property interpolation 0')
        self.create_itor_timers(self.rate_itor, 'well controls interpolation')

        # define well control factories
        # Injection wells (upwind method requires both bhp and inj_stream for bhp controlled injection wells):
        self.new_bhp_inj = lambda bhp, inj_stream: bhp_inj_well_control(bhp, value_vector(inj_stream))
        self.new_rate_inj = lambda rate, inj_stream, iph: rate_inj_well_control(self.phases, iph, self.n_vars, self.n_vars, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        # Production wells:
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_prod = lambda rate, iph: rate_prod_well_control(self.phases, iph, self.nc, self.nc, rate, self.rate_itor)

    # Define some class methods:
    def init_wells(self, wells):
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_vars, self.phases, self.rate_itor)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_composition: list):
        assert isinstance(mesh, conn_mesh)

        nb = mesh.n_blocks
        nb_res = mesh.n_res_blocks
        """ Uniform Initial conditions """
        # set initial pressure
        pz_bounds = np.array(mesh.pz_bounds, copy=False)

        pressure_grad = -0.09995
        pressure = np.array(mesh.pressure, copy=False)
        if isinstance( uniform_pressure, (np.ndarray, np.generic) ):
            pressure[:nb_res] = uniform_pressure
            pz_bounds[::self.nc] = np.mean(uniform_pressure)
        else:
            depth = np.array(mesh.depth, copy=True)
            nonuniform_pressure = depth[:nb] * pressure_grad + uniform_pressure
            pressure[:] = nonuniform_pressure
            # pressure = np.array(mesh.pressure, copy=False)
            # pressure.fill(uniform_pressure)
            pz_bounds[::self.nc] = np.mean(nonuniform_pressure)

        # set initial composition
        mesh.composition.resize(nb * (self.nc - 1))
        composition = np.array(mesh.composition, copy=False)
        # composition[:] = np.array(uniform_composition)
        if self.nc == 2:
            for c in range(self.nc - 1):
                composition[c:(self.nc - 1) * nb_res:(self.nc - 1)] = uniform_composition[:]
                pz_bounds[1+c::self.nc] = np.mean(uniform_composition[:])
        else:
            for c in range(self.nc - 1):  # Denis
                composition[c::(self.nc - 1)] = uniform_composition[c]
                pz_bounds[1+c::self.nc] = np.mean(uniform_composition[c])

    def set_uniform_T_initial_conditions(self, mesh, uniform_pressure, uniform_composition: list, uniform_temp):
        """""
        Function to set uniform initial reservoir condition
        Arguments:
            -mesh: mesh object
            -uniform_pressure: uniform pressure setting
            -uniform_composition: uniform uniform_composition setting
        """
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks

        # set initial pressure
        depth = np.array(mesh.depth, copy=True)
        pressure_grad = 0.09995
        pressure = np.array(mesh.pressure, copy=False)
        nonuniform_pressure = depth * pressure_grad + 1
        pressure[:] = nonuniform_pressure
        # pressure = np.array(mesh.pressure, copy=False)
        # pressure.fill(uniform_pressure)

        temperature = np.array(mesh.temperature, copy=False)
        temperature.fill(uniform_temp)

        # set initial composition
        mesh.composition.resize(nb * (self.nc - 1))
        composition = np.array(mesh.composition, copy=False)
        if self.nc == 2:
            for c in range(self.nc - 1):
                composition[c::(self.nc - 1)] = uniform_composition[:]
        else:
            for c in range(self.nc - 1):
                composition[c::(self.nc - 1)] = uniform_composition[c]

    def set_boundary_conditions(self, mesh, uniform_pressure, uniform_composition):
        assert isinstance(mesh, conn_mesh)

        # Class methods which can create constant pressure and composition boundary condition:
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        mesh.composition.resize(mesh.n_blocks * (self.nc - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.nc - 1):
            composition[c::(self.nc - 1)] = uniform_composition[c]
