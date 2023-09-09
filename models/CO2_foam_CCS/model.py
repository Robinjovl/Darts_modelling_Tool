from reservoir import UnstructReservoir
from darts.models.cicd_model import CICDModel
from darts.physics.super.physics import Compositional
from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.density import DensityBasic, DensityBrineCO2
from darts.physics.properties.flash import ConstantK

from property_container import PropertyContainer
import numpy as np
from operator_evaluator import AccFluxGravityEvaluator, AccFluxGravityWellEvaluator, RateEvaluator, PropertyEvaluator
from darts.engines import *


class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_wells()
        self.set_physics()

        self.set_sim_params(first_ts=1e-4, mult_ts=1.5, max_ts=1, runtime=10, tol_newton=1e-3, tol_linear=1e-4,
                            it_newton=10, it_linear=50, newton_type=sim_params.newton_local_chop)
        self.params.newton_params[0] = 0.25

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        """Reservoir"""
        self.const_perm = 100
        self.poro = 0.15
        mesh_file = 'wedgesmall.msh'
        self.reservoir = UnstructReservoir(permx=self.const_perm, permy=self.const_perm, permz=self.const_perm, frac_aper=0,
                                           mesh_file=mesh_file, poro=self.poro)

        return

    def set_wells(self):
        return

    def set_physics(self):
        zero = 1e-8

        """Physical properties"""
        # Create property containers:
        components = ['CO2', 'H2O']
        Mw = np.array([44.01, 18.015])
        phases = ['gas', 'wat']
        property_container = PropertyContainer(phase_name=phases, component_name=components, min_z=zero, Mw=Mw)

        """ properties correlations """
        # foam parameter, fmmob, fmdry, epdry, fmmob = 0 no foam generation
        foam_paras = np.array([100, 0.35, 1000])

        ki = np.array([44.5, 2.05e-2])
        # ki = np.array([40, 2.47e-4])
        property_container.flash_ev = ConstantK(nc=2, ki=ki, min_z=1e-12)
        # property_container.flash_ev = Flash(components)
        # property_container.density_ev = dict([('wat', DensityBrine()),
        #                                       ('gas', DensityVap())])
        property_container.density_ev = dict([('wat', DensityBrineCO2(components, dens0=980., co2_mult=4./0.0125)),
                                              ('gas', DensityBasic(dens0=733., compr=1e-7, p0=1.))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(0.511)),
                                                ('gas', ConstFunc(0.2611))])
        property_container.rel_perm_ev = dict([('wat', RelPerm("wat", swc=0.2, sgr=0.2, kre=0.2, n=4.2)),
                                               ('gas', RelPerm("gas", swc=0.2, sgr=0.2, kre=0.94, n=1.3))])
        property_container.foam_STARS_FM_ev = FMEvaluator(foam_paras)

        """ Activate physics """
        self.physics = CustomPhysics(components, phases, self.timer,
                                     n_points=200, min_p=1., max_p=1000., min_z=zero/10, max_z=1.-zero/10, cache=False)
        self.physics.add_property_region(property_container)
        self.physics.init_physics()
        return

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        """ initialize conditions for all scenarios"""
        # equilibrium pressure
        pressure = 90
        composition = [1e-6]

        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, pressure, composition)

    def set_boundary_conditions(self):
        self.inj_CO2 = [0.3]
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_rate_gas_inj(1, self.inj_CO2)
            else:
                w.control = self.physics.new_bhp_prod(85)


class CustomPhysics(Compositional):
    def __init__(self, components, phases, timer, n_points, min_p, max_p, min_z, max_z, min_t=None, max_t=None, thermal=0,
                 discr_type='tpfa', cache=False):
        super().__init__(components, phases, timer, n_points, min_p, max_p, min_z, max_z, min_t, max_t, thermal, discr_type, cache)

    def set_operators(self, regions, output_properties=None):
        for region, prop_container in self.property_containers.items():
            self.reservoir_operators[region] = AccFluxGravityEvaluator(prop_container)
        self.wellbore_operators = AccFluxGravityWellEvaluator(self.property_containers[regions[0]])

        self.rate_operators = RateEvaluator(self.property_containers[regions[0]])

        if output_properties is None:
            self.property_operators = PropertyEvaluator(self.vars, self.property_containers[regions[0]])
        else:
            self.property_operators = output_properties

        return

    def set_well_controls(self):
        # define well control factories
        # Injection wells (upwind method requires both bhp and inj_stream for bhp controlled injection wells):
        self.new_bhp_inj = lambda bhp, inj_stream: bhp_inj_well_control(bhp, value_vector(inj_stream))
        self.new_rate_gas_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 0, self.nc, self.nc, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        # Production wells:
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_gas_prod = lambda rate: rate_prod_well_control(self.phases, 0, self.nc, self.nc, rate,
                                                                     self.rate_itor)
        self.new_rate_water_prod = lambda rate: rate_prod_well_control(self.phases, 1, self.nc, self.nc, rate,
                                                                       self.rate_itor)

        return


class RelPerm:
    def __init__(self, phase, swc=0., sgr=0., kre=1., n=2.):
        self.phase = phase

        self.Swc = swc
        self.Sgr = sgr
        if phase == "wat":
            self.kre = kre
            self.sr = self.Swc
            self.sr1 = self.Sgr
            self.n = n

        else:
            self.kre = kre
            self.sr = self.Sgr
            self.sr1 = self.Swc
            self.n = n

    def evaluate(self, sat):
        # sat = sat_w
        if sat >= 1 - self.sr1:
            kr = self.kre
        elif sat <= self.sr:
            kr = 0
        else:
            # general Brook-Corey
            kr = self.kre * ((sat - self.sr) / (1 - self.Sgr - self.Swc)) ** self.n

            # if self.kre == 0.2:
            #     Se = (sat - self.Swc) / (1 - self.Swc)
            #     kr = Se**4
            # else:
            #     Se = (1 - sat - self.Swc) / (1 - self.Swc)
            #     Swa = 1 - self.Sgr
            #     Sea = (Swa - self.Swc) / (1 - self.Swc)
            #     krna = 0.4 * (1 - Sea ** 2) * (1 - Sea) ** 2
            #     C = krna
            #
            #     kr = 0.4 * (1 - Se ** 2) * (1 - Se) ** 2 - C
            #
            # if kr > 1:
            #     kr = 1
            # elif kr < 0:
            #     kr = 0

        return kr


class FMEvaluator:
    def __init__(self, foam_paras):
        foam = foam_paras
        self.fmmob = foam[0]
        self.fmdry = foam[1]
        self.epdry = foam[2]

    def evaluate(self, sg):
        water_sat = 1 - sg

        Fw = 0.5 + np.arctan(self.epdry * (water_sat - self.fmdry)) / np.pi

        FM = 1/(1 + self.fmmob * Fw)

        return FM
