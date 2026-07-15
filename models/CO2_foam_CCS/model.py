from darts.engines import *
from darts.models.cicd_model import CICDModel
from darts.nonlinear_solvers import NewtonSpec, ChopSpec

from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from darts.physics.super.physics import Compositional
from property_container import PropertyContainer
from operator_evaluator import AccFluxGravityEvaluator, AccFluxGravityWellEvaluator, RateEvaluator, PropertyEvaluator

from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.density import DensityBasic, DensityBrineCO2
from darts.physics.properties.flash import ConstantK

import numpy as np


class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        self.nonlinear_solver = NewtonSpec(tolerance=1e-3, max_iterations=10,
                                           chop=ChopSpec(mode='local', factor=0.25))
        self.set_sim_params(first_ts=1e-4, mult_ts=1.5, max_ts=1, runtime=10, tol_linear=1e-4,
                            it_linear=50)

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        const_perm = 100
        poro = 0.15
        mesh_file = 'wedgesmall.msh'
        self.reservoir = UnstructReservoir(self.timer, poro=poro, permx=const_perm, permy=const_perm, permz=const_perm,
                                           frac_aper=0, mesh_file=mesh_file, cache=False)

        # Add injection well for CO2:
        self.reservoir.add_well("I1", well_diameter=0.1)
        # Perforate all boundary cells:
        for nth_perf in range(len(self.left_boundary_cells)):
            well_index = mesh.volume[self.left_boundary_cells[nth_perf]] / self.max_well_vol * self.well_index
            well_indexD = 0.
            self.reservoir.add_perforation(well_name=self.wells[-1], res_cell_idx=self.left_boundary_cells[nth_perf],
                                           well_index=well_index, well_indexD=well_indexD)

        return

    def set_physics(self):
        zero = 1e-8

        """Physical properties"""
        # Create property containers:
        components = ['CO2', 'H2O']
        Mw = np.array([44.01, 18.015])
        phases = ['gas', 'wat']

        self.ini_stream = [1e-6]
        self.inj_composition = [0.3]

        property_container = PropertyContainer(phase_name=phases, component_name=components, eps_z=zero, Mw=Mw)

        """ properties correlations """
        # foam parameter, fmmob, fmdry, epdry, fmmob = 0 no foam generation
        foam_paras = np.array([100, 0.35, 1000])

        ki = np.array([44.5, 2.05e-2])
        # ki = np.array([40, 2.47e-4])
        property_container.flash_ev = ConstantK(nc=len(components), ki=ki, eps_z=1e-12)
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
        thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        nz = len(components) - 1
        ax_step = [5.0] + [5e-3] * nz
        ax_origin = [1.0] + [eps_z] * nz
        self.physics = CustomPhysics(components, phases, self.timer,
                                     axes_step=ax_step, axes_origin=ax_origin, epsilon_z=eps_z,
                                     state_spec=state_spec, cache=False)
        self.physics.add_property_region(property_container)
        return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 90,
                              self.physics.vars[1]: self.ini_stream[0],
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MOLAR_RATE,
                                               is_inj=True, phase_name='gas', target=1., inj_composition=self.inj_composition)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=85.)


class CustomPhysics(Compositional):
    def __init__(self, components, phases, timer, axes_step, axes_origin=None, epsilon_z=1e-9,
                 state_spec=Compositional.StateSpecification.P, cache=False):
        super().__init__(components, phases, timer, axes_step=axes_step, axes_origin=axes_origin,
                         epsilon_z=epsilon_z, state_spec=state_spec, cache=cache)

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
