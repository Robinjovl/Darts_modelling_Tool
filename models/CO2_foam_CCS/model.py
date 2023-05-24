from reservoir import UnstructReservoir
from physics import CO2Brine
from darts.models.darts_model import DartsModel
import numpy as np
from operator_evaluator import *
from darts.engines import value_vector

# Model class creation here!
class Model(DartsModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        """Reservoir"""
        self.const_perm = 100
        self.poro = 0.15
        mesh_file = 'wedgesmall.msh'
        self.reservoir = UnstructReservoir(permx=self.const_perm, permy=self.const_perm, permz=self.const_perm, frac_aper=0,
                                           mesh_file=mesh_file, poro=self.poro)

        self.zero = 1e-8
        """Physical properties"""
        # Create property containers:
        self.property_container = property_container(phase_name=['gas', 'wat'], component_name=['CO2', 'H2O'],
                                                     min_z=self.zero)
        self.components = self.property_container.component_name
        self.phases = self.property_container.phase_name

        """ properties correlations """
        # foam parameter, fmmob, fmdry, epdry, fmmob = 0 no foam generation
        self.foam_paras = np.array([100, 0.35, 1000])
        self.property_container.flash_ev = Flash(self.components)
        self.property_container.density_ev = dict([('wat', DensityBrine()),
                                                     ('gas', DensityVap())])
        self.property_container.viscosity_ev = dict([('wat', ViscosityBrine()),
                                                     ('gas', ViscosityVap())])
        self.property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("wat")),
                                                     ('gas', PhaseRelPerm("gas"))])
        self.property_container.foam_STARS_FM_ev = FMEvaluator(self.foam_paras)

        """ Activate physics """
        self.physics = CO2Brine(self.timer, n_points=200, min_p=1, max_p=1000, min_z=self.zero/10,
                                max_z=1-self.zero/10, property_container=self.property_container)

        # Some newton parameters for non-linear solution:
        self.params.first_ts = 1e-4
        self.params.mult_ts = 1.5
        self.params.max_ts = 1

        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-4
        self.params.max_i_newton = 10
        self.params.max_i_linear = 50
        self.params.newton_type = sim_params.newton_local_chop
        self.params.newton_params[0] = 0.25

        self.runtime = 10

        self.timer.node["initialization"].stop()

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        """ initialize conditions for all scenarios"""
        # equilibrium pressure
        pressure = 90
        composition = np.ones(self.reservoir.nb)*1e-6

        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, pressure, composition)

    def set_boundary_conditions(self):
        self.inj_CO2 = [0.3]
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_rate_gas_inj(1, self.inj_CO2)
            else:
                w.control = self.physics.new_bhp_prod(85)

