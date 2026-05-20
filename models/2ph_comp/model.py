from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params, well_control_iface, ms_well
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic


class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        self.set_sim_params(first_ts=0.001, mult_ts=2, max_ts=1, runtime=1000, tol_newton=1e-2, tol_linear=1e-3,
                            it_newton=10, it_linear=50, newton_type=sim_params.newton_local_chop)

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        nx = 1000
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=1, dy=10, dz=10,
                                         permx=100, permy=100, permz=10, poro=0.3, depth=1000)
        return

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", res_cell_idx=(1, 1, 1))
        self.reservoir.add_well("P1")
        self.reservoir.add_perforation("P1", res_cell_idx=(self.reservoir.nx, 1, 1))

    def set_physics(self):
        zero = 1e-8
        epsilon = 1e-9

        components = ['CO2', 'C1', 'H2O']
        phases = ['gas', 'aqueous']
        Mw = [44.01, 16.04, 18.015]

        # Create a property container
        property_container = PropertyContainer(phases_name=phases, components_name=components,
                                               Mw=Mw, eps_z=epsilon, temperature=1.)

        """ properties correlations """
        property_container.flash_ev = ConstantK(len(components), [4, 2, 1e-1], zero)
        property_container.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                                              ('aqueous', DensityBasic(compr=1e-5, dens0=600))])
        property_container.viscosity_ev = dict([('gas', ConstFunc(0.05)),
                                                ('aqueous', ConstFunc(0.5))])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                               ('aqueous', PhaseRelPerm("oil"))])

        """ Activate physics """
        thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        # New axes_step-based API: pass per-axis cell size directly. With the adaptive
        # multi-index-keyed interpolator the cache extends past any prescribed window on
        # demand, so axes_max / n_points become advisory. The step values below
        # reproduce the legacy grid exactly (n_points=200 over P in [1, 300] and z in [0, 1]).
        n_points = 200
        p_step = (300 - 1) / (n_points - 1)
        z_step = (1 - 3 * epsilon) / (n_points - 1)
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     axes_step=[p_step, z_step, z_step],
                                     min_p=1, min_z=0., epsilon_z=epsilon,
                                     n_points=n_points,  # advisory: drives legacy pickle export window
                                     extrapolation_flag=True)
        # property_container.output_props = {
        #     "sat0": lambda: property_container.sat[0],
        #     "dens0": lambda: property_container.dens[0],
        #     "nu0": lambda: property_container.nu[0],
        #     "x00": lambda: property_container.x[0,0]
        #     }

        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 50,
                              self.physics.vars[1]: 0.1,
                              self.physics.vars[2]: 0.2
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        zero = self.physics.axes_min[1]
        inj_composition = [1.0 - 2 * zero*10, zero*10]
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=140., inj_composition=inj_composition)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=50.)
                # Control total mass rate of the produced fluid
                # self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE,
                #                                is_inj=False, target=4000)
