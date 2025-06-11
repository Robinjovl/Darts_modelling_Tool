from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic, DensityBrineCO2


# Model class creation here!
class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        self.set_sim_params(first_ts=0.01, mult_ts=2, max_ts=20, runtime=1000, tol_newton=1e-2, tol_linear=1e-3,
                            it_newton=10, it_linear=50, newton_type=sim_params.newton_local_chop)

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        nx = 1000
        self.reservoir = StructReservoir(self.timer, nx, ny=1, nz=1, dx=1, dy=10, dz=10, permx=100, permy=100,
                                         permz=10, poro=0.3, depth=1000)

        return

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", cell_index=(1, 1, 1))
        self.reservoir.add_well("P1")
        self.reservoir.add_perforation("P1", cell_index=(self.reservoir.nx, 1, 1))

    def set_physics(self):
        """Physical properties"""
        # Create property containers:
        zero = 1e-8
        components = ['g', 'o', 'w']
        phases = ['gas', 'oil', 'wat']
        Mw = [1, 1, 1]
        self.inj_composition = [1 - 2 * zero, zero]
        self.ini_stream = [0.05, 0.2 - zero]

        """ properties correlations """
        property_container = ModelProperties(phases_name=phases, components_name=components, Mw=Mw, min_z=zero / 10)

        property_container.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                                              ('oil', DensityBasic(compr=1e-5, dens0=600)),
                                              ('wat', DensityBrineCO2(components, compr=1e-5, dens0=1000, co2_mult=0))])
        property_container.viscosity_ev = dict([('gas', ConstFunc(0.05)),
                                                ('oil', ConstFunc(0.5)),
                                                ('wat', ConstFunc(0.5))])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                               ('oil', PhaseRelPerm("oil")),
                                               ('wat', PhaseRelPerm("wat"))])

        """ Activate physics """
        thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=100, min_p=1, max_p=200, min_z=zero / 10, max_z=1 - zero / 10)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 100.,
                              self.physics.vars[1]: self.ini_stream[0],
                              self.physics.vars[2]: self.ini_stream[1],
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=120., inj_composition=self.inj_composition)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=60.)


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, Mw, min_z=1e-11, rock_comp=1e-6):
        # Call base class constructor
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, min_z=min_z,
                         rock_comp=rock_comp, temperature=1.)

    def run_flash(self, pressure, temperature, zc, evaluate_PT: bool = None):
        # evaluate_PT argument is required in PropertyContainer but is not needed in this model
        
        ph = np.array([0, 1, 2], dtype=np.intp)

        for i in range(self.nc):
            self.x[i][i] = 1
        self.nu = zc

        return ph
