from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params, well_control_iface
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic

from dartsflash.libflash import Flash, PXFlash
from dartsflash.libflash import CubicEoS, FlashParams, EoS
from dartsflash.components import CompData

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
                                         permx=100, permy=100, permz=10, poro=0.3, depth=1000,
                                         forch_coef=1e9)
        return

    # def set_wells(self):
    #     self.reservoir.add_well("I1")
    #     self.reservoir.add_perforation("I1", cell_index=(1, 1, 1))
    #
    #     self.reservoir.add_well("P1")
    #     self.reservoir.add_perforation("P1", cell_index=(self.reservoir.nx, 1, 1))

    def set_physics(self):
        """Physical properties"""
        self.zero = 1e-8
        # Create property containers:
        components_names = ['CO2', 'C1']
        phases = ['gas']

        comp_data = CompData(components_names, setprops=True)

        flash_params = FlashParams(comp_data)
        pr = CubicEoS(comp_data, CubicEoS.PR)

        flash_params.add_eos("PR", pr)
        flash_params.eos_order = ["PR"]

        property_container = PropertyContainer(phases_name=phases, components_name=components_names,
                                               Mw=comp_data.Mw, min_z=self.zero / 10, temperature=320.)

        """ properties correlations """
        property_container.flash_ev = Flash(flash_params)
        property_container.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                                                ])
        property_container.viscosity_ev = dict([('gas', ConstFunc(0.05)),
                                                ])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                                ])

        """ Activate physics """
        thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components_names, phases, self.timer, state_spec=state_spec,
                                     n_points=200, min_p=1, max_p=300, min_z=self.zero/10, max_z=1-self.zero/10)
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
                              self.physics.vars[1]: 0.01,
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    # def set_well_controls(self):
    #     zero = self.physics.axes_min[1]
    #     inj_composition = [1.0 - zero*10]
    #     for i, w in enumerate(self.reservoir.wells):
    #         if i == 0:
    #             self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
    #                                            is_inj=True, target=140., inj_composition=inj_composition)
    #         else:
    #             self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
    #                                            is_inj=False, target=50.)

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        rhs_flux = np.zeros(self.reservoir.mesh.n_res_blocks * self.physics.n_vars)
        inj_cell_idx = 0
        inj_rate = 10000
        inj_comp = np.array([1 - self.zero, self.zero])
        inj_flux = inj_rate * inj_comp
        cell_start_idx = inj_cell_idx * self.physics.n_vars
        rhs_flux[cell_start_idx:cell_start_idx + self.physics.n_vars:] = - inj_flux  # inflow (e.g., injection) becomes minus for rhs

        return rhs_flux
