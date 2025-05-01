from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import Flash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, EoS
from dartsflash.components import CompData


class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        # self.set_physics()
        self.zero = 1e-10
        self.set_physics_3_phases()
        """ Note: By reducing both first_ts and max_ts divergence occurred later """
        self.set_sim_params(first_ts=0.00001, mult_ts=2, max_ts=0.0005, runtime=1000, tol_newton=1e-2, tol_linear=1e-3,
                            it_newton=10, it_linear=50, newton_type=sim_params.newton_local_chop)

        self.timer.node["initialization"].stop()

        """ Note: By increasing the mole fraction of H2O (initial water saturation), convergence happens much better."""
        """ NOte: If initial conditions are changed, sw_init_res should get updated as well. """
        # calculate the state of the reservoir for the following p_init_res, sw_init_res, and zCO2_init_res
        p_init_res = 20
        T_init_res = 370

        zCO2_init_res = self.zero
        zC1_range = np.linspace(self.zero, 1 - self.zero, 10000)
        for zC1 in zC1_range:
            state = [p_init_res, zCO2_init_res, zC1, T_init_res]
            self.physics.property_containers[0].compute_saturation_full(state)
            if self.physics.property_containers[0].sat[self.physics.phases.index("aqueous")] < self.sw_init_res:
                break

        self.initial_values = {self.physics.vars[0]: state[0],
                               self.physics.vars[1]: state[1],
                               self.physics.vars[2]: state[2],
                               self.physics.vars[3]: state[3],
                               }

    def set_reservoir(self):
        nx = 15
        ny = 15
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=1, dx=1, dy=1, dz=5,
                                         permx=100, permy=100, permz=100, poro=0.3, depth=1000)
        return

    def set_physics_3_phases(self):
        components_names = ["CO2", "C1", "H2O"]
        h2o_idx = components_names.index("H2O")
        phases_names = ['aqueous', 'gas', "LCO2"]
        comp_data = CompData(components_names, setprops=True)

        """ Activate physics """
        self.physics = Compositional(components_names, phases_names, self.timer, thermal=True, n_points=1001,
                                     min_p=1, max_p=500, min_z=self.zero / 10, max_z=1 - self.zero / 10,
                                     min_t=200, max_t=500)

        """ Initialize flash """
        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        pr = CubicEoS(comp_data, CubicEoS.PR)
        pr.set_preferred_roots(h2o_idx, 0.75, EoS.MAX)
        aq = AQEoS(comp_data, {AQEoS.CompType.water: AQEoS.Jager2003,
                               AQEoS.CompType.solute: AQEoS.Ziabakhsh2012,
                               })
        aq.set_eos_range(h2o_idx, [0.6, 1.])

        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["AQ", "PR"]
        flash_params.eos_params["PR"].root_order = [EoS.MAX, EoS.MIN]

        # Define initial guesses for stability + flash
        params = flash_params.eos_params["PR"]
        params.initial_guesses = [i for i in range(comp_data.nc)]
        params.stability_tol = 1e-20
        params.stability_switch_tol = 1e-2
        params.stability_max_iter = 50
        params.use_gmix = False

        params = flash_params.eos_params["AQ"]
        params.initial_guesses = [h2o_idx]
        params.stability_max_iter = 10
        params.use_gmix = True

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3
        # flash_params.split_tol = 1e-14
        flash_params.comp_tol = 1e-2
        # flash_params.verbose = True

        """ PropertyContainer object and correlations """
        property_container = PropertyContainer(phases_names, components_names, Mw=comp_data.Mw, min_z=self.zero / 10,
                                               temperature=None, rock_comp=0)

        property_container.flash_ev = Flash(flash_params)

        property_container.density_ev = dict([('gas', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                              ('LCO2', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                              ('aqueous', Garcia2001(components_names)), ])
        property_container.viscosity_ev = dict([('gas', Fenghour1998()),
                                                ('LCO2', Fenghour1998()),
                                                ('aqueous', Islam2012(components_names)), ])

        diff = 8.64e-6
        property_container.diffusion_ev = dict([('gas', ConstFunc(np.ones(len(components_names)) * diff)),
                                                ('LCO2', ConstFunc(np.ones(len(components_names)) * diff)),
                                                ('aqueous', ConstFunc(np.ones(len(components_names)) * diff * 1e-3))])

        property_container.enthalpy_ev = dict([('gas', EoSEnthalpy(eos=pr)),
                                               ('LCO2', EoSEnthalpy(eos=pr)),
                                               ('aqueous', EoSEnthalpy(eos=aq)), ])

        property_container.conductivity_ev = dict([('gas', ConstFunc(10.)),
                                                   ('LCO2', ConstFunc(10.)),
                                                   ('aqueous', ConstFunc(180.)), ])

        self.sw_init_res = 0.25
        swc = self.sw_init_res
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas", swc=swc, sgr=swc, n=1.5)),
                                               ('LCO2', PhaseRelPerm("oil", swc=swc, sgr=swc, n=1.5)),
                                               ('aqueous', PhaseRelPerm("wat", swc=swc, sgr=swc, n=4))])

        """ Add property region """
        self.physics.add_property_region(property_container)

        property_container.output_props = {}
        for j, ph in enumerate(phases_names):
            property_container.output_props['sat_' + ph] = lambda jj=j: property_container.sat[jj]
            property_container.output_props['rho_' + ph] = lambda jj=j: property_container.dens[jj]
            property_container.output_props['miu_' + ph] = lambda jj=j: property_container.mu[jj]
            property_container.output_props['enth_' + ph] = lambda jj=j: property_container.enthalpy[jj]
            for i, comp in enumerate(components_names):
                property_container.output_props[comp + '_in_' + ph] = lambda jj=j, ii=i: property_container.x[jj, ii]

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        rhs_flux = np.zeros(self.reservoir.mesh.n_res_blocks * self.physics.n_vars)

        inj_cell_idx = int(self.reservoir.n / 2)

        inj_rate = 1963
        inj_comp = np.array([1 - 10 * self.zero, 10 * self.zero, 10 * self.zero])
        inj_flux = inj_rate * inj_comp

        injected_fluid_specific_enthalpy = - 13200
        injected_heat_rate = inj_rate * injected_fluid_specific_enthalpy
        inj_flux = np.append(inj_flux, injected_heat_rate)

        cell_start_idx = inj_cell_idx * self.physics.n_vars
        rhs_flux[cell_start_idx:cell_start_idx+self.physics.n_vars:] = - inj_flux   # inflow (e.g., injection) becomes minus for rhs

        return rhs_flux
