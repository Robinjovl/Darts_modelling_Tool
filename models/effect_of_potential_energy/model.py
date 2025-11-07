import numpy as np
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.super.initialize import Initialize

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData


class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.reservoir.grav_acceleration_for_spe = 9.80665
        self.zero = 1e-10
        self.set_physics(n_points=10001)

        self.set_sim_params(first_ts=1e-5, mult_ts=1.5, max_ts=5, tol_newton=1e-3,
                            tol_linear=1e-5, it_newton=10, it_linear=50)

        self.timer.node["initialization"].stop()

        return

    def set_reservoir(self):
        nx = 1
        ny = 1
        nz = 20
        dx = 10
        dy = 10
        self.linear_inverval = 10 * np.ones(int(nz / 2))   # Reservoir interval
        log_inverval = np.logspace(start=1, stop=2, num=int(nz / 2))   # Underburden interval
        dz = np.concatenate((self.linear_inverval, log_inverval))

        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                         permx=100, permy=100, permz=1000, hcap=2200, rcond=100, poro=0.2)
        self.reservoir.boundary_volumes['xy_plus'] = 1e20

        return

    def set_physics(self, n_points):
        """Physical properties"""
        components = ["H2O", "CO2"]
        phases = ["aq", "CO2_rich_phase"]
        comp_data = CompData(components, setprops=True)

        pr = CubicEoS(comp_data, CubicEoS.PR)
        # aq = Jager2003(comp_data)
        aq = AQEoS(comp_data, AQEoS.Ziabakhsh2012)

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["AQ", "PR"]

        state_spec = Compositional.StateSpecification.PT

        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw,
                                               min_z=self.zero/10)

        property_container.flash_ev = NegativeFlash(flash_params, ["AQ", "PR"], [InitialGuess.Henry_AV])
        property_container.density_ev = dict([('CO2_rich_phase', EoSDensity(pr, comp_data.Mw)),
                                              ('aq', Garcia2001(components))])
        property_container.viscosity_ev = dict([('CO2_rich_phase', Fenghour1998()),
                                                ('aq', Islam2012(components))])
        property_container.rel_perm_ev = dict([('CO2_rich_phase', PhaseRelPerm("gas")),
                                               ('aq', PhaseRelPerm("oil"))])

        property_container.enthalpy_ev = dict([('CO2_rich_phase', EoSEnthalpy(pr)),
                                               ('aq', EoSEnthalpy(aq))])
        property_container.conductivity_ev = dict([('CO2_rich_phase', ConstFunc(10.)),
                                                   ('aq', ConstFunc(180.)), ])

        property_container.output_props = {"sat_aq": lambda: property_container.sat[0],
                                           "sat_CO2_rich_phase": lambda: property_container.sat[1],
                                           "xCO2": lambda: property_container.x[0, 1],
                                           "yH2O": lambda: property_container.x[1, 0]
                                           }

        self.physics = Compositional(components, phases, self.timer, n_points, min_p=1, max_p=600, min_z=self.zero/10,
                                     max_z=1-self.zero/10, min_t=220, max_t=500, state_spec=state_spec, cache=False)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        base_depth = self.reservoir.mesh.depth[0]
        boundary_state = {'H2O': 1 - self.zero, 'pressure': 100., 'temperature': 350}
        init = Initialize(physics=self.physics)
        X = init.solve(depth_bottom=self.reservoir.global_data['depth'].max(),
                       depth_top=self.reservoir.global_data['depth'].min(),
                       depth_known=base_depth, boundary_state=boundary_state,
                       primary_specs={'H2O': 1 - self.zero}, secondary_specs={})
        self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh, input_depth=init.depths,
                                                             input_distribution={var: X[i::self.physics.n_vars] for i, var in
                                                                                 enumerate(self.physics.vars)})

        return

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)

        # Inject CO2
        inj_rate = 1963.19 * 10   # 10 kg/s CO2
        inj_comp = np.array([10 * self.zero, 1 - 10 * self.zero])
        inj_flux = inj_rate * inj_comp

        cell_idx = 0
        co2_idx = self.physics.components.index("CO2")
        inj_fluid_molar_enthalpy = - 2000

        inj_fluid_specific_potential_energy = self.reservoir.mesh.cell_spe[cell_idx]
        inj_fluid_molar_potential_energy = inj_fluid_specific_potential_energy * self.physics.property_containers[0].Mw[co2_idx]
        inj_fluid_molar_energy = inj_fluid_molar_enthalpy + inj_fluid_molar_potential_energy

        injected_heat_rate = inj_rate * inj_fluid_molar_energy
        inj_flux = np.append(inj_flux, injected_heat_rate)

        cell_start_idx = cell_idx * self.physics.n_vars
        rhs_flux[cell_start_idx:cell_start_idx + self.physics.n_vars:] = - inj_flux  # inflow (e.g., injection) becomes minus for rhs

        return rhs_flux
