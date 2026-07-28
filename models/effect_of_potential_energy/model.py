import numpy as np
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel

from darts.physics.base.physics import PhysicsBase
from darts.physics.base.property_container import PropertyContainer
from darts.physics.base.initialize import Initialize

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from darts.nonlinear_solvers import NewtonSolver

from dartsflash.libflash import EoS, InitialGuess
from dartsflash.components import CompData
from dartsflash.mixtures import DARTSFlash, VLAq


class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.reservoir.grav_acceleration_for_spe = 9.80665
        self.zero = 1e-10
        self.set_physics()

        self.nonlinear_solver = NewtonSolver(tolerance=1e-3, max_iterations=10)
        self.set_sim_params(first_ts=1e-5, mult_ts=1.5, max_ts=5,
                            tol_linear=1e-5, it_linear=50,
                            runtime=50, # This runtime will be used when CI test is conducted without the main file
                            )

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

    def set_physics(self):
        """Physical properties"""
        components = ["H2O", "CO2"]
        self.components = components
        phases = ["aq", "CO2_rich_phase"]
        comp_data = CompData(components, setprops=True)

        """ Define flash """
        flash_ev = VLAq(comp_data, hybrid=True)
        flash_ev.set_vl_eos("PR", root_order=[EoS.STABLE],
                            trial_comps=[InitialGuess.Yi.Wilson],
                            stability_tol=1e-20, switch_tol=1e-2, max_iter=50, use_gmix=False
                            )
        flash_ev.set_aq_eos("Aq", stability_tol=1e-20, max_iter=10, use_gmix=True)
        pr = flash_ev.eos["VL"]
        aq = flash_ev.eos["Aq"]

        flash_ev.init_flash(eos_order=["Aq", "VL"],
                            flash_type=DARTSFlash.FlashType.PTFlash,
                            )

        """ properties correlations """
        epsilon = self.zero / 10
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw,
                                               eps_z=epsilon)

        property_container.flash_ev = flash_ev
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

        state_spec = PhysicsBase.StateSpecification.PT
        # 1 p + (nc-1) z + 1 T
        ax_step = [0.0599] + [1e-4] * (len(components) - 1) + [0.028]
        ax_origin = [1.0] + [epsilon] * (len(components) - 1) + [220.0]
        self.physics = PhysicsBase(components, phases, self.timer,
                                     axes_step=ax_step, axes_origin=ax_origin,
                                     epsilon_z=epsilon, state_spec=state_spec, cache=False,
                                     extrapolation_flag=True)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        base_depth = self.reservoir.mesh.depth[0]
        boundary_state = {'H2O': 1 - self.zero, 'pressure': 100., 'temperature': 350}
        init = Initialize(physics=self.physics)

        # Solve boundary state
        X0 = init.solve_state(Xi=[100., 1.-self.zero, 350.],
                              specs=boundary_state,
                              )

        # Initialize depth table
        X, bc_idx = init.init_depth_table(depth_bottom=self.reservoir.global_data['depth'].max(),
                                          depth_top=self.reservoir.global_data['depth'].min(),
                                          depth_known=base_depth,
                                          X0=X0,
                                          nb=self.reservoir.nz,
                                          dTdh=0.03
                                          )

        # Solve vertical equilibrium in region below
        specs = {'H2O': 1 - self.zero}
        X = init.solve(X=X, bc_idx=bc_idx, specs=specs, downward=True).flatten()

        # Pass depth table
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
        Mw_avg = np.sum(self.physics.property_containers[0].Mw * inj_comp)
        inj_fluid_molar_potential_energy = inj_fluid_specific_potential_energy * Mw_avg
        inj_fluid_molar_energy = inj_fluid_molar_enthalpy + inj_fluid_molar_potential_energy

        injected_heat_rate = inj_rate * inj_fluid_molar_energy
        inj_flux = np.append(inj_flux, injected_heat_rate)

        cell_start_idx = cell_idx * self.physics.n_vars
        rhs_flux[cell_start_idx:cell_start_idx + self.physics.n_vars:] = - inj_flux  # inflow (e.g., injection) becomes minus for rhs

        return rhs_flux
