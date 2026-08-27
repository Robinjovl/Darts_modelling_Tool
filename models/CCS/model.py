import numpy as np
from darts.models.darts_model import DartsModel
from darts.engines import ms_well

from darts.physics.base.physics import PhysicsBase
from darts.physics.eos_physics import EoSPhysics
from darts.physics.base.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from darts.nonlinear_solvers import NewtonSolver


class Model(DartsModel):
    def __init__(self, logspace: bool = True):
        # Call base class constructor
        super().__init__()

        if logspace:
            dr = 1.
            R1 = 1000.
            nz = 24

            poro = np.ones(nz) * 0.001
            perm = np.ones(nz) * 0.001
            poro[4:20] = 0.2
            perm[4:20] = 20
            perm[6:12] = 100

            self.set_reservoir(dr=dr, R1=R1, logspace=logspace, nz=nz, dz=5, poro=poro, perm=perm)

        else:
            # lower detfurth
            (nr, nz) = (10, 24)
            dr = 1.
            R1 = nr * dr

            if 0:
                poro = 0.2
                perm = 100

            elif 1:
                poro = np.ones((nr, nz)) * 0.001
                perm = np.ones((nr, nz)) * 0.001
                poro[:, 4:20] = 0.2
                perm[:, 4:20] = 20
                perm[:, 6:12] = 100

            else:
                poro = np.ones((nr, nz)) * 0.075
                perm = np.ones((nr, nz)) * 0.29
                # upper detfurth
                perm[:, :16] = 12.6
                # hardegsen
                poro[:, :10] = 0.09
                perm[:, :10] = 24
                # hardegsen high perm
                poro[:, :8] = 0.2
                perm[:, :8] = 550
                # hardegsen
                poro[:, :6] = 0.09
                perm[:, :6] = 24
                # caprock
                poro[:, :4] = 0.01
                perm[:, :4] = 0.01

            self.set_reservoir(dr=dr, R1=R1, logspace=False, nz=nz, dz=5, poro=poro, perm=perm)

        # Specify initial and injection conditions
        self.p_init = 100.
        self.p_inj = self.p_init + 1.
        self.t_init = 350.
        self.t_inj = 300.
        self.swc = 0.25

        zero = 1e-12
        self.set_physics(zero, temperature=None, ph=False, vl_phases=False)
        self.inj_stream = [0.001] if self.components[0] == "H2O" else [0.999]

        # Solver/time-stepping configuration moved to set_solver() (called from base reset()).

    def set_solver(self):
        if self.linear_solver is None:
            from darts.linear_solvers import LinearSolver
            self.linear_solver = LinearSolver(model=self)
        self.linear_solver.set_sim_params(first_ts=1e-7, mult_ts=2, max_ts=20.,   runtime=1,
                            #   # Type of newton method (related to chopping strategy?)
                            #   # Probably chop-criteria(?)
                            )
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-6, max_iterations=8)
        self.linear_solver.spec.tolerance = 1e-6
        self.linear_solver.spec.max_iterations = 50
        # self.params.nonlinear_norm_type = self.params.L1
        # self.params.linear_type = self.params.cpu_superlu

    def set_reservoir(self, dr, R1, logspace, nz, dz, poro, perm):
        self.seg_ratio = 1

        from darts.reservoirs.struct_radial_reservoir import StructRadialReservoir
        self.reservoir = StructRadialReservoir(self.timer, nz=nz, dr=dr, dz=dz, permr=perm, permz=perm/10, poro=poro,
                                               hcap=2200, rcond=100, R1=R1, nr=40, logspace=logspace, boundary_volume=4.3e7)
        # self.reservoir.well_dict = {'I1': [(1, 1, k+1) for k in range(1, self.reservoir.nz-1)]}
        # self.reservoir.well_dict = {'I1': [(1, 1, k + 1) for k in range(10, 20)]}
        # self.reservoir.well_dict = {'I1': [(1, 1, self.reservoir.nz)]}
        #self.boundary_flux = True

        self.reservoir.boundary_volumes['xy_plus'] = 1e10
        self.reservoir.boundary_volumes['xy_minus'] = 1e10

        return

    def set_physics(self, zero, temperature: float = None, ph: bool = False, vl_phases: bool = False):
        """Physical properties"""
        self.zero = zero
        epsilon = zero/10

        from dartsflash.libflash import EoSParams, EoS
        from dartsflash.components import CompData
        from dartsflash.mixtures import DARTSFlash, Mixture
        # Fluid components, ions and solid
        components = ["H2O", "CO2"]
        self.components = components
        phases = ["Aq", "V", "L"] if vl_phases else ["Aq", "V"]
        comp_data = CompData(components, setprops=True)
        nc = len(components)

        """ Define state specification and initialize Physics object """
        if temperature is None:  # if None, then thermal=True
            state_spec = PhysicsBase.StateSpecification.PH if ph else PhysicsBase.StateSpecification.PT
        else:
            state_spec = PhysicsBase.StateSpecification.P

        # [p, z_1, ..., z_{nc-1}, T?]
        nz = len(components) - 1
        ax_step = [0.399] + [1e-3] * nz
        ax_origin = [1.0] + [epsilon] * nz
        if state_spec >= PhysicsBase.StateSpecification.PT:
            ax_step.append(0.1)
            ax_origin.append(273.15)
        self.physics = EoSPhysics(components, phases, self.timer,
                                  axes_step=ax_step, axes_origin=ax_origin,
                                  epsilon_z=epsilon, state_spec=state_spec, cache=False,
                                  extrapolation_flag=True)

        """ Define flash """
        mixture = Mixture(comp_data)
        mixture.set_vl_eos(vl_eos_name="VL", hybrid_aq_eos_name="Aq",
                           root_order=[EoS.MAX, EoS.MIN] if vl_phases else [EoS.STABLE],
                           trial_comps=[EoSParams.Yi.Wilson, 1],
                           stability_tol=1e-20, switch_tol=1e-2, max_iter=50, use_gmix=False
                           )
        mixture.set_aq_eos(aq_eos_name="Aq", stability_tol=1e-20, max_iter=10, use_gmix=True)

        mixture.init_flash(flash_type=DARTSFlash.FlashType.PHFlash if ph else DARTSFlash.FlashType.PTFlash,
                            eos_order=["Aq", "VL"],
                            t_min=270., t_max=500., t_init=300.,
                            # pxflash_switch_ttol=1e-3, near_zero_px=1e-2,
                            )
        self.physics.set_mixture(mixture)

        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw,
                                               temperature=temperature, eps_z=epsilon)
        self.physics.add_property_region(property_container)

        property_container.flash_ev = self.physics.get_flash_ev()
        property_container.density_ev = dict([('V', EoSDensity(eos=mixture.eos["VL"], root_flag=EoS.MAX if vl_phases else EoS.STABLE)),
                                              ('L', EoSDensity(eos=mixture.eos["VL"], root_flag=EoS.MIN)),
                                              ('Aq', Garcia2001(components, ions=None, combined_ions=None)), ])
        property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                ('L', Fenghour1998()),
                                                ('Aq', Islam2012(components, ions=None, combined_ions=None))])
        diff = 8.64e-6
        property_container.diffusion_ev = dict([('V', ConstFunc(np.ones(nc) * diff)),
                                                ('L', ConstFunc(np.ones(nc) * diff)),
                                                ('Aq', ConstFunc(np.ones(nc) * diff * 1e-3))])

        # property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=mixture.eos["VL"], root_flag=EoS.MAX if vl_phases else EoS.STABLE)),
        #                                        ('L', EoSEnthalpy(eos=mixture.eos["VL"], root_flag=EoS.MIN)),
        #                                        ('Aq', EoSEnthalpy(eos=mixture.eos["Aq"])), ])
        property_container.enthalpy_ev = dict([('V', self.physics.get_enthalpy_ev_from_flash(phase_idx=1)),
                                               ('L', self.physics.get_enthalpy_ev_from_flash(phase_idx=2)),
                                               ('Aq', self.physics.get_enthalpy_ev_from_flash(phase_idx=0)), ])

        property_container.conductivity_ev = dict([('V', ConstFunc(10.)),
                                                   ('L', ConstFunc(10.)),
                                                   ('Aq', ConstFunc(180.)), ])

        property_container.rel_perm_ev = dict([('V', PhaseRelPerm("gas", swc=self.swc, sgr=self.swc, n=1.5)),
                                               ('L', PhaseRelPerm("oil", swc=self.swc, sgr=self.swc, n=1.5)),
                                               ('Aq', PhaseRelPerm("wat", swc=self.swc, sgr=self.swc, n=4))])

        property_container.output_props = {'temperature': lambda: property_container.temperature,
                                           'satA': lambda: property_container.sat[phases.index("Aq")],
                                           'satV': lambda: property_container.sat[phases.index("V")],
                                           'satL': lambda: property_container.sat[phases.index("L")] if vl_phases else 0.,
                                           'rho_a': lambda: property_container.dens[phases.index("Aq")],
                                           'rho_g': lambda: property_container.dens[phases.index("V")],
                                           }

        return

    def set_initial_conditions(self):
        if 1:
            from darts.physics.base.initialize import Initialize
            init = Initialize(physics=self.physics)

            # Solve boundary state
            boundary_state = {'H2O': 1 - self.zero, 'pressure': 100., 'temperature': 350.}
            X0 = init.solve_state(Xi=[boundary_state['pressure'], 1 - self.zero, boundary_state['temperature']],
                                  specs=boundary_state,
                                  )

            # Initialize depth table
            nb = 100
            min_depth = self.reservoir.global_data['depth'].min()
            max_depth = self.reservoir.global_data['depth'].max()
            b_depth = min_depth + (max_depth - min_depth) / 4.
            X, bc_idx = init.init_depth_table(depth_bottom=max_depth,
                                              depth_top=min_depth,
                                              depth_known=b_depth,
                                              X0=X0,
                                              nb=nb,
                                              dTdh=0.03
                                              )

            # Solve vertical equilibrium
            specs = {'H2O': 1. - self.zero}
            X = init.solve(X=X, bc_idx=bc_idx, specs=specs, downward=False)  # solve above
            X = init.solve(X=X, bc_idx=bc_idx, specs=specs, downward=True)  # solve below

            # assign initial condition with evaluated initialized properties
            self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh, input_depth=init.depths,
                                                                 input_distribution={var: X[:, i] for i, var in
                                                                                     enumerate(self.physics.vars)})
        else:
            input_distribution = {self.physics.vars[0]: 100.,
                                  self.physics.vars[1]: 0.99995,
                                  self.physics.vars[2]: 350.,
                                  }
            return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                                  input_distribution=input_distribution)

    def set_wells(self):
        if 1:
            self.reservoir.add_well("I1")
            for k in range(4, 20):
            # for k in range(24):
                self.reservoir.add_perforation("I1", res_cell_idx=(1, 1, k + 1), well_indexD=0)
                # self.reservoir.add_perforation("I1", res_cell_idx=(1, 1, k + 1), well_indexD=0, ms_epm=True)

        if 0:
            self.reservoir.add_well("P1")
            for k in range(4, 20):
                self.reservoir.add_perforation("P1", res_cell_idx=(1, 1, k + 1), well_index=100, well_indexD=100)

    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if 'I' in w.name:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=self.p_inj, inj_composition=self.inj_stream,
                                               inj_temp=self.t_inj)
                # self.physics.set_well_controls(wctrl=w.control, is_inj=True,
                #                                control_type=well_control_iface.BHP, target=self.p_inj,
                #                                # control_type=well_control_iface.MOLAR_RATE, target=4000.,
                #                                inj_composition=self.inj_stream,
                #                                inj_temp=self.t_inj)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=self.p_prod)
