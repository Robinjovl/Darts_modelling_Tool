from darts.models.thmc_model import THMCModel
from reservoir import UnstructReservoirCustom, get_mesh_filename
import numpy as np
import os
from darts.engines import sim_params
from darts.reservoirs.mesh.transcalc import TransCalculations as TC
from darts.reservoirs.unstruct_reservoir_mech import get_bulk_modulus, get_rock_compressibility, get_isotropic_stiffness
from darts.reservoirs.unstruct_reservoir_mech import get_biot_modulus, bound_cond
from darts.input.input_data import InputData


class Model(THMCModel):
    def __init__(self, discretizer='mech_discretizer', case='mandel', mesh='rect'):
        self.case = case
        self.mesh = mesh
        self.discretizer_name = discretizer
        super().__init__()

    def set_solver(self):
        # data_ts is used only for linear solver params for PETSc
        self.data_ts = self.idata.sim.DataTS  # this needed as mech models have their own run_python implementation

        # Open-source FS-CPR by default for BOTH discretizers -- inject the spec;
        # the engine bypasses sim_params.linear_type. FS-CPR is a PRECONDITIONER
        # (single application), not an outer Krylov loop -- wrap it in GMRES to
        # mirror the proprietary path (bos_gmres + bos_fs_cpr).
        #
        # The two mechanics engines lay their Jacobian block out differently, and
        # the FS-CPR must be told which block is pressure and which is
        # displacement or it splits the wrong subsystem and never converges
        # (displacement residual plateaus, linear cap exhausts every Newton step,
        # dt collapses to ~0 -> the whole job hangs). engine_super_elastic_cpu
        # (mech_discretizer) uses the spec's default convention (P_VAR=0, Z_VAR=1,
        # U_VAR=NE); engine_pm_cpu (pm_discretizer) puts displacement first
        # (U_VAR=0), pressure at ND=3, and carries no composition variable
        # (Z_VAR=255 sentinel). Read the layout off the engine so this stays
        # correct if the conventions change.
        from darts.linear_solvers import LinearSolver
        from darts.linear_solvers.specs import FSCPRSolverSpec, GMRESSolverSpec
        engine = self.physics.engine
        mesh = self.reservoir.mesh
        n_blocks = mesh.n_blocks
        n_res_blks = mesh.n_res_blocks
        n_matrix = getattr(self.reservoir, 'n_matrix', n_res_blks)
        n_fracs_mesh = getattr(self.reservoir, 'n_fracs', 0)
        # Match proprietary engine_pm_cpu.cpp:136 convention:
        #   n_res  = n_matrix + n_fracs  (matrix + fracture cells treated as "reservoir")
        #   n_fracs= 0   (zero gap-DOF rows -- FS_UPG not yet supported)
        #   n_wells= n_blocks - n_res_blocks
        fs_cpr_kwargs = dict(
            force_amg_asymmetric=True,
            n_res=n_matrix + n_fracs_mesh,
            n_fracs=0,
            n_wells=n_blocks - n_res_blks,
        )
        if self.discretizer_name == 'pm_discretizer':
            # ND = 3 (3D mechanics); for the isothermal poroelastic cases the
            # flow subsystem is a single pressure equation (NE = N_VARS - ND = 1
            # = NC). Thermoporoelasticity (bai) is gated off upstream.
            fs_cpr_kwargs.update(
                p_var=engine.P_VAR,
                z_var=engine.Z_VAR,
                u_var=engine.U_VAR,
                nc=engine.N_VARS - 3,
            )
        fs_cpr = FSCPRSolverSpec(**fs_cpr_kwargs)
        if self.discretizer_name == 'mech_discretizer':
            lin_tol, lin_max_it = 1e-10, 5000
        else:
            lin_tol, lin_max_it = 1e-5, 50
        self.linear_solver = LinearSolver(model=self)
        self.linear_solver.spec = GMRESSolverSpec(
            prec=fs_cpr,
            tolerance=lin_tol,
            max_iterations=lin_max_it,
            restart=50,
            proprietary_linear_type=sim_params.cpu_gmres_fs_cpr,
        )
        super().set_solver()

    def set_reservoir(self):
        self.reservoir = UnstructReservoirCustom(timer=self.timer, idata=self.idata, case=self.case,
                                                 discretizer=self.discretizer_name, fluid_vars=self.physics.vars)

    def set_input_data(self):
        case = self.case  # short name
        if case == 'bai':
            type_hydr = 'thermal'
            type_mech = 'thermoporoelasticity'
        else:
            type_hydr = 'isothermal'
            type_mech = 'poroelasticity'  # Note: not supported with thermal
        self.idata = InputData(type_hydr=type_hydr, type_mech=type_mech, init_type='uniform')

        self.idata.other.case_name = self.case

        self.idata.rock.density = 2650.  # kg/m3
        self.idata.fluid.Mw = 18.015  # kg/kmol molar density (water)
        self.idata.fluid.density = self.idata.fluid.Mw
        self.idata.fluid.heat_capacity = 167.2 # [kJ/kg/K] the same as for the rock
        self.idata.fluid.heat_capacity *= self.idata.fluid.Mw / self.idata.fluid.density  # convert from [kJ/m3/K] to [kJ/kmol/K]
        self.idata.fluid.thermal_conductivity = 0. # it is not used in the mech. engines

        self.bc_type = bound_cond()  # get predefined constants for boundary conditions
        NO_FLOW = self.bc_type.NO_FLOW  # short name
        self.idata.mesh.bnd_tags = {}
        bnd_tags = self.idata.mesh.bnd_tags  # short name

        self.idata.mesh.mesh_filename = get_mesh_filename(self.mesh)

        self.idata.initial.initial_temperature = 0  # [K]
        self.idata.initial.initial_pressure = 0  # [bar]
        self.idata.initial.initial_displacements = [0., 0., 0.]  # [m]
        self.idata.initial.initial_composition = None  # not used in this test

        if 'box' in self.mesh or 'cylinder' in self.mesh:
            # 1 to 500 kN
            # 1 kilonewton/square meter	= 0.01 bar
            # R = 0.025 => area = 0.0196
            # 100 kN / area = 1 bar / area = 50 bar/m2
            self.idata.other.load_vertic = -100
            # 0 to 55 MPa (550 bars)
            self.idata.other.load_horiz = -50

            confining = self.bc_type.LOAD(self.idata.other.load_horiz, [0.0, 0.0, 0.0])
            p_init = self.idata.initial.initial_pressure
            f_top = self.bc_type.AQUIFER(p_init)

            inflow = False
            #inflow = True
            if inflow:
                vertic = self.bc_type.STUCK(0, [0,0,0])
                f_bottom = self.bc_type.AQUIFER(p_init + 0.5)
            else:
                vertic = self.bc_type.LOAD(self.idata.other.load_vertic, [0.0, 0.0, 0.0])
                f_bottom = self.bc_type.AQUIFER(p_init)

        bnd_tags['BND_X-'] = 991
        bnd_tags['BND_X+'] = 992
        bnd_tags['BND_Y-'] = 993
        bnd_tags['BND_Y+'] = 994
        bnd_tags['BND_Z-'] = 995
        bnd_tags['BND_Z+'] = 996
        self.idata.mesh.matrix_tags = [99991]

        self.idata.fluid.compressibility = 1.e-5
        if 'rect' in self.mesh:
            self.idata.fluid.compressibility = 1.e-10

        if case == 'mandel':
            self.idata.rock.porosity = 0.375
            self.idata.rock.perm = 10.0 / 9.81
            self.idata.rock.E = 10000  # in bars
            self.idata.rock.nu = 0.25
            self.idata.rock.biot = 0.9
            self.idata.rock.compressibility = get_rock_compressibility(
                kd=get_bulk_modulus(E=self.idata.rock.E, nu=self.idata.rock.nu),
                biot=self.idata.rock.biot, poro0=self.idata.rock.porosity)
            self.idata.fluid.viscosity = 1.0
            self.idata.other.Fa = -100.0  # bar * m

            self.idata.boundary = {}
            nf_r = {'flow': NO_FLOW, 'mech': self.bc_type.ROLLER}
            self.idata.boundary[bnd_tags['BND_X-']] = nf_r
            self.idata.boundary[bnd_tags['BND_X+']] = {'flow': self.bc_type.AQUIFER(self.idata.initial.initial_pressure), 'mech': self.bc_type.FREE}
            self.idata.boundary[bnd_tags['BND_Y-']] = nf_r
            self.idata.boundary[bnd_tags['BND_Y+']] = {'flow': NO_FLOW, 'mech': self.bc_type.STUCK_ROLLER(0.)}
            self.idata.boundary[bnd_tags['BND_Z-']] = nf_r
            self.idata.boundary[bnd_tags['BND_Z+']] = nf_r
        elif case == 'terzaghi':
            self.idata.rock.porosity = 0.375
            self.idata.rock.perm = 10.0 / 9.81
            self.idata.rock.E = 10000  # in bars
            self.idata.rock.nu = 0.25
            self.idata.rock.biot = 0.9
            self.idata.rock.compressibility = get_rock_compressibility(
                kd=get_bulk_modulus(E=self.idata.rock.E, nu=self.idata.rock.nu),
                biot=self.idata.rock.biot, poro0=self.idata.rock.porosity)
            self.idata.fluid.viscosity = 1.0

            self.idata.other.F = -100.0 # bar * m

            self.idata.boundary = {}
            nf_r = {'flow': NO_FLOW, 'mech': self.bc_type.ROLLER}
            self.idata.boundary[bnd_tags['BND_X-']] = nf_r
            self.idata.boundary[bnd_tags['BND_X+']] = {'flow': self.bc_type.AQUIFER(self.idata.initial.initial_pressure),
                                                       'mech': self.bc_type.LOAD(self.idata.other.F, [0.0, 0.0, 0.0])}
            self.idata.boundary[bnd_tags['BND_Y-']] = nf_r
            self.idata.boundary[bnd_tags['BND_Y+']] = nf_r
            self.idata.boundary[bnd_tags['BND_Z-']] = nf_r
            self.idata.boundary[bnd_tags['BND_Z+']] = nf_r
        elif case == 'terzaghi_two_layers':
            biot_1 = 0.9; biot_2 = 0.01
            poro_1 = 0.15; poro_2 = 0.001
            nu_1 = 0.15
            self.idata.rock.porosity = np.array([poro_1, poro_2])
            self.idata.rock.perm = 1.
            self.idata.rock.E = 10000  # in bars
            self.idata.rock.biot = np.array([biot_1, biot_2])
            self.idata.fluid.compressibility = 1.e-10
            self.idata.fluid.viscosity = 1.0
            self.idata.other.h = np.array([0.25, 0.75])  # geometric size of two layers
            # compute nu_2
            x = (biot_2 / biot_1 * ( 3 * (biot_1 - poro_1) * (1 - biot_1) * (1 - nu_1) / (1 + nu_1) + biot_1 ** 2) -
                 biot_2 ** 2) / 3 / (biot_2 - poro_2) / (1 - biot_2)
            nu_2 = (1 - x) / (1 + x)
            assert (nu_2 < 0.5 and nu_2 > 0)
            self.idata.rock.nu = np.array([nu_1, nu_2])
            self.idata.rock.compressibility = get_rock_compressibility(
                kd=get_bulk_modulus(E=self.idata.rock.E, nu=self.idata.rock.nu),
                biot=self.idata.rock.biot, poro0=self.idata.rock.porosity)
            self.idata.other.kd = get_bulk_modulus(self.idata.rock.E, self.idata.rock.nu)
            self.idata.other.M = get_biot_modulus(biot=self.idata.rock.biot, poro0=self.idata.rock.porosity,
                                                  kd=self.idata.other.kd, cf=self.idata.rock.compressibility)
            self.idata.make_prop_arrays()

            self.idata.other.F = -100.0 # bar * m

            self.idata.mesh.mesh_filename = get_mesh_filename(self.mesh, suffix='_two_layers')
            self.idata.mesh.matrix_tags = [99991, 99992]

            self.idata.boundary = {}
            nf_r = {'flow': NO_FLOW, 'mech': self.bc_type.ROLLER}
            self.idata.boundary[bnd_tags['BND_X-']] = nf_r
            self.idata.boundary[bnd_tags['BND_X+']] = {'flow': self.bc_type.AQUIFER(self.idata.initial.initial_pressure),
                                                       'mech': self.bc_type.LOAD(self.idata.other.F, [0.0, 0.0, 0.0])}
            self.idata.boundary[bnd_tags['BND_Y-']] = nf_r
            self.idata.boundary[bnd_tags['BND_Y+']] = nf_r
            self.idata.boundary[bnd_tags['BND_Z-']] = nf_r
            self.idata.boundary[bnd_tags['BND_Z+']] = nf_r
        elif case == 'bai':
            self.idata.rock.porosity = 0.2
            self.idata.rock.perm = 4.e+6 / 0.9869  # [mD]
            self.idata.rock.E = 0.06  # Young modulus [bars]
            self.idata.rock.nu = 0.4  # Poisson ratio
            self.idata.rock.biot = 1.0 # Biot coefficient
            self.idata.rock.compressibility = get_rock_compressibility(
                kd=get_bulk_modulus(E=self.idata.rock.E, nu=self.idata.rock.nu),
                biot=self.idata.rock.biot, poro0=self.idata.rock.porosity)
            self.idata.rock.th_expn = 9.0 * 1.E-7  # [1/K]
            self.idata.rock.th_expn *= get_bulk_modulus(E=self.idata.rock.E, nu=self.idata.rock.nu) # # Couchy book formula 4.19a, 4.21a
            self.idata.rock.thermal_conductivity = 0.836 * 86400.0 # [kJ/m/day/K]
            self.idata.rock.heat_capacity = 167.2 # [kJ/m3/K]
            self.idata.rock.th_expn_poro = 0.0   # mechanical term in porosity update
            self.idata.fluid.compressibility = 0.0
            self.idata.fluid.viscosity = 1.0  # [cP]

            self.idata.other.F = -1.e-5

            self.idata.mesh.mesh_filename = get_mesh_filename(self.mesh, suffix='_bai')

            self.idata.boundary = {}
            nf_r = {'flow': NO_FLOW, 'mech': self.bc_type.ROLLER, 'temp': NO_FLOW}
            self.idata.boundary[bnd_tags['BND_X-']] = nf_r
            self.idata.boundary[bnd_tags['BND_X+']] = nf_r
            self.idata.boundary[bnd_tags['BND_Y-']] = nf_r
            self.idata.boundary[bnd_tags['BND_Y+']] = {'flow': self.bc_type.AQUIFER(self.idata.initial.initial_pressure),
                                                       'mech': self.bc_type.LOAD(self.idata.other.F, [0.0, 0.0, 0.0]),
                                                       'temp': self.bc_type.AQUIFER(self.idata.initial.initial_temperature + 50)}
            self.idata.boundary[bnd_tags['BND_Z-']] = nf_r
            self.idata.boundary[bnd_tags['BND_Z+']] = nf_r
        self.idata.rock.stiffness = get_isotropic_stiffness(self.idata.rock.E, self.idata.rock.nu)

        if case == 'terzaghi_two_layers':
            # short names
            b = self.idata.rock.biot; nu = self.idata.rock.nu; E = self.idata.rock.E;
            M = get_biot_modulus(biot=b, poro0=self.idata.rock.porosity, cf=self.idata.fluid.compressibility,
                                 kd=get_bulk_modulus(E=self.idata.rock.E, nu=self.idata.rock.nu));
            # some numbers for analytics
            self.idata.other.m = m = (1 + nu) * (1 - 2 * nu) / E / (1 - nu)
            self.idata.other.skempton = b * m * M / (1 + b ** 2 * m * M)
            self.idata.other.c = TC.darcy_constant * self.idata.rock.perm / self.idata.fluid.viscosity * M / (1 + b ** 2 * m * M)
            assert (np.fabs(self.idata.other.skempton[1] - self.idata.other.skempton[0]) < 1.e-6)

        if case == 'bai':
            nt = 60
            max_dt = 0.1
            self.idata.sim.time_steps = np.logspace(-7, np.log10(max_dt), nt)
        else:
            nt = 60  # number of timesteps
            max_dt = 30  # timestep length, days
            self.idata.sim.time_steps = np.logspace(-3, np.log10(max_dt), nt)

        # optional: use PETSc / Pardiso linear solver (set in set_solver())
        #   from darts.linear_solvers import PETScSolverSpec, PardisoSolverSpec
        #   self.linear_solver.spec = PETScSolverSpec(variant="fs")
        #   self.linear_solver.spec = PardisoSolverSpec()
        from darts.models.darts_model import DataTS
        self.idata.sim.DataTS = DataTS(n_vars=0)

        self.idata.obl.zero = 1e-9
        self.idata.obl.epsilon_z = 1e-10
        self.idata.obl.p_step = 1.0
        self.idata.obl.p_origin = -5.
        self.idata.obl.z_step = 2e-3
        self.idata.obl.z_origin = 0.
        self.idata.obl.t_step = 0.25
        self.idata.obl.t_origin = -10.

        super().set_input_data()  # check
