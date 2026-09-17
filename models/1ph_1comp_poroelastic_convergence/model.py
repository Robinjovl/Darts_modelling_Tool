from darts.models.thmc_model import THMCModel
from reservoir import UnstructReservoirCustom
from darts.reservoirs.unstruct_reservoir_mech import bound_cond
import numpy as np
import os
from darts.input.input_data import InputData
from darts.engines import value_vector, sim_params, mech_operators

class Model(THMCModel):
    def __init__(self, mode, mesh_filename, discretizer='mech_discretizer', heat_cond_mult=1.):
        self.mode = mode
        self.mesh_filename = mesh_filename
        self.discretizer_name = discretizer
        self.physics_type = 'poromechanics'  # folder name for vtk output
        self.heat_cond_mult = heat_cond_mult
        super().__init__()

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        if self.mode == 'thermoporoelastic':
            vol_strain_trans = np.array(self.reservoir.mesh.vol_strain_tran, copy=False)
            vol_strain_rhs = np.array(self.reservoir.mesh.vol_strain_rhs, copy=False)
            vol_strain_trans[:] = 0.0
            vol_strain_rhs[:] = 0.0

        Xref = np.array(self.physics.engine.Xref, copy=False)
        Xn_ref = np.array(self.physics.engine.Xn_ref, copy=False)
        Xref[:] = 0.0
        Xn_ref[:] = 0.0

    def set_solver(self):
        # Open-source FS-CPR by default -- inject the spec; the engine bypasses
        # sim_params.linear_type. FS-CPR is a PRECONDITIONER (single application),
        # not an outer Krylov loop -- wrap it in GMRES to mirror the proprietary
        # path (bos_gmres + bos_fs_cpr).
        from darts.linear_solvers.specs import FSCPRSolverSpec, GMRESSolverSpec
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
            # engine_pm_cpu lays the block out as U_VAR=0, P_VAR=ND, Z_VAR=255 --
            # not the spec's default (P_VAR=0, Z_VAR=1, U_VAR=NE). Without this
            # FS-CPR splits the wrong subsystem and preconditions poorly (the
            # analytics model passes the same four fields for the same reason).
            engine = self.physics.engine
            fs_cpr_kwargs.update(
                p_var=engine.P_VAR,
                z_var=engine.Z_VAR,
                u_var=engine.U_VAR,
                nc=engine.get_n_vars() - 3,
            )
        fs_cpr = FSCPRSolverSpec(**fs_cpr_kwargs)
        # Single solver declaration. The FS-CPR spec drives _apply_solver on the
        # open-source CPU build. On the proprietary build _apply_solver applies
        # proprietary_linear_type (bos_fs_cpr) to params.linear_type -- but only for
        # mech_discretizer; pm_discretizer keeps its mechanics multi-stage backend
        # (engine.ls_params), so its spec carries no proprietary fallback (None).
        # The spec is set here, before super().set_solver() below, so the platform
        # default is never materialized.
        self.linear_solver.spec = GMRESSolverSpec(
            prec=fs_cpr,
            # NOTE: 1e-5 / 50 are the values this model has always effectively run with.
            # Until !280 the engine overwrote a spec's tolerance/max_iterations at init()
            # with sim_params (defaults 1e-5 / 50, globals.h:117), so the spec's numbers were
            # decorative. The spec is authoritative now, so state the values the model has
            # really been running -- keeping behaviour unchanged. FS-CPR does not reach 1e-8 on
            # these systems anyway: asking for it only burns the iteration budget (on SPE10_mech
            # 22 of 48 solves exhaust the 200-iteration cap; 99 vs 41 linear iters per Newton).
            tolerance=1e-5,
            max_iterations=50,
            restart=50,
            proprietary_linear_type=(sim_params.cpu_gmres_fs_cpr
                                     if self.discretizer_name == 'mech_discretizer' else None),
        )
        super().set_solver()
        if self.discretizer_name == 'pm_discretizer':
            self.physics.engine.ls_params[-1].linear_type = (
                sim_params.cpu_superlu if self.linear_solver.open_source_solvers_available()
                else sim_params.cpu_gmres_fs_cpr)

    def set_reservoir(self):
        self.reservoir = UnstructReservoirCustom(timer=self.timer, idata=self.idata, discretizer=self.discretizer_name,
                                                 fluid_vars=self.physics.vars, mode=self.mode, mesh_filename=self.mesh_filename)

    def set_input_data(self):
        if self.mode == 'thermoporoelastic':
            type_hydr = 'thermal'
            type_mech = 'thermoporoelasticity'
        elif self.mode == 'poroelastic':
            type_hydr = 'isothermal'
            type_mech = 'poroelasticity'  # Note: not supported with thermal
        self.idata = InputData(type_hydr=type_hydr, type_mech=type_mech, init_type='uniform')

        self.bc_type = bound_cond()  # get predefined constants for boundary conditions

        self.idata.mesh.bnd_tags = {}
        bnd_tags = self.idata.mesh.bnd_tags  # short name
        bnd_tags['BND_X-'] = 991
        bnd_tags['BND_X+'] = 992
        bnd_tags['BND_Y-'] = 993
        bnd_tags['BND_Y+'] = 994
        bnd_tags['BND_Z-'] = 995
        bnd_tags['BND_Z+'] = 996
        self.idata.mesh.matrix_tags = [99991]

        self.idata.boundary = {}
        nf_s = {'flow': self.bc_type.AQUIFER(0), 'temp': self.bc_type.AQUIFER(0), 'mech': self.bc_type.STUCK(0.0, [0.0, 0.0, 0.0])}
        self.idata.boundary[bnd_tags['BND_X-']] = nf_s
        self.idata.boundary[bnd_tags['BND_X+']] = nf_s
        self.idata.boundary[bnd_tags['BND_Y-']] = nf_s
        self.idata.boundary[bnd_tags['BND_Y+']] = nf_s
        self.idata.boundary[bnd_tags['BND_Z-']] = nf_s
        self.idata.boundary[bnd_tags['BND_Z+']] = nf_s

        self.idata.rock.density = 2650.0
        self.idata.rock.porosity = 0.1
        self.idata.rock.perm = [1.5,    0.5,    0.35,
                                0.5,    1.5,    0.45,
                                0.35,   0.45,   1.5]
        self.idata.rock.biot = [1.5,    0.1,    0.5,
                                0.1,    1.5,    0.15,
                                0.5,    0.15,   1.5]
        self.idata.rock.stiffness = [1.323, 0.0726, 0.263, 0.108, -0.08, -0.239,
                                     0.0726, 1.276, -0.318, 0.383, 0.108, 0.501,
                                     0.263, -0.318, 0.943, -0.183, 0.146, 0.182,
                                     0.108, 0.383, -0.183, 1.517, -0.0127, -0.304,
                                     -0.08, 0.108, 0.146, -0.0127, 1.209, -0.326,
                                     -0.239, 0.501, 0.182, -0.304, -0.326, 1.373]

        if self.mode == 'thermoporoelastic':
            self.idata.rock.compressibility = 0.
            self.idata.rock.th_expn =  [1.5,    0.5,    0.35,
                                        0.5,    1.5,    0.45,
                                        0.35,   0.45,   1.5]
            self.idata.rock.th_expn_poro = 0.0  # mechanical term in porosity update
            self.idata.rock.heat_capacity = 1.0
            self.idata.rock.thermal_conductivity = self.heat_cond_mult * 1.e+6 * np.array([1.5, 0.1, 0.5,
                                                             0.1, 1.5, 0.15,
                                                             0.5, 0.15, 1.5])
            self.idata.rock.compressibility = 0.0
        else:
            self.idata.rock.compressibility = self.idata.rock.porosity * 1.4503768e-05

        self.idata.fluid.compressibility = 0.0
        self.idata.fluid.viscosity = 1e-2
        self.idata.fluid.Mw = 1.0
        self.idata.fluid.density = 978.0
        if self.mode == 'thermoporoelastic':
            self.idata.fluid.heat_capacity = self.idata.rock.heat_capacity # [kJ/kg/K] the same as for the rock
            #self.idata.fluid.heat_capacity *= self.idata.fluid.Mw / self.idata.fluid.density  # convert from [kJ/m3/K] to [kJ/kmol/K]
        else:
            self.idata.fluid.heat_capacity = 0.
        self.idata.fluid.thermal_conductivity = 0. # it is not used in the mech. engines

        self.idata.obl.zero = 1e-9
        self.idata.obl.epsilon_z = 1e-10
        self.idata.obl.p_step = 2.0
        self.idata.obl.p_origin = -500.0
        self.idata.obl.z_step = 2e-3
        self.idata.obl.z_origin = self.idata.obl.epsilon_z
        self.idata.obl.t_step = 0.4
        self.idata.obl.t_origin = -100.0

        super().set_input_data()

    def set_initial_conditions(self):
        input_distribution = {'pressure': self.reservoir.p_init}
        input_distribution.update({comp: self.reservoir.z_init[i] for i, comp in enumerate(self.physics.components[:-1])})
        if self.reservoir.thermoporoelasticity:
            input_distribution['temperature'] = self.reservoir.t_init
            input_displacement = [0.0, 0.0, 0.0]
        else:
            input_displacement = self.reservoir.u_init

        self.physics.set_initial_conditions_from_array(self.reservoir.mesh,
                                                       input_distribution=input_distribution,
                                                       input_displacement=input_displacement)
        return 0
