from darts.models.thmc_model import THMCModel
from reservoir import UnstructReservoirCustom
from darts.reservoirs.unstruct_reservoir_mech import bound_cond
import numpy as np
import os
from darts.input.input_data import InputData
from darts.engines import value_vector, sim_params, mech_operators

class Model(THMCModel):
    def __init__(self, mode, mesh_filename, n_points=64, discretizer='mech_discretizer', heat_cond_mult=1.):
        self.mode = mode
        self.mesh_filename = mesh_filename
        self.discretizer_name = discretizer
        self.physics_type = 'poromechanics'  # folder name for vtk output
        self.heat_cond_mult = heat_cond_mult
        super().__init__(n_points=n_points, discretizer=discretizer)

    def init(self):
        super().init()
        if self.mode == 'thermoporoelastic':
            vol_strain_trans = np.array(self.reservoir.mesh.vol_strain_tran, copy=False)
            vol_strain_rhs = np.array(self.reservoir.mesh.vol_strain_rhs, copy=False)
            vol_strain_trans[:] = 0.0
            vol_strain_rhs[:] = 0.0

        Xref = np.array(self.physics.engine.Xref, copy=False)
        Xn_ref = np.array(self.physics.engine.Xn_ref, copy=False)
        Xref[:] = 0.0
        Xn_ref[:] = 0.0

    def set_solver_params(self):
        super().set_solver_params()
        if os.getenv('ODLS') != None and os.getenv('ODLS') == '-a':
            linear_type = sim_params.cpu_gmres_fs_cpr
        else:
            linear_type = sim_params.cpu_superlu

        if self.discretizer_name == 'mech_discretizer':
            self.params.linear_type = linear_type
        elif self.discretizer_name == 'pm_discretizer':
            self.physics.engine.ls_params[-1].linear_type = linear_type
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
            self.idata.rock.conductivity = self.heat_cond_mult * 1.e+6 * np.array([1.5, 0.1, 0.5,
                                                             0.1, 1.5, 0.15,
                                                             0.5, 0.15, 1.5])
            self.idata.rock.compressibility = 0.0
        else:
            self.idata.rock.compressibility = self.idata.rock.porosity * 1.4503768e-05 

        self.idata.fluid.compressibility = 0.0
        self.idata.fluid.viscosity = 1e-2
        self.idata.fluid.Mw = 1.0
        self.idata.fluid.density = 978.0

        self.idata.obl.n_points = 500
        self.idata.obl.zero = 1e-9
        self.idata.obl.min_p = -500.
        self.idata.obl.max_p = 500.
        self.idata.obl.min_t = -100.
        self.idata.obl.max_t = 100.
        self.idata.obl.min_z = self.idata.obl.zero
        self.idata.obl.max_z = 1 - self.idata.obl.zero

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
