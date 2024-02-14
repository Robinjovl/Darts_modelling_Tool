from darts.models.one_phase_thermoporoelastic import OnePhaseThermoPoroElasticModel
from reservoir import UnstructReservoirCustom
import numpy as np
from darts.input.input_data import InputData

class Model(OnePhaseThermoPoroElasticModel):
    def __init__(self, mode, mesh_filename, n_points=64, discretizer='mech_discretizer'):
        self.mode = mode
        self.mesh_filename = mesh_filename
        self.discretizer_name = discretizer
        self.physics_type = 'poromechanics'  # folder name for vtk output
        super().__init__(n_points=n_points, discretizer=discretizer)

    def set_reservoir(self):
        self.reservoir = UnstructReservoirCustom(timer=self.timer, idata=self.idata, discretizer=self.discretizer_name,
                                                 mode=self.mode, mesh_filename=self.mesh_filename)
    def set_input_data(self):
        if self.mode == 'thermoporoelastic':
            type_hydr = 'thermal'
            type_mech = 'thermoporoelasticity'
        elif self.mode == 'poroelastic':
            type_hydr = 'isothermal'
            type_mech = 'poroelasticity'  # Note: not supported with thermal
        self.idata = InputData(type_hydr=type_hydr, type_mech=type_mech)

        self.idata.rock.compressibility = 1.

        self.idata.rock.porosity = 0.1
        self.idata.rock.perm = [1.5,    0.5,    0.35,
                                0.5,    1.5,    0.45,
                                0.35,   0.45,   1.5]
        self.idata.rock.E = 0.06  # in bars
        self.idata.rock.nu = 0.4
        self.idata.rock.biot = [1.5,    0.1,    0.5,
                                0.1,    1.5,    0.15,

                                0.5,    0.15,   1.5]
        self.idata.rock.stiffness = [1.323, 0.0726, 0.263, 0.108, -0.08, -0.239,
                                     0.0726, 1.276, -0.318, 0.383, 0.108, 0.501,
                                     0.263, -0.318, 0.943, -0.183, 0.146, 0.182,
                                     0.108, 0.383, -0.183, 1.517, -0.0127, -0.304,
                                     -0.08, 0.108, 0.146, -0.0127, 1.209, -0.326,
                                     -0.239, 0.501, 0.182, -0.304, -0.326, 1.373]
        self.idata.rock.kd_cur = 0. # TODO: why parent ask self.th_expn_coef and self.kd_cur?

        if self.mode == 'thermoporoelastic':
            self.idata.rock.th_expn =  [1.5,    0.5,    0.35,
                                        0.5,    1.5,    0.45,
                                        0.35,   0.45,   1.5]
            self.idata.rock.conductivity = 0.836 * 86400.0 * 1000
            self.idata.rock.th_expn_poro = 0.0  # mechanical term in porosity update
            self.idata.rock.heat_capacity = 1.0
            self.idata.rock.conductivity = 1.e+6 * np.array([1.5, 0.1, 0.5,
                                                             0.1, 1.5, 0.15,
                                                             0.5, 0.15, 1.5])

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


    def init(self):
        super().init()

        if self.reservoir.thermoporoelasticity:
            vol_strain_trans = np.array(self.reservoir.mesh.vol_strain_tran, copy=False)
            vol_strain_rhs = np.array(self.reservoir.mesh.vol_strain_rhs, copy=False)
            vol_strain_trans[:] = 0.0
            vol_strain_rhs[:] = 0.0

        Xref = np.array(self.engine.Xref, copy=False)
        Xn_ref = np.array(self.engine.Xn_ref, copy=False)
        Xref[:] = 0.0
        Xn_ref[:] = 0.0

    def set_initial_conditions(self):
        if self.reservoir.thermoporoelasticity:
            self.physics.set_nonuniform_initial_conditions(self.reservoir.mesh,
                                                           initial_pressure=self.reservoir.p_init,
                                                           initial_temperature=self.reservoir.t_init,
                                                           initial_displacement=[0.0, 0.0, 0.0])
        else:
            self.physics.set_nonuniform_initial_conditions(self.reservoir.mesh,
                                                           initial_pressure=self.reservoir.p_init,
                                                           initial_displacement=self.reservoir.u_init)
        return 0