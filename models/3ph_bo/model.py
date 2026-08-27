from darts.input.input_data import InputData
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.physics.base.property_container import PropertyContainer
from darts.engines import ms_well
from darts.nonlinear_solvers import NewtonSolver

from darts.physics.properties.black_oil import *
from darts.physics.blackoil import BlackOil, BlackOilFluidProps

# Model class creation here!
class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        idata = self.set_input_data('')
        self.set_physics(idata)

        # Solver/time-stepping configuration moved to set_solver() (called from base reset())

        self.timer.node["initialization"].stop()

    def set_solver(self):
        self.linear_solver.set_sim_params(first_ts=1e-6, mult_ts=2, max_ts=10, runtime=100 )
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-3, max_iterations=10)
        self.linear_solver.spec.tolerance = 1e-7
        self.linear_solver.spec.max_iterations = 50

    def set_reservoir(self):
        """Reservoir"""
        (nx, ny, nz) = (10, 10, 3)
        kx = np.array([500] * 100 + [50] * 100 + [200] * 100)
        ky = kx
        kz = np.array([80] * 100 + [42] * 100 + [20] * 100)
        dz = np.array([6] * 100 + [9] * 100 + [15] * 100)
        depth = np.array([2540] * 100 + [2548] * 100 + [2560] * 100)
        dx = 3000 / nx
        dy = 3000 / ny

        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                         permx=kx, permy=ky, permz=kz, poro=0.3, depth=depth)
        return

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", res_cell_idx=(1, 1, 1))
        self.reservoir.add_well("P1")
        self.reservoir.add_perforation("P1", res_cell_idx=(10, 10, 3))

    def set_physics(self, idata: InputData):
        self.physics = BlackOil(idata, self.timer, thermal=False)
        zero = 1e-12
        # epsilon = 1e-13
        self.inj_composition = [1 - 2 * zero, zero]
        self.ini_stream = [0.001225901537, 0.7711341309]

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 330.,
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
                                               is_inj=True, target=400., inj_composition=self.inj_composition)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=70.)


    def set_input_data(self, case):
        idata = InputData(type_hydr='isothermal', type_mech='none', init_type='uniform')
        pvt = 'physics.in'
        # this sets default properties
        idata.fluid = BlackOilFluidProps(pvt=pvt)
        # example - how to change the properties
        # idata.fluid.density['water'] = DensityBasic(compr=1e-5, dens0=1014)

        idata.obl.zero = 1e-12
        idata.obl.epsilon_z = 1e-13
        idata.obl.p_step = 0.0898   # bar
        idata.obl.p_origin = 1.0
        idata.obl.z_step = 2e-4
        idata.obl.z_origin = idata.obl.epsilon_z
        idata.obl.t_step = 0.05
        idata.obl.t_origin = -10.0

        return idata
