from darts.models.darts_model import DartsModel
from darts.engines import value_vector, ms_well
from darts.nonlinear_solvers import NewtonSolver
import numpy as np

from darts.physics.base.physics import PhysicsBase
from darts.physics.dead_oil import DeadOilProperties

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.enthalpy import EnthalpyBasic

from reservoir import UnstructReservoir

class Model(DartsModel):
    def __init__(self, discr_type='mpfa', mesh_file='meshes/wedge.msh'):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.discr_type = discr_type
        self.physics_type = 'dead_oil'

        self.set_physics()
        self.set_reservoir(mesh_file)

        # solver/time-stepping config moved to set_solver() (called at top of reset())
        self.timer.node["initialization"].stop()

    def set_solver(self):
        self.ts_control.dt_first = 1e-4
        self.ts_control.dt_min = 1e-15
        self.ts_control.dt_mult = 2
        self.ts_control.dt_max = 5
        self.ts_control.runtime = 1000
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-3)
        self.linear_solver.spec.tolerance = 1e-6

    def init(self, platform='cpu'):
        DartsModel.init(self, discr_type=self.discr_type, platform=platform)

    def set_reservoir(self, mesh_file):
        self.reservoir = UnstructReservoir(self.discr_type, mesh_file, n_vars=self.physics.n_vars)

        hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)
        hcap.fill(2200)
        rcond.fill(181.44)

        if self.discr_type == 'mpfa':
            self.reservoir.mesh.pz_bounds.resize(self.physics.n_vars * self.reservoir.n_bounds)
            pz_bounds = np.array(self.reservoir.mesh.pz_bounds, copy=False)
            pz_bounds[::self.reservoir.n_vars] = self.p_init
            for i in range(1, self.physics.nc):
                pz_bounds[i::self.reservoir.n_vars] = self.ini[i-1]

            pz_bounds[self.physics.n_vars-1::self.reservoir.n_vars] = self.init_temp

            self.reservoir.P_VAR = 0

    def set_wells(self):
        # well model or boundary conditions
        if self.discr_type == 'mpfa':
            Lx = max([pt.values[0] for pt in self.reservoir.discr_mesh.nodes])
            Ly = max([pt.values[1] for pt in self.reservoir.discr_mesh.nodes])
            Lz = max([pt.values[2] for pt in self.reservoir.discr_mesh.nodes])
            dx = np.sqrt(np.mean(self.reservoir.volume_all_cells) / \
                    ( max([pt.values[2] for pt in self.reservoir.discr_mesh.nodes]) - min([pt.values[2] for pt in self.reservoir.discr_mesh.nodes])))

            n_cells = self.reservoir.discr_mesh.n_cells
            pt_x = np.array([c.values[0] for c in self.reservoir.discr_mesh.centroids])[:n_cells]
            pt_y = np.array([c.values[1] for c in self.reservoir.discr_mesh.centroids])[:n_cells]
            pt_z = np.array([c.values[2] for c in self.reservoir.discr_mesh.centroids])[:n_cells]

            x0 = 0.1 * Lx
            y0 = 0.1 * Ly
            self.id1 = ((pt_x - x0) ** 2 + (pt_y - y0) ** 2 + pt_z ** 2).argmin()

            x0 = 0.9 * Lx
            y0 = 0.9 * Ly
            self.id2 = ((pt_x - x0) ** 2 + (pt_y - y0) ** 2 + pt_z ** 2).argmin()
        else:
            pts = self.reservoir.unstr_discr.mesh_data.points
            Lx = np.max(pts[:,0])
            Ly = np.max(pts[:,1])
            Lz = np.max(pts[:,2])

            c = np.array([c.centroid for c in self.reservoir.unstr_discr.mat_cell_info_dict.values()])

            x0 = 0.1 * Lx
            y0 = 0.1 * Ly
            self.id1 = ((c[:,0] - x0) ** 2 + (c[:,1] - y0) ** 2 + c[:,2] ** 2).argmin()

            x0 = 0.9 * Lx
            y0 = 0.9 * Ly
            self.id2 = ((c[:,0] - x0) ** 2 + (c[:,1] - y0) ** 2 + c[:,2] ** 2).argmin()

        self.reservoir.add_well("PROD001", depth=0)
        self.reservoir.add_perforation(self.reservoir.wells[-1], int(self.id1), well_index=self.reservoir.well_index)

        self.reservoir.add_well("INJ001", depth=0)
        self.reservoir.add_perforation(self.reservoir.wells[-1], int(self.id2), well_index=self.reservoir.well_index)

    def set_physics(self):
        """Physical properties"""
        zero = 1e-13
        epsilon = 1e-14
        components = ['w', 'o']
        phases = ['wat', 'oil']
        self.cell_property = ['pressure'] + ['water']
        self.cell_property += ['temperature']

        property_container = DeadOilProperties(phases_name=phases, components_name=components,
                                             Mw=np.ones(len(phases)), eps_z=epsilon, temperature=None)

        # Define property evaluators based on custom properties
        property_container.density_ev = dict([('wat', DensityBasic(compr=1e-5, dens0=1014)),
                                              ('oil', DensityBasic(compr=5e-3, dens0=50))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(0.3)),
                                                ('oil', ConstFunc(0.03))])
        property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("gas", 0.1, 0.1)),
                                               ('oil', PhaseRelPerm("oil", 0.1, 0.1))])
        property_container.enthalpy_ev = dict([('wat', EnthalpyBasic(hcap=4.18)),
                                               ('oil', EnthalpyBasic(hcap=0.035))])
        property_container.conductivity_ev = dict([('wat', ConstFunc(1.)),
                                                   ('oil', ConstFunc(1.))])

        property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0)

        # create physics
        thermal = True
        state_spec = PhysicsBase.StateSpecification.PT if thermal else PhysicsBase.StateSpecification.P
        self.physics = PhysicsBase(components, phases, self.timer, state_spec=state_spec,
                                axes_step=[2.5, 2.5e-3, 0.451],  # p [bar], z, T [K]
                                axes_origin=[0.0, epsilon, 273.15 + 20],
                                epsilon_z=epsilon, extrapolation_flag=True)
        self.physics.add_property_region(property_container)

        self.ts_control.runtime = 1000
        self.p_init = 200
        self.init_temp = 350
        self.inj = value_vector([1 - zero, self.init_temp - 30])
        self.ini = value_vector([zero])

        return

    def set_initial_conditions(self):
        input_distribution = {'pressure': self.p_init}
        input_distribution.update({comp: self.ini[i] for i, comp in enumerate(self.physics.components[:-1])})
        if self.physics.thermal:
            input_distribution['temperature'] = self.init_temp

        return self.physics.set_initial_conditions_from_array(self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_boundary_conditions(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=self.p_init-10.)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=self.p_init+10., inj_composition=self.inj[:-1],
                                               inj_temp=self.inj[-1])
