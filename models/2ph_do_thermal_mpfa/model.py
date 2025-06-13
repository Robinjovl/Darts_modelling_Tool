from darts.models.cicd_model import CICDModel
from darts.models.darts_model import DartsModel
from darts.engines import value_vector
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.enthalpy import EnthalpyBasic

from reservoir import UnstructReservoir

class Model(CICDModel):
    def __init__(self, discr_type='mpfa', mesh_file='meshes/wedge.msh'):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.discr_type = discr_type
        self.physics_type = 'dead_oil'

        self.set_physics()
        self.set_reservoir(mesh_file)

        self.set_sim_params(first_ts=1e-4, mult_ts=2, max_ts=5, tol_newton=1e-3, tol_linear=1e-6)
        self.timer.node["initialization"].stop()

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
        components = ['w', 'o']
        phases = ['wat', 'oil']
        self.cell_property = ['pressure'] + ['water']
        self.cell_property += ['temperature']

        property_container = ModelProperties(phases_name=phases, components_name=components, min_z=zero/10)

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
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                n_points=400, min_p=0, max_p=1000, min_z=zero, max_z=1-zero,
                                min_t=273.15 + 20, max_t=273.15 + 200)
        self.physics.add_property_region(property_container)

        self.runtime = 1000
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


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name, components_name, Mw=Mw, min_z=min_z, temperature=None)

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]
        self.temperature = vec_state_as_np[-1] if self.thermal else self.temperature

        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        self.ph = np.array([0, 1], dtype=np.intp)

        for j in self.ph:
            # molar weight of mixture
            M = np.sum(self.x[j, :] * self.Mw)
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]

        self.nu = zc
        self.compute_saturation(self.ph)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc[j] = 0

        mass_source = np.zeros(self.nc)

        return self.ph, self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, mass_source

    def evaluate_at_cond(self, pressure, zc):

        self.sat[:] = 0

        ph = [0, 1]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.dens_m = [1025, 0.77]  # to match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m
