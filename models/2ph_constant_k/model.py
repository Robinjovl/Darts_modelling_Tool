from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import sim_params, value_vector, index_vector
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from scipy.interpolate import interp1d
import os

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.super.initialize import Initialize


class Model(DartsModel):
    def __init__(self, obl_points, reservoir_type, nx: int = None, components: list = [], itor_type: str = 'multilinear',
                 itor_mode: str = 'adaptive', is_barycentric: bool = False):
        # Call base class constructor
        super().__init__()

        self.nx = nx
        self.obl_points = obl_points
        self.discr_type = 'tpfa'
        self.reservoir_type = reservoir_type
        self.components = components
        self.itor_type = itor_type
        self.itor_mode = itor_mode
        self.is_barycentric = is_barycentric

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        if len(self.components) > 14:
            max_ts_mult = 1.
        else:
            max_ts_mult = 5.
        max_ts = min(4., max_ts_mult * 1000 / self.nx)
        self.set_sim_params(first_ts=0.001, mult_ts=2, max_ts=max_ts, runtime=1000, tol_newton=1e-2, tol_linear=1e-3,
                            it_newton=10, it_linear=50, newton_type=sim_params.newton_local_chop)
        # self.params.linear_type = sim_params.cpu_superlu

        self.timer.node["initialization"].stop()

        self.initial_values = {
                self.physics.vars[0]: self.p_init,
                **{self.physics.vars[i + 1]: self.ini_comp[i] for i in range(len(self.physics.vars) - 1)} }
        
        self.inj_composition = self.inj_comp[:self.physics.nc-1]
        self.physics.components = self.components

    def set_reservoir(self):
        if self.reservoir_type == '1D':
            self.p_init = 100.
            self.ny = self.nz = 1
            self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=1000. / self.nx,
                                             dy=1, dz=1, permx=100, permy=100, permz=100, poro=0.3, depth=1000)
            self.well_cell_id = [[1, 1], [self.nx, 1]]
        elif self.reservoir_type == '2D':
            self.p_init = 100.
            self.ny = self.nx
            self.nz = 1
            self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=1000. / self.nx,
                                             dy=1000. / self.nx, dz=1, permx=100, permy=100, permz=100, poro=0.3, depth=1000)
            self.well_cell_id = [[1, 1], [self.nx, self.ny]]
        else: # SPE10
            # read properties
            self.nx, self.ny, self.nz = [int(n) for n in self.reservoir_type.split('_')[1:]]
            input_folder = os.path.join('input', self.reservoir_type.split('_')[0], '_'.join(self.reservoir_type.split('_')[1:]))
            porosity = np.flip(np.swapaxes(load_single_keyword(os.path.join(input_folder, 'poro.txt'), 'PORO', cache=0).
                                     reshape(self.nz, self.ny, self.nx), 0, 2), axis=2)
            permeability = np.flip(np.swapaxes(load_single_keyword(os.path.join(input_folder, 'perm.txt'), 'PERM', cache=0).
                                     reshape(self.nz, self.ny, self.nx, 3), 0, 2), axis=2)
            self.p_init = load_single_keyword(os.path.join(input_folder, 'ref_pres.txt'), 'REF_PRESSURE', cache=0)
            # SPE10 reservoir geometry
            foot2meter = 0.3048
            Lx, Ly, Lz = np.array([960., 2080., 160.]) * foot2meter # feet to meters
            dx, dy, dz = Lx / self.nx, Ly / self.ny, Lz / self.nz
            depth = 12000 * foot2meter# + Lz

            self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz,
                                                         dx=dx, dy=dy, dz=dz,
                                                         permx=permeability[:,:,:,0],
                                                         permy=permeability[:,:,:,1],
                                                         permz=permeability[:,:,:,2],
                                                         poro=porosity, start_z=depth)


            # find well cells
            eps = 0.1 * dx
            self.pt_wells = [[Lx / 2 + eps, Ly / 2 + 200], [Lx / 2 + eps, Ly / 2 - 200]]

            # extend initial pressure array for well bodies/heads
            # n_wells = len(self.well_cell_id)
            # self.p_init = np.append(self.p_init, 2 * n_wells * [np.mean(self.p_init)])

        return

    def set_wells(self):
        if self.reservoir_type == '1D' or self.reservoir_type == '2D':
            self.reservoir.add_well("I1")
            self.reservoir.add_well("P1")
            for k in range(1, self.nz + 1):
                self.reservoir.add_perforation("I1", cell_index=(self.well_cell_id[0][0], self.well_cell_id[0][1], k))
                self.reservoir.add_perforation("P1", cell_index=(self.well_cell_id[1][0], self.well_cell_id[1][1], k))
        return

    def set_wells_spe10(self):
        # evaluate well cells and adjust transmissibilities between them
        cell_m = np.asarray(self.reservoir.mesh.block_m)
        cell_p = np.asarray(self.reservoir.mesh.block_p)
        tran = np.asarray(self.reservoir.mesh.tran)
        rw = 0.1
        dz = np.unique(self.reservoir.global_data['dz'])[0]
        k_poiselle = rw ** 2 / 8 / 0.9869e-15

        self.well_ids = []
        for pt in self.pt_wells:
            # find well cells
            dist = np.linalg.norm(self.reservoir.discretizer.centroids_all_cells[:, :2] - pt, axis=1)
            id_dist_sort = np.argsort(dist)
            id_closest_cells = id_dist_sort[:self.reservoir.nz]
            self.well_ids.append(id_closest_cells)

            # find connections
            mask_m = np.isin(cell_m, id_closest_cells)
            mask_p = np.isin(cell_p, id_closest_cells)
            id_conn = np.where(mask_m & mask_p)[0]

            # tran[id_conn] += k_poiselle * np.pi * rw ** 2 / dz

    def set_physics(self):
        """Physical properties"""
        self.zero = 1e-8
        # Create property containers:

        n_comps = len(self.components)
        self.inj_comp = [1. - (n_comps - 1) * 0.001] + (n_comps - 1) * [0.001]
        if n_comps == 3:
            self.ini_comp = [0.005, 0.550, 0.445]
        elif n_comps == 4:
            self.ini_comp = [0.005, 0.500, 0.300, 0.195]
        elif n_comps == 6:
            self.ini_comp = [0.005, 0.350, 0.250, 0.195, 0.125, 0.075]
        elif n_comps == 8:
            self.ini_comp = [0.005, 0.350, 0.170, 0.150, 0.125, 0.100, 0.075, 0.025]
        elif n_comps == 10:
            self.ini_comp = [0.005, 0.300, 0.170, 0.150, 0.100, 0.100, 0.075, 0.050, 0.025, 0.025]
        elif n_comps == 12:
            self.ini_comp = [0.005, 0.300, 0.150, 0.100, 0.075, 0.075, 0.065, 0.065, 0.060, 0.050, 0.035, 0.020]
        elif n_comps == 14:
            self.ini_comp = [0.005, 0.275, 0.125, 0.100, 0.075, 0.075, 0.065, 0.065, 0.060, 0.050, 0.040, 0.030, 0.020, 0.015]
        elif n_comps == 16:
            self.ini_comp = [0.005, 0.250, 0.125, 0.100, 0.070, 0.070, 0.060, 0.060, 0.050, 0.045,
                             0.040, 0.035, 0.030, 0.025, 0.020, 0.015]
        elif n_comps == 20:
            self.ini_comp = [0.005, 0.240, 0.120, 0.090, 0.070, 0.070, 0.060, 0.060, 0.050, 0.045,
                             0.040, 0.035, 0.030, 0.025, 0.020, 0.015, 0.010, 0.007, 0.005, 0.003]
        assert (np.isclose(sum(self.ini_comp), 1.0, atol=1.e-8))

        Mw_comps = {
            'CO2': 44.010,
            'C1': 16.043,
            'C2': 30.069,
            'C3': 44.096,
            'C4': 58.123,
            'nC4': 58.123,
            'C5': 72.150,
            'nC5': 72.150,
            'C6': 86.178,
            'C7': 100.205,
            'C8': 114.232,
            'C9': 128.259,
            'C10': 142.286,
            'C11': 156.313,
            'C12': 170.340,
            'C13': 184.367,
            'C14': 198.394,
            'C15': 212.421,
            'C16': 226.448,
            'C17': 240.475,
            'C18': 254.502,
            'C19': 268.529,
            'C20': 282.556
        }
        K_comps = {
            'CO2': 1.5,
            'C1': 2.5,
            'C2': 2.0,
            'C3': 1.0,
            'C4': 0.7,
            'nC4': 0.5,
            'C5': 0.4,
            'nC5': 0.3,
            'C6': 0.2,
            'C7': 0.15,
            'C8': 0.12,
            'C9': 0.10,
            'C10': 0.05,
            'C11': 0.04,
            'C12': 0.035,
            'C13': 0.03,
            'C14': 0.025,
            'C15': 0.02,
            'C16': 0.01,
            'C17': 0.01,
            'C18': 0.01,
            'C19': 0.01,
            'C20': 0.01
        }


        Mw = [Mw_comps[c] for c in self.components]
        K = [K_comps[c] for c in self.components]

        phases = ['gas', 'oil']
        thermal = 0

        property_container = ModelProperties(phases_name=phases, components_name=self.components,
                                               Mw=Mw, min_z=self.zero / 10, temperature=1.)

        """ properties correlations """
        property_container.flash_ev = ConstantK(len(self.components), K, self.zero)
        property_container.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                                              ('oil', DensityBasic(compr=1e-5, dens0=600))])
        property_container.viscosity_ev = dict([('gas', ConstFunc(0.1)),
                                                ('oil', ConstFunc(1.0))])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                               ('oil', PhaseRelPerm("oil"))])

        """ Activate physics """
        max_p = 500.
        if n_comps != 20:
            axes_max = [max_p, 1.-self.zero/10, 0.9]
            if n_comps > 3:
                axes_max += [0.7]
            if n_comps > 4:
                axes_max += [0.5]
            if n_comps > 5:
                axes_max += (n_comps - 5) * [0.4]
            assert(len(axes_max) == n_comps)
        else:
            axes_max = np.array([max_p, 1-self.zero/10, 0.240, 0.120, 0.090, 0.070, 0.070, 0.060, 0.060, 0.050, 0.045,
                                 0.040, 0.035, 0.030, 0.025, 0.020, 0.015, 0.010, 0.007, 0.005])
            axes_max[2:] *= 2
            assert(axes_max.size == n_comps)

        if self.reservoir_type != '1D' and self.reservoir_type != '2D':
            max_p = 1.4 * np.max(self.p_init)
            max_p = 500.0
            axes_max[0] = max_p

        thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(self.components, phases, self.timer, state_spec=state_spec, n_points=self.obl_points,
                                     min_p=40, max_p=max_p, min_z=self.zero/10, max_z=1-self.zero/10, cache=False,
                                     axes_max=axes_max)
        self.physics.add_property_region(property_container)
        
        return

    def set_initial_conditions(self):
        if self.reservoir_type == '1D' or self.reservoir_type == '2D':
            input_distribution = {'pressure': self.p_init}
            input_distribution.update({comp: self.ini_comp[i] for i, comp in self.physics.components[:-1]})
            # if self.physics.thermal:
            #     input_distribution['temperature'] = self.init_temp

            return self.physics.set_initial_conditions_from_array(self.reservoir.mesh,
                                                                  input_distribution=input_distribution)
        else:
            # get depths
            depths = np.asarray(self.reservoir.mesh.depth)
            min_depth = np.min(depths)
            max_depth = np.max(depths)

            # calculate phase equilibrium for given uniform composition
            props = self.physics.reservoir_operators[0].property
            state = [np.mean(self.p_init)] + self.ini_comp
            props.evaluate(state)

            # run initialization over depth with specified GOC, pure liquid above, pure vapour under
            from darts.physics.super.initialize import Initialize
            nc = self.physics.nc
            # top boundary
            boundary_state = {var: props.x[0, c] for c, var in enumerate(self.physics.components[:-1])}
            boundary_state['temperature'] = 350.
            boundary_state['pressure'] = np.min(self.p_init)
            init = Initialize(physics=self.physics, algorithm=self.itor_type, mode=self.itor_mode, is_barycentric=self.is_barycentric)
            # GOC
            nb = 100
            mid_depth = (min_depth + max_depth) / 2
            z = np.zeros((nb, nc - 1))
            init_depths = np.linspace(start=min_depth, stop=max_depth, num=nb)

            z[init_depths < mid_depth, :] = props.x[0, :-1]  # vapour is above GOC
            z[init_depths >= mid_depth, :] = props.x[1, :-1]  # liquid is below GOC
            primary_specs = {var: z[:, i] for i, var in enumerate(self.physics.components[:-1])}
            # run initialization
            X = init.solve(depth_bottom=max_depth, depth_top=min_depth, depth_known=min_depth,
                           nb=nb, primary_specs=primary_specs, boundary_state=boundary_state,
                           dTdh=0.).reshape((nb, self.physics.n_vars))

            # assign initial condition with evaluated initialized properties
            self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh, input_depth=init.depths,
                                                                 input_distribution={var: X[:, i] for i, var in enumerate(self.physics.vars)})

    def set_well_controls(self):
        from darts.engines import well_control_iface
        injector = self.reservoir.get_well('I1')
        producer = self.reservoir.get_well('P1')

        zero = self.physics.axes_min[1]
        if self.reservoir_type == '1D':
            self.physics.set_well_controls(wctrl=injector.control, is_control=True, control_type=well_control_iface.MOLAR_RATE,
                                           is_inj=True, target=1., phase_name='gas', inj_composition=self.inj_composition)
            self.physics.set_well_controls(wctrl=producer.control, is_control=True, control_type=well_control_iface.BHP,
                                           is_inj=False, target=50.)
        elif self.reservoir_type == '2D':
            self.physics.set_well_controls(wctrl=injector.control, is_control=True, control_type=well_control_iface.MOLAR_RATE,
                                           is_inj=True, target=300., phase_name='gas', inj_composition=self.inj_composition)
            self.physics.set_well_controls(wctrl=producer.control, is_control=True, control_type=well_control_iface.BHP,
                                           is_inj=False, target=50.)

    def set_rhs_flux(self, t: float = None):
        nv = self.physics.n_vars
        nb = self.reservoir.mesh.n_res_blocks
        rhs_flux = np.zeros(nb * nv)

        if self.reservoir_type != '1D' and self.reservoir_type != '2D':
            self.inj_rate = [-10., 10.]
            Mw = self.physics.property_containers[0].Mw
            for k, ids in enumerate(self.well_ids):
                for c in range(len(self.components)):
                    rhs_flux[ids * nv + c] += self.inj_rate[k] * self.inj_comp[c] / Mw[c]
        return rhs_flux


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, Mw, min_z=1e-11, temperature = 1.):
        # Call base class constructor
        self.nph = len(phases_name)
        # Mw = np.ones(self.nph)
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, min_z=min_z, temperature=1.)

    def evaluate(self, state: value_vector):
        """
        Class methods which evaluates the state operators for the element based physics

        :param state: state variables [pres, comp_0, ..., comp_N-1, temperature (optional)]
        :type state: value_vector

        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        pressure, temperature, zc = self.get_state(state)

        self.clean_arrays()

        self.ph = self.run_flash(pressure, temperature, zc)

        for j in self.ph:
            M = np.sum(self.Mw * self.x[j][:])

            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :])  # output in [kg/m3]
            
            ########################################################## 
            self.dens_m[j] = self.dens[j] / M  # molar density [kg/m3]/[kg/kmol]=[kmol/m3]
            ##########################################################
            
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :], self.dens[j])  # output in [cp]
        self.compute_saturation(self.ph)

        self.pc = np.array(self.capillary_pressure_ev.evaluate(self.sat))

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc = np.array([0, 0])

        mass_source = self.evaluate_mass_source(pressure, temperature, zc)

        return self.ph, self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, mass_source