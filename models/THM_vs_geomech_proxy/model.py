from scipy.interpolate import interp1d
import numpy as np
from darts.physics.base.property_container import PropertyContainer
from darts.physics.properties.density import DensityBasic
from darts.engines import well_control_iface
from darts.models.thmc_model import THMCModel
from darts.engines import sim_params
from darts.physics.base.initialize import Initialize
from darts.physics.properties.viscosity import MaoDuan2009

from reservoir import UnstructReservoirCustom

class MaoDuan2009Shifted(MaoDuan2009):
    """MaoDuan2009 viscosity driven by the model's relative temperature scale.

    MaoDuan2009 requires ABSOLUTE temperature in Kelvin, but this model runs on a
    relative temperature scale with baseline 0 (OBL range -50..50). This wrapper maps
    the model baseline T=0 to t_abs0 (default 373.15 K), so MaoDuan2009 always sees a
    physical absolute temperature (~323..423 K over the OBL range) and returns a
    positive viscosity instead of the negative values that previously stalled the well
    residual. See set_input_data() t_ref note.
    """
    def __init__(self, components, t_abs0=373.15, ions=None, combined_ions=None):
        super().__init__(components, ions, combined_ions)
        self.t_abs0 = t_abs0

    def evaluate(self, pressure, temperature, x, rho):
        return super().evaluate(pressure, temperature + self.t_abs0, x, rho)


class DensityBasicTdep(DensityBasic):
    """DensityBasic + linear thermal expansion.
    rho(p,T) = dens0 * (1 + compr*(p - p0) - thermal_expn*(T - t0))
    Unlike MaoDuan2009/Spivey2004, it does NOT need absolute Kelvin: t0 is the
    model's baseline temperature (this model uses a relative scale with baseline 0),
    so it stays well-behaved over the operating range and converges.
    """
    def __init__(self, dens0, compr=0.0, p0=1.0, thermal_expn=0.0, t0=0.0):
        super().__init__(dens0, compr, p0)
        self.thermal_expn = thermal_expn
        self.t0 = t0

    def evaluate(self, pressure, temperature: float = None, x: list = None):
        rho = super().evaluate(pressure, temperature, x)
        if temperature is not None and self.thermal_expn != 0.0:
            rho *= (1.0 - self.thermal_expn * (temperature - self.t0))
        return rho


def fmt_e(x : float):
    return "{:.3e}".format(x) if np.isscalar(x) else str(x)

def fmt(x : float):
    return "{:.3}".format(x) if np.isscalar(x) else str(x)

class Model(THMCModel):
    def __init__(self, model_folder, physics_type='single_phase',
                 uniform_props=False, wells_type=None,
                 decouple_geomech=False, generate_mesh=False, dummy='no',
                 solver_type='by_env_var', cache_discretization=None):
        self.model_folder = model_folder
        # reuse the discretization of an earlier run with the same input, see
        # UnstructReservoirCustom.discretization_cache_key; None: on unless DARTS_DISCR_CACHE=0
        self.cache_discretization = cache_discretization
        self.uniform_props = uniform_props
        self.physics_type = physics_type
        self.discretizer_name = 'mech_discretizer'
        self.thermal = self.physics_type == 'single_phase_thermal'
        self.decouple_geomech = decouple_geomech
        self.generate_mesh = generate_mesh
        self.wells_type = wells_type
        self.solver_type = solver_type  # 'superlu', 'fs_cpr', 'by_env_var'

        if dummy == 'yes':  # save time for proxy run
            return
        # call base class constructor
        super().__init__()


    def set_solver(self):
        # Open-source FS-CPR by default (mech_discretizer / engine_super_elastic_cpu).
        # The spec drives _apply_solver in the open-source build; in the proprietary
        # build it is ignored and the engine factory uses params.linear_type
        # (bos_fs_cpr).
        from darts.linear_solvers.specs import FSCPRSolverSpec, GMRESSolverSpec
        mesh = self.reservoir.mesh
        n_res_blks = mesh.n_res_blocks
        n_matrix = getattr(self.reservoir, 'n_matrix', n_res_blks)
        n_fracs_mesh = getattr(self.reservoir, 'n_fracs', 0)
        fs_cpr = FSCPRSolverSpec(
            force_amg_asymmetric=True,
            n_res=n_matrix + n_fracs_mesh,
            n_fracs=0,
            n_wells=mesh.n_blocks - n_res_blks,
        )
        # Single solver declaration: the spec drives _apply_solver on the open-source
        # CPU build; on the proprietary build _apply_solver applies
        # proprietary_linear_type (bos_fs_cpr) to params.linear_type. No model-level
        # params.linear_type needed (its open-source value was the engine default).
        # Pre-!280 this model solved at tolerance_linear=1e-8 / max_i_linear=5000.
        # The spec owns those parameters now, so it must carry the same values.
        # The spec is set here, before super().set_solver() below, so the platform
        # default is never materialized.
        self.linear_solver.spec = GMRESSolverSpec(prec=fs_cpr, tolerance=1e-8, max_iterations=5000, restart=50,
                                  proprietary_linear_type=sim_params.cpu_gmres_fs_cpr)
        super().set_solver()
        self.ts_control.dt_first = 0.01
        self.ts_control.dt_mult = 8
        self.ts_control.dt_max = 5
        self.nonlinear_solver.spec.tolerance = 1e-6
        self.params.tolerance_linear = self.linear_solver.spec.tolerance
        self.params.max_i_linear = self.linear_solver.spec.max_iterations
        self.nonlinear_solver.spec.max_iterations = 20

    def set_reservoir(self):
        mesh_folder = self.idata.other.mesh_dir if self.idata.other.mesh_dir is not None else self.model_folder
        self.reservoir = UnstructReservoirCustom(timer=self.timer, fluid_vars=self.physics.vars,
                                                 idata=self.idata, model_folder=mesh_folder,
                                                 uniform_props=self.uniform_props, generate_mesh=self.generate_mesh,
                                                 cache_discretization=self.cache_discretization)

    def set_input_data(self):
        from set_case import set_input_data

        self.idata = set_input_data(
            case=self.model_folder,
            model_folder=self.model_folder,
            physics_type=self.physics_type,
            wells_type=self.wells_type,
        )

        super().set_input_data()
        return


    def set_physics(self):
        super().set_physics()
        if self.physics_type == 'single_phase':
            pass
        elif self.physics_type == 'single_phase_thermal':
            # pass
            # MaoDuan2009 requires ABSOLUTE temperature in Kelvin; this model runs on a relative
            # temperature scale with baseline 0 (OBL range -50..50). MaoDuan2009Shifted maps the
            # model baseline T=0 to 373.15 K (assume 100 degrees C in the reservoir)
            # so the correlation always sees a physical absolute temperature
            # and returns a positive viscosity. See set_input_data() t_ref note.
            components = self.physics.components
            property_container = self.physics.property_containers[0]
            property_container.viscosity_ev = dict([('wat', MaoDuan2009Shifted(components, t_abs0=373.15))])
            property_container.density_ev = dict([('wat', DensityBasicTdep(dens0=1000))])
        return

    def set_wells(self):
        #return
        centroids_3d = np.array([np.array([c.values[0], c.values[1], c.values[2]]) for
                              c in self.reservoir.discr_mesh.centroids])[:self.reservoir.n_matrix]

        if self.wells_type == 'none':
            well_names = []
            well_coords = []
        elif self.wells_type == 'prod':  # one well (prod)
            well_names = ['PRD1']
            well_coords = np.array([self.idata.other.prod_well_coords])
        elif self.wells_type == 'inj': # one well (inj)
            well_names = ['INJ1']
            well_coords = np.array([self.idata.other.inj_well_coords])
        elif self.wells_type == 'doublet':# two wells (doublet)
            well_names = ['PRD1', 'INJ1']
            well_coords = np.array([self.idata.other.prod_well_coords, self.idata.other.inj_well_coords])

        print('well_coords:', well_coords)
        print('centroids_mean depth:', centroids_3d[:, 2].mean())

        self.well_cell_ids = []

        nodes = np.array(self.reservoir.discr_mesh.nodes)
        elems = np.array(self.reservoir.discr_mesh.elems)

        step_z_perf = 1 # [m] should be smaller that cell dz

        # Restrict nearest-cell matching to explicitly allowed
        # physical tags so a tiny centroid-distance difference cannot put a
        # perforation into overburden or underburden.
        perforation_tags = getattr(self.idata.other, 'well_perforation_tags', None)
        if perforation_tags is None:
            candidate_ids = np.arange(self.reservoir.n_matrix, dtype=int)
        else:
            matrix_tags = np.asarray(self.reservoir.tags[:self.reservoir.n_matrix])
            candidate_ids = np.flatnonzero(np.isin(matrix_tags, perforation_tags))
            if candidate_ids.size == 0:
                raise ValueError(
                    f'No matrix cells found with well perforation tags {perforation_tags}'
                )

        for i, coord in enumerate(well_coords): # process each well
            # find mesh cells which
            z1, z2 = coord[2], coord[3]
            z_points = np.arange(z1, z2, step_z_perf)
            ids = set()
            for z in z_points: # find a cell with the closest center
                distances_sq = (
                    (centroids_3d[candidate_ids, 0] - coord[0]) ** 2
                    + (centroids_3d[candidate_ids, 1] - coord[1]) ** 2
                    + (centroids_3d[candidate_ids, 2] - z) ** 2
                )
                cell = candidate_ids[distances_sq.argmin()]
                ids.add(int(cell))
            ids_1 = list(ids)

            # sort perforations by depth
            perf_depths = centroids_3d[ids_1, 2]
            perf_sorted_indices = np.argsort(perf_depths)
            ids_1 = np.array(ids_1)[perf_sorted_indices]

            self.well_cell_ids.append(ids_1)
            # adding a well
            self.reservoir.add_well(well_names[i])
            # adding perforations
            for cell_id in ids_1:
                cell = elems[cell_id]
                pt_ids = self.reservoir.discr_mesh.elem_nodes[cell.pts_offset:cell.pts_offset + cell.n_pts]
                pts = np.array([nodes[id].values for id in pt_ids])
                # Calculate well_index (very primitive way....):
                rw = 0.0762 # m
                # onstruct a right hexahedron around cell nodes
                dx = np.max(pts, axis=0)[0] - np.min(pts, axis=0)[0]
                dy = np.max(pts, axis=0)[1] - np.min(pts, axis=0)[1]
                dz = np.max(pts, axis=0)[2] - np.min(pts, axis=0)[2]
                perm = np.array(self.reservoir.discr.perms[cell_id].values)
                mean_perm_xx = perm[0]
                mean_perm_yy = perm[4]
                mean_perm_zz = perm[8]
                rp_z = 0.28 * np.sqrt((mean_perm_yy / mean_perm_xx) ** 0.5 * dx ** 2 +
                                      (mean_perm_xx / mean_perm_yy) ** 0.5 * dy ** 2) / \
                       ((mean_perm_xx / mean_perm_yy) ** 0.25 + (mean_perm_yy / mean_perm_xx) ** 0.25)
                wi_x = 0.0
                wi_y = 0.0
                wi_z = 2 * np.pi * np.sqrt(mean_perm_xx * mean_perm_yy) * dz / np.log(rp_z / rw)
                well_index = np.sqrt(wi_x ** 2 + wi_y ** 2 + wi_z ** 2)
                # add perforation
                self.reservoir.add_perforation(self.reservoir.wells[-1].name, res_cell_idx=cell_id,
                                               well_index=well_index, well_indexD=0., ms_epm=True, verbose=True)
                print('well perf added to the cell', cell_id, 'with a center=', centroids_3d[cell_id], 'for the requested point=', centroids_3d[cell_id,:])

    def set_boundary_conditions(self): # for initial mechanical equilibrium initialization, wells are switched off
        for i, w in enumerate(self.reservoir.wells):
                self.physics.set_well_controls(w.control,
                                               control_type=well_control_iface.MOLAR_RATE,
                                               is_inj=False, target=0., phase_name='wat')


    def set_boundary_conditions_after_initialization(self): # set well controls
        #return
        """
        Class method called in the init() class method of parents class
        :return:
        """
        # Takes care of well controls, argument of the function is (in case of bhp) the bhp pressure and (in case of
        # rate) water/oil rate:

        for i, w in enumerate(self.reservoir.wells):
            p_cell = self.initial_pressure[self.well_cell_ids[i][0]] # from the first perforation of the current well
            t_cell = self.initial_temperature[self.well_cell_ids[i][0]]

            delta_temp_inj = self.idata.other.delta_temp_inj
            delta_p = self.idata.other.delta_p
            well_rate = self.idata.other.well_rate
            wctrl_type = self.idata.other.wctrl_type

            if 'PRD' in w.name:
                target = p_cell - delta_p if wctrl_type == well_control_iface.BHP else well_rate
                print('prod well', w.name, 'control', wctrl_type, 'target', fmt(target))
                self.physics.set_well_controls(wctrl=w.control,
                                               control_type=wctrl_type,
                                               is_inj=False,
                                               target=target)
            elif 'INJ' in w.name:
                inj = []
                inj_temp = None
                if self.physics_type == 'single_phase_thermal':
                    inj_temp = t_cell - delta_temp_inj
                target = p_cell + delta_p if wctrl_type == well_control_iface.BHP else well_rate
                print('inj well', w.name, 'control', wctrl_type, 'target ' + fmt(target), 'inj_temp = ' + fmt(inj_temp))
                self.physics.set_well_controls(wctrl=w.control,
                                               control_type=wctrl_type,
                                               is_inj=True,
                                               target=target,
                                               inj_composition=inj,
                                               inj_temp=inj_temp)
        return 0

    def set_initial_conditions(self):

        if True: # compute fluid equilibrium from given p,T at the surface
            # pressure gradient might vary as the density depends on the temperature
            boundary_state = {}
            if self.thermal:
                boundary_state['temperature'] = self.idata.initial.temperature_at_ref_depth
            boundary_state['pressure'] = self.idata.initial.pressure_at_ref_depth
            init = Initialize(physics=self.physics, algorithm='multilinear',
                              is_barycentric=False)

            nb = 100
            nc = self.physics.nc
            z = np.zeros((nb, nc))
            z[:, 0] = 1 - self.idata.obl.zero  # liquid is below GOC
            primary_specs = {var: z[:, i] for i, var in enumerate(self.physics.components[:-1])}
            # run initialization
            min_depth = self.reservoir.depths.min()
            max_depth = self.reservoir.depths.max()
            X = init.solve_up_and_downwards(depth_bottom=max_depth, depth_top=min_depth, depth_known=min_depth,
                           nb=nb, primary_specs=primary_specs, boundary_state=boundary_state,
                           dTdh=self.idata.initial.temperature_gradient).reshape((nb, self.physics.n_vars))
            input_distribution = {var: X[:, i] for i, var in enumerate(self.physics.vars)}
            set_initial_conditions_from_depth_table(self=self.physics, mesh=self.reservoir.mesh, input_depth=init.depths,
                                                                 input_distribution=input_distribution,
                                                                 input_displacement=self.reservoir.u_init,
                                                            depths=self.reservoir.depths[:self.reservoir.mesh.n_blocks])
        else:
            #self.reservoir.p_init[:] = 500  # uniform init pressure
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
        s = np.asarray(self.reservoir.mesh.initial_state)
        if self.reservoir.thermoporoelasticity:
            self.initial_pressure = s[0::2][:self.reservoir.n_matrix]
            self.initial_temperature = s[1::2][:self.reservoir.n_matrix]
        else:
            self.initial_pressure = s[:self.reservoir.n_matrix]
            self.initial_temperature = np.zeros_like(s)[:self.reservoir.n_matrix]
        print('Initial pressure: min/mean/max:', fmt(self.initial_pressure.min()), fmt(self.initial_pressure.mean()), fmt(self.initial_pressure.max()))
        print('Initial temperature: min/mean/max:', fmt(self.initial_temperature.min()), fmt(self.initial_temperature.mean()), fmt(self.initial_temperature.max()))

        centroids_3d = np.array([np.array([c.values[0], c.values[1], c.values[2]]) for
                              c in self.reservoir.discr_mesh.centroids])[:self.reservoir.n_matrix]
        eps = 1
        from functools import reduce
        rsv = reduce(np.logical_and, [self.reservoir.rsv_top - eps < centroids_3d[:,2], centroids_3d[:,2] < self.reservoir.rsv_bottom + eps])
        print('Initial pressure rsv: min/mean/max:', fmt(self.initial_pressure[rsv].min()), fmt(self.initial_pressure[rsv].mean()), fmt(self.initial_pressure[rsv].max()))
        print('Initial temperature rsv: min/mean/max:', fmt(self.initial_temperature[rsv].min()), fmt(self.initial_temperature[rsv].mean()), fmt(self.initial_temperature[rsv].max()))

        return 0


def set_initial_conditions_from_depth_table(self, mesh, input_distribution: dict, input_displacement: dict,
                                            input_depth, depths):
    """
    Function to set initial conditions from given distribution of properties over depth.

    :param mesh: conn_mesh object
    :param input_distribution: Initial distributions of unknowns over depth, must have keys equal to self.vars
                               and each entry is scalar or array of length equal to depths
    :param input_depth: Array of depths over which depth table has been specified
    """
    # Assertions of consistent depth table specification
    assert np.all([variable in input_distribution.keys() for variable in self.vars[1:self.nc]]), \
        "Initial state for must be specified for all primary variables"
    assert not self.thermal or ('temperature' in input_distribution.keys() or
                                'enthalpy' in input_distribution.keys()), \
        "Temperature or enthalpy must be specified for thermal models"
    input_depth = input_depth if hasattr(input_depth, "__len__") else np.array([input_depth])
    for key, input_values in input_distribution.items():
        input_values = input_values if hasattr(input_values, "__len__") else np.ones(len(input_depth)) * input_values
        assert len(input_values) == len(input_depth)

    # Get depths and primary variable arrays from mesh object
    #depths = np.asarray(mesh.depth)[:mesh.n_blocks]

    # adjust the size of initial_state array in c++
    mesh.initial_state.resize(mesh.n_blocks * self.n_vars)

    # Loop over variables to fill initial_state vector in c++
    for ith_var, variable in enumerate(self.vars):
        if variable == "enthalpy" and "enthalpy" not in input_distribution.keys():
            # If temperature has been provided, interpolate pressure and temperature to compute enthalpies
            p_itor = interp1d(input_depth, input_distribution['pressure'], kind='linear', fill_value='extrapolate')
            pressure = p_itor(depths)

            t_itor = interp1d(input_depth, input_distribution['temperature'], kind='linear', fill_value='extrapolate')
            temperature = t_itor(depths)

            values = np.empty(mesh.n_blocks)
            for j in range(mesh.n_blocks):
                state = np.array([pressure[j], temperature[j]])
                values[j] = self.property_containers[0].compute_total_enthalpy(state, temperature[j])
        else:
            # Else, interpolate primary variable
            itor = interp1d(input_depth, input_distribution[variable], kind='linear', fill_value='extrapolate')
            values = itor(depths)

        np.asarray(mesh.initial_state)[ith_var::self.n_vars] = values

    # set initial displacements
    for i in range(self.n_dim):
        np.asarray(mesh.displacement)[i::self.n_dim] = input_displacement[i]


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, eps_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        from tools import darts_version_ge
        kwargs = dict(phases_name=phases_name, components_name=components_name, Mw=Mw, temperature=None)
        # PropertyContainer renamed the small-composition floor min_z -> eps_z after 1.5.0
        kwargs['eps_z' if darts_version_ge((1, 5, 1)) else 'min_z'] = eps_z
        super().__init__(**kwargs)

    def evaluate(self, state):
        super().evaluate(state)
        return self.ph, self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, self.mass_source
