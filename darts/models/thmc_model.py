from darts.models.darts_model import DartsModel
from darts.engines import value_vector, sim_params, mech_operators, rsf_props, friction, contact_state, state_law, contact_solver, critical_stress, linear_solver_params
from darts.reservoirs.unstruct_reservoir_mech import UnstructReservoirMech
import numpy as np
from darts.reservoirs.mesh.transcalc import TransCalculations as TC
from darts.physics.mech.poroelasticity import Poroelasticity
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.flash import SinglePhase
from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.enthalpy import EnthalpyBasic
from darts.input.input_data import InputData

class THMCModel(DartsModel):
    def __init__(self, n_points=64, discretizer='mech_discretizer'):
        super().__init__()
        self.set_input_data()
        self.set_physics()
        self.set_reservoir()
        self.reservoir.P_VAR = self.physics.engine.P_VAR
        self.reservoir.U_VAR = self.physics.engine.U_VAR
        if self.idata.type_mech == 'thermoporoelasticity':
            self.reservoir.T_VAR = self.physics.engine.T_VAR
        self.set_solver_params()
        self.timer.node["initialization"].stop()

    def reinit(self, zero_conduction):
        self.reservoir.turn_off_equilibrium(zero_conduction)
        self.reservoir.write_to_vtk(self.output_directory, 0, self.physics.engine)
        self.reservoir.eps_vol_ref = np.array(self.reservoir.mesh.ref_eps_vol, copy=False)
        self.reservoir.eps_vol_ref[:] = self.physics.engine.eps_vol[:]
        self.physics.engine.t = 0.0

    def set_reservoir(self, timer):
        self.reservoir = UnstructReservoirMech(timer=timer, discretizer=discretizer,
                                               thermoporoelasticity=self.idata.type_mech == 'thermal',
                                               fluid_vars=self.physics.vars)

    def set_solver_params(self):
        self.params.tolerance_newton = 1e-6 # Tolerance of newton residual norm ||residual||<tol_newt
        self.params.newton_type = sim_params.newton_global_chop  # Type of newton method (related to chopping strategy?)
        self.params.newton_params = value_vector([0.2])  # Probably chop-criteria(?)
        self.params.max_i_newton = 10

        if self.discretizer_name == 'mech_discretizer':
            self.params.tolerance_linear = 1e-10  # Tolerance for linear solver ||Ax - b||<tol_linslv
            self.params.linear_type = sim_params.cpu_superlu  # cpu_gmres_fs_cpr # cpu_superlu
            self.params.max_i_linear = 5000
        elif self.discretizer_name == 'pm_discretizer':
            ls1 = linear_solver_params()
            ls1.linear_type = sim_params.cpu_superlu # cpu_gmres_fs_cpr # cpu_superlu
            ls1.tolerance_linear = 1.e-12
            ls1.max_i_linear = 500
            self.physics.engine.ls_params.append(ls1)

    def set_input_data(self):
        self.idata.check()
        
    def set_physics(self):
        # Create property containers:
        components = ['H2O']
        phases = ['wat']
        Mw = [self.idata.fluid.Mw]

        if self.idata.type_hydr == 'thermal':
            property_container = PropertyContainer(phases_name=phases, components_name=components,
                                                   Mw=Mw, min_z=self.idata.obl.min_z)
        else:
            property_container = PropertyContainer(phases_name=phases, components_name=components,
                                                   Mw=Mw, min_z=self.idata.obl.min_z, temperature=1.)

        """ properties correlations """
        property_container.flash_ev = SinglePhase(nc=1)
        property_container.density_ev = dict([('wat', DensityBasic(compr=self.idata.fluid.compressibility,
                                                                   dens0=self.idata.fluid.density))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(self.idata.fluid.viscosity))])

        property_container.rel_perm_ev = dict([('wat', ConstFunc(1.0))])
        # rock compressibility is treated inside engine
        property_container.rock_compr_ev = ConstFunc(1.0)
        property_container.rock_density_ev = ConstFunc(self.idata.rock.density)
        # create physics
        if self.idata.type_mech == 'thermoporoelasticity':
            property_container.enthalpy_ev = dict([('wat', EnthalpyBasic(hcap=self.idata.rock.heat_capacity, tref=0.0))])
            property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0, tref=0.0)  #TODO use hcap from idata? see https://gitlab.com/open-darts/open-darts/-/issues/19
            property_container.conductivity_ev = dict([('wat', ConstFunc(1.0))])
            self.physics = Poroelasticity(components, phases, self.timer, n_points=self.idata.obl.n_points,
                                          min_p=self.idata.obl.min_p, max_p=self.idata.obl.max_p, 
                                          min_z=self.idata.obl.min_z, max_z=self.idata.obl.max_z,
                                          thermal=True, min_t=self.idata.obl.min_t, max_t=self.idata.obl.max_t,
                                          discretizer=self.discretizer_name)
        else:
            self.physics = Poroelasticity(components, phases, self.timer, n_points=self.idata.obl.n_points,
                                          min_p=self.idata.obl.min_p, max_p=self.idata.obl.max_p, 
                                          min_z=self.idata.obl.min_z, max_z=self.idata.obl.max_z,
                                          discretizer=self.discretizer_name)
        self.physics.add_property_region(property_container)

        self.physics.init_physics(discr_type=self.discretizer_name, platform='cpu')
        return

    def init(self):
        if self.discretizer_name == 'pm_discretizer':
            self.reservoir.mech_operators = mech_operators()
            self.reservoir.mech_operators.init(self.reservoir.mesh, self.reservoir.pm,
                                     self.physics.engine.P_VAR, self.physics.engine.Z_VAR, self.physics.engine.U_VAR,
                                     self.physics.engine.N_VARS, self.physics.engine.N_OPS, self.physics.engine.NC,
                                     self.physics.engine.ACC_OP, self.physics.engine.FLUX_OP, self.physics.engine.GRAV_OP)
            self.reservoir.mech_operators.prepare()

        self.set_wells()

        # TODO: move init_physics here for consistence with DartsModel
        # assert self.physics is not None, "Physics object has not been defined"
        # self.physics.init_physics(discr_type=discr_type, platform=platform, verbose=verbose)

        self.reservoir.init_wells()
        self.physics.init_wells(self.reservoir.wells)

        self.set_op_list()
        self.set_boundary_conditions()
        self.set_initial_conditions()
        self.set_well_controls()
        self.reset()  # engine is created here
        #TODO replace the lines above with this;
        # remove self.physics.engine arg from self.physics.init_wells as engine is located is in self.physics in the current dev branch
        # self.reservoir.init_wells()  keep this?
        #super().init()  # init base model and engine

        # link engine with discretizer
        if self.discretizer_name == 'mech_discretizer':
            self.physics.engine.set_discretizer(self.reservoir.discr)
            self.physics.engine.gravity = self.reservoir.discr.grav_vec.values

    def set_wells(self):
        pass

    def add_wells(self):
        layers_num = 1
        # unstructured
        dist = 1.E+10
        mid = (np.min(self.reservoir.unstr_discr.mesh_data.points, axis=0) +
               np.max(self.reservoir.unstr_discr.mesh_data.points, axis=0)) / 2
        id = -1
        for cell_id, cell in self.reservoir.unstr_discr.mat_cell_info_dict.items():
            cur_dist = (cell.centroid[0] - mid[0]) ** 2 + (cell.centroid[1] - mid[1]) ** 2 + cell.centroid[2] ** 2
            if dist > cur_dist:
                dist = cur_dist
                id = cell_id

        self.reservoir.add_well("PROD001", depth=0)
        for kk in range(layers_num):
            self.reservoir.add_perforation(self.reservoir.wells[-1], int(id + kk),
                                           well_index=self.reservoir.well_index)

    def set_initial_conditions(self):
        if self.reservoir.thermoporoelasticity:
            self.physics.set_uniform_initial_conditions(self.reservoir.mesh,
                                                        uniform_pressure=self.reservoir.p_init,
                                                        uniform_composition=self.reservoir.z_init,
                                                        uniform_temperature=self.reservoir.t_init,
                                                        uniform_displacement=self.reservoir.u_init)
        else:
            self.physics.set_uniform_initial_conditions(self.reservoir.mesh,
                                                        uniform_pressure=self.reservoir.p_init,
                                                        uniform_composition=self.reservoir.z_init,
                                                        uniform_displacement=self.reservoir.u_init)
        return 0

    def set_boundary_conditions(self):
        """
        Class method called in the init() class method of parents class
        :return:
        """
        return 0

    def set_op_list(self):
        self.op_list = [self.physics.acc_flux_itor[0], self.physics.acc_flux_w_itor]

    def get_performance_data(self, is_last_ts: bool = False):
        """
        Function to get the needed performance data
        """
        perf_data = dict()
        perf_data['solution'] = np.copy(self.physics.engine.X)
        perf_data['variables'] = ['ux', 'uy', 'uz', 'p']
        perf_data['reservoir blocks'] = self.reservoir.mesh.n_blocks

        if is_last_ts:
            perf_data['OBL resolution'] = list(self.physics.n_axes_points)
            perf_data['operators'] = self.physics.n_ops
            perf_data['timesteps'] = self.physics.engine.stat.n_timesteps_total
            perf_data['wasted timesteps'] = self.physics.engine.stat.n_timesteps_wasted
            perf_data['newton iterations'] = self.physics.engine.stat.n_newton_total
            perf_data['wasted newton iterations'] = self.physics.engine.stat.n_newton_wasted
            perf_data['linear iterations'] = self.physics.engine.stat.n_linear_total
            perf_data['wasted linear iterations'] = self.physics.engine.stat.n_linear_wasted

            sim = self.timer.node['simulation']
            jac = sim.node['jacobian assembly']
            perf_data['simulation time'] = sim.get_timer()
            perf_data['linearization time'] = jac.get_timer()
            perf_data['linear solver time'] = sim.node['linear solver solve'].get_timer() + sim.node[
                'linear solver setup'].get_timer()
            interp = jac.node['interpolation']
            perf_data['interpolation incl. generation time'] = interp.get_timer()

        return perf_data

    def save_performance_data(self, data, file_name):
        import platform
        import pickle
        """
        Function to save performance data for future comparison.
        :param file_name:
        :return:
        """
        with open(file_name, "wb") as fp:
            pickle.dump(data, fp, 4)

    def load_performance_data(self, file_name=''):
        import os
        import pickle
        """
        Function to load the performance pkl file at previous simulation.
        :param file_name: performance filename
        """
        if os.path.exists(file_name):
            with open(file_name, "rb") as fp:
                return pickle.load(fp)
        return 0
    
    # it doesn't use model object, put inside the class just for the convenience of import 
    def check_performance_data(self, ref_data, cur_data, prev_fail,
                               diff_max_tol=1e-6,
                               diff_max_normalized_tol=1e-4,
                               rel_diff_tol=1, plot=False, png_suffix=''):
        fail = 0
        # the difference lower than eps will not be accounted
        eps_sol = {'p': 1e-5, 'ux': 1e-5, 'uy': 1e-5, 'uz': 1e-5}
        get_eps = lambda var_name: eps_sol[var_name] if var_name in eps_sol.keys() else 0.
        # data = self.get_performance_data()
        nb = ref_data['reservoir blocks']
        vars = cur_data['variables']
        nv = len(vars)
        # Check final solution - data[0]
        # Check every variable separately
        for v in range(nv):
            sol_et = ref_data['solution'][v:nb * nv:nv]
            sol_cur = cur_data['solution'][v:nb * nv:nv]
            sol_range = np.max(sol_et) - np.min(sol_et) + 1.e-12
            # replace small values in solution with eps to avoid difference in normalized diff
            sol_et[np.fabs(sol_et) < get_eps(vars[v])] = 0.
            sol_cur[np.fabs(sol_cur) < get_eps(vars[v])] = 0.
            diff = sol_cur - sol_et
            diff_abs = np.abs(diff)
            diff_max_abs = diff_abs.max()
            diff_norm = np.linalg.norm(diff)
            diff_norm_normalized = diff_norm / len(sol_et) / sol_range
            diff_abs_max_normalized = np.max(diff_abs) / sol_range
            if diff_max_abs > diff_max_tol and diff_abs_max_normalized > diff_max_normalized_tol:
                fail += 1
                print(
                    '#%d solution check failed for variable %d %s (range %.2E): max(abs(diff))/range %.2E (tol %.2E), max(abs(diff)) = %.2E' \
                    % (fail, v, vars[v], sol_range, diff_abs_max_normalized, diff_max_normalized_tol, diff_max_abs))
            if plot:
                # plot two solutions and difference between them
                from matplotlib import pyplot as plt
                fig, (ax1, ax2) = plt.subplots(2, sharex=True)
                ax1.plot(sol_et, 'r', label='ref')
                ax1.plot(sol_cur, 'b--', label='cur')
                ax1.set_title(vars[v])
                ax1.legend()
                ax2.plot(diff, 'b')
                ax2.set_title('diff')
                plt.savefig(vars[v] + '_' + png_suffix + '.png', dpi=500)
                plt.clf()
                plt.close()
    
    
        for key, value in sorted(cur_data.items()):
            if key == 'solution' or type(value) != int:
                continue
            reference = ref_data[key]
    
            if reference == 0:
                if value != 0:
                    print('#%d parameter %s is %d (was 0)' % (fail, key, value))
                    fail += 1
            else:
                rel_diff = (value - ref_data[key]) / reference * 100
                if abs(rel_diff) > rel_diff_tol:
                    print('#%d parameter %s is %d (was %d, %+.2f%%)' % (fail, key, value, reference, rel_diff))
                    fail += 1
    
            if not fail:
                return 0
            else:
                return 1



