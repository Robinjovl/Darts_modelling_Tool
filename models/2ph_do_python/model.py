from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import value_vector, index_vector, sim_params
import numpy as np
from math import fabs
import jax
import jax.numpy as jnp
from jax import lax

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer, PropertyContainerJAX
from darts.physics.super.operator_evaluator import ReservoirOperators

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic

class Model(CICDModel):
    def __init__(self, python_assembly):
        # call base class constructor
        super().__init__()

        self.python_assembly = python_assembly

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics(jax_support=python_assembly)

        self.set_sim_params(first_ts=0.01, mult_ts=2, max_ts=5, runtime=300, tol_newton=1e-3, tol_linear=1e-6)
        self.params.nonlinear_norm_type = self.params.LINF

        self.timer.node["initialization"].stop()

        self.initial_values = {self.physics.vars[0]: 200,
                               self.physics.vars[1]: self.ini,
                               }

    def set_reservoir(self):
        self.nx = 100
        self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=1, nz=1, dx=1.0, dy=1.0, dz=1,
                                         permx=100, permy=100, permz=100, poro=0.2, hcap=0, rcond=0, depth=1000)
        self.well_cell_id = [[1, 1], [self.nx, 1]]
        return

    def set_wells(self):
        volume = np.array(self.reservoir.mesh.volume, copy=False)
        volume[self.well_cell_id[1][0] - 1] = 1.e+6

    def set_rhs_flux(self, t: float = None):
        nv = self.physics.n_vars
        nb = self.reservoir.mesh.n_res_blocks
        rhs_flux = np.zeros(nb * nv)
        # injection
        rhs_flux[(self.well_cell_id[0][0] - 1) * nv + 1] = -20.

        return rhs_flux

    def set_properties(self, comps, phases, zero, jax_support):
        if jax_support:
            property_container = ModelPropertiesJAX(phases_name=phases, components_name=comps, min_z=zero / 10)
        else:
            property_container = ModelProperties(phases_name=phases, components_name=comps, min_z=zero / 10)

        property_container.density_ev = dict([('wat', DensityBasic(compr=1e-5, dens0=1014)),
                                              ('oil', DensityBasic(compr=5e-3, dens0=500))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(0.3)),
                                                ('oil', ConstFunc(0.03))])
        property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("wat", 0.1, 0.1)),
                                               ('oil', PhaseRelPerm("oil", 0.1, 0.1))])
        return property_container

    def set_physics(self, jax_support=False):
        """Physical properties"""
        self.zero = 1e-13
        self.components = ["w", "o"]
        phases = ["wat", "oil"]

        self.inj = value_vector([self.zero])
        self.ini = value_vector([1. - self.zero])

        property_container = self.set_properties(comps=self.components, phases=phases, zero=self.zero,
                                                 jax_support=jax_support)

        # create physics
        self.physics = Compositional(self.components, phases, self.timer, n_points=400, min_p=0, max_p=1000,
                                     min_z=self.zero, max_z=1 - self.zero)
        self.physics.add_property_region(property_container)

        return

    def run(self, days: float = None, restart_dt: float = 0., verbose: bool = True):
        """
        Method to run simulation for specified time. Optional argument to specify dt to restart simulation with.

        :param days: Time increment [days]
        :type days: float
        :param restart_dt: Restart value for timestep size [days, optional]
        :type restart_dt: float
        :param verbose: Switch for verbose, default is True
        :type verbose: bool
        """
        days = days if days is not None else self.runtime

        # get current engine time
        t = self.physics.engine.t
        stop_time = t + days

        # same logic as in engine.run
        if fabs(t) < 1e-15 or not hasattr(self, 'prev_dt'):
            dt = self.params.first_ts
        elif restart_dt > 0.:
            dt = restart_dt
        else:
            dt = min(self.prev_dt * self.params.mult_ts, self.params.max_ts)
        self.prev_dt = dt

        ts = 0

        while t < stop_time:
            converged = self.run_timestep(dt, t, verbose)

            if converged:
                t += dt
                self.physics.engine.t = t
                ts += 1
                if verbose:
                    print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                          % (ts, t, dt, self.physics.engine.n_newton_last_dt, self.physics.engine.n_linear_last_dt))

                dt = min(dt * self.params.mult_ts, self.params.max_ts)

                if t + dt > stop_time:
                    dt = stop_time - t
                else:
                    self.prev_dt = dt

                self.save_data_to_h5(kind='well')

            else:
                dt /= self.params.mult_ts
                if verbose:
                    print("Cut timestep to %2.10f" % dt)
                if dt < self.params.min_ts:
                    break

        # update current engine time
        self.physics.engine.t = stop_time

        # save solution vector
        self.save_data_to_h5(kind='solution')

        if verbose:
            print("TS = %d(%d), NI = %d(%d), LI = %d(%d)"
                  % (self.physics.engine.stat.n_timesteps_total, self.physics.engine.stat.n_timesteps_wasted,
                     self.physics.engine.stat.n_newton_total, self.physics.engine.stat.n_newton_wasted,
                     self.physics.engine.stat.n_linear_total, self.physics.engine.stat.n_linear_wasted))

    def run_timestep(self, dt: float, t: float, verbose: bool = True):
        """
        Method to solve Newton loop for specified timestep

        :param dt: Timestep size [days]
        :type dt: float
        :param t: Current time [days]
        :type t: float
        :param verbose: Switch for verbose, default is True
        :type verbose: bool
        """
        max_newt = self.params.max_i_newton
        max_residual = np.zeros(max_newt + 1)
        self.physics.engine.n_linear_last_dt = 0
        self.timer.node['simulation'].start()
        for i in range(max_newt+1):
            # self.physics.engine.run_single_newton_iteration(dt)
            if self.python_assembly:
                self.update_residual(dt)
            else:
                self.physics.engine.assemble_linear_system(dt)  # assemble Jacobian and residual of reservoir and well blocks
            self.apply_rhs_flux(dt, t)  # apply RHS flux
            self.physics.engine.newton_residual_last_dt = self.physics.engine.calc_newton_residual()  # calc norm of residual

            max_residual[i] = self.physics.engine.newton_residual_last_dt
            counter = 0
            for j in range(i):
                if abs(max_residual[i] - max_residual[j])/max_residual[i] < self.params.stationary_point_tolerance:
                    counter += 1
            if counter > 2:
                if verbose:
                    print("Stationary point detected!")
                break

            self.physics.engine.well_residual_last_dt = self.physics.engine.calc_well_residual()
            self.physics.engine.n_newton_last_dt = i
            #  check tolerance if it converges
            if ((self.physics.engine.newton_residual_last_dt < self.params.tolerance_newton and
                 self.physics.engine.well_residual_last_dt < self.params.well_tolerance_coefficient * self.params.tolerance_newton) or
                    self.physics.engine.n_newton_last_dt == self.params.max_i_newton):
                if i > 0:  # min_i_newton
                    break
            r_code = self.physics.engine.solve_linear_equation()
            self.timer.node["newton update"].start()
            self.physics.engine.apply_newton_update(dt)
            self.timer.node["newton update"].stop()
        # End of newton loop
        converged = self.physics.engine.post_newtonloop(dt, t)

        self.timer.node['simulation'].stop()
        return converged

    def test_operators(self, dt):
        X = jnp.asarray(self.physics.engine.X)
        RHS = np.asarray(self.physics.engine.RHS)
        jac_vals = self.physics.engine.jac_vals
        jac_rows = self.physics.engine.jac_rows
        jac_cols = self.physics.engine.jac_cols
        jac_diags = self.physics.engine.jac_diags
        PV = np.asarray(self.physics.engine.PV)
        RV = np.asarray(self.physics.engine.RV)
        cell_m = np.asarray(self.reservoir.mesh.block_m)
        cell_p = np.asarray(self.reservoir.mesh.block_p)

        n_res_blocks = self.reservoir.mesh.n_res_blocks
        n_vars = self.physics.n_vars
        n_state = n_vars
        n_conns = cell_m.size

        # JAX gradient evaluators of operators
        if not hasattr(self, 'jax_jac_evaluators'):
            self.jax_jac_evaluators = {}
            for region, op in self.physics.reservoir_operators.items():
                self.jax_jac_evaluators[region] = jax.jacfwd(op.evaluate_jax)

        # to test operators & derivatives
        if not hasattr(self, 'reservoir_operators'):
            self.property_containers = {}
            self.reservoir_operators = {}
            self.acc_flux_itor = {}
            for region in self.physics.regions:
                self.property_containers[region] = self.set_properties(comps=self.components, phases=self.physics.phases,
                                                                       zero=self.zero, jax_support=False)
                self.reservoir_operators[region] = ReservoirOperators(self.property_containers[region],
                                                                      self.physics.thermal)
                self.acc_flux_itor[region] = self.physics.create_interpolator(self.reservoir_operators[region],
                                                                              n_ops=self.physics.n_ops,
                                                                              platform='cpu', algorithm='multilinear',
                                                                              mode='adaptive', precision='d',
                                                                              timer_name='reservoir %d interpolation' % region, region=str(region))

        i = 0
        op_values = value_vector(np.zeros(self.physics.n_ops * n_res_blocks))
        op_dvalues = value_vector(np.zeros(self.physics.n_ops * n_res_blocks * self.physics.n_vars))
        op_values_np = np.asarray(op_values)
        op_dvalues_np = np.asarray(op_dvalues)
        for region, itor in self.acc_flux_itor.items():
            region_cell_idx = np.where(self.op_num == region)[0].astype(np.int32)
            itor.evaluate_with_derivatives(self.physics.engine.X, index_vector(region_cell_idx), op_values, op_dvalues)
            i += 1

        # loop over cells
        id_conn = 0
        for i in range(n_res_blocks):
            csr_idx_start = jac_rows[i]
            csr_idx_end = jac_rows[i + 1]
            cols = jac_cols[csr_idx_start:csr_idx_end]
            # states = X[np.concatenate([
            #     np.arange(col * n_vars, col * n_vars + n_state) for col in cols])]
            region = self.op_num[i]

            state = X[i * n_vars:i * n_vars + n_state]
            values = self.physics.reservoir_operators[region].evaluate_jax(state)
            grad_values = self.jax_jac_evaluators[region](state)

            # check operators
            assert(np.isclose(values, op_values_np[i * self.physics.n_ops:(i + 1) * self.physics.n_ops],
                       rtol=1e-5, atol=1e-5).all())
            assert(np.isclose(grad_values.flatten(),
                       op_dvalues_np[i * self.physics.n_ops * self.physics.n_vars:(i + 1) * self.physics.n_ops * self.physics.n_vars],
                       rtol=1e-4, atol=1e-5).all())

        aa=3

    def init_assembly(self):
        self.PV = np.asarray(self.physics.engine.PV)
        self.RV = np.asarray(self.physics.engine.RV)
        self.n_vars = self.physics.n_vars
        self.n_ops = self.physics.n_ops
        self.n_state = self.n_vars
        self.n_ph = self.physics.nph
        self.n_dim = 3
        self.n_res_blocks = self.reservoir.mesh.n_res_blocks

        op = self.physics.reservoir_operators[self.op_num[0]]
        self.ACC_OP = op.ACC_OP
        self.FLUX_OP = op.FLUX_OP
        self.GRAV_OP = op.GRAV_OP
        self.PC_OP = op.PC_OP

        self.cell_m = np.asarray(self.reservoir.mesh.block_m)
        self.cell_p = np.asarray(self.reservoir.mesh.block_p)
        self.tran = np.asarray(self.reservoir.mesh.tran)
        self.grav_coef = np.asarray(self.reservoir.mesh.grav_coef)
        self.n_conns = self.cell_m.size

        self.jacobian_fun = jax.jacfwd(self.compute_residual, argnums=0)

    def compute_residual(self, states, state_n, cell_id, stencil, conn_id, dt):
        op_m = self.operator_cache.evaluate(cell_id, states[self.n_vars * stencil[cell_id]:self.n_vars * (stencil[cell_id] + 1)])
        op_m_n = self.physics.reservoir_operators[self.op_num[cell_id]].evaluate_jax(state_n)

        rhs_contribution = self.PV[cell_id] * (op_m[self.ACC_OP:self.ACC_OP + self.n_vars] -
                                               op_m_n[self.ACC_OP:self.ACC_OP + self.n_vars])

        assert(self.cell_m[conn_id] == cell_id)
        while (conn_id < self.n_conns and self.cell_m[conn_id] == cell_id):
            cell_m = self.cell_m[conn_id]
            cell_p = self.cell_p[conn_id]
            op_p = self.operator_cache.evaluate(cell_p, states[self.n_vars * stencil[cell_p]:self.n_vars * (stencil[cell_p] + 1)])

            dp = states[self.n_vars * stencil[cell_p]] - states[self.n_vars * stencil[cell_m]]
            avg_phase_density = (op_m[self.GRAV_OP:self.GRAV_OP + self.n_ph] + op_p[self.GRAV_OP:self.GRAV_OP + self.n_ph]) / 2
            phase_dp = (dp + avg_phase_density * self.grav_coef[conn_id] +
                        op_m[self.PC_OP:self.PC_OP + self.n_ph] - op_p[self.PC_OP:self.PC_OP + self.n_ph])
            flux_op = jnp.zeros((self.n_ph, self.n_state))
            left = phase_dp < 0
            right = jnp.logical_not(left)
            flux_op.at[left].set(op_m[self.FLUX_OP:self.FLUX_OP + self.n_state * self.n_ph].reshape(self.n_ph, self.n_state)[left])
            flux_op.at[right].set(op_p[self.FLUX_OP:self.FLUX_OP + self.n_state * self.n_ph].reshape(self.n_ph, self.n_state)[right])
            rhs_contribution -= dt * self.tran[conn_id] * jnp.matmul(phase_dp, flux_op)
            conn_id += 1

        return rhs_contribution

    def update_residual(self, dt):
        X = jnp.asarray(self.physics.engine.X)
        Xn = np.asarray(self.physics.engine.Xn)
        RHS = jnp.asarray(self.physics.engine.RHS)
        jac_vals = self.physics.engine.jac_vals
        jac_rows = self.physics.engine.jac_rows
        jac_cols = self.physics.engine.jac_cols
        jac_diags = self.physics.engine.jac_diags

        n_res_blocks = self.reservoir.mesh.n_res_blocks
        n_vars = self.physics.n_vars
        n_ops = self.physics.n_ops
        n_state = n_vars

        # loop over cells
        conn_id = 0
        self.operator_cache = OperatorCache(self.physics.reservoir_operators, self.op_num)

        for i in range(n_res_blocks):
            csr_idx_start = jac_rows[i]
            csr_idx_end = jac_rows[i + 1]
            cols = jac_cols[csr_idx_start:csr_idx_end]

            states = X[np.concatenate([
                np.arange(col * n_vars, col * n_vars + n_state) for col in cols])]
            state_n = Xn[i * n_vars:i * n_vars + n_state]
            state_n = lax.stop_gradient(state_n)
            stencil = {cols[idx]: idx for idx in range(cols.size)}

            RHS_contribution = self.compute_residual(states=states, state_n=state_n,
                                                     cell_id=i, stencil=stencil, conn_id=conn_id, dt=dt)
            RHS = RHS.at[i * n_vars:(i + 1) * n_vars].set(RHS_contribution)
            Jacobian_i = self.jacobian_fun(states, state_n, i, stencil, conn_id, dt)
            conn_id += cols.size - 1

        return RHS

class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, min_z=min_z, temperature=1.)

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

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        self.ph = [0, 1]

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

        return

    def evaluate_at_cond(self, pressure, zc):
        self.sat[:] = 0

        ph = [0, 1]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.dens_m = [1025, 0.77]  # to match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m

class OperatorCache:
    def __init__(self, operators, op_num):
        self.operators = operators
        self.op_num = op_num
        self.cache = {}
    def evaluate(self, cell_id, state):
        if cell_id in self.cache:
            return self.cache[cell_id]
        else:
            op_value = self.operators[self.op_num[cell_id]].evaluate_jax(state)
            self.cache[cell_id] = op_value
            return op_value

class ModelPropertiesJAX(PropertyContainerJAX):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        self.x = jnp.eye(self.nph)
        self.ph = jnp.array([0, 1])  # Assuming a two-phase system
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, min_z=min_z, temperature=1.)

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

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        self.ph = [0, 1]

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

        return

    def evaluate_jax(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        pressure = state[0]
        zc = jnp.append(state[1:], 1. - jnp.sum(state[1:]))
        self.x = jnp.eye(self.nph)

        M = jnp.sum(self.x * self.Mw, axis=1)
        density = jnp.array([self.density_ev[name].evaluate(pressure) for name in self.phases_name])
        dens_m = density / M
        viscosity = jnp.array([self.viscosity_ev[name].evaluate() for name in self.phases_name])

        # Compute saturation assuming `compute_saturation` is a pure function
        sat = self.compute_saturation(zc, dens_m)

        kr = jnp.array([self.rel_perm_ev[name].evaluate_jax(sat[j]) for j, name in enumerate(self.phases_name)])
        pc = jnp.zeros_like(density)  # Simplified assumption for capillary pressure

        return {
            'density': density,
            'molar_density': dens_m,
            'viscosity': viscosity,
            'saturation': sat,
            'relative_permeability': kr,
            'capillary_pressure': pc
        }

    def evaluate_at_cond(self, pressure, zc):
        self.sat[:] = 0

        ph = [0, 1]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.dens_m = [1025, 0.77]  # to match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m
