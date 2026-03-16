import numpy as np

from darts.engines import index_vector, value_vector
from darts.physics.base.operators_base import PropertyOperators
from darts.physics.base.physics_base import PhysicsBase


class Initialize:
    def __init__(
        self,
        physics,
        algorithm: str = 'multilinear',
        mode: str = 'adaptive',
        is_barycentric: bool = False,
        aq_idx: int = None,
        h2o_idx: int = None,
    ):
        """
        Constructor for Initialize class. It solves the equilibrated vertical distribution in the PT-domain

        :param physics: Physics object
        :param algorithm: Type of interpolation (multilinear/linear), default is multilinear
        :param mode: Interpolation mode (static/adaptive), default is adaptive
        :param is_barycentric: Bool for barycentric interpolation, default is False
        :param aq_idx: Index of Aq phase
        :param h2o_idx: Index of H2O-component
        """
        self.physics = physics
        self.nv = physics.n_vars
        self.thermal = physics.thermal
        self.nc = self.nv - self.thermal
        self.nph = physics.nph

        # Index of pressure, temperature and components
        self.vars = (
            ['pressure']
            + self.physics.components[:-1]
            + (['temperature'] if self.thermal else [])
        )
        self.var_idxs = {var: i for i, var in enumerate(self.vars)}

        # Add evaluators of phase saturations, rhoT and dX (if kinetic reactions are defined)
        pc = physics.property_containers[0]
        continuous_sat = lambda: np.sum(
            [pc.sat[j] for j in range(pc.np_fl) if pc.kr[j] > 1e-8]
        )
        self.props = {
            'rhoT': lambda: np.sum(
                [pc.sat[j] * pc.dens[j] for j in range(pc.np_fl) if pc.kr[j] > 1e-8]
            )
            / continuous_sat(),
            'pressure': lambda: pc.pressure,
            'temperature': lambda: pc.temperature,
        }
        self.props.update(
            {
                comp: lambda i=i: np.nansum(pc.nu * pc.x[:, i])
                for i, comp in enumerate(self.physics.components)
            }
        )
        self.props.update(
            {
                'm_' + comp: lambda i=i: np.nansum(
                    pc.dens_m * pc.sat * pc.x[:, i] * pc.Mw[i]
                )
                for i, comp in enumerate(self.physics.components)
            }
        )  # kg/m3 of component i
        self.props.update(
            {
                'pot' + ph: lambda j=j: pc.pressure - pc.pc[j]
                for j, ph in enumerate(self.physics.phases)
            }
        )
        self.props.update(
            {
                'mob' + ph: lambda j=j: pc.kr[j] / pc.mu[j] if pc.mu[j] else 0.0
                for j, ph in enumerate(physics.phases)
            }
        )
        self.props.update(
            {'sat' + ph: lambda j=j: pc.sat[j] for j, ph in enumerate(physics.phases)}
        )
        self.props.update(
            {
                'x' + str(i) + ph: lambda i=i, j=j: pc.x[j, i]
                for i in range(pc.nc_fl)
                for j, ph in enumerate(physics.phases[: pc.np_fl])
            }
        )
        if aq_idx is not None:
            self.props.update(
                {
                    'm' + str(i): lambda i=i: 55.509
                    * pc.x[aq_idx, i]
                    / pc.x[aq_idx, h2o_idx]
                    for i in range(pc.nc_fl)
                }
            )
        self.props.update(
            {
                'dX' + str(k): lambda k=k: pc.dX[k]
                for k, kr in enumerate(pc.kinetic_rate_ev)
            }
        )

        self.props_idxs = {prop: i for i, prop in enumerate(self.props.keys())}
        self.primary_specs = {}
        self.secondary_specs = {}

        # If PH-formulation, evaluate_PT method must be called in the evaluate() during Initialize
        self.evaluate_PT_bool = physics.state_spec > PhysicsBase.StateSpecification.PT

        # Create PropertyOperators and interpolators
        self.etor = PropertyOperators(
            pc,
            self.thermal,
            self.props,
            extrapolation_flag=self.physics.extrapolation_flag,
            dz=self.physics.dz,
        )
        self.itor, n_ops = physics.create_interpolator(
            evaluator=self.etor,
            n_ops=physics.n_ops,
            axes_min=value_vector(self.physics.PT_axes_min),
            axes_max=value_vector(self.physics.PT_axes_max),
            timer_name='initialization itor',
            algorithm=algorithm,
            mode=mode,
            is_barycentric=is_barycentric,
        )
        self.n_ops = n_ops

    def evaluate(self, Xi: list, region_idx: int = 0):
        """
        Function to return array of properties.
        Primary variables (vars) are obtained from engine, secondary variables (props) are interpolated by property_itor.

        :param Xi: State
        :type Xi: list
        :param region_idx: Property region index, default is 0
        :returns: property_array
        :rtype: np.ndarray
        """
        # Set flag to evaluate_PT in case of PH/PS-formulation
        pc = self.physics.property_containers[region_idx]
        pc.evaluate_PT_bool = self.evaluate_PT_bool

        # Interpolate values and derivatives in property_itor
        state_idxs = index_vector([0])
        values = value_vector(np.zeros(self.n_ops))
        derivs = value_vector(np.zeros(self.n_ops * self.nv))

        self.itor.evaluate_with_derivatives(
            value_vector(Xi), state_idxs, values, derivs
        )

        # Switch evaluate_PT boolean off to flash.evaluate() during simulation again
        pc.evaluate_PT_bool = False

        return values, derivs

    def init_depth_table(
        self,
        depth_bottom: float,
        depth_top: float,
        depth_known: float,
        X0: np.ndarray,
        nb: int = 100,
        dTdh: float = 0.03,
    ):
        """
        Method to initialize depth table of nb grid blocks from min, max and known depth.

        :param depth_bottom: Depth of bottom block
        :param depth_top: Depth of top block
        :param depth_known: Depth of known conditions
        :param X0: Known conditions
        :param nb: Number of grid blocks
        :param dTdh: Temperature gradient [K/m]
        :return:
        """
        if nb == 1:
            self.depths = np.array([depth_known])
            bc_idx = 0
        else:
            # Check input and create depths
            assert depth_bottom >= depth_top, "Top depth is below bottom depth"
            assert depth_top <= depth_known <= depth_bottom, (
                "Known depth is not in range [bottom, top]"
            )
            self.depths = np.linspace(start=depth_top, stop=depth_bottom, num=nb)
            bc_idx = (np.fabs(self.depths - depth_known)).argmin()
            self.depths[bc_idx] = depth_known

        # Define thermal gradient
        values0, _ = self.evaluate(X0)
        if self.thermal:
            self.T = (
                lambda i: values0[self.props_idxs['temperature']]
                + (self.depths[i] - self.depths[bc_idx]) * dTdh
            )

        # Set state in known cell
        X = np.zeros((nb, self.nv))
        X[bc_idx] = X0

        return X, bc_idx

    def solve_state(
        self,
        Xi: list,
        specs: dict,
        max_iter: int = 100,
    ):
        """
        Solve for boundary state

        :param Xi: State
        :type Xi: list
        :param specs: Specifications (primary and secondary variables)
        :type specs: dict
        :param max_iter: Maximum number of iterations
        """
        assert (
            int(np.sum([not np.isnan(np.float64(spec)) for spec in specs.values()]))
            == self.nv
        ), (
            "Not enough variables specified for well-defined system of equations, {} specified but {} needed".format(
                int(
                    np.sum([not np.isnan(np.float64(spec)) for spec in specs.values()])
                ),
                self.nv,
            )
        )

        for _it in range(max_iter):
            res = np.zeros(self.nv)
            Jac = np.zeros((self.nv, self.nv))
            values, derivs = self.evaluate(Xi)

            # Specification equations of primary and secondary variables defined in self.props
            res_idx = 0
            for var, spec in specs.items():
                if not np.isnan(np.float64(spec)):
                    prop_idx = self.props_idxs[var]
                    res[res_idx] = values[prop_idx] - spec

                    for jj in range(self.nv):
                        Jac[res_idx, jj] = derivs[prop_idx * self.nv + jj]
                    res_idx += 1

            # Solve Newton step
            dX = np.linalg.solve(Jac, res)

            # Calculate damping factor to remain within all positive mole fractions
            betas_min, betas_max = np.empty(self.nc - 1), np.empty(self.nc - 1)
            for i in range(1, self.nc):
                if np.abs(dX[i]) > 1e-15:
                    betas_min[i - 1] = Xi[i] / dX[i]
                    betas_max[i - 1] = -(1.0 - Xi[i]) / dX[i]
                else:
                    betas_min[i - 1] = (
                        np.sign(dX[i]) * np.inf if np.sign(dX[i]) else np.inf
                    )
                    betas_max[i - 1] = (
                        -np.sign(dX[i]) * np.inf if np.sign(dX[i]) else -np.inf
                    )

            beta = min(
                1,
                min(
                    (
                        np.amin(betas_min[betas_min > 0])
                        if len(betas_min[betas_min > 0]) > 0
                        else 1
                    ),
                    (
                        np.amin(betas_max[betas_max > 0])
                        if len(betas_max[betas_max > 0]) > 0
                        else 1
                    ),
                ),
            )

            beta = 1.0 if beta == 1.0 else beta * 0.1
            Xi[1:-1] -= beta * dX[1:-1]
            Xi[0] -= dX[0]
            Xi[-1] -= dX[-1]

            norm = np.linalg.norm(res)
            if norm < 1e-10:
                return Xi

        print("MAX ITER REACHED FOR INITIALIZATION", Xi)
        return Xi

    def solve(
        self,
        X: np.ndarray,
        bc_idx: int,
        specs: dict,
        downward: bool = True,
        region_idx: int = 0,
        max_iter: int = 100,
    ):
        """
        Solve for all depths

        :param X: State vector for all depths
        :type X: np.ndarray
        :param bc_idx: Index of known cell
        :type bc_idx: bool
        :param specs: Set of specifications (primary and secondary variables)
        :type specs: dict
        :param region_idx: Index of property region for property container
        :type region_idx: int
        :param max_iter: Maximum number of iterations
        :type max_iter: int
        """
        # If only one block has been specified, return the boundary state
        if len(self.depths) == 1:
            return X

        # Else, solve specification equations (primary and secondary variables)
        temp_idx = self.props_idxs['temperature']
        rhoT_idx = self.props_idxs['rhoT']
        pot_idx = self.props_idxs['pot' + self.physics.phases[0]]
        # mob_idx = self.props_idxs['mob' + self.physics.phases[0]]
        # sat_idx = self.props_idxs['sat' + self.physics.phases[0]]

        n_vars = self.nv  # - self.thermal

        # Solve cells from specified cell upwards
        cell_range = (
            range(bc_idx, len(self.depths) - 1) if downward else range(bc_idx, 0, -1)
        )
        for i in cell_range:
            # Find neighbouring cell for which state is known
            cell_idx = i + 1 if downward else i - 1
            known_idx = cell_idx - 1 if downward else cell_idx + 1

            values0, _ = self.evaluate(X[known_idx])
            gh0 = 9.81 * self.depths[known_idx] * 1e-5
            gh1 = 9.81 * self.depths[cell_idx] * 1e-5

            # Solve nonlinear unknowns
            # Initialize using same composition, recalculate pressure and evaluate temperature gradient
            X[cell_idx, 0] = values0[pot_idx] + values0[rhoT_idx] * (gh1 - gh0)
            X[cell_idx, 1:] = X[known_idx, 1:]
            if self.thermal:
                X[cell_idx, -1] = self.T(cell_idx)

            for _it in range(max_iter):
                # nc variables for pressure and nc-1 compositions, temperature is calculated from gradient
                res = np.zeros(n_vars)
                Jac = np.zeros((n_vars, n_vars))

                # Evaluate operators and derivatives at current state Xi
                values1, derivs1 = self.evaluate(X[cell_idx])

                # Zero mass flux equations for fluid phases
                mgh = (values1[rhoT_idx] + values0[rhoT_idx]) * (gh1 - gh0) / 2

                potential_diff = values0[pot_idx] - values1[pot_idx] + mgh
                res[0] = potential_diff
                for k in range(n_vars):
                    Jac[0, k] += (
                        -derivs1[pot_idx * self.nv + k]
                        + derivs1[rhoT_idx * self.nv + k] * (gh1 - gh0) / 2
                    )

                # Specification equations of primary and secondary variables defined in self.props
                res_idx = 1
                for var, spec in specs.items():
                    if not np.isnan(np.float64(spec)):
                        prop_idx = self.props_idxs[var]
                        res[res_idx] = values1[prop_idx] - spec

                        for jj in range(self.nv):
                            Jac[res_idx, jj] = derivs1[prop_idx * self.nv + jj]
                        res_idx += 1

                if self.thermal:
                    res[-1] = values1[temp_idx] - self.T(cell_idx)
                    for k in range(n_vars):
                        Jac[-1, k] = derivs1[temp_idx * self.nv + k]

                # Solve Newton step
                dX = np.linalg.solve(Jac, res)

                # Calculate damping factor to remain within all positive mole fractions
                betas_min, betas_max = np.empty(self.nc - 1), np.empty(self.nc - 1)
                for i in range(1, self.nc):
                    if np.abs(dX[i]) > 1e-15:
                        betas_min[i - 1] = X[cell_idx, i] / dX[i]
                        betas_max[i - 1] = -(1.0 - X[cell_idx, i]) / dX[i]
                    else:
                        betas_min[i - 1] = (
                            np.sign(dX[i]) * np.inf if np.sign(dX[i]) else np.inf
                        )
                        betas_max[i - 1] = (
                            -np.sign(dX[i]) * np.inf if np.sign(dX[i]) else -np.inf
                        )

                beta = min(
                    1,
                    min(
                        (
                            np.amin(betas_min[betas_min > 0])
                            if len(betas_min[betas_min > 0]) > 0
                            else 1
                        ),
                        (
                            np.amin(betas_max[betas_max > 0])
                            if len(betas_max[betas_max > 0]) > 0
                            else 1
                        ),
                    ),
                )

                X[cell_idx, :n_vars] -= beta * dX

                if np.linalg.norm(res) < 1e-10:
                    break

            if _it > max_iter:
                print("MAX ITER REACHED FOR INITIALIZATION", X[cell_idx, :])

        return X

    def solve_region(
        self,
        X: np.ndarray,
        mobile_phases: list,
        wetting_phase: str,
        residual_saturations: dict,
        bc_idx: int,
        specs: dict = None,
        downward: bool = True,
        region_idx: int = 0,
        max_iter: int = 100,
    ):
        """
        Solve for specific depth

        :param X: State vector of all depths
        :type X: np.ndarray
        :param cell_idx: Index of cell to solve
        :param downward: Bool to indicate if known cell is above or below
        :param max_iter: Maximum number of iterations
        """
        # Set primary and secondary specs to empty dictionary if None
        specs = specs if specs is not None else {}

        # Find indices of phases and properties
        mobile_phases_idxs = [self.physics.phases.index(ph) for ph in mobile_phases]
        wetting_phase_idx = self.physics.phases.index(wetting_phase)

        temp_idx = self.props_idxs['temperature']
        rhoT_idx = self.props_idxs['rhoT']
        pot_idx = self.props_idxs['pot' + self.physics.phases[0]]
        mob_idx = self.props_idxs['mob' + self.physics.phases[0]]
        sat_idx = self.props_idxs['sat' + self.physics.phases[0]]

        n_vars = self.nv  # - self.thermal

        # Solve cells from specified cell upwards
        cell_range = (
            range(bc_idx, len(self.depths) - 1) if downward else range(bc_idx, 0, -1)
        )
        for i in range(cell_range):
            # Find neighbouring cell for which state is known
            cell_idx = i + 1 if downward else i - 1
            known_idx = cell_idx - 1 if downward else cell_idx + 1

            values0, _ = self.evaluate(X[known_idx])
            gh0 = 9.81 * self.depths[known_idx] * 1e-5
            gh1 = 9.81 * self.depths[cell_idx] * 1e-5

            # Solve nonlinear unknowns
            # Initialize using same composition, recalculate pressure and evaluate temperature gradient
            X[cell_idx, 0] = values0[pot_idx] + values0[rhoT_idx] * (gh1 - gh0)
            X[cell_idx, 1:] = X[known_idx, 1:]

            for _it in range(max_iter):
                # nc variables for pressure and nc-1 compositions, temperature is calculated from gradient
                res = np.zeros(n_vars)
                Jac = np.zeros((n_vars, n_vars))

                # Evaluate operators and derivatives at current state Xi
                values1, derivs1 = self.evaluate(X[cell_idx])

                # Zero mass flux equations for fluid phases
                mgh = (values1[rhoT_idx] + values0[rhoT_idx]) * (gh1 - gh0) / 2
                # j1, j2 = 0, 0
                for j, phase_idx in enumerate(mobile_phases_idxs):
                    # Potential difference of phase j: P[1] - Pc[1] - (P[2] - Pc[2]) + mgh[av]
                    potential_diff = (
                        values0[pot_idx + phase_idx]
                        - values1[pot_idx + phase_idx]
                        + mgh
                    )

                    if j == wetting_phase_idx:
                        if (
                            values1[sat_idx + phase_idx]
                            < residual_saturations[mobile_phases[j]]
                        ):
                            res[j] = (
                                values1[sat_idx + phase_idx]
                                - residual_saturations[mobile_phases[j]]
                            )

                            for k in range(n_vars):
                                Jac[j, k] = derivs1[(sat_idx + phase_idx) * self.nv + k]
                        else:
                            res[j] = values1[mob_idx + phase_idx] * potential_diff
                            for k in range(n_vars):
                                Jac[j, k] += values1[mob_idx + phase_idx] * (
                                    -derivs1[(pot_idx + phase_idx) * self.nv + k]
                                    + derivs1[rhoT_idx * self.nv + k] * (gh1 - gh0) / 2
                                )
                                Jac[j, k] += (
                                    derivs1[(mob_idx + phase_idx) * self.nv + k]
                                    * potential_diff
                                )
                    else:
                        res[j] = potential_diff
                        for k in range(n_vars):
                            Jac[j, k] += (
                                -derivs1[(pot_idx + phase_idx) * self.nv + k]
                                + derivs1[rhoT_idx * self.nv + k] * (gh1 - gh0) / 2
                            )

                # Specification equation
                res_idx = len(mobile_phases)
                for var, spec in specs.items():
                    spec_ = (
                        np.float64(spec[cell_idx])
                        if hasattr(spec, '__len__')
                        else np.float64(spec)
                    )
                    if not np.isnan(spec_):
                        prop_idx = self.props_idxs[var]
                        res[res_idx] = values1[prop_idx] - spec_

                        for jj in range(n_vars):
                            Jac[res_idx, jj] = derivs1[prop_idx * self.nv + jj]
                        res_idx += 1

                if self.thermal:
                    res[-1] = values1[temp_idx] - self.T(cell_idx)
                    for k in range(n_vars):
                        Jac[-1, k] = derivs1[temp_idx * self.nv + k]

                # Solve Newton step
                dX = np.linalg.solve(Jac, res)

                # Calculate damping factor to remain within all positive mole fractions
                betas_min, betas_max = np.empty(self.nc - 1), np.empty(self.nc - 1)
                for i in range(1, self.nc):
                    if np.abs(dX[i]) > 1e-15:
                        betas_min[i - 1] = X[cell_idx, i] / dX[i]
                        betas_max[i - 1] = -(1.0 - X[cell_idx, i]) / dX[i]
                    else:
                        betas_min[i - 1] = (
                            np.sign(dX[i]) * np.inf if np.sign(dX[i]) else np.inf
                        )
                        betas_max[i - 1] = (
                            -np.sign(dX[i]) * np.inf if np.sign(dX[i]) else -np.inf
                        )

                beta = min(
                    1.0,
                    min(
                        (
                            np.amin(betas_min[betas_min > 0.0])
                            if len(betas_min[betas_min > 0.0]) > 0
                            else 1.0
                        ),
                        (
                            np.amin(betas_max[betas_max > 0.0])
                            if len(betas_max[betas_max > 0.0]) > 0
                            else 1.0
                        ),
                    ),
                )

                beta = 1.0 if beta == 1.0 else beta * 0.1
                X[cell_idx, 1:-1] -= beta * dX[1:-1]
                X[cell_idx, 0] -= dX[0]
                X[cell_idx, -1] -= dX[-1]

                norm = np.linalg.norm(res)
                if norm < 1e-10:
                    break

            if _it > max_iter:
                print("MAX ITER REACHED FOR INITIALIZATION", X[cell_idx, :])

        return X
