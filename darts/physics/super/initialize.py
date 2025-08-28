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

        # Index of pressure, temperature and components
        self.vars = (
            ['pressure']
            + self.physics.components[:-1]
            + (['temperature'] if self.thermal else [])
        )
        self.var_idxs = {var: i for i, var in enumerate(self.vars)}

        # Add evaluators of phase saturations, rhoT and dX (if kinetic reactions are defined)
        pc = physics.property_containers[0]
        self.props = {}
        self.props.update({'rhoT': lambda: np.sum(pc.sat * pc.dens)})
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
                    'm'
                    + str(i): lambda i=i: 55.509
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
        pc.evaluate_PT_bool = physics.state_spec > PhysicsBase.StateSpecification.PT

        # Create PropertyOperators and interpolators
        self.etor = PropertyOperators(pc, self.thermal, self.props)
        self.itor = physics.create_interpolator(
            evaluator=self.etor,
            n_ops=physics.n_ops,
            axes_min=value_vector(self.physics.PT_axes_min),
            axes_max=value_vector(self.physics.PT_axes_max),
            timer_name='initialization itor',
            algorithm=algorithm,
            mode=mode,
            is_barycentric=is_barycentric,
        )

    def evaluate(self, Xi: list):
        """
        Function to return array of properties.
        Primary variables (vars) are obtained from engine, secondary variables (props) are interpolated by property_itor.

        :param Xi: State
        :type Xi: list
        :returns: property_array
        :rtype: np.ndarray
        """
        # Interpolate values and derivatives in property_itor
        state_idxs = index_vector([0])
        values = value_vector(np.zeros(self.physics.n_ops))
        derivs = value_vector(np.zeros(self.physics.n_ops * self.nv))

        self.itor.evaluate_with_derivatives(
            value_vector(Xi), state_idxs, values, derivs
        )

        return values, derivs

    def solve(
        self,
        depth_bottom: float,
        depth_top: float,
        depth_known: float,
        boundary_state: dict,
        primary_specs: dict = None,
        secondary_specs: dict = None,
        nb: int = 100,
        dTdh: float = 0.03,
    ):
        """
        Solve for all depths

        :param depth_bottom: Bottom depth
        :param depth_top: Top depth
        :param depth_known: Depth of known conditions
        :param boundary_state: Known conditions, set of primary and secondary specifications
        :type boundary_state: dict
        :param primary_specs: Primary specifications
        :type primary_specs: dict
        :param secondary_specs: Secondary specifications
        :type secondary_specs: dict
        :param nb: Number of depth values
        :param dTdh: Temperature gradient
        """
        # If only one block has been specified, return the boundary state
        if nb == 1:
            self.depths = np.array([depth_known])
            return np.array(
                [boundary_state[variable] for variable in self.var_idxs.keys()]
            )

        # Else, check input and create depths
        assert depth_bottom >= depth_top, "Top depth is below bottom depth"
        assert (
            depth_top <= depth_known <= depth_bottom
        ), "Known depth is not in range [bottom, top]"
        self.depths = np.linspace(start=depth_top, stop=depth_bottom, num=nb)
        bc_idx = (np.fabs(self.depths - depth_known)).argmin()
        self.depths[bc_idx] = depth_known

        # Set primary and secondary specifications
        if primary_specs:
            for spec, values in primary_specs.items():
                self.primary_specs[spec] = (
                    values
                    if isinstance(values, (list, np.ndarray))
                    else np.ones(nb) * values
                )
                assert len(self.primary_specs[spec]) == nb, (
                    "Length of " + spec + " not compatible"
                )
        if secondary_specs:
            for spec, values in secondary_specs.items():
                self.secondary_specs[spec] = (
                    values
                    if isinstance(values, (list, np.ndarray))
                    else np.ones(nb) * values
                )
                assert len(self.secondary_specs[spec]) == nb, (
                    "Length of " + spec + " not compatible"
                )
        for i in range(nb):
            assert (
                int(
                    np.sum(
                        [
                            not np.isnan(np.float64(spec[i]))
                            for spec in self.primary_specs.values()
                        ]
                    )
                    + np.sum(
                        [
                            not np.isnan(np.float64(spec[i]))
                            for spec in self.secondary_specs.values()
                        ]
                    )
                )
                == self.nv - 1 - self.thermal
            ), "Not the right number of variables specified for well-defined system of equations in block {}, need {}".format(
                i, self.nv - 1 - self.thermal
            )

        # Define thermal gradient
        if self.thermal:
            self.T = (
                lambda i: boundary_state['temperature']
                + (self.depths[i] - self.depths[bc_idx]) * dTdh
            )

        # Set state in known cell
        X = np.zeros((nb, self.nv))
        X[bc_idx] = np.array([boundary_state[v] for v in self.vars])

        # Solve cells from specified cell upwards
        for i in range(bc_idx, 0, -1):
            self.solve_cell(X, cell_idx=i - 1, downward=False)

        # Solve cells from specified cell downwards
        for i in range(bc_idx, nb - 1):
            self.solve_cell(X, cell_idx=i + 1, downward=True)

        # Switch evaluate_PT boolean off to flash.evaluate() during simulation again
        self.physics.property_containers[0].evaluate_PT_bool = False

        return X.flatten()

    def solve_state(
        self,
        Xi: list,
        primary_specs: dict = None,
        secondary_specs: dict = None,
        max_iter: int = 100,
    ):
        """
        Solve for all depths

        :param Xi: State
        :type Xi: list
        :param primary_specs: Primary specifications
        :type primary_specs: dict
        :param secondary_specs: Secondary specifications
        :type secondary_specs: dict
        :param max_iter: Maximum number of iterations
        """
        assert (
            int(
                np.sum(
                    [not np.isnan(np.float64(spec)) for spec in primary_specs.values()]
                )
                + np.sum(
                    [
                        not np.isnan(np.float64(spec))
                        for spec in secondary_specs.values()
                    ]
                )
            )
            == self.nv
        ), "Not enough variables specified for well-defined system of equations, {} specified but {} needed".format(
            int(
                np.sum(
                    [not np.isnan(np.float64(spec)) for spec in primary_specs.values()]
                )
                + np.sum(
                    [
                        not np.isnan(np.float64(spec))
                        for spec in secondary_specs.values()
                    ]
                )
            ),
            self.nv,
        )

        for it in range(max_iter):
            res = np.zeros(self.nv)
            Jac = np.zeros((self.nv, self.nv))
            values, derivs = self.evaluate(Xi)

            # Specification of primary variables
            j1 = 0
            for var, spec in primary_specs.items():
                if not np.isnan(np.float64(spec)):
                    var_idx = self.var_idxs[var]
                    Xi[var_idx] = spec

                    res[j1] = Xi[var_idx] - spec
                    Jac[j1, :] = 0.0
                    Jac[j1, var_idx] = 1.0
                    j1 += 1

            # Specification of secondary variables
            j2 = 0
            for var, spec in secondary_specs.items():
                if not np.isnan(np.float64(spec)):
                    prop_idx = self.props_idxs[var]
                    res_idx = j1 + j2
                    res[res_idx] = values[prop_idx] - spec

                    for jj in range(self.nv):
                        Jac[res_idx, jj] = derivs[prop_idx * self.nv + jj]
                    j2 += 1

            # Solve Newton step
            dX = np.linalg.solve(Jac, res)
            Xi -= dX

            if np.linalg.norm(res) < 1e-10:
                return Xi

        print("MAX ITER REACHED FOR INITIALIZATION", Xi)
        return Xi

    def solve_cell(
        self, X: np.ndarray, cell_idx: int, downward: bool = True, max_iter: int = 100
    ):
        """
        Solve for specific depth

        :param X: State vector of all depths
        :type X: np.ndarray
        :param cell_idx: Index of cell to solve
        :param downward: Bool to indicate if known cell is above or below
        :param max_iter: Maximum number of iterations
        """
        # Find neighbouring cell for which state is known
        known_idx = cell_idx - 1 if downward else cell_idx + 1
        rhoT_idx = self.props_idxs['rhoT']

        n_vars = self.nv - self.thermal
        values0, _ = self.evaluate(X[known_idx])
        gh0 = 9.81 * self.depths[known_idx] * 1e-5
        gh1 = 9.81 * self.depths[cell_idx] * 1e-5

        # Solve nonlinear unknowns
        # Initialize using same composition, recalculate pressure and evaluate temperature gradient
        X[cell_idx, 0] = X[known_idx, 0] + values0[rhoT_idx] * (gh1 - gh0)
        X[cell_idx, 1:] = X[known_idx, 1:]
        if self.thermal:
            X[cell_idx, -1] = self.T(cell_idx)

        for it in range(max_iter):
            # nc variables for pressure and nc-1 compositions, temperature is calculated from gradient
            res = np.zeros(n_vars)
            Jac = np.zeros((n_vars, n_vars))

            # Evaluate operators and derivatives at current state Xi
            values1, derivs1 = self.evaluate(X[cell_idx])

            # Pressure equation
            mgh = (values1[rhoT_idx] + values0[rhoT_idx]) * (gh1 - gh0) / 2
            res[0] = X[known_idx, 0] + mgh - X[cell_idx, 0]
            Jac[0, 0] -= 1.0
            for j in range(n_vars):
                Jac[0, j] += derivs1[rhoT_idx * self.nv + j] * (gh1 - gh0) / 2

            # Specification equation
            j1 = 0
            for var, spec in self.primary_specs.items():
                if not np.isnan(np.float64(spec[cell_idx])):
                    var_idx = self.var_idxs[var]
                    X[cell_idx, var_idx] = spec[cell_idx]

                    res_idx = j1 + 1
                    res[res_idx] = X[cell_idx, var_idx] - spec[cell_idx]
                    Jac[res_idx, :] = 0.0
                    Jac[res_idx, var_idx] = 1.0
                    j1 += 1

            # Specification of secondary variables
            j2 = 0
            for var, spec in self.secondary_specs.items():
                if not np.isnan(np.float64(spec[cell_idx])):
                    prop_idx = self.props_idxs[var]
                    res_idx = j1 + j2 + 1
                    res[res_idx] = values1[prop_idx] - spec[cell_idx]

                    for jj in range(n_vars):
                        Jac[res_idx, jj] = derivs1[prop_idx * self.nv + jj]
                    j2 += 1

            dX = np.linalg.solve(Jac, res)
            X[cell_idx, :n_vars] -= dX

            if np.linalg.norm(res) < 1e-10:
                return X

        print("MAX ITER REACHED FOR INITIALIZATION", X[cell_idx, :])
        return X
