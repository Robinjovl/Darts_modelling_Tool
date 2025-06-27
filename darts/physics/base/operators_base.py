import numpy as np
import abc
from darts.engines import operator_set_evaluator_iface, value_vector
from darts.physics.base.property_base import PropertyBase
from darts.engines import value_vector, index_vector

class OperatorsBase(operator_set_evaluator_iface):
    n_ops: int

    def __init__(self, property_container: PropertyBase, thermal: bool):
        super().__init__()

        self.property = property_container

        self.thermal = thermal

        self.nc = property_container.nc
        self.ne = self.nc + self.thermal
        self.nph = property_container.nph

    @abc.abstractmethod
    def evaluate(self, state, values):
        pass

    def apply_extrapolation(self, state, values):
        # Find composition, if last composition is negative, apply extrapolation
        zc = np.append(state[1:self.nc], 1 - np.sum(state[1:self.nc]))

        if zc[-1] < -1e-12 / 10:
            self.extrapolate(state, values)
            return 1
        else:
            return 0

    # Least square method
    # def extrapolate(self, state, values):
    #     """
    #     When sun(z_i) > 1 (or some z_i < 0), fit a hyperplane through nearby valid points
    #     in the d = (n_comps)-dimensional z-space and use it to extrapolate all n_ops operators.
    #
    #     State layout: [ p, z₁, z₂, …, z_d, T ]
    #     """
    #     print(state)
    #     if self.thermal:
    #         vec = state.to_numpy()
    #         p, T = vec[0], vec[-1]
    #         z = vec[1:-1]  # array of length d
    #
    #     else:
    #         vec = state.to_numpy()
    #         p= vec[0]
    #         z = vec[1:]  # array of length d
    #
    #
    #     # n = vec.size
    #     # # enforce exactly p + d comps + T
    #     # d = n - 2
    #     # if d < 1:
    #     #     # nothing to do
    #     #     return 0
    #
    #     d = z.size
    #     dz = abs(1.0 - z.sum())  # “distance” outside the simplex
    #     # build candidate reference points: d of them by subtracting dz along each axis,
    #     # plus one extra “diagonal” point
    #     vec_values = np.array(values, copy=False)
    #     cand = []
    #     for i in range(d):
    #         zp = z.copy()
    #         zp[i] -= dz
    #         cand.append(zp)
    #     cand.append(z - dz / d)  # diagonal
    #     cand.append(z - dz)
    #
    #     A_rows = []
    #     B_rows = []
    #     # gather only the valid ones (0 ≤ ∑z ≤ 1 and each z_i ≥ 0)
    #     for zp in cand:
    #         if (zp >= 0).all() and zp.sum() <= 1.0:
    #             if self.thermal:
    #                 ref_state = value_vector(np.concatenate(([p], zp, [T])))
    #             else:
    #                 ref_state = value_vector(np.concatenate(([p], zp)))
    #
    #             # # make a DARTS state with this composition
    #             # ref_state = value_vector(np.concatenate(([p], zp, [T])))
    #             # print("ref",ref_state)
    #             ref_vals = value_vector(np.zeros(self.n_ops))
    #             # evaluate your normal operator function
    #             self.evaluate(ref_state, ref_vals)
    #
    #             A_rows.append(np.concatenate((zp, [1.0])))  # [z₁, …, z_d, 1]
    #             B_rows.append(ref_vals.to_numpy())  # shape (n_ops,)
    #
    #     # A = np.vstack(A_rows)  # shape (M, d+1)
    #     # B = np.vstack(B_rows)  # shape (M, n_ops)
    #     A = np.array(A_rows)  # shape (M, d+1)
    #     B = np.array(B_rows)  # shape (M, n_ops)
    #     if 0:#full least square solve
    #         # solve A · C = B in a least-squares sense → C has shape (d+1, n_ops)
    #         C, *_ = np.linalg.lstsq(A, B, rcond=None)
    #         # now extrapolate at our original z:  ext = z·C[0:d,:] + C[d,:]
    #         ext = z.dot(C[:d, :]) + C[d, :]  # shape (n_ops,)
    #     elif 1:
    #         # Solve for each operator separately (more efficient small systems)
    #         ext = np.zeros(self.n_ops, dtype=float)
    #         for j in range(self.n_ops):
    #             # find coefficients c_j of length (d+1) for operator j
    #             coeffs_j, *_ = np.linalg.lstsq(A, B[:, j], rcond=None)
    #             # extrapolate at original z
    #             ext[j] = z.dot(coeffs_j[:d]) + coeffs_j[d]
    #
    #     for i, value in enumerate(ext):
    #         vec_values[i] = np.float64(value)
    #     # copy into the DARTS values vector
    #
    #     return vec_values

    ### USING HYPERCUBE LINEAR FOURMULATION
    # def extrapolate(self, state, values):
    #     """
    #     When sum(z_i) > 1 (or some z_i < 0), fit a hyperplane through nearby valid points
    #     in the d=(n_comps)-dimensional z-space and use it to extrapolate all n_ops operators.
    #
    #     Uses null-space method per operator to solve smaller systems.
    #     State layout: [ p, z₁, z₂, …, z_d, T ]
    #     """
    #     if self.thermal:
    #         vec = state.to_numpy()
    #         p, T = vec[0], vec[-1]
    #         z = vec[1:-1]
    #     else:
    #         vec = state.to_numpy()
    #         p = vec[0]
    #         z = vec[1:]
    #
    #     d = z.size
    #     dz = abs(1.0 - z.sum())
    #     cand = []
    #     for i in range(d):
    #         zp = z.copy()
    #         zp[i] -= dz
    #         cand.append(zp)
    #     # cand.append(z - dz / d)
    #     cand.append(z - dz)
    #
    #     # Gather valid reference points and their operator values
    #     zps = []
    #     vals = []
    #     for zp in cand:
    #         if (zp >= 0).all() and zp.sum() <= 1.0:
    #             if self.thermal:
    #                 ref_state = value_vector(np.concatenate(([p], zp, [T])))
    #             else:
    #                 ref_state = value_vector(np.concatenate(([p], zp)))
    #
    #             ref_vals = value_vector(np.zeros(self.n_ops))
    #             self.evaluate(ref_state, ref_vals)
    #
    #             zps.append(zp)
    #             vals.append(ref_vals.to_numpy())
    #
    #     zps = np.array(zps)  # shape (M, d)
    #     vals = np.array(vals)  # shape (M, n_ops)
    #     n_refs = zps.shape[0]
    #
    #     # Extrapolate each operator via null-space hyperplane
    #     ext = np.zeros(self.n_ops, dtype=float)
    #     for j in range(self.n_ops):
    #         # Build P_j matrix: [z1 ... z_d, alpha, 1]
    #         Pj = np.hstack((zps, vals[:, j:j + 1], np.ones((n_refs, 1))))
    #         # Compute null vector h (last singular vector)
    #         _, _, vh = np.linalg.svd(Pj)
    #         h = vh[-1, :]
    #         # h: [h_z (d), h_alpha, h_const]
    #         h_z = h[:d]
    #         h_alpha = h[d]
    #         h_const = h[d + 1]
    #         # Solve for alpha at original z: h_z·z + h_alpha·alpha + h_const = 0
    #         ext[j] = -(h_z.dot(z) + h_const) / h_alpha
    #
    #     # Copy into values vector
    #     vec_values = np.array(values, copy=False)
    #     for i, val in enumerate(ext):
    #         vec_values[i] = np.float64(val)
    #
    #     return vec_values

    def extrapolate(self, state, values):
        """
        When composition lies outside the simplex (∑ z_i ≠ 1 or some z_i < 0), perform exact hyperplane extrapolation:
        Fit each operator value via val = a·z + c through exactly d+1 valid reference points ,
        then evaluate at the out‑of‑bounds composition. Pressure (and temperature) remain constant.
        State layout: [ p, z₁, …, z_d, (T) ]
        """
        # Unpack state
        vec = state.to_numpy()
        if self.thermal:
            p, T = vec[0], vec[-1]
            z = vec[1:-1].copy()
        else:
            p = vec[0]
            z = vec[1:].copy()

        d = z.size
        dz = abs(1.0 - z.sum())

        # Build candidate points by subtracting dz along each axis and uniformly
        candidates = []
        for i in range(d):
            zp = z.copy()
            zp[i] -= dz
            candidates.append(zp)
        candidates.append(z - dz)

        # Gather valid reference points
        zps_list = []
        vals_list = []
        for zp in candidates:
            if (zp >= 0).all() and zp.sum() <= 1.0:
                if self.thermal:
                    ref_state = value_vector(np.concatenate(([p], zp, [T])))
                else:
                    ref_state = value_vector(np.concatenate(([p], zp)))
                ref_vals = value_vector(np.zeros(self.n_ops))
                self.evaluate(ref_state, ref_vals)
                zps_list.append(zp)
                vals_list.append(ref_vals.to_numpy())

        # Use the first d+1 valid points to define hyperplane implicitly via val = a·z + c
        zps = np.stack(zps_list[:d + 1])  # shape (d+1, d)
        vals = np.stack(vals_list[:d + 1])  # shape (d+1, n_ops)

        # Build and solve B · X = vals, where B = [zps | 1]
        B = np.hstack((zps, np.ones((d + 1, 1))))  # shape (d+1, d+1)
        X = np.linalg.solve(B, vals)  # shape (d+1, n_ops)

        # Separate coefficients
        a = X[:-1, :]  # shape (d, n_ops)
        c = X[-1, :]  # shape (n_ops,)

        # Extrapolate values
        ext = a.T.dot(z) + c

        # Write back into values array
        out = np.array(values, copy=False)
        out[:] = ext
        return out

class WellControlOperators(OperatorsBase):
    """
    Set of operators for well controls. It contains the pressure, composition and temperature of the wellhead,
    plus a set of rate-control operators for different types of rates: molar-, mass-, volumetric- or advective
    heat rate controls
    """
    def __init__(self, property_container: PropertyBase, thermal: bool):
        super().__init__(property_container, thermal)

        self.n_ops = 2 + self.nph * 4

    def evaluate(self, state, values):
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state, values):
            return 0

        vec_state_as_np = state.to_numpy()
        vec_values_as_np = values.to_numpy()
        vec_values_as_np[:] = 0

        self.property.evaluate(vec_state_as_np)

        # Store rate controls
        mobility = self.property.kr[self.property.ph] / self.property.mu[self.property.ph]

        # Molar rate
        idx = 0
        vec_values_as_np[idx + self.property.ph] = self.property.dens_m[self.property.ph] * mobility

        # Mass rate
        idx += self.nph
        vec_values_as_np[idx + self.property.ph] = self.property.dens[self.property.ph] * mobility

        # Volumetric rate
        idx += self.nph
        vec_values_as_np[idx + self.property.ph] = mobility

        # Advective heat rate
        idx += self.nph
        if self.thermal:
            self.property.evaluate_thermal(vec_state_as_np)
            vec_values_as_np[idx + self.property.ph] = \
                    self.property.enthalpy[self.property.ph] * self.property.dens_m[self.property.ph] * mobility

        # Store P, T and composition of current state
        idx += self.nph
        vec_values_as_np[idx + 0] = state[0]
        vec_values_as_np[idx + 1] = self.property.temperature

        return 0


class WellInitOperators(OperatorsBase):
    def __init__(self, property_container: PropertyBase, thermal: bool, is_pt: bool = True):
        super().__init__(property_container, thermal)

        self.n_ops = 1
        self.is_pt = is_pt

    def evaluate(self, state_pt, values):
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state_pt, values):
            return 0

        vec_values_as_np = values.to_numpy()
        vec_values_as_np[:] = 0

        if self.is_pt:
            vec_values_as_np[0] = state_pt[-1]
        else:
            state_pt = np.array(list(state_pt[:self.nc]) + [state_pt[-1] if self.thermal else self.temperature])
            vec_values_as_np[0] = self.property.compute_total_enthalpy(state_pt=state_pt)

        return 0


class PropertyOperators(OperatorsBase):
    """
    This class contains a set of operators for evaluation of output properties.
    A set of interpolators is created in the :class:`Physics` object to rapidly obtain properties after simulation.
    """
    def __init__(self, property_container: PropertyBase, thermal: bool, props: dict = None):
        """
        This is the constructor for PropertyOperator.
        The properties to be obtained from the PropertyOperators are passed to PropertyContainer as a dictionary.

        :param property_container: PropertyBase object to evaluate properties at given state
        :param thermal: Bool for thermal
        :param props: Optional dictionary of properties, default is taken from PropertyContainer
        """
        super().__init__(property_container, thermal)

        self.props = property_container.output_props if props is None else props
        self.props_name = [key for key in self.props.keys()]
        self.props_idx = {prop: j for j, prop in enumerate(self.props_name)}
        self.n_ops = len(self.props_name)

    def evaluate(self, state: value_vector, values: value_vector):
        """
        This function evaluates the properties at given `state` (P,z) or (P,T,z) from the :class:`PropertyContainer` object.
        The user-specified properties are stored in the `values` object.

        :param state: Vector of state variables [pres, comp_0, ..., comp_N-1, (temp)]
        :type state: darts.engines.value_vector
        :param values: Vector for storage of operator values
        :type values: darts.engines.value_vector
        """
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state, values):
            return 0

        _ = self.property.evaluate(state)
        if self.thermal:
            _ = self.property.evaluate_thermal(state)

        for i, prop in enumerate(self.props_name):
            output = self.props[prop]()
            values[i] = output if not np.isnan(output) else 0.

        return 0
