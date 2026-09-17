import warnings
from itertools import product

import numpy as np

from darts.interpolators import operator_set_evaluator_iface, value_vector
from darts.physics.base.property_container import PropertyContainer


class OperatorsBase(operator_set_evaluator_iface):
    n_ops: int

    def __init__(
        self,
        property_container: PropertyContainer,
        thermal: bool,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        Constructor of OperatorsBase base class

        :param property_container: Property container of type PropertyContainer
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[dependent_comp_idx] < 0 in case nc >= 3)
        :param dz: Composition OBL cell size(s) used to step onto neighbouring grid nodes
                    during boundary extrapolation. Scalar (uniform spacing) or a per-axis
                    vector of length nc-1 (non-uniform cell size across composition axes).
        """
        super().__init__()

        self.property = property_container

        self.thermal = thermal

        self.nc = property_container.nc
        self.dependent_comp_idx = property_container.dependent_comp_idx
        self.ne = self.nc + self.thermal
        self.nph = property_container.nph
        self.eps_z = (
            property_container.eps_z if hasattr(property_container, 'eps_z') else 1e-13
        )

        self.extrapolation_flag = extrapolation_flag
        # dz: composition-axis OBL cell size(s) used to step onto neighbouring grid
        # nodes during boundary extrapolation. Accepts a scalar / length-1 value
        # (uniform cell size across all composition axes — the legacy case) or a
        # per-axis vector of length nc-1 (non-uniform OBL cell size across
        # composition axes). Stored as a 1-D float array; a length-1 array is
        # broadcast to every composition axis.
        self.dz = np.atleast_1d(np.asarray(dz, dtype=float)) if dz is not None else None
        assert self.nc <= 2 or not extrapolation_flag or self.dz is not None, (
            "Please provide dz for extrapolation"
        )
        if self.dz is not None:
            # Validate shape/values, not just length: a stray 2-D array, NaN/inf, or
            # non-positive step would otherwise corrupt the per-axis stepping silently.
            assert self.dz.ndim == 1, (
                f"dz must be a scalar or 1-D vector, got ndim={self.dz.ndim}"
            )
            assert np.all(np.isfinite(self.dz)), "dz entries must be finite"
            if extrapolation_flag and self.nc > 2:
                assert self.dz.size in (1, self.nc - 1), (
                    f"dz must be scalar or length nc-1={self.nc - 1}, got {self.dz.size}"
                )
                assert np.all(self.dz > 0), "dz entries must be strictly positive"

    def evaluate_batch(self, states, n_points, values, n_ops):
        """
        Default serial batch evaluation: loops calling evaluate() per point.
        Override in a subclass or wrapper (e.g. ParallelEvaluator) for parallel dispatch.

        :param states: Flat array of coordinates [n_points * n_dims]
        :param n_points: Number of points to evaluate
        :param values: Flat output array [n_points * n_ops], pre-allocated
        :param n_ops: Number of operators per point
        :return: 0 if successful
        """
        states_np = np.asarray(states)
        values_np = np.asarray(values)
        n_dims = len(states_np) // n_points
        for i in range(n_points):
            sv = value_vector(states_np[i * n_dims : (i + 1) * n_dims].copy())
            vv = value_vector(np.zeros(n_ops))
            self.evaluate(sv, vv)
            values_np[i * n_ops : (i + 1) * n_ops] = np.asarray(vv)
        return 0

    def apply_extrapolation(self, state, values):
        """
        Method that determines whether or not extrapolation should be applied to current state
        (z[dependent_comp_idx] < 0). If so, it will call extrapolate() and return True, such that
        evaluate() skips further evaluation of operators

        :param state: Vector with state [P, z, (T/H)]
        :param values: Vector with operator values
        :return: Whether or not extrapolation has been applied to this state
        """
        # Find composition, if the implicit (closure) composition is negative, apply extrapolation
        zc = np.insert(
            state[1 : self.nc],
            self.dependent_comp_idx,
            1 - np.sum(state[1 : self.nc]),
        )

        if (
            len(zc) > 2
            and zc[self.dependent_comp_idx] < 0.99 * self.eps_z
            and self.extrapolation_flag
        ):
            # TODO: Fix second condition, this is problematic for small eps_z values (~1e-14)
            self.extrapolate(state, values)
            return True
        else:
            return False

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

        zero_comps = [i for i in range(self.nc - 1) if z[i] <= 2 * self.eps_z]
        nonzero_comps = [1 if z[i] > 2 * self.eps_z else 0 for i in range(self.nc - 1)]
        nonzero_comp_idxs = [
            i for i, is_nonzero in enumerate(nonzero_comps) if is_nonzero
        ]
        dims = np.sum(nonzero_comps)

        # Per-axis composition cell size (length nc-1). A scalar / length-1 dz is
        # broadcast to every composition axis, so uniform OBL grids reproduce the
        # original behaviour exactly; a length nc-1 dz gives each composition axis
        # its own step, i.e. non-uniform OBL cell size across composition axes.
        dz_axis = (
            self.dz
            if self.dz.size == (self.nc - 1)
            else np.full(self.nc - 1, self.dz[0])
        )

        # --- Candidate supporting points --------------------------------------
        # Step −offset_i · dz_axis[i] along each active composition axis using
        # INDEPENDENT per-axis offsets (0..n_steps_max), excluding the all-zero
        # offset (the incoming point). Independent offsets are required on a
        # non-uniform grid: a valid *physical* support can need a mixed move such
        # as (1·dz0, 2·dz1) that a single shared multiplier never produces. Each
        # candidate lands on a real OBL grid node.
        #
        # n_steps_max gives reach 2 per axis, enough to recover `dims + 1` physical
        # supports for boundary nodes (which sit within one cell of the surface).
        n_steps_max = max(2, dims + 1 - (2**dims - 1))
        candidates = []  # (dist2_index, offsets, zp, admissible)
        for offsets in product(range(n_steps_max + 1), repeat=dims):
            if not any(offsets):
                continue
            zp = z.copy()
            for i, axis in enumerate(nonzero_comp_idxs):
                if offsets[i]:
                    zp[axis] -= offsets[i] * dz_axis[axis]
            # Distance in grid-index units (the offset magnitude), so the selection
            # geometry is independent of the per-axis cell sizes and not biased
            # toward coarser axes.
            dist2 = float(sum(o * o for o in offsets))
            last_z = 1.0 - np.sum(zp)
            # Admissible = the support is physical: it must lie inside the simplex
            # on BOTH the normalization constraint (last_z = 1 − Σz ≥ 0) AND the
            # axis-aligned constraints (every explicit composition ≥ 0). Checking
            # last_z alone would accept a support with a negative component and
            # flash it as physical (the paper requires *physical* supporting
            # points). The small tolerance absorbs round-off on exactly-on-axis
            # grid nodes. This per-node half-space test needs no explicit
            # hypercube/normalization-surface intersection.
            admissible = (last_z >= 0.0) and bool(np.all(zp >= -1e-12))
            candidates.append((dist2, offsets, zp, admissible))

        # --- Rank-revealing selection of d+1 physical, independent supports ----
        # Walk the admissible candidates nearest-first and greedily keep a point
        # only if it adds a NEW affine direction (raises the rank). This guarantees
        # the selected d+1 supports are affinely independent (so B is non-singular)
        # AND physical — closing the gap where a distance-only "furthest+closest"
        # rule can pick collinear or non-physical points. Independence is tracked by
        # modified Gram–Schmidt on the offset vectors; because zp − z = −offset·dz
        # (a fixed positive per-axis scaling), offset-space rank equals zp-space
        # rank, so this is exact and scale-free.
        admissible_sorted = sorted((c for c in candidates if c[3]), key=lambda c: c[0])
        n_supporting_points = dims + 1
        selected = []
        anchor = None
        basis = []  # orthonormal directions already spanned (offset/index units)
        for c in admissible_sorted:
            ov = np.asarray(c[1], dtype=float)
            if anchor is None:
                anchor = ov
                selected.append(c)
                continue
            v = ov - anchor
            for b in basis:
                v = v - np.dot(v, b) * b
            nv = np.linalg.norm(v)
            if nv > 1e-9:
                basis.append(v / nv)
                selected.append(c)
            if len(selected) == n_supporting_points:
                break

        # Fallback: the admissible set genuinely spans fewer than `dims` directions
        # (the physical neighbourhood of this node is lower-dimensional than dims).
        # Top up nearest-first from the remaining candidates so the fit can still
        # proceed; evaluating a non-physical top-up node recurses into another
        # extrapolation rather than flashing it.
        if len(selected) < n_supporting_points:
            chosen = {c[1] for c in selected}
            for c in sorted(
                (c for c in candidates if c[1] not in chosen), key=lambda c: c[0]
            ):
                selected.append(c)
                if len(selected) == n_supporting_points:
                    break

        supporting_points = [c[2] for c in selected]

        # Gather valid reference points
        zps_list = []
        vals_list = []
        for zp in supporting_points:
            if self.thermal:
                ref_state = value_vector(np.concatenate(([p], zp, [T])))
            else:
                ref_state = value_vector(np.concatenate(([p], zp)))
            ref_vals = value_vector(np.zeros(self.n_ops))
            self.evaluate(ref_state, ref_vals)
            zps_list.append(zp)
            vals_list.append(ref_vals.to_numpy())

        # Use the first d+1 valid points to define hyperplane implicitly via val = a·z + c
        zps = np.stack(zps_list[: dims + 1])  # shape (dims+1, dims)
        vals = np.stack(vals_list[: dims + 1])  # shape (dims+1, n_ops)

        # Build and solve B · X = vals, where B = [zps | 1]
        B = np.hstack((zps, np.ones((dims + 1, 1))))  # shape (dims+1, dims+1)
        B = np.delete(B, zero_comps, axis=1)
        try:
            X = np.linalg.solve(B, vals)  # shape (dims+1, n_ops)
        except np.linalg.LinAlgError:
            # A singular system means support SELECTION produced affinely-dependent
            # points — a selection problem, not an expected numerical condition.
            # Don't abort the run, but surface it loudly (per-process warning, which
            # works in serial and in multiprocessing workers alike) rather than
            # silently least-squares'ing it away, so it can be investigated.
            warnings.warn(
                f"OBL extrapolation: singular supporting set (dims={int(dims)}) — "
                "falling back to least-squares. This indicates degenerate support "
                "selection and should be investigated.",
                RuntimeWarning,
                stacklevel=2,
            )
            X = np.linalg.lstsq(B, vals, rcond=None)[0]

        # Separate coefficients
        a = X[:-1, :]  # shape (dims, n_ops)
        c = X[-1, :]  # shape (n_ops,)

        # Extrapolate values
        z_nonzero = z[nonzero_comp_idxs]
        ext = a.T.dot(z_nonzero) + c

        # Write back into values array
        out = np.array(values, copy=False)
        out[: self.n_ops] = ext
        return out


class WellCtrlOperators(OperatorsBase):
    """
    Set of operators for well controls of EPM and DFM wells.

    Operator layout:
    NP EPM molar-rate, NP EPM mass-rate, NP EPM volumetric-rate,
    NP EPM advective-heat-rate, pressure, temperature,
    NP DFM molar-rate, NP DFM mass-rate, NP DFM volumetric-rate,
    NP DFM advective-heat-rate.
    """

    def __init__(
        self,
        property_container: PropertyContainer,
        thermal: bool,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        Constructor of WellCtrlOperators class

        :param property_container: Property container of type PropertyContainer
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[dependent_comp_idx] < 0 in case nc >= 3)
        :param dz: Composition OBL cell size(s) used to step onto neighbouring grid nodes
                    during boundary extrapolation. Scalar (uniform spacing) or a per-axis
                    vector of length nc-1 (non-uniform cell size across composition axes).
        """
        super().__init__(
            property_container, thermal, extrapolation_flag=extrapolation_flag, dz=dz
        )

        self.n_rate_ctrl_types = 4  # molar, mass, volumetric, and advective heat rates
        self.n_state_ctrl_ops = 2  # pressure and temperature
        self.epm_rate_ctrl_ops_offset = 0
        self.state_ctrl_ops_offset = self.n_rate_ctrl_types * self.nph
        self.dfm_rate_ctrl_ops_offset = (
            self.state_ctrl_ops_offset + self.n_state_ctrl_ops
        )
        self.n_ops = self.n_state_ctrl_ops + 2 * self.n_rate_ctrl_types * self.nph

    def _fill_rate_ctrl_ops(self, values, offset, rate_factor, eq_phase_idxs_mobile):
        # Molar rate ctrl operator
        idx = offset
        values[idx + eq_phase_idxs_mobile] = (
            self.property.dens_m[eq_phase_idxs_mobile] * rate_factor
        )

        # Mass rate ctrl operator
        idx += self.nph
        values[idx + eq_phase_idxs_mobile] = (
            self.property.dens[eq_phase_idxs_mobile] * rate_factor
        )

        # Volumetric rate ctrl operator
        idx += self.nph
        values[idx + eq_phase_idxs_mobile] = rate_factor

        # Advective heat rate ctrl operator
        idx += self.nph
        if self.thermal:
            values[idx + eq_phase_idxs_mobile] = (
                self.property.enthalpy[eq_phase_idxs_mobile]
                * self.property.dens_m[eq_phase_idxs_mobile]
                * rate_factor
            )

    def evaluate(self, state, values):
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state, values):
            return 0

        state_np = state.to_numpy()
        values_np = values.to_numpy()
        values_np[:] = 0

        self.property.evaluate(state_np)
        if self.thermal:
            self.property.evaluate_thermal(state_np)

        eq_phase_idxs_mobile = self.property.eq_phase_idxs_mobile()
        epm_rate_factor = (
            self.property.kr[eq_phase_idxs_mobile]
            / self.property.mu[eq_phase_idxs_mobile]
        )
        self._fill_rate_ctrl_ops(
            values_np,
            self.epm_rate_ctrl_ops_offset,
            epm_rate_factor,
            eq_phase_idxs_mobile,
        )

        # Store pressure (P) and temperature (T) of the current state for a generic state specification.
        # This is needed when pressure or temperature is not part of the state variables
        # (e.g., volume instead of pressure, or enthalpy instead of temperature).
        idx = self.state_ctrl_ops_offset
        values_np[idx + 0] = state[0]
        values_np[idx + 1] = self.property.temperature

        dfm_rate_factor = self.property.sat[eq_phase_idxs_mobile]
        self._fill_rate_ctrl_ops(
            values_np,
            self.dfm_rate_ctrl_ops_offset,
            dfm_rate_factor,
            eq_phase_idxs_mobile,
        )

        return 0


class ThermalVarOperator(OperatorsBase):
    """
    ThermalVarOperator gives the thermal variable for generic state specification
    """

    def __init__(
        self,
        property_container: PropertyContainer,
        thermal: bool,
        is_pt: bool = True,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        Constructor of ThermalVarOperator class

        :param property_container: Property container of type PropertyContainer
        :param thermal: Switch to indicate if energy conservation equation is there
        :param is_pt: Switch to indicate if state specification is P, PT, or PH
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[dependent_comp_idx] < 0 in case nc >= 3)
        :param dz: Composition OBL cell size(s) used to step onto neighbouring grid nodes
                    during boundary extrapolation. Scalar (uniform spacing) or a per-axis
                    vector of length nc-1 (non-uniform cell size across composition axes).
        """
        super().__init__(property_container, thermal, extrapolation_flag, dz)

        self.n_ops = 1
        self.is_pt = is_pt

    def evaluate(self, state_pt, values):
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state_pt, values):
            return 0

        values_np = values.to_numpy()
        values_np[:] = 0

        if not self.thermal:
            values_np[0] = self.property.temperature
        elif self.is_pt:
            values_np[0] = state_pt[-1]
        else:
            values_np[0] = self.property.compute_total_enthalpy(state_pt=state_pt)

        return 0


class PropertyOperators(OperatorsBase):
    """
    This class contains a set of operators for evaluation of output properties.
    A set of interpolators is created in the :class:`Physics` object to rapidly obtain properties after simulation.
    """

    def __init__(
        self,
        property_container: PropertyContainer,
        thermal: bool,
        props: dict = None,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        This is the constructor for PropertyOperator.
        The properties to be obtained from the PropertyOperators are passed to PropertyContainer as a dictionary.

        :param property_container: PropertyContainer object to evaluate properties at given state
        :param thermal: Bool for thermal
        :param props: Optional dictionary of properties, default is taken from PropertyContainer
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[dependent_comp_idx] < 0 in case nc >= 3)
        :param dz: Composition OBL cell size(s) used to step onto neighbouring grid nodes
                    during boundary extrapolation. Scalar (uniform spacing) or a per-axis
                    vector of length nc-1 (non-uniform cell size across composition axes).
        """
        super().__init__(property_container, thermal, extrapolation_flag, dz)

        self.props = property_container.output_props if props is None else props
        self.props_name = [key for key in self.props.keys()]
        self.props_idx = {prop: j for j, prop in enumerate(self.props_name)}
        self.n_ops = len(self.props_name)

    def evaluate(self, state: value_vector, values: value_vector):
        """
        This function evaluates the properties at given `state` (P,z) or (P,T,z) from the :class:`PropertyContainer` object.
        The user-specified properties are stored in the `values` object.

        :param state: Vector of state variables [pres, comp_0, ..., comp_N-1, (temp)]
        :type state: darts.interpolators.value_vector
        :param values: Vector for storage of operator values
        :type values: darts.interpolators.value_vector
        """
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state, values):
            return 0

        state_np = state.to_numpy()
        values_np = values.to_numpy()
        _ = self.property.evaluate(state_np)
        if self.thermal:
            _ = self.property.evaluate_thermal(state_np)

        for i, prop in enumerate(self.props_name):
            output = self.props[prop]()
            values_np[i] = output if not np.isnan(output) else 0.0

        return 0


class OperatorsSuper(OperatorsBase):
    property: PropertyContainer

    def __init__(
        self,
        property_container: PropertyContainer,
        thermal: bool,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        Constructor of OperatorsSuper base class

        :param property_container: Property container of type PropertyContainer
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[dependent_comp_idx] < 0 in case nc >= 3)
        :param dz: Composition interval along OBL composition axes to obtain consistent points for extrapolation
                    (must be equal along all composition axes in current setup)
        """
        super().__init__(
            property_container, thermal, extrapolation_flag=extrapolation_flag, dz=dz
        )  # Initialize base-class

        self.nc_eq = property_container.nc_eq
        self.nc_kin = property_container.nc_kin
        self.np_eq = property_container.np_eq
        self.fluid_phase_idxs = property_container.fluid_phase_idxs
        self.solid_phase_idxs = property_container.solid_phase_idxs
        self.kin_phase_idxs = property_container.kin_phase_idxs
        self.mole_kin_phase_idxs = property_container.mole_kin_phase_idxs
        self.mole_kin_comp_idxs = property_container.mole_kin_comp_idxs
        self.bulk_kin_phase_idxs = property_container.bulk_kin_phase_idxs
        self.bulk_kin_comp_idxs = property_container.bulk_kin_comp_idxs
        self.solid_comp_idxs = property_container.solid_comp_idxs
        self.fluid_comp_idxs = property_container.fluid_comp_idxs

        # Operator order
        self.ACC_OP = 0  # accumulation operator - ne
        self.FLUX_OP = self.ACC_OP + self.ne  # flux operator - ne * nph
        self.DENS_OP = self.FLUX_OP + self.ne * self.nph  # density operator
        self.UPSAT_OP = self.DENS_OP + self.nph  # saturation operator
        self.GRAD_OP = self.UPSAT_OP + self.nph  # gradient operator - ne * nph
        self.KIN_OP = self.GRAD_OP + self.ne * self.nph  # kinetic operator - ne
        self.GRAV_OP = self.KIN_OP + self.ne  # gravity operator - nph
        self.PC_OP = self.GRAV_OP + self.nph  # capillary operator - nph
        self.MULT_OP = self.PC_OP + self.nph  # permeability multiplier operator - 1
        self.LAMBDA_OP = self.MULT_OP + 1  # mobility operator - nph
        self.SAT_OP = self.LAMBDA_OP + self.nph  # saturation operator - nph
        self.ENTH_OP = self.SAT_OP + self.nph  # enthalpy operator - nph
        self.TEMP_OP = self.ENTH_OP + self.nph  # temperature operator - 1
        self.PRES_OP = self.TEMP_OP + 1  # pressure operator - 1
        self.n_ops = self.PRES_OP + 1

        # Operator names
        self.op_names = [
            (self.ACC_OP, "ACC"),
            (self.FLUX_OP, "FLUX"),
            (self.DENS_OP, "DENS"),
            (self.UPSAT_OP, "UPSAT"),
            (self.GRAD_OP, "GRAD"),
            (self.KIN_OP, "KIN"),
            (self.GRAV_OP, "GRAV"),
            (self.PC_OP, "PC"),
            (self.MULT_OP, "MULT"),
            (self.LAMBDA_OP, "LAMBDA"),
            (self.SAT_OP, "SAT"),
            (self.ENTH_OP, "ENTH"),
            (self.TEMP_OP, "TEMP"),
            (self.PRES_OP, "PRES"),
        ]

    def print_operators(self, state, values):
        """Method for printing operators, grouped"""
        print("================================================")
        print("STATE", state)
        print("ALPHA (accumulation)", values[self.ACC_OP : self.FLUX_OP])
        for j in range(self.nph):
            idx0, idx1 = self.FLUX_OP + j * self.ne, self.FLUX_OP + (j + 1) * self.ne
            print(f"BETA (flux) {j}", values[idx0:idx1])
        print("GAMMA (diffusion)", values[self.UPSAT_OP : self.GRAD_OP])
        for j in range(self.nph):
            idx0, idx1 = self.GRAD_OP + j * self.ne, self.GRAD_OP + (j + 1) * self.ne
            print(f"CHI (diffusion) {j}", values[idx0:idx1])
        print("DELTA (reaction)", values[self.KIN_OP : self.GRAV_OP])
        print("GRAVITY", values[self.GRAV_OP : self.PC_OP])
        print("CAPILLARITY", values[self.PC_OP : self.MULT_OP])
        print("PERM_MULT", values[self.MULT_OP])
        print("LAMBDA", values[self.LAMBDA_OP : self.SAT_OP])
        print("SAT", values[self.SAT_OP : self.ENTH_OP])
        print("ENTHALPY", values[self.ENTH_OP : self.ENTH_OP + self.nph])
        print("TEMPERATURE, PRESSURE", values[self.TEMP_OP], values[self.PRES_OP])
        return


class ReservoirOperators(OperatorsSuper):
    def evaluate(self, state, values):
        """
        Evaluate the non-thermal reservoir operators for the super engine

        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]
        :type state: value_vector in open-darts, pylvarray.Array in GEOS
        :param values: values of the operators (used for storing the operator values)
        :type values: value_vector in open-darts, pylvarray.Array in GEOS
        :return: updated value for operators, stored in values
        """
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state, values):
            return 0

        # Composition vector and pressure from state:
        state_np = state.to_numpy()
        values_np = values.to_numpy()
        values_np[:] = 0

        # Evaluate properties at current state
        self.property.evaluate(state_np)
        self.compr = self.property.rock_compr_ev.evaluate(state_np[0])

        # Computed fresh (not cached on self.property) each call -- see
        # PropertyContainer.mole_basis_phase_idxs()/eq_phase_idxs_mobile().
        mole_basis_phase_idxs = self.property.mole_basis_phase_idxs()
        eq_phase_idxs_mobile = self.property.eq_phase_idxs_mobile()

        # Average molar density of the equilibrium + KineticFormulation.MOLE_FRACTION phase pool
        # (mole_basis_phase_idxs, see compute_saturation()) -- zc for
        # equilibrium AND KineticFormulation.MOLE_FRACTION components are both "modified
        # variables" on that same shared basis (see Flash's is_mole_fraction), so
        # both use this density_tot directly below.
        density_tot = np.sum(
            self.property.sat[mole_basis_phase_idxs]
            * self.property.dens_m[mole_basis_phase_idxs]
        )
        zc = np.insert(
            state_np[1 : self.nc],
            self.dependent_comp_idx,
            1 - np.sum(state_np[1 : self.nc]),
        )

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        # zc is the raw z_c* (get_state() docstring), not Flash-normalized: z_c* already
        # bakes in phi_f, so density_tot (per fluid volume) * z_c* gives kmol per bulk volume.
        # fluid mass accumulation: c_r [1/bar] z_c* [-] rho_m^T [kmol/m3]
        values_np[self.ACC_OP : self.ACC_OP + self.nc_eq] = (
            self.compr * density_tot * zc[: self.nc_eq]
        )
        # KineticFormulation.MOLE_FRACTION components: same formula, same density_tot -- their
        # raw zc is on the same basis as the equilibrium components' own.
        values_np[self.ACC_OP + self.mole_kin_comp_idxs] = (
            self.compr * density_tot * zc[self.mole_kin_comp_idxs]
        )

        """ and alpha for bulk (volume-based) kinetic components """
        # solid mass accumulation: c_r [1/bar] rho_ms [kmol/m3] -- always 1 component
        # per phase (see PropertyContainer), so this maps directly, no splitting needed.
        values_np[self.ACC_OP + self.bulk_kin_comp_idxs] = (
            self.compr
            * self.property.dens_m[self.bulk_kin_phase_idxs]
            * self.property.sat[self.bulk_kin_phase_idxs]
        )

        """ Beta operator """
        for j in eq_phase_idxs_mobile:
            # fluid convective mass flux: x_cj [-] rho_mj [kmol/m3] (kmol/m3)
            values_np[
                self.FLUX_OP + j * self.ne : self.FLUX_OP + j * self.ne + self.nc_eq
            ] = self.property.x[j][: self.nc_eq] * self.property.dens_m[j]

        """ Molar density operator """
        # molar density: rho_mj [kmol/m3]
        values_np[self.DENS_OP + self.property.eq_phase_idxs] = self.property.dens_m[
            self.property.eq_phase_idxs
        ]

        # Diffusion isn't mobility-gated for KineticFormulation.MOLE_FRACTION phases (unlike
        # convection/FLUX_OP): pool them in alongside eq_phase_idxs_mobile even
        # when immobile, since they share the fluid's diffusion machinery.
        diffusive_phase_idxs = np.union1d(
            eq_phase_idxs_mobile, self.mole_kin_phase_idxs
        )

        """ Gamma operator for diffusion (for heat conduction and molecular diffusion) """
        # fluid diffusive flux sat: c_r [1/bar] phi_f s_j (1/bar)
        values_np[self.UPSAT_OP + diffusive_phase_idxs] = (
            self.compr * self.property.phi_f * self.property.sat[diffusive_phase_idxs]
        )

        """ Chi operator for diffusion """
        for j in diffusive_phase_idxs:
            D = self.property.diffusion_ev[self.property.phases_name[j]].evaluate()
            # fluid diffusive flux: D_cj [m2/day] x_cj [-] (m2/day)
            values_np[
                self.GRAD_OP + j * self.ne : self.GRAD_OP + j * self.ne + self.nc_eq
            ] = D[: self.nc_eq] * self.property.x[j][: self.nc_eq]

        """ Delta operator for reaction """
        # fluid/solid mass source: n_c [kmol/m3/day] (kmol/m3/day)
        values_np[self.KIN_OP : self.KIN_OP + self.nc] = self.property.mass_source

        """ Gravity and capillarity operators """
        # E3-> gravity
        values_np[self.GRAV_OP + self.property.eq_phase_idxs] = self.property.dens[
            self.property.eq_phase_idxs
        ]

        # E4-> capillarity
        values_np[self.PC_OP + eq_phase_idxs_mobile] = self.property.pc[
            eq_phase_idxs_mobile
        ]

        """ Permeability multiplier k/kmax """
        # E5_> permeability multiplier due to permporo relationship
        values_np[self.MULT_OP] = self.property.permporo_mult

        """ Lambda operator (phase mobility) """
        # phase mobility: k_rj [-] / mu_j [cP ∝ bar.day] (1/(bar.day))
        values_np[self.LAMBDA_OP + eq_phase_idxs_mobile] = (
            self.property.kr[eq_phase_idxs_mobile]
            / self.property.mu[eq_phase_idxs_mobile]
        )

        """ Saturation operator for phase volumetric calculations in the wellbore """
        # phase saturation: s_j [-]
        values_np[self.SAT_OP + self.property.eq_phase_idxs] = self.property.sat[
            self.property.eq_phase_idxs
        ]

        """ Pressure operator """
        # Pressure operator (for generic state specification where no pressure in the state, for instance V,T)
        values_np[self.PRES_OP] = state_np[0]

        if self.thermal:
            self.evaluate_thermal(
                state_np, values_np, mole_basis_phase_idxs, eq_phase_idxs_mobile
            )

        # self.print_operators(state, values)

        return 0

    def evaluate_thermal(
        self, state, values, mole_basis_phase_idxs, eq_phase_idxs_mobile
    ):
        """
        Evaluate the thermal reservoir operators for the super engine

        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        pressure = state[0]

        # Evaluate thermal properties at current state
        self.property.evaluate_thermal(state)

        """ Alpha operator represents accumulation term """
        # fluid enthalpy: phi_f[-] s_j [-] rho_mj [kmol/m3] H_j [kJ/kmol] (kJ/m3) --
        # eq_phase_idxs + KineticFormulation.MOLE_FRACTION phases share this pooled basis.
        values[self.ACC_OP + self.nc] += (
            self.compr
            * self.property.phi_f
            * np.sum(
                self.property.sat[mole_basis_phase_idxs]
                * self.property.dens_m[mole_basis_phase_idxs]
                * self.property.enthalpy[mole_basis_phase_idxs]
            )
        )  # fluid enthalpy (kJ/m3)
        # solid enthalpy: phi_s[-] s_j [-] rho_mj [kmol/m3] H_j [kJ/kmol] (kJ/m3)
        values[self.ACC_OP + self.nc] += (
            self.compr
            * self.property.phi_s
            * np.sum(
                self.property.sat[self.bulk_kin_phase_idxs]
                * self.property.dens_m[self.bulk_kin_phase_idxs]
                * self.property.enthalpy[self.bulk_kin_phase_idxs]
            )
        )
        # Enthalpy to internal energy conversion
        values[self.ACC_OP + self.nc] -= self.compr * 100 * pressure

        """ Beta operator """
        # fluid convective energy flux: H_j [kJ/kmol] rho_mj [kmol/m3] (kJ/m3)
        values[self.FLUX_OP + eq_phase_idxs_mobile * self.ne + self.nc] = (
            self.property.enthalpy[eq_phase_idxs_mobile]
            * self.property.dens_m[eq_phase_idxs_mobile]
        )

        """ Chi operator for temperature in conduction """
        # fluid/solid conductive flux: kappa_j [kJ/m.K.day] T [K] (kJ/m.day) --
        # eq_phase_idxs + KineticFormulation.MOLE_FRACTION phases share this pooled basis.
        values[self.GRAD_OP + mole_basis_phase_idxs * self.ne + self.nc] = (
            self.property.temperature * self.property.cond[mole_basis_phase_idxs]
        )

        """ Delta operator for reaction """
        # energy source: Q [kJ/m3/day] (kJ/m3/day)
        values[self.KIN_OP + self.nc] = self.property.energy_source

        """ Phase enthalpy operator """
        for j in range(self.nph):
            values[self.ENTH_OP + j] = self.property.enthalpy[j]

        """ Additional energy operators """
        # Temperature operator
        values[self.TEMP_OP] = self.property.temperature

        return 0


class WellOperators(OperatorsSuper):
    def evaluate(self, state, values):
        """
        Evaluate the non-thermal well operators for the super engine

        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]
        :type state: value_vector in open-darts, pylvarray.Array in GEOS
        :param values: values of the operators (used for storing the operator values)
        :type state: value_vector in open-darts, pylvarray.Array in GEOS
        :return: updated value for operators, stored in values
        """
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state, values):
            return 0

        # Composition vector and pressure from state:
        state_np = state.to_numpy()
        values_np = values.to_numpy()
        values_np[:] = 0

        # Evaluate properties at current state
        self.property.evaluate(state_np)

        # Computed fresh (not cached on self.property) each call -- see
        # PropertyContainer.mole_basis_phase_idxs()/eq_phase_idxs_mobile().
        mole_basis_phase_idxs = self.property.mole_basis_phase_idxs()
        eq_phase_idxs_mobile = self.property.eq_phase_idxs_mobile()

        # Average molar density of the equilibrium + KineticFormulation.MOLE_FRACTION phase pool
        # (mole_basis_phase_idxs, see compute_saturation()) -- zc for
        # equilibrium AND KineticFormulation.MOLE_FRACTION components are both "modified
        # variables" on that same shared basis (see Flash's is_mole_fraction), so
        # both use this density_tot directly below.
        density_tot = np.sum(
            self.property.sat[mole_basis_phase_idxs]
            * self.property.dens_m[mole_basis_phase_idxs]
        )
        zc = np.insert(
            state_np[1 : self.nc],
            self.dependent_comp_idx,
            1 - np.sum(state_np[1 : self.nc]),
        )

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        # zc is the raw z_c* (get_state() docstring), not Flash-normalized: z_c* already
        # bakes in phi_f, so density_tot (per fluid volume) * z_c* gives kmol per bulk volume.
        # fluid mass accumulation: z_c* [-] rho_m^T [kmol/m3]
        values_np[self.ACC_OP : self.ACC_OP + self.nc_eq] = (
            density_tot * zc[: self.nc_eq]
        )
        # KineticFormulation.MOLE_FRACTION components: same formula, same density_tot.
        values_np[self.ACC_OP + self.mole_kin_comp_idxs] = (
            density_tot * zc[self.mole_kin_comp_idxs]
        )

        """ and alpha for bulk (volume-based) kinetic components """
        # solid mass accumulation: rho_ms [kmol/m3] -- always 1 component per phase.
        values_np[self.ACC_OP + self.bulk_kin_comp_idxs] = (
            self.property.dens_m[self.bulk_kin_phase_idxs]
            * self.property.sat[self.bulk_kin_phase_idxs]
        )

        """ Beta operator """
        for j in eq_phase_idxs_mobile:
            # fluid convective mass flux: x_cj [-] rho_mj [kmol/m3] (kmol/m3)
            values_np[
                self.FLUX_OP + j * self.ne : self.FLUX_OP + j * self.ne + self.nc_eq
            ] = self.property.x[j][: self.nc_eq] * self.property.dens_m[j]

        """ Molar density operator """

        """ Gamma operator for diffusion (for heat conduction and molecular diffusion) """

        """ Chi operator for diffusion """

        """ Delta operator for reaction """
        # fluid/solid mass source: n_c [kmol/m3/day] (kmol/m3/day)
        values_np[self.KIN_OP : self.KIN_OP + self.nc] = self.property.mass_source

        """ Gravity and capillarity operators """
        # E3-> gravity
        values_np[self.GRAV_OP + self.property.eq_phase_idxs] = self.property.dens[
            self.property.eq_phase_idxs
        ]

        # E4-> capillarity

        """ Permeability multiplier k/kmax """
        # E5_> permeability multiplier due to permporo relationship
        values_np[self.MULT_OP] = 1.0

        """ Lambda operator (phase mobility) """
        # phase mobility: k_rj [-] / mu_j [cP ∝ bar.day] (1/(bar.day))
        values_np[self.LAMBDA_OP + eq_phase_idxs_mobile] = (
            self.property.kr[eq_phase_idxs_mobile]
            / self.property.mu[eq_phase_idxs_mobile]
        )

        """ Saturation operator for phase volumetric calculations in the wellbore """
        # phase saturation: s_j [-]
        values_np[self.SAT_OP + self.property.eq_phase_idxs] = self.property.sat[
            self.property.eq_phase_idxs
        ]

        """ Pressure operator """
        # Pressure operator (for generic state specification where no pressure in the state, for instance V,T)
        values_np[self.PRES_OP] = state_np[0]

        if self.thermal:
            self.evaluate_thermal(
                state_np, values_np, mole_basis_phase_idxs, eq_phase_idxs_mobile
            )

        # self.print_operators(state, values)

        return 0

    def evaluate_thermal(
        self, state, values, mole_basis_phase_idxs, eq_phase_idxs_mobile
    ):
        """
        Evaluate the thermal well operators for the super engine

        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        pressure = state[0]

        # Evaluate thermal properties at current state
        self.property.evaluate_thermal(state)

        """ Alpha operator represents accumulation term """
        # fluid enthalpy: s_j [-] rho_mj [kmol/m3] H_j [kJ/kmol] (kJ/m3) --
        # eq_phase_idxs + KineticFormulation.MOLE_FRACTION phases share this pooled basis.
        values[self.ACC_OP + self.nc] += self.property.phi_f * np.sum(
            self.property.sat[mole_basis_phase_idxs]
            * self.property.dens_m[mole_basis_phase_idxs]
            * self.property.enthalpy[mole_basis_phase_idxs]
        )  # fluid enthalpy (kJ/m3)
        # solid enthalpy: s_j [-] rho_mj [kmol/m3] H_j [kJ/kmol] (kJ/m3)
        values[self.ACC_OP + self.nc] += self.property.phi_s * np.sum(
            self.property.sat[self.bulk_kin_phase_idxs]
            * self.property.dens_m[self.bulk_kin_phase_idxs]
            * self.property.enthalpy[self.bulk_kin_phase_idxs]
        )

        # Enthalpy to internal energy conversion
        values[self.ACC_OP + self.nc] -= 100 * pressure

        """ Beta operator """
        # fluid convective energy flux: H_j [kJ/kmol] rho_mj [kmol/m3] (kJ/m3)
        values[self.FLUX_OP + eq_phase_idxs_mobile * self.ne + self.nc] = (
            self.property.enthalpy[eq_phase_idxs_mobile]
            * self.property.dens_m[eq_phase_idxs_mobile]
        )

        """ Chi operator for temperature in conduction """
        # fluid/solid conductive flux: kappa_j [kJ/m.K.day] T [K] (kJ/m.day) --
        # eq_phase_idxs + KineticFormulation.MOLE_FRACTION phases share this pooled basis.
        values[self.GRAD_OP + mole_basis_phase_idxs * self.ne + self.nc] = (
            self.property.temperature * self.property.cond[mole_basis_phase_idxs]
        )

        """ Delta operator for reaction """
        # energy source: V [m3] dt [day] c_r phi^T Q [kJ/m3.days] (kJ/m3)
        values[self.KIN_OP + self.nc] = self.property.energy_source

        """ Phase enthalpy operator """
        for j in range(self.nph):
            values[self.ENTH_OP + j] = self.property.enthalpy[j]

        """ Additional energy operators """
        # Temperature operator
        values[self.TEMP_OP] = self.property.temperature

        return 0


class GeomechanicsReservoirOperators(ReservoirOperators):
    def __init__(
        self,
        property_container: PropertyContainer,
        thermal: bool,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        Constructor of GeomechanicsReservoirOperators class

        :param property_container: Property container of type PropertyContainer
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[dependent_comp_idx] < 0 in case nc >= 3)
        :param dz: Composition interval along OBL composition axes to obtain consistent points for extrapolation
                    (must be equal along all composition axes in current setup)
        """
        super().__init__(
            property_container, thermal, extrapolation_flag, dz
        )  # Initialize base-class

        self.ROCK_DENS_OP = self.PRES_OP + 1  # used only in mechanical engine
        self.n_ops = self.ROCK_DENS_OP + 1

    def evaluate(self, state, values):
        """
        Evaluate the geomechanical reservoir operators for the super engine

        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]
        :type state: value_vector in open-darts, pylvarray.Array in GEOS
        :param values: values of the operators (used for storing the operator values)
        :type values: value_vector in open-darts, pylvarray.Array in GEOS
        :return: updated value for operators, stored in values
        """
        # Reservoir operators
        super().evaluate(state, values)

        # Rock density operator
        self.n_ops = self.ROCK_DENS_OP + 1
        # TODO: function of matrix pressure = I1 / 3 = (s_xx + s_yy + s_zz) / 3
        values.to_numpy()[self.ROCK_DENS_OP] = self.property.rock_density_ev.evaluate()

        return 0

    def print_operators(self, state, values):
        """Method for printing operators, grouped"""
        super().print_operators(state, values)
        print("ROCK DENSITY", values[self.ROCK_DENS_OP])
        return


class SinglePhaseGeomechanicsOperators(OperatorsBase):
    def evaluate(self, state, values):
        """
        Evaluate the single-phase geomechanical operators for the super engine

        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]
        :type state: value_vector in open-darts, pylvarray.Array in GEOS
        :param values: values of the operators (used for storing the operator values)
        :type values: value_vector in open-darts, pylvarray.Array in GEOS
        :return: updated value for operators, stored in values
        """
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state, values):
            return 0

        state_np = state.to_numpy()
        values_np = values.to_numpy()
        self.property.evaluate(state_np)
        values_np[0] = self.property.dens[0]
        values_np[1] = self.property.dens[0] / self.property.mu[0]

        return 0
