import warnings
from itertools import product

import numpy as np

from darts.interpolators import operator_set_evaluator_iface, value_vector
from darts.physics.base.property_base import PropertyBase


class OperatorsBase(operator_set_evaluator_iface):
    n_ops: int

    def __init__(
        self,
        property_container: PropertyBase,
        thermal: bool,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        Constructor of OperatorsBase base class

        :param property_container: Property container of type PropertyBase
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition OBL cell size(s) used to step onto neighbouring grid nodes
                    during boundary extrapolation. Scalar (uniform spacing) or a per-axis
                    vector of length nc-1 (non-uniform cell size across composition axes).
        """
        super().__init__()

        self.property = property_container

        self.thermal = thermal

        self.nc = property_container.nc
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
        Method that determines whether or not extrapolation should be applied to current state (z[-1] < 0).
        If so, it will call extrapolate() and return True, such that evaluate() skips further evaluation of operators

        :param state: Vector with state [P, z, (T/H)]
        :param values: Vector with operator values
        :return: Whether or not extrapolation has been applied to this state
        """
        # Find composition, if last composition is negative, apply extrapolation
        zc = np.append(state[1 : self.nc], 1 - np.sum(state[1 : self.nc]))

        if len(zc) > 2 and zc[-1] < 0.99 * self.eps_z and self.extrapolation_flag:
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
        property_container: PropertyBase,
        thermal: bool,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        Constructor of WellCtrlOperators class

        :param property_container: Property container of type PropertyBase
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
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

    def _fill_rate_ctrl_ops(self, values, offset, rate_factor):
        # Molar rate ctrl operator
        idx = offset
        values[idx + self.property.ph] = (
            self.property.dens_m[self.property.ph] * rate_factor
        )

        # Mass rate ctrl operator
        idx += self.nph
        values[idx + self.property.ph] = (
            self.property.dens[self.property.ph] * rate_factor
        )

        # Volumetric rate ctrl operator
        idx += self.nph
        values[idx + self.property.ph] = rate_factor

        # Advective heat rate ctrl operator
        idx += self.nph
        if self.thermal:
            values[idx + self.property.ph] = (
                self.property.enthalpy[self.property.ph]
                * self.property.dens_m[self.property.ph]
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

        epm_rate_factor = (
            self.property.kr[self.property.ph] / self.property.mu[self.property.ph]
        )
        self._fill_rate_ctrl_ops(
            values_np, self.epm_rate_ctrl_ops_offset, epm_rate_factor
        )

        # Store pressure (P) and temperature (T) of the current state for a generic state specification.
        # This is needed when pressure or temperature is not part of the state variables
        # (e.g., volume instead of pressure, or enthalpy instead of temperature).
        idx = self.state_ctrl_ops_offset
        values_np[idx + 0] = state[0]
        values_np[idx + 1] = self.property.temperature

        dfm_rate_factor = self.property.sat[self.property.ph]
        self._fill_rate_ctrl_ops(
            values_np, self.dfm_rate_ctrl_ops_offset, dfm_rate_factor
        )

        return 0


class ThermalVarOperator(OperatorsBase):
    """
    ThermalVarOperator gives the thermal variable for generic state specification
    """

    def __init__(
        self,
        property_container: PropertyBase,
        thermal: bool,
        is_pt: bool = True,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        Constructor of ThermalVarOperator class

        :param property_container: Property container of type PropertyBase
        :param thermal: Switch to indicate if energy conservation equation is there
        :param is_pt: Switch to indicate if state specification is P, PT, or PH
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
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
        property_container: PropertyBase,
        thermal: bool,
        props: dict = None,
        extrapolation_flag: bool = True,
        dz: float = None,
    ):
        """
        This is the constructor for PropertyOperator.
        The properties to be obtained from the PropertyOperators are passed to PropertyContainer as a dictionary.

        :param property_container: PropertyBase object to evaluate properties at given state
        :param thermal: Bool for thermal
        :param props: Optional dictionary of properties, default is taken from PropertyContainer
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
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
