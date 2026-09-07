import abc

import numpy as np
from numba import jit


class Flash:
    def __init__(self, nph, nc, nsalt=0):
        self.np_eq = nph
        self.nc = nc
        self.nsalt = nsalt
        self.nc_eq = nc + nsalt

        # One entry per registered kinetic phase: {'component_map', 'composition', 'nc_kin'}
        self._kinetic_phases: list[dict] = []
        self.nc_kin = 0
        self.np_kin = 0

        self.nu = []
        self.X = []
        self.temperature: float = 0.0

    def set_kinetic_phase(
        self,
        component_map: list,
        composition: list = None,
        is_mole_fraction: bool = True,
    ):
        """
        Register one kinetic (non-equilibrium) phase, appended after
        the equilibrium phases handled by this flash. Call once per kinetic phase;
        the kinetic zc entries consumed by each phase are read from ``zc`` in
        registration order, right after the ``nc_eq`` equilibrium components.

        The phase's composition is a mapping onto this flash's regular
        (equilibrium) components, given by ``component_map`` (indices into
        those components). Two ways to specify it:

        - **Fixed composition** (pass ``composition``): a single kinetic zc
          entry gives the phase's total molar amount; its mole fractions over
          ``component_map`` are the fixed ``composition`` (must sum to 1) and
          never change -- e.g. a pure mineral, or one with fixed stoichiometry.
        - **Variable composition** (leave ``composition`` as ``None``): one
          kinetic zc entry per entry of ``component_map``, giving the molar
          amount mapped onto each regular component; the phase's mole
          fractions are their normalized (renormalized) share, so the
          composition can vary.

        :param component_map: Indices into this flash's regular (equilibrium)
                               components that this kinetic phase's composition maps onto.
        :param composition: Fixed mole fractions over ``component_map`` (must sum to
                             1, one kinetic zc entry consumed), or ``None`` for a
                             variable composition (one kinetic zc entry per mapped
                             component consumed).
        :param is_mole_fraction: Whether this phase's kinetic zc entries should be
                             subtracted from the same sum-to-1 budget the fluid feed
                             is renormalized against. This is NOT simply "this
                             component uses MoleFractionKinetic" (see
                             :mod:`darts.physics.properties.kinetic_formulation`) --
                             it also has to be ``True`` whenever the fluid primary
                             variables were pre-scaled by the model to leave room for
                             this entry in a shared budget, regardless of what the
                             entry itself represents. E.g. every kinetic-component
                             model in this repo today uses BulkVolumeFractionKinetic
                             (the entry is a volume fraction) yet still needs
                             ``is_mole_fraction=True``, because that entry is their
                             state vector's closure component (``1 - sum(others)``)
                             and the fluid primaries are pre-scaled against it --
                             setting ``False`` would silently feed an unnormalized,
                             sub-1 composition into evaluate_equilibrium().

                             - ``True`` (default): included in the total subtracted
                               from the fluid feed before renormalizing it, and this
                               phase's own ``nu`` is set to the (normalized) zc entry
                               directly, matching the combined-total simplex. Always
                               correct for a MoleFractionKinetic component (that's
                               its definition); for a BulkVolumeFractionKinetic one,
                               correct only if the fluid primaries were pre-scaled
                               against this same entry as above.
                             - ``False``: excluded from the fluid-feed
                               normalization -- appropriate for a
                               BulkVolumeFractionKinetic component that is instead
                               an independently-given primary variable, with fluid
                               primaries that are already true, unscaled mole
                               fractions needing no adjustment for it. Flash has no
                               phase-density access and so cannot convert a volume
                               fraction to a mole fraction itself; this phase's
                               ``nu`` is still set to the raw zc entry, but is then
                               only a placeholder -- not a true mole fraction -- and
                               isn't meant to be used as one downstream.
        :type is_mole_fraction: bool
        """
        if composition is not None:
            assert len(composition) == len(component_map), (
                "composition must have one entry per mapped component"
            )
            nc_kin_phase = 1
        else:
            nc_kin_phase = len(component_map)

        self._kinetic_phases.append(
            {
                "component_map": list(component_map),
                "composition": None
                if composition is None
                else np.asarray(composition, dtype=float),
                "nc_kin": nc_kin_phase,
                "is_mole_fraction": is_mole_fraction,
            }
        )
        self.nc_kin += nc_kin_phase
        self.np_kin += 1

    @abc.abstractmethod
    def evaluate_equilibrium(self, pressure, temperature, zc):
        pass

    @staticmethod
    def _snap_single_phase_composition(nu, x, zc_eq):
        """If exactly one equilibrium phase is present: pin its composition to ``zc_eq``.

        :param nu: Phase molar fractions (equilibrium phases only), mutated in place.
        :param x: Phase compositions (equilibrium phases only), mutated in place.
        :param zc_eq: Normalized equilibrium-component feed composition.
        """
        present = [j for j in range(len(nu)) if nu[j] > 0]
        if len(present) == 1:
            x[present[0]] = zc_eq

    def evaluate(self, pressure, temperature, zc):
        """Evaluate flash, normalizing for kinetic components/phases if configured.

        If set_kinetic_phase() has not been called, this is equivalent to
        evaluate_equilibrium() over the full ``zc``. Otherwise, this normalizes
        the equilibrium part of ``zc`` (kinetic components removed), evaluates
        evaluate_equilibrium() on it, then re-appends the kinetic phase
        fractions to ``nu``/``X``.

        If evaluate_equilibrium() fails and leaves its phase compositions
        mis-shaped, the equilibrium part of ``X`` is filled with NaN and the
        error count is incremented instead of raising, so the caller (OBL
        point generation) can detect the failed supporting point downstream.

        :param pressure: Pressure at the evaluated point.
        :param temperature: Temperature (may be None for isothermal physics).
        :param zc: Overall composition, including kinetic components if configured.
        :return: Number of flash errors encountered (0 on success).
        """
        if self.np_kin == 0:
            error_output = self.evaluate_equilibrium(pressure, temperature, zc)
            self.nu = np.asarray(self.nu)
            try:
                self.X = np.asarray(self.X).reshape(self.np_eq, self.nc_eq)
            except ValueError as e:
                print(e.args[0], pressure, temperature, zc)
                error_output += 1
                # failed flash left X mis-shaped; keep going with NaN phase
                # compositions so the error is detectable downstream instead of
                # crashing on the undefined local
                self.X = np.full((self.np_eq, self.nc_eq), np.nan)
            self._snap_single_phase_composition(self.nu, self.X, zc)
            return error_output

        # Normalize compositions. Only the kinetic phases registered with
        # is_mole_fraction=True share the fluid components' mole-fraction simplex,
        # so only their zc entries are subtracted from it here -- a bulk-volume-
        # fraction phase's raw value can't validly enter this sum (Flash has no
        # phase-density access to convert it to a mole fraction itself).
        zc_kin = zc[self.nc_eq :]
        zc_kin_tot = 0.0
        offset = 0
        for kin in self._kinetic_phases:
            z_j = zc_kin[offset : offset + kin["nc_kin"]]
            offset += kin["nc_kin"]
            if kin["is_mole_fraction"]:
                zc_kin_tot += np.sum(z_j)
        zc_norm = zc[: self.nc_eq] / (1.0 - zc_kin_tot)

        # Evaluate flash for normalized composition
        error_output = self.evaluate_equilibrium(pressure, temperature, zc_norm)
        flash_results = self.get_flash_results()
        nu = np.array(flash_results.nu)
        try:
            x = np.array(flash_results.X).reshape(self.np_eq, self.nc_eq)
        except ValueError as e:
            print(e.args[0], pressure, temperature, zc)
            error_output += 1
            # failed flash left X mis-shaped; keep going with NaN phase
            # compositions so the error is detectable downstream instead of
            # crashing on the undefined local
            x = np.full((self.np_eq, self.nc_eq), np.nan)

        self._snap_single_phase_composition(nu, x, zc_norm)

        # Re-normalize kinetic phases and append to nu, x -- kinetic phase compositions
        # are written into the regular (equilibrium) component columns via each phase's
        # component_map, so X stays nc_eq components wide.
        NU = np.zeros(self.np_eq + self.np_kin)
        X = np.zeros((self.np_eq + self.np_kin, self.nc_eq))
        for j in range(self.np_eq):
            NU[j] = nu[j] * (1.0 - zc_kin_tot)
            X[j, :] = x[j, :]

        # NU[row] is set to the raw (registration-order) zc entry either way. It's a
        # true mole fraction of the combined total only for is_mole_fraction=True
        # phases (matching MoleFractionKinetic); for is_mole_fraction=False phases it
        # is just a placeholder passthrough -- not meant to be used as a mole
        # fraction downstream (see BulkVolumeFractionKinetic).
        offset = 0
        for j, kin in enumerate(self._kinetic_phases):
            z_j = zc_kin[offset : offset + kin["nc_kin"]]
            offset += kin["nc_kin"]
            row = self.np_eq + j
            if kin["composition"] is not None:
                NU[row] = z_j[0]
                X[row, kin["component_map"]] = kin["composition"]
            else:
                total = np.sum(z_j)
                NU[row] = total
                X[row, kin["component_map"]] = z_j / total if total > 0 else 0.0

        self.nu = NU
        self.X = X
        self.temperature = flash_results.temperature

        return error_output

    def get_flash_results(self):
        """Exists to keep the same call pattern as DARTS-flash, whose flash objects
        return a separate results object from this getter rather than storing
        nu/X/temperature on self; here evaluate() already sets them on self, so this
        just returns self."""
        return self


class SinglePhase(Flash):
    def __init__(self, nc):
        super().__init__(nph=1, nc=nc)

    def evaluate_equilibrium(self, pressure, temperature, zc):
        self.nu, self.X = np.array([1.0]), np.array([zc])
        self.temperature = temperature
        return 0


class ConstantK(Flash):
    def __init__(self, nc, ki, eps=1e-11, use_dartsflash: bool = False):
        """
        Constant K-value flash, option to make use of DARTS-flash RR solver.
        K-values are defined as Ki = yi/xi, with the y-phase returned as phase 0.
        The DARTS-flash RR solver defines K-values with phase 0 as reference phase
        (Ki = xi1/xi0), so K-values are inverted internally when use_dartsflash is enabled.

        :param nc: Number of components
        :param ki: K-values per component (Ki = yi/xi)
        :param eps: Epsilon value for composition
        :param use_dartsflash: Use DARTS-flash RR solver, default is False
        """
        super().__init__(nph=2, nc=nc)

        self.rr_eps = eps
        self.K_values = np.array(ki)

        self.use_dartsflash = use_dartsflash
        if use_dartsflash:
            from dartsflash.libflash import RR_EqConvex2

            self.rr = RR_EqConvex2(nc=nc, min_z=eps, rr_tol=1e-12, max_iter=100)

    def evaluate_equilibrium(self, pressure, temperature, zc):
        if self.use_dartsflash:
            # darts-flash uses phase 0 as reference phase (Ki = xi1/xi0), so invert
            # the K-values to keep the y-phase as phase 0 in the flash output
            self.rr.solve_rr(zc, 1.0 / self.K_values, np.array([]))
            self.nu, self.X = self.rr.getnu(), self.rr.getx()
            self.X = np.array(self.X).reshape(2, self.nc)

        else:
            self.nu, self.X = RR2(self.K_values, zc, self.rr_eps)

        self.temperature = temperature
        return 0


@jit(nopython=True)
def RR2(k, zc, eps):
    a = 1 / (1 - np.max(k)) + eps
    b = 1 / (1 - np.min(k)) - eps
    k_minus_1 = k - 1

    max_iter = 200  # use enough iterations for V to converge
    tol = 1e-12  # convergence tolerance

    for _i in range(1, max_iter):
        V = 0.5 * (a + b)
        r = np.sum(zc * k_minus_1 / (V * k_minus_1 + 1))
        if abs(r) < tol:
            break

        if r > 0:
            a = V
        else:
            b = V

    if _i >= max_iter:
        print("Flash warning!!!")

    x = zc / (V * k_minus_1 + 1)
    y = k * x

    return [V, 1 - V], [y, x]
