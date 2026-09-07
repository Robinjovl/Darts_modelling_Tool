from typing import Any

import numpy as np

from darts.engines import value_vector
from darts.physics.properties.basic import ConstFunc, RockCompactionEvaluator
from darts.physics.properties.flash import Flash
from darts.physics.properties.hysteresis import (
    HistoryAwareCapPressure,
    HistoryAwareRelPerm,
)


class PropertyContainer:
    # Interface attributes (folded in from the former PropertyBase):
    nc: int
    nph: int
    output_props = {}

    def __init__(
        self,
        phases_name: list,
        components_name: list,
        Mw: list,
        nc_kin: int = 0,
        np_kin: int = 0,
        solid_phase_idxs: list = None,
        solid_comp_idxs: list = None,
        eps_z: float = 1e-11,
        rock_comp: float = 1e-6,
        temperature: float = None,
        n_history: int = 0,
    ):
        """
        This is the PropertyContainer class for the compositional engine.

        :param phases_name: List of phases
        :type phases_name: list[str]
        :param components_name: List of components
        :type components_name: list[str]
        :param Mw: List of molecular weights [g/mol] for the nc_eq equilibrium
                   components (matching x's columns), ordered like components_name.
                   Kinetic phases' own molar mass is derived from it (see evaluate()).
        :type Mw: list[float]
        :param nc_kin: Number of kinetic (non-equilibrium) components, default is 0
        :type nc_kin: int
        :param np_kin: Number of kinetic (non-equilibrium) phases, default is 0
        :type np_kin: int
        :param solid_phase_idxs: Indices into ``phases_name`` of the non-flowing phases
                      (no kr/mu/pc/diffusion needed), default is ``None`` (the kinetic
                      phases, i.e. the last ``np_kin`` phases). A non-flowing phase need
                      not be a kinetic one -- e.g. an equilibrium-computed precipitate --
                      so this is independent of ``np_kin`` and may be overridden.
        :type solid_phase_idxs: list[int], optional
        :param solid_comp_idxs: Indices into ``components_name`` of the components with
                      no flux/diffusion contribution to their conservation equation
                      (purely accumulation + local reaction source, hence block-diagonal
                      in the Jacobian and eligible for Schur-complement elimination),
                      default is ``None`` (the kinetic components, i.e. the last
                      ``nc_kin`` components -- FLUX_OP/GRAD_OP are only ever populated
                      for the first ``nc_eq`` columns). Independent of ``nc_kin`` and may
                      be overridden, e.g. if a component is only ever present in
                      non-flowing phases.
        :type solid_comp_idxs: list[int], optional
        :param eps_z: Minimum bound of component mole fractions in OBL grid, default is 1e-11
        :type eps_z: float
        :param rock_comp: Rock compressibility, default is 1e-6
        :type rock_comp: float
        :param temperature: Constant temperature for isothermal simulation, default is None (thermal)
        :type temperature: float, optional
        :param n_history: Number of OBL history variables (e.g. ``sg_max``) appended to the state
                      after the primary Newton unknowns. ``0`` disables history-aware dispatch
                      and matches legacy behaviour; set by :class:`PhysicsBase` through
                      ``add_property_region``
        :type n_history: int
        """
        # This class contains all the property evaluators required for simulation
        self.components_name = components_name
        self.phases_name = phases_name
        self.nc = len(components_name)
        self.nph = len(phases_name)
        self.nc_kin = nc_kin
        self.np_kin = np_kin
        self.nc_eq = self.nc - nc_kin
        self.np_eq = self.nph - np_kin

        # Kinetic phases (np_kin of them) are, structurally, always the last phases
        # in phases_name/nu/X -- that ordering comes from Flash.set_kinetic_phase(),
        # which appends kinetic phases after the equilibrium ones, and is not a
        # user choice. This index set is used only by the kinetic-phase machinery
        # below (run_flash's reshape, and the zc-driven kinetic loops in evaluate()).
        self.kin_phase_idxs = np.arange(self.np_eq, self.nph)

        # solid_phase_idxs is a separate, independent concept: which phases don't
        # flow (no kr/mu/pc/diffusion), used to derive fluid_phase_idxs. A
        # non-flowing phase need not be a kinetic phase -- an equilibrium phase can
        # equally be non-flowing (e.g. a precipitated salt phase from the flash) --
        # so this is not required to have the same length as np_kin, and defaults
        # to (but is not tied to) the kinetic phases.
        self.solid_phase_idxs = (
            np.asarray(solid_phase_idxs, dtype=int)
            if solid_phase_idxs is not None
            else self.kin_phase_idxs
        )
        self.fluid_phase_idxs = np.setdiff1d(
            np.arange(self.nph), self.solid_phase_idxs, assume_unique=True
        )

        # solid_comp_idxs: components whose conservation equation has no flux/diffusion
        # contribution (only ACC_OP + local KIN_OP reaction source), i.e. purely
        # diagonal in the Jacobian across grid blocks -- eligible for Schur-complement
        # elimination. Independent of solid_phase_idxs/fluid_phase_idxs (that's a
        # phase-level, mobility concept; this is component-level and equation-level),
        # though today's default (the kinetic components) coincides with them since
        # kinetic phases are exactly the non-flowing ones by default.
        self.solid_comp_idxs = (
            np.asarray(solid_comp_idxs, dtype=int)
            if solid_comp_idxs is not None
            else np.arange(self.nc_eq, self.nc)
        )
        self.fluid_comp_idxs = np.setdiff1d(
            np.arange(self.nc), self.solid_comp_idxs, assume_unique=True
        )

        # Mw covers only the nc_eq equilibrium components, matching x's columns.
        # Dict-keyed Mw (used by the chemistry PropertyContainer subclass, keyed by
        # component name rather than position) is passed through as-is.
        self.Mw = Mw if isinstance(Mw, dict) else np.asarray(Mw)
        self.eps_z = eps_z
        # Number of OBL history variables (e.g. sg_max) appended to the state vector after
        # the primary Newton unknowns. 0 disables history-aware dispatch entirely.
        self.n_history = int(n_history)
        # Ordered labels for the appended history variables (populated by PhysicsBase.add_property_region).
        # When present and non-empty, evaluate() extracts one trailing scalar per label and passes
        # them as kwargs to HistoryAware* evaluators.
        self.history_labels: list[str] = []
        # Last extracted {label: value} map; refreshed on every evaluate() call. Evaluators that
        # consume more than one history variable can read this directly.
        self.history_values: dict[str, float] = {}

        if temperature:  # constant T specified
            self.thermal = False
            self.temperature = temperature
        else:
            self.thermal = True
            self.temperature = None

        # In case of PH-formulation, PT flashes are required for calculating initial distribution (Initialize class)
        self.evaluate_PT_bool = False  # set to True when PH-formulation but PT-flash needs to be calculated (Initialize)

        # Allocate (empty) evaluators for functions
        self.density_ev = {}
        self.viscosity_ev = {}
        self.enthalpy_ev = {}
        self.conductivity_ev = {}
        self.IFT_ev = {}

        self.rel_perm_ev = []
        self.rel_well_perm_ev = []
        self.rock_compr_ev = RockCompactionEvaluator(compres=rock_comp)
        self.rock_density_ev = ConstFunc(2650.0)
        self.capillary_pressure_ev = ConstFunc(np.zeros(self.np_eq))
        self.diffusion_ev = {
            phases_name[j]: ConstFunc(np.zeros(self.nc_eq))
            for j in self.fluid_phase_idxs
        }
        self.kinetic_rate_ev = {}
        self.energy_source_ev = {}
        self.flash_ev: Flash = 0
        # Set by run_flash(): whether flash_ev itself handles kinetic phases (via
        # Flash.set_kinetic_phase()), in which case self.x for kinetic phases is
        # populated and meaningful; see run_flash().
        self.flash_handles_kinetics = False
        self.permporo_mult_ev = ConstFunc(1.0)

        # passing arguments
        # x/x_mass cover all phases (equilibrium + kinetic): phase compositions are
        # (np_eq + np_kin) x nc_eq, since kinetic-phase composition is expressed over
        # the regular (equilibrium) component columns (see Flash.set_kinetic_phase).
        self.x = np.zeros((self.nph, self.nc_eq))
        self.x_mass = np.zeros((self.nph, self.nc_eq))
        self.dens = np.zeros(self.nph)
        self.dens_m = np.zeros(self.nph)
        self.sat = np.zeros(self.nph)
        self.nu = np.zeros(self.nph)
        self.mu = np.zeros(self.np_eq)
        self.kr = np.zeros(self.np_eq)
        self.pc = np.zeros(self.np_eq)
        self.enthalpy = np.zeros(self.nph)
        self.cond = np.zeros(self.nph)
        self.dX = []
        self.mass_source = np.zeros(self.nc)
        self.energy_source = 0.0

        self.phase_props = [
            self.dens,
            self.dens_m,
            self.sat,
            self.mu,
            self.kr,
            self.pc,
            self.enthalpy,
            self.cond,
            self.mass_source,
        ]

        self.output_props = {"sat0": lambda: self.sat[0]}

    def check_properties(self):
        """
        Check consistency of input properties
        """
        # Check that all phases have a density and enthalpy/conductivity evaluator in case of thermal
        # and all mobile phases have a viscosity/diffusion/relperm evaluator
        acc_evs = {"density": self.density_ev} | (
            {"enthalpy": self.enthalpy_ev, "conductivity": self.conductivity_ev}
            if self.thermal
            else {}
        )
        flux_evs = {
            "viscosity": self.viscosity_ev,
            "diffusion": self.diffusion_ev,
            "rel_perm": self.rel_perm_ev,
        }

        for name, ev in acc_evs.items():
            for phase in self.phases_name:
                assert phase in ev.keys() and ev[phase] is not None, (
                    f"Acc evaluator '{name}' missing for phase '{phase}'"
                )
        for name, ev in flux_evs.items():
            for j in self.fluid_phase_idxs:
                phase = self.phases_name[j]
                assert phase in ev.keys() and ev[phase] is not None, (
                    f"Flux evaluator '{name}' missing for phase '{phase}'"
                )

        for name, kinetic_ev in self.kinetic_rate_ev.items():
            assert kinetic_ev is not None, (
                f"Kinetic rate evaluator missing for '{name}'"
            )
        for name, energy_ev in self.energy_source_ev.items():
            assert energy_ev is not None, (
                f"Energy source evaluator missing for '{name}'"
            )

    def validate_history_consistency(self) -> None:
        """Assert that every history-aware evaluator in this container that owns a
        :class:`~darts.physics.properties.hysteresis.KilloughLandModel` (or any other
        trapping-model object exposed as ``.history_model``) agrees on its parameters.

        Catches silent drift between :attr:`rel_perm_ev` and
        :attr:`capillary_pressure_ev` whose evaluators each build their own model from
        independent Corey/parameter sources. Call once after the container is fully
        populated; :class:`~darts.physics.base.physics.PhysicsBase` invokes this
        from :meth:`init_physics`.

        :raises AssertionError: If two evaluators expose ``history_model`` instances of
                                the same type but with non-equal parameter values.
        """
        seen: dict[str, Any] = {}
        sources: dict[str, list[str]] = {}

        def _consider(label: str, ev) -> None:
            model = getattr(ev, "history_model", None)
            if model is None:
                return
            key = type(model).__name__
            if key not in seen:
                seen[key] = model
                sources[key] = [label]
                return
            sources[key].append(label)
            if model != seen[key]:
                raise AssertionError(
                    f"{key} parameters disagree across evaluators in this "
                    f"PropertyContainer: {sources[key][0]} has {seen[key]}, "
                    f"{label} has {model}. All hysteresis-bearing evaluators in one "
                    f"region must share the same trapping parameters."
                )

        if isinstance(self.rel_perm_ev, dict):
            for ph, ev in self.rel_perm_ev.items():
                _consider(f"rel_perm_ev[{ph!r}]", ev)
        if isinstance(self.capillary_pressure_ev, dict):
            for ph, ev in self.capillary_pressure_ev.items():
                _consider(f"capillary_pressure_ev[{ph!r}]", ev)

    def get_state(self, state):
        """
        Get tuple of (pressure, state_spec_2 (temperature/enthalpy/entropy),
                      [z0, ... zn-1]) at current OBL point (state)
        If isothermal, temperature returns initial temperature.
        If kinetic components are present, the modified variables zc* sum to 1 and correspond to saturation for the
        kinetic components. To obtain mole fractions of the equilibrium components, one needs to normalize zc* for
        the equilibrium components.
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(
            vec_state_as_np[1 : self.nc], 1 - np.sum(vec_state_as_np[1 : self.nc])
        )
        if zc[-1] < 0.99 * self.eps_z:
            zc = self.comp_out_of_bounds(zc)

        if self.thermal:
            # Primary thermal state: [P, z_0..z_{nc-2}, T] = nc+1 elements. If n_history history
            # variables are appended, T sits at nc (end of primary block), not at [-1].
            state_spec_2 = vec_state_as_np[self.nc]
        else:
            state_spec_2 = self.temperature

        return pressure, state_spec_2, zc

    def comp_out_of_bounds(self, vec_composition):
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_sum = 0
        count_corr = 0
        check_vec = np.zeros((len(vec_composition),))

        for ith_comp, zi in enumerate(vec_composition):
            if zi < 0.99 * self.eps_z:
                # print(vec_composition)
                vec_composition[ith_comp] = self.eps_z
                count_corr += 1
                check_vec[ith_comp] = 1
            elif zi > 1 - (self.nc - 1) * self.eps_z - 1e-15:
                # print(vec_composition)
                vec_composition[ith_comp] = 1 - (self.nc - 1) * self.eps_z
                temp_sum += vec_composition[ith_comp]
            else:
                temp_sum += vec_composition[ith_comp]

        for ith_comp, zi in enumerate(vec_composition):
            if check_vec[ith_comp] != 1:
                vec_composition[ith_comp] = (
                    zi / temp_sum * (1 - count_corr * self.eps_z)
                )
        return vec_composition

    def clean_arrays(self):
        for a in self.phase_props:
            a[:] = 0
        for j in range(self.nph):
            self.x[j][:] = 0

    def compute_saturation(self, state_pt=None, evaluate_PT_from_PHflash: bool = False):
        """
        Compute phase saturations from molar phase fractions and phase densities.

        Two uses:
        - ``state_pt=None`` (default): used from within :meth:`evaluate`, where flash and
          phase densities (``self.ph``, ``self.dens_m``) have already been computed for
          the current state earlier in that call.
        - ``state_pt`` given: used for initial-conditions calculation (previously the
          separate ``compute_saturation_full()`` method). Runs the flash for the given
          PT-state and computes phase densities before computing saturations.

        :param state_pt: State (pressure, [temperature], compositions) to flash; if
                          ``None``, uses the already-computed ``self.ph``/``self.dens_m``
        :param evaluate_PT_from_PHflash: Passed to :meth:`run_flash` when ``state_pt`` is
                                          given, to evaluate PT-state from a PH-flash object
        :returns: Saturation of the first phase, ``self.sat[0]``
        """
        if state_pt is not None:
            pressure, temperature, zc = self.get_state(state_pt)
            self.clean_arrays()
            self.ph = self.run_flash(
                pressure, temperature, zc, evaluate_PT=evaluate_PT_from_PHflash
            )

            for j in self.ph:
                M = np.sum(self.Mw * self.x[j][:])
                self.dens_m[j] = (
                    self.density_ev[self.phases_name[j]].evaluate(
                        pressure, temperature, self.x[j, :]
                    )
                    / M
                )

        # Get saturations [volume fraction]
        vol = [self.nu[j] / self.dens_m[j] for j in self.ph]
        self.sat[self.ph] = vol / np.sum(vol)

        return self.sat[0]

    def compute_total_enthalpy(self, state_pt):
        # Evaluate flash at PT
        pressure, temperature, zc = self.get_state(state_pt)
        flash_type = getattr(self.flash_ev, "flash_type", 0)
        flash_type_value = getattr(flash_type, "value", flash_type)
        ph = self.run_flash(
            pressure,
            temperature,
            zc,
            evaluate_PT=self.evaluate_PT_bool or flash_type_value > 0,
        )

        # Compute molar enthalpy of multiphase mixture
        enthalpy = 0.0
        for j in ph:
            self.enthalpy_ev[self.phases_name[j]].evaluate_PT_bool = True
            enthalpy += self.nu[j] * self.enthalpy_ev[self.phases_name[j]].evaluate(
                pressure, temperature, self.x[j, :]
            )  # kJ/kmol
            self.enthalpy_ev[self.phases_name[j]].evaluate_PT_bool = False

        return enthalpy

    def run_flash(self, pressure, state_spec_2, zc, evaluate_PT: bool = False):
        # flash_ev only handles kinetic components/phases itself if it was configured
        # via Flash.set_kinetic_phase() (flash.py) -- signalled by its own np_kin > 0.
        # In that case it normalizes the kinetic part of zc internally and returns
        # (np_eq + np_kin) rows of nu/X, expressed over the mapped equilibrium
        # component columns, so self.x for those kinetic phases is meaningful and
        # can be used to derive e.g. their effective molar mass (see evaluate()).
        # Otherwise (flash_ev only knows about the nc_eq equilibrium components),
        # PropertyContainer must normalize the kinetic part of zc away itself before
        # calling it, and only the np_eq equilibrium phases' nu/X come back --
        # evaluate() then asserts, since there is no other source for a kinetic
        # phase's composition/molar mass in that case (see kin_phase_idxs loops).
        self.flash_handles_kinetics = getattr(self.flash_ev, "np_kin", 0) > 0
        if self.flash_handles_kinetics:
            zc_flash = zc
            n_rows = self.np_eq + self.np_kin
        elif self.nc_kin:
            zc_flash = zc[: self.nc_eq] / (1.0 - np.sum(zc[self.nc_eq :]))
            n_rows = self.np_eq
        else:
            zc_flash = zc
            n_rows = self.np_eq

        # Evaluates flash, then uses getter for nu and x - for compatibility with DARTS-flash
        if evaluate_PT:
            # In case of PH-formulation, PT flashes are required for calculating initial distribution
            error_output = self.flash_ev.evaluate(
                pressure, state_spec_2, zc_flash, evaluate_PT=True
            )
            flash_results = self.flash_ev.get_flash_results(evaluate_PT=True)
            self.temperature = state_spec_2
        else:
            error_output = self.flash_ev.evaluate(pressure, state_spec_2, zc_flash)
            flash_results = self.flash_ev.get_flash_results()
            self.temperature = flash_results.temperature

        self.nu[:n_rows] = np.array(flash_results.nu)

        # Flash.evaluate() (flash.py) already guarantees this shape (NaN-filled with
        # error_output incremented on failure) for flash_ev implementations that
        # subclass it. This is kept as a second line of defense because flash_ev can
        # also be an external implementation -- e.g. get_flash_results(evaluate_PT=...)
        # above isn't even part of the local Flash interface -- with no such guarantee.
        try:
            self.x[:n_rows] = np.array(flash_results.X).reshape(n_rows, self.nc_eq)
        except ValueError as e:
            print(e.args[0], pressure, state_spec_2, zc)
            error_output += 1

        # Set present equilibrium phase idxs -- kinetic phases are always present by
        # definition and are handled separately via kin_phase_idxs. An equilibrium
        # phase can be non-flowing (present in solid_phase_idxs) and still needs its
        # density computed below, so this is not restricted to fluid_phase_idxs.
        # (When only one equilibrium phase is present, Flash.evaluate() itself pins
        # that phase's composition to the equilibrium feed -- see
        # Flash._snap_single_phase_composition -- so no correction is needed here.)
        ph = np.array([j for j in range(self.np_eq) if self.nu[j] > 0], dtype=int)

        return ph

    def evaluate_mass_source(self, pressure, temperature, zc):
        self.dX = np.zeros(len(self.kinetic_rate_ev))
        for _j, reaction in self.kinetic_rate_ev.items():
            dm, self.dX[_j] = reaction.evaluate(
                pressure, temperature, self.x, self.sat[-1]
            )
            self.mass_source += dm

        return self.mass_source

    def evaluate(self, state: value_vector):
        """
        Evaluate the phase properties. Phase properties used only in the energy conservation equation
        are evaluated using a different method.

        :param state: state variables [pres, comp_0, ..., comp_N-1, temperature (optional)]
        :type state: value_vector
        """
        # Composition vector and pressure from state:
        pressure, state_spec_2, zc = self.get_state(state)

        self.clean_arrays()

        # Run flash
        self.ph = self.run_flash(
            pressure, state_spec_2, zc, evaluate_PT=self.evaluate_PT_bool
        )
        self.pressure = pressure
        assert self.pressure is not None, (
            "PropertyContainer does not specify self.pressure, should be set to "
            "pressure in case of pressure-based flash, "
            "self.flash.pressure in case of volume-based flash"
        )
        assert self.temperature is not None, (
            "PropertyContainer does not specify self.temperature, should be set to "
            "constant temperature in case of isothermal physics, "
            "self.flash.temperature in case of thermal"
        )

        for j in self.ph:
            # Density (and mass fraction) is needed for every present equilibrium
            # phase, whether or not it flows -- a non-flowing equilibrium phase
            # (in solid_phase_idxs) still contributes to the mixture.
            M = np.sum(self.Mw * self.x[j][: self.nc_eq])

            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(
                self.pressure, self.temperature, self.x[j, :]
            )  # output in [kg/m3]
            self.dens_m[j] = (
                self.dens[j] / M
            )  # molar density [kg/m3]/[kg/kmol]=[kmol/m3]

            self.x_mass[j, :] = (self.x[j, : self.nc_eq] * self.Mw) / sum(
                self.x[j, : self.nc_eq] * self.Mw
            )

        # Viscosity is a mobility property: only needed for present, flowing phases.
        for j in np.intersect1d(self.ph, self.fluid_phase_idxs, assume_unique=True):
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(
                self.pressure, self.temperature, self.x[j, :], self.dens[j]
            )  # output in [cp]

        self.compute_saturation()

        # Extract every appended history variable by label, preserving the physics-declared
        # order. history_labels is populated by PhysicsBase.add_property_region; when it's
        # empty but n_history > 0 we fall back to the legacy single-trailing-scalar layout and
        # assume the sole variable is named "sg_max".
        if self.n_history:
            tail = np.asarray(state)[-self.n_history :]
            labels = (
                self.history_labels
                if len(self.history_labels) == self.n_history
                else ["sg_max"]
            )
            self.history_values = {
                label: float(tail[k]) for k, label in enumerate(labels)
            }
        else:
            self.history_values = {}

        # Dispatch to history-aware evaluators: unpack {label: value} as kwargs so concrete
        # evaluators can accept any subset of history variables by name (e.g. sg_max=...).
        # Plain evaluators without the mixin are called with sat only, unchanged.
        # Capillary pressure and relative permeability are mobility properties: only
        # needed for present, flowing phases.
        ph_fluid = np.intersect1d(self.ph, self.fluid_phase_idxs, assume_unique=True)

        if isinstance(self.capillary_pressure_ev, dict):
            for j in ph_fluid:
                pc_ev = self.capillary_pressure_ev[self.phases_name[j]]
                if self.history_values and isinstance(pc_ev, HistoryAwareCapPressure):
                    self.pc[j] = pc_ev.evaluate(self.sat[j], **self.history_values)
                else:
                    self.pc[j] = pc_ev.evaluate(self.sat[j])
        else:
            self.pc[:] = self.capillary_pressure_ev.evaluate(self.sat)

        for j in ph_fluid:
            kr_ev = self.rel_perm_ev[self.phases_name[j]]
            if self.history_values and isinstance(kr_ev, HistoryAwareRelPerm):
                self.kr[j] = kr_ev.evaluate(self.sat[j], **self.history_values)
            else:
                self.kr[j] = kr_ev.evaluate(self.sat[j])

        if self.np_kin:
            assert self.flash_handles_kinetics, (
                "flash_ev must be configured via Flash.set_kinetic_phase() for every "
                "kinetic phase -- there is no other source for a kinetic phase's "
                "composition/molar mass."
            )
        for idx in self.kin_phase_idxs:
            j = idx - self.np_eq
            self.sat[idx] = zc[self.nc_eq + j]
            self.dens[idx] = self.density_ev[self.phases_name[idx]].evaluate(
                self.pressure, self.temperature
            )
            # x[idx, :] is meaningful here (populated via Flash.set_kinetic_phase's
            # component_map): a kinetic phase is a stoichiometric compound of its
            # mapped equilibrium components (e.g. CaCO3 = 1 mole Ca + 1 mole CO3), not
            # a mole-fraction blend of them, so its molar mass is the unweighted sum
            # of the mapped components' Mw -- read off which components are mapped
            # from x's nonzero entries (their actual fractions don't factor in).
            M = np.sum(self.Mw[self.x[idx, : self.nc_eq] > 0])
            self.dens_m[idx] = self.dens[idx] / M

        self.mass_source = self.evaluate_mass_source(
            self.pressure, self.temperature, zc
        )

        return

    def evaluate_thermal(self, state):
        """
        Evaluate the phase properties used only in the energy conservation equation

        :param state: state variables [pres, comp_0, ..., comp_N-1, temperature (optional)]
        :type state: value_vector
        """
        for j in self.ph:
            self.enthalpy[j] = self.enthalpy_ev[self.phases_name[j]].evaluate(
                self.pressure, self.temperature, self.x[j, :]
            )  # kJ/kmol
            self.cond[j] = self.conductivity_ev[self.phases_name[j]].evaluate(
                self.pressure, self.temperature, self.x[j, :], self.dens[j]
            )

        for idx in self.kin_phase_idxs:
            self.enthalpy[idx] = self.enthalpy_ev[self.phases_name[idx]].evaluate(
                self.pressure, self.temperature, self.x[0, :]
            )
            self.cond[idx] = self.conductivity_ev[self.phases_name[idx]].evaluate()

        # Heat source and Reaction enthalpy
        self.energy_source = 0.0
        for _, energy_source in self.energy_source_ev.items():
            self.energy_source += energy_source.evaluate(state)

        for _, reaction in self.kinetic_rate_ev.items():
            self.energy_source += reaction.evaluate_enthalpy(
                self.pressure, self.temperature, self.x, self.sat[-1]
            )

        return

    def set_output_props(self, props: dict):
        """
        :param props: Dictionary of lambdas with output properties to be evaluated
        :type props: dict[str, lambda]
        """
        self.output_props = props
        return
