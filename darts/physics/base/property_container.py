import warnings
from enum import Enum
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

    class KineticFormulation(Enum):
        """What a kinetic (non-equilibrium) phase's raw zc entry/entries represent."""

        BULK_VOLUME_FRACTION = 0
        """The raw zc entry already is the bulk volume fraction;
        Restricted to exactly one component per phase"""

        MOLE_FRACTION = 1
        """The raw zc entry is a mole fraction of the combined (fluid + kinetic) total;
        isn't necessarily restricted to one component per phase."""

    def __init__(
        self,
        phases_name: list,
        components_name: list,
        Mw: list,
        nc_kin: int = 0,
        np_kin: int = 0,
        solid_phase_idxs: list = None,
        solid_comp_idxs: list = None,
        kin_formulation: list = None,
        nc_kin_per_phase: list = None,
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
        :param solid_phase_idxs: Indices into ``phases_name`` of non-flowing phases (no
                      kr/mu/pc/diffusion). Default ``None`` (the kinetic phases);
                      independent of ``np_kin`` and overridable.
        :type solid_phase_idxs: list[int], optional
        :param solid_comp_idxs: Indices into ``components_name`` of components with no
                      flux/diffusion term (block-diagonal, Schur-eliminable). Default
                      ``None`` (the kinetic components); independent of ``nc_kin`` and
                      overridable.
        :type solid_comp_idxs: list[int], optional
        :param kin_formulation: One :class:`KineticFormulation` per kinetic phase
                      (``np_kin`` entries, Flash.set_kinetic_phase() order), or a
                      single :class:`KineticFormulation` broadcast to every kinetic
                      phase. Default ``None`` (``BULK_VOLUME_FRACTION`` for all).
        :type kin_formulation: list[KineticFormulation] | KineticFormulation, optional
        :param nc_kin_per_phase: Number of kinetic components each kinetic phase maps to
                      (``np_kin`` entries), or a single ``int`` broadcast to every
                      kinetic phase. Default ``None`` (``1`` per phase).
        :type nc_kin_per_phase: list[int] | int, optional
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

        self._setup_kinetic_and_phase_idxs(
            nc_kin,
            np_kin,
            kin_formulation,
            nc_kin_per_phase,
            solid_phase_idxs,
            solid_comp_idxs,
        )

        # Mw covers only nc_eq (matches x's columns); dict-keyed Mw (chemistry subclass) passed through as-is.
        self.Mw = Mw if isinstance(Mw, dict) else np.asarray(Mw)
        self.eps_z = eps_z
        # Number of OBL history variables (e.g. sg_max) appended to the state vector after
        # the primary Newton unknowns. 0 disables history-aware dispatch entirely.
        self.n_history = int(n_history)
        # Set by PhysicsBase.add_property_region(); evaluate() extracts one trailing scalar per label.
        self.history_labels: list[str] = []
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
            for j in self.mole_frac_phase_idxs
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

        # phi_s/phi_f: fraction of bulk volume not/available to the pooled eq_phase_idxs + MoleFractionKinetic (self.mole_basis_phase_idxs()) basis
        self.phi_s = 0.0
        self.phi_f = 1.0
        self.permporo_mult = 1.0

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

    def _setup_kinetic_and_phase_idxs(
        self,
        nc_kin: int,
        np_kin: int,
        kin_formulation: list,
        nc_kin_per_phase: list,
        solid_phase_idxs: list,
        solid_comp_idxs: list,
    ):
        """kin_formulation/nc_kin_per_phase (one entry per kinetic phase), the derived
        kin_phase_idxs/bulk_kin_phase_idxs/bulk_kin_comp_idxs/mole_kin_phase_idxs/
        mole_kin_comp_idxs, and the independent solid/fluid phase and component idx sets."""
        # A single (non-list) value is shorthand for "every kinetic phase";
        # the length/sum asserts below will still catch a genuine mismatch.
        if isinstance(kin_formulation, PropertyContainer.KineticFormulation):
            kin_formulation = [kin_formulation] * np_kin
        if isinstance(nc_kin_per_phase, int):
            nc_kin_per_phase = [nc_kin_per_phase] * np_kin

        # One formulation per kinetic phase; default: bulk volume fraction (legacy).
        if kin_formulation is None and np_kin:
            warnings.warn(
                f"kin_formulation not specified for {np_kin} kinetic phase(s); "
                "defaulting to BULK_VOLUME_FRACTION for all of them.",
                stacklevel=2,
            )
        self.kin_formulation: list[PropertyContainer.KineticFormulation] = (
            list(kin_formulation)
            if kin_formulation is not None
            else [self.KineticFormulation.BULK_VOLUME_FRACTION] * np_kin
        )
        assert len(self.kin_formulation) == np_kin, (
            f"kin_formulation has {len(self.kin_formulation)} entries, expected np_kin={np_kin}"
        )

        # Kinetic components per phase; default: 1 each (fixed composition).
        self.nc_kin_per_phase = (
            list(nc_kin_per_phase) if nc_kin_per_phase is not None else [1] * np_kin
        )
        assert len(self.nc_kin_per_phase) == np_kin, (
            f"nc_kin_per_phase has {len(self.nc_kin_per_phase)} entries, expected np_kin={np_kin}"
        )
        assert sum(self.nc_kin_per_phase) == nc_kin, (
            f"nc_kin_per_phase sums to {sum(self.nc_kin_per_phase)}, expected nc_kin={nc_kin}"
        )
        # Starting offset of each phase's own kinetic component(s) in the nc_kin block.
        self.kin_comp_offsets = np.concatenate(
            ([0], np.cumsum(self.nc_kin_per_phase)[:-1])
        )

        # Kinetic phases are always the last np_kin phases (Flash.set_kinetic_phase() order).
        self.kin_phase_idxs = np.arange(self.np_eq, self.nph)

        # BULK_VOLUME_FRACTION phases/components: dens_m*sat can't be split across
        # multiple components (no per-component fraction data), so exactly 1
        # component per such phase. Used for phi_s, the thermal solid-enthalpy term,
        # and the "bulk" ACC_OP term (operator_evaluator.py) -- a MOLE_FRACTION
        # phase's sat is normalized on a different (combined-total) basis not meant
        # to be summed directly alongside these.
        self.bulk_kin_phase_idxs = []
        self.bulk_kin_comp_idxs = []
        for j, idx in enumerate(self.kin_phase_idxs):
            if self.kin_formulation[j] == self.KineticFormulation.BULK_VOLUME_FRACTION:
                assert self.nc_kin_per_phase[j] == 1, (
                    f"kinetic phase {j} uses BULK_VOLUME_FRACTION but maps "
                    f"{self.nc_kin_per_phase[j]} kinetic components; only "
                    "MOLE_FRACTION supports more than 1"
                )
                self.bulk_kin_phase_idxs.append(idx)
                self.bulk_kin_comp_idxs.append(self.nc_eq + self.kin_comp_offsets[j])
        self.bulk_kin_phase_idxs = np.array(self.bulk_kin_phase_idxs, dtype=int)
        self.bulk_kin_comp_idxs = np.array(self.bulk_kin_comp_idxs, dtype=int)

        # MOLE_FRACTION phases/components: their raw zc is on the same combined-total
        # basis as the equilibrium components, so they're folded into the same
        # ACC_OP term as those (scaled by density_tot), not the bulk one.
        self.mole_kin_phase_idxs = np.setdiff1d(
            self.kin_phase_idxs, self.bulk_kin_phase_idxs, assume_unique=True
        )
        self.mole_kin_comp_idxs = np.setdiff1d(
            np.arange(self.nc_eq, self.nc), self.bulk_kin_comp_idxs, assume_unique=True
        )

        # All mole-fraction-based phases: equilibrium phases + MoleFractionKinetic
        # ones. Diffusion is defined on this set regardless of mobility (unlike
        # kr/mu/pc, which are mobility-gated -- see fluid_phase_idxs below).
        self.mole_frac_phase_idxs = np.concatenate(
            [np.arange(self.np_eq), self.mole_kin_phase_idxs]
        )

        # Non-flowing phases (no kr/mu/pc)
        self.solid_phase_idxs = (
            np.asarray(solid_phase_idxs, dtype=int)
            if solid_phase_idxs is not None
            else self.kin_phase_idxs
        )
        self.fluid_phase_idxs = np.setdiff1d(
            np.arange(self.nph), self.solid_phase_idxs, assume_unique=True
        )

        # Components with no flux/diffusion term (Schur-eliminable); independent of
        # solid_phase_idxs (phase/mobility concept vs. component/equation concept).
        self.solid_comp_idxs = (
            np.asarray(solid_comp_idxs, dtype=int)
            if solid_comp_idxs is not None
            else np.arange(self.nc_eq, self.nc)
        )
        self.fluid_comp_idxs = np.setdiff1d(
            np.arange(self.nc), self.solid_comp_idxs, assume_unique=True
        )

    def check_properties(self):
        """
        Check consistency of input properties
        """
        # Needed for every phase, mobile or not.
        all_phase_evs = {"density": self.density_ev} | (
            {"enthalpy": self.enthalpy_ev, "conductivity": self.conductivity_ev}
            if self.thermal
            else {}
        )
        # Needed only for mobile (flowing) phases.
        mobile_phase_evs = {
            "viscosity": self.viscosity_ev,
            "rel_perm": self.rel_perm_ev,
        }
        # Needed for mole-fraction-based phases (equilibrium + MoleFractionKinetic),
        # regardless of mobility -- diffusion isn't mobility-gated.
        mole_frac_phase_evs = {"diffusion": self.diffusion_ev}

        for name, ev in all_phase_evs.items():
            for phase in self.phases_name:
                assert phase in ev.keys() and ev[phase] is not None, (
                    f"Evaluator '{name}' missing for phase '{phase}'"
                )
        for name, ev in mobile_phase_evs.items():
            for j in self.fluid_phase_idxs:
                phase = self.phases_name[j]
                assert phase in ev.keys() and ev[phase] is not None, (
                    f"Evaluator '{name}' missing for mobile phase '{phase}'"
                )
        for name, ev in mole_frac_phase_evs.items():
            for j in self.mole_frac_phase_idxs:
                phase = self.phases_name[j]
                assert phase in ev.keys() and ev[phase] is not None, (
                    f"Evaluator '{name}' missing for mole-fraction-based phase '{phase}'"
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

    def mole_basis_phase_idxs(self) -> np.ndarray:
        """Present equilibrium phases (self.eq_phase_idxs) + MoleFractionKinetic
        phases (share saturation/accumulation/diffusion/conduction). Computed fresh
        from self.eq_phase_idxs on every call -- never cached, so it can't go stale
        across an evaluate()/compute_saturation() call or a subclass override that
        forgets to refresh it."""
        return np.concatenate([self.eq_phase_idxs, self.mole_kin_phase_idxs])

    def eq_phase_idxs_mobile(self) -> np.ndarray:
        """The flowing (fluid_phase_idxs) subset of mole_basis_phase_idxs(). Computed
        fresh on every call, same reasoning as mole_basis_phase_idxs()."""
        return np.intersect1d(
            self.mole_basis_phase_idxs(), self.fluid_phase_idxs, assume_unique=True
        )

    def compute_saturation(self, state_pt=None, evaluate_PT_from_PHflash: bool = False):
        """
        Compute phase saturations -- fluid AND kinetic -- from molar phase fractions
        and phase densities/molar masses. The only place PropertyContainer converts a
        kinetic phase's amount (self.nu, from Flash) to a physical volume quantity
        (via kin_formulation), so no other module (Flash, operator_evaluator) has to.

        Two uses:
        - ``state_pt=None`` (default): used from within :meth:`evaluate`, where flash,
          phase densities and equilibrium x/Mw (``self.eq_phase_idxs``, ``self.dens_m``, ``self.nu``)
          have already been computed for the current state earlier in that call.
        - ``state_pt`` given: used for initial-conditions calculation (previously the
          separate ``compute_saturation_full()`` method). Runs the flash for the given
          PT-state and computes phase densities.

        :param state_pt: State (pressure, [temperature], compositions) to flash; if
                          ``None``, uses the already-computed ``self.eq_phase_idxs``/``self.dens_m``
        :param evaluate_PT_from_PHflash: Passed to :meth:`run_flash` when ``state_pt`` is
                                          given, to evaluate PT-state from a PH-flash object
        :returns: Saturation of the first phase, ``self.sat[0]``
        """
        if state_pt is not None:
            pressure, temperature, zc = self.get_state(state_pt)
            self.clean_arrays()
            self.eq_phase_idxs = self.run_flash(
                pressure, temperature, zc, evaluate_PT=evaluate_PT_from_PHflash
            )
            self.pressure = pressure

            for j in self.eq_phase_idxs:
                M = np.sum(self.Mw * self.x[j][: self.nc_eq])
                self.dens_m[j] = (
                    self.density_ev[self.phases_name[j]].evaluate(
                        pressure, temperature, self.x[j, :]
                    )
                    / M
                )
        else:
            pressure, temperature = self.pressure, self.temperature

        if self.np_kin:
            assert self.flash_handles_kinetics, (
                "flash_ev must be configured via Flash.set_kinetic_phase() for every "
                "kinetic phase -- there is no other source for a kinetic phase's "
                "composition/molar mass."
            )

        # Molar mass = unweighted sum of mapped components' Mw (stoichiometric compound, not a blend).
        for idx in self.kin_phase_idxs:
            self.dens[idx] = self.density_ev[self.phases_name[idx]].evaluate(
                pressure, temperature
            )
            M = np.sum(self.Mw[self.x[idx, : self.nc_eq] > 0])
            self.dens_m[idx] = self.dens[idx] / M

        # BulkVolumeFractionKinetic: raw nu already is the bulk volume fraction.
        self.sat[self.bulk_kin_phase_idxs] = self.nu[self.bulk_kin_phase_idxs]

        # Equilibrium phases + MoleFractionKinetic phases, saturation-normalized
        # together: a MoleFractionKinetic phase's nu is on the same combined-total
        # basis as the (Flash-rescaled) equilibrium phases' nu (see
        # Flash.set_kinetic_phase(is_mole_fraction=True)), so they're one pool.
        mole_basis_phase_idxs = self.mole_basis_phase_idxs()
        vol = self.nu[mole_basis_phase_idxs] / self.dens_m[mole_basis_phase_idxs]
        self.sat[mole_basis_phase_idxs] = vol / np.sum(vol)

        return self.sat[0]

    def compute_total_enthalpy(self, state_pt):
        # Evaluate flash at PT
        pressure, temperature, zc = self.get_state(state_pt)
        flash_type = getattr(self.flash_ev, "flash_type", 0)
        flash_type_value = getattr(flash_type, "value", flash_type)
        eq_phase_idxs = self.run_flash(
            pressure,
            temperature,
            zc,
            evaluate_PT=self.evaluate_PT_bool or flash_type_value > 0,
        )

        # Compute molar enthalpy of multiphase mixture
        enthalpy = 0.0
        for j in eq_phase_idxs:
            self.enthalpy_ev[self.phases_name[j]].evaluate_PT_bool = True
            enthalpy += self.nu[j] * self.enthalpy_ev[self.phases_name[j]].evaluate(
                pressure, temperature, self.x[j, :]
            )  # kJ/kmol
            self.enthalpy_ev[self.phases_name[j]].evaluate_PT_bool = False

        return enthalpy

    def run_flash(self, pressure, state_spec_2, zc, evaluate_PT: bool = False):
        # flash_ev handles kinetics itself only if configured via Flash.set_kinetic_phase();
        # otherwise PropertyContainer normalizes the kinetic part of zc away itself.
        self.flash_handles_kinetics = getattr(self.flash_ev, "np_kin", 0) > 0
        if self.flash_handles_kinetics:
            # BULK_VOLUME_FRACTION components aren't mole fractions, so Flash never
            # normalizes for them (Flash.set_kinetic_phase(is_mole_fraction=False));
            # strip them from the fluid budget here instead, before Flash sees zc.
            # Exact given get_state()'s single closure (zc[:nc_eq].sum() +
            # zc_bulk_tot + zc_mole_tot == 1): dividing both the fluid slice and the
            # MOLE_FRACTION-kinetic slice by (1 - zc_bulk_tot) leaves them summing to
            # 1, so Flash's own (unchanged) is_mole_fraction=True normalization over
            # the now-already-bulk-adjusted MOLE_FRACTION entries is still exact.
            zc_flash = zc.copy()
            if self.bulk_kin_comp_idxs.size:
                zc_bulk_tot = np.sum(zc[self.bulk_kin_comp_idxs])
                zc_flash[: self.nc_eq] /= 1.0 - zc_bulk_tot
                zc_flash[self.mole_kin_comp_idxs] /= 1.0 - zc_bulk_tot
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

        # Second line of defense: flash_ev can be an external implementation with no shape guarantee.
        try:
            self.x[:n_rows] = np.array(flash_results.X).reshape(n_rows, self.nc_eq)
        except ValueError as e:
            print(e.args[0], pressure, state_spec_2, zc)
            error_output += 1

        # Present equilibrium phases; not restricted to fluid_phase_idxs since a
        # present phase can be non-flowing and still needs its density computed.
        eq_phase_idxs = np.array(
            [j for j in range(self.np_eq) if self.nu[j] > 0], dtype=int
        )

        return eq_phase_idxs

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
        self.eq_phase_idxs = self.run_flash(
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

        for j in self.eq_phase_idxs:
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
        eq_phase_idxs_mobile = self.eq_phase_idxs_mobile()
        for j in eq_phase_idxs_mobile:
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(
                self.pressure, self.temperature, self.x[j, :], self.dens[j]
            )  # output in [cp]

        self.compute_saturation()

        # phi_s: fraction of bulk volume NOT covered by self.mole_basis_phase_idxs() (sums only bulk_kin_phase_idxs)
        self.phi_s = np.sum(self.sat[self.bulk_kin_phase_idxs])
        self.phi_f = 1.0 - self.phi_s
        self.permporo_mult = self.permporo_mult_ev.evaluate(self.phi_f)

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

        if isinstance(self.capillary_pressure_ev, dict):
            for j in eq_phase_idxs_mobile:
                pc_ev = self.capillary_pressure_ev[self.phases_name[j]]
                if self.history_values and isinstance(pc_ev, HistoryAwareCapPressure):
                    self.pc[j] = pc_ev.evaluate(self.sat[j], **self.history_values)
                else:
                    self.pc[j] = pc_ev.evaluate(self.sat[j])
        else:
            self.pc[:] = self.capillary_pressure_ev.evaluate(self.sat)

        for j in eq_phase_idxs_mobile:
            kr_ev = self.rel_perm_ev[self.phases_name[j]]
            if self.history_values and isinstance(kr_ev, HistoryAwareRelPerm):
                self.kr[j] = kr_ev.evaluate(self.sat[j], **self.history_values)
            else:
                self.kr[j] = kr_ev.evaluate(self.sat[j])

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
        for j in self.eq_phase_idxs:
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
