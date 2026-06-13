import os
import sys
import warnings
from pathlib import Path

import numpy as np

import darts
from darts.physics.properties.flash_exceptions import FlashError

try:
    from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc
except ImportError:
    from phreeqpy.iphreeqc.phreeqc_com import IPhreeqc

# Databases directory co-located with this module
_DEFAULT_DB_DIR = Path(__file__).parent / 'databases'


class PhreeqcFlashError(FlashError):
    """
    Raised when PHREEQC cannot equilibrate a state.

    A :class:`FlashError` subclass so callers (e.g. the model's Newton loop) can catch
    *any* flash non-convergence and convert it into a timestep cut, without masking
    unrelated bugs. Raised when the primary and backup databases both fail AND either the
    dilution fallback is disabled or even maximal dilution does not converge.
    """


def _resolve_phreeqc_db_path(db_spec: str | os.PathLike) -> str:
    """Resolve PHREEQC database path.

    Accepts either an explicit path or a file name. If a name is given, the
    function looks under the default databases directory.
    """
    candidate = Path(db_spec)
    if candidate.exists():
        return str(candidate.resolve())
    # Try within default dir, with and without .dat suffix
    name = candidate.name
    attempts = [
        _DEFAULT_DB_DIR / name,
    ]
    if not candidate.suffix:
        attempts.append(_DEFAULT_DB_DIR / f"{name}.dat")
    for path in attempts:
        if path.exists():
            return str(path.resolve())
    # Fall back to provided string; load will likely fail but error will be clear
    return str(candidate)


class Flash:
    """
    Calculates chemical and vapour-liquid equilibrium using PHREEQC.

    Write doc
    :param min_z: minimal composition value
    :type min_z: float
    :param minerals: list of minerals
    :type minerals: list[str]
    :param components: list of components (elements in this case)
    :type components: list[str]
    :param temperature: temperature for isothermal case
    :type temperature: float | None
    :param gas_species: gas species to include in the GAS_PHASE section
    :type gas_species: list[str] | tuple[str, ...]
    :param tolerance: convergence tolerance for the equilibrium solver
    :type tolerance: float
    :param database_filename: path to PHREEQC database file for primary engine
    :type database_filename: str
    :param backup_database_filename: path to database file as a backup for primary database
    :type backup_database_filename: str
    """

    def __init__(
        self,
        min_z: float,
        minerals: list[str],
        components: list[str],
        temperature: float | None = None,
        gas_species: list[str] | tuple[str, ...] = ("CO2(g)", "H2O(g)"),
        tolerance: float = 1e-10,
        database_filename: str = "phreeqc.dat",
        backup_database_filename: str = "pitzer.dat",
        dilution_fallback: bool = True,
        dilution_step: float = 1.5,
        dilution_max_steps: int = 14,
        dilution_refine_steps: int = 8,
    ):
        """
        :param min_z: minimal composition value
        :param minerals: list of minerals
        :param components: list of components (elements in this case)
        :param temperature: temperature for isothermal case
        :param gas_species: gas species to include in the GAS_PHASE section
        :param database_filename: path to PHREEQC database file for primary engine
        :param backup_database_filename: path to database file as a backup for primary database
        :param dilution_fallback: if True, when both databases fail (typically at an
            unreachable, over-concentrated OBL supporting point), retry with progressively
            more solvent water until PHREEQC converges, and report the diluted-edge result
            instead of raising. If False, raise :class:`PhreeqcFlashError` on failure.
        :param dilution_step: geometric factor by which the solvent water mass is scaled up
            each escalation step while searching for a converging dilution.
        :param dilution_max_steps: max number of geometric escalation steps before giving up
            (dilution_step ** dilution_max_steps is the largest factor tried).
        :param dilution_refine_steps: number of bisection steps used to refine the dilution
            factor back toward the convergence edge (smoother operators, closer to physical).
        """
        self.minerals = minerals
        self.components = components
        self.n_fluid = len(self.components)
        self.n_solid = len(self.minerals)
        # maps of component names to their indices for local treatment
        self.fc_idx = {comp: i for i, comp in enumerate(self.components)}
        # boolean mask for extraction of fluid components from state vector
        self.f_mask_state = np.concatenate(
            [[False] * (self.n_solid + 1), [True] * (self.n_fluid - 1)]
        )
        # TODO: unify mineral and miniral_names
        self.mineral_names = [item.split('_', 1)[1] for item in self.minerals]

        if temperature is None:
            self.thermal = True
        else:
            self.thermal = False
            self.temperature = temperature - 273.15
        self.min_z = min_z
        self.total_moles = 1000
        self.molar_weight_h2o = 0.018016

        # phreeqc
        self.tolerance = tolerance
        root = os.path.dirname(darts.__file__)
        if sys.platform.startswith("win"):
            libname = "IPhreeqc.dll"
        else:
            libname = "libIPhreeqc.so"
        self.phreeqc = IPhreeqc(os.path.join(root, libname))
        self.load_database(self.phreeqc, database_filename)
        self.backup_phreeqc = IPhreeqc(os.path.join(root, libname))
        self.load_database(self.backup_phreeqc, backup_database_filename)
        # self.phreeqc.set_output_file_on()
        # self.phreeqc.set_selected_output_file_on()

        self.gas_species = list(gas_species)
        # Precompute gas-related helpers
        self._gases_selected_output = " ".join(self.gas_species)
        self._gas_phase_entries = "\n".join([f"{sp}    0.0" for sp in self.gas_species])
        # Initial guess partial pressures [atm] per gas species
        self.gas_partial_pressures = {sp: 0.0 for sp in self.gas_species}
        # Parse gas species formulas to element stoichiometry for dynamic aggregation
        self._gas_species_element_stoich = {}
        for sp in self.gas_species:
            formula = sp.split("(")[0]  # e.g., 'CO2(g)' -> 'CO2'
            self._gas_species_element_stoich[sp] = self._parse_formula_to_elements(
                formula
            )
        if set(self.minerals) == {'Solid_CaCO3'}:  # pure calcite
            self.spec = 0
            self.aqueous_species = [
                "OH-",
                "H+",
                "H2O",
                "CH4",
                "HCO3-",
                "CO2",
                "CO3-2",
                "CaHCO3+",
                "CaCO3",
                "(CO2)2",
                "Ca+2",
                "CaOH+",
                "H2",
                "O2",
            ]
            self.species_2_element_moles = np.array(
                [2, 1, 3, 5, 5, 3, 4, 6, 5, 6, 1, 3, 2, 2]
            )
        elif set(self.minerals) == {
            'Solid_CaCO3',
            'Solid_CaMg(CO3)2',
        }:  # calcite and dolomite
            self.spec = 1
            self.aqueous_species = [
                "OH-",
                "H+",
                "H2O",
                "CH4",
                "HCO3-",
                "CO2",
                "CO3-2",
                "CaHCO3+",
                "MgHCO3+",
                "CaCO3",
                "MgCO3",
                "(CO2)2",
                "Ca+2",
                "CaOH+",
                "H2",
                "Mg+2",
                "MgOH+",
                "O2",
            ]
            self.species_2_element_moles = np.array(
                [2, 1, 3, 5, 5, 3, 4, 6, 6, 5, 5, 6, 1, 3, 2, 1, 3, 2]
            )
        elif set(self.minerals) == {
            'Solid_CaCO3',
            'Solid_CaMg(CO3)2',
            'Solid_MgCO3',
        }:  # calcite, dolomite and magnesite
            self.spec = 2
            self.aqueous_species = [
                "OH-",
                "H+",
                "H2O",
                "CH4",
                "HCO3-",
                "CO2",
                "CO3-2",
                "CaHCO3+",
                "MgHCO3+",
                "CaCO3",
                "MgCO3",
                "(CO2)2",
                "Ca+2",
                "CaOH+",
                "H2",
                "Mg+2",
                "MgOH+",
                "O2",
            ]
            self.species_2_element_moles = np.array(
                [2, 1, 3, 5, 5, 3, 4, 6, 6, 5, 5, 6, 1, 3, 2, 1, 3, 2]
            )

        # Unify PHREEQC template across specs; use high_precision=false and dynamic sections
        fluid_elements_order = [
            el for el, _ in sorted(self.fc_idx.items(), key=lambda kv: kv[1])
        ]
        element_headings = " ".join([f"{el}(mol)" for el in fluid_elements_order])
        element_punch = " ".join([f"TOTMOLE(\"{el}\")" for el in fluid_elements_order])

        # TODO: unify mineral and miniral_names
        mineral_label_map = {
            'Solid_CaCO3': 'Calcite',
            'Solid_CaMg(CO3)2': 'Dolomite',
            'Solid_MgCO3': 'Magnesite',
        }
        mineral_labels = [mineral_label_map.get(m, m) for m in self.minerals]
        if len(mineral_labels) > 0:
            sr_headings = " ".join([f"SR_{lbl}" for lbl in mineral_labels])
            sr_punch = " ".join([f"SR(\"{lbl}\")" for lbl in mineral_labels])
        else:
            sr_headings = "SR"
            sr_punch = "SR(\"Calcite\")"

        species_headings = " ".join([f'MOL("{sp}")' for sp in self.aqueous_species])
        species_punch = " ".join([f'MOL("{sp}")' for sp in self.aqueous_species])

        self.phreeqc_template = f"""
                USER_PUNCH
                -headings   {element_headings}   Vol_aq   {sr_headings}   ACT("H+") ACT("CO2") ACT("H2O") {species_headings}
                10 PUNCH    {element_punch} SOLN_VOL {sr_punch} ACT("H+") ACT("CO2") ACT("H2O") {species_punch}

                SELECTED_OUTPUT
                -selected_out    true
                -user_punch      true
                -reset           false
                -high_precision  false
                -gases           {self._gases_selected_output}

                SOLUTION 1
                temp      {{temperature:.4f}}
                pressure  {{pressure:.6f}}
                pH        7 charge
                -water    {{water_mass:.12f}} # kg

                REACTION 1
                {{reaction_lines}}
                1

                KNOBS
                -convergence_tolerance  {self.tolerance}

                GAS_PHASE 1
                pressure  {{pressure:.6f}}
                temp      {{temperature:.4f}}
                {{gas_phase_entries}}

                END
                """

        # Dilution fallback configuration (see __init__ docstring)
        self.dilution_fallback = dilution_fallback
        self.dilution_step = dilution_step
        self.dilution_max_steps = dilution_max_steps
        self.dilution_refine_steps = dilution_refine_steps
        # Per-nonlinear-iteration dilution tracker. Accumulated across all
        # diluted supporting points within one assembly pass; the model reads
        # and clears it once per Newton iteration to emit a single warning.
        self.reset_dilution_tracker()

    def reset_dilution_tracker(self):
        """
        Clear the accumulated record of diluted states (call once per Newton iteration).
        """
        self._diluted_states = []
        self._diluted_factors = []
        self._diluted_molality = []

    def pop_dilution_report(self):
        """
        Return per-iteration dilution statistics and clear the tracker.

        :return: ``None`` if no dilution occurred since the last reset, otherwise a dict with
            ``count`` (number of diluted supporting points), component-wise ``state_min`` /
            ``state_max`` of those states, ``factor_min`` / ``factor_max`` (dilution factors
            applied) and ``molality_min`` / ``molality_max`` (nominal pre-dilution molalities).
        :rtype: dict | None
        """
        if not self._diluted_states:
            return None
        states = np.array(self._diluted_states)
        report = {
            'count': len(self._diluted_states),
            'state_min': states.min(axis=0),
            'state_max': states.max(axis=0),
            'factor_min': float(min(self._diluted_factors)),
            'factor_max': float(max(self._diluted_factors)),
            'molality_min': float(min(self._diluted_molality)),
            'molality_max': float(max(self._diluted_molality)),
        }
        self.reset_dilution_tracker()
        return report

    def load_database(self, database, db_path):
        """
        Loads a PHREEQC database into the given database object.
        :param database: PHREEQC database object
        :param db_path: path to the PHREEQC database file
        """
        resolved = _resolve_phreeqc_db_path(db_path)
        try:
            database.load_database(resolved)
        except Exception as e:
            warnings.warn(f"Failed to load '{resolved}': {e}.", Warning, stacklevel=2)

    def interpret_results(self, database, water_mass):
        """
        Interprets the results of a PHREEQC simulation.
        :param database: PHREEQC database object
        :param water_mass: mass of water in kg
        :type water_mass: float
        :return: (nu_v) vapour phase molar fraction, (x) molar composition of aqueous
         and (y) vapour phases, (rho_phases) phase molar densities,
         (kin_state) kinetic params, (volume_aq + volume_gas) fluid volume,
         (species_aq_molar_fractions) species molar fractions in aqueous phase,
         and (species_gas_molar_fractions) species molar fractions in gaseous phase
        :rtype: tuple[float, np.ndarray, np.ndarray, dict, dict, float, np.ndarray, np.ndarray]
        """
        results_array = np.array(database.get_selected_output_array()[2])

        # Gas phase: volume and moles per species (in order of self.gas_species)
        volume_gas = results_array[2] / 1000  # liters to m3
        n_gases = len(self.gas_species)
        gas_moles = np.array(results_array[3 : 3 + n_gases])

        # Compute total gas "element moles" (sum over atoms) for consistent basis with aqueous TOTMOLE
        total_mole_gas = 0.0
        element_moles_in_gas = {el: 0.0 for el in self.fc_idx.keys()}
        # Aggregate elemental contributions from each gas species
        for sp, moles in zip(self.gas_species, gas_moles, strict=False):
            stoich = self._gas_species_element_stoich.get(sp, {})
            # Sum total atoms (only elements of interest)
            atoms_in_sp = 0
            for el, count in stoich.items():
                if el in element_moles_in_gas:
                    element_moles_in_gas[el] += count * moles
                    atoms_in_sp += count
            total_mole_gas += atoms_in_sp * moles

        # Gas species fractional composition by molecules
        sum_gas_moles = gas_moles.sum()
        if sum_gas_moles > 0:
            species_gas_molar_fractions = gas_moles / sum_gas_moles
        else:
            species_gas_molar_fractions = np.zeros_like(gas_moles)

        # interpret aqueous phase
        aq_start = 3 + n_gases
        mole_aq = results_array[aq_start : aq_start + self.n_fluid]

        volume_aq = results_array[aq_start + self.n_fluid] / 1000  # liters to m3
        total_mole_aq = mole_aq.sum()  # mol
        rho_aq = total_mole_aq / volume_aq / 1000  # kmol/m3

        # molar fraction of elements in aqueous phase
        nc = self.n_solid + self.n_fluid
        x = np.zeros(nc)
        x[self.n_solid :] = mole_aq / total_mole_aq

        # in gaseous phase
        y = np.zeros(nc)
        if total_mole_gas > 1.0e-10:
            rho_g = total_mole_gas / volume_gas / 1000  # kmol/m3
            # Fill elemental fractions in gas on an elemental basis
            for el, idx in self.fc_idx.items():
                y[self.n_solid + idx] = (
                    element_moles_in_gas.get(el, 0.0) / total_mole_gas
                )
        else:
            rho_g = 0.0

        # molar densities
        rho_phases = {'aq': rho_aq, 'gas': rho_g}
        # molar fraction of gaseous phase in fluid
        nu_v = total_mole_gas / (total_mole_aq + total_mole_gas)

        # interpret kinetic parameters
        counter = self.n_fluid + aq_start + 1
        kin_state = {}
        for name in self.mineral_names:
            kin_state['SR_' + name] = results_array[counter]
            counter += 1
        kin_state['Act(H+)'] = results_array[counter]
        kin_state['Act(CO2)'] = results_array[counter + 1]
        kin_state['Act(H2O)'] = results_array[counter + 2]
        species_molalities = results_array[counter + 3 :]

        species_aq_molar_fractions = (
            species_molalities * water_mass * self.species_2_element_moles
        )
        species_aq_molar_fractions /= species_aq_molar_fractions.sum()

        return (
            nu_v,
            x,
            y,
            rho_phases,
            kin_state,
            volume_aq + volume_gas,
            species_aq_molar_fractions,
            species_gas_molar_fractions,
        )

    def get_fluid_composition(self, state):
        """
        Extracts the fluid composition from the state vector using the boolean mask.
        :param state: state vector
        :type state: np.ndarray
        :return: fluid composition
        :rtype: np.ndarray
        """
        if self.thermal:
            z = state[-1][self.f_mask_state]
        else:
            z = state[self.f_mask_state]
        z_last = min(max(1 - np.sum(z), self.min_z), 1 - self.min_z)
        z = np.concatenate([z, [z_last]])
        return z

    def evaluate(self, state):
        """
        Calculates chemical and vapour-liquid equilibrium using PHREEQC.
        :param state: state vector
        :type state: np.ndarray
        :return: (nu_v) vapour phase molar fraction, (x) molar composition of aqueous
         and (y) vapour phases, (rho_phases) phase molar densities,
         (kin_state) kinetic params, (fluid_volume) fluid volume,
         (species_aq_molar_fractions) species molar fractions in aqueous phase,
         and (species_gas_molar_fractions) species molar fractions in gaseous phase
        :rtype: tuple[float, np.ndarray, np.ndarray, dict, dict, float, np.ndarray, np.ndarray]
        """
        # extract pressure and fluid composition
        pressure_atm = state[0] / 1.01325  # bar to atm

        # check for negative composition occurrence
        fluid_composition = self.get_fluid_composition(state)

        # calculate amount of moles of each component in 1000 moles of mixture
        fluid_moles = self.total_moles * fluid_composition

        # adjust oxygen and hydrogen moles for water formation
        init_h_moles, init_o_moles = (
            fluid_moles[self.fc_idx['H']],
            fluid_moles[self.fc_idx['O']],
        )
        if init_h_moles / 2 <= init_o_moles:
            water_mass = init_h_moles / 2 * self.molar_weight_h2o
            fluid_moles[self.fc_idx['H']] = 0
            fluid_moles[self.fc_idx['O']] = init_o_moles - init_h_moles / 2
        else:
            water_mass = init_o_moles * self.molar_weight_h2o
            fluid_moles[self.fc_idx['H']] = init_h_moles - 2 * init_o_moles
            fluid_moles[self.fc_idx['O']] = 0

        # Check if solvent (water) is enough
        # ion_strength = np.sum(fluid_moles) / (water_mass + 1.0e-8)
        # if ion_strength > 20:
        #     print(f'ion_strength = {ion_strength}')
        # assert ion_strength < 7, "Not enough water to form a realistic brine"

        # Generate and execute PHREEQC input
        # Build GAS_PHASE entries using current per-species initial guess partial pressures
        gas_phase_entries = "\n".join(
            [
                f"{sp}    {self.gas_partial_pressures.get(sp, 0.0):.6f}"
                for sp in self.gas_species
            ]
        )

        # Build REACTION block lines for present elements
        reaction_lines_list = []
        for el in self.components:
            reaction_lines_list.append(f"{el:<9}{fluid_moles[self.fc_idx[el]]:.12f}")
        reaction_lines = "\n                    ".join(reaction_lines_list)

        # Solve PHREEQC equilibrium. The element moles (reaction_lines) are fixed; only the
        # solvent water mass is scaled by the dilution fallback, so build the input lazily.
        def _build_input(wm):
            return self.phreeqc_template.format(
                temperature=self.temperature,
                pressure=pressure_atm,
                water_mass=wm,
                gas_phase_entries=gas_phase_entries,
                reaction_lines=reaction_lines,
            )

        try:
            self.phreeqc.run_string(_build_input(water_mass))
            results = self.interpret_results(self.phreeqc, water_mass)
        except Exception as e_primary:
            # Primary database (e.g. phreeqc.dat) failed to converge. Try the backup
            # database (e.g. pitzer.dat) at the same solvent mass first.
            try:
                self.backup_phreeqc.run_string(_build_input(water_mass))
                results = self.interpret_results(self.backup_phreeqc, water_mass)
            except Exception as e_backup:
                if not self.dilution_fallback:
                    raise PhreeqcFlashError(
                        f"PHREEQC did not converge (primary+backup) at p={state[0]:.6g} bar, "
                        f"water={water_mass:.6g} kg, "
                        f"fluid_moles={dict(zip(self.components, np.round(fluid_moles, 4), strict=False))}: "
                        f"{e_primary}"
                    ) from e_backup
                # Over-concentrated point (typically an unreachable OBL composition-box
                # corner). Dilute with extra solvent water until the primary database
                # converges, report the diluted-edge result, and record it for the model's
                # per-nonlinear-iteration warning instead of raising/spamming per point.
                results, factor = self._solve_with_dilution(
                    _build_input, water_mass, state
                )
                self._diluted_states.append(np.asarray(state, dtype=float).copy())
                self._diluted_factors.append(factor)
                self._diluted_molality.append(
                    float(np.sum(fluid_moles) / max(water_mass, 1.0e-30))
                )

        (
            nu_v,
            x,
            y,
            rho_phases,
            kin_state,
            fluid_volume,
            species_aq_molar_fractions,
            species_gas_molar_fractions,
        ) = results

        return (
            nu_v,
            x,
            y,
            rho_phases,
            kin_state,
            fluid_volume,
            species_aq_molar_fractions,
            species_gas_molar_fractions,
        )

    def _solve_with_dilution(self, build_input, water_mass, state):
        """
        Find the smallest extra-solvent dilution at which PHREEQC converges.

        Geometric escalation brackets a converging dilution factor, then a few bisection
        steps refine it back toward the convergence edge so the reported speciation stays as
        close as possible to the (unreachable) physical point. The diluted solvent mass is
        used for result interpretation so species fractions remain self-consistent. Densities
        and volumes are returned exactly as PHREEQC reports them (they are intensive and vary
        smoothly with the dilution factor); no 1/factor rescaling is applied.

        :param build_input: callable mapping a solvent water mass [kg] to a PHREEQC input string
        :param water_mass: base (undiluted) solvent water mass [kg]
        :param state: original state vector (for diagnostics only)
        :return: tuple ``(results, dilution_factor)`` where results is the interpret_results tuple
        :rtype: tuple
        """
        factor = 1.0
        results = None
        for _ in range(self.dilution_max_steps):
            factor *= self.dilution_step
            try:
                self.phreeqc.run_string(build_input(water_mass * factor))
                results = self.interpret_results(self.phreeqc, water_mass * factor)
                break
            except Exception:
                continue
        if results is None:
            raise PhreeqcFlashError(
                f"PHREEQC did not converge even after diluting solvent water by "
                f"x{factor:.1f} (base water={water_mass:.6g} kg) at p={state[0]:.6g} bar"
            )
        # Refine toward the convergence edge: bisect between the last failing factor
        # (factor / step) and the first converging factor.
        f_lo = factor / self.dilution_step
        f_hi = factor
        for _ in range(self.dilution_refine_steps):
            f_mid = 0.5 * (f_lo + f_hi)
            try:
                self.phreeqc.run_string(build_input(water_mass * f_mid))
                results = self.interpret_results(self.phreeqc, water_mass * f_mid)
                f_hi = f_mid
            except Exception:
                f_lo = f_mid
        return results, f_hi

    def set_gas_partial_pressures(self, gas_to_pressure_atm):
        """Set/update initial guess partial pressures [atm] for gas species in GAS_PHASE.
        Any species not provided keeps its current value.
        :param gas_to_pressure_atm: dictionary of gas species to partial pressures [atm]
        :type gas_to_pressure_atm: dict[str, float]
        """
        for sp, p in gas_to_pressure_atm.items():
            if sp in self.gas_partial_pressures:
                self.gas_partial_pressures[sp] = float(p)

    def _parse_formula_to_elements(self, formula):
        """
        Convert a chemical formula string (e.g., 'CO2', 'H2O', 'CH4') into a mapping of element symbol to atom count.
        Only elements present in the current fluid component set will be relevant.
        :param formula: chemical formula string
        :type formula: str
        :return: dictionary of element symbol to atom count
        :rtype: dict[str, int]
        """
        import re

        tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
        stoich = {}
        for el, count_str in tokens:
            count = int(count_str) if count_str else 1
            stoich[el] = stoich.get(el, 0) + count
        return stoich
