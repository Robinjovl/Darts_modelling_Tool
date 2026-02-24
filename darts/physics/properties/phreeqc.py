import os
import sys
import warnings
from pathlib import Path

import numpy as np

import darts

try:
    from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc
except ImportError:
    from phreeqpy.iphreeqc.phreeqc_com import IPhreeqc

# Databases directory co-located with this module
_DEFAULT_DB_DIR = Path(__file__).parent / 'databases'


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
    ):
        """
        :param min_z: minimal composition value
        :param minerals: list of minerals
        :param components: list of components (elements in this case)
        :param temperature: temperature for isothermal case
        :param gas_species: gas species to include in the GAS_PHASE section
        :param database_filename: path to PHREEQC database file for primary engine
        :param backup_database_filename: path to database file as a backup for primary database
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

        self._build_element_species_matrices()

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

    def interpret_results(self, database, water_mass, pressure_bar):
        """
        Interprets the results of a PHREEQC simulation.
        :param database: PHREEQC database object
        :param water_mass: mass of water in kg
        :type water_mass: float
        :param pressure_bar: pressure in bar
        :type pressure_bar: float
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

        co2_gas_idx = next(
            (
                i
                for i, sp in enumerate(self.gas_species)
                if sp == "CO2(g)" or sp == "CO2" or sp.startswith("CO2(")
            ),
            None,
        )
        if co2_gas_idx is not None and co2_gas_idx < len(species_gas_molar_fractions):
            y_co2 = float(species_gas_molar_fractions[co2_gas_idx])
            if not np.isfinite(y_co2):
                y_co2 = 0.0
            y_co2 = max(y_co2, 0.0)
        else:
            y_co2 = 0.0

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
        kin_state['P(CO2)'] = y_co2 * pressure_bar
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
        pressure_bar = float(state[0])
        pressure_atm = pressure_bar / 1.01325  # bar to atm

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
        ion_strength = np.sum(fluid_moles) / (water_mass + 1.0e-8)
        if ion_strength > 20:
            print(f'ion_strength = {ion_strength}')
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

        input_string = self.phreeqc_template.format(
            temperature=self.temperature,
            pressure=pressure_atm,
            water_mass=water_mass,
            gas_phase_entries=gas_phase_entries,
            reaction_lines=reaction_lines,
        )

        try:
            self.phreeqc.run_string(input_string)
            (
                nu_v,
                x,
                y,
                rho_phases,
                kin_state,
                fluid_volume,
                species_aq_molar_fractions,
                species_gas_molar_fractions,
            ) = self.interpret_results(self.phreeqc, water_mass, pressure_bar)
        except Exception as e:
            warnings.warn(f"Failed to run PHREEQC: {e}", Warning, stacklevel=2)
            if self.spec == 0:
                print(
                    f"h20_mass={water_mass}, p={state[0]}, Ca={fluid_moles[self.fc_idx['Ca']]}, C={fluid_moles[self.fc_idx['C']]}, O={fluid_moles[self.fc_idx['O']]}, H={fluid_moles[self.fc_idx['H']]}"
                )
            elif self.spec == 1 or self.spec == 2:
                print(
                    f"h20_mass={water_mass}, p={state[0]}, Ca={fluid_moles[self.fc_idx['Ca']]}, Mg={fluid_moles[self.fc_idx['Mg']]}, C={fluid_moles[self.fc_idx['C']]}, O={fluid_moles[self.fc_idx['O']]}, H={fluid_moles[self.fc_idx['H']]}"
                )
            self.backup_phreeqc.run_string(input_string)
            (
                nu_v,
                x,
                y,
                rho_phases,
                kin_state,
                fluid_volume,
                species_aq_molar_fractions,
                species_gas_molar_fractions,
            ) = self.interpret_results(self.backup_phreeqc, water_mass, pressure_bar)

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

    def _parse_species_formula_to_elements(self, species_name):
        """
        Parse a PHREEQC species name into element stoichiometry.

        Handles common PHREEQC notation:
        - charge suffixes (e.g., H+, Ca+2, CO3-2)
        - phase suffixes (e.g., CO2(g), H2O(aq))
        - grouped formulas with multipliers (e.g., (CO2)2, CaMg(CO3)2)
        """
        import re

        formula = species_name.strip()
        # Remove trailing phase marker, e.g., "(g)" or "(aq)"
        formula = re.sub(r"\([A-Za-z]+\)$", "", formula)
        # Remove trailing charge notation, e.g., "+", "-2", "+2", "2+"
        formula = re.sub(r"([+-]\d*|\d*[+-])$", "", formula)

        stack = [{}]
        i = 0
        while i < len(formula):
            ch = formula[i]

            if ch == "(":
                stack.append({})
                i += 1
                continue

            if ch == ")":
                i += 1
                j = i
                while j < len(formula) and formula[j].isdigit():
                    j += 1
                multiplier = int(formula[i:j] or "1")
                group = stack.pop() if len(stack) > 1 else {}
                for el, count in group.items():
                    stack[-1][el] = stack[-1].get(el, 0) + count * multiplier
                i = j
                continue

            if ch.isupper():
                j = i + 1
                if j < len(formula) and formula[j].islower():
                    j += 1
                element = formula[i:j]
                k = j
                while k < len(formula) and formula[k].isdigit():
                    k += 1
                count = int(formula[j:k] or "1")
                stack[-1][element] = stack[-1].get(element, 0) + count
                i = k
                continue

            # Skip any other symbols (charges, separators, etc.)
            i += 1

        return stack[0]

    def _build_element_species_matrices(self):
        """
        Build element-species stoichiometric matrices for aqueous and gas phases.

        Stored matrices:
        - *_element_to_species_matrix: shape (n_elements, n_species_in_phase)
        - *_species_to_element_matrix: shape (n_species_in_phase, n_elements)
        """
        element_symbols = [
            el for el, _ in sorted(self.fc_idx.items(), key=lambda kv: kv[1])
        ]

        self.system_elements = element_symbols
        self.system_species = list(self.aqueous_species) + list(self.gas_species)

        n_elements = len(element_symbols)
        self.aqueous_element_to_species_matrix = np.zeros(
            (n_elements, len(self.aqueous_species)), dtype=float
        )
        self.gas_element_to_species_matrix = np.zeros(
            (n_elements, len(self.gas_species)), dtype=float
        )

        for j, sp in enumerate(self.aqueous_species):
            stoich = self._parse_species_formula_to_elements(sp)
            for i, el in enumerate(element_symbols):
                self.aqueous_element_to_species_matrix[i, j] = stoich.get(el, 0.0)

        for j, sp in enumerate(self.gas_species):
            stoich = self._gas_species_element_stoich.get(sp, {})
            for i, el in enumerate(element_symbols):
                self.gas_element_to_species_matrix[i, j] = stoich.get(el, 0.0)

        self.aqueous_species_to_element_matrix = (
            self.aqueous_element_to_species_matrix.T
        )
        self.gas_species_to_element_matrix = self.gas_element_to_species_matrix.T
