import os
import sys
import warnings

import numpy as np

import darts

try:
    from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc
except ImportError:
    from phreeqpy.iphreeqc.phreeqc_com import IPhreeqc


class Flash:
    def __init__(
        self,
        min_z,
        fc_mask,
        fc_idx,
        f_mask_state,
        minerals,
        temperature=None,
        is_gas_spec=False,
    ):
        """
        :param min_z: minimal composition value
        :param fc_mask: boolean mask for extraction of fluid components from all components
        :param fc_idx: dictionary for mapping names of fluid components to filtered (via mask) state
        :param temperature: temperature for isothermal case
        """
        self.fc_mask = fc_mask
        self.fc_idx = fc_idx
        self.f_mask_state = f_mask_state
        self.n_fluid = np.count_nonzero(self.fc_mask)
        self.n_solid = np.count_nonzero(~self.fc_mask)
        self.minerals = minerals
        self.mineral_names = [item.split('_', 1)[1] for item in self.minerals]
        self.is_gas_spec = is_gas_spec

        if temperature is None:
            self.thermal = True
        else:
            self.thermal = False
            self.temperature = temperature - 273.15
        self.min_z = min_z
        self.total_moles = 1000
        self.molar_weight_h2o = 0.018016

        # phreeqc
        root = os.path.dirname(darts.__file__)
        if sys.platform.startswith("win"):
            libname = "IPhreeqc.dll"
        else:
            libname = "libIPhreeqc.so"
        self.phreeqc = IPhreeqc(os.path.join(root, libname))
        self.load_database(self.phreeqc, "phreeqc.dat")
        self.pitzer = IPhreeqc(os.path.join(root, libname))
        self.load_database(self.pitzer, "pitzer.dat")
        # self.phreeqc.phreeqc.OutputFileOn = True
        # self.phreeqc.phreeqc.SelectedOutputFileOn = True

        self.gas_species = ['CO2(g)', 'H2O(g)']
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
            self.phreeqc_species = [
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
            self.phreeqc_species = [
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
            self.phreeqc_species = [
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

        species_headings = " ".join([f'MOL("{sp}")' for sp in self.phreeqc_species])
        species_punch = " ".join([f'MOL("{sp}")' for sp in self.phreeqc_species])

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
                -convergence_tolerance  1e-10

                GAS_PHASE 1
                pressure  {{pressure:.6f}}
                temp      {{temperature:.4f}}
                {{gas_phase_entries}}

                END
                """

    def load_database(self, database, db_path):
        try:
            database.load_database(db_path)
        except Exception as e:
            warnings.warn(f"Failed to load '{db_path}': {e}.", Warning, stacklevel=2)

    def interpret_results(self, database):
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
            gas_fractions = gas_moles / sum_gas_moles
        else:
            gas_fractions = np.zeros_like(gas_moles)

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

        return (
            nu_v,
            x,
            y,
            rho_phases,
            kin_state,
            volume_aq + volume_gas,
            species_molalities,
            gas_fractions,
        )

    def get_fluid_composition(self, state):
        if self.thermal:
            z = state[-1][self.f_mask_state]
        else:
            z = state[self.f_mask_state]
        z_last = min(max(1 - np.sum(z), self.min_z), 1 - self.min_z)
        z = np.concatenate([z, [z_last]])
        return z

    def evaluate(self, state):
        """
        :param state: state vector with fluid composition accessible by fc_mask
        :type state: np.ndarray
        :return: phase molar fraction, molar composition of aqueous and vapour phases, kinetic params, solution volume
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
        reaction_elements = ['Ca', 'Mg', 'C', 'O', 'H']
        reaction_lines_list = []
        for el in reaction_elements:
            if el in self.fc_idx:
                reaction_lines_list.append(
                    f"{el:<9}{fluid_moles[self.fc_idx[el]]:.12f}"
                )
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
                species_aq_molalities,
                species_gas_molar_fractions,
            ) = self.interpret_results(self.phreeqc)
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
            self.pitzer.run_string(input_string)
            (
                nu_v,
                x,
                y,
                rho_phases,
                kin_state,
                fluid_volume,
                species_aq_molalities,
                species_gas_molar_fractions,
            ) = self.interpret_results(self.pitzer)

        species_aq_molar_fractions = (
            species_aq_molalities
            * water_mass
            * self.species_2_element_moles
            / self.total_moles
        )
        species_gas_molar_fractions *= nu_v
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
        """
        for sp, p in gas_to_pressure_atm.items():
            if sp in self.gas_partial_pressures:
                self.gas_partial_pressures[sp] = float(p)

    def _parse_formula_to_elements(self, formula):
        """
        Convert a chemical formula string (e.g., 'CO2', 'H2O', 'CH4') into a mapping of element symbol to atom count.
        Only elements present in the current fluid component set will be relevant.
        """
        import re

        tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
        stoich = {}
        for el, count_str in tokens:
            count = int(count_str) if count_str else 1
            stoich[el] = stoich.get(el, 0) + count
        return stoich


class KineticRate:
    def __init__(self, temperature, min_z, mineral, kinetic_mechanisms):
        self.temperature = temperature
        self.min_z = min_z
        self.mineral = mineral
        self.specific_sa = 0.925  # [m2/mol], default = 0.925
        self.SR_name = 'SR_' + self.mineral

        # doi: 10.3133/ofr20041068 for 25 celsius
        t_ref = 273.15 + 25
        if mineral == 'CaCO3':  # calcite
            acidic = self.ReactionMechanism(
                name='acidic', temperature_ref=t_ref, k=10 ** (-0.3), Ea=14400, n=1
            )
            neutral = self.ReactionMechanism(
                name='neutral', temperature_ref=t_ref, k=10 ** (-5.81), Ea=23500, n=0
            )
            carbonate = self.ReactionMechanism(
                name='carbonate', temperature_ref=t_ref, k=10 ** (-3.48), Ea=35400, n=1
            )
        elif mineral == 'CaMg(CO3)2':  # dolomite
            acidic = self.ReactionMechanism(
                name='acidic', temperature_ref=t_ref, k=10 ** (-3.19), Ea=36100, n=0.5
            )
            neutral = self.ReactionMechanism(
                name='neutral', temperature_ref=t_ref, k=10 ** (-7.53), Ea=52200, n=0
            )
            carbonate = self.ReactionMechanism(
                name='carbonate',
                temperature_ref=t_ref,
                k=10 ** (-5.11),
                Ea=34800,
                n=0.5,
            )
        elif mineral == 'MgCO3':  # magnesite
            acidic = self.ReactionMechanism(
                name='acidic', temperature_ref=t_ref, k=10 ** (-6.38), Ea=14400, n=1
            )
            neutral = self.ReactionMechanism(
                name='neutral', temperature_ref=t_ref, k=10 ** (-9.34), Ea=23500, n=0
            )
            carbonate = self.ReactionMechanism(
                name='carbonate', temperature_ref=t_ref, k=10 ** (-5.22), Ea=62800, n=1
            )
        self.mechanisms = []
        if 'acidic' in kinetic_mechanisms:
            self.mechanisms.append(acidic)
        if 'neutral' in kinetic_mechanisms:
            self.mechanisms.append(neutral)
        if 'carbonate' in kinetic_mechanisms:
            self.mechanisms.append(carbonate)

    def evaluate(self, kin_state, solid_saturation, rho_s):
        activities = [kin_state['Act(H+)'], 1.0, kin_state['Act(CO2)']]
        rates = [
            m.evaluate(
                temperature=self.temperature,
                activity=activities[i],
                SR=kin_state[self.SR_name],
            )
            for i, m in enumerate(self.mechanisms)
        ]

        # [mol/s/m3]
        kinetic_rate = (
            -self.specific_sa * solid_saturation * (rho_s * 1000) * sum(rates)
        )

        # [kmol/d/m3]
        kinetic_rate *= 60 * 60 * 24 / 1000
        return kinetic_rate

    class ReactionMechanism:
        def __init__(self, name, temperature_ref, k, Ea, n, p=1, q=1):
            self.SR_threshold = 100
            self.R = 8.314472

            self.name = name
            self.temperature_ref = temperature_ref  # gas constant [J/mol/Kelvin]
            self.k = k  # [mol * m-2 * s-1]
            self.Ea = Ea  # [J * mol-1]
            self.n = n  # reaction order with respect to given activity/anything
            self.p = p  # chemical affinity parameter
            self.q = q  # chemical affinity parameter

        def evaluate(self, temperature, activity, SR):
            k_arr = self.k * np.exp(
                (-self.Ea / self.R) * (1 / temperature - 1 / self.temperature_ref)
            )
            SR_bound = min(SR, self.SR_threshold)
            k_aff = (1 - SR_bound**self.p) ** self.q
            rate = k_arr * k_aff * activity**self.n
            return rate
