import os
import sys
import warnings

import numpy as np

import darts
from darts.physics.properties.kinetic_registry import KINETIC_REGISTRY

try:
    from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc
except ImportError:
    from phreeqpy.iphreeqc.phreeqc_com import IPhreeqc


# Pydantic is used to validate user-provided kinetic configuration
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


# -----------------------------
# Pydantic input schema
# -----------------------------
class MineralSpec(BaseModel):
    """Pydantic schema for a single-mineral kinetic configuration.

    Example:
        mineral_name='CaCO3', mechanisms=['acidic', 'neutral', 'carbonate']
    """

    mineral_name: str
    mechanisms: list[str]

    model_config = ConfigDict(extra='forbid')

    @field_validator('mineral_name')
    @classmethod
    def validate_mineral_name(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("'mineral_name' must be a non-empty string")
        mineral_key = value.strip()
        if mineral_key not in KINETIC_REGISTRY:
            raise ValueError(
                f"Unsupported mineral '{value}'. Supported: {sorted(KINETIC_REGISTRY.keys())}"
            )
        return mineral_key

    @field_validator('mechanisms')
    @classmethod
    def validate_mechanisms(cls, value: list[str], info) -> list[str]:
        if not isinstance(value, list | tuple) or len(value) == 0:
            raise ValueError("'mechanisms' must be a non-empty list of strings")
        # Access already-validated mineral_name from model fields
        # Pydantic v2 passes validated fields via 'data' on the model instance after init,
        # but here we re-validate against the full registry after normalization below.
        clean: list[str] = []
        for mech in value:
            if not isinstance(mech, str):
                raise ValueError("Mechanism names must be strings")
            mk = mech.strip().lower()
            if mk not in {'acidic', 'neutral', 'carbonate'}:
                raise ValueError(
                    "Unsupported mechanism '{mk}'. Supported: ['acidic','neutral','carbonate']"
                )
            if mk not in clean:
                clean.append(mk)
        return clean


class FlashSpec(BaseModel):
    """Pydantic spec for constructing a Flash instance.

    This model is JSON-schema friendly and validates only user inputs.
    """

    model_config = ConfigDict(extra='forbid')

    min_z: float = Field(gt=0)
    minerals: list[str] = Field(min_length=1)
    components: list[str] = Field(min_length=1)
    temperature: float | None = None

    # Optional overrides
    phreeqc_db: str = 'phreeqc.dat'
    pitzer_db: str = 'pitzer.dat'
    gas_species: list[str] = ['CO2(g)', 'H2O(g)']

    @field_validator('minerals')
    @classmethod
    def _validate_minerals(cls, value: list[str]) -> list[str]:
        if not all(isinstance(m, str) and m.strip() for m in value):
            raise ValueError('minerals must be non-empty strings')
        return [m.strip() for m in value]

    @field_validator('components')
    @classmethod
    def _validate_components(cls, comps: list[str]) -> list[str]:
        comps = [c.strip() for c in comps]
        # Require at least H and O due to water-formation logic
        for required in ('H', 'O'):
            if required not in comps:
                raise ValueError(f"components must include '{required}'")
        return comps


class ReactionSurfaceAreaSpec(BaseModel):
    """Spec for reaction surface area evaluator used by KineticRate.

    Currently supports a linear evaluator only.
    """

    model_config = ConfigDict(extra='forbid')

    kind: str = Field(default='linear', pattern='^(linear)$')
    initial_area_per_mol: float = Field(gt=0)


class KineticRateSpec(MineralSpec):
    """Pydantic spec for constructing a KineticRate instance.

    Inherits mineral/mechanism validation; adds `min_z` and surface area spec.
    """

    min_z: float = Field(gt=0)
    surface_area: ReactionSurfaceAreaSpec


class Flash:
    """
    Calculates chemical and vapour-liquid equilibrium using PHREEQC.
    Currently phreeqc.dat is main database while pitzer.dat is used as a back-up.

    """

    def __init__(
        self,
        min_z: float,
        minerals: list[str],
        components: list[str],
        temperature: float | None = None,
    ):
        """
        :param min_z: minimal composition value
        :param minerals: list of minerals
        :param components: list of components (elements in this case)
        :param temperature: temperature for isothermal case
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
        """
        Loads a PHREEQC database into the given database object.
        :param database: PHREEQC database object
        :param db_path: path to the PHREEQC database file
        """
        try:
            database.load_database(db_path)
        except Exception as e:
            warnings.warn(f"Failed to load '{db_path}': {e}.", Warning, stacklevel=2)

    @classmethod
    def from_spec(cls, spec: FlashSpec) -> 'Flash':
        """Construct a Flash instance from a validated FlashSpec.

        Converts JSON-friendly inputs to runtime types and applies optional
        overrides for databases and gas settings.
        """
        inst = cls(
            min_z=spec.min_z,
            minerals=spec.minerals,
            components=spec.components,
            temperature=spec.temperature,
        )

        # Override databases if requested
        if spec.phreeqc_db and spec.phreeqc_db != 'phreeqc.dat':
            try:
                inst.load_database(inst.phreeqc, spec.phreeqc_db)
            except Exception:
                pass
        if spec.pitzer_db and spec.pitzer_db != 'pitzer.dat':
            try:
                inst.load_database(inst.pitzer, spec.pitzer_db)
            except Exception:
                pass

        # Apply gas settings
        if spec.gas_species != inst.gas_species:
            inst.gas_species = list(spec.gas_species)
            inst._gases_selected_output = " ".join(inst.gas_species)
            # Recompute stoichiometry map for gases
            inst._gas_species_element_stoich = {}
            for sp in inst.gas_species:
                formula = sp.split("(")[0]
                inst._gas_species_element_stoich[sp] = inst._parse_formula_to_elements(
                    formula
                )

        return inst

    def interpret_results(self, database):
        """
        Interprets the results of a PHREEQC simulation.
        :param database: PHREEQC database object
        :return: (nu_v) vapour phase molar fraction, (x) molar composition of aqueous
         and (y) vapour phases, (rho_phases) phase molar densities,
         (kin_state) kinetic params, (volume_aq + volume_gas) fluid volume,
         (species_molalities) aqueous species molalities,
         and (gas_fractions) gas species molar fractions
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
         (species_aq_molar_fractions) aqueous species molar fractions,
         and (species_gas_molar_fractions) gas species molar fractions
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
    """Evaluate mineral kinetic rates.

    Input is validated against the kinetics registry via Pydantic.
    """

    def __init__(self, min_z, mineral_name, mechanisms, surface_area_ev):
        """Create a kinetic rate evaluator for a single mineral.

        Parameters
        - min_z: Minimum composition value (kept for API symmetry)
        - mineral_name: Mineral identifier, e.g. 'CaCO3'
        - mechanisms: List of mechanism names, e.g. ['acidic','neutral']
        - surface_area_ev: Surface area evaluator, e.g. LinearReactionSurfaceArea
        """
        self.min_z = min_z
        self.surface_area_ev = surface_area_ev
        try:
            spec = MineralSpec(mineral_name=mineral_name, mechanisms=mechanisms)
        except ValidationError as exc:
            raise exc
        self.mineral = spec.mineral_name
        mech_defs = KINETIC_REGISTRY[self.mineral]
        self.mechanisms = [
            ReactionMechanism(
                name=mech,
                temperature_ref=mech_defs[mech]['T_ref_K'],
                k=mech_defs[mech]['k'],
                Ea=mech_defs[mech]['Ea'],
                n=mech_defs[mech]['n'],
                p=mech_defs[mech]['p'],
                q=mech_defs[mech]['q'],
            )
            for mech in mechanisms
        ]

    def evaluate(self, kin_state, solid_saturation, rho_s, temperature):
        """Compute kinetic rate [kmol/d/m3] for the configured mineral.

        Parameters
        - kin_state: Dict-like with activities/saturation ratios from PHREEQC; expects keys
          'Act(H+)', 'Act(CO2)', and 'SR_<mineral>'
        - solid_saturation: Solid saturation (volume fraction) of the mineral
        - rho_s: Solid molar density [kmol/m3]
        - temperature: Temperature [K]
        """
        if not self.mechanisms:
            return 0.0

        # TODO: avoid dependence on PHREEQC format of kin_state
        # gather activities by mechanism
        activity_by_mech = {
            'acidic': kin_state['Act(H+)'],
            'neutral': 1.0,
            'carbonate': kin_state['Act(CO2)'],
        }

        # calculate rates by mechanism
        sr_key = 'SR_' + self.mineral
        rates = [
            mech.evaluate(
                temperature=temperature,
                activity=activity_by_mech[mech.name],
                SR=kin_state[sr_key],
            )
            for mech in self.mechanisms
        ]

        # calculate surface area [m2/mol]
        surface_area = self.surface_area_ev.evaluate(solid_saturation)

        # calculate kinetic rate [mol/s/m3]
        kinetic_rate = -surface_area * (rho_s * 1000) * sum(rates)

        # convert to [kmol/d/m3]
        kinetic_rate = kinetic_rate * 60 * 60 * 24 / 1000
        return kinetic_rate

    @classmethod
    def from_spec(cls, spec: KineticRateSpec) -> 'KineticRate':
        """Construct a KineticRate instance from a validated KineticRateSpec."""
        # Build surface area evaluator (extensible by kind)
        if spec.surface_area.kind == 'linear':
            sa_ev = LinearReactionSurfaceArea(spec.surface_area.initial_area_per_mol)
        else:
            raise ValueError(f"Unsupported surface area kind: {spec.surface_area.kind}")

        return cls(
            min_z=spec.min_z,
            mineral_name=spec.mineral_name,
            mechanisms=spec.mechanisms,
            surface_area_ev=sa_ev,
        )


class LinearReactionSurfaceArea:
    def __init__(self, initial_area_per_mol: float):
        """
        Initialize the reaction surface area evaluator.
        :param initial_area_per_mol: initial area per mol [m2/mol]
        :type initial_area_per_mol: float
        """
        self.s_init = initial_area_per_mol

    def evaluate(self, vol_fraction):
        return self.s_init * vol_fraction


class ReactionMechanism:
    """
    Class representing an Arrhenius-type reaction mechanism
    with chemical affinity term (1-SR**p)**q.
    """

    def __init__(self, name, temperature_ref, k, Ea, n, p=1, q=1):
        # maximum saturation ratio threshold for chemical affinity term
        self.SR_threshold = 100
        # universal gas constant [J/mol/K]
        self.R = 8.314472
        # name of the mechanism
        self.name = name
        # reference temperature when rate parameters are given [K]
        self.temperature_ref = temperature_ref
        # pre-exponential factor [mol/m2/s]
        self.k = k
        # activation energy [J/mol]
        self.Ea = Ea
        # reaction order with respect to given activity/anything
        self.n = n
        # chemical affinity parameter in (1-SR**p)**q term
        self.p = p
        # chemical affinity parameter in (1-SR**p)**q term
        self.q = q

    def evaluate(self, temperature, activity, SR):
        """
        Evaluate the reaction rate for a given temperature, activity, and saturation ratio.
        :param temperature: temperature [K]
        :param activity: activity of the reactant relevant to the mechanism
        :param SR: saturation ratio
        :return: reaction rate [mol/s/m2]
        """
        # calculate the Arrhenius factor
        k_arr = self.k * np.exp(
            (-self.Ea / self.R) * (1 / temperature - 1 / self.temperature_ref)
        )
        # impose maximum saturation ratio threshold on chemical affinity term
        SR_bound = min(SR, self.SR_threshold)
        # calculate the chemical affinity factor
        k_aff = (1 - SR_bound**self.p) ** self.q
        # calculate the reaction rate
        rate = k_arr * k_aff * activity**self.n
        return rate
