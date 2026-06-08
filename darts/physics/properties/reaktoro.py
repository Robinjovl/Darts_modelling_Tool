import numpy as np

from darts.physics.properties.flash_exceptions import FlashError

try:
    # Reaktoro v2 Python API
    from reaktoro import (
        AqueousPhase,
        AqueousProps,
        ChemicalState,
        ChemicalSystem,
        EquilibriumConditions,
        EquilibriumOptions,
        EquilibriumSolver,
        GaseousPhase,
        PhreeqcDatabase,
        SupcrtDatabase,
        speciate,
    )
except Exception as _exc:  # pragma: no cover - optional dependency
    PhreeqcDatabase = None
    SupcrtDatabase = None
    AqueousPhase = None
    GaseousPhase = None
    MineralPhase = None
    ChemicalSystem = None
    EquilibriumConditions = None
    EquilibriumSolver = None
    ChemicalState = None
    speciate = None
    _REAKTORO_IMPORT_ERROR = _exc
else:
    _REAKTORO_IMPORT_ERROR = None


class ReaktoroFlashError(FlashError):
    """Raised when the Reaktoro equilibrium solver fails or does not converge.

    A :class:`FlashError` subclass so the model's Newton loop catches it the same way as a
    PHREEQC failure and converts it into a timestep cut (keeping the simulation alive),
    rather than silently returning non-physical zeros that would inject NaN/Inf operators.
    """


class Flash:
    """
    Calculates chemical and vapour-liquid equilibrium using Reaktoro.

    This class mirrors the public runtime interface of the PHREEQC-based
    Flash in `phreeqc.py` as closely as possible, but uses Reaktoro
    for equilibrium calculations.

    Supported databases:
    - PHREEQC: phreeqc.dat
    - Supcrtbl: supcrtbl
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
    ):
        """
        :param min_z: minimal composition value
        :type min_z: float
        :param minerals: list of minerals (e.g., ['Solid_CaCO3'])
        :type minerals: list[str]
        :param components: list of elements used as components (e.g., ['Ca','Mg','C','O','H'])
        :type components: list[str]
        :param temperature: temperature [K] for isothermal case
        :type temperature: float | None
        :param gas_species: gas species to include in the GAS_PHASE section
        :type gas_species: list[str] | tuple[str, ...]
        :param tolerance: convergence tolerance for the equilibrium solver
        :type tolerance: float
        :param database_filename: path to database file (PHREEQC or supcrtbl) for primary engine
        :type database_filename: str
        """
        if _REAKTORO_IMPORT_ERROR is not None:  # pragma: no cover
            raise ImportError(
                "Reaktoro is required but could not be imported"
            ) from _REAKTORO_IMPORT_ERROR

        # Store composition/model parameters (keep parity with PHREEQC Flash)
        self.minerals = minerals
        self.components = components
        self.n_fluid = len(self.components)
        self.n_solid = len(self.minerals)

        # Local indexing helpers for element components
        self.fc_idx = {comp: i for i, comp in enumerate(self.components)}

        # Boolean mask for extraction of fluid components from DARTS state vector
        self.f_mask_state = np.concatenate(
            [[False] * (self.n_solid + 1), [True] * (self.n_fluid - 1)]
        )

        # Keep human-friendly mineral names (remove 'Solid_' prefix)
        self.mineral_names = [item.split('_', 1)[1] for item in self.minerals]

        # Thermal handling consistent with PHREEQC Flash
        if temperature is None:
            self.thermal = True
        else:
            self.thermal = False
            # Store Celsius for parity with PHREEQC template (which expects °C)
            self.temperature = temperature - 273.15

        self.min_z = min_z
        self.total_moles = 1000

        # Gas setup (names must match database species names)
        self.gas_species = list(gas_species)

        # Initialize Reaktoro system (PHREEQC database backend)
        self.database_filename = database_filename
        self.tolerance = tolerance
        self._build_reaktoro_system()

    def get_fluid_composition(self, state):
        """
        Extract the fluid composition from the DARTS state vector using the boolean mask.
        Mirrors the logic in the PHREEQC-based Flash.

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
        Calculates chemical and vapour-liquid equilibrium using Reaktoro.
        Returns a tuple with the same structure as PHREEQC Flash.evaluate:
        (
            nu_v, x, y, rho_phases, kin_state, fluid_volume,
            species_aq_molar_fractions, species_gas_molar_fractions
        )
        :param state: state vector
        :type state: np.ndarray
        :return: (nu_v) vapour phase molar fraction, (x) molar composition of aqueous
         and (y) vapour phases, (rho_phases) phase molar densities,
         (kin_state) kinetic params, (fluid_volume) fluid volume,
         (species_aq_molar_fractions) aqueous species molar fractions,
         and (species_gas_molar_fractions) gas species molar fractions
        :rtype: tuple[float, np.ndarray, np.ndarray, dict, dict, float, np.ndarray, np.ndarray]
        """
        # Pressure in bar (DARTS), convert to SI units required by Reaktoro conditions if needed
        pressure_bar = float(state[0])
        temperature_c = float(self.temperature) if not self.thermal else 25.0

        # Component composition (elements) and moles basis
        fluid_composition = self.get_fluid_composition(state)
        fluid_moles = self.total_moles * fluid_composition

        # Build Reaktoro equilibrium conditions with element amounts
        conds = EquilibriumConditions(self.system)
        conds.pressure(pressure_bar, "bar")
        conds.temperature(temperature_c, "celsius")
        # conds.pH(7.0)

        # Set initial element amounts + neutral charge
        e_moles = np.array(
            [fluid_moles[self.fc_idx[el.symbol()]] for el in self.system.elements()]
            + [0.0]
        )
        conds.setInitialComponentAmounts(e_moles)

        state = ChemicalState(self.system)

        init_h_moles, init_o_moles = (
            fluid_moles[self.fc_idx['H']],
            fluid_moles[self.fc_idx['O']],
        )
        if init_h_moles / 2 <= init_o_moles:
            water_moles = init_h_moles / 2
            fluid_moles[self.fc_idx['H']] = 0
            fluid_moles[self.fc_idx['O']] = init_o_moles - init_h_moles / 2
        else:
            water_moles = init_o_moles
            fluid_moles[self.fc_idx['H']] = init_h_moles - 2 * init_o_moles
            fluid_moles[self.fc_idx['O']] = 0
        state.set("H2O" + self.aq_ending, water_moles, "mol")

        solver = EquilibriumSolver(self.system)
        op = EquilibriumOptions()
        op.optima.convergence.tolerance = 1e-12
        solver.setOptions(op)
        try:
            result = solver.solve(state, conds)
            if hasattr(result, "succeeded") and not result.succeeded():
                raise RuntimeError("Reaktoro equilibrium solver failed to converge")
            props = state.props()
        except Exception as exc:
            # Do NOT return zeros: rho_aq=0 would feed divide-by-zero / NaN operators into
            # the OBL table silently. Raise a FlashError so the model's Newton loop treats
            # it as a non-convergence and cuts the timestep (keeping the simulation alive).
            raise ReaktoroFlashError(
                f"Reaktoro equilibrium failed at p={pressure_bar:.6g} bar, "
                f"T={temperature_c:.4g} C: {exc}"
            ) from exc

        volume_m3 = props.volume().val()  # m3
        elem_moles_aq = props.elementAmountsInPhase("AqueousPhase").asarray()
        elem_moles_gas = props.elementAmountsInPhase("GaseousPhase").asarray()
        elem_moles_aq_sum = elem_moles_aq.sum()
        elem_moles_gas_sum = elem_moles_gas.sum()

        nu_v = elem_moles_gas_sum / (elem_moles_aq_sum + elem_moles_gas_sum)
        species_aq_molar_fractions = (
            props.phaseProps("AqueousPhase").speciesMoleFractions().asarray()
        )
        species_gas_molar_fractions = (
            props.phaseProps("GaseousPhase").speciesMoleFractions().asarray()
        )
        rho_phases = {
            'aq': elem_moles_aq_sum
            / props.phaseProps("AqueousPhase").volume().val()
            / 1000.0,
            'gas': elem_moles_gas_sum
            / props.phaseProps("GaseousPhase").volume().val()
            / 1000.0,
        }

        # Build x (aqueous) and y (gas) on element basis (positions after solids)
        nc = self.n_solid + self.n_fluid
        x = np.zeros(nc)
        y = np.zeros(nc)

        if elem_moles_aq_sum > 0:
            moles_aq_elements = props.elementAmountsInPhase("AqueousPhase").asarray()
            total_mole_aq_elements = sum(moles_aq_elements)
            x[self.n_solid :] = np.array(
                [
                    moles_aq_elements[self.system.elements().index(c)]
                    / total_mole_aq_elements
                    for c in self.components
                ]
            )

        if elem_moles_gas_sum > 1.0e-10:
            moles_gas_elements = props.elementAmountsInPhase("GaseousPhase").asarray()
            total_mole_gas_elements = sum(moles_gas_elements)
            y[self.n_solid :] = np.array(
                [
                    moles_gas_elements[self.system.elements().index(c)]
                    / total_mole_gas_elements
                    for c in self.components
                ]
            )

        # Kinetic state: saturation ratios and activities
        aq_props = AqueousProps(state)
        kin_state = {
            'Act(H+)': props.speciesActivity("H+").val(),
            'Act(CO2)': props.speciesActivity("CO2" + self.aq_ending).val(),
            'Act(H2O)': props.speciesActivity("H2O" + self.aq_ending).val(),
            #'pH': aq_props.pH().val(),
        }

        n_saturation_species = aq_props.saturationSpecies().size()
        for m in self.mineral_names:
            id = aq_props.saturationSpecies().findWithFormula(m)
            if id < n_saturation_species:
                kin_state[f"SR_{m}"] = aq_props.saturationRatio(id).val()
            else:
                kin_state[f"SR_{m}"] = 0.0

        return (
            nu_v,
            x,
            y,
            rho_phases,
            kin_state,
            volume_m3,
            species_aq_molar_fractions,
            species_gas_molar_fractions,
        )

    def _build_reaktoro_system(self):
        try:
            if self.database_filename == "supcrtbl":
                # Load SUPCRT database
                self.db = SupcrtDatabase(self.database_filename)
                self.aq_ending = "(aq)"
            else:
                # Load PHREEQC database
                self.db = PhreeqcDatabase(self.database_filename)
                self.aq_ending = ""
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {self.database_filename} with Reaktoro: {exc}"
            ) from exc

        # Define chemical system with aqueous and gaseous phases
        phases = []
        # Aqueous phase with species generated from selected elements
        try:
            elements_str = " ".join(self.components)
            aq = AqueousPhase(speciate(elements_str))
            phases.append(aq)
        except Exception as exc:
            raise RuntimeError(f"Failed to create AqueousPhase: {exc}") from exc

        # Gaseous phase with selected gas species
        try:
            # Accept both list and space-separated string depending on API
            try:
                gas = GaseousPhase(self.gas_species)
            except Exception:
                gas = GaseousPhase(" ".join(self.gas_species))
            phases.append(gas)
        except Exception as exc:
            raise RuntimeError(f"Failed to create GaseousPhase: {exc}") from exc

        # define equilibrium system
        try:
            self.system = ChemicalSystem(self.db, *phases)
        except Exception as exc:
            raise RuntimeError(f"Failed to build ChemicalSystem: {exc}") from exc

        # get aqueous and sort them for consistent output
        phase_idx = self.system.phases().find("AqueousPhase")
        self.aqueous_species = [
            sp.name() for sp in self.system.phases()[phase_idx].species()
        ]
        phase_idx = self.system.phases().find("GaseousPhase")
        self.gas_species = [
            sp.name() for sp in self.system.phases()[phase_idx].species()
        ]
