import numpy as np

from darts.models.zerod_model import ZerodModel
from darts.physics.chemistry.physics import ElementBasedReactiveFlow
from darts.physics.chemistry.property_container import (
    OutputPropertyContainer,
    PropertyContainer,
)
from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.kinetics import KineticRate, LinearReactionSurfaceArea
from darts.physics.properties.reaktoro import Flash as ReaktoroFlash
from reaktoro import (
    ActivityModelPengRobinsonPhreeqcOriginal,
    ActivityModelPitzer,
    AqueousPhase,
    ChemicalSystem,
    GaseousPhase,
    SupcrtDatabase,
    speciate,
)


class Model(ZerodModel):
    """
    0D reaktoro kinetics model driven by OBL operators.

    State variables follow ElementBasedReactiveFlow ordering:
    [p, Solid_CaCO3, Solid_CaMg(CO3)2, Solid_MgCO3, Ca, Mg, C, O].
    """

    def __init__(
        self,
        n_obl_mult: int = 1,
        runtime: float = 1500.0,
        first_ts: float = 1e-5,
        max_ts: float = 10.0,
        fixed_pressure: bool = True,
        poro: float = 0.2,
    ):
        super().__init__(fixed_pressure=fixed_pressure, fixed_temperature=True)

        self.n_obl_mult = int(n_obl_mult)
        self.mode = "obl"

        # These are used by ZerodModel when building dA/dx.
        self.poro = poro
        self.dens_rock = 0.0
        self.c_r = 0.0

        self.timer.node["initialization"].start()
        self.set_physics()
        self.set_sim_params(
            n_vars=self.physics.n_vars,
            first_ts=first_ts,
            mult_ts=2.0,
            max_ts=max_ts,
            runtime=runtime,
        )
        self.data_ts.eta = np.array([1e20] + [1e-5] * 7)
        self.timer.node["initialization"].stop()

    def set_physics(self):
        self.min_z = 1e-15

        # Ambient conditions
        self.temperature = 298.15  # K
        self.pressure_init = 1.0  # bar

        # Initial inventory
        h2o_mass = 1.0  # kg
        h2o_mole = h2o_mass / 0.018016  # mol
        co2_mole = 5.0  # mol
        calcite_mole = 1.0  # mol
        dolomite_mole = 0.0  # mol
        magnesite_mole = 1.0  # mol
        o2_mole = 1.0e-6  # mol

        # Initial elemental molar fractions
        fluid_moles = 3.0 * h2o_mole + 3.0 * co2_mole + 2.0 * o2_mole
        self.zCa = max(self.min_z, 0.0 / fluid_moles)
        self.zMg = max(self.min_z, 0.0 / fluid_moles)
        self.zC = max(self.min_z, co2_mole / fluid_moles)
        self.zO = max(self.min_z, (h2o_mole + 2.0 * (co2_mole + o2_mole)) / fluid_moles)
        self.zH = max(self.min_z, 2.0 * h2o_mole / fluid_moles)

        solid_moles = calcite_mole + dolomite_mole + magnesite_mole
        total_moles = solid_moles + fluid_moles
        self.zCalcite = max(self.min_z, calcite_mole / total_moles)
        self.zDolomite = max(self.min_z, dolomite_mole / total_moles)
        self.zMagnesite = max(self.min_z, magnesite_mole / total_moles)

        self.obl_min = self.min_z / 10.0
        gas = "gas"
        liq = "liq"
        self.phases = {gas: 0, liq: 1}
        phase_name = [
            list(self.phases.keys())[list(self.phases.values()).index(i)]
            for i in range(len(self.phases))
        ]

        self.minerals = ["calcite", "dolomite", "magnesite"]
        self.n_solid = len(self.minerals)

        self.elements = [
            "Solid_CaCO3",
            "Solid_CaMg(CO3)2",
            "Solid_MgCO3",
            "Ca",
            "Mg",
            "C",
            "O",
            "H",
        ]
        mw = {
            "Solid_CaCO3": 100.0869,
            "Solid_CaMg(CO3)2": 184.401,
            "Solid_MgCO3": 84.31,
            "Ca": 40.078,
            "Mg": 24.305,
            "C": 12.0096,
            "O": 15.999,
            "H": 1.007,
        }

        self.n_points = list(
            self.n_obl_mult
            * np.array([11, 5001, 5001, 5001, 501, 501, 501, 501], dtype=np.intp)
        )
        self.fc_mask = np.array(
            [False, False, False, True, True, True, True, True], dtype=bool
        )
        self.axes_min = [self.pressure_init - 0.01] + [
            self.obl_min,
            self.obl_min,
            self.obl_min,
            self.obl_min,
            self.obl_min,
            self.obl_min,
            0.35,
        ]
        self.axes_max = [self.pressure_init + 0.01] + [
            0.01,
            0.01,
            0.01,
            0.0001,
            0.0001,
            0.05,
            0.37,
        ]

        stoich_matrix = np.array(
            [
                [-1, 0, 0, 1, 0, 1, 3, 0],
                [0, -1, 0, 1, 1, 2, 6, 0],
                [0, 0, -1, 0, 1, 1, 3, 0],
            ]
        )
        rock_props = {
            "Solid_CaCO3": {"density": 2710.0, "compressibility": 1.0e-6},
            "Solid_CaMg(CO3)2": {"density": 2840.0, "compressibility": 1.0e-6},
            "Solid_MgCO3": {"density": 2958.0, "compressibility": 1.0e-6},
        }

        kinetic_mechanisms = ["acidic", "neutral", "carbonate"]
        property_container = PropertyContainer(
            phases=self.phases,
            components_name=self.elements,
            Mw=mw,
            stoich_matrix=stoich_matrix,
            eps_z=self.obl_min,
            temperature=self.temperature,
            fc_mask=self.fc_mask,
        )

        property_container.permporo_mult_ev = ConstFunc(1.0)
        property_container.diffusion_ev = {
            ph: ConstFunc(
                np.concatenate([np.zeros(self.n_solid), np.ones(property_container.nc - self.n_solid)])
                * 5.2e-10
                * 86400.0
            )
            for ph in self.phases
        }
        property_container.rel_perm_ev = {ph: CustomRelPerm(2) for ph in self.phases}
        property_container.viscosity_ev = {gas: GasViscosity(), liq: LiquidViscosity()}

        property_container.flash_ev = Flash(
            min_z=property_container.eps_z,
            minerals=property_container.minerals,
            components=property_container.components_name[property_container.fc_mask],
            temperature=property_container.temperature,
            database_filename="supcrtbl",
            mineral_saturation_names={
                "CaCO3": "Calcite",
                "CaMg(CO3)2": "Dolomite",
                "MgCO3": "Magnesite",
            },
        )

        for mineral, props in rock_props.items():
            property_container.rock_compr_ev[mineral] = ConstFunc(props["compressibility"])
            property_container.rock_density_ev[mineral] = DensityBasic(
                compr=props["compressibility"], dens0=props["density"], p0=1.0
            )

        # Scale kinetic surface area from 6 cm2/cm3 to m2/mol using mineral molar density.
        area_per_volume = 6.0 * 100.0
        for mineral in property_container.minerals:
            rho_m = (
                property_container.rock_density_ev[mineral].evaluate(
                    self.pressure_init, self.temperature
                )
                / property_container.Mw[mineral]
            )
            area_per_mol = area_per_volume / rho_m / 1000.0
            surface_area_ev = LinearReactionSurfaceArea(
                initial_area_per_mol=area_per_mol
            )
            property_container.kinetic_rate_ev[mineral] = KineticRate(
                min_z=self.obl_min,
                mineral_name=mineral.split("_", 1)[1],
                mechanisms=kinetic_mechanisms,
                surface_area_ev=surface_area_ev,
            )

        output_property_container = MyOutputPropertyContainer(property_container)

        self.physics = ElementBasedReactiveFlow(
            timer=self.timer,
            elements=self.elements,
            phases=phase_name,
            n_points=self.n_points,
            axes_min=self.axes_min,
            axes_max=self.axes_max,
            epsilon_z=property_container.eps_z,
            extrapolation_flag=False,
            cache=False,
        )
        self.physics.add_property_region(property_container, output_property_container, 0)
        self.property_container = property_container

    def set_initial_conditions(self):
        self.initial_state = np.array(
            [
                self.pressure_init,
                self.zCalcite,
                self.zDolomite,
                self.zMagnesite,
                self.zCa,
                self.zMg,
                self.zC,
                self.zO,
            ],
            dtype=float,
        )

    def extract_property_history(self, prop_names=None):
        if not self.state_history:
            return np.array([]), {}

        output_container = self.physics.output_property_containers[0]
        names = (
            list(output_container.output_props.keys())
            if prop_names is None
            else list(prop_names)
        )
        data = {name: np.zeros(len(self.state_history), dtype=float) for name in names}

        for i, state in enumerate(self.state_history):
            output_container.evaluate(np.asarray(state, dtype=float))
            for name in names:
                data[name][i] = float(output_container.output_props[name]())

        return np.asarray(self.time_history, dtype=float), data


class CustomRelPerm:
    def __init__(self, exp, sr=0.0):
        self.exp = exp
        self.sr = sr

    def evaluate(self, sat):
        del sat
        return 0.0


class GasViscosity:
    def evaluate(self, pressure, temperature):
        del pressure, temperature
        return 0.0278


class LiquidViscosity:
    def evaluate(self, density, temperature):
        del density, temperature
        return 1.0


class Flash(ReaktoroFlash):
    def _build_reaktoro_system(self):
        try:
            self.db = SupcrtDatabase(self.database_filename)
            self.aq_ending = "(aq)"
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {self.database_filename} with Reaktoro: {exc}"
            ) from exc

        try:
            elements_str = " ".join(self.components)
            aq = AqueousPhase(speciate(elements_str))
            gas = GaseousPhase("CO2(g)")
            self.system = ChemicalSystem(
                self.db,
                aq.set(ActivityModelPitzer()),
                gas.set(ActivityModelPengRobinsonPhreeqcOriginal()),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to build ChemicalSystem: {exc}") from exc

        phase_idx = self.system.phases().find("AqueousPhase")
        self.aqueous_species = [
            sp.name() for sp in self.system.phases()[phase_idx].species()
        ]
        phase_idx = self.system.phases().find("GaseousPhase")
        self.gas_species = [sp.name() for sp in self.system.phases()[phase_idx].species()]


class MyOutputPropertyContainer(OutputPropertyContainer):
    def __init__(self, property_container, props_name: list[str] | None = None):
        super().__init__(property_container, props_name)

        self.dens_m = np.zeros(2)
        self.sat = np.zeros(2)
        self.sat_minerals = np.zeros(len(self.property.flash_ev.mineral_names))

        for i, ph in enumerate(self.property.phases_name):
            self.output_props["dens_m_" + ph] = lambda i=i: self.dens_m[i]
            self.output_props["sat_" + ph] = lambda i=i: self.sat[i]

        for i, mineral in enumerate(self.property.flash_ev.mineral_names):
            self.output_props["sat_" + mineral] = lambda i=i: self.sat_minerals[i]

    def evaluate(self, state):
        super().evaluate(state)
        self.property.evaluate(state)

        self.dens_m = self.property.dens_m
        self.sat = self.property.sat
        self.sat_minerals = self.property.sat_minerals
