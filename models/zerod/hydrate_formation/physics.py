import numpy as np

from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.eos_properties import (
    EoSDensity,
    EoSEnthalpy,
    VdWPDensity,
    VdWPEnthalpy,
)
from darts.physics.properties.flash import SolidFlash
from darts.physics.properties.kinetics import Kinetics
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from dartsflash.components import CompData
from dartsflash.dartsflash import DARTSFlash
from dartsflash.libflash import AQEoS, Ballard, CubicEoS, EoS


class ZeroCapillaryPressure:
    def __init__(self, nph: int):
        self.nph = int(nph)

    def evaluate(self, sat):
        del sat
        return np.zeros(self.nph, dtype=float)


class HydrateKineticsPeter(Kinetics):
    """
    CO2-hydrate kinetics used in the Li et al. publication case.
    """

    def __init__(
        self,
        components: list[str],
        phases: list[str],
        mw,
        hydrate_eos,
        fluid_eos: list,
        stoich: list[float],
        saturations,
        perm: float,
        poro: float,
        k: float,
    ):
        super().__init__(stoich)

        self.hydrate_eos = hydrate_eos
        self.fluid_eos = fluid_eos
        self.water_idx = components.index("H2O")
        self.guest_idx = 0 if self.water_idx == 1 else 1
        self.aq_idx = phases.index("Aq")
        self.vap_idx = phases.index("V")
        self.hyd_idx = phases.index("sI")
        self.mw = mw
        self.k = float(k)
        self.poro = float(poro)
        self.perm = float(perm) * 1e-15
        self.saturations = saturations

        r_p = 1.645e-4
        beta = 5.3
        self.area = lambda sat: (
            (1.0 - self.poro)
            / r_p
            * sat[self.vap_idx] ** (2.0 / 3.0)
            * sat[self.aq_idx] ** beta
            * (1.0 - sat[self.hyd_idx]) ** beta
        )

    def calc_df(self, pressure, temperature, x):
        j = 1 if np.isnan(x[0, 0]) or x[0, 0] == 0.0 else 0
        f0 = self.fluid_eos[j].fugacity(pressure, temperature, x[j, :])
        fw_h = self.hydrate_eos.fw(pressure, temperature, f0)
        return fw_h - f0[self.water_idx]

    def evaluate(self, pressure, temperature, x, sat_hyd):
        del sat_hyd
        df = self.calc_df(pressure, temperature, x)
        d_e = -1.0288e5
        gas_constant = 8.3145
        rate = (
            3.0
            * self.k
            * self.area(self.saturations)
            * np.exp(d_e / (gas_constant * temperature))
            * df
        )
        return [stoich * rate for stoich in self.stoich], df


class HydrateKineticsMoridis(Kinetics):
    """
    Methane-hydrate kinetics used in the Moridis/Yin publication case.
    """

    def __init__(
        self,
        components: list[str],
        phases: list[str],
        hydrate_eos,
        fluid_eos: list,
        stoich: list[float],
        perm: float,
        poro: float,
        k: float,
        f_a: float = 0.23,
    ):
        super().__init__(stoich)

        self.hydrate_eos = hydrate_eos
        self.fluid_eos = fluid_eos
        self.water_idx = components.index("H2O")
        self.guest_idx = 0 if self.water_idx == 1 else 1
        self.k = float(k)
        self.f_a = float(f_a)
        self.poro = float(poro)
        self.perm = float(perm) * 1e-15

        r_p = np.sqrt(45.0 * self.perm * (1.0 - self.poro) ** 2 / self.poro**3)
        self.area = (
            lambda sat_hyd: 0.879
            * self.f_a
            * (1.0 - self.poro)
            / r_p
            * sat_hyd ** (2.0 / 3.0)
        )

    def calc_df(self, pressure, temperature, x):
        j = 1 if np.isnan(x[0, 0]) or x[0, 0] == 0.0 else 0
        f0 = self.fluid_eos[j].fugacity(pressure, temperature, x[j, :])
        fw_h = self.hydrate_eos.fw(pressure, temperature, f0)
        return fw_h - f0[self.water_idx]

    def evaluate(self, pressure, temperature, x, sat_hyd):
        df = self.calc_df(pressure, temperature, x)
        d_e = -81e3
        gas_constant = 8.3145
        rate = (
            self.k
            * self.area(sat_hyd)
            * np.exp(d_e / (gas_constant * temperature))
            * df
        )
        return [stoich * rate for stoich in self.stoich], df


class HydrateBatchPhysics(Compositional):
    """
    Minimal hydrate PT-physics for 0D kinetics-driven runs.

    This keeps the flash, density, enthalpy, and kinetics setup from the
    publication model while using simple transport correlations because the 0D
    reduction only needs accumulation operators and kinetic source terms.
    """

    def __init__(
        self,
        timer,
        guest_component: str,
        case_name: str,
        n_points: int = 101,
        min_p: float = 10.0,
        max_p: float = 300.0,
        min_t: float = 250.0,
        max_t: float = 333.15,
        cache: bool = False,
        extrapolation_flag: bool = True,
        epsilon: float = 1e-13,
    ):
        self.case_name = str(case_name)
        self.guest_component = str(guest_component)

        components = ["H2O", self.guest_component]
        phases = ["Aq", "V", "sI"]

        comp_data = CompData(components, setprops=True)
        h2o_idx = components.index("H2O")

        flash_ev = DARTSFlash(comp_data=comp_data)
        cubic_kind = CubicEoS.PR if self.case_name == "ch4" else CubicEoS.SRK
        ceos = CubicEoS(comp_data, cubic_kind, volume_shift=False)
        aq = AQEoS(
            comp_data,
            {
                AQEoS.CompType.water: AQEoS.Jager2003,
                AQEoS.CompType.solute: AQEoS.Ziabakhsh2012,
                AQEoS.CompType.ion: AQEoS.Jager2003,
            },
        )

        flash_ev.add_eos(
            "CEOS",
            ceos,
            trial_comps=[i for i in range(comp_data.nc)],
            root_order=[EoS.STABLE],
            preferred_roots=[(h2o_idx, 0.75, EoS.MAX)],
            stability_tol=1e-20,
            switch_tol=1e-2,
            max_iter=50,
            use_gmix=False,
        )
        flash_ev.add_eos(
            "AQ",
            aq,
            trial_comps=[h2o_idx],
            eos_range={h2o_idx: [0.6, 1.0]},
            max_iter=10,
            use_gmix=True,
        )

        self.hydrate_eos = Ballard(comp_data, "sI")

        n_h = 6.0 if self.guest_component == "CO2" else 6.1
        self.stoich = [-n_h, -1.0, 1.0]
        mw = np.append(
            comp_data.Mw,
            np.abs(self.stoich[0] * comp_data.Mw[0] + self.stoich[1] * comp_data.Mw[1]),
        )
        x_h = np.array([n_h, 1.0], dtype=float) / (n_h + 1.0)

        flash_ev.init_flash(
            flash_type=DARTSFlash.FlashType.PTFlash,
            eos_order=["AQ", "CEOS"],
            split_switch_tol=1e1,
            split_negative_flash_iter=10,
            t_min=min_t,
            t_max=max_t,
        )

        species = components + ["H"]
        super().__init__(
            components=species,
            phases=phases,
            timer=timer,
            n_points=n_points,
            min_p=min_p,
            max_p=max_p,
            min_z=0.0,
            max_z=1.0,
            epsilon_z=epsilon,
            min_t=min_t,
            max_t=max_t,
            state_spec=Compositional.StateSpecification.PT,
            cache=cache,
            extrapolation_flag=extrapolation_flag,
        )

        property_container = PropertyContainer(
            phases_name=phases,
            components_name=species,
            Mw=mw,
            temperature=None,
            eps_z=epsilon,
            nc_sol=0,
            np_sol=0,
        )
        property_container.flash_ev = SolidFlash(
            flash_ev, nc_fl=len(components), np_fl=2, nc_sol=1, np_sol=1
        )
        property_container.density_ev = {
            "V": EoSDensity(ceos, mw),
            "Aq": Garcia2001(components, None),
            "sI": VdWPDensity(self.hydrate_eos, comp_data.Mw, x_h),
        }
        property_container.viscosity_ev = {
            "V": ConstFunc(0.02),
            "Aq": ConstFunc(1.0),
            "sI": ConstFunc(1.0),
        }
        property_container.rel_perm_ev = {
            "V": ConstFunc(1.0),
            "Aq": ConstFunc(1.0),
            "sI": ConstFunc(0.0),
        }
        property_container.enthalpy_ev = {
            "V": EoSEnthalpy(ceos),
            "Aq": EoSEnthalpy(ceos),
            "sI": VdWPEnthalpy(self.hydrate_eos, x_h),
        }
        property_container.conductivity_ev = {
            "V": ConstFunc(0.016 * 86.4),
            "Aq": ConstFunc(0.6 * 86.4),
            "sI": ConstFunc(0.6 * 86.4),
        }
        property_container.capillary_pressure_ev = ZeroCapillaryPressure(len(phases))
        property_container.output_props = {"sat_hydrate": lambda: property_container.sat[2]}

        self.add_property_region(property_container, 0)
        self.property_container = property_container
        self.fluid_eos = [aq, ceos]

    def set_kinetic_ev(self, k: float, poro: float, perm: float, region: int = None):
        regions = [region] if region is not None else self.regions
        for i in regions:
            pc = self.property_containers[i]
            if self.guest_component == "CO2":
                pc.kinetic_rate_ev[0] = HydrateKineticsPeter(
                    components=pc.components_name,
                    phases=pc.phases_name,
                    mw=pc.Mw,
                    hydrate_eos=self.hydrate_eos,
                    fluid_eos=self.fluid_eos,
                    stoich=self.stoich,
                    saturations=pc.sat,
                    perm=perm,
                    poro=poro,
                    k=k,
                )
            else:
                pc.kinetic_rate_ev[0] = HydrateKineticsMoridis(
                    components=pc.components_name,
                    phases=pc.phases_name,
                    hydrate_eos=self.hydrate_eos,
                    fluid_eos=self.fluid_eos,
                    stoich=self.stoich,
                    perm=perm,
                    poro=poro,
                    k=k,
                    f_a=0.23,
                )
