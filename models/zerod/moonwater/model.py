import numpy as np

from darts.engines import value_vector
from darts.models.zerod_model import ZerodModel
from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from property_container import PropertyContainerDerivatives
from darts.reservoirs.struct_reservoir import StructReservoir
from dartsflash.dartsflash import DARTSFlash, CompData
from dartsflash.libflash import EoS, IdealGas, PureSolid, CubicEoS, StateSpecification

class EnergySource:
    def __init__(self, value=0.0):
        """
        Initialize a constant energy source.

        :param value: Energy source value.
        :type value: float
        """
        self.value = float(value)

    def evaluate(self, t):
        """
        Evaluate the energy source at a given time.

        :param t: Simulation time.
        :type t: float
        :return: Energy source value.
        :rtype: float
        """
        return self.value


class Model(ZerodModel):
    def __init__(
        self,
        mode='analytical',
        p_init=1e-3,
        t_init=230.0,
        sv_init=0.535,
        fixed_pressure=False,
        fixed_temperature=False,
        energy_source=-2e3,
        volume=1.0,
        poro=0.3,
        dens_rock=3100.0,
        c_r=0.920,
    ):
        """
        Initialize the model.
        :param mode: str, 'analytical' or 'obl'
        :type mode: str
        :param p_init: float, initial pressure, in bar
        :type p_init: float
        :param t_init: float, initial temperature, in Kelvin
        :type t_init: float
        :param sv_init: float, initial vapour saturation
        :type sv_init: float
        :param fixed_pressure: flag to fix pressure or not
        :type fixed_pressure: bool
        :param fixed_temperature: flag to fix temperature or not
        :type fixed_temperature: bool
        :param energy_source: float, energy source, in kJ/day
        :type energy_source: float
        :param volume: float, volume of the reservoir, in m^3
        :type volume: float
        :param poro: float, porosity of the reservoir
        :type poro: float
        :param dens_rock: float, density of the rock, in kg/m^3
        :type dens_rock: float
        :param c_r: float, heat capacity of the rock, in kJ/kg/K
        :type c_r: float
        """
        super().__init__(fixed_pressure=fixed_pressure, fixed_temperature=fixed_temperature)

        self.timer.node["initialization"].start()
        self.mode = mode
        self.p_init = p_init
        self.t_init = t_init
        self.sv_init = sv_init
        self.energy_source = EnergySource(energy_source)
        self.volume = volume
        self.poro = poro
        self.c_r = c_r
        self.dens_rock = dens_rock

        self.set_physics()

        self.set_sim_params(n_vars=self.property_container.n_vars, first_ts=1e-4, mult_ts=2, max_ts=1e-3)
        self.timer.node["initialization"].stop()

    def set_physics(self):
        """
        Set up flash, property container, and evaluators.
        """
        components = ["H2O"]
        phases = ["ice", "steam"]
        zero = 1e-12

        # flash
        comp_data = CompData(components=components, setprops=True)
        ice_eos = PureSolid(comp_data, "Ice")
        steam_eos = IdealGas(comp_data)

        flash_ph = DARTSFlash(comp_data=comp_data)
        flash_ph.add_eos("ice", ice_eos)
        flash_ph.add_eos("steam", steam_eos)
        flash_ph.init_flash(
            flash_type=DARTSFlash.FlashType.PHFlash,
            eos_order=phases,
            f_tol=1e-8,
            t_tol=1e-1,
            t_min=180.0,
            t_max=650.0,
        )
        self.flash_ph = flash_ph

        # property container
        if self.mode == "analytical":
            property_container = PropertyContainerDerivatives(
                phases_name=phases,
                components_name=components,
                Mw=comp_data.Mw,
                eps_z=zero / 10,
                state_spec=StateSpecification.ENTHALPY,
            )
        else:
            property_container = PropertyContainer(
                phases_name=phases,
                components_name=components,
                Mw=comp_data.Mw,
                min_z=zero,
            )
        property_container.flash_ev = flash_ph
        property_container.density_ev = {
            "ice": EoSDensity(eos=ice_eos, Mw=comp_data.Mw),
            "steam": EoSDensity(eos=steam_eos, Mw=comp_data.Mw, root_flag=EoS.RootFlag.MAX),
        }
        property_container.enthalpy_ev = {
            "ice": EoSEnthalpy(eos=ice_eos),
            "steam": EoSEnthalpy(eos=steam_eos, root_flag=EoS.RootFlag.MAX),
        }
        property_container.viscosity_ev = {
            "ice": ConstFunc(1.0),
            "steam": ConstFunc(0.01),
        }
        property_container.rel_perm_ev = {
            "ice": ConstFunc(1.0),
            "steam": ConstFunc(1.0),
        }
        property_container.conductivity_ev = {
            "ice": ConstFunc(2.0),
            "steam": ConstFunc(0.1),
        }
        property_container.energy_source_ev = self.energy_source

        if self.mode == "obl":
            # self.physics = Compositional(
            #     components,
            #     phases,
            #     self.timer,
            #     state_spec=state_spec,
            #     n_points=self.n_points,
            #     min_p=self.p_min,
            #     max_p=self.p_max,
            #     min_z=zero / 10,
            #     max_z=1.0 - zero / 10,
            #     min_t=min_t,
            #     max_t=max_t,
            # )
            # self.physics.add_property_region(property_container)
            pass
        self.property_container = property_container

    def set_initial_conditions(self):
        """
        Initialize the model state from input parameters.
        """
        # Get enthalpy evaluators
        h_ice_ev = self.property_container.enthalpy_ev['ice']
        h_steam_ev = self.property_container.enthalpy_ev['steam']
        rho_ice_ev = self.property_container.density_ev['ice']
        rho_steam_ev = self.property_container.density_ev['steam']

        h_ice = h_ice_ev.evaluate(pressure=self.p_init, temperature=self.t_init, x=[1.0])
        rho_ice = rho_ice_ev.evaluate(pressure=self.p_init, temperature=self.t_init, x=[1.0])
        h_steam = h_steam_ev.evaluate(pressure=self.p_init, temperature=self.t_init, x=[1.0])
        rho_steam = rho_steam_ev.evaluate(pressure=self.p_init, temperature=self.t_init, x=[1.0])

        rho_regolith = 3100.  # density=3100 kg/m3
        poro = self.poro

        # Mass fractions for the mixed phase (5.6% ice, 94.4% steam)
        mass_fraction_ice = poro * (1 - self.sv_init) * rho_ice / \
                            (poro * ((1 - self.sv_init) * rho_ice + self.sv_init * rho_steam) + (1 - poro) * rho_regolith)
        molar_fraction_ice = rho_ice * (1 - self.sv_init) / (rho_ice * (1 - self.sv_init) + self.sv_init * rho_steam)

        # Calculate enthalpy for ice and steam
        self.h_init = molar_fraction_ice * h_ice + (1 - molar_fraction_ice) * h_steam
        self.initial_state = np.array([self.p_init, self.h_init])

        print(f"Setting intial conditions for p={self.p_init}, t={self.t_init}, sv={self.sv_init}")
        print(f"wt_ice={mass_fraction_ice}")
        print(f"Calculated h_ice: {h_ice}, h_steam: {h_steam}, h_total: {self.h_init}")
        print(f"Ice density = {rho_ice}, Steam density = {rho_steam}")
