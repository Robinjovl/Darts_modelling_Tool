import numpy as np
from scipy.optimize import root_scalar

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
        p_init=1e-5,
        sv_init=0.535,
        fixed_pressure=False,
        fixed_temperature=False,
        energy_source=1e5,
        poro=0.3,
        dens_rock=1100.0,
        c_r=0.920,
    ):
        """
        Initialize the model.
        :param mode: str, 'analytical' or 'obl'
        :type mode: str
        :param p_init: float, initial pressure, in bar
        :type p_init: float
        :param sv_init: float, initial vapour saturation
        :type sv_init: float
        :param fixed_pressure: flag to fix pressure or not
        :type fixed_pressure: bool
        :param fixed_temperature: flag to fix temperature or not
        :type fixed_temperature: bool
        :param energy_source: float, energy source, in kJ/day/m3
        :type energy_source: float
        :param poro: float, porosity of the reservoir
        :type poro: float
        :param dens_rock: float, density of the rock, in kg/m3
        :type dens_rock: float
        :param c_r: float, heat capacity of the rock, in kJ/kg/K
        :type c_r: float
        """
        super().__init__(fixed_pressure=fixed_pressure, fixed_temperature=fixed_temperature)

        self.timer.node["initialization"].start()
        self.mode = mode
        self.p_init = p_init
        self.sv_init = sv_init
        self.energy_source = EnergySource(energy_source)
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
        self.temp_min = 180.0
        self.temp_max = 650.0
        flash_ph.init_flash(
            flash_type=DARTSFlash.FlashType.PHFlash,
            eos_order=phases,
            f_tol=1e-8,
            t_tol=1e-1,
            t_min=self.temp_min,
            t_max=self.temp_max,
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
        phases = self.property_container.phases_name
        try:
            steam_idx = phases.index("steam")
        except ValueError as exc:
            raise RuntimeError("Expected 'steam' in phase list.") from exc

        if not 0.0 <= self.sv_init <= 1.0:
            raise ValueError("sv_init must be in [0, 1].")

        t_min = getattr(self, "temp_min", 180.0)
        t_max = getattr(self, "temp_max", 650.0)

        def saturation_error(enthalpy):
            state = np.array([self.p_init, enthalpy], dtype=float)
            self.property_container.evaluate(state)
            return float(self.property_container.sat[steam_idx] - self.sv_init)

        # Bracket enthalpy using pure-phase enthalpies at temperature bounds
        h_low = h_ice_ev.evaluate(pressure=self.p_init, temperature=t_min, x=[1.0])
        h_high = h_steam_ev.evaluate(pressure=self.p_init, temperature=t_max, x=[1.0])

        if h_low > h_high:
            h_low, h_high = h_high, h_low

        f_low = saturation_error(h_low)
        f_high = saturation_error(h_high)

        if np.isclose(f_low, 0.0, atol=1e-10):
            self.h_init = float(h_low)
        elif np.isclose(f_high, 0.0, atol=1e-10):
            self.h_init = float(h_high)
        else:
            if f_low * f_high > 0:
                bracket = None
                grid = np.linspace(h_low, h_high, 25)
                h_prev, f_prev = h_low, f_low
                for h_val in grid[1:]:
                    f_val = saturation_error(h_val)
                    if f_prev * f_val <= 0:
                        bracket = (h_prev, h_val)
                        break
                    h_prev, f_prev = h_val, f_val
                if bracket is None:
                    raise RuntimeError(
                        f"Failed to bracket enthalpy for sv_init={self.sv_init} "
                        f"at p_init={self.p_init}."
                    )
                h_low, h_high = bracket

            sol = root_scalar(
                saturation_error,
                bracket=(h_low, h_high),
                method="brentq",
                xtol=1e-10,
                rtol=1e-8,
                maxiter=200,
            )
            if not sol.converged:
                raise RuntimeError(
                    f"Enthalpy solve did not converge for p_init={self.p_init}, "
                    f"sv_init={self.sv_init}."
                )
            self.h_init = float(sol.root)

        # Evaluate properties at solved enthalpy
        self.property_container.evaluate(np.array([self.p_init, self.h_init], dtype=float))
        self.temp = float(self.property_container.temperature)

        sv_effective = float(self.property_container.sat[steam_idx])
        h_ice = h_ice_ev.evaluate(pressure=self.p_init, temperature=self.temp, x=[1.0])
        rho_ice = rho_ice_ev.evaluate(pressure=self.p_init, temperature=self.temp, x=[1.0])
        h_steam = h_steam_ev.evaluate(pressure=self.p_init, temperature=self.temp, x=[1.0])
        rho_steam = rho_steam_ev.evaluate(pressure=self.p_init, temperature=self.temp, x=[1.0])

        rho_regolith = 3100.  # density=3100 kg/m3
        poro = self.poro

        # Mass fractions for the mixed phase
        mass_fraction_ice = poro * (1 - sv_effective) * rho_ice / \
                            (poro * ((1 - sv_effective) * rho_ice + sv_effective * rho_steam) + (1 - poro) * rho_regolith)

        self.initial_state = np.array([self.p_init, self.h_init])

        print(f"Setting initial conditions for given p={self.p_init}, sv={sv_effective}")
        print(f"Target sv={self.sv_init}, solved h_total={self.h_init} temp={self.temp}")
        print(f"wt_ice={mass_fraction_ice}")
        print(f"Calculated h_ice: {h_ice}, h_steam: {h_steam}")
        print(f"Ice density = {rho_ice}, Steam density = {rho_steam}")
