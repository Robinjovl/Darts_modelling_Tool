import abc
import numpy as np
from darts.engines import value_vector

from darts.input.input_data import InputData, FluidProps
from darts.physics.geothermal.physics import Geothermal
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm

from darts.physics.properties.iapws.iapws_property import *
from darts.physics.properties.iapws.custom_rock_property import *
from darts.physics.base.property_base import PropertyBase


class GeothermalIAPWS(Geothermal):
    def __init__(self, idata: InputData, timer):
        super().__init__(timer, idata.obl.n_points, idata.obl.min_p, idata.obl.max_p,
                         idata.obl.min_e, idata.obl.max_e)

        property_container = GeothermalIAPWSProperties()

        property_container.Mw = [18.015]

        property_container.rock = [value_vector([idata.rock.compressibility_ref_p, idata.rock.compressibility, idata.rock.compressibility_ref_T])]
        property_container.rock_compaction_ev = custom_rock_compaction_evaluator(property_container.rock)
        property_container.rock_energy_ev = custom_rock_energy_evaluator(property_container.rock)  # Create rock_energy object

        property_container.temperature_ev = idata.fluid.temperature_ev
        property_container.density_ev = idata.fluid.density_ev
        property_container.viscosity_ev = idata.fluid.viscosity_ev
        property_container.relperm_ev = idata.fluid.relperm_ev
        property_container.enthalpy_ev = idata.fluid.enthalpy_ev
        property_container.saturation_ev = idata.fluid.saturation_ev
        property_container.conduction_ev = idata.fluid.conduction_ev

        self.add_property_region(property_container)


class GeothermalPH(Geothermal):
    def __init__(self, idata: InputData, timer):
        # Call base class constructor
        super().__init__(timer, idata.obl.n_points, idata.obl.min_p, idata.obl.max_p,
                         idata.obl.min_e, idata.obl.max_e)
        self.idata = idata
        property_container = GeothermalPHProperties()

        property_container.flash_ev = idata.fluid.flash_ev
        property_container.Mw = idata.fluid.Mw

        property_container.rock = [value_vector([idata.rock.compressibility_ref_p, idata.rock.compressibility, idata.rock.compressibility_ref_T])]
        property_container.rock_compaction_ev = custom_rock_compaction_evaluator(property_container.rock)
        property_container.rock_energy_ev = custom_rock_energy_evaluator(property_container.rock)  # Create rock_energy object

        property_container.density_ev = idata.fluid.density_ev
        property_container.viscosity_ev = idata.fluid.viscosity_ev
        property_container.relperm_ev = idata.fluid.relperm_ev
        property_container.enthalpy_ev = idata.fluid.enthalpy_ev
        property_container.conduction_ev = idata.fluid.conduction_ev

        self.add_property_region(property_container)


class GeothermalPropertiesBase(PropertyBase):
    nc = 1
    nph = 2

    def __init__(self):
        self.Mw = np.zeros(self.nc)
        self.nu = np.zeros(self.nph)
        self.x = np.zeros((self.nph, self.nc))
        self.density = np.zeros(self.nph)
        self.dens_m = np.zeros(self.nph)
        self.saturation = np.zeros(self.nph)
        self.viscosity = np.zeros(self.nph)
        self.relperm = np.zeros(self.nph)
        self.pc = np.zeros(self.nph)
        self.enthalpy = np.zeros(self.nph)
        self.conduction = np.zeros(self.nph)
        self.dX = []
        self.mass_source = np.zeros(self.nc)
        self.energy_source = 0.
        self.temperature = 0.

        self.phase_props = [self.density, self.dens_m, self.saturation, self.nu, self.viscosity, self.relperm, self.pc,
                            self.enthalpy, self.conduction, self.mass_source]

        self.output_props = {'temperature': lambda: self.temperature}

    @abc.abstractmethod
    def compute_total_enthalpy(self, state, temperature):
        pass


class GeothermalIAPWSProperties(GeothermalPropertiesBase):

    def evaluate(self, state):
        self.temperature = self.temperature_ev.evaluate(state)

        for j, phase in enumerate(['water', 'steam']):
            self.enthalpy[j] = self.enthalpy_ev[phase].evaluate(state)
            self.density[j] = self.density_ev[phase].evaluate(state)
            self.dens_m[j] = self.density[j] / self.Mw[0]
            self.saturation[j] = self.saturation_ev[phase].evaluate(state)
            self.viscosity[j] = self.viscosity_ev[phase].evaluate(state)
            self.conduction[j] = self.conduction_ev[phase].evaluate(state)
            self.relperm[j] = self.relperm_ev[phase].evaluate(state)

        self.ph = np.array([j for j in range(self.nph) if self.saturation[j] > 0])
        return

    def compute_total_enthalpy(self, state, temperature):
        return self.enthalpy_ev['total'].evaluate(state, temperature)


class GeothermalIAPWSFluidProps(FluidProps):
    def __init__(self):
        super().__init__()

        self.components = ['water']
        self.phases = ["water", "steam"]
        self.temperature_ev = iapws_temperature_evaluator()  # Create temperature object
        self.enthalpy_ev = {'water': iapws_water_enthalpy_evaluator(),
                            'steam': iapws_steam_enthalpy_evaluator(),
                            'total': iapws_total_enthalpy_evalutor()}
        self.density_ev = {'water': iapws_water_density_evaluator(),
                           'steam': iapws_steam_density_evaluator()}
        self.saturation_ev = {'water': iapws_water_saturation_evaluator(),
                              'steam': iapws_steam_saturation_evaluator()}
        self.viscosity_ev = {'water': iapws_water_viscosity_evaluator(),
                             'steam': iapws_steam_viscosity_evaluator()}
        self.conduction_ev = {'water': ConstFunc(172.8),
                              'steam': ConstFunc(0.)}
        self.relperm_ev = {'water': iapws_water_relperm_evaluator(),
                           'steam': iapws_steam_relperm_evaluator()}


class GeothermalPHProperties(GeothermalPropertiesBase):
    def __init__(self):

        super().__init__()
        self.phases = ["water", "steam"]

    def run_flash(self, pressure, enthalpy):
        _ = self.flash_ev.evaluate(pressure, enthalpy)
        flash_results = self.flash_ev.get_flash_results()
        self.nu = np.array(flash_results.nu)
        self.x = np.array(flash_results.X).reshape(self.nph, self.nc)
        self.temperature = flash_results.T

        ph = np.array([j for j in range(self.nph) if self.nu[j] > 0])

        return ph

    def compute_saturation(self, ph):
        # Get saturations [volume fraction]
        if len(ph) == 1:
            self.saturation[ph] = 1.
        else:
            Vtot = 0
            for j in ph:
                Vtot += self.nu[j] / self.dens_m[j]

            for j in ph:
                self.saturation[j] = (self.nu[j] / self.dens_m[j]) / Vtot

        return

    def compute_total_enthalpy(self, state, temperature):
        _ = self.flash_ev.evaluate_PT(state[0], temperature)
        flash_results = self.flash_ev.get_flash_results()
        nu = np.array(flash_results.nu)
        x = np.array(flash_results.X).reshape(self.nph, self.nc)

        ph = np.array([j for j in range(self.nph) if nu[j] > 0])

        enthalpy = 0.
        for j in ph:
            enthalpy += nu[j] * self.enthalpy_ev[self.phases[j]].evaluate(state[0], temperature, x[j, :])

        return enthalpy

    def evaluate(self, state):
        # Clean arrays
        for a in self.phase_props:
            a[:] = 0

        # Evaluate flash
        self.ph = self.run_flash(state[0], state[1])

        # Evaluate phase properties
        for j in self.ph:
            phase = self.phases[j]
            Mw = np.sum(self.Mw * self.x[j, :])
            self.density[j] = self.density_ev[phase].evaluate(state[0], self.temperature, self.x[j, :])
            self.dens_m[j] = self.density[j] / Mw
            self.viscosity[j] = self.viscosity_ev[phase].evaluate(state[0], self.temperature, self.x[j, :], self.density[j])
            self.enthalpy[j] = self.enthalpy_ev[phase].evaluate(state[0], self.temperature, self.x[j, :])
            self.conduction[j] = self.conduction_ev[phase].evaluate(state)

        # Compute saturation and saturation-based properties
        self.compute_saturation(self.ph)

        # self.pc = self.capillary_pressure_ev.evaluate(self.sat)
        for j in self.ph:
            self.relperm[j] = self.relperm_ev[self.phases[j]].evaluate(self.saturation[j])

        return


class GeothermalPHFluidProps(FluidProps):
    def __init__(self, ):
        super().__init__()
        self.components = ["H2O"]
        self.phases = ['water', 'steam']

        from dartsflash.libflash import PHFlash, FlashParams, EoSParams, EoS
        from dartsflash.libflash import CubicEoS, AQEoS
        from dartsflash.components import CompData
        comp_data = CompData(components=self.components, setprops=True)
        self.Mw = comp_data.Mw
        ceos = CubicEoS(comp_data, CubicEoS.PR)
        ceos.set_preferred_roots(0, 0.75, EoS.MAX)
        aq = AQEoS(comp_data, AQEoS.Jager2003)
        aq.set_eos_range(0, [0.6, 1.])

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        flash_params.add_eos("CEOS", ceos)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["AQ", "CEOS"]

        flash_params.T_min = 250.
        flash_params.T_max = 575.
        flash_params.phflash_Htol = 1e-3
        flash_params.phflash_Ttol = 1e-8

        self.flash_ev = PHFlash(flash_params)

        # properties implemented in python
        from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
        from darts.physics.properties.density import Spivey2004
        from darts.physics.properties.viscosity import MaoDuan2009
        self.enthalpy_ev = {'water': EoSEnthalpy(aq),
                            'steam': EoSEnthalpy(ceos)}
        self.density_ev = {'water': Spivey2004(self.components),
                           'steam': EoSDensity(ceos, comp_data.Mw)}
        self.viscosity_ev = {'water': MaoDuan2009(self.components),
                             'steam': ConstFunc(0.01)}
        self.conduction_ev = {'water': ConstFunc(172.8),
                              'steam': ConstFunc(0.)}
        self.relperm_ev = dict([('water', PhaseRelPerm("water")),
                                ('steam', PhaseRelPerm("gas"))])
