import numpy as np
from darts.engines import value_vector

from darts.input.input_data import InputData, FluidProps
from darts.physics.geothermal.physics import Geothermal
from darts.physics.properties.basic import ConstFunc

from darts.physics.properties.iapws.iapws_property import *
from darts.physics.properties.iapws.custom_rock_property import *
from darts.physics.property_base import PropertyBase


class GeothermalIAPWS(Geothermal):
    def __init__(self, idata: InputData, timer):
        super().__init__(timer, idata.obl.n_points, idata.obl.min_p, idata.obl.max_p,
                         idata.obl.min_e, idata.obl.max_e)
        idata.rock.compr_ev = [value_vector([idata.rock.compressibility_ref_p, idata.rock.compressibility, idata.rock.compressibility_ref_T])]
        idata.rock.compaction_ev = custom_rock_compaction_evaluator(property_container.rock)
        idata.rock.energy_ev = custom_rock_energy_evaluator(property_container.rock)  # Create rock_energy object

        property_container = PropertiesIAPWS(idata)

        self.add_property_region(property_container)


class GeothermalPH(Geothermal):
    def __init__(self, idata: InputData, timer):
        # Call base class constructor
        super().__init__(timer, idata.obl.n_points, idata.obl.min_p, idata.obl.max_p,
                         idata.obl.min_e, idata.obl.max_e)
        idata.rock.compr_ev = [value_vector([idata.rock.compressibility_ref_p, idata.rock.compressibility, idata.rock.compressibility_ref_T])]
        idata.rock.compaction_ev = custom_rock_compaction_evaluator(property_container.rock)
        idata.rock.energy_ev = custom_rock_energy_evaluator(property_container.rock)  # Create rock_energy object

        property_container = PropertiesPH(idata)
        self.add_property_region(property_container)


class PropertiesIAPWS(PropertyBase):
    def __init__(self, idata: InputData):
        super().__init__(idata)

        self.temperature_ev = idata.fluid.temperature_ev
        self.saturation_ev = idata.fluid.saturation_ev
        self.output_props = {'temperature': lambda: self.temperature}

    def evaluate(self, state):
        self.temperature = self.temperature_ev.evaluate(state)

        for j, phase in enumerate(['water', 'steam']):
            self.enthalpy[j] = self.enthalpy_ev[phase].evaluate(state)
            self.dens[j] = self.density_ev[phase].evaluate(state)
            self.sat[j] = self.saturation_ev[phase].evaluate(state)
            self.mu[j] = self.viscosity_ev[phase].evaluate(state)
            self.cond[j] = self.conductivity_ev[phase].evaluate(state)
            self.kr[j] = self.rel_perm_ev[phase].evaluate(state)
        return


class PropertiesPH(PropertyBase):
    def __init__(self, idata: InputData):
        super().__init__(idata=idata)

        self.enthalpy_ev['total'] = lambda: np.nansum(self.nu * self.enthalpy)
        self.output_props = {'temperature': lambda: self.temperature}

    def run_flash(self, pressure, enthalpy, composition):
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
            self.sat[ph] = 1.
        else:
            Vtot = 0
            for j in ph:
                Vtot += self.nu[j] / self.dens_m[j]

            for j in ph:
                self.sat[j] = (self.nu[j] / self.dens_m[j]) / Vtot

        return

    def evaluate(self, state):
        # Clean arrays
        for a in self.phase_props:
            a[:] = 0

        # Evaluate flash
        self.ph = self.run_flash(state[0], state[1], state[2:])

        # Evaluate phase properties
        for j in self.ph:
            Mw = np.sum(self.Mw * self.x[j, :])
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(state[0], self.temperature, self.x[j, :])
            self.dens_m[j] = self.dens[j] / Mw
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(state[0], self.temperature, self.x[j, :], self.dens[j])
            self.enthalpy[j] = self.enthalpy_ev[self.phases_name[j]].evaluate(state[0], self.temperature, self.x[j, :])
            self.cond[j] = self.conductivity_ev[self.phases_name[j]].evaluate(state)

        # Compute saturation and saturation-based properties
        self.compute_saturation(self.ph)

        # self.pc = self.capillary_pressure_ev.evaluate(self.sat)
        for j in self.ph:
            self.relperm[j] = self.rel_perm_ev[self.phases[j]].evaluate(state)

        return


class FluidPropsIAPWS(FluidProps):
    def __init__(self):
        super().__init__(phases_name=["water", "steam"], components_name=["water"], Mw=[])

        self.temperature_ev = iapws_temperature_evaluator()  # Create temperature object
        self.enthalpy_ev = {'water': iapws_water_enthalpy_evaluator(),
                            'steam': iapws_steam_enthalpy_evaluator(),
                            'total': iapws_total_enthalpy_evalutor}
        self.density_ev = {'water': iapws_water_density_evaluator(),
                           'steam': iapws_steam_density_evaluator()}
        self.saturation_ev = {'water': iapws_water_saturation_evaluator(),
                              'steam': iapws_steam_saturation_evaluator()}
        self.viscosity_ev = {'water': iapws_water_viscosity_evaluator(),
                             'steam': iapws_steam_viscosity_evaluator()}
        self.conduction_ev = {'water': ConstFunc(172.8),
                              'steam': ConstFunc(0.)}
        self.rel_perm_ev = {'water': iapws_water_relperm_evaluator(),
                            'steam': iapws_steam_relperm_evaluator()}


class FluidPropsPH(FluidProps):
    def __init__(self):
        super().__init__(phases_name=['water', 'steam'], components_name=["H2O"], Mw=[])

        from dartsflash.libflash import PHFlash, FlashParams
        from dartsflash.libflash import CubicEoS, AQEoS
        from dartsflash.components import CompData
        comp_data = CompData(components=self.components_name, setprops=True)
        self.Mw = comp_data.Mw
        ceos = CubicEoS(comp_data, CubicEoS.PR)
        aq = AQEoS(comp_data, {AQEoS.water: AQEoS.Jager2003,
                               AQEoS.solute: AQEoS.Ziabakhsh2012,
                               AQEoS.ion: AQEoS.Jager2003})

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        flash_params.add_eos("PR", ceos)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["AQ", "PR"]

        # Flash-related parameters
        flash_params.T_min = 250.
        flash_params.T_max = 600.
        flash_params.T_init = 300.

        # Create instance of PHFlash object
        self.flash_ev = PHFlash(flash_params)

        # properties implemented in python
        from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
        from darts.physics.properties.density import Spivey2004
        from darts.physics.properties.viscosity import MaoDuan2009
        self.enthalpy_ev = {'water': EoSEnthalpy(aq),
                            'steam': EoSEnthalpy(ceos)}
        self.density_ev = {'water': Spivey2004(self.components_name),
                           'steam': EoSDensity(ceos, comp_data.Mw)}
        self.viscosity_ev = {'water': MaoDuan2009(self.components_name),
                             'steam': ConstFunc(0.01)}
        self.conduction_ev = {'water': ConstFunc(172.8),
                              'steam': ConstFunc(0.)}
        self.rel_perm_ev = {'water': iapws_water_relperm_evaluator(),
                            'steam': iapws_steam_relperm_evaluator()}
