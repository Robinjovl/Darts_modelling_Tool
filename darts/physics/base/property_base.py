import numpy as np
from darts.engines import value_vector
import abc

from darts.input.input_data import InputData
from darts.physics.properties.basic import ConstFunc, RockCompactionEvaluator, RockEnergyEvaluator


class PropertyBase:
    def __init__(self, idata: InputData):
        """
        This is the constructor of the base PropertyContainer class. Here, all properties that are required for simulation
        are defined:
        - Mass components and phases, fluid/solid, composition bounds, element reduction matrix (for chemistry, optional)
        - Thermodynamic equilibrium solver (flash), phase densities, saturation, enthalpies
        - Phase viscosities, relative permeabilities, capillary pressure curves, thermal conductivities
        - Equilibrium and kinetic chemistry
        """
        # This class contains all the property evaluators required for simulation
        self.components_name = idata.fluid.components_name
        self.phases_name = idata.fluid.phases_name
        self.nc = len(self.components_name)
        self.nph = len(self.phases_name)
        self.ns = idata.fluid.nc_sol
        self.nc_fl = self.nc - self.ns
        self.np_fl = self.nph - idata.fluid.np_sol

        self.rate_ann_mat = idata.fluid.rate_ann_mat if idata.fluid.rate_ann_mat is not None else np.eye(len(self.components_name))
        self.nelem = self.rate_ann_mat.shape[0]

        self.Mw = idata.fluid.Mw
        self.min_z = idata.obl.min_z

        # Allocate (empty) evaluators for functions
        self.flash_ev = idata.fluid.flash_ev
        self.density_ev = idata.fluid.density_ev
        self.viscosity_ev = idata.fluid.viscosity_ev
        self.enthalpy_ev = idata.fluid.enthalpy_ev
        self.conductivity_ev = idata.fluid.conductivity_ev

        self.rel_perm_ev = idata.fluid.rel_perm_ev
        self.rel_well_perm_ev = idata.fluid.rel_well_perm_ev
        self.capillary_pressure_ev = idata.fluid.capillary_pressure_ev
        self.diffusion_ev = idata.fluid.diffusion_ev
        self.kinetic_rate_ev = idata.fluid.kinetic_rate_ev
        self.energy_source_ev = idata.fluid.energy_source_ev

        self.rock_energy_ev = RockEnergyEvaluator() if idata.rock.energy_ev is None else idata.rock.energy_ev
        self.rock_compr_ev = RockCompactionEvaluator(idata.rock.compressibility) if idata.rock.compr_ev is None else idata.rock.compr_ev
        self.rock_density_ev = ConstFunc(idata.rock.density) if idata.rock.density_ev is None else idata.rock.density_ev

        # Property arrays
        self.pressure = 0.  # store pressure for generic state specification
        self.temperature = 0.  # store temperature for generic state specification

        self.nu = np.zeros(self.np_fl)  # phase fractions
        self.x = np.zeros((self.np_fl, self.nc_fl))  # phase compositions
        self.dens = np.zeros(self.nph)  # phase densities
        self.dens_m = np.zeros(self.nph)  # phase molar densities
        self.sat = np.zeros(self.nph)  # phase saturations
        self.mu = np.zeros(self.np_fl)  # phase viscosities
        self.kr = np.zeros(self.np_fl)  # phase relperms
        self.pc = np.zeros(self.np_fl)  # capillary pressures
        self.enthalpy = np.zeros(self.nph)  # phase enthalpies
        self.cond = np.zeros(self.nph)  # thermal conductivities
        self.dX = []  # reaction products
        self.mass_source = np.zeros(self.nc)  # mass sources
        self.energy_source = 0.  # energy source

        self.phase_props = [self.dens, self.dens_m, self.sat, self.nu, self.mu, self.kr, self.pc, self.enthalpy,
                            self.cond, self.mass_source]

        self.output_props = {}

    @abc.abstractmethod
    def get_state(self, state: value_vector):
        pass

    @abc.abstractmethod
    def evaluate(self, state: value_vector):
        pass

    @abc.abstractmethod
    def evaluate_thermal(self, state: value_vector):
        pass

    @abc.abstractmethod
    def evaluate_at_cond(self, state: value_vector):
        pass

    @abc.abstractmethod
    def evaluate_output_properties(self, state):
        pass
