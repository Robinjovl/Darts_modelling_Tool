import numpy as np
from darts.engines import value_vector
from darts.physics.base.property_base import PropertyBase
from darts.physics.properties.flash import Flash
from darts.physics.properties.basic import ConstFunc, RockCompactionEvaluator, RockEnergyEvaluator


class PropertyContainer(PropertyBase):
    def __init__(self, phases_name: list, components_name: list, Mw: list, nc_sol: int = 0, np_sol: int = 0,
                 min_z: float = 1e-11, rock_comp: float = 1e-6, rate_ann_mat=None, temperature: float = None):
        """
        This is the PropertyContainer class for the Compositional engine.

        :param phases_name: List of phases
        :param components_name: List of components
        :param Mw: List of molecular weights [g/mol]
        :param nc_sol: Number of solid components, default is 0
        :param np_sol: Number of solid phases, default is 0
        :param min_z: Minimum bound of component mole fractions in OBL grid, default is 1e-11
        :param rock_comp: Rock compressibility, default is 1e-6
        :param rate_ann_mat: Rate annihilation matrix, optional
        :param temperature: Constant temperature for isothermal simulation, default is None (thermal)
        """
        # This class contains all the property evaluators required for simulation
        self.components_name = components_name
        self.phases_name = phases_name
        self.nc = len(components_name)
        self.nph = len(phases_name)
        self.ns = nc_sol
        self.nc_fl = self.nc - nc_sol
        self.np_fl = self.nph - np_sol

        self.rate_ann_mat = rate_ann_mat if rate_ann_mat is not None else np.eye(len(components_name))
        self.nelem = self.rate_ann_mat.shape[0]

        self.Mw = Mw
        self.min_z = min_z

        if temperature:  # constant T specified
            self.thermal = False
            self.temperature = temperature
        else:
            self.thermal = True
            self.temperature = None

        # Allocate (empty) evaluators for functions
        self.density_ev = {}
        self.viscosity_ev = {}
        self.enthalpy_ev = {}
        self.conductivity_ev = {}

        self.rel_perm_ev = []
        self.rel_well_perm_ev = []
        self.rock_energy_ev = RockEnergyEvaluator()
        self.rock_compr_ev = RockCompactionEvaluator(compres=rock_comp)
        self.rock_density_ev = ConstFunc(2650.0)
        self.capillary_pressure_ev = ConstFunc(np.zeros(self.np_fl))
        self.diffusion_ev = {ph: ConstFunc(np.zeros(self.nc_fl)) for ph in phases_name[:self.np_fl]}
        self.kinetic_rate_ev = {}
        self.energy_source_ev = []
        self.flash_ev: Flash = 0

        # passing arguments
        self.x = np.zeros((self.np_fl, self.nc_fl))
        self.dens = np.zeros(self.nph)
        self.dens_m = np.zeros(self.nph)
        self.sat = np.zeros(self.nph)
        self.nu = np.zeros(self.np_fl)
        self.mu = np.zeros(self.np_fl)
        self.kr = np.zeros(self.np_fl)
        self.pc = np.zeros(self.np_fl)
        self.enthalpy = np.zeros(self.nph)
        self.cond = np.zeros(self.nph)
        self.dX = []
        self.mass_source = np.zeros(self.nc)
        self.energy_source = 0.

        self.phase_props = [self.dens, self.dens_m, self.sat, self.nu, self.mu, self.kr, self.pc, self.enthalpy,
                            self.cond, self.mass_source]

        self.output_props = {"sat0": lambda: self.sat[0]}

    def get_state(self, state):
        """
        Get tuple of (pressure, temperature, [z0, ... zn-1]) at current OBL point (state)
        If isothermal, temperature returns initial temperature.
        If solids are present, the modified variables zc* sum to 1 and correspond to saturation for the solid components.
        To obtain mole fractions of the fluid components, one needs to normalize zc* for the fluid components.
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))
        if zc[-1] < self.min_z:
            zc = self.comp_out_of_bounds(zc)

        if self.thermal:
            temperature = vec_state_as_np[-1]
        else:
            temperature = self.temperature

        return pressure, temperature, zc

    def comp_out_of_bounds(self, vec_composition):
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_sum = 0
        count_corr = 0
        check_vec = np.zeros((len(vec_composition),))

        for ith_comp, zi in enumerate(vec_composition):
            if zi < self.min_z:
                #print(vec_composition)
                vec_composition[ith_comp] = self.min_z
                count_corr += 1
                check_vec[ith_comp] = 1
            elif zi > 1 - self.min_z:
                #print(vec_composition)
                vec_composition[ith_comp] = 1 - self.min_z
                temp_sum += vec_composition[ith_comp]
            else:
                temp_sum += vec_composition[ith_comp]

        for ith_comp, zi in enumerate(vec_composition):
            if check_vec[ith_comp] != 1:
                vec_composition[ith_comp] = zi / temp_sum * (1 - count_corr * self.min_z)
        return vec_composition

    def clean_arrays(self):
        for a in self.phase_props:
            a[:] = 0
        for j in range(self.np_fl):
            self.x[j][:] = 0

    def compute_saturation(self, ph):
        # Get saturations [volume fraction]
        Vtot = 0
        for j in ph:
            Vtot += self.nu[j] / self.dens_m[j]

        for j in ph:
            self.sat[j] = (self.nu[j] / self.dens_m[j]) / Vtot

        return
        
    def compute_saturation_full(self, state):
        pressure, temperature, zc = self.get_state(state)
        self.clean_arrays()
        self.ph = self.run_flash(pressure, temperature, zc)

        for j in self.ph:
            M = np.sum(self.Mw * self.x[j][:])
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :]) / M

        self.compute_saturation(self.ph)

        return self.sat[0]

    def run_flash(self, pressure, temperature, zc):
        # Normalize fluid compositions
        if self.ns > 0:
            norm = 1. - np.sum(zc[self.nc_fl:])
            zc = zc[:self.nc_fl] / norm

        # Evaluates flash, then uses getter for nu and x - for compatibility with DARTS-flash
        error_output = self.flash_ev.evaluate(pressure, temperature, zc)
        flash_results = self.flash_ev.get_flash_results()
        self.nu = np.array(flash_results.nu)
        self.x = np.array(flash_results.X).reshape(self.np_fl, self.nc_fl)

        ph = np.array([j for j in range(self.np_fl) if self.nu[j] > 0])

        if ph.size == 1:
            self.x[ph[0]] = zc

        return ph

    def evaluate_mass_source(self, pressure, temperature, zc):
        self.dX = np.zeros(len(self.kinetic_rate_ev))

        for j, reaction in self.kinetic_rate_ev.items():
            dm, self.dX[j] = reaction.evaluate(pressure, temperature, self.x, zc[self.nc_fl + j])
            self.mass_source += dm

        return self.mass_source

    def evaluate(self, state: value_vector):
        """
        Class methods which evaluates the state operators for the element based physics

        :param state: state variables [pres, comp_0, ..., comp_N-1, temperature (optional)]
        :type state: value_vector

        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        pressure, temperature, zc = self.get_state(state)

        self.clean_arrays()

        self.ph = self.run_flash(pressure, temperature, zc)

        for j in self.ph:
            M = np.sum(self.Mw[:self.nc_fl] * self.x[j][:])

            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :])  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M  # molar density [kg/m3]/[kg/kmol]=[kmol/m3]
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :], self.dens[j])  # output in [cp]
        self.compute_saturation(self.ph)

        self.pc = self.capillary_pressure_ev.evaluate(self.sat)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])

        for j in range(self.ns):
            idx = self.np_fl + j
            self.sat[idx] = zc[self.nc_fl + j]
            self.dens[idx] = self.density_ev[self.phases_name[idx]].evaluate(pressure, temperature)
            self.dens_m[idx] = self.dens[idx] / self.Mw[self.nc_fl + j]

        self.mass_source = self.evaluate_mass_source(pressure, temperature, zc)

        return

    def evaluate_thermal(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        pressure, temperature, zc = self.get_state(state)

        for j in self.ph:
            self.enthalpy[j] = self.enthalpy_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :])  # kJ/kmol
            self.cond[j] = self.conductivity_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :], self.dens[j])

        for j in range(self.ns):
            idx = self.np_fl + j
            self.enthalpy[idx] = self.enthalpy_ev[self.phases_name[idx]].evaluate(pressure, temperature, self.x[0, :])
            self.cond[idx] = self.conductivity_ev[self.phases_name[idx]].evaluate()

        # Heat source and Reaction enthalpy
        self.energy_source = 0.
        if self.energy_source_ev:
            self.energy_source += self.energy_source_ev.evaluate(state)

        for j, reaction in self.kinetic_rate_ev.items():
            self.energy_source += reaction.evaluate_enthalpy(pressure, temperature, self.x, zc[self.nc_fl + j])

        return

    def evaluate_at_cond(self, state):
        # Composition vector and pressure from state:
        pressure, temperature, zc = self.get_state(state)

        ph = self.run_flash(pressure, temperature, zc)

        for j in ph:
            M = np.sum(self.Mw * self.x[j][:])  # molar weight of mixture
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j][:]) / M

        self.compute_saturation(ph)

        return self.sat, self.dens_m
