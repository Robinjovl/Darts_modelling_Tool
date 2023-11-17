import numpy as np
from darts.physics.properties.flash import Flash
from darts.physics.properties.basic import CapillaryPressure, Diffusion, RockCompactionEvaluator, RockEnergyEvaluator


class PropertyContainer:
    def __init__(self, phases_name, components_name, Mw, min_z=1e-11,
                 diff_coef=0., rock_comp=1e-6, solid_dens=None, rate_ann_mat=None, temperature=None):

        # This class contains all the property evaluators required for simulation
        self.components_name = components_name
        self.rate_ann_mat = rate_ann_mat if rate_ann_mat is not None else np.eye(len(components_name))
        # self.nc = len(components_name)
        self.nc = self.rate_ann_mat.shape[1]
        self.nelem = self.rate_ann_mat.shape[0]

        self.phases_name = phases_name
        self.nph = len(phases_name)

        self.Mw = Mw
        self.min_z = min_z

        self.solid_dens = solid_dens if solid_dens is not None else []
        self.nm = len(self.solid_dens)
        self.ncfl = self.nc - self.nm
        for i in range(self.nm):
            self.solid_dens[i] /= Mw[i + self.ncfl]

        if temperature:  # constant T specified
            self.thermal = False
            self.temperature = temperature
        else:
            self.thermal = True
            self.temperature = None

        self.diff_coef = diff_coef

        # Allocate (empty) evaluators for functions
        self.density_ev = {}
        self.viscosity_ev = {}
        self.enthalpy_ev = {}
        self.conductivity_ev = {}

        self.rel_perm_ev = []
        self.rel_well_perm_ev = []
        self.rock_energy_ev = RockEnergyEvaluator()
        self.rock_compr_ev = RockCompactionEvaluator(compres=rock_comp)
        self.capillary_pressure_ev = CapillaryPressure(self.nph)
        self.diffusion_ev = Diffusion(diff_coeff=diff_coef)
        self.kinetic_rate_ev = {}
        self.energy_source_ev = []
        self.flash_ev: Flash = 0

        # passing arguments
        self.x = np.zeros((self.nph, self.nc))
        self.dens = np.zeros(self.nph)
        self.dens_m = np.zeros(self.nph)
        self.sat = np.zeros(self.nph)
        self.nu = np.zeros(self.nph)
        self.mu = np.zeros(self.nph)
        self.kr = np.zeros(self.nph)
        self.pc = np.zeros(self.nph)
        self.enthalpy = np.zeros(self.nph)
        self.kappa = np.zeros(self.nph)

        self.phase_props = [self.dens, self.dens_m, self.sat, self.nu, self.mu, self.kr, self.pc, self.enthalpy, self.kappa]

    def get_state(self, state):
        """
        Get tuple of (pressure, temperature, [z0, ... zn-1]) at current OBL point (state)
        If isothermal, temperature returns initial temperature
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
        for j in range(self.nph):
            self.x[j][:] = 0

    def compute_saturation(self, ph):
        # Get saturations [volume fraction]
        Vtot = 0
        for j in ph:
            Vtot += self.nu[j] / self.dens_m[j]

        for j in ph:
            self.sat[j] = (self.nu[j] / self.dens_m[j]) / Vtot

        return

    def run_flash(self, pressure, temperature, zc):
        # Evaluates flash, then uses getter for nu and x - for compatibility with DARTS-flash
        error_output = self.flash_ev.evaluate(pressure, temperature, zc)
        self.nu = np.array(self.flash_ev.getnu())
        self.x = np.array(self.flash_ev.getx())

        ph = []
        for j in range(self.nph):
            if self.nu[j] > 0:
                ph.append(j)

        if len(ph) == 1:
            self.x[ph[0]] = zc

        return ph

    def evaluate_mass_source(self, pressure, temperature, zc):
        mass_source = np.zeros(self.nc)
        for j, reaction in self.kinetic_rate_ev.items():
            # mass_source += reaction.evaluate(pressure, temperature, self.x, zc[-1])
            mass_source += reaction.evaluate(pressure, temperature, self.x, self.sat[-1])
        return mass_source

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1, temperature (optional)]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        pressure, temperature, zc = self.get_state(state)

        self.clean_arrays()

        self.ph = self.run_flash(pressure, temperature, zc)

        for j in self.ph:
            M = np.sum(self.Mw * self.x[j][:])

            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :])  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M  # molar density [kg/m3]/[kg/kmol]=[kmol/m3]
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :], self.dens[j])  # output in [cp]
        self.compute_saturation(self.ph)

        self.pc = self.capillary_pressure_ev.evaluate(self.sat)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])

        mass_source = self.evaluate_mass_source(pressure, temperature, zc)

        return self.ph, self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, mass_source

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
            self.kappa[j] = self.conductivity_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :], self.dens[j])

        # Heat source and Reaction enthalpy
        energy_source = 0.
        if self.energy_source_ev:
            energy_source += self.energy_source_ev.evaluate(state)

        for j, reaction in self.kinetic_rate_ev.items():
            energy_source += reaction.evaluate_enthalpy(pressure, temperature, self.x, self.sat[-1])

        return self.enthalpy, self.kappa, energy_source

    def evaluate_at_cond(self, state):
        # Composition vector and pressure from state:
        pressure, temperature, zc = self.get_state(state)

        ph = self.run_flash(pressure, temperature, zc)

        for j in ph:
            M = np.sum(self.Mw * self.x[j][:])  # molar weight of mixture
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j][:]) / M

        self.compute_saturation(ph)

        return self.sat, self.dens_m
