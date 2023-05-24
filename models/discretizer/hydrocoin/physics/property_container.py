import numpy as np
from darts_flash.thermodynamics import *


class PropertyContainer:
    def __init__(self, thermodynamics: Thermodynamics, Mw, min_z=1e-11, diff_coef=0, rock_comp=1e-6, temperature=None):
        """
        Class for evaluation of thermodynamics and phase properties
        Parameters
        ----------
        thermodynamics: object for evaluating thermodynamics
        Mw: molecular weight of each component in composition vector
        min_z: minimum z value in OBL mesh
        diff_coef: diffusion coefficient
        rock_comp: rock compressibility
        temperature: if None, thermal, else isothermal at specified temperature
        """
        # Define thermal/isothermal
        self.temperature = temperature
        if temperature is not None:  # constant T specified
            self.thermal = False
        else:
            self.thermal = True

        """ Thermodynamics class """
        self.thermodynamics = thermodynamics

        """ Components and phases """
        self.components_name = thermodynamics.comp_in_z
        self.phases_name = thermodynamics.phases

        self.nc = len(self.components_name)  # Total number of components
        self.nm = 0 if thermodynamics.kin_in_vx else thermodynamics.nc_kin  # Number of mineral components
        self.nc_fl = self.nc - self.nm  # Number of fluid components
        self.nelem = len(self.components_name)

        self.np_fl = thermodynamics.np_fl  # Number of fluid phases
        self.np_kin = thermodynamics.np_kin  # Number of solid phases
        self.nph = self.np_fl + self.np_kin
        self.kin_in_vx = thermodynamics.kin_in_vx

        self.min_z = min_z
        self.Mw = Mw

        self.rock_comp = rock_comp
        self.p_ref = 1.0
        self.diff_coef = diff_coef

        # Allocate (empty) evaluators for functions
        self.density_ev = []
        self.viscosity_ev = []
        self.rel_perm_ev = []
        self.rel_well_perm_ev = []
        self.enthalpy_ev = []
        self.conductivity_ev = []
        self.rock_energy_ev = []
        # self.rock_compress_ev = []
        self.capillary_pressure_ev = []
        self.heat_source_ev = []

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
        self.solid_dens = np.zeros(self.nm)

        self.phase_props = [self.dens, self.dens_m, self.sat, self.nu, self.mu, self.kr, self.pc, self.enthalpy, self.kappa]

    def comp_out_of_bounds(self, vec_composition):
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_sum = 0
        count_corr = 0
        check_vec = np.zeros((len(vec_composition),))

        for ith_comp in range(len(vec_composition)):
            if vec_composition[ith_comp] < self.min_z:
                #print(vec_composition)
                vec_composition[ith_comp] = self.min_z
                count_corr += 1
                check_vec[ith_comp] = 1
            elif vec_composition[ith_comp] > 1 - self.min_z:
                #print(vec_composition)
                vec_composition[ith_comp] = 1 - self.min_z
                temp_sum += vec_composition[ith_comp]
            else:
                temp_sum += vec_composition[ith_comp]

        for ith_comp in range(len(vec_composition)):
            if check_vec[ith_comp] != 1:
                vec_composition[ith_comp] = vec_composition[ith_comp] / temp_sum * (1 - count_corr * self.min_z)
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

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))
        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        if self.thermal:
            temperature = vec_state_as_np[-1]
        else:
            temperature = self.temperature

        self.clean_arrays()

        # Flash
        self.ph, self.nu, self.x = self.thermodynamics.evaluate_flash(pressure, temperature, zc)

        # Density and viscosity
        for j in self.ph:
            M = 0
            # molar weight of mixture (note: self.nc is fluid components here!)
            for i in range(self.nc_fl):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j][:])  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M  # molar density [kg/m3]/[kg/kmol]=[kmol/m3]
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j][:], self.dens[j])  # output in [cp]

        # Saturation
        self.compute_saturation(self.ph)

        # Capillary pressure
        if self.capillary_pressure_ev:
            pc = self.capillary_pressure_ev.evaluate(self.sat[1])
            pc = np.append(pc, 0.)
            for j in self.ph:
                self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
                self.pc[j] = pc[j]
        else:
            self.pc = np.zeros(self.nph)

        # Kinetic rates
        rates, _ = self.thermodynamics.evaluate_kinetics(pressure, temperature, self.x, self.sat)

        return self.sat, self.x, self.dens, self.dens_m, self.mu, rates, self.kr, self.pc, self.ph

    def evaluate_thermal(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]
        temperature = vec_state_as_np[-1]
        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))
        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        for j in self.ph:
            self.enthalpy[j] = self.enthalpy_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j][:])  # kJ/kmol
            self.kappa[j] = self.conductivity_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j][:], self.dens[j])

        rock_energy = self.rock_energy_ev.evaluate(temperature)

        # Heat source and Reaction enthalpy
        heat_source = 0
        if self.heat_source_ev:
            heat_source += self.heat_source_ev.evaluate(state)

        _, reac_enthalpy = self.thermodynamics.evaluate_kinetics(pressure, temperature, self.x, self.sat)
        heat_source += reac_enthalpy

        return self.enthalpy, self.kappa, rock_energy, heat_source

    def evaluate_at_cond(self, state):
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))
        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        if self.thermal:
            temperature = vec_state_as_np[-1]
        else:
            temperature = self.temperature

        self.sat[:] = 0

        ph, self.nu, self.x = self.thermodynamics.evaluate_flash(pressure, temperature, zc)

        for j in ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j][:]) / M

        self.compute_saturation(ph)

        return self.sat, self.dens_m
