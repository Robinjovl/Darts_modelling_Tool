import numpy as np
from darts.engines import value_vector
from darts.input.input_data import InputData
from darts.physics.property_base import PropertyBase


class PropertyContainer(PropertyBase):
    def __init__(self, idata: InputData, constant_temperature: float = None):
        """
        This is the PropertyContainer class for the Compositional engine.

        :param constant_temperature: Constant temperature for isothermal simulation, default is None (thermal)
        """
        super().__init__(idata)
        if constant_temperature is not None:  # constant T specified
            self.thermal = False
            self.temperature = constant_temperature
        else:
            self.thermal = True
            self.temperature = None

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

        ph = np.where(self.nu > 0)[0]
        assert(ph.size > 0)

        if ph.size == 1:
            self.x[ph[0]] = zc

        return ph

    def evaluate_mass_source(self, pressure, temperature, zc):
        self.dX = np.zeros(len(self.kinetic_rate_ev))

        for j, reaction in enumerate(self.kinetic_rate_ev):
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

        for j, reaction in enumerate(self.kinetic_rate_ev):
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
