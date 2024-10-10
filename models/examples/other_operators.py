def evaluate(self, state, values):
    for i in range(self.n_ops):
        values[i] = 0

    self.property.evaluate(state)
    if self.thermal:
        # Evaluate thermal properties at current state
        self.property.evaluate_thermal(state)

    # For calculating molar rates of components in each phase
    for j in range(self.nph):
        for i in range(self.nc_fl):
            values[self.nc_fl * j + i] = self.property.x[j][i] * self.property.dens_m[j] * self.property.kr[j] / self.property.mu[j]

    # For calculating molar rates of phases [kmole/day]
    for j in self.property.ph:
        values[j] = self.property.dens_m[j] * self.property.kr[j] / self.property.mu[j]

    # For calculating mass rates of phases [kg/day]
    for j in self.property.ph:
        values[j] = self.property.dens[j] * self.property.kr[j] / self.property.mu[j]

    # For calculating volumetric rates of phases [m3/day]
    for j in self.property.ph:
        values[j] = self.property.kr[j] / self.property.mu[j]

    # For calculating advective heat rate of phases
    if self.thermal:
        for j in self.property.ph:
            # H_j [kJ/kmol] rho_mj [kmol/m3] k_rj [-] / mu_j [cP ∝ bar.day] (kJ/m3.bar.day)
            values[self.FLUX_OP + j * self.ne + self.nc] = (self.property.enthalpy[j] * self.property.dens_m[j] *
                                                            self.property.kr[j] / self.property.mu[j])