import numpy as np

from darts.physics.super.property_container import PropertyContainer
from dartsflash.libflash import StateSpecification

class PropertyContainerDerivatives(PropertyContainer):
    def __init__(
        self,
        phases_name: list,
        components_name: list,
        Mw: list,
        nc_sol: int = 0,
        np_sol: int = 0,
        min_z: float = 1e-11,
        rock_comp: float = 1e-6,
        rate_ann_mat=None,
        temperature: float = None,
        state_spec: StateSpecification = StateSpecification.TEMPERATURE
    ):
        super().__init__(phases_name, components_name, Mw, nc_sol, np_sol, min_z, rock_comp, rate_ann_mat, temperature)

        self.state_spec = state_spec
        self.Mw = np.array(Mw)

        ## derivatives
        self.n_vars = self.nc + self.thermal
        # store current state vector and per-variable timestep tolerances (eta)
        self.state = None
        self.eta = 1e20 * np.ones(self.n_vars)
        self.T0 = None

        # flash derivatives nu, x, T w.r.t. P, X, T, zk
        self.dnudP = np.zeros(self.nph)
        self.dnudX = np.zeros(self.nph)
        self.dnudT = np.zeros(self.nph)
        self.dnudzk = np.zeros((self.nph, self.nc))

        self.dxdP = np.zeros((self.nph, self.nc))
        self.dxdX = np.zeros((self.nph, self.nc))
        self.dxdT = np.zeros((self.nph, self.nc))
        self.dxdzk = np.zeros((self.nph, self.nc, self.nc))

        self.dTdP = 0.0
        self.dTdX = 0.0
        self.dTdzk = np.zeros(self.nc)

        # property derivatives
        self.dens_m_ders = np.zeros((self.nph, self.n_vars))
        self.enthalpy_ders = np.zeros((self.nph, self.n_vars))
        self.sat_ders = np.zeros((self.nph, self.n_vars))

        self.props_derivatives = [
            # flash derivatives
            self.dnudP,
            self.dnudX,
            self.dnudT,
            self.dnudzk,
            self.dxdP,
            self.dxdX,
            self.dxdT,
            self.dxdzk,
            self.dTdP,
            self.dTdX,
            self.dTdzk,
            # property derivatives
            self.dens_m_ders,
            self.enthalpy_ders,
            self.sat_ders
        ]

        #Extraction parameters TODO: improve organisation
        self.Qe = 0
        self.phi = 0

    def clean_arrays(self):
        super().clean_arrays()
        for a in self.props_derivatives:
            if isinstance(a, np.ndarray):
                a[:] = 0
            else:
                a = 0.0

    def evaluate(self, state):
        self.clean_arrays()

        # store current state for timestep adaptivity
        self.state = np.array(state, copy=True)

        pressure, enthalpy = state[0], state[1]
        zc_norm = [1. - self.min_z]
        error_output = self.flash_ev.evaluate(pressure, enthalpy, zc_norm)
        flash_results = self.flash_ev.get_flash_results(derivs=True)

        # flash results and derivatives
        self.nu = np.array(flash_results.nu)
        self.x = np.array(flash_results.X).reshape(self.nph, self.nc)
        self.temperature = flash_results.temperature
        self.ph = np.array([j for j in range(self.nph) if self.nu[j] > 0])
        if self.ph.size == 1:
            self.x[self.ph[0]] = zc_norm

        # flash derivatives (PXFlashResults: derivatives w.r.t. P, X, z)
        dnudP = np.empty(self.nph)
        dnudX = np.empty(self.nph)
        dnudzk = np.empty(self.nph * self.nc)
        dxdP = np.empty(self.nph * self.nc)
        dxdX = np.empty(self.nph * self.nc)
        dxdzk = np.empty(self.nph * self.nc * self.nc)
        flash_results.get_derivs(dnudP, dnudX, dnudzk, dxdP, dxdX, dxdzk)

        self.dnudP[:] = dnudP
        self.dnudX[:] = dnudX
        self.dnudT[:] = 0.0  # PX flash does not provide d/dT directly
        self.dnudzk[:, :] = dnudzk.reshape(self.dnudzk.shape)

        self.dxdP[:, :] = dxdP.reshape(self.dxdP.shape)
        self.dxdX[:, :] = dxdX.reshape(self.dxdX.shape)
        self.dxdT[:, :] = 0.0  # PX flash does not provide d/dT directly
        self.dxdzk[:, :, :] = dxdzk.reshape(self.dxdzk.shape)

        # temperature derivatives
        dTdP_arr, dTdX_arr, dTdzk_arr = np.empty(1), np.empty(1), np.empty(self.nc)
        flash_results.get_dT_derivs(dTdP_arr, dTdX_arr, dTdzk_arr)
        self.dTdP = float(dTdP_arr[0])
        self.dTdX = float(dTdX_arr[0])
        self.dTdzk[:] = dTdzk_arr

        # properties and their derivatives
        for j in self.ph:
            # phase molar weight
            M = np.sum(self.Mw * self.x[j])
            dM_dP = np.sum(self.Mw * self.dxdP[j])
            dM_dX = np.sum(self.Mw * self.dxdX[j])

            # phase mass densities
            self.dens[j], drho_dP, drho_dX = \
                self.density_ev[self.phases_name[j]].evaluate(
                    pressure=pressure,
                    temperature=self.temperature,
                    x=self.x[j, :],
                    derivs=True,
                    state_spec=self.state_spec,
                    dTdP=self.dTdP,
                    dTdX=self.dTdX)  # output in [kg/m3]

            # phase molar densities
            self.dens_m[j] = self.dens[j] / M  # molar density [kg/m3]/[kg/kmol]=[kmol/m3]
            self.dens_m_ders[j] = np.array([
                (drho_dP * M - self.dens[j] * dM_dP),
                (drho_dX * M - self.dens[j] * dM_dX)
            ]) / M ** 2

            # phase enthalpies
            self.enthalpy[j], self.enthalpy_ders[j, 0], self.enthalpy_ders[j, 1] = \
                self.enthalpy_ev[self.phases_name[j]].evaluate(
                    pressure=pressure,
                    temperature=self.temperature,
                    x=self.x[j, :],
                    derivs=True,
                    state_spec=self.state_spec,
                    dTdP=self.dTdP,
                    dTdX=self.dTdX)

        # saturations
        vol = np.array([self.nu[j] / self.dens_m[j] for j in self.ph])
        tot_vol = np.sum(vol)
        self.sat[self.ph] = vol / tot_vol

        # derivative of saturation only for present phases; keep others at zero
        vol_dP = np.zeros(self.nph)
        vol_dX = np.zeros(self.nph)
        vol_dP[self.ph] = (self.dnudP[self.ph] * self.dens_m[self.ph] - self.nu[self.ph] * self.dens_m_ders[self.ph, 0]) / self.dens_m[self.ph] ** 2
        vol_dX[self.ph] = (self.dnudX[self.ph] * self.dens_m[self.ph] - self.nu[self.ph] * self.dens_m_ders[self.ph, 1]) / self.dens_m[self.ph] ** 2

        sat_dP = (vol_dP[self.ph] * tot_vol - vol * np.sum(vol_dP[self.ph])) / tot_vol ** 2
        sat_dX = (vol_dX[self.ph] * tot_vol - vol * np.sum(vol_dX[self.ph])) / tot_vol ** 2
        self.sat_ders[self.ph, 0] = sat_dP
        self.sat_ders[self.ph, 1] = sat_dX
