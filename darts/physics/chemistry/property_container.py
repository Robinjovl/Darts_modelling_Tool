import numpy as np

from darts.physics.properties.kinetic_registry import (
    KineticRate,
    LinearReactionSurfaceArea,
)
from darts.physics.properties.phreeqc import Flash as PhreeqcFlash
from darts.physics.properties.reaktoro import Flash as ReaktoroFlash
from darts.physics.super.property_container import PropertyContainer


class PropertyContainer(PropertyContainer):
    """
    This is the Property container class for the element-based reactive flow model.
    Chemical equilibrium is calculated using third-party geochemical equilibrium solver.
    """

    def __init__(
        self,
        phases_name,
        components_name,
        Mw,
        kinetic_mechanisms,
        stoich_matrix,
        nc_sol=0,
        np_sol=0,
        min_z=1e-11,
        rate_ann_mat=None,
        temperature=None,
        fc_mask=None,
        flash='phreeqc',
    ):
        """
        Constructor for PropertyContainer class.
        :param phases_name: List of phases names
        :type phases_name: List[str]
        :param components_name: List of components names
        :type components_name: List[str]
        :param Mw: Dictionary of molar weights of components [kg/kmol]
        :type Mw: Dict[str, float]
        :param kinetic_mechanisms: List of kinetic mechanisms
        :type kinetic_mechanisms: List[str]
        :param stoich_matrix: Stoichiometric matrix
        :type stoich_matrix: np.ndarray
        :param nc_sol: Number of components in solid phase
        :type nc_sol: int
        :param np_sol: Number of components in pure phase
        :type np_sol: int
        :param min_z: Minimum composition value
        :type min_z: float
        :param rate_ann_mat: Rate annihilation matrix, optional
        :type rate_ann_mat: np.ndarray
        :param temperature: Temperature, for isothermal simulation
        :type temperature: float | None
        :param fc_mask: Fluid component mask
        :type fc_mask: List[bool]
        :param flash: Flash type
        :type flash: str (phreeqc or reaktoro)
        """
        super().__init__(
            phases_name=phases_name,
            components_name=components_name,
            Mw=Mw,
            nc_sol=nc_sol,
            np_sol=np_sol,
            min_z=min_z,
            rate_ann_mat=rate_ann_mat,
            temperature=temperature,
        )
        self.components_name = np.array(self.components_name)
        self.stoich_matrix = stoich_matrix

        # Define primary fluid constituents
        if fc_mask is None:
            self.fc_mask = self.nc * [True]
        else:
            self.fc_mask = fc_mask
        self.fc_idx = {
            comp: i for i, comp in enumerate(self.components_name[self.fc_mask])
        }
        self.Mw_array = np.array([self.Mw[c] for c in self.components_name])
        self.n_solid = (~self.fc_mask).sum()

        # to retrieve fluid component fractions from state
        self.f_mask_state = np.concatenate([[False], self.fc_mask[:-1]])
        # to retrieve solid component fractions from state
        self.s_mask_state = np.concatenate([[False], ~self.fc_mask[:-1]])

        # figure out spec
        self.minerals = self.components_name[~self.fc_mask]

        self.sat_overall = np.zeros(self.nph + 1)
        self.diffusivity = np.zeros((self.nph, self.nc))
        self.sat_minerals = np.zeros(self.n_solid)
        self.kin_rates = np.zeros(self.n_solid)
        self.rock_compr = np.zeros(self.n_solid)

        # Define custom evaluators
        self.rock_density_ev = {}
        self.rock_compr_ev = {}
        if flash == 'phreeqc':
            self.flash_ev = PhreeqcFlash(
                min_z=self.min_z,
                minerals=self.minerals,
                components=self.components_name[self.fc_mask],
                temperature=self.temperature,
            )
        elif flash == 'reaktoro':
            self.flash_ev = ReaktoroFlash(
                min_z=self.min_z,
                minerals=self.minerals,
                components=self.components_name[self.fc_mask],
                temperature=self.temperature,
            )
        else:
            raise ValueError(f'Invalid flash type: {flash}')

        # Build one evaluator per mineral using the single-mineral API
        surface_area_ev = LinearReactionSurfaceArea(initial_area_per_mol=0.925)
        self.kinetic_rate_ev = {
            m: KineticRate(
                min_z=self.min_z,
                mineral_name=m.split('_', 1)[1],
                mechanisms=kinetic_mechanisms,
                surface_area_ev=surface_area_ev,
            )
            for m in self.minerals
        }

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics

        :param state: state variables [pres, comp_0, ..., comp_N-1, temperature (optional)]
        :type state: value_vector

        :return: updated value for operators, stored in values
        """
        nu_v, x, y, rho_phases, self.kin_state, _, _, _ = self.flash_ev.evaluate(state)
        self.nu_solid = state[self.s_mask_state]
        self.nu[0] = nu_v * (
            1 - self.nu_solid.sum()
        )  # convert to overall molar fraction
        self.nu[1] = 1 - nu_v - self.nu_solid.sum()

        pressure = state[0]
        # molar densities in kmol/m3
        self.dens_m[1], self.dens_m[0] = rho_phases['aq'], rho_phases['gas']
        self.dens_m_solid = np.array(
            [v.evaluate(pressure) / self.Mw[k] for k, v in self.rock_density_ev.items()]
        )
        self.ph = np.array([0, 1], dtype=np.intp)

        # Get saturations
        if nu_v > 0:
            sum = (
                self.nu[0] / self.dens_m[0]
                + self.nu[1] / self.dens_m[1]
                + (self.nu_solid / self.dens_m_solid).sum()
            )
            self.sat_overall[0] = self.nu[0] / self.dens_m[0] / sum
            self.sat_overall[1] = self.nu[1] / self.dens_m[1] / sum
            self.sat_overall[2] = (self.nu_solid / self.dens_m_solid).sum() / sum
        else:
            sum = (
                self.nu[1] / self.dens_m[1] + (self.nu_solid / self.dens_m_solid).sum()
            )
            self.sat_overall[0] = 0
            self.sat_overall[1] = self.nu[1] / self.dens_m[1] / sum
            self.sat_overall[2] = (self.nu_solid / self.dens_m_solid).sum() / sum
        self.sat_minerals = self.nu_solid / self.dens_m_solid / sum

        self.x = np.array([y, x])

        for j in self.ph:
            M = np.sum(self.Mw_array * self.x[j])
            self.dens[j] = self.dens_m[j] * M
            self.sat[j] = self.sat_overall[j] / np.sum(self.sat_overall[: self.nph])
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.diffusivity[j] = self.diffusion_ev[self.phases_name[j]].evaluate()

        # gas
        self.mu[0] = self.viscosity_ev[self.phases_name[0]].evaluate(
            pressure=pressure, temperature=self.temperature
        )
        # liquid
        self.mu[1] = self.viscosity_ev[self.phases_name[1]].evaluate(
            density=self.dens[1], temperature=self.temperature
        )

        for i, k in enumerate(self.rock_compr_ev.keys()):
            self.rock_compr[i] = self.rock_compr_ev[k].evaluate(pressure)
            self.kin_rates[i] = self.kinetic_rate_ev[k].evaluate(
                self.kin_state,
                self.sat_minerals[i],
                self.dens_m_solid[i],
                self.temperature,
            )
