import numpy as np
from pydantic import BaseModel, Field

from darts.engines import value_vector
from darts.physics.base.property_base import PropertyBase
from darts.physics.properties.basic import ConstFunc, RockCompactionEvaluator
from darts.physics.properties.flash import Flash


class PropertyContainerConfig(BaseModel):
    """Pydantic configuration for PropertyContainer construction.

    Fields mirror ``PropertyContainer.__init__`` parameters.
    """

    phases_name: list[str] | None = None
    components_name: list[str] | None = None
    Mw: list[float] | None = None
    min_z: float | None = Field(
        1e-9, ge=0, description="Backward-compat alias for eps_z"
    )
    eps_z: float | None = Field(
        None, ge=0, description="Min composition bound in OBL grid"
    )
    temperature: float = Field(
        1.0, description="Constant temperature for isothermal simulation"
    )
    nc_sol: int | None = None
    np_sol: int | None = None
    rock_comp: float | None = None


class PropertyContainer(PropertyBase):
    def __init__(
        self,
        phases_name: list,
        components_name: list,
        Mw: list,
        nc_sol: int = 0,
        np_sol: int = 0,
        eps_z: float = 1e-11,
        rock_comp: float = 1e-6,
        rate_ann_mat=None,
        temperature: float = None,
    ):
        """
        This is the PropertyContainer class for the Compositional engine.

        :param phases_name: List of phases
        :param components_name: List of components
        :param Mw: List of molecular weights [g/mol]
        :param nc_sol: Number of solid components, default is 0
        :param np_sol: Number of solid phases, default is 0
        :param eps_z: Minimum bound of component mole fractions in OBL grid, default is 1e-11
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

        self.rate_ann_mat = (
            rate_ann_mat if rate_ann_mat is not None else np.eye(len(components_name))
        )
        self.nelem = self.rate_ann_mat.shape[0]

        self.Mw = Mw
        self.eps_z = eps_z

        if temperature:  # constant T specified
            self.thermal = False
            self.temperature = temperature
        else:
            self.thermal = True
            self.temperature = None

        # In case of PH-formulation, PT flashes are required for calculating initial distribution (Initialize class)
        self.evaluate_PT_bool = False  # set to True when PH-formulation but PT-flash needs to be calculated (Initialize)

        # Allocate (empty) evaluators for functions
        self.density_ev = {}
        self.viscosity_ev = {}
        self.enthalpy_ev = {}
        self.conductivity_ev = {}
        self.IFT_ev = {}

        self.rel_perm_ev = []
        self.rel_well_perm_ev = []
        self.rock_compr_ev = RockCompactionEvaluator(compres=rock_comp)
        self.rock_density_ev = ConstFunc(2650.0)
        self.capillary_pressure_ev = ConstFunc(np.zeros(self.np_fl))
        self.diffusion_ev = {
            ph: ConstFunc(np.zeros(self.nc_fl)) for ph in phases_name[: self.np_fl]
        }
        self.kinetic_rate_ev = {}
        self.energy_source_ev = []
        self.flash_ev: Flash = 0
        self.permporo_mult_ev = ConstFunc(1.0)

        # passing arguments
        self.x = np.zeros((self.np_fl, self.nc_fl))
        self.x_mass = np.zeros((self.np_fl, self.nc_fl))
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
        self.energy_source = 0.0

        self.phase_props = [
            self.dens,
            self.dens_m,
            self.sat,
            self.mu,
            self.kr,
            self.pc,
            self.enthalpy,
            self.cond,
            self.mass_source,
        ]

        self.output_props = {"sat0": lambda: self.sat[0]}

    @classmethod
    def from_config(cls, config: PropertyContainerConfig) -> "PropertyContainer":
        """Construct a PropertyContainer from a validated config object."""
        eps_z = config.eps_z if config.eps_z is not None else config.min_z
        kwargs: dict = dict(
            phases_name=config.phases_name,
            components_name=config.components_name,
            Mw=config.Mw,
            eps_z=eps_z,
            temperature=config.temperature,
        )
        if config.nc_sol is not None:
            kwargs["nc_sol"] = config.nc_sol
        if config.np_sol is not None:
            kwargs["np_sol"] = config.np_sol
        if config.rock_comp is not None:
            kwargs["rock_comp"] = config.rock_comp
        return cls(**kwargs)

    def get_state(self, state):
        """
        Get tuple of (pressure, state_spec_2 (temperature/enthalpy/entropy),
                      [z0, ... zn-1]) at current OBL point (state)
        If isothermal, temperature returns initial temperature.
        If solids are present, the modified variables zc* sum to 1 and correspond to saturation for the solid components.
        To obtain mole fractions of the fluid components, one needs to normalize zc* for the fluid components.
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(
            vec_state_as_np[1 : self.nc], 1 - np.sum(vec_state_as_np[1 : self.nc])
        )
        if zc[-1] < 0.99 * self.eps_z:
            zc = self.comp_out_of_bounds(zc)

        if self.thermal:
            state_spec_2 = vec_state_as_np[-1]
        else:
            state_spec_2 = self.temperature

        return pressure, state_spec_2, zc

    def comp_out_of_bounds(self, vec_composition):
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_sum = 0
        count_corr = 0
        check_vec = np.zeros((len(vec_composition),))

        for ith_comp, zi in enumerate(vec_composition):
            if zi < 0.99 * self.eps_z:
                # print(vec_composition)
                vec_composition[ith_comp] = self.eps_z
                count_corr += 1
                check_vec[ith_comp] = 1
            elif zi > 1 - (self.nc - 1) * self.eps_z - 1e-15:
                # print(vec_composition)
                vec_composition[ith_comp] = 1 - (self.nc - 1) * self.eps_z
                temp_sum += vec_composition[ith_comp]
            else:
                temp_sum += vec_composition[ith_comp]

        for ith_comp, zi in enumerate(vec_composition):
            if check_vec[ith_comp] != 1:
                vec_composition[ith_comp] = (
                    zi / temp_sum * (1 - count_corr * self.eps_z)
                )
        return vec_composition

    def clean_arrays(self):
        for a in self.phase_props:
            a[:] = 0
        for j in range(self.np_fl):
            self.x[j][:] = 0

    def compute_saturation(self, ph):
        # Get saturations [volume fraction]
        vol = [self.nu[j] / self.dens_m[j] for j in ph]
        self.sat[ph] = vol / np.sum(vol)

        return

    def compute_saturation_full(self, state_pt, evaluate_PT_from_PHflash: bool = False):
        pressure, temperature, zc = self.get_state(state_pt)
        self.clean_arrays()
        self.ph = self.run_flash(
            pressure, temperature, zc, evaluate_PT=evaluate_PT_from_PHflash
        )

        for j in self.ph:
            M = np.sum(self.Mw * self.x[j][:])
            self.dens_m[j] = (
                self.density_ev[self.phases_name[j]].evaluate(
                    pressure, temperature, self.x[j, :]
                )
                / M
            )

        self.compute_saturation(self.ph)

        return self.sat[0]

    def compute_total_enthalpy(self, state_pt):
        # Evaluate flash at PT
        pressure, temperature, zc = self.get_state(state_pt)
        ph = self.run_flash(pressure, temperature, zc, evaluate_PT=True)

        # Compute molar enthalpy of multiphase mixture
        enthalpy = 0.0
        for j in ph:
            enthalpy += self.nu[j] * self.enthalpy_ev[self.phases_name[j]].evaluate(
                pressure, temperature, self.x[j, :]
            )  # kJ/kmol

        return enthalpy

    def run_flash(self, pressure, state_spec_2, zc, evaluate_PT: bool = False):
        # Normalize fluid compositions
        zc_norm = (
            zc if not self.ns else zc[: self.nc_fl] / (1.0 - np.sum(zc[self.nc_fl :]))
        )

        # Evaluates flash, then uses getter for nu and x - for compatibility with DARTS-flash
        if evaluate_PT:
            # In case of PH-formulation, PT flashes are required for calculating initial distribution
            error_output = self.flash_ev.evaluate(
                pressure, state_spec_2, zc_norm, evaluate_PT=True
            )
            flash_results = self.flash_ev.get_flash_results(evaluate_PT=True)
            self.temperature = state_spec_2
        else:
            error_output = self.flash_ev.evaluate(pressure, state_spec_2, zc_norm)
            flash_results = self.flash_ev.get_flash_results()
            self.temperature = flash_results.temperature

        self.nu = np.array(flash_results.nu)

        try:
            self.x = np.array(flash_results.X).reshape(self.np_fl, self.nc_fl)
        except ValueError as e:
            print(e.args[0], pressure, state_spec_2, zc)
            error_output += 1

        # Set present phase idxs
        ph = np.array([j for j in range(self.np_fl) if self.nu[j] > 0])

        if ph.size == 1:
            self.x[ph[0]] = zc_norm

        return ph

    def evaluate_mass_source(self, pressure, temperature, zc):
        self.dX = np.zeros(len(self.kinetic_rate_ev))
        for _j, reaction in self.kinetic_rate_ev.items():
            dm, self.dX[_j] = reaction.evaluate(
                pressure, temperature, self.x, self.sat[-1]
            )
            self.mass_source += dm

        return self.mass_source

    def evaluate(self, state: value_vector):
        """
        Evaluate the phase properties. Phase properties used only in the energy conservation equation
        are evaluated using a different method.

        :param state: state variables [pres, comp_0, ..., comp_N-1, temperature (optional)]
        :type state: value_vector
        """
        # Composition vector and pressure from state:
        pressure, state_spec_2, zc = self.get_state(state)

        self.clean_arrays()

        # Run flash
        self.ph = self.run_flash(
            pressure, state_spec_2, zc, evaluate_PT=self.evaluate_PT_bool
        )
        self.pressure = pressure
        assert self.pressure is not None, (
            "PropertyContainer does not specify self.pressure, should be set to "
            "pressure in case of pressure-based flash, "
            "self.flash.pressure in case of volume-based flash"
        )
        assert self.temperature is not None, (
            "PropertyContainer does not specify self.temperature, should be set to "
            "constant temperature in case of isothermal physics, "
            "self.flash.temperature in case of thermal"
        )

        for j in self.ph:
            M = np.sum(self.Mw[: self.nc_fl] * self.x[j][: self.nc_fl])

            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(
                self.pressure, self.temperature, self.x[j, :]
            )  # output in [kg/m3]
            self.dens_m[j] = (
                self.dens[j] / M
            )  # molar density [kg/m3]/[kg/kmol]=[kmol/m3]
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(
                self.pressure, self.temperature, self.x[j, :], self.dens[j]
            )  # output in [cp]

            self.x_mass[j, :] = (self.x[j, : self.nc_fl] * self.Mw[: self.nc_fl]) / sum(
                self.x[j, : self.nc_fl] * self.Mw[: self.nc_fl]
            )

        self.compute_saturation(self.ph)

        self.pc = self.capillary_pressure_ev.evaluate(self.sat)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])

        for j in range(self.ns):
            idx = self.np_fl + j
            self.sat[idx] = zc[self.nc_fl + j]
            self.dens[idx] = self.density_ev[self.phases_name[idx]].evaluate(
                self.pressure, self.temperature
            )
            self.dens_m[idx] = self.dens[idx] / self.Mw[self.nc_fl + j]

        self.mass_source = self.evaluate_mass_source(
            self.pressure, self.temperature, zc
        )

        return

    def evaluate_thermal(self, state):
        """
        Evaluate the phase properties used only in the energy conservation equation

        :param state: state variables [pres, comp_0, ..., comp_N-1, temperature (optional)]
        :type state: value_vector
        """
        for j in self.ph:
            self.enthalpy[j] = self.enthalpy_ev[self.phases_name[j]].evaluate(
                self.pressure, self.temperature, self.x[j, :]
            )  # kJ/kmol
            self.cond[j] = self.conductivity_ev[self.phases_name[j]].evaluate(
                self.pressure, self.temperature, self.x[j, :], self.dens[j]
            )

        for j in range(self.ns):
            idx = self.np_fl + j
            self.enthalpy[idx] = self.enthalpy_ev[self.phases_name[idx]].evaluate(
                self.pressure, self.temperature, self.x[0, :]
            )
            self.cond[idx] = self.conductivity_ev[self.phases_name[idx]].evaluate()

        # Heat source and Reaction enthalpy
        self.energy_source = 0.0
        if self.energy_source_ev:
            self.energy_source += self.energy_source_ev.evaluate(state)

        for _, reaction in self.kinetic_rate_ev.items():
            self.energy_source += reaction.evaluate_enthalpy(
                self.pressure, self.temperature, self.x, self.sat[-1]
            )

        return

    def evaluate_at_cond(self, state):
        # Composition vector and pressure from state:
        pressure, state_spec_2, zc = self.get_state(state)

        ph = self.run_flash(
            pressure, state_spec_2, zc, evaluate_PT=self.evaluate_PT_bool
        )

        for j in ph:
            M = np.sum(self.Mw * self.x[j][:])  # molar weight of mixture
            self.dens_m[j] = (
                self.density_ev[self.phases_name[j]].evaluate(
                    self.pressure, self.temperature, self.x[j][:]
                )
                / M
            )

        self.compute_saturation(ph)

        return self.sat, self.dens_m
