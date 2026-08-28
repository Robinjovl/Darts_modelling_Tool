from darts.input.input_data import InputData
from darts.physics.base.physics import PhysicsBase
from darts.physics.base.property_container import PropertyContainer
from darts.physics.properties.black_oil import *


class BlackOil(PhysicsBase):
    def __init__(self, idata: InputData, timer, thermal):
        state_spec = (
            PhysicsBase.StateSpecification.PT
            if thermal
            else PhysicsBase.StateSpecification.P
        )
        # Build axes_step / axes_origin from idata.obl. The grid is uniform: same z_step
        # across all composition axes. (idata.obl.* now carries axes_step / axes_origin
        # — see darts.input.input_data.)
        nc = len(idata.fluid.components)
        nz = nc - 1
        ax_step = [idata.obl.p_step] + [idata.obl.z_step] * nz
        ax_origin = [idata.obl.p_origin] + [idata.obl.z_origin] * nz
        if thermal:
            ax_step.append(idata.obl.t_step)
            ax_origin.append(idata.obl.t_origin)
        super().__init__(
            components=idata.fluid.components,
            phases=idata.fluid.phases,
            timer=timer,
            axes_step=ax_step,
            axes_origin=ax_origin,
            epsilon_z=idata.obl.epsilon_z,
            state_spec=state_spec,
            extrapolation_flag=True,
        )

        temperature = None if thermal else 1.0
        property_container = BlackOilProperties(
            phases_name=idata.fluid.phases,
            components_name=idata.fluid.components,
            Mw=idata.fluid.Mw,
            eps_z=idata.obl.epsilon_z,
            # eps_z=idata.obl.min_z,
            temperature=temperature,
        )

        property_container.flash_ev = idata.fluid.flash_ev
        property_container.density_ev = idata.fluid.density
        property_container.viscosity_ev = idata.fluid.viscosity
        property_container.rel_perm_ev = idata.fluid.rel_perm
        property_container.capillary_pressure_ev = idata.fluid.capillary_pressure

        property_container.rock_compress_ev = RockCompactionEvaluator(idata.fluid.pvt)

        self.add_property_region(property_container)


class BlackOilProperties(PropertyContainer):
    def __init__(
        self,
        phases_name,
        components_name,
        Mw,
        eps_z: float = 1e-11,
        temperature: float = None,
    ):
        # Call base class constructor
        super().__init__(
            phases_name, components_name, Mw, eps_z=eps_z, temperature=temperature
        )
        # self.surf_dens = get_table_keyword(idata.fluid.pvt, 'DENSITY')[0]
        # self.surf_oil_dens = self.surf_dens[0]
        # self.surf_wat_dens = self.surf_dens[1]
        # self.surf_gas_dens = self.surf_dens[2]

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
        self.temperature = vec_state_as_np[-1] if self.thermal else self.temperature

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        xgo, V, pbub = self.flash_ev.evaluate(pressure, zc)
        for i in range(self.nph):
            self.x[i, i] = 1

        if V < 0:
            self.ph = np.array([1, 2])
        else:  # assume oil and water are always exists
            self.x[1][0] = xgo
            self.x[1][1] = 1 - xgo
            self.ph = np.array([0, 1, 2])

        for j in self.ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(
                pressure, pbub, xgo
            )  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(
                pressure, pbub
            )  # output in [cp]

        self.nu[2] = zc[2]
        # two phase undersaturated condition
        if pressure > pbub:
            self.nu[0] = 0
            self.nu[1] = zc[1]
        else:
            self.nu[1] = zc[1] / (1 - xgo)
            self.nu[0] = 1 - self.nu[1] - self.nu[2]

        self.compute_saturation(self.ph)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(
                self.sat[0], self.sat[2]
            )

        pcow = self.capillary_pressure_ev['pcow'].evaluate(self.sat[2])
        pcgo = self.capillary_pressure_ev['pcgo'].evaluate(self.sat[0])

        self.pc = np.array([-pcgo, 0, pcow])

        return
