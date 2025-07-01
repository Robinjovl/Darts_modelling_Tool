from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import DartsModel
from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic, DensityBrineCO2
from darts.physics.properties.black_oil import *

class Model(DartsModel):
    def __init__(self, n_points=256, physics: str = 'geo'):
        # call base class constructor
        super().__init__()
        self.physics_name = physics
        self.timer.node["initialization"].start()
        zero = 1e-12

        self.set_reservoir()
        self.inj_temp = -1  # for isothermal cases
        dt_max = 31

        if physics == 'geo':
            self.set_geo_physics(n_points, zero)
            self.init_state = [200, 350]
            self.inj_temp = 300
            self.inj_comp = []
        elif physics == 'do':
            self.set_do_physics(n_points, zero)
            self.init_state = [200, 0.05]
            self.inj_comp = [1 - zero]
        elif physics == 'bo':
            self.set_bo_physics(n_points, zero)
            self.init_state = [200, 0.001225901537, 0.7711341309]
            self.inj_comp = [1 - 2 * zero, zero]
        elif physics == 'comp':
            self.set_comp_physics(n_points, zero)
            self.init_state = [200, 1.0 - 3 * zero, zero, zero]
            self.inj_comp = [0.1, 0.2, 0.5 - zero]
            dt_max = 3
        elif physics == 'CO2':
            components = ["CO2"]
            self.set_vl_physics(components, n_points)
            self.init_state = [200, 350]
            self.inj_temp = 300
            self.inj_comp = []
        elif physics == 'iapws':
            self.set_iapws_physics(n_points)
            self.init_state = [200, 350]
            self.inj_temp = 300
            self.inj_comp = []

        self.set_sim_params(first_ts=1e-3, mult_ts=4, max_ts=dt_max, tol_newton=1e-2)

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        (nx, ny, nz) = (60, 60, 3)
        nb = nx * ny * nz
        perm = np.ones(nb) * 2000
        perm = load_single_keyword('permXVanEssen.in', 'PERMX')
        perm = perm[:nb]

        poro = np.ones(nb) * 0.2
        dx = 30
        dy = 30
        dz = np.ones(nb) * 30

        # discretize structured reservoir
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                         permx=perm, permy=perm, permz=perm*0.1, poro=poro, depth=2000,
                                         hcap=2200, rcond=500)

        self.reservoir.boundary_volumes['yz_minus'] = 1e8
        self.reservoir.boundary_volumes['yz_plus'] = 1e8
        self.reservoir.boundary_volumes['xz_minus'] = 1e8
        self.reservoir.boundary_volumes['xz_plus'] = 1e8

        return

    def set_wells(self):
        # add well's locations
        iw = [30, 30]
        jw = [14, 46]

        # add well
        self.reservoir.add_well("INJ")
        for k in range(1, self.reservoir.nz):
            self.reservoir.add_perforation("INJ", cell_index=(iw[0], jw[0], k + 1),
                                           well_radius=0.16, multi_segment=True)

        # add well
        self.reservoir.add_well("PRD")
        for k in range(1, self.reservoir.nz):
            self.reservoir.add_perforation("PRD", cell_index=(iw[1], jw[1], k + 1),
                                           well_radius=0.16, multi_segment=True)

    def set_geo_physics(self, n_points, zero):
        from darts.physics.geothermal.physics import Geothermal
        from darts.physics.geothermal.property_container import PropertyContainer
        # create pre-defined physics for geothermal
        property_container = PropertyContainer()
        property_container.output_props = {'T,degrees': lambda: property_container.temperature - 273.15}

        self.physics = Geothermal(self.timer, n_points, 1, 351, 1000, 10000, cache=False)
        self.physics.add_property_region(property_container)
        self.physics.init_physics()

    def set_do_physics(self, n_points, zero):
        from darts.physics.super.physics import Compositional
        # create pre-defined physics for geothermal
        components = ["w", "o"]
        phases = ["wat", "oil"]

        property_container = DOProperties(phases_name=phases, components_name=components, min_z=zero)

        property_container.density_ev = dict([('wat', DensityBasic(compr=1e-5, dens0=1014)),
                                              ('oil', DensityBasic(compr=5e-3, dens0=500))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(0.3)),
                                                ('oil', ConstFunc(0.03))])
        property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("water", 0.1, 0.1)),
                                               ('oil', PhaseRelPerm("oil", 0.1, 0.1))])

        # create physics
        thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=n_points, min_p=0, max_p=1000, min_z=zero/10, max_z=1 - zero/10)
        self.physics.add_property_region(property_container)

        return

    def set_bo_physics(self, n_points, zero):
        from darts.physics.super.physics import Compositional

        """Physical properties"""
        # Create property containers:
        components = ['g', 'o', 'w']
        phases = ['gas', 'oil', 'wat']

        pvt = 'physics.in'

        """ properties correlations """
        pvt = 'physics.in'
        Mw = np.ones(len(components))
        property_container = BOProperties(phases_name=phases, components_name=components, Mw=Mw, min_z=zero)

        property_container.flash_ev = flash_black_oil(pvt)
        property_container.density_ev = dict([('gas', DensityGas(pvt)),
                                              ('oil', DensityOil(pvt)),
                                              ('wat', DensityWat(pvt))])
        property_container.viscosity_ev = dict([('gas', ViscGas(pvt)),
                                                ('oil', ViscOil(pvt)),
                                                ('wat', ViscWat(pvt))])
        property_container.rel_perm_ev = dict([('gas', GasRelPerm(pvt)),
                                               ('oil', OilRelPerm(pvt)),
                                               ('wat', WatRelPerm(pvt))])
        property_container.capillary_pressure_ev = dict([('pcow', CapillaryPressurePcow(pvt)),
                                                         ('pcgo', CapillaryPressurePcgo(pvt))])

        property_container.rock_compress_ev = RockCompactionEvaluator(pvt)

        """ Activate physics """
        self.physics = Compositional(components, phases, self.timer,
                                     n_points=n_points, min_p=1, max_p=450, min_z=zero/10, max_z=1-zero/10)
        self.physics.add_property_region(property_container)

        return

    def set_comp_physics(self, n_points, zero):
        from darts.physics.super.physics import Compositional
        from darts.physics.properties.flash import ConstantK
        """Physical properties"""
        # Create property containers:
        components = ['CO2', 'C1', 'H2S', 'H2O']
        phases = ['gas', 'oil', 'wat']
        nc = len(components)
        Mw = [44.01, 16.04, 34.081, 18.015]

        property_container = CompProperties(phases_name=phases, components_name=components, Mw=Mw, min_z=zero)

        """ properties correlations """
        property_container.flash_ev = ConstantK(nc - 1, [4, 2, 1e-2], zero)
        property_container.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                                              ('oil', DensityBasic(compr=1e-5, dens0=600)),
                                              ('wat', DensityBrineCO2(components, compr=1e-5, dens0=1000, co2_mult=0))])
        property_container.viscosity_ev = dict([('gas', ConstFunc(0.05)),
                                                ('oil', ConstFunc(0.5)),
                                                ('wat', ConstFunc(0.5))])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas", swc=0.2)),
                                               ('oil', PhaseRelPerm("oil", swc=0.2)),
                                               ('wat', PhaseRelPerm("wat", swc=0.2))])

        """ Activate physics """
        thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=n_points, min_p=1, max_p=500, min_z=zero/10, max_z=1-zero/10)
        self.physics.add_property_region(property_container)

        return

    def set_vl_physics(self, components, n_points):
        from darts.physics.super.physics import Compositional
        from dartsflash.mixtures import DARTSFlash, CompData, EoS, VL

        from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
        from darts.physics.properties.viscosity import Fenghour1998
        """Physical properties"""
        # Create property containers:
        phases = ['V', 'L']
        nc = len(components)
        zero = 1e-12

        comp_data = CompData(components=components, setprops=True)

        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw, min_z=zero)

        """ properties correlations """
        pt = False
        flash_ev = VL(comp_data)
        flash_ev.set_vl_eos("PR")
        flash_ev.init_flash(flash_type=DARTSFlash.FlashType.PTFlash if pt else DARTSFlash.FlashType.PHFlash)
        property_container.flash_ev = flash_ev
        property_container.density_ev = dict([('V', EoSDensity(eos=flash_ev.eos["VL"], Mw=comp_data.Mw, root_flag=EoS.RootFlag.MAX)),
                                              ('L', EoSDensity(eos=flash_ev.eos["VL"], Mw=comp_data.Mw, root_flag=EoS.RootFlag.MAX))])
        property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                ('L', Fenghour1998())])
        property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=flash_ev.eos["VL"], root_flag=EoS.RootFlag.MAX)),
                                               ('L', EoSEnthalpy(eos=flash_ev.eos["VL"], root_flag=EoS.RootFlag.MIN))])
        property_container.rel_perm_ev = dict([('V', PhaseRelPerm("gas", swc=0.2)),
                                               ('L', PhaseRelPerm("oil", swc=0.2))])
        property_container.conductivity_ev = dict([('V', ConstFunc(10.)),
                                                   ('L', ConstFunc(180.)), ])


        """ Activate physics """
        state_spec = Compositional.StateSpecification.PT if pt else Compositional.StateSpecification.PH
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=n_points, min_p=1, max_p=500, min_z=zero / 10, max_z=1 - zero / 10,
                                     min_t=273.15, max_t=473.15)
        self.physics.add_property_region(property_container)
        return

    def set_iapws_physics(self, n_points):
        from darts.physics.super.physics import Compositional
        from dartsflash.mixtures import DARTSFlash, CompData, EoS, IAPWS

        from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
        from darts.physics.properties.viscosity import Fenghour1998
        """Physical properties"""
        # Create property containers:
        components = ["H2O"]
        phases = ['V', 'L']
        nc = len(components)
        zero = 1e-12

        comp_data = CompData(components=components, setprops=True)

        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw,
                                               min_z=zero)

        """ properties correlations """
        pt = True
        flash_ev = IAPWS(iapws_ideal=True, ice_phase=False)
        flash_ev.init_flash(flash_type=DARTSFlash.FlashType.PTFlash if pt else DARTSFlash.FlashType.PHFlash)
        property_container.flash_ev = flash_ev
        property_container.density_ev = dict([('V', EoSDensity(eos=flash_ev.eos["IAPWS"], Mw=comp_data.Mw, root_flag=EoS.RootFlag.MAX)),
                                              ('L', EoSDensity(eos=flash_ev.eos["IAPWS"], Mw=comp_data.Mw, root_flag=EoS.RootFlag.MIN))])
        property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                ('L', Fenghour1998())])
        property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=flash_ev.eos["IAPWS"], root_flag=EoS.RootFlag.MAX)),
                                               ('L', EoSEnthalpy(eos=flash_ev.eos["IAPWS"], root_flag=EoS.RootFlag.MIN))])
        property_container.rel_perm_ev = dict([('V', PhaseRelPerm("gas", swc=0.2)),
                                               ('L', PhaseRelPerm("oil", swc=0.2))])
        property_container.conductivity_ev = dict([('V', ConstFunc(10.)),
                                                   ('L', ConstFunc(180.)), ])


        """ Activate physics """
        state_spec = Compositional.StateSpecification.PT if pt else Compositional.StateSpecification.PH
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=n_points, min_p=1, max_p=500, min_z=zero / 10, max_z=1 - zero / 10,
                                     min_t=273.15, max_t=473.15)
        self.physics.add_property_region(property_container)
        return

    def set_initial_conditions(self):
        if self.physics_name == 'geo' or self.physics_name == 'CO2' or self.physics_name == 'iapws':
            input_distribution = {'pressure': self.init_state[0],
                                  'temperature': self.init_state[1]
                                  }
        else:
            input_distribution = dict(zip(self.physics.vars, self.init_state))

        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)


    def set_well_controls(self):
        from darts.engines import well_control_iface
        nc = self.physics.nc
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=230,
                                               inj_composition=self.inj_comp, inj_temp=self.inj_temp)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=170)


    def compute_temperature(self, X):
        nb = self.reservoir.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        return temp

from darts.physics.super.property_container import PropertyContainer

class CompProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, Mw, min_z=1e-11):
        # Call base class constructor
        super().__init__(phases_name, components_name, Mw, min_z=min_z, temperature=1.)

    def run_flash(self, pressure, temperature, zc, evaluate_PT: bool = None):
        # evaluate_PT argument is required in PropertyContainer but is not needed in this model

        zc_r = zc[:-1] / (1 - zc[-1])
        self.flash_ev.evaluate(pressure, temperature, zc_r)
        flash_results = self.flash_ev.get_flash_results()
        nu = np.array(flash_results.nu)
        xr = np.array(flash_results.X).reshape(self.nph-1, self.nc-1)
        V = nu[0]

        if V <= 0:
            V = 0
            xr[1] = zc_r
            ph = [1, 2]
        elif V >= 1:
            V = 1
            xr[0] = zc_r
            ph = [0, 2]
        else:
            ph = [0, 1, 2]

        for i in range(self.nc - 1):
            for j in range(2):
                self.x[j][i] = xr[j][i]

        self.x[-1][-1] = 1

        self.nu[0] = V * (1 - zc[-1])
        self.nu[1] = (1 - V) * (1 - zc[-1])
        self.nu[2] = zc[-1]

        return np.array(ph, dtype=np.intp)

class DOProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, min_z=min_z, temperature=1.)

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

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        self.ph = np.array([0, 1], dtype=np.intp)

        for j in self.ph:
            # molar weight of mixture
            M = np.sum(self.x[j, :] * self.Mw)
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]

        self.nu = zc
        self.compute_saturation(self.ph)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc[j] = 0

        return

    def evaluate_at_cond(self, pressure, zc):
        self.sat[:] = 0

        ph = [0, 1]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.dens_m = [1025, 0.77]  # to match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m


class BOProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, Mw, min_z: float = 1e-11, temperature: float = None):
        # Call base class constructor
        super().__init__(phases_name, components_name, Mw, min_z=min_z, temperature=1.)

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

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        (xgo, V, pbub) = self.flash_ev.evaluate(pressure, zc)
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
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, pbub, xgo)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, pbub)  # output in [cp]

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
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[0], self.sat[2])

        pcow = self.capillary_pressure_ev['pcow'].evaluate(self.sat[2])
        pcgo = self.capillary_pressure_ev['pcgo'].evaluate(self.sat[0])

        self.pc = np.array([-pcgo, 0, pcow])

        return

    def evaluate_at_cond(self, pressure, zc):

        self.sat[:] = 0

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        self.ph = []
        for j in range(self.nph):
            if zc[j] > self.min_z:
                self.ph.append(j)
            self.dens_m[j] = self.density_ev[self.phases_name[j]].dens_sc

        self.nu = zc
        self.compute_saturation(self.ph)

        return self.sat, self.dens_m
