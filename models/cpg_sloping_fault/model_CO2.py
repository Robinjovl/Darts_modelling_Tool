import numpy as np
import pandas as pd
from scipy import interpolate

from darts.input.input_data import InputData
from darts.engines import value_vector
from darts.physics.deadoil import DeadOil, DeadOil2PFluidProps
from darts.engines import well_control_iface

from model_cpg import Model_CPG, fmt
from set_case import set_input_data

from dataclasses import dataclass
from darts.engines import well_control_iface
from darts.physics.base.physics import PhysicsBase
from darts.physics.eos_physics import EoSPhysics
from darts.physics.base.property_container import PropertyContainer
from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from scipy.special import erf

@dataclass
class Corey:
    nw: float
    ng: float
    swc: float
    sgc: float
    krwe: float
    krge: float
    labda: float
    p_entry: float
    pcmax: float
    c2: float
    def modify(self, std, mult):
        i = 0
        for attr, value in self.__dict__.items():
            if attr != 'type':
                setattr(self, attr, value * (1 + mult[i] * float(getattr(std, attr))))
            i += 1

    def random(self, std):
        for attr, value in self.__dict__.items():
            if attr != 'type':
                std_in = value * float(getattr(std, attr))
                param = np.random.normal(value, std_in)
                if param < 0:
                    param = 0
                setattr(self, attr, param)



class ModelCCS(Model_CPG):
    def __init__(self, comps):
        self.zero = 1e-10
        super().__init__()
        self.components = comps
        self.nc = len(self.components)

    def set_physics(self):
        corey_params = Corey(nw=1.5, ng=1.5, swc=0.32, sgc=0.10, krwe=1.0, krge=1.0, labda=2.,
                             p_entry=2, pcmax=300, c2=1.5)
        self.salinity = 0

        self.ini = value_vector([1 - self.zero])

        # Fluid components, ions and solid
        from dartsflash.libflash import NegativeFlash
        from dartsflash.components import CompData
        from dartsflash.mixtures import DARTSFlash, Mixture
        comp_data = CompData(self.components, setprops=True)
        nc, ni = comp_data.nc, comp_data.ni
        # len(components)
        phases = ["gas", "wat"]

        state_spec = PhysicsBase.StateSpecification.P

        nz = len(self.components) - 1
        ax_step = [self.idata.obl.p_step] + [self.idata.obl.z_step] * nz
        ax_origin = [self.idata.obl.p_origin] + [self.idata.obl.z_origin] * nz
        self.physics = EoSPhysics(self.components, phases, timer=self.timer,
                                  axes_step=ax_step, axes_origin=ax_origin,
                                  epsilon_z=self.idata.obl.epsilon_z,
                                  state_spec=state_spec, cache=False)
        #self.physics.n_axes_points[0] = 1001  # sets OBL points for pressure

        mixture = Mixture(comp_data)
        mixture.set_vl_eos(vl_eos_name="PR", hybrid_aq_eos_name="Aq")
        mixture.set_aq_eos(aq_eos_name="Aq")
        mixture.init_flash(flash_type=DARTSFlash.FlashType.NegativeFlash,
                           eos_order=["PR", "Aq"], nf_initial_guess=[NegativeFlash.Ki.Henry_VA])
        self.physics.set_mixture(mixture)

        self.physics.dispersivity = {}

        diff_w = 1e-9 * 86400
        diff_g = 2e-8 * 86400
        property_container = PropertyContainer(components_name=self.components, phases_name=phases, Mw=comp_data.Mw,
                                               min_z=self.zero, temperature=350)

        # property_container.flash_ev = ConstantK(nc=2, ki=[0.001, 100])
        property_container.flash_ev = self.physics.get_flash_ev()
        property_container.density_ev = dict([('gas', EoSDensity(eos=mixture.eos["PR"])),
                                              ('wat', Garcia2001(self.components)), ])
        property_container.viscosity_ev = dict([('gas', Fenghour1998()),
                                                ('wat', Islam2012(self.components)), ])
        property_container.diffusion_ev = dict([('gas', ConstFunc(np.ones(nc) * diff_g)),
                                                ('wat', ConstFunc(np.ones(nc) * diff_w))])
        property_container.enthalpy_ev = dict([('gas', self.physics.get_enthalpy_ev_from_flash(phase_idx=0)),
                                               ('wat', self.physics.get_enthalpy_ev_from_flash(phase_idx=1)), ])
        property_container.conductivity_ev = dict([('gas', ConstFunc(8.4)),
                                                   ('wat', ConstFunc(170.)), ])
        property_container.rel_perm_ev = dict([('gas', ModBrooksCorey(corey_params, 'gas')),
                                               ('wat', ModBrooksCorey(corey_params, 'wat'))])
        property_container.capillary_pressure_ev = ModCapillaryPressure(corey_params)

        i = 0
        self.physics.add_property_region(property_container, 0)

        property_container.output_props = {"satV": lambda ii=i: self.physics.property_containers[ii].sat[0],
                                           "rhoV": lambda ii=i: self.physics.property_containers[ii].dens[0],
                                           "rho_mA": lambda ii=i: self.physics.property_containers[ii].dens_m[1],
                                           "enthV": lambda ii=i: self.physics.property_containers[ii].enthalpy[0]}

        for j, phase_name in enumerate(phases):
            for c, component_name in enumerate(self.components):
                key = f"x{component_name}" if phase_name == 'wat' else f"y{component_name}"
                property_container.output_props[key] = lambda ii=i, jj=j, cc=c: \
                self.physics.property_containers[ii].x[jj, cc]

        self.physics.dispersivity[0] = np.zeros((self.physics.nph, self.physics.nc))

    def set_initial_conditions(self):  # override origin set_initial_conditions function from darts_model
        if self.reservoir.nz == 1 or True:
            # uniform initial conditions, # pressure in bars # composition
            # Specify reference depth, values and gradients to construct depth table in super().set_initial_conditions()
            input_depth = [0., np.amax(self.reservoir.mesh.depth)]
            P_at_surface = 1.  # bar
            input_distribution = {'pressure': [P_at_surface, P_at_surface + input_depth[1] * 0.1],  # gradient 0.1 bar/m
                                  self.physics.vars[1]: [self.ini[0], self.ini[0]]
                                  }
            return self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh,
                                                                        input_distribution=input_distribution,
                                                                        input_depth=input_depth)
        else:
            nb = self.reservoir.mesh.n_res_blocks
            depth_array = np.array(self.reservoir.mesh.depth, copy=False)[:nb]
            water_table_depth = depth_array.mean()  # specify your value here

            def sat_to_z(p, s):
                # find composition corresponding to particular saturation
                z_range = np.linspace(self.zero * 100, self.zero * 100, 2000)
                for z in z_range:
                    # state is pressure and 1 molar fractions out of 2
                    state = [p, z]
                    sat = self.physics.property_containers[0].compute_saturation_full(state)
                    if sat > s:
                        break
                return z
            def p_by_depth(depth):  # depth in meters
                return 1 + depth * 0.1  # gradient 0.1 bars/m
            def Sw_by_depth(depth):
                return 0 if depth > water_table_depth else 0.9

            # compute composition at few depth values
            n_depth_discr = self.reservoir.nz
            tbl_depth = np.linspace(depth_array.min(), depth_array.max(), n_depth_discr)
            tbl_z = np.zeros(n_depth_discr)
            for i in range(n_depth_discr):
                p = p_by_depth(tbl_depth[i])
                Sw = Sw_by_depth(tbl_depth[i])
                tbl_z[i] = sat_to_z(p, Sw)

            # and interpolate the resulting tbl_z to the full array (as loop over the variables would be slow)
            z_interp_func = interpolate.interp1d(tbl_depth, tbl_z, fill_value='extrapolate')
            Z_initial = z_interp_func(depth_array)

            P_initial = p_by_depth(depth_array)

            # set initial array for each variable: pressure and composition
            input_distribution = {self.physics.vars[0]: P_initial,
                                  self.physics.vars[1]: Z_initial
                                  }

            return self.physics.set_initial_conditions_from_array(self.reservoir.mesh,
                                                                  input_distribution=input_distribution)
    def get_arrays(self):
        '''
        :return: dictionary of current unknown arrays (p, T)
        '''
        a = self.reservoir.input_arrays  # include initial arrays and the grid

        nv = self.physics.n_vars
        nb = nv * self.reservoir.mesh.n_res_blocks
        Xn = np.array(self.physics.engine.X, copy=False)
        P = Xn[:nb:nv]
        a.update({'PRESSURE': P})

        print('P range [bars]:', fmt(P.min()), '-', fmt(P.max()))

        return a

    def print_well_rate(self):
        return

    def set_input_data(self, case=''):
        self.idata = InputData(type_hydr='isothermal', type_mech='none', init_type='uniform')
        set_input_data(self.idata, case)
        self.idata.sim.TimestepControl.dt_first = 1e-5

        self.idata.geom.burden_layers = 0

        # this sets default properties
        self.idata.fluid = DeadOil2PFluidProps() #if twophase else DeadOil3PFluidProps

        # example - how to change the properties
        # self.idata.fluid.density['water'] = DensityBasic(compr=1e-5, dens0=1014)
        # well controls
        wdata = self.idata.well_data
        wells = wdata.wells  # short name
        # set default injection composition
        inj_comp = value_vector([self.zero])  # injection composition - water

        if 'wbhp' in case:
            for w in wells:
                if self.well_is_inj(w):
                    wdata.add_inj_bhp_control(name=w, bhp=250, phase_name='gas',
                                              inj_composition=inj_comp, temperature=350)  # kmol/day | bars | K
                else:  # prod
                    wdata.add_prd_bhp_control(name=w, bhp=100)  # kmol/day | bars
        elif 'wrate' in case:
            for w in wells:
                if self.well_is_inj(w): # inject water
                    wdata.add_inj_rate_control(name=w, rate=1e6, rate_type=well_control_iface.MOLAR_RATE,
                                               phase_name='wat', inj_composition=inj_comp, bhp_constraint=250)  # kmol/day | bars | K
                else:  # prod
                    wdata.add_prd_rate_control(name=w, rate=1e6, rate_type=well_control_iface.MOLAR_RATE,
                                               phase_name='gas', bhp_constraint=100)  # kmol/day | bars
        elif 'wperiodic' in case:
            y2d = 2*365.25
            nper = 4
            for w in wells:
                if self.well_is_inj(w):  # inject water
                    for y in range(nper):
                        wdata.add_inj_rate_control(time=2*y*y2d, name=w, rate=1e5, rate_type=well_control_iface.MOLAR_RATE,
                                                   phase_name='wat', inj_composition=inj_comp, bhp_constraint=300)  # kmol/day | bars | K
                        wdata.add_inj_rate_control(time=(2*y+1)*y2d, name=w, rate=1e6, rate_type=well_control_iface.MOLAR_RATE,
                                                   phase_name='wat', inj_composition=inj_comp, bhp_constraint=300)  # kmol/day | bars | K
                else:  # prod
                    for y in range(nper):
                        wdata.add_prd_rate_control(time=2*y*y2d, name=w, rate=1e5, rate_type=well_control_iface.MOLAR_RATE,
                                                   phase_name='gas', bhp_constraint=70)  # kmol/day | bars
                        wdata.add_prd_rate_control(time=(2*y+1)*y2d, name=w, rate=1e6, rate_type=well_control_iface.MOLAR_RATE,
                                                   phase_name='gas', bhp_constraint=70)  # kmol/day | bars

        self.idata.obl.zero = 1e-13
        self.idata.obl.epsilon_z = self.idata.obl.zero
        self.idata.obl.p_step = 2.5
        self.idata.obl.p_origin = 0.0
        self.idata.obl.z_step = 2.5e-3
        self.idata.obl.z_origin = self.idata.obl.zero
        self.idata.obl.t_step = 0.25
        self.idata.obl.t_origin = 10.0


class ModBrooksCorey:
    def __init__(self, corey, phase):

        self.phase = phase

        if self.phase == "wat":
            self.k_rw_e = corey.krwe
            self.swc = corey.swc
            self.sgc = 0
            self.nw = corey.nw
        else:
            self.k_rg_e = corey.krge
            self.sgc = corey.sgc
            self.swc = 0
            self.ng = corey.ng

    def evaluate(self, sat):
        if self.phase == "wat":
            Se = (sat - self.swc)/(1 - self.swc - self.sgc)
            if Se > 1:
                Se = 1
            elif Se < 0:
                Se = 0
            k_r = self.k_rw_e * Se ** self.nw
        else:
            Se = (sat - self.sgc) / (1 - self.swc - self.sgc)
            if Se > 1:
                Se = 1
            elif Se < 0:
                Se = 0
            k_r = self.k_rg_e * Se ** self.ng

        return k_r

class ModCapillaryPressure:
    def __init__(self, corey):
        self.swc = corey.swc
        self.p_entry = corey.p_entry
        self.labda = corey.labda
        # self.labda = 3
        self.eps = 1e-10
        self.pcmax = corey.pcmax
        self.c2 = corey.c2

    def evaluate(self, sat):
        sat_w = sat[1]
        Se = (sat_w - self.swc)/(1 - self.swc)
        if Se < self.eps:
            Se = self.eps

        pc_b = self.p_entry * Se ** (-1/self.c2) # basic capillary pressure
        pc = self.pcmax * erf((pc_b * np.sqrt(np.pi)) / (self.pcmax * 2)) # smoothened capillary pressure

        Pc = np.array([0, pc], dtype=object)  # V, Aq
        return Pc
