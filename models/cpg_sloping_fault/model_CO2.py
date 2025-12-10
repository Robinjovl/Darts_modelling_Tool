import numpy as np
from scipy import interpolate

from darts.input.input_data import InputData

from model_base import Model_CPG, fmt

from dataclasses import dataclass
from darts.engines import well_control_iface
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from dartsflash.libflash import NegativeFlash, FlashParams, InitialGuess
from dartsflash.libflash import CubicEoS, AQEoS
from dartsflash.components import CompData

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
    def __init__(self):
        self.zero = 1e-10
        super().__init__()
        self.components = ['CO2', 'H2O']
        self.nc = len(self.components)
        self.physics_type = 'CCS'

    def set_physics(self):
        corey_params = Corey(nw=1.5, ng=1.5, swc=0.32, sgc=0.10, krwe=1.0, krge=1.0, labda=2.,
                             p_entry=2, pcmax=300, c2=1.5)
        self.salinity = 0

        # Fluid components, ions and solid
        comp_data = CompData(self.components, setprops=True)
        nc, ni = comp_data.nc, comp_data.ni
        self.ini = [self.zero]

        flash_params = FlashParams(comp_data)
        flash_params.add_eos("PR", CubicEoS(comp_data, CubicEoS.PR))
        flash_params.add_eos("AQ", AQEoS(comp_data, {AQEoS.CompType.water: AQEoS.Jager2003,
                                                     AQEoS.CompType.solute: AQEoS.Ziabakhsh2012,
                                                     AQEoS.CompType.ion: AQEoS.Jager2003
                                                     }))
        pr = flash_params.eos_params["PR"].eos
        aq = flash_params.eos_params["AQ"].eos
        flash_params.eos_order = ["PR", "AQ"]
        phases = ["gas", "wat"]

        state_spec = Compositional.StateSpecification.P

        self.physics = Compositional(self.components, phases, timer=self.timer, n_points=self.idata.obl.n_points,
                                     min_p=self.idata.obl.min_p, max_p=self.idata.obl.max_p,
                                     min_z=self.idata.obl.min_z, max_z=self.idata.obl.max_z,
                                     state_spec=state_spec, cache=False)

        self.physics.dispersivity = {}

        diff_w = 1e-9 * 86400
        diff_g = 2e-8 * 86400
        property_container = PropertyContainer(components_name=self.components, phases_name=phases, Mw=comp_data.Mw,
                                               min_z=self.zero, temperature=350)

        # property_container.flash_ev = ConstantK(nc=2, ki=[0.001, 100])
        property_container.flash_ev = NegativeFlash(flash_params, ["PR", "AQ"], [InitialGuess.Henry_VA])
        property_container.density_ev = dict([('gas', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                              ('wat', Garcia2001(self.components)), ])
        property_container.viscosity_ev = dict([('gas', Fenghour1998()),
                                                ('wat', Islam2012(self.components)), ])
        property_container.diffusion_ev = dict([('gas', ConstFunc(np.ones(nc) * diff_g)),
                                                ('wat', ConstFunc(np.ones(nc) * diff_w))])
        property_container.enthalpy_ev = dict([('gas', EoSEnthalpy(eos=pr)),
                                               ('wat', EoSEnthalpy(eos=aq)), ])
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
