import numpy as np
from dataclasses import dataclass, field

from darts.models.darts_model import DartsModel
from darts.engines import value_vector, index_vector, sim_params, conn_mesh

try:
    from darts.engines import copy_data_to_device, copy_data_to_host, allocate_device_data
except ImportError:
    pass

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, CapillaryPressure, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.flash import ConstantK
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash, FlashParams, InitialGuess
from dartsflash.libflash import CubicEoS, AQEoS
from dartsflash.components import CompData

from scipy.special import erf

# region Dataclasses
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


@dataclass
class PorPerm:
    type: str
    poro: float
    perm: float
    anisotropy: list = None
    hcap: float = 2125
    rcond: float = 181.44

# endregion


class Model(DartsModel):
    def set_physics(self, corey: dict = {}, zero: float = 1e-12, temperature: float = None, n_points: int = 10001,
                    diff = 1e-9):
        """Physical properties"""
        # Fluid components, ions and solid
        components = ["H2O", "CO2"]
        phases = ["V", "Aq"]
        comp_data = CompData(components, setprops=True)
        nc = len(components)

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        flash_params.add_eos("PR", CubicEoS(comp_data, CubicEoS.PR))
        flash_params.add_eos("AQ", AQEoS(comp_data, {AQEoS.CompType.water: AQEoS.Jager2003,
                                                     AQEoS.CompType.solute: AQEoS.Ziabakhsh2012,
                                                     AQEoS.CompType.ion: AQEoS.Jager2003
                                                     }))
        pr = flash_params.eos_params["PR"].eos
        aq = flash_params.eos_params["AQ"].eos

        flash_params.eos_order = ["PR", "AQ"]

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3

        if temperature is None:  # if None, then thermal=True
            thermal = True
        else:
            thermal = False
        pres_in = 210 # (pressure at depth of well 1 will be 300 bar)
        min_t = 273.15 if temperature is None else None
        max_t = 373.15 if temperature is None else None
        self.physics = Compositional(components, phases, timer=self.timer,
                                     n_points=n_points, min_p=200, max_p=450,
                                     min_z=zero / 10, max_z=1 - zero / 10, min_t=min_t, max_t=max_t,
                                     thermal=thermal, cache=False)
        self.physics.n_axes_points[0] = 101  # sets OBL points for pressure

        dispersivity = 10.
        self.physics.dispersivity = {}

        for i, (region, corey_params) in enumerate(corey.items()):
            diff_w = diff * 86400
            diff_g = diff * 86400
            property_container = PropertyContainer(components_name=components, phases_name=phases, Mw=comp_data.Mw[:2],
                                                   min_z=zero / 10, temperature=temperature)

            # property_container.flash_ev = ConstantK(nc=2, ki=[0.001, 100])
            property_container.flash_ev = NegativeFlash(flash_params, ["PR", "AQ"], [InitialGuess.Henry_VA])
            property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=comp_data.Mw[:2])),
                                                  ('Aq', Garcia2001(components)), ])
            property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                    ('Aq', Islam2012(components)), ])
            property_container.diffusion_ev = dict([('V', ConstFunc(np.ones(nc) * diff_g)),
                                                    ('Aq', ConstFunc(np.ones(nc) * diff_w))])

            if thermal:
                property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr)),
                                                       ('Aq', EoSEnthalpy(eos=aq)), ])

                property_container.conductivity_ev = dict([('V', ConstFunc(8.4)),
                                                           ('Aq', ConstFunc(170.)), ])

            property_container.rel_perm_ev = dict([('V', ModBrooksCorey(corey_params, 'V')),
                                                   ('Aq', ModBrooksCorey(corey_params, 'Aq'))])
            property_container.capillary_pressure_ev = ModCapillaryPressure(corey_params)

            self.physics.add_property_region(property_container, i)

            property_container.output_props = {"satV": lambda ii=i: self.physics.property_containers[ii].sat[0],
                                               "xCO2": lambda ii=i: self.physics.property_containers[ii].x[1, 1],
                                               "rhoV": lambda ii=i: self.physics.property_containers[ii].dens[0],
                                               "rho_mA": lambda ii=i: self.physics.property_containers[ii].dens_m[1],
                                               "yCO2": lambda ii=i: self.physics.property_containers[ii].x[0, 1],
                                               "enthV": lambda ii=i: self.physics.property_containers[ii].enthalpy[0]}

            if region == 0 or region == 6:
                self.physics.dispersivity[region] = np.zeros((self.physics.nph, self.physics.nc))
            else:
                disp = dispersivity * np.ones((self.physics.nph, self.physics.nc))
                disp[0, :] /= diff_g
                disp[1, :] /= diff_w
                self.physics.dispersivity[region] = disp

    def init_dispersion(self):
        # activate reconstruction of velocities
        self.reconstruct_velocities()

        # set dispersion coefficients
        nph = self.physics.nph
        nc = self.physics.nc
        self.physics.engine.dispersivity.resize(len(self.physics.regions) * self.physics.nph * self.physics.nc)
        dispersivity = np.asarray(self.physics.engine.dispersivity)
        for i, region in enumerate(self.physics.regions):
            dispersivity[i * nph * nc:(i + 1) * nph * nc] = self.physics.dispersivity[region].flatten()

        # allocate & transfer dispersivities to device
        if self.platform == 'gpu':
            dispersivity_d = self.physics.engine.get_dispersivity_d()
            allocate_device_data(self.physics.engine.dispersivity, dispersivity_d)
            copy_data_to_device(self.physics.engine.dispersivity, dispersivity_d)
    

    def output_properties_old(self):
        """
        Function to return array of properties.
        Primary variables (vars) are obtained from engine, secondary variables (props) are interpolated by property_itor.

        :returns: property_array
        :rtype: np.ndarray
        """
        # Initialize property_array
        n_vars = self.physics.n_vars
        n_props = self.physics.property_operators[next(iter(self.physics.property_operators))].n_ops
        tot_props = n_vars + n_props
        nb = self.reservoir.mesh.n_res_blocks
        property_array = np.zeros((tot_props, nb))

        # Obtain primary variables from engine
        X = np.array(self.physics.engine.X, copy=False)
        for j, variable in enumerate(self.physics.vars):
            property_array[j, :] = X[j:nb * n_vars:n_vars]

        n_ops = self.physics.n_ops
        state = value_vector(property_array[:n_vars, :].T.flatten())
        values = value_vector(np.zeros(n_ops * nb))
        values_numpy = np.array(values, copy=False)
        dvalues = value_vector(np.zeros(n_ops * nb * n_vars))
        for region, prop_itor in self.physics.property_itor.items():
            block_idx = np.where(self.op_num == region)[0].astype(np.int32)
            prop_itor.evaluate_with_derivatives(state, index_vector(block_idx), values, dvalues)
            # copy values immediately to avoid problems with host-device communication under GPU
            for j in range(n_props):
                property_array[j + n_vars, block_idx] = values_numpy[block_idx * n_ops + j]

        return property_array

    def set_str_boundary_volume_multiplier(self):
        self.reservoir.boundary_volumes['yz_minus'] = 5e4 * (1200 / self.reservoir.nz)
        self.reservoir.boundary_volumes['yz_plus']  = 5e4 * (1200 / self.reservoir.nz)

        return

    def get_mass_CO2(self, property_array):
        n_vars = self.physics.n_vars
        M_CO2 = 44.01  # kg/kmol
        M_H2O = 18.01528  # kg/kmol
        sg = property_array[n_vars]
        xCO2 = property_array[n_vars + 1]
        rhoV = property_array[n_vars + 2]
        rho_m_Aq = property_array[n_vars + 3]
        yCO2 = property_array[n_vars + 4]
        w_co2 = yCO2 * M_CO2 / (yCO2 * M_CO2 + (1 - yCO2) * M_H2O)  # co2 vapor mass fraction
        V = np.asarray(self.reservoir.mesh.volume)
        phi = np.asarray(self.reservoir.mesh.poro)
        mass_CO2 = 0
        mass_CO2 += np.sum(V * phi * w_co2 * sg * rhoV)  # vapor CO2
        mass_CO2 += np.sum(V * phi * (1 - sg) * xCO2 * rho_m_Aq * M_CO2)  # aqueous CO2 - m3*kmol/m3*kg/kmol = [kg]

        return mass_CO2


    def set_top_bot_temp(self):
        nv = self.physics.n_vars
        for bot_cell in self.reservoir.bot_cells:
            # T = 70 - 0.025 * z  - origin at bottom
            T_spec_bot = 273.15 + 70 - self.reservoir.centroids[bot_cell, 2] * 0.025
            self.physics.engine.X[bot_cell*nv+nv-1] = T_spec_bot

        for top_cell in self.reservoir.top_cells:
            # T = 70 - 0.025 * z  - origin at bottom
            T_spec_top = 273.15 + 70 - self.reservoir.centroids[top_cell, 2] * 0.025
            self.physics.engine.X[top_cell*nv+nv-1] = T_spec_top
        return

    def set_rhs_flux(self, t: float = None):
        M_CO2 = 44.01  # kg/kmol
        nv = self.physics.n_vars
        nb = self.reservoir.mesh.n_res_blocks
        rhs_flux = np.zeros(nb * nv)
        # wells
        enth_idx = list(self.physics.property_containers[0].output_props.keys()).index("enthV")
        for i, well_cell in enumerate(self.reservoir.well_cells):
            # Obtain state from engine
            p_wellcell = self.physics.engine.X[well_cell * nv]
            CO2_idx = well_cell * nv + 1  # second equation
            temp_idx = well_cell * nv + nv - 1  # last equation
            state = value_vector([p_wellcell] + self.inj_stream)

            # calculate properties
            values = value_vector(np.zeros(self.physics.n_ops))
            self.physics.property_itor[self.op_num[well_cell]].evaluate(state, values)

            enthV = values[enth_idx]
            n_CO2 = self.inj_rate[i] / M_CO2
            rhs_flux[CO2_idx] -= n_CO2
            rhs_flux[temp_idx] -= enthV * n_CO2
        return rhs_flux


class ModBrooksCorey:
    def __init__(self, corey, phase):

        self.phase = phase

        if self.phase == "Aq":
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
        if self.phase == "Aq":
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
        # sat_w = sat
        Se = (sat_w - self.swc)/(1 - self.swc)
        if Se < self.eps:
            Se = self.eps
        # pc = self.p_entry * self.eps ** (1/self.labda) * Se ** (-1/self.labda)  # for p_entry to non-wetting phase
        pc_b = self.p_entry * Se ** (-1/self.c2) # basic capillary pressure
        pc = self.pcmax * erf((pc_b * np.sqrt(np.pi)) / (self.pcmax * 2)) # smoothened capillary pressure
        # if Se > 1 - self.eps:
        #     pc = 0

        # pc = self.p_entry
        Pc = np.array([0, pc], dtype=object)  # V, Aq
        return Pc


class BrooksCorey:
    def __init__(self, wetting: bool):
        self.sat_wr = 0.15
        # self.sat_nwr = 0.1

        self.lambda_w = 4.2
        self.lambda_nw = 3.7

        self.wetting = wetting

    def evaluate(self, sat_w):
        # From Brooks-Corey (1964)
        Se = (sat_w - self.sat_wr)/(1-self.sat_wr)
        if Se > 1:
            Se = 1
        elif Se < 0:
            Se = 0

        if self.wetting:
            k_r = Se**((2+3*self.lambda_w)/self.lambda_w)
        else:
            k_r = (1-Se)**2 * (1-Se**((2+self.lambda_nw)/self.lambda_nw))

        if k_r > 1:
            k_r = 1
        elif k_r < 0:
            k_r = 0

        return k_r

######################## HIDE THIS ######################## 
cmult = 86.4
layer_props = {900001: PorPerm(type='7', poro=1e-6, perm=1e-6, anisotropy=[1, 1, 0.1], rcond=2.0 * cmult),
               900002: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1], rcond=0.92 * cmult),
               900003: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1], rcond=0.92 * cmult),
               900004: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1], rcond=0.92 * cmult),
               900005: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1], rcond=0.92 * cmult),
               900006: PorPerm(type='5', poro=0.25, perm=1013.24997, anisotropy=[1, 1, 0.1], rcond=0.92 * cmult),
               900007: PorPerm(type='1', poro=0.1, perm=0.101324997, anisotropy=[1, 1, 0.1], rcond=1.9 * cmult),
               900008: PorPerm(type='1', poro=0.1, perm=0.101324997, anisotropy=[1, 1, 0.1], rcond=1.9 * cmult),
               900009: PorPerm(type='1', poro=0.1, perm=0.101324997, anisotropy=[1, 1, 0.1], rcond=1.9 * cmult),
               900010: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900011: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900012: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900013: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900014: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900015: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900016: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900017: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900018: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900019: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900020: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900021: PorPerm(type='3', poro=0.2, perm=202.649994, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900022: PorPerm(type='4', poro=0.2, perm=506.624985, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900023: PorPerm(type='6', poro=0.35, perm=2026.49994, anisotropy=[1, 1, 0.1], rcond=0.26 * cmult),
               900024: PorPerm(type='6', poro=0.35, perm=2026.49994, anisotropy=[1, 1, 0.1], rcond=0.26 * cmult),
               900025: PorPerm(type='6', poro=0.35, perm=2026.49994, anisotropy=[1, 1, 0.1], rcond=0.26 * cmult),
               900026: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900027: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900028: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900029: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900030: PorPerm(type='2', poro=0.2, perm=101.324997, anisotropy=[1, 1, 0.1], rcond=1.25 * cmult),
               900031: PorPerm(type='7', poro=1e-6, perm=1e-6, anisotropy=[1, 1, 0.1], rcond=2.0 * cmult),
               900032: PorPerm(type='1', poro=0.1, perm=0.101324997, anisotropy=[1, 1, 0.1], rcond=1.9 * cmult),
               }
######################## ######################## ######################## 