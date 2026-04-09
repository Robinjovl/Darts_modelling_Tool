import numpy as np
from math import fabs
import h5py
from dataclasses import dataclass, field
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from darts.models.darts_model import DartsModel
from darts.models.output import Output
from darts.physics.super.physics import Compositional

from physics import CompositionalCapillary
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData
from darts.engines import value_vector, index_vector, sim_params, conn_mesh
from darts.physics.properties.flash import ConstantK
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
import os
from scipy.interpolate import interp1d
from scipy.special import erf

from Kr_hysteresis import kr_hysteresis


# region Dataclasses
@dataclass
class Corey:
    Pc_drainage_section:str
    Pc_imbibition_section:str
    # nowetting_type: str
    nowetting_d:str
    nowetting_i: str
    wetting_type: str
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
    sgrmax: float   # Land upper bound Sgr(max)
    a: float  # Killough exponent
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
# Class SpecialProperty_container(PropertyContainer):
#     def run_flash(self):

class ALProperty_container(PropertyContainer):

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

        # Run flash
        self.ph = self.run_flash(pressure, temperature, zc)
        self.temperature = self.flash_ev.get_flash_results().temperature if not isinstance(self.flash_ev, int) else self.temperature
        assert self.temperature is not None, ("PropertyContainer does not specify self.temperature, should be set to "
                                              "constant temperature in case of isothermal physics, "
                                              "self.flash.temperature in case of thermal")

        for j in self.ph:
            M = np.sum(self.Mw[:self.nc_fl] * self.x[j][:self.nc_fl])

            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, self.temperature, self.x[j, :])  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M  # molar density [kg/m3]/[kg/kmol]=[kmol/m3]
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, self.temperature, self.x[j, :], self.dens[j])  # output in [cp]
        self.compute_saturation(self.ph)
        for j in self.ph:
            self.pc[j] = self.capillary_pressure_ev[self.phases_name[j]].evaluate(self.sat[j],state[-1])

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j], state[-1])

        for j in range(self.ns):
            idx = self.np_fl + j
            self.sat[idx] = zc[self.nc_fl + j]
            self.dens[idx] = self.density_ev[self.phases_name[idx]].evaluate(pressure, self.temperature)
            self.dens_m[idx] = self.dens[idx] / self.Mw[self.nc_fl + j]

        self.mass_source = self.evaluate_mass_source(pressure, self.temperature, zc)

        return

    def run_flash(self, pressure, temperature, zc):
        # Normalize fluid compositions
        zc_norm = zc if not self.ns else zc[:self.nc_fl] / (1. - np.sum(zc[self.nc_fl:]))

        # Evaluates flash, then uses getter for nu and x - for compatibility with DARTS-flash
        zc_array = np.asarray(zc_norm, dtype=float)
        if isinstance(self.flash_ev, ConstantK):
            error_output = self.flash_ev.evaluate(
                float(pressure),
                float(temperature),
                zc_array,
            )
        else:
            zc_list = zc_array.tolist()
            nonzero_comp_idxs = list(range(len(zc_list)))
            error_output = self.flash_ev.evaluate(
                float(pressure),
                float(temperature),
                zc_list,
                nonzero_comp_idxs,
            )
        flash_results = self.flash_ev.get_flash_results()
        self.nu = np.array(flash_results.nu)
        try:
            self.x = np.array(flash_results.X).reshape(self.np_fl, self.nc_fl)
        except ValueError as e:
            print(e.args[0], pressure, temperature, zc)
            error_output += 1

        # Set present phase idxs
        ph = np.array([j for j in range(self.np_fl) if self.nu[j] > 0])

        if ph.size == 1:
            self.x[ph[0]] = zc_norm

        return ph
class Model(DartsModel):

    # def __init__(self):
    #     # Initialize self.sgr as an array of zeros with the same length as self.reservoir.mesh.n_res_blocks
    #     self.sgr = np.zeros(self.reservoir.mesh.n_res_blocks)
    #     self.label = np.zeros(self.reservoir.mesh.n_res_blocks)
    def init(self, discr_type: str = 'tpfa', platform: str = 'cpu', restart: bool = False,
             verbose: bool = False, itor_mode: str = 'adaptive',
             itor_type: str = 'multilinear', is_barycentric: bool = False):
        """
        Function to initialize the model, which includes:
        - initialize well (perforation) position
        - initialize well rate parameters
        - initialize reservoir initial conditions
        - initialize well control settings
        - define list of operator interpolators for accumulation-flux regions and wells
        - initialize engine

        :param discr_type: 'tpfa' for using Python implementation of TPFA, 'mpfa' activates C++ implementation of MPFA
        :type discr_type: str
        :param platform: 'cpu' for CPU, 'gpu' for using GPU for matrix assembly/solvers/interpolators
        :type platform: str
        :param restart: Boolean to check if existing file should be overwritten or appended
        :type restart: bool
        :param verbose: Switch for verbose
        :type verbose: bool
        :param itor_mode: specifies either 'static' or 'adaptive' interpolator
        :type itor_mode: str
        :param itor_type: specifies either 'linear' or 'multilinear' interpolator
        :type itor_type: str
        :param is_barycentric: Flag which turn on barycentric interpolation on Delaunay simplices
        :type is_barycentric: bool
        """
        # Initialize reservoir and Mesh object
        assert self.reservoir is not None, "Reservoir object has not been defined"
        self.reservoir.init_reservoir(verbose)
        # self.reservoir.mesh.volume[0] *= 1e+12#set the first cell as a huge volume.

        self.set_wells()
        # self.set_wells_2D()
        self.has_dfm_well = False


        # Initialize physics and Engine object
        assert self.physics is not None, "Physics object has not been defined"
        self.platform = platform
        self.physics.init_physics(discr_type=discr_type, platform=platform, verbose=verbose,
                                  itor_mode=itor_mode, itor_type=itor_type, is_barycentric=is_barycentric)
        if platform == 'gpu':
            self.params.linear_type = sim_params.gpu_gmres_cpr_amgx_ilu
        self.params.sim_eps = self.physics.sim_eps

        # Initialize well objects
        self.reservoir.init_wells()
        self.physics.init_wells(self.reservoir.wells)

        self.set_op_list()
        self.set_boundary_conditions()
        self.set_well_controls()

        # when restarting the initial conditions are set in self.load_restart_data() and the engine is reset.
        self.restart = restart
        if restart is False:
            self.set_initial_conditions()
            self.reset()
        self.data_ts.print()

    def set_reservoir(self, nx=100,layer_props =None, layers_to_regions = None,logscale =True, rad_grid=False):
        L = 100  # length of reservoir
        ny = 1
        nz = 1
        nb = nx * ny * nz
        # self.sg = np.zeros(nb)
        if logscale:
            self.x_axes = np.logspace(-4, np.log10(L), nx+1)
        else:
            self.x_axes = np.linspace(0, L, nx+1)
        dx_slice = self.x_axes[1:] - self.x_axes[:-1]
        self.dx = dx_slice
        dx = np.tile(dx_slice, nz)

        dy_slice = np.zeros(nx)
        if rad_grid:
            for i in range(nx):
                dy_slice[i] = 2 * np.pi * (self.x_axes[i+1])
        else:
            dy_slice[:] = 1

        dy = np.tile(dy_slice, nz)

        dz = 1
        depth = np.zeros(nb)
        cell_to_layer = np.zeros((nx,ny,nz))
        actnum = np.zeros((nx,ny,nz))
        op_num = np.zeros(nb)
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    global_idx = k * nx * ny + j * nx + i
                    depth[global_idx] = 50. - k * dz
                    actnum[i,j,k] = 1
                    if i < nx / 2:
                        cell_to_layer[i, j, k] = 0
                    else:
                        cell_to_layer[i, j, k] = 0
                    tag = cell_to_layer[i, j, k] + 900001
                    op_num[global_idx] = layers_to_regions[layer_props[tag].type]

        self.reservoir = StructReservoir(self.timer, nx, ny, nz, dx=dx, dy=dy, dz=dz,
                                         permx=40, permy=40, permz=40, hcap=0, rcond=0, poro=0.2, op_num=op_num,depth=depth,actnum = actnum)
        # self.inj = True
        if not self.prod:
            self.reservoir.boundary_volumes['yz_plus'] = 1e8

    def set_wells(self, verbose: bool = False):
        if not self.rate_rhs:
            from darts.reservoirs.reservoir_base import ReservoirBase
            if type(self.reservoir).set_wells is not ReservoirBase.set_wells:
                # If the function has not been overloaded, pass
                self.reservoir.set_wells()
            else:
                if self.inj:
                    self.reservoir.add_well("I1",0.001)
                    # self.reservoir.add_well("I2",0.001)
                    self.reservoir.add_perforation(
                        "I1",
                        res_cell_idx=(1, 1, self.reservoir.nz),
                        well_index=1e4,
                        well_indexD=0,
                    )
                    # self.reservoir.add_perforation("I2", cell_index=(self.reservoir.nx-1, 1, self.reservoir.nz),
                    # well_index = 1e4, well_indexD = 0)
        else:
            # self.well_centers
            self.well_cells = []
            for name, center in self.well_centers.items():
                cell_index = self.reservoir.find_cell_index(center)
                self.well_cells.append(cell_index)
        if self.prod:
            self.reservoir.add_well("P1")
            # self.reservoir.add_well("P2")
            self.reservoir.add_perforation(
                "P1",
                res_cell_idx=(self.reservoir.nx, 1, self.reservoir.nz),
                well_index=10000,
                well_indexD=0,
            )
            # self.reservoir.add_perforation("P2", cell_index=(2, 1, self.reservoir.nz),
            #                                well_index=10000, well_indexD=0
            #                                )
        return

    def set_wells_2D(self):
        if not self.rate_rhs:
            self.reservoir.add_well("I1")
            for k in range(1, self.reservoir.nz//2):
                self.reservoir.add_perforation(
                    "I1",
                    res_cell_idx=(1, 1, k),
                    ms_epm=False,
                    well_index=10000,
                    well_indexD=0,
                )
        else:
            # self.well_centers
            self.well_cells = []
            for name, centers in self.well_centers.items():
                for center in centers:
                    cell_index = self.reservoir.find_cell_index(center)
                    self.well_cells.append(cell_index)
        if self.prod:
            self.reservoir.add_well("P1")
            for k in range(self.reservoir.nz):
                self.reservoir.add_perforation(
                    "P1",
                    res_cell_idx=(self.reservoir.nx, self.reservoir.ny, k + 1),
                    well_index=10000,
                    well_indexD=0,
                )

    def set_physics(self,corey,  zero, n_points, components, temperature=None, temp_inj=350.):
        """Physical properties"""
        # Fluid components, ions and solid

        phases = ["Aq", "V"]
        self.inj = [zero, 1-zero]
        self.ini = value_vector([1 - zero])
        # self.sg = np.zeros(self.reservoir.mesh.n_res_blocks)
        comp_data = CompData(components, setprops=True)

        pr = CubicEoS(comp_data, CubicEoS.PR)
        # aq = Jager2003(comp_data)
        # aq = AQEoS(comp_data, AQEoS.Ziabakhsh2012)
        aq_evaluators = {AQEoS.water: AQEoS.Jager2003,
                         AQEoS.solute: AQEoS.Ziabakhsh2012,
                         # AQEoS.ion: AQEoS.Jager2003
                         }
        aq = AQEoS(comp_data, aq_evaluators)

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["AQ", "PR"]

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3

        if temperature is None:  # if None, then thermal=True
            thermal = True
        else:
            thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        axes_min = [1, zero / 10, zero]
        axes_max = [500, 1 - zero / 10, 1]
        n_axes_points = [n_points, n_points, int(n_points)]
        self.physics = CompositionalCapillary(
            components, phases, self.timer, n_points,
            min_p=1, max_p=500, min_z=zero / 10, max_z=1 - zero / 10,
            state_spec = state_spec, cache=False, axes_min=axes_min, axes_max=axes_max,
            n_axes_points=n_axes_points)
        """ properties correlations """
        for i, (region, hysteresis_params) in enumerate(corey.items()):
            property_container = ALProperty_container(
                phases_name=phases,
                components_name=components,
                Mw=comp_data.Mw,
                temperature=temperature,
                rock_comp=0,
                eps_z=zero / 10,
            )

            # property_container.flash_ev = NegativeFlash(flash_params, ["AQ", "PR"], [InitialGuess.Henry_AV])
            property_container.flash_ev =ConstantK(len(components), [1./0.011406373765724964, 1./67.29641667035624], zero)
            # K_val = np.array([110, 0.016, 0.0015])
            # property_container.flash_ev = ConstantK(len(components), K_val, zero)
            property_container.density_ev = dict( [('V', EoSDensity(pr, comp_data.Mw)),
                                                 ('Aq', Garcia2001(components))])
                                                 # [('V',ConstFunc(800.)),
                                                 #    ('Aq',ConstFunc(1000.))])
            property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                    ('Aq', Islam2012(components)), ])
            # property_container.viscosity_ev = dict([#('V', Fenghour1998()),
            #                                          ('V', ConstFunc(6.4e-2)),
            #                                         ('Aq', ConstFunc(0.47))])#mPa.s
            # property_container.rel_perm_ev = dict([#('V', PhaseRelPermTable(hysteresis_params,'V')),
            #                                        ('V', PhaseRelPermLinear('gas',swc=0.2)),
            #                                         ('Aq', PhaseRelPermLinear('oil', swc=0.2))])
            #                                        # ('Aq', PhaseRelPerm("oil",swc = 0.2))])
            #                                        #('Aq', PhaseRelPermTable(hysteresis_params,'Aq'))])
            property_container.rel_perm_ev = dict([('V', kr_hysteresis(hysteresis_params,'gas')),
                                                    ('Aq', kr_hysteresis(hysteresis_params,'water'))])
            # property_container.rel_perm_ev = dict([('V', PhaseRelPermTable_h(hysteresis_params, 'gas')),
            #                                        ('Aq', PhaseRelPermTable_h(hysteresis_params, 'Aq'))])

            property_container.capillary_pressure_ev =dict([('V', PhaseCapillaryHysteresis(hysteresis_params,'gas')),
                                                            ('Aq', PhaseCapillaryHysteresis(hysteresis_params,'Aq'))])
            # property_container.capillary_pressure_ev = ModCapillaryPressure(hysteresis_params)
            property_container.enthalpy_ev = dict([('V', EoSEnthalpy(pr)),
                                                   ('Aq', EoSEnthalpy(aq))])
            # property_container.conductivity_ev = dict([('V', ConstFunc(10.)),
            #                                            ('Aq', ConstFunc(180.)), ])
            property_container.conductivity_ev = dict([('V', ConstFunc(0.)),
                                                       ('Aq', ConstFunc(0.)), ])

            property_container.output_props = {'satA': lambda ii=i : self.physics.property_containers[ii].sat[0],
                                               'satV': lambda ii=i: self.physics.property_containers[ii].sat[1],
                                               'densA': lambda ii=i: self.physics.property_containers[ii].dens[0],
                                               'densV': lambda ii=i: self.physics.property_containers[ii].dens[1],
                                               "enthA": lambda ii=i: self.physics.property_containers[ii].enthalpy[0],
                                               "enthV": lambda ii=i: self.physics.property_containers[ii].enthalpy[1],
                                               "muA": lambda ii=i: self.physics.property_containers[ii].mu[0],
                                               "muV": lambda ii=i: self.physics.property_containers[ii].mu[1],
                                               'x' + components[0]: lambda ii=i: self.physics.property_containers[ii].x[0, 0],
                                               'x' + components[1]: lambda ii=i: self.physics.property_containers[ii].x[0, 1],
                                               'y' + components[0]: lambda ii=i: self.physics.property_containers[ii].x[1, 0],
                                               'y' + components[1]: lambda ii=i: self.physics.property_containers[ii].x[1, 1],
                                               "RelPermV": lambda ii=i: self.physics.property_containers[ii].kr[1],
                                               "RelPermAq": lambda ii=i: self.physics.property_containers[ii].kr[0],
                                               "CapV": lambda ii=i: self.physics.property_containers[ii].pc[1],
                                               "CapAq": lambda ii=i: self.physics.property_containers[ii].pc[0]
                                               }
            self.physics.add_property_region(property_container, region)

    def set_initial_conditions(self):
        zero = 1e-12
        pressure = 250. * np.ones(self.reservoir.mesh.n_res_blocks)
        z_h2o = (1 - zero) * np.ones(self.reservoir.mesh.n_res_blocks)
        self.sg = np.zeros(self.reservoir.mesh.n_res_blocks)
        # pressure[0] = 251.
        # z_h2o[0] = zero
        input_distribution = {self.physics.vars[0]: pressure,
                              self.physics.vars[1]: z_h2o,
                              }
        n_blocks = self.reservoir.mesh.n_blocks
        # sgmax = np.asarray(self.physics.engine.sg_max)
        sgmax = value_vector([0.0] * n_blocks)
        self.physics.engine.sg_max = sgmax
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self, rate = None):
        # # define all wells as closed
        # if rate is not None:
        #     self.rate = rate
        # for i, w in enumerate(self.reservoir.wells):
        #     if 'I' in w.name:
        #         if self.rate:
        #
        #             w.control = self.physics.new_rate_inj(self.inj_rate[i], self.inj_stream[i], self.iph[i])
        #             w.constraint = self.physics.new_bhp_inj(450, self.inj_stream[i])
        #         else:
        #             w.control = self.physics.new_bhp_inj(self.p_inj, self.inj_stream[i])
        #
        #     else:
        #         if "P2" in w.name:
        #             w.control = self.physics.new_rate_prod(self.prod_rate, 1)
        #         else:
        #             w.control = self.physics.new_bhp_prod(self.p_prod)
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if 'I' in w.name:
                self.physics.set_well_controls( wctrl=w.control ,control_type=well_control_iface.MASS_RATE,
                                               is_inj=True, target=self.inj_rate[0], phase_name='V',
                                               inj_composition=self.inj)
                # self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                #                                is_inj=True, target=257.)
            else:
                self.physics.set_well_controls(wctrl=w.control,control_type=well_control_iface.BHP,
                                               is_inj=False, target=250.)
    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        nv = self.physics.n_vars# length of variables
        nb = self.reservoir.mesh.n_blocks
        rhs_flux = np.zeros(nb * nv)

        if self.rate_rhs:
            M_CO2 = 44.01  # kg/kmol
            M_H2O = 18
            sg_max = np.asarray(self.physics.engine.sg_max)
            X = np.asarray(self.physics.engine.X)
            for i, well_cell in enumerate(self.well_cells):
                # Obtain state from engine
                p_wellcell = X[well_cell * nv]
                CO2_idx = well_cell * nv + 1  # second equation
                H2O_idx = well_cell * nv # first equation
                energy_idx = well_cell * nv + nv - 1  # last equation
                state = value_vector([p_wellcell] + self.inj_stream[0] + [sg_max[well_cell]])  # for CO2 injection

                # calculate properties
                region = int(self.op_num[well_cell])
                property_container = self.physics.property_containers[region]
                property_container.evaluate(state)
                enthV = property_container.enthalpy[1]
                densV = property_container.dens[1]
                enthA = property_container.enthalpy[0]
                densA = property_container.dens[0]
                n_CO2 = self.inj_rate[0]/M_CO2
                n_H2O = self.inj_rate[1]/M_H2O
                rhs_flux[CO2_idx] -= n_CO2
                rhs_flux[H2O_idx] -= n_H2O
                if self.physics.thermal:
                    rhs_flux[energy_idx] -= enthV * n_CO2+enthA * n_H2O

        # rhs_flux[:] = 0.0
        return rhs_flux

    def plot(self, output_properties: list, fig=None, lims: dict = None):
        output_data = self.output_properties()

        tot_props = self.physics.vars + self.physics.property_operators[0].props_name
        output_idxs = {prop: tot_props.index(prop) for prop in output_properties}

        return self.reservoir.plot(output_idxs, output_data, fig=fig, lims=lims)


    def run_hysteresis(self, days: float = None, transition_days:float=None,restart_dt: float = 0., save_well_data : bool = True, save_solution_data : bool = True,
            log_3d_body_path: bool = False, verbose: bool = True):
        """
        Method to run simulation for specified time. Optional argument to specify dt to restart simulation with.

        :param days: Time increment [days]
        :type days: float
        :param restart_dt: Restart value for timestep size [days, optional]
        :type restart_dt: float
        :param verbose: Switch for verbose, default is True
        :type verbose: bool
        :param save_well_data: if True save states of well blocks at every time step to 'well_data.h5', default is True
        :type save_well_data: bool
        :param save_solution_data: if True save states of all reservoir blocks at the end of run to 'solution.h5', default is True
        :type save_solution_data: bool
        :param log_3d_body_path: hypercube output
        :type verbose: bool
        """
        assert hasattr(self, 'output'), "self.output does not exist, please call m.set_output() after m.init()"
        days = days if days is not None else self.runtime
        data_ts = self.data_ts

        # get current engine time
        t = self.physics.engine.t
        stop_time = t + days

        # same logic as in engine.run
        if fabs(t) < 1e-15 or not hasattr(self, 'prev_dt'):
            dt = data_ts.dt_first
        elif restart_dt > 0.:
            dt = restart_dt
        else:
            dt = min(self.prev_dt * data_ts.dt_mult, days, data_ts.dt_max)

        self.prev_dt = dt

        ts = 0

        nc = self.physics.n_vars
        nb = self.reservoir.mesh.n_res_blocks
        max_dx = np.zeros(nc)

        if np.fabs(data_ts.dt_mult - 1) < 1e-10:
            omega = 0.
        else:
            omega = 1 / (data_ts.dt_mult - 1)  # inversion assuming mult = (1 + omega) / omega

        while t < stop_time:
            xn = np.array(self.physics.engine.Xn, copy=True)[:nb * nc]  # need to copy since Xn will be updated Xn = X
            converged = self.run_timestep(dt, t, verbose)

            if converged:
                t += dt
                self.physics.engine.t = t
                ts += 1
                x = np.array(self.physics.engine.X, copy=False)[:nb * nc]
                dt_mult_new = data_ts.dt_mult
                for i in range(nc):
                    max_dx[i] = np.max(abs(xn[i::nc] - x[i::nc]))
                    mult = ((1 + omega) * data_ts.eta[i]) / (max_dx[i] + omega * data_ts.eta[i])
                    if mult < dt_mult_new:
                        dt_mult_new = mult

                if verbose:
                    print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d\tDT_MULT=%3.3g\tdX=%4s"
                          % (ts, t, dt, self.physics.engine.n_newton_last_dt, self.physics.engine.n_linear_last_dt,
                             dt_mult_new, np.round(max_dx, 3)))

                dt = min(dt * dt_mult_new, data_ts.dt_max)

                if np.fabs(t + dt - stop_time) < data_ts.dt_min:
                    dt = stop_time - t

                if t + dt > stop_time:
                    dt = stop_time - t
                else:
                    self.prev_dt = dt
                # # save well data at every converged time step
                # if save_well_data and save_well_data_after_run is False:
                #     self.output.save_data_to_h5(kind='well')

                if 0:
                    sg_max = np.asarray(self.physics.engine.sg_max)
                    Timevetor, sg2 = self.output.output_properties(output_properties = ['satV'], engine=True)
                    sg1 = np.array(next(iter(sg2.values())))
                    sg1 = sg1.squeeze()
                    for i in range (nb):
                        if sg_max[i] >= 0.99 and sg1[i]-self.sg[i] < 0: #and abs(sg1[i]-self.sg[i]) >= 1e-6 :
                            sg_max[i] = self.sg[i]
                            print("switch from drainage to imbibition at cell:",i)
                        elif sg_max[i] <0.99 and sg1[i] > self.sg[i] and sg1[i] >= sg_max[i]:
                            sg_max[i] = 1
                            print("switch from imbibition to drainage at cell:", i)
                    self.sg = sg1
                if 0:
                    sg_max = np.asarray(self.physics.engine.sg_max)
                    Timevetor, sg2 = self.output.output_properties(output_properties = ['satV'], engine=True)
                    sg1 = np.array(next(iter(sg2.values())))
                    sg1 = sg1.squeeze()

                    for i in range (nb):
                        if sg1[i] >= sg_max[i]:
                            sg_max[i] = sg1[i]
                            # print("update sgmax to eliminate the numerical oscillation,in cell:",i)
                if self.hys:
                    self.hysteresis()

            else:
                dt /= self.params.mult_ts
                if verbose:
                    print("Cut timestep to %2.10f" % dt)
                if dt < self.params.min_ts:
                    break

        # update current engine time
        self.physics.engine.t = stop_time
        if save_solution_data:
            self.output.save_data_to_h5(kind='reservoir')

        if verbose:
            print("TS = %d(%d), NI = %d(%d), LI = %d(%d)"
                  % (self.physics.engine.stat.n_timesteps_total, self.physics.engine.stat.n_timesteps_wasted,
                     self.physics.engine.stat.n_newton_total, self.physics.engine.stat.n_newton_wasted,
                     self.physics.engine.stat.n_linear_total, self.physics.engine.stat.n_linear_wasted))
    def hysteresis(self):
        nb = self.reservoir.mesh.n_res_blocks
        corey = {
            0: Corey( Pc_drainage_section='Pc_drainage', Pc_imbibition_section='Pc_imbibition',
                     nowetting_d="nonwetting_drainage_kr",
                     nowetting_i="nonwetting_imbibition_kr", wetting_type='wetting_kr', nw=2, ng=1.5, swc=0.2, sgc=0.1,
                     krwe=1.0, krge=0.8, labda=2., p_entry=2., pcmax=30, c2=1.5,sgrmax = 0.4, a = 0.8),
            1: Corey(Pc_drainage_section='Pc_drainage', Pc_imbibition_section='Pc_imbibition',
                     nowetting_d="nonwetting_drainage_kr",
                     nowetting_i="nonwetting_imbibition_kr", wetting_type='wetting_kr', nw=2, ng=1.5, swc=0.2, sgc=0.1,
                     krwe=1.0, krge=0.8, labda=2., p_entry=2., pcmax=30, c2=1.5,sgrmax = 0.4, a = 0.8)
        }
        sg_max = np.asarray(self.physics.engine.sg_max)
        Timevetor, sg2 = self.output.output_properties(output_properties=['satV'], engine=True)
        sg1 = np.array(next(iter(sg2.values())))
        sg1 = sg1.squeeze()
        for i in range(nb):
            if sg1[i] >= sg_max[i]:
                sg_max[i] = sg1[i]
            else:
                # pick the correct Corey set via op_num
                key = self.op_num[i]
                if key not in corey:
                    raise RuntimeError(f"Invalid op_num {key} at block {i} (corey keys: {sorted(corey.keys())})")
                c = corey[key]
                C_land = 1 / c.sgrmax - 1 / (1 - c.swc)
                sgr = sg_max[i]/(1+C_land*sg_max[i])
                if sg1[i] < sgr: #  CO2 dissolution
                    sg_max[i] = float(np.clip(sg1[i]/(1-C_land*sg1[i]), 0.0, 1.0))
        return 0

    def set_output(self, output_folder: str = 'output', sol_filename: str = 'reservoir_solution.h5',
                   well_filename: str = 'well_data.h5', save_initial: bool = True, all_phase_props : bool = False,
                   precision : str = 'd', compression : str = 'gzip', compression_level: int = 0,
                   verbose : bool = False):
        """
        Function to initialize output class

        : param output_folder: folder for h5 output files
        : param sol_filename: filename of output file
        : param save_inital:
        : param all_phase_props: Boolean to output all phase properties
        : param precision: data precision of saved data ('s' single precision, 'd' double precision)
        : param compression: default 'gzip'
        : param verbose:
        """

        self.output_folder = output_folder
        self.sol_filename  = sol_filename
        self.well_filename = well_filename
        self.sol_filepath  = os.path.join(self.output_folder, self.sol_filename)
        self.well_filepath = os.path.join(self.output_folder, self.well_filename)

        if self.restart:
            save_initial = False

        self.output = output1(
            timer=self.timer,
            reservoir=self.reservoir,
            physics=self.physics,
            op_list=self.op_list,
            params=self.params,
            output_folder=self.output_folder,
            sol_filename=self.sol_filename,
            well_filename=self.well_filename,
            save_initial=save_initial,
            all_phase_props=all_phase_props,
            precision=precision,
            compression=compression,
            compression_level=compression_level,
            verbose=verbose,
            wells=self.wells,
            has_dfm_well=self.has_dfm_well,
        )

        return
class output1(Output):
    def output_properties(self, filepath: str = None, output_properties: list = None, timestep: int = None,
                          engine=False) -> tuple[np.ndarray, dict]:
        """
        Evaluates and returns properties from saved data (HDF5 file) or a simulation engine.

        :param filepath: Path to the solution HDF5 file. Defaults to None, in which case the dartsmodel.sol_filepath is used.
        :type filepath: str, optional
        :param output_properties: List of properties to evaluate. Defaults to None, which returns an array containing only state variables.
        :type output_properties: list, optional
        :param timestep: Timestep at which to evaluate properties. Defaults to None, which will evaluate all saved timesteps.
        :type timestep: int, optional
        :param engine: If true, state variables are evaluated directly from engine.X. Defaults to False, which reads properties from the HDF5 file.
        :type engine: bool, optional

        :return property_array: A dictionary where keys are primary/secondary variables and values are NumPy arrays of the requested properties for each grid block. The shape of each array is (number_of_timesteps, number_of_gridblocks).
        :type property_array: dict
        :return timesteps: A NumPy array of the time labels.
        :type timesteps: np.ndarray

        :raises KeyError: If specified property in `output_properties` is not found in any property container
        :raises TypeError: If output_properties is not a list
        """

        if output_properties is not None and not isinstance(output_properties, list):
            raise TypeError(f"Expected 'output_properties' to be a list, but got {type(output_properties).__name__}.")
        ntotal = self.reservoir.mesh.n_blocks
        n_vars = self.physics.n_vars  # number of primary variables
        if not engine:
            # Evaluate properties from the HDF5 file
            if filepath is None:  # Establish filepath/name to HDF5 file
                path = os.path.join(self.output_folder, self.sol_filename)
            else:
                path = filepath
            timesteps, cell_id, X, var_names = self.read_specific_data(path, timestep)  # Read data from HDF5 file
            var_names = self.physics.vars + ['Sgmax']
            X = np.asarray(self.physics.engine.Xop).reshape(ntotal, n_vars + 1)[np.newaxis]
        else:
            # Evaluate properties from the physics.engine.X
            timesteps = np.array(self.physics.engine.t).reshape(1, )  # current time
            cell_id = np.arange(self.reservoir.mesh.n_res_blocks)  # cell ids
            X = np.array(self.physics.engine.Xop[:(self.physics.n_vars+1) * self.reservoir.mesh.n_res_blocks],
                         copy=True)  # solution at current time
            var_names = self.physics.vars + ['Sgmax'] # primary variable names


        nb = len(cell_id)  # number of grid blocks
        output_properties = output_properties if output_properties is not None else list(
            self.physics.vars)  # complete list of properties
        alias_map = {
            "satA": "sat_Aq",
            "satV": "sat_V",
            "densA": "dens_Aq",
            "densV": "dens_V",
            "enthA": "enthalpy_Aq",
            "enthV": "enthalpy_V",
            "muA": "mu_Aq",
            "muV": "mu_V",
            "RelPermV": "kr_V",
            "RelPermAq": "kr_Aq",
            "CapV": "pc_V",
            "CapAq": "pc_Aq",
            "xH2O": "x_Aq_H2O",
            "xCO2": "x_Aq_CO2",
            "yH2O": "x_V_H2O",
            "yCO2": "x_V_CO2",
        }

        # List of primary variables i.e. state variables
        primary_props = [prop for prop in output_properties if prop in var_names]
        primary_prop_idxs = {prop: list(var_names).index(prop) for prop in primary_props}

        # List of secondary properties
        secondary_props = [prop for prop in output_properties if prop not in var_names]
        secondary_prop_idxs = {}
        for prop in secondary_props:
            canonical_prop = alias_map.get(prop, prop)
            for container in self.physics.property_containers.values():
                if canonical_prop in container.output_props:
                    secondary_prop_idxs[prop] = list(container.output_props.keys()).index(canonical_prop)
                    break
            else:
                raise KeyError(f"Secondary property '{prop}' not found in any property container.")

        # define property array dictionary
        property_array = {prop: np.zeros((len(timesteps), nb)) for prop in primary_props + secondary_props}


        # Loop over available timesteps
        for k, t in enumerate(timesteps):
            # Extract primary properties from X vector
            for var_name, var_idx in primary_prop_idxs.items():
                if engine is False:
                    property_array[var_name][k] = X[k, :nb, var_idx]
                else:
                    property_array[var_name][k] = X[var_idx::(n_vars+1)]

            # Interpolate secondary properties
            if secondary_props:  # if empty this part is skipped
                if engine is False:
                    state = value_vector(np.stack([X[k, :nb, j] for j in range(n_vars+1)]).T.flatten().astype(np.float64))
                else:
                    state = value_vector(np.stack([X[j::(n_vars+1)] for j in range(n_vars+1)]).T.flatten())

                values = value_vector(np.zeros(self.n_ops * nb))
                values_numpy = np.array(values, copy=False)
                dvalues = value_vector(np.zeros(self.n_ops * nb * (n_vars+1)))

                for region, prop_itor in self.physics.property_itor.items():

                    block_idx = np.where(self.op_num == region)[0].astype(np.int32)
                    prop_itor.evaluate_with_derivatives(state, index_vector(block_idx), values, dvalues)

                    for prop_name, prop_idx in secondary_prop_idxs.items():
                        temp = values_numpy[prop_idx::self.n_ops]
                        property_array[prop_name][k][block_idx] = temp[block_idx]

        return timesteps, property_array
class ModCapillaryPressure:
    def __init__(self, corey):
        self.swc = corey.swc
        self.sgc = corey.sgc
        self.p_entry = corey.p_entry
        self.labda = corey.labda
        # self.labda = 3
        self.eps = 1e-10
        self.pcmax = corey.pcmax
        self.c2 = corey.c2

    def evaluate(self, sat):
        sat_w = sat[0]
        # sat_w = sat
        Se = (sat_w - self.swc)/(1 - self.swc-self.sgc)
        if Se < self.eps:
            Se = self.eps
        # pc = self.p_entry * self.eps ** (1/self.labda) * Se ** (-1/self.labda)  # for p_entry to non-wetting phase
        pc_b = self.p_entry * Se ** (-1/self.c2) # basic capillary pressure
        pc = self.pcmax * erf((pc_b * np.sqrt(np.pi)) / (self.pcmax * 2)) # smoothened capillary pressure
        # if Se > 1 - self.eps:
        #     pc = 0

        # pc = self.p_entry
        Pc = np.array([pc, 0], dtype=object)  # Aq, V
        return Pc
class PhaseRelPermLinear:
    def __init__(self, phase, swc=0., sgr=0., kre=1., n=1.):
        self.phase = phase

        self.Swc = swc
        self.Sgr = sgr
        if phase == "oil":
            self.kre = kre
            self.sr = swc
            self.sr1 = sgr
            self.n = n
        elif phase == 'gas':
            self.kre = kre
            self.sr = sgr
            self.sr1 = swc
            self.n = n
        else:  # water
            self.kre = kre
            self.sr = sgr
            self.sr1 = swc
            self.n = n

    def evaluate(self, sat,sg_max):
        if sat >= 1 - self.sr1:
            kr = self.kre
        elif sat <= self.sr:
            kr = 0
        else:
            # linear
            if self.phase == "gas":
                kr = sat / (1 - self.Swc)
            elif self.phase == "oil":
                kr = 1-(1-sat) / (1 - self.Swc)


        return kr
class PhaseRelPermTable:
    def __init__(self,corey,phase:str):
        """

        sat_data: 1D array of saturation values (sorted in ascending order)
        kr_data: 1D array of corresponding relative permeability values
        interp_kind: type of interpolation ('linear', 'cubic', etc.)
        """
        self.kind = 'linear'
        lookup_file = "LookupTable.txt"
        self.filename = lookup_file
        self.phase = phase
        self.Pc_type =corey.Pc_type
        self.nowetting_type = corey.nowetting_d
        self.wetting_type = corey.wetting_type
        if self.phase == 'V':
            self.section = self.nowetting_type
        elif self.phase == 'Aq':
            self.section = self.wetting_type
        else:
            self.section = self.Pc_type

        # Load data from file if not provided

        self.sat_data, self.kr_data = self.load_lookup_table(self.filename, self.section)

        self.kr_interpolator = interp1d(
            self.sat_data,
            self.kr_data,
            kind=self.kind,
            bounds_error=False,
            fill_value=(self.kr_data[0], self.kr_data[-1])
        )

    def evaluate(self, sat):
        """
        Evaluate the relative permeability at a given saturation value.
        """
        if self.phase == 'Pc':
            Sw = sat[0]
            pc=self.kr_interpolator(Sw)*1e-5
            Pc = np.array([pc, 0], dtype=object)
            return Pc
        else:
            Sw = sat
        # Create an interpolation function; setting fill_value to the boundary values for extrapolation

        return float(self.kr_interpolator(Sw))


    @staticmethod
    def load_lookup_table(filename, section):
        """
        Loads lookup data from a text file for a given section.
        The file should contain sections marked by headers starting with '--'.

        Parameters:
          filename (str): Path to the lookup table file.
          section (str): The section to extract (e.g., "wetting_kr").

        Returns:
          tuple: Two numpy arrays, one for saturation (sat_data) and one for relative permeability (kr_data).
        """
        sat_list = []
        kr_list = []
        in_section = False

        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue

                # Check if the line is a section header.
                if line.startswith('--'):
                    # Set the flag true if this is the section we want; false otherwise.
                    if line[2:].strip() == section:
                        in_section = True
                    else:
                        in_section = False
                    continue

                # If we are in the desired section, parse the numbers.
                if in_section:
                    try:
                        parts = line.split()
                        original_sat = float(parts[0])
                        # Use the section name instead of checking the data line itself
                        if 'nonwetting' in section.lower():
                            sat_val = 1.0 - original_sat
                        else:
                            sat_val = original_sat
                        sat_list.append(sat_val)
                        kr_list.append(float(parts[1]))
                    except Exception as e:
                        print("Error parsing line:", line, e)

        return np.array(sat_list), np.array(kr_list)

class PhaseRelPermTable_h:
    def __init__(self, corey, phase: str, lookup_file="LookupTable.txt"):
        """
        corey:       object exposing .nowetting_type, .wetting_type, .Pc_type
        phase:       'V' (nonwetting), 'Aq' (wetting), or 'Pc'
        lookup_file: path to your LookupTable.txt
        """
        self.phase = phase
        self.kind  = 'linear'
        self.filename = lookup_file

        # 1) Load the correct primary table for interpolation:
        if phase == 'gas':
            section = corey.nowetting_d
        elif phase == 'Aq':   section = corey.wetting_type
        else:                 section = corey.Pc_type

        self.sat_data, self.kr_data = self.load_lookup_table(
            self.filename, section
        )
        self.kr_interpolator = interp1d(
            self.sat_data, self.kr_data,
            kind=self.kind,
            bounds_error=False,
            fill_value=(self.kr_data[0], self.kr_data[-1])
        )

        # 2) ALSO load both primary drainage & imbibition for scanning curves
        #    (always nonwetting vs wetting tables regardless of phase)
        self.sat_dr, self.kr_dr = self.load_lookup_table(
            lookup_file, corey.nowetting_d
        )
        idx = np.argsort(self.sat_dr)
        self.sat_dr = self.sat_dr[idx]
        self.kr_dr = self.kr_dr[idx]
        # pos_dr = self.kr_dr > 0
        self.S_gc = self.sat_dr[0]
        self.S_gmax = self.sat_dr[-2]
        self.sat_im, self.kr_im = self.load_lookup_table(
            lookup_file, corey.nowetting_i
        )
        idx = np.argsort(self.sat_im)
        self.sat_im = self.sat_im[idx]
        self.kr_im = self.kr_im[idx]
        self.Sgc_max = self.sat_im[1]
        # Cache of {sg_max: interp1d}
        self._scan_cache = {}
        self.C = 1.0 / self.Sgc_max - 1.0 / self.S_gmax
    @staticmethod
    def load_lookup_table(filename, section):
        sat_list, kr_list = [], []
        in_sec = False
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('--'):
                    in_sec = (line[2:].strip() == section)
                    continue
                if in_sec:
                    x,y = line.split()[:2]
                    s0 = float(x)
                    sat = 1 - s0 if 'nonwetting' in section.lower() else s0
                    sat_list.append(sat)
                    kr_list .append(float(y))
        return np.array(sat_list), np.array(kr_list)

    def _make_scanning_interp(self, sg_max):
        """
        Build and cache an interp1d for a given reversal sg_max
        following Killough’s composite-perm formula.
        """
        # clamp reversal
        sg_max = min(sg_max, 1.0)
        # Land residual
        # sgr = sg_max / (1+self.C*sg_max) #Killough
        sgr = sg_max /2 # linear

        # 1) define a fine grid on [sgr … sg_max]
        npts = 100
        s1 = np.linspace(sgr, sg_max, npts)
        # 2) map imbibition curve into [sgr…sg_max]
        #    S* = sgr + (s - sgr)*(1 - sgr)/(sg_max - sgr)
        s_star = self.Sgc_max + (s1 - sgr) * ( self.S_gmax - self.Sgc_max) / (sg_max - sgr)
        kr_inf = np.interp(sg_max, self.sat_dr, self.kr_dr)
        kr1 = np.interp(s_star, self.sat_im, self.kr_im) * kr_inf / np.interp(self.S_gmax, self.sat_dr, self.kr_dr)

        # 3) tail from primary drainage above sg_max
        mask = self.sat_dr > sg_max
        s2 = self.sat_dr[mask]
        kr2 = self.kr_dr[mask]

        sats = np.concatenate([s1, s2])
        krs  = np.concatenate([kr1, kr2])

        interp = interp1d(
            sats, krs,
            kind=self.kind,
            bounds_error=False,
            fill_value=(krs[0], krs[-1])
        )
        self._scan_cache[sg_max] = interp
        return interp

    def evaluate(self, sat, sg_max: float = None):
        """
        sat:    single saturation (float) or 2-tuple for Pc
        sg_max: historical maximum for hysteresis (None → pure drainage)
        """
        # --- Pc branch (unchanged) ---
        if self.phase == 'Pc':
            Sw = sat[0]
            pc = self.kr_interpolator(Sw)*1e-5
            return np.array([pc, 0], dtype=object)

        # --- no hysteresis for wetting or no reversal ---
        if self.phase == 'Aq' or sat >= sg_max:
            return float(self.kr_interpolator(sat))

        # --- gas with hysteresis: use or build scanning interp ---
        interp = self._scan_cache.get(sg_max) or self._make_scanning_interp(sg_max)
        return float(interp(sat))

class PhaseCapillaryHysteresis:
    """
    Killough/Lenhard–Parker hysteresis for capillary pressure without precomputed scanning curves.

    Parameters
    ----------
    corey: object with attributes
        - swc: irreducible water saturation
        - sgc: residual gas saturation
        - Pc_drainage_section: section name for primary drainage Pc
        - Pc_imbibition_section: section name for primary imbibition Pc
    lookup_file: str
        Path to LookupTable.txt containing sections:
        --<section_name>
        <sat> <Pc>
    """
    def __init__(self, corey, phase:str,lookup_file="LookupTable.txt", epsilon=0.1):
        # Load and sort primary drainage Pc
        self.phase = phase
        sat_dr, pc_dr = self.load_lookup_table(lookup_file, corey.Pc_drainage_section)
        idx = np.argsort(sat_dr)
        sat_dr, pc_dr = sat_dr[idx], pc_dr[idx]
        self.Pc_dr_interp = interp1d(
            sat_dr, pc_dr, kind='linear',
            bounds_error=False, fill_value=(pc_dr[0], pc_dr[-1])
        )
        # Maximum gas saturation Sg_max = last point of drainage
        self.Sg_max = sat_dr[-2]
        # Load primary imbibition Pc
        sat_im, pc_im = self.load_lookup_table(lookup_file, corey.Pc_imbibition_section)
        idx = np.argsort(sat_im)
        sat_im, pc_im = sat_im[idx], pc_im[idx]
        self.Pc_im_interp = interp1d(
            sat_im, pc_im, kind='linear',
            bounds_error=False, fill_value=(pc_im[0], pc_im[-1])
        )
        # Residual critical imbibition saturation sgci_max = last saturation where Pc = 0
        zero_im = np.where(np.isclose(pc_im, 0.0))[0]
        self.sgci_max = sat_im[zero_im[-1]] if len(zero_im) > 0 else sat_im[0]

        # Land’s trapping coefficient based on imbibition and max
        self.C = 1.0 / self.sgci_max - 1.0 / self.Sg_max
        self.epsilon = epsilon
    @staticmethod
    def load_lookup_table(filename, section):
        sat_list, pc_list = [], []
        in_section = False
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('--'):
                    in_section = (line[2:].strip() == section)
                    continue
                if in_section:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    s0 = float(parts[0])
                    # invert for nonwetting (gas) tables if needed
                    sat = 1.0 - s0 if 'PC' in section.upper() else s0
                    sat_list.append(sat)
                    pc_list.append(float(parts[1]))
        return np.array(sat_list), np.array(pc_list)

    def evaluate(self, Sw, Sg_max):
        """
        Compute hysteretic capillary pressure P_c at (Sg, Sg_max).

        Returns
        -------
        Pc: float
        """
        # Clamp inputs

        if self.phase =='gas':
            Pc_gas = 0
            return Pc_gas
        Sg = 1 - Sw
        Pc_dr = float(self.Pc_dr_interp(Sg))
        Sg = np.clip(Sg, 0.0, self.Sg_max)

        if Sg_max >= self.Sg_max or Sg >= Sg_max:
            # Primary drainage if beyond reversal

            return Pc_dr*1e-5

        # Compute Land-trapped residual gas saturation at reversal
        Sgr = Sg_max / (1.0 + self.C * Sg_max)
        # Sgr = Sg_max/2
        # Fraction F (Lenhard–Parker)
        num = 1.0 / (1.0 - Sg - (1.0 - Sg_max) + self.epsilon) - 1.0 / self.epsilon
        denom = 1.0 / ((1.0 - Sgr) - (1.0 - Sg_max) + self.epsilon) - 1.0 / self.epsilon
        F = num / denom if abs(denom) > 0 else 0.0
        F = np.clip(F, 0.0, 1.0)
        # Imbibition capillary pressure at current saturation
        Pc_im = float(self.Pc_im_interp(Sg))

        # Composite hysteretic Pc
        return (Pc_dr + F * (Pc_im - Pc_dr))*1e-5

# Example usage:
# corey.Pc_drainage_section = 'Pc_drainage'
# corey.Pc_imbibition_section = 'Pc_imbibation'
# model = PhaseCapillaryHysteresis(corey, "LookupTable.txt", epsilon=0.1)
# Pc_val = model.evaluate(Sg=0.5, Sg_max=0.7)


cmult = 86.4
layer_props = {900001: PorPerm(type='1', poro=0.3, perm=10000, anisotropy=[1, 1, 1], rcond=1 * cmult),
               900002: PorPerm(type='2', poro=0.25, perm=100, anisotropy=[1, 1, 1], rcond=3 * cmult),
               }
