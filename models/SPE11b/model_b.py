import numpy as np
import matplotlib.pyplot as plt
import os

from dataclasses import dataclass
from darts.models.darts_model import DartsModel
from darts.engines import value_vector
from math import fabs
try:
    from darts.engines import copy_data_to_device, copy_data_to_host, allocate_device_data
except ImportError:
    pass
from darts.engines import well_control_iface
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.basic import ConstFunc, CapillaryPressure, PhaseRelPerm
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
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

def build_output_dir(spec, base_dir="results"):
    iso_tag = "iso" if spec.get("temperature") is not None else "niso"
    rhs_tag = "rhs" if spec.get("RHS") else "wells"
    disp_tag = "disp" if spec.get("dispersion") else "nodisp"
    components = "-".join(spec.get("components", []))
    nx = spec.get("nx", "nx?")
    nz = spec.get("nz", "nz?")
    device = 'CPU' if spec.get('gpu_device') is False else 'GPU'

    # Construct folder name
    dir_name = f"{iso_tag}__{rhs_tag}__{disp_tag}__{components}__nx{nx}_nz{nz}_{device}"
    return os.path.join(base_dir, dir_name)

########
import pickle
from darts.engines import redirect_darts_output, sim_params
from fluidflower_str_b import FluidFlowerStruct
try:
    from darts.engines import set_gpu_device
except:
    from darts.engines import set_num_threads

# For each of the facies within the SPE11b model we define a set of operators in the physics.
property_regions  = [0, 1, 2, 3, 4, 5, 6]
layers_to_regions = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6}
######

class Model(DartsModel):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        self.components = self.specs['components']
        self.nc = len(self.components)
        self.zero = 1e-10
        self.salinity = 0

        """ set up output directory """
        if self.specs['output_dir'] is None:
            self.specs["output_dir"] = build_output_dir(self.specs)

        if self.specs['post_process'] is None:
            self.output_dir = self.specs['output_dir']
        else:
            self.output_dir = os.path.join(self.specs['output_dir'], self.specs['post_process'])

        # save specs to a .pkl file
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'specs.pkl'), 'wb') as f:
            pickle.dump(self.specs, f)

        redirect_darts_output(os.path.join(self.output_dir, 'model.log'))

        """Define physics"""
        self.set_physics(temperature=specs['temperature'], n_points=1001)
        self.set_sim_params(first_ts=1e-6, mult_ts=2, max_ts=365, tol_linear=1e-2, tol_newton=1e-3,
                            it_linear=50, it_newton=12, newton_type=sim_params.newton_global_chop)
        self.params.newton_params[0] = 0.05
        self.data_ts.eta = np.ones(self.physics.n_vars)
        self.params.nonlinear_norm_type = self.params.L1 #self.params.LINF  # linf if you use m.set_rhs() for injection

        """Define the reservoir and wells """
        well_centers = {
            "I1": [2700.0, 0.0, 300.0],
            "I2": [5100.0, 0.0, 700.0]
        }

        self.reservoir = FluidFlowerStruct(timer=self.timer, layer_properties=layer_props,
                                            layers_to_regions=layers_to_regions,
                                                model_specs=specs, well_centers=well_centers)
        self.nx, self.ny, self.nz = self.reservoir.nx, self.reservoir.ny, self.reservoir.nz
        self.grid = np.meshgrid(np.linspace((8400 / self.nx / 2), 8400 - (8400 / self.nx / 2), self.nx),
                           np.linspace((1200 / self.nz / 2), 1200 - (1200 / self.nz / 2), self.nz))
        self.set_str_boundary_volume_multiplier()  # right and left boundary volume multiplier

        """ Define initial and boundary conditions """
        self.inj_stream = specs['inj_stream'] # define injection stream of the wells
        inj_rate = 3024  # mass rate per well, kg/day
        if specs['1000years'] is False:
            self.inj_rate = [inj_rate, self.zero]
        else:
            self.inj_rate = [0, 0]

        if specs['gpu_device'] is False:
            self.platform = 'cpu'
            set_num_threads(12)
        else:
            self.platform = 'gpu'
            set_gpu_device(0)

    def set_wells(self):
        self.reservoir.set_wells(False)
        return

    def set_well_controls(self):
        if self.specs['RHS'] is False:
            for i, w in enumerate(self.reservoir.wells):
                self.physics.set_well_controls(wctrl = w.control,
                                               control_type = well_control_iface.MASS_RATE,
                                               is_inj = True,
                                               target = self.inj_rate[i],
                                               phase_name = 'V',
                                               inj_composition = self.inj_stream[:-1],
                                               inj_temp = self.inj_stream[-1]
                                               )
                print(f'Set well {w.name} to {self.inj_rate[i]} kg/day with {self.inj_stream[:-1]} {self.components[:-1]} at 10°C...')
            return
        else:
            pass

    def apply_rhs_flux(self, dt: float, t: float):
        if self.specs['RHS'] is False:
            # If the function has not been overloaded, pass
            return
        rhs = np.array(self.physics.engine.RHS, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks * self.physics.n_vars
        rhs[:n_res] += self.set_rhs_flux(t) * dt
        return

    def set_rhs_flux(self, t: float = None):
        if self.specs['RHS'] is True:
            nc = self.physics.nc
            nv = self.physics.n_vars
            nb = self.reservoir.mesh.n_res_blocks
            rhs = np.zeros(nb * nv)

            region = 0
            molar_masses = self.physics.property_containers[region].Mw
            mole_fractions = self.inj_stream[:nc - 1]
            n_comp = np.zeros(nc - 1)
            enth_idx = list(self.physics.property_containers[region].output_props.keys()).index("enthalpy_V")

            for i, well_cell in enumerate(self.reservoir.well_cells):
                p_wellcell = self.physics.engine.X[well_cell * nv]
                state = value_vector([p_wellcell] + self.inj_stream) if self.physics.thermal else value_vector([p_wellcell] + self.inj_stream[:-1])
                values = value_vector(np.zeros(self.physics.n_ops))
                # values_np = np.array(values)
                self.physics.property_itor[self.op_num[well_cell]].evaluate(state, values)
                enthV = values[enth_idx]

                avg_molar_mass = sum(mf * M for mf, M in zip(mole_fractions, molar_masses))

                tot_moles = self.inj_rate[i] / avg_molar_mass

                for comp_idx in range(nc - 1):
                    comp_flux_idx = well_cell * nv + comp_idx  # Index
                    n_comp[comp_idx] = tot_moles * mole_fractions[comp_idx]  # Compute component moles
                    rhs[comp_flux_idx] -= n_comp[comp_idx]  # Update rhs

                if self.physics.thermal:
                    temp_idx = well_cell * nv + nv - 1  # Last equation index (temperature)
                    rhs[temp_idx] -= enthV * np.sum(n_comp)
            return rhs
        # else:
        #     # rhs = np.zeros(self.reservoir.mesh.n_res_blocks * self.physics.n_vars)
        #     # return rhs
        #     pass
        
    def set_physics(self, temperature: float = None, n_points: int = 10001):
        """Physical properties"""

        # define the Corey parameters for each layer (rock type) according to the technical description of the CSP
        corey = {
            0: Corey(nw=1.5, ng=1.5, swc=0.32, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=1.935314, pcmax=300, c2=1.5),
            1: Corey(nw=1.5, ng=1.5, swc=0.14, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.08655, pcmax=300, c2=1.5),
            2: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.0612, pcmax=300, c2=1.5),
            3: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.038706, pcmax=300, c2=1.5),
            4: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.0306, pcmax=300, c2=1.5),
            5: Corey(nw=1.5, ng=1.5, swc=0.10, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.025602, pcmax=300, c2=1.5),
            6: Corey(nw=1.5, ng=1.5, swc=1e-8, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=1e-2, pcmax=300, c2=1.5)
        }

        # Fluid components, ions and solid
        comp_data = CompData(self.components, setprops=True)
        nc, ni = comp_data.nc, comp_data.ni
        # len(components)
        flash_params = FlashParams(comp_data)
        flash_params.add_eos("PR", CubicEoS(comp_data, CubicEoS.PR))
        flash_params.add_eos("AQ", AQEoS(comp_data, {AQEoS.CompType.water: AQEoS.Jager2003,
                                                      AQEoS.CompType.solute: AQEoS.Ziabakhsh2012,
                                                      AQEoS.CompType.ion: AQEoS.Jager2003
                                                      }))
        pr = flash_params.eos_params["PR"].eos
        aq = flash_params.eos_params["AQ"].eos
        flash_params.eos_order = ["PR", "AQ"]
        phases = ["V", "Aq"]

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3

        if temperature is None:  # if None, then thermal=True
            thermal = True
            state_spec = Compositional.StateSpecification.PT
        else:
            thermal = False
            state_spec = Compositional.StateSpecification.P
            
        pres_in = 210 # (pressure at depth of well 1 will be 300 bar)
        min_t = 273.15 if temperature is None else None
        max_t = 373.15 if temperature is None else None
        self.physics = Compositional(self.components, phases, timer=self.timer,
                                     n_points=n_points, min_p=200, max_p=450,
                                     min_z=self.zero/10, max_z=1-self.zero/10, min_t=min_t, max_t=max_t,
                                     state_spec = state_spec, 
                                     cache=False)
        self.physics.n_axes_points[0] = 1001  # sets OBL points for pressure

        dispersivity = 10.
        self.physics.dispersivity = {}

        for i, (region, corey_params) in enumerate(corey.items()):
            diff_w = 1e-9 * 86400
            diff_g = 2e-8 * 86400
            property_container = PropertyContainer(components_name=self.components, phases_name=phases, Mw=comp_data.Mw,
                                                   min_z=self.zero / 10, temperature=temperature)

            # property_container.flash_ev = ConstantK(nc=2, ki=[0.001, 100])
            property_container.flash_ev = NegativeFlash(flash_params, ["PR", "AQ"], [InitialGuess.Henry_VA])
            property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                                  ('Aq', Garcia2001(self.components)), ])
            property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                    ('Aq', Islam2012(self.components)), ])
            property_container.diffusion_ev = dict([('V', ConstFunc(np.ones(nc) * diff_g)),
                                                    ('Aq', ConstFunc(np.ones(nc) * diff_w))])
            property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr)),
                                                   ('Aq', EoSEnthalpy(eos=aq)), ])
            property_container.conductivity_ev = dict([('V', ConstFunc(8.4)),
                                                       ('Aq', ConstFunc(170.)),])
            property_container.rel_perm_ev = dict([('V', ModBrooksCorey(corey_params, 'V')),
                                                   ('Aq', ModBrooksCorey(corey_params, 'Aq'))])
            property_container.capillary_pressure_ev = ModCapillaryPressure(corey_params)

            self.physics.add_property_region(property_container, i)

            property_container.output_props = {"sat_V": lambda ii=i: self.physics.property_containers[ii].sat[0],
                                               "dens_V": lambda ii=i: self.physics.property_containers[ii].dens[0],
                                               "densm_Aq": lambda ii=i: self.physics.property_containers[ii].dens_m[1],
                                               "enthalpy_V": lambda ii=i: self.physics.property_containers[ii].enthalpy[0]}

            for j, phase_name in enumerate(phases):
                for c, component_name in enumerate(self.components):
                    key = f"x_{phase_name}_{component_name}" #if phase_name == 'Aq' else f"y{component_name}"
                    property_container.output_props[key] = lambda ii=i, jj=j, cc=c: self.physics.property_containers[ii].x[jj, cc]
            
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

        # self.physics.engine.is_fickian_energy_transport_on = False
        print('Fickian energy is', self.physics.engine.is_fickian_energy_transport_on)

        # allocate & transfer dispersivities to device
        if self.platform == 'gpu':
            dispersivity_d = self.physics.engine.get_dispersivity_d()
            allocate_device_data(self.physics.engine.dispersivity, dispersivity_d)
            copy_data_to_device(self.physics.engine.dispersivity, dispersivity_d)

    def set_initial_conditions(self):
        if 0:
            pres_in = 212
            input_depths = [np.amin(self.reservoir.mesh.depth), np.amax(self.reservoir.mesh.depth)]
            
            self.input_distribution = {"pressure": [pres_in, pres_in + input_depths[1] * 0.09775]}
            for i in range(self.nc):
                if self.components[i] == 'H2O':
                    self.input_distribution[self.components[i]] = [1-(self.nc-1)*self.zero, 1-(self.nc-1)*self.zero]
                else:
                    self.input_distribution[self.components[i]] = [self.zero, self.zero]
                
            if self.specs['temperature'] is None:
                self.input_distribution["temperature"] = [313.4, 342.9]
    
            self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh, 
                                                                 input_depth=input_depths,
                                                                 input_distribution=self.input_distribution)
        else:
            self.temp = lambda depth: 273.15 + 70. - depth * 0.025
            
            depths = np.asarray(self.reservoir.mesh.depth)
            min_depth = np.min(depths)
            max_depth = np.max(depths)
            nb = 100
            depths = np.linspace(min_depth, max_depth, nb)

            from darts.physics.super.initialize import Initialize
            init = Initialize(self.physics, aq_idx=0, h2o_idx=0)

            # Known conditions at well I1
            known_depth = self.reservoir.well_centers["I1"][2]
            known_idx = np.argmin(np.abs(depths-known_depth))
            depths[known_idx] = known_depth
            pres_I1 = 300.
            temp_I1 = self.temp(max_depth-known_depth)  # depths in grid have z=0 at the bottom

            nc = len(self.components)
            primary_specs = {comp: np.ones(nb) * np.nan for comp in self.components}
            secondary_specs = {}
            known_specs = {}

            if nc == 2:
                # H2O-CO2, initially pure brine
                # need 1 specification: H2O = 1-zero
                # primary_specs["H2O"][:] = 1.-self.zero
                primary_specs["CO2"][:] = self.zero
                
            else:
                # H2O-CO2-C1, initially pure brine
                # need 2 specifications: H2O = 1-(nc-1)*zero, CO2 = zero
                primary_specs["H2S"][:] = self.zero
                primary_specs["CO2"][:] = self.zero
                
            if self.salinity:
                # + ions, need extra specification for ion molality
                # H2O cannot be specified because of salinity, last component instead
                mol = np.ones(nb) * self.salinity
                secondary_specs.update({'m' + str(nc): mol})
                known_specs.update({'m' + str(nc): mol[known_idx]})
                primary_specs["H2O"][:] = None
                primary_specs[self.components[-1]][:] = self.zero

            # Solve boundary state
            X0 = ([pres_I1, 0.98] +  # pressure, H2O
                  ([self.zero] if nc > 2 else []) +  # CO2
                  ([1. - 0.98 - mol[known_idx] * 0.98 / 55.509] if self.salinity else []) +  # last component if ions
                  ([temp_I1] if init.thermal else []))  # temperature
            X0 = init.solve_state(X0, primary_specs={'pressure': pres_I1, 
                                                     'temperature': temp_I1 if init.thermal else None} |
                                                    {comp: primary_specs[comp][known_idx] for comp in self.components},
                                  secondary_specs=known_specs)
            boundary_state = {v: X0[i] for i, v in enumerate(self.physics.vars)}

            # Solve vertical equilibrium
            X = init.solve(depth_bottom=max_depth, depth_top=min_depth, depth_known=known_depth, nb=nb,
                            primary_specs=primary_specs, secondary_specs=secondary_specs,
                            boundary_state=boundary_state, dTdh=0.025).reshape((nb, self.physics.n_vars))

            self.physics.set_initial_conditions_from_depth_table(mesh = self.reservoir.mesh, 
                                                                 input_depth = init.depths,
                                                                 input_distribution = {v: X[:, i] for i, v in enumerate(self.physics.vars)})

    def set_str_boundary_volume_multiplier(self):
        self.reservoir.boundary_volumes['yz_minus'] = 5e4 * (1200 / self.reservoir.nz)
        self.reservoir.boundary_volumes['yz_plus']  = 5e4 * (1200 / self.reservoir.nz)
        return

    def get_mass_components(self, property_array):
        component_names = self.physics.property_containers[0].components_name
        Mw = np.array(self.physics.property_containers[0].Mw).reshape(-1, 1)

        # Extract properties from property_array
        sg = property_array['sat_V'][0]
        rhoV = property_array['dens_V'][0]
        rho_m_Aq = property_array['densm_Aq'][0]

        self.x_components, self.y_components = [], []
        # for phase_name in self.physics.phases:
        for component_name in self.physics.components:
            self.x_components.append(property_array[f'x_Aq_{component_name}'][0])
            self.y_components.append(property_array[f'x_V_{component_name}'][0])
            # self.x_components.append(property_array['x' + component_name][0])
            # self.y_components.append(property_array['y' + component_name][0])
            
        self.x_components, self.y_components = np.array(self.x_components), np.array(self.y_components)
        
        # Compute molecular weight of the aqueous phase
        MWAq = np.sum(self.y_components[1:, :] * Mw[1:], axis = 0 ) + (1 - np.sum(self.y_components[1:, :], axis = 0)) * Mw[0]
    
        # Mass fractions in vapor phase
        w_components_vapor = (self.y_components * Mw) / MWAq
    
        # Pore volume 
        V = np.array(self.reservoir.mesh.volume, copy=False)[:self.reservoir.n]
        phi = np.array(self.reservoir.mesh.poro, copy=False)[:self.reservoir.n]

        # Calculate total mass for each component
        mass_components = {}
        mass_aqueous = {}
        mass_vapor = {}
        for i, component_name in enumerate(component_names):
            # Vapor phase mass contribution
            mass_vapor[component_name] = phi * V * w_components_vapor[i] * sg * rhoV
            
            # Aqueous phase mass contribution
            mass_aqueous[component_name] = phi * V * (1 - sg) * self.x_components[i] * rho_m_Aq * Mw[i]
            
            # Total mass
            mass_components[component_name] = mass_vapor[component_name] + mass_aqueous[component_name]
    
        return mass_components, mass_vapor, mass_aqueous

    def set_top_bot_temp(self):
        nv = self.physics.n_vars
        for bot_cell in self.reservoir.bot_cells:
            # T = 70 - 0.025 * z  - origin at bottom
            T_spec_bot = 273.15 + 70 - self.reservoir.centroids[bot_cell, 2] * 0.025
            target_cell = bot_cell*nv+nv-1
            self.physics.engine.X[target_cell] = T_spec_bot

        for top_cell in self.reservoir.top_cells:
            # T = 70 - 0.025 * z  - origin at bottom
            T_spec_top = 273.15 + 70 - self.reservoir.centroids[top_cell, 2] * 0.025
            target_cell = top_cell*nv+nv-1
            self.physics.engine.X[target_cell] = T_spec_top
        return

    def my_output_to_vtk(self, property_array, timesteps, ith_step: int = None, output_directory: str = None, ):
        self.timer.start(); self.timer.node["vtk_output"].start()

        # Set default output directory
        if output_directory is None:
            output_directory = os.path.join(self.output_folder, 'vtk_files')
        os.makedirs(output_directory, exist_ok=True)

        # units to prop names
        prop_names = {}
        for i, name in enumerate(property_array.keys()):
            prop_names[name] = name
            
        for t, time in enumerate(timesteps):
            data = np.zeros((len(property_array), self.reservoir.mesh.n_res_blocks))
            for i, name in enumerate(property_array.keys()):
                data[i, :] = property_array[name][t]

            if ith_step is None:
                self.reservoir.output_to_vtk(t, time, output_directory, prop_names, data)
            else:
                self.reservoir.output_to_vtk(ith_step, time, output_directory, prop_names, data)

        self.timer.node["vtk_output"].stop(); self.timer.stop()

    def run(self, days: float = None, restart_dt: float = 0., save_well_data: bool = True, save_well_data_after_run: bool = False,
            save_reservoir_data: bool = True, verbose: bool = True):

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
                dt_mult_new = data_ts.dt_mult # current multiplier
                for i in range(nc):
                    # propose new multiplier based on change of solution
                    max_dx[i] = np.max(abs(xn[i::nc] - x[i::nc]))
                    mult = ((1 + omega) * data_ts.eta[i]) / (max_dx[i] + omega * data_ts.eta[i])
                    if mult < dt_mult_new: # if the proposed multiplier is smaller than the specified maximum timestep
                        dt_mult_new = mult # the new multiplier = proposed multiplier

                if verbose:
                    print("# %d \tT = %.3g\tDT = %.2g\tNI = %d\tLI=%d\tDT_MULT=%3.3g\tmax_dX=%4s"
                          % (ts, t, dt, self.physics.engine.n_newton_last_dt, self.physics.engine.n_linear_last_dt,
                             dt_mult_new, np.round(max_dx, 3)))

                dt = min(dt * dt_mult_new, data_ts.dt_max) # define the new timestep according to the new multiplier

                if np.fabs(t + dt - stop_time) < data_ts.dt_min:
                    dt = stop_time - t

                if t + dt > stop_time:
                    dt = stop_time - t
                else:
                    self.prev_dt = dt

                # save well data at every converged time step
                if save_well_data and save_well_data_after_run is False:
                    self.output.save_data_to_h5(kind='well')

            else:
                dt /= data_ts.dt_mult
                if verbose:
                    print("Cut timestep to %2.10f" % dt)
                assert dt > data_ts.dt_min, ('Stop simulation. Reason: reached min. timestep '
                                             + str(data_ts.dt_min) + ' dt=' + str(dt))

        # update current engine time
        self.physics.engine.t = stop_time

        # save well data after run
        if save_well_data and save_well_data_after_run is True:
            self.output.save_data_to_h5(kind='well')

        # save solution vector
        if save_reservoir_data:
            self.output.save_data_to_h5(kind='reservoir')

        if verbose:
            print("TS = %d(%d), NI = %d(%d), LI = %d(%d)"
                  % (self.physics.engine.stat.n_timesteps_total, self.physics.engine.stat.n_timesteps_wasted,
                     self.physics.engine.stat.n_newton_total, self.physics.engine.stat.n_newton_wasted,
                     self.physics.engine.stat.n_linear_total, self.physics.engine.stat.n_linear_wasted))

        return 0

    def run_timestep(self, dt: float, t: float, verbose: bool = True):
        max_newt = self.data_ts.newton_max_iter
        max_residual = np.zeros(max_newt + 1)
        self.physics.engine.n_linear_last_dt = 0
        self.data_ts.newton_tol_wel_mult = 1e2

        self.timer.node['simulation'].start()
        for i in range(max_newt + 1):
            if self.platform == 'gpu':
                copy_data_to_host(self.physics.engine.X, self.physics.engine.get_X_d())

            if self.physics.thermal:
                self.set_top_bot_temp()

            if self.platform == 'gpu':
                copy_data_to_device(self.physics.engine.X, self.physics.engine.get_X_d())

            self.physics.engine.assemble_linear_system(dt)
            self.apply_rhs_flux(dt, t)

            if self.platform == 'gpu':
                copy_data_to_device(self.physics.engine.RHS, self.physics.engine.get_RHS_d())

            self.physics.engine.newton_residual_last_dt = self.physics.engine.calc_newton_residual()  # calc norm of residual

            max_residual[i] = self.physics.engine.newton_residual_last_dt
            counter = 0
            for j in range(i):
                if abs(max_residual[i] - max_residual[j]) / max_residual[i] < self.data_ts.newton_tol_stationary:
                    counter += 1
            if counter > 2:
                if verbose:
                    print("Stationary point detected!")
                break

            self.physics.engine.well_residual_last_dt = self.physics.engine.calc_well_residual()
            self.physics.engine.n_newton_last_dt = i
            #  check tolerance if it converges
            if ((self.physics.engine.newton_residual_last_dt < self.data_ts.newton_tol and
                 self.physics.engine.well_residual_last_dt < self.data_ts.newton_tol * self.data_ts.newton_tol_wel_mult) or
                    self.physics.engine.n_newton_last_dt == max_newt):
                if i > 0:  # min_i_newton
                    break

            # line search
            if self.data_ts.line_search and i > 0 and residual_history[-1][0] > 0.9 * residual_history[-2][0]:
                coef = np.array([0.0, 1.0])
                history = np.array([residual_history[-2], residual_history[-1]])
                residual_history[-1] = self.line_search(dt, t, coef, history, verbose)
                max_residual[i] = residual_history[-1][0]

                # check stationary point after line search
                counter = 0
                for j in range(i):
                    if abs(max_residual[i] - max_residual[j]) / max_residual[i] < self.data_ts.newton_tol_stationary:
                        counter += 1
                if counter > 2:
                    if verbose:
                        print("Stationary point detected!")
                    break
            else:
                r_code = self.physics.engine.solve_linear_equation()
                self.timer.node["newton update"].start()
                self.physics.engine.apply_newton_update(dt)
                self.timer.node["newton update"].stop()
        # End of newton loop
        converged = self.physics.engine.post_newtonloop(dt, t)

        self.timer.node['simulation'].stop()
        return converged

    def set_well_rhs(self, Dt, inj_rate, event1, event2):
        if self.physics.engine.t >= 25 * Dt and self.physics.engine.t < 50 * Dt and event1:
            print('At 25 years, start injecting in the second well')
            self.inj_rate = [inj_rate, inj_rate]
            event1 = False
        elif self.physics.engine.t >= 50 * Dt and event2:
            print('At 50 years, stop injection for both wells')
            self.inj_rate = [self.zero, self.zero]
            self.specs['check_rates'] = False  # after injection stop checking rates
            event2 = False

        return event1, event2

    def set_well_rates(self, Dt, inj_rate, event1, event2):
        if self.physics.engine.t >= 25 * Dt and self.physics.engine.t < 50 * Dt and event1:
            print('At 25 years, start injecting in the second well')
            self.inj_rate = [inj_rate, inj_rate]
            self.physics.set_well_controls(wctrl=self.reservoir.wells[1].control,
                                            control_type=well_control_iface.MASS_RATE,
                                            is_inj=True,
                                            target=self.inj_rate[1],
                                            phase_name='V',
                                            inj_composition=self.inj_stream[:-1],
                                            inj_temp=self.inj_stream[-1]
                                        )
            event1 = False

        elif self.physics.engine.t >= 50 * Dt and event2:
            print('At 50 years, stop injection in both wells')
            self.inj_rate = [self.zero, self.zero]
            for i in range(2):
                self.physics.set_well_controls(wctrl=self.reservoir.wells[i].control,
                                            control_type=well_control_iface.MASS_RATE,
                                            is_inj=True,
                                            target=self.inj_rate[i],
                                            phase_name='V',
                                            inj_composition=self.inj_stream[:-1],
                                            inj_temp=self.inj_stream[-1]
                                            )
            self.specs['check_rates'] = False  # after injection stop checking rates
            event2 = False

        return event1, event2

    def a_quiver_plot(self, ids_cells, centroids):
        # Get coordinates of start and end points
        start = centroids[ids_cells[:, 0]]  # shape (n_edges, 3)
        end = centroids[ids_cells[:, 1]]

        # Choose 2D plane: x-z
        x_start, z_start = start[:, 0], start[:, 2]
        x_dir = end[:, 0] - start[:, 0]
        z_dir = end[:, 2] - start[:, 2]

        # Plot the quiver plot
        plt.figure(figsize=(10, 6))
        plt.quiver(
            x_start, z_start,  # origins
            x_dir, z_dir,  # directions
            angles='xy',
            scale_units='xy',
            scale=1,
            color='blue',
            width=0.0025
        )
        # plt.xlabel("X")
        # plt.ylabel("Z")
        # plt.title("Flow Directions (x–z plane)")
        # # plt.axis("equal")
        # plt.grid(True)
        # plt.show()

    def map_mesh_faces(self):
        """
        Identifies directional face connections (NS, SN, EW, WE) between mesh blocks.
        Groups these face pairs by direction and stores the indices and associated cell pairs.
        Returns those indices in a structured form for further use (e.g., plots).
        """
        nx, ny, nz = self.reservoir.nx, self.reservoir.ny, self.reservoir.nz
        cell_m = np.asarray(self.reservoir.mesh.block_m)
        cell_p = np.asarray(self.reservoir.mesh.block_p)
        cell_ = np.vstack([cell_m.flatten(), cell_p.flatten()]).T
        self.centroids = self.reservoir.discretizer.centroids_all_cells

        # z-direction
        x0 = np.unique(self.centroids[:, 0])

        ids_SN = np.zeros(len(x0) * nz, dtype=np.int32)
        ids_cells_SN = np.zeros((len(x0) * nz, 2), dtype=np.int32)

        ids_NS = np.zeros(len(x0) * nz, dtype=np.int32)
        ids_cells_NS = np.zeros((len(x0) * nz, 2), dtype=np.int32)

        for i, x in enumerate(x0):
            ids = np.where(np.logical_and(
                np.logical_and(np.fabs(self.centroids[cell_m, 0] - x) < 1.e-6,
                               np.fabs(self.centroids[cell_p, 0] - x) < 1.e-6),
                self.centroids[cell_m, 2] < self.centroids[cell_p, 2]))[0]
            ids_SN[i * (nz - 1):(i + 1) * (nz - 1)] = ids
            ids_cells_SN[i * (nz - 1):(i + 1) * (nz - 1)] = cell_[ids]

            ids = np.where(np.logical_and(
                np.logical_and(np.fabs(self.centroids[cell_m, 0] - x) < 1.e-6,
                               np.fabs(self.centroids[cell_p, 0] - x) < 1.e-6),
                self.centroids[cell_m, 2] > self.centroids[cell_p, 2]))[0]
            ids_NS[i * (nz - 1):(i + 1) * (nz - 1)] = ids
            ids_cells_NS[i * (nz - 1):(i + 1) * (nz - 1)] = cell_[ids]
        # a_quiver_plot(ids_cells_NS, self.centroids); plt.title('NS'); plt.show()
        # a_quiver_plot(ids_cells_SN, self.centroids); plt.title('SN'); plt.show()

        # x-direction
        z0 = np.unique(self.centroids[:, 2])

        ids_EW = np.zeros(len(z0) * nx, dtype=np.int32)
        ids_cells_EW = np.zeros((len(z0) * nx, 2), dtype=np.int32)

        ids_WE = np.zeros(len(z0) * nx, dtype=np.int32)
        ids_cells_WE = np.zeros((len(z0) * nx, 2), dtype=np.int32)

        for i, z in enumerate(z0):
            ids = np.where(np.logical_and(
                np.logical_and(np.fabs(self.centroids[cell_m, 2] - z) < 1.e-6,
                               np.fabs(self.centroids[cell_p, 2] - z) < 1.e-6),
                self.centroids[cell_m, 0] > self.centroids[cell_p, 0]))[0]
            ids_EW[i * (nx - 1):(i + 1) * (nx - 1)] = ids
            ids_cells_EW[i * (nx - 1):(i + 1) * (nx - 1)] = cell_[ids]

            ids = np.where(np.logical_and(
                np.logical_and(np.fabs(self.centroids[cell_m, 2] - z) < 1.e-6,
                               np.fabs(self.centroids[cell_p, 2] - z) < 1.e-6),
                self.centroids[cell_m, 0] < self.centroids[cell_p, 0]))[0]
            ids_WE[i * (nx - 1):(i + 1) * (nx - 1)] = ids
            ids_cells_WE[i * (nx - 1):(i + 1) * (nx - 1)] = cell_[ids]
        # a_quiver_plot(ids_cells_EW, self.centroids); plt.title('EW'); plt.show()
        # a_quiver_plot(ids_cells_WE, self.centroids); plt.title('WE'); plt.show()

        self.ids_list = {'NS': ids_NS, 'SN': ids_SN, 'EW': ids_EW, 'WE': ids_WE}
        self.ids_cells_list = {'NS': ids_cells_NS, 'SN': ids_cells_SN, 'EW': ids_cells_EW, 'WE': ids_cells_WE}

        return self.ids_list, self.ids_cells_list

    def plot_fluxes(self, property_array, time_vector, ts):
        nx, ny, nz = self.reservoir.nx, self.reservoir.ny, self.reservoir.nz
        figure_folder = os.path.join(self.output_folder, 'figures', 'fluxes')
        os.makedirs(figure_folder, exist_ok=True)

        for id_key in ['SN', 'EW']:
            for phase_idx, phase_name in enumerate(self.physics.phases):
                for comp_idx, comp_name in enumerate(self.components):

                    if 0:
                        # Get coordinates of start and end points
                        start = self.centroids[ids_cells_list[id_key][:, 0]]  # shape (n_edges, 3)
                        end = self.centroids[ids_cells_list[id_key][:, 1]]

                        # Choose 2D plane: x-z
                        x_start, z_start = start[:, 0], start[:, 2]
                        if 'N' in id_key:
                            x_dir = end[:, 0] - start[:, 0]
                            z_dir = field
                        else:
                            x_dir = field
                            z_dir = end[:, 2] - start[:, 2]

                        # Plot the quiver plot
                        plt.figure(figsize=(10, 6), dpi=200)
                        plt.quiver(
                            x_start, z_start,  # origins
                            x_dir, z_dir,  # directions
                            angles='xy',
                            scale_units='xy',
                            scale=1,
                            color='blue',
                            width=0.0025
                        )
                        # plt.xlabel("X")
                        # plt.ylabel("Z")
                        plt.title(
                            f"Diffusion Flux for {component_name} in {phase_name}, in the {id_key} direction")
                        plt.grid(True)
                        plt.show()

                    else:

                        face_centroids_x = (self.centroids[self.ids_cells_list[id_key][:, 0], 0] + self.centroids[
                            self.ids_cells_list[id_key][:, 1], 0]) / 2
                        face_centroids_z = (self.centroids[self.ids_cells_list[id_key][:, 0], 2] + self.centroids[
                            self.ids_cells_list[id_key][:, 1], 2]) / 2

                        # Determine shape and indexing logic
                        if id_key in {'SN', 'NS'}:
                            shape = (nz - 1, nx)
                            index_fn = lambda i, j: i + j * (nz - 1)
                        elif id_key in {'EW', 'WE'}:
                            shape = (nz, nx - 1)
                            index_fn = lambda i, j: j + i * (nx - 1)
                        else:
                            raise ValueError(f"Unsupported id_key: {id_key}")

                        temp_x, temp_z = np.zeros(shape), np.zeros(shape)
                        diff_flux = np.zeros(shape)
                        darcy_flux = np.zeros(shape)
                        disp_flux = np.zeros(shape)

                        for i in range(shape[0]):
                            for j in range(shape[1]):
                                idx = index_fn(i, j)
                                temp_x[i, j] = face_centroids_x[idx]
                                temp_z[i, j] = face_centroids_z[idx]
                                diff_flux[i, j] = property_array[f'diff_fluxes_{phase_name}_{comp_name}_{id_key}'][idx]
                                darcy_flux[i, j] = property_array[f'darcy_fluxes_{phase_name}_{comp_name}_{id_key}'][idx]
                                disp_flux[i, j] = property_array[f'disp_fluxes_{phase_name}_{comp_name}_{id_key}'][idx]

                        plt.figure(dpi = 100, figsize=(8, 8))
                        plt.suptitle(f"{comp_name}, {phase_name}, in the {id_key} direction @ {time_vector[0]} days")

                        plt.subplot(4, 1, 1)
                        # pc1 = plt.scatter(face_centroids_x, face_centroids_z, c=property_array[f'diff_fluxes_{phase_name}_{comp_idx}_{id_key}'], s=1, cmap='coolwarm')
                        pc1 = plt.pcolor(temp_x, temp_z, diff_flux, vmin = -np.max(np.abs(diff_flux)), vmax = np.max(np.abs(diff_flux)), cmap='coolwarm')
                        plt.colorbar(pc1, aspect = 10, label="Diffusion Flux")
                        plt.ylim(0, 1200); plt.xlim(0, 8400)
                        plt.ylabel('z [m]')

                        plt.subplot(4, 1, 2)
                        # pc2 = plt.scatter(self.centroids[:, 0], self.centroids[:, 2], c = property_array[f'vel_{phase_name}'], s=1, cmap='coolwarm')
                        velocity = property_array[f'vel_{phase_name}'].reshape(nz, nx)
                        pc2 = plt.pcolor(self.centroids[:, 0].reshape(nz, nx),
                                         self.centroids[:, 2].reshape(nz, nx),
                                         velocity,
                                         # vmin=-np.max(np.abs(velocity)), vmax=np.max(np.abs(velocity)),
                                         cmap='coolwarm')
                        plt.colorbar(pc2, aspect = 10, label="Velocities")
                        plt.ylim(0, 1200); plt.xlim(0, 8400)
                        plt.ylabel('z [m]')

                        plt.subplot(4, 1, 3)
                        # pc3 = plt.scatter(face_centroids_x, face_centroids_z, c = property_array[f'disp_fluxes_{phase_name}_{comp_idx}_{id_key}'], s=1, cmap='coolwarm')
                        pc3 = plt.pcolor(temp_x, temp_z, disp_flux, vmin = -np.max(np.abs(disp_flux)), vmax = np.max(np.abs(disp_flux)), cmap='coolwarm')
                        plt.colorbar(pc3, aspect = 10, label="Disp Flux")
                        plt.ylim(0, 1200); plt.xlim(0, 8400)
                        plt.ylabel('z [m]')

                        plt.subplot(4, 1, 4)
                        # pc4 = plt.scatter(face_centroids_x, face_centroids_z, c = property_array[f'darcy_fluxes_{phase_name}_{comp_idx}_{id_key}'] , s=1, cmap='coolwarm')
                        pc4 = plt.pcolor(temp_x, temp_z, darcy_flux, vmin = -np.max(np.abs(darcy_flux)), vmax = np.max(np.abs(darcy_flux)), cmap='coolwarm')
                        plt.colorbar(pc4, aspect = 10, label="Darcy Flux")
                        plt.ylim(0, 1200); plt.xlim(0, 8400)
                        plt.xlabel('x [m]');
                        plt.ylabel('z [m]')
                        plt.tight_layout()
                        plt.savefig(os.path.join(figure_folder, f'fluxes_{id_key}_{phase_name}_{comp_name}_at_ts_{ts}.png'))
                        plt.close()

    def plot_properties(self, property_array, time_vector, ts):
        for i, name in enumerate(property_array.keys()):
            if 'fluxes' not in name:
                plt.figure(figsize=(10, 2))
                plt.title(f'{name} @ year {time_vector[0] / 365}')
                c = plt.pcolor(self.grid[0], self.grid[1], property_array[name][0].reshape(self.nz, self.nx), cmap='cividis')
                try:
                    plt.colorbar(c, aspect=10, label=self.output.variable_units[name])
                except:
                    pass
                plt.xlabel('x [m]');
                plt.ylabel('z [m]')
                fig_dir = os.path.join(self.output_folder, 'figures', f'{name}')
                os.makedirs(fig_dir, exist_ok=True)
                fig_path = os.path.join(fig_dir, f'{name}_ts_{ts}.png')
                plt.savefig(fig_path, bbox_inches='tight')
                plt.close()
        return

    def plot_reservoir(self):
        nx = self.reservoir.nx
        nz = self.reservoir.nz
        nb = self.reservoir.n
        n_vars = self.physics.n_vars
        vars = self.physics.vars
        self.grid = np.meshgrid(np.linspace(0, 8400, nx), np.linspace(0, 1200, nz))
        poro = self.reservoir.global_data['poro']
        op_num = np.array(self.reservoir.mesh.op_num)[:self.reservoir.n] + 1

        plt.figure(dpi=100, figsize=(10, 2))
        plt.title('Facies')
        c = plt.pcolor(self.grid[0], self.grid[1], op_num.reshape(nz, nx), cmap='jet', vmin=min(op_num), vmax=max(op_num))
        plt.colorbar(c, ticks=np.arange(1, 8))
        plt.xlabel('x [m]');
        plt.ylabel('z [m]')
        # centroids = m.reservoir.discretizer.centroids_all_cells
        centroids = self.reservoir.centroids
        plt.scatter(centroids[self.reservoir.well_cells[0], 0], centroids[self.reservoir.well_cells[0], 2], marker='x', c='r', s=5)
        plt.scatter(centroids[self.reservoir.well_cells[1], 0], centroids[self.reservoir.well_cells[1], 2], marker='x', c='r', s=5)
        plt.savefig(os.path.join(self.output_folder, f'op_num.png'), bbox_inches='tight')
        plt.close()

        solution_vector = np.array(self.physics.engine.X)
        for i, name in enumerate(vars):
            plt.figure(figsize=(10, 2))
            plt.title(name)
            c = plt.pcolor(self.grid[0], self.grid[1], np.round(solution_vector[i::n_vars][:nb], 2).reshape(nz, nx), cmap='jet')
            plt.colorbar(c, aspect=10)
            plt.xlabel('x [m]');
            plt.ylabel('z [m]')
            plt.savefig(os.path.join(self.output_folder, f'initial_conditions_{name}.png'), bbox_inches='tight')
            plt.close()

    def print_darts(self):
        print(r"""
        ------------------------------------------------------
         _____                 _____     _______     _____
        |  __ \       /\      |  __ \   |__   __|  /  ____|
        | |  | |     /  \     | |__) |     | |     | (___
        | |  | |    / /\ \    |  _  /      | |      \___ \
        | |__| |   / ____ \   | | \ \      | |      ____) |
        |_____/   /_/    \_\  |_|  \_\     |_|     |_____/
        ------------------------------------------------------
        """)


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