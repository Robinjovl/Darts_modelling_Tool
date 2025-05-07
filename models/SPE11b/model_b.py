import os
import numpy as np

from dataclasses import dataclass, field
from darts.models.darts_model import DartsModel
from darts.engines import value_vector, index_vector, sim_params, conn_mesh

try:
    from darts.engines import copy_data_to_device, copy_data_to_host, allocate_device_data
except ImportError:
    pass

from darts.engines import well_control_iface
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
    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        self.components = self.specs['components']
        self.nc = len(self.components)

    def set_wells(self):
        if self.specs['RHS'] is False:
            self.reservoir.well_cells = []
            for name, center in self.reservoir.well_centers.items():
                cell_index = self.reservoir.find_cell_index(center)
                self.reservoir.well_cells.append(cell_index)

            for well_nr in range(2):
                k = int(self.reservoir.well_cells[well_nr] / (self.reservoir.nx * self.reservoir.ny) - 1)
                i = int(np.abs(self.reservoir.nx - (self.reservoir.well_cells[well_nr] - k * (self.reservoir.nx * self.reservoir.ny))))
                j = 1

                try:
                    assert k * self.reservoir.nx * self.reservoir.ny + j * self.reservoir.nx + i == self.reservoir.well_cells[well_nr]
                except:
                    print(f"Assertion Failed: (i={i}, j={j}, k={k})")
                    print(f"Computed Index: {k * self.reservoir.nx * self.reservoir.ny + j * self.reservoir.nx + i}")
                    print(f"Expected Index: {self.reservoir.well_cells[well_nr]}")
                    raise

                self.reservoir.add_well("I%d"%well_nr)
                self.reservoir.add_perforation("I%d"%well_nr, cell_index = (i, j, k), well_index=100, well_indexD=100)

            return
        else:
            self.reservoir.set_wells(False)
            return

    def set_well_controls(self):
        if self.specs['RHS'] is False:
            T_inj = 10+273.15
            for i, w in enumerate(self.reservoir.wells):
                if 'I' in w.name:
                    if self.inj_rate[i] == 0:
                        self.physics.set_well_controls(well = w,
                                                       is_control = True,
                                                       control_type = well_control_iface.MASS_RATE,
                                                       is_inj = True,
                                                       target = 0.001,
                                                       phase_name = 'V',
                                                       inj_composition = self.inj_stream[:-1],
                                                       inj_temp = T_inj)
                    else:
                        self.physics.set_well_controls(well = w,
                                                       is_control = True,
                                                       control_type = well_control_iface.MASS_RATE,
                                                       is_inj = True,
                                                       target = self.inj_rate[i],
                                                       phase_name = 'V',
                                                       inj_composition = self.inj_stream[:-1],
                                                       inj_temp = T_inj)

                    print(f'Set well {w.name} to {self.inj_rate[i]} kg/day at 10°C')
            return
        else:
            pass

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
            enth_idx = list(self.physics.property_containers[region].output_props.keys()).index("enthV")

            for i, well_cell in enumerate(self.reservoir.well_cells):
                p_wellcell = self.physics.engine.X[well_cell * nv]
                state = value_vector([p_wellcell] + self.inj_stream)
                values = value_vector(np.zeros(self.physics.n_ops))
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
        else:
            rhs = np.zeros(self.reservoir.mesh.n_res_blocks * self.physics.n_vars)
            return rhs
        
    def set_physics(self, zero: float = 1e-12, temperature: float = None, n_points: int = 10001, diff = 1e-9):
        """Physical properties"""

        # define the Corey parameters for each layer (rock type) according to the technical description of the CSP
        corey = {
            0: Corey(nw=1.5, ng=1.5, swc=0.32, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=1.935314, pcmax=300,
                     c2=1.5),
            1: Corey(nw=1.5, ng=1.5, swc=0.14, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.08655, pcmax=300,
                     c2=1.5),
            2: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.0612, pcmax=300,
                     c2=1.5),
            3: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.038706, pcmax=300,
                     c2=1.5),
            4: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.0306, pcmax=300,
                     c2=1.5),
            5: Corey(nw=1.5, ng=1.5, swc=0.10, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.025602, pcmax=300,
                     c2=1.5),
            6: Corey(nw=1.5, ng=1.5, swc=1e-8, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=1e-2, pcmax=300, c2=1.5)
        }

        self.zero = zero
        self.salinity = 0
        
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
                                     min_z=zero/10, max_z=1-zero/10, min_t=min_t, max_t=max_t,
                                     state_spec = state_spec, 
                                     cache=False)
        self.physics.n_axes_points[0] = 101  # sets OBL points for pressure

        dispersivity = 10.
        self.physics.dispersivity = {}

        for i, (region, corey_params) in enumerate(corey.items()):
            diff_w = 1e-9 * 86400
            diff_g = 2e-8 * 86400
            property_container = PropertyContainer(components_name=self.components, phases_name=phases, Mw=comp_data.Mw,
                                                   min_z=zero / 10, temperature=temperature)

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

            property_container.output_props = {"satV": lambda ii=i: self.physics.property_containers[ii].sat[0],
                                               "rhoV": lambda ii=i: self.physics.property_containers[ii].dens[0],
                                               "rho_mA": lambda ii=i: self.physics.property_containers[ii].dens_m[1],
                                               "enthV": lambda ii=i: self.physics.property_containers[ii].enthalpy[0]}

            for j, phase_name in enumerate(phases):
                for c, component_name in enumerate(self.components):    
                    key = f"x{component_name}" if phase_name == 'Aq' else f"y{component_name}" 
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

        # allocate & transfer dispersivities to device
        if self.platform == 'gpu':
            dispersivity_d = self.physics.engine.get_dispersivity_d()
            allocate_device_data(self.physics.engine.dispersivity, dispersivity_d)
            copy_data_to_device(self.physics.engine.dispersivity, dispersivity_d)

    def set_initial_conditions(self):
        if 1:
            pres_in = 212
            input_depths = [np.amin(self.reservoir.mesh.depth), np.amax(self.reservoir.mesh.depth)]
            
            self.input_distribution = {"pressure": [pres_in, pres_in + input_depths[1] * 0.09775]}
            for i in range(self.nc):
                if self.components[i] == 'H2O':
                    self.input_distribution[self.components[i]] = [1-(self.nc-1)*self.zero, 1-(self.nc-1)*self.zero]
                else:
                    self.input_distribution[self.components[i]] = [self.zero, self.zero]
                
            if self.specs['temperature'] is None:
                self.input_distribution["temperature"] = [313.45, 342.9]
    
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
                primary_specs["H2O"][:] = 1.-self.zero
                # primary_specs["CO2"][:] = self.zero 
                
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
        sg = property_array['satV'][0]
        rhoV = property_array['rhoV'][0]
        rho_m_Aq = property_array['rho_mA'][0]

        self.x_components, self.y_components = [], []
        for component_name in self.physics.components:
            self.x_components.append(property_array['x' + component_name][0])
            self.y_components.append(property_array['y' + component_name][0])
            
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
        for i, component_name in enumerate(component_names):
            # Vapor phase mass contribution
            mass_vapor = phi * V * w_components_vapor[i] * sg * rhoV
            
            # Aqueous phase mass contribution
            mass_aqueous = phi * V * (1 - sg) * self.x_components[i] * rho_m_Aq * Mw[i]
            
            # Total mass
            mass_components[component_name] = mass_vapor + mass_aqueous
    
        return mass_components, mass_vapor, mass_aqueous

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
    
    def run_python_my(self, days=0, restart_dt=0, verbose=1, max_res=1e20):
        runtime = days

        mult_dt = self.params.mult_ts
        max_dt = self.params.max_ts

        # get current engine time
        t = self.physics.engine.t
        # same logic as in engine.run
        if np.fabs(t) < 1e-15:
            dt = self.params.first_ts
        elif restart_dt > 0:
            dt = restart_dt
        else:
            dt = self.params.max_ts

        # evaluate end time
        runtime += t
        ts = 0
        omega = 0.8  # tuning factor between 0 and 1
        if t < 50:
            eta = np.array([0.7, 0.03, 0.7])
            # eta = np.array([0.1, 0.001, 1])
        else:
            eta = np.array([1., 0.05, 1.])

        nc = self.physics.n_vars
        nb = self.reservoir.mesh.n_res_blocks
        max_x = np.zeros(nc)

        while t < runtime:
            xn = np.array(self.physics.engine.Xn[:nb * nc])
            converged = self.run_timestep_python(dt, t, verbose=0, max_res=max_res)

            if converged:
                t += dt
                ts = ts + 1
                x = np.array(self.physics.engine.X[:nb * nc])

                dt_mult_new = mult_dt
                for i in range(nc):
                    max_x[i] = np.max(abs(xn[i::nc] - x[i::nc]))
                    # mult = ((1 + omega) * eta[i]) / (max_x[i] + omega * eta[i])
                    # if mult < dt_mult_new:
                    #   dt_mult_new = mult

                if verbose:
                    print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d\tdX=%s\tmt=%2g"
                          % (ts, t, dt, self.physics.engine.n_newton_last_dt, self.physics.engine.n_linear_last_dt,
                             max_x, dt_mult_new))

                # if self.physics.engine.n_newton_last_dt < 5:
                dt *= dt_mult_new  # new dt multiplier
                # else:
                #     dt /= mult_dt

                if dt > self.params.max_ts:
                    dt = self.params.max_ts

                if t + dt > runtime:
                    dt = runtime - t
            else:
                dt /= mult_dt
                if verbose:
                    print("Cut timestep to %g" % dt)

                if dt < 1e-10:
                    break
        # update current engine time
        self.physics.engine.t = runtime

        print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (
        self.physics.engine.stat.n_timesteps_total, self.physics.engine.stat.n_timesteps_wasted,
        self.physics.engine.stat.n_newton_total, self.physics.engine.stat.n_newton_wasted,
        self.physics.engine.stat.n_linear_total, self.physics.engine.stat.n_linear_wasted))
        self.max_dt = max_dt

    def run_timestep_python(self, dt, t, verbose=0, max_res=1e20):
        max_newt = self.params.max_i_newton
        max_residual = np.zeros(max_newt + 1)
        self.physics.engine.n_linear_last_dt = 0
        well_tolerance_coefficient = 1e1
        self.timer.node['simulation'].start()
        # self.set_top_bot_temp()
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

            self.physics.engine.newton_residual_last_dt = self.physics.engine.calc_newton_residual()

            max_residual[i] = self.physics.engine.newton_residual_last_dt
            counter = 0
            for j in range(i):
                if abs(max_residual[i] - max_residual[j]) / max_residual[i] < 1e-3:
                    counter += 1
            if counter > 2:
                if verbose:
                    print("Stationary point detected!")
                break

            if max_residual[i] / max_residual[0] > max_res:
                if verbose:
                    print("Residual growing over the limit!")
                break

            self.physics.engine.well_residual_last_dt = self.physics.engine.calc_well_residual()
            self.physics.engine.n_newton_last_dt = i
            #  check tolerance if it converges
            if ((self.physics.engine.newton_residual_last_dt < self.params.tolerance_newton and
                 self.physics.engine.well_residual_last_dt < well_tolerance_coefficient * self.params.tolerance_newton)
                    or self.physics.engine.n_newton_last_dt == self.params.max_i_newton):
                if (i > 0):  # min_i_newton
                    break
            r_code = self.physics.engine.solve_linear_equation()
            self.timer.node["newton update"].start()
            self.physics.engine.apply_newton_update(dt)
            self.timer.node["newton update"].stop()
        # End of newton loop
        converged = self.physics.engine.post_newtonloop(dt, t)
        self.timer.node['simulation'].stop()

        return converged

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