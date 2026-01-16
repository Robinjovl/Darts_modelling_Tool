import numpy as np

from darts.models.cicd_model import CICDModel
from darts.engines import sim_params, ms_well, value_vector

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.viscosity import Fenghour1998
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.set_initial_conditions import LinearAmbientTemperature
from darts.pipes.ramp_up_rate import RampUpRate
from darts.pipes.pipe import Pipe
from darts.pipes.interfacial_tension import IFT_multicomponent_MCM

from nearwellbore import RadialStruct


class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.reservoir.grav_acceleration_for_spe = 9.80665
        self.zero = 1e-10
        self.set_physics()

        self.set_sim_params(first_ts=0.0001/(24*60*60), mult_ts=2, max_ts=2/(24*60*60), tol_newton=1e-3, tol_linear=1e-4,
                            it_newton=10, it_linear=10, newton_type=sim_params.newton_local_chop,
                            coupled_well_res_norm_method=2,
                            runtime=1/24/60, # This runtime will be used when CI test is conducted without the main file
                            )

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        (nr, nz) = (1000, 1)
        (dr, dz) = (0.01, 50)

        poro = np.ones((nr, nz)) * 0.21
        permr = np.ones((nr, nz)) * 100

        permz = permr

        self.well_1_ID = 0.1016
        self.reservoir = RadialStruct(self.timer, nr=nr, nz=nz, dr=dr, dz=dz, poro=poro.flatten(order='F'),
                                      permr=permr.flatten(order='F'), permz=permz.flatten(order='F'),
                                      R0=self.well_1_ID / 2, R1=1000, logspace=True, rcond=259.2, hcap=5250,
                                      top_depth=2000)  # depth is the depth of the top exterface of the reservoir
        self.reservoir.boundary_volumes['yz_plus'] = 1e20

        return

    def set_initial_conditions(self):
        p_init_res = 11   # from the pressure of the perforated segment of the wellbore
        T_init_res = 348.15   # from the temperature of the perforated segment of the wellbore

        input_distribution = {self.physics.vars[0]: p_init_res,
                              "temperature": T_init_res,
                              }
        self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, input_distribution=input_distribution)

        for well in self.reservoir.wells:
            well.init_state = value_vector(self.wells[well.name].initial_conditions.initial_conditions_vector)

        return

    def set_physics(self):
        from dartsflash.libflash import CubicEoS, FlashParams, EoS
        from dartsflash.components import CompData
        from dartsflash.mixtures import DARTSFlash, VL
        components_names = ['CO2']
        phases_names = ['gas', 'LCO2']
        comp_data = CompData(components_names, setprops=True)
        epsilon = self.zero / 10

        """ Define state specification and initialize physics object """
        ph = True
        state_spec = Compositional.StateSpecification.PH if ph else Compositional.StateSpecification.PT
        self.physics = Compositional(components_names, phases_names, self.timer, state_spec=state_spec,
                                     n_points=10000, min_p=1, max_p=500, min_z=0, max_z=1, epsilon_z=epsilon,
                                     min_t=150, max_t=500)

        """ PropertyContainer object and correlations """
        property_container = PropertyContainer(phases_names, components_names, Mw=comp_data.Mw, eps_z=epsilon,
                                               temperature=None, rock_comp=0)

        """ Define flash """
        flash_ev = VL(comp_data)
        flash_ev.set_vl_eos("PR", root_order=[EoS.MAX, EoS.MIN])
        flash_ev.init_flash(flash_type=DARTSFlash.FlashType.PHFlash if ph else DARTSFlash.FlashType.PTFlash)
        property_container.flash_ev = flash_ev

        """ Define phase properties """
        pr = flash_ev.eos["VL"]
        property_container.density_ev = dict([('gas', EoSDensity(eos=pr, Mw=comp_data.Mw, root_flag=EoS.RootFlag.MAX)),
                                              ('LCO2', EoSDensity(eos=pr, Mw=comp_data.Mw, root_flag=EoS.RootFlag.MIN)),
                                              ])
        property_container.enthalpy_ev = dict([('gas', EoSEnthalpy(eos=pr, root_flag=EoS.RootFlag.MAX)),
                                               ('LCO2', EoSEnthalpy(eos=pr, root_flag=EoS.RootFlag.MIN)),
                                               ])
        property_container.viscosity_ev = dict([('gas', Fenghour1998()),
                                                ('LCO2', Fenghour1998()),
                                                ])

        # diff = 8.64e-6
        # property_container.diffusion_ev = dict([('gas', ConstFunc(np.ones(len(components_names)) * diff)),
        #                                         ('LCO2', ConstFunc(np.ones(len(components_names)) * diff)),
        #                                         ('aqueous', ConstFunc(np.ones(len(components_names)) * diff * 1e-3))])

        property_container.conductivity_ev = dict([('gas', ConstFunc(3.5)),
                                                   ('LCO2', ConstFunc(7.)),
                                                   ])

        self.sw_init_res = 0
        swc = self.sw_init_res
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas", swc=swc, sgr=swc, n=1.5)),
                                               ('LCO2', PhaseRelPerm("oil", swc=swc, sgr=swc, n=1.5)),
                                               ])

        property_container.IFT_ev = IFT_multicomponent_MCM(components_names)

        """ Add property region """
        self.physics.add_property_region(property_container)

        property_container.output_props = {}
        property_container.output_props['temperature'] = lambda: property_container.temperature
        for j, ph in enumerate(phases_names):
            property_container.output_props['sat_' + ph] = lambda jj=j: property_container.sat[jj]
            property_container.output_props['rho_' + ph] = lambda jj=j: property_container.dens[jj]
            property_container.output_props['miu_' + ph] = lambda jj=j: property_container.mu[jj]
            property_container.output_props['enth_' + ph] = lambda jj=j: property_container.enthalpy[jj]
            for i, comp in enumerate(components_names):
                property_container.output_props[comp + '_in_' + ph] = lambda jj=j, ii=i: property_container.x[jj, ii]

        return

    def set_wells(self):
        """================================================= Well 1 ================================================="""
        well_1_name = "I1"
        well_1_ms_type = ms_well.MS_Type.DFM
        # Lengths of the well segments are specified here.
        # The lengths of the well segments in front of the reservoir must be equal to the height of the reservoir cells.
        well_1_segments_lengths = 50 * np.ones(41)  # From top to bottom of the wellbore
        well_1_ID = self.well_1_ID
        well_1_inclination_angle = 0.  # in degrees relative to the vertical direction
        well_1_wall_roughness = 2.032e-5
        verbose = True
        well_1_geometry = PipeGeometry(well_1_name, well_1_segments_lengths, well_1_ID, well_1_inclination_angle,
                                       well_1_wall_roughness, verbose)

        #%% Set initial conditions in the pipe using LinearAmbientTemperature
        pipe_head_pressure = 7.81438  # bar
        pipe_head_temperature = 15 + 273.15  # Kelvin
        temp_grad = 0.03  # deg C/meter
        pipe_head_segment_index = 0  # index starts from zero

        initial_conditions_dict = {'phases_names': ['gas'], 'phases_compositions': [[1.]],
                                   'pipe_intervals': [[0, well_1_geometry.pipe_length]]}  # 0 is the beginning of the pipe and pipe_intervals are TVD

        well_1_initial_conditions = LinearAmbientTemperature(well_1_name, well_1_geometry, self.physics,
                                                             pipe_head_pressure, pipe_head_temperature, temp_grad,
                                                             pipe_head_segment_index, initial_conditions_dict)

        #%% Add source/sink terms
        inj_segment_idx = 0
        inflow_or_outflow = "inflow"
        target_inj_rate = 58895.98  # in kmol/day
        ramp_up_period = 3 / (24 * 60)  # in day

        inj_phase_comp = np.array([1.])
        # There is no difference if the phase used in the following line for evaluating enthalpy is either gas
        # or LCO2 because for both the same EoSs are used.
        inj_phase_name = "gas"
        injected_fluid_pressure = 60.
        injected_fluid_temperature = 10 + 273.15
        inj_fluid_props = {"composition": inj_phase_comp, "phase_name": inj_phase_name, "pressure": injected_fluid_pressure, "temperature": injected_fluid_temperature}

        ramp_up_rate = RampUpRate(well_1_name, well_1_geometry, self.physics, self.data_ts.dt_first, inj_segment_idx, inflow_or_outflow, target_inj_rate, ramp_up_period, inj_fluid_props)
        # The following dict will be used in set_rhs_flux and pipe velocity evaluation
        source_sinks = {"RampUpRate1": ramp_up_rate}

        # %% Store well props
        self.wells = {'I1': Pipe('I1', well_1_geometry, self.physics, self.reservoir, well_1_initial_conditions, source_sinks=source_sinks)}

        self.reservoir.add_well(well_1_name, well_1_ms_type, well_geometry=well_1_geometry)

        # Well with a single perforation
        well_1_perforated_segment = well_1_geometry.num_segments
        self.reservoir.add_perforation(well_1_name, res_cell_idx=(1, 1, 1), well_seg_idx=well_1_perforated_segment, well_diameter=well_1_geometry.pipe_ID)

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        inj_comp = self.wells["I1"].source_sinks["RampUpRate1"].inj_fluid_props["composition"]

        # Get updated ramp-up injection rate (rate is updated in pipe.py)
        inj_rate = self.wells["I1"].source_sinks["RampUpRate1"].current_rate

        component_rate = inj_rate * inj_comp

        # Get inj_fluid_molar_enthalpy
        inj_fluid_molar_enthalpy = self.wells["I1"].source_sinks["RampUpRate1"].inj_fluid_props["molar_enthalpy"]
        # Get inj_fluid_molar_potential_energy
        inj_segment_idx = self.wells["I1"].source_sinks["RampUpRate1"].segment_idx
        inj_fluid_specific_potential_energy = self.reservoir.mesh.cell_spe[self.reservoir.mesh.n_res_blocks + inj_segment_idx]
        inj_fluid_molar_potential_energy = inj_fluid_specific_potential_energy * self.physics.property_containers[0].Mw[0]
        inj_fluid_energy = inj_fluid_molar_enthalpy + inj_fluid_molar_potential_energy

        inj_energy_rate = inj_rate * inj_fluid_energy

        inj_rates = np.append(component_rate, inj_energy_rate)

        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        well_head_start_idx = (self.reservoir.mesh.n_res_blocks + inj_segment_idx) * self.physics.n_vars
        rhs_flux[well_head_start_idx:well_head_start_idx+self.physics.n_vars:] = - inj_rates

        return rhs_flux

    def populate_data_for_radial_vtk_output(self, data):
        new_data = {}
        n_cells = self.reservoir.mesh.n_res_blocks
        for prop, val in data.items():
            # populate r-z data to all angles
            new_data[prop] = np.tile(val, self.reservoir.nphi)

        return new_data

    def output_to_vtk(self, ith_step: int = None, output_directory: str = None, output_properties: list = None):
        if output_directory is None:
            output_directory = self.output_folder

        timestep, output_data = self.output.output_properties(output_properties=output_properties, timestep=ith_step)

        data = self.populate_data_for_radial_vtk_output(output_data)
        self.reservoir.output_to_vtk(output_directory=output_directory, data=data, ith_step=ith_step, t=timestep,
                                     prop_names=list(output_data.keys()))

    def get_unknowns_for_radial_vtk_output(self):
        X = np.array(self.physics.engine.X, copy=False)

        # prepare data
        data = {}
        n_cells = self.reservoir.mesh.n_res_blocks
        for i, var in enumerate(self.physics.vars):
            # write r-z data
            data[var] = X[i:self.physics.n_vars * n_cells:self.physics.n_vars]
            # populate r-z data to all angles
            data[var] = np.tile(data[var], self.reservoir.nphi)

        return data
