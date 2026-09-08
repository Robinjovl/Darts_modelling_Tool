import numpy as np

from darts.models.darts_model import DartsModel
from darts.pipes.viz.plot_live import DartsModelWithLivePlots
from darts.engines import sim_params, ms_well, value_vector, well_control_iface
from darts.nonlinear_solvers import NewtonSolver, ChopSpec

from darts.reservoirs.struct_radial_reservoir import StructRadialReservoir

from darts.physics.base.physics import PhysicsBase
from darts.physics.eos_physics import EoSPhysics
from darts.physics.base.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.viscosity import Fenghour1998
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.set_initial_conditions import LinearAmbientTemperature
from darts.pipes.ramp_up_rate import RampUpRate
from darts.pipes.pipe import Pipe
from darts.pipes.interfacial_tension import IFT_multicomponent_MCM

# class Model(DartsModelWithLivePlots):
class Model(DartsModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Use DartsModelWithLivePlots as the super class and enable plots below for live plotting
        # self.live_plot_config.enable_solver_props = True

        # self.live_plot_config.enable_ph_diagram = True
        # self.live_plot_config.tracked_block_idx = 1000

        # self.live_plot_config.enable_well_res_profiles = True
        # self.live_plot_config.plot_till_this_res_cell = 0

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.reservoir.grav_acceleration_for_spe = 9.80665
        self.zero = 1e-10
        self.set_physics()

        # For isenthalpic injection and injection at a constant gas rate
        # NOTE: set_sim_params stays in __init__ (not moved to set_solver): set_wells()
        # builds RampUpRate from self.ts_control.dt_first and runs during init() before
        # reset()/set_solver(). dfm_well is the documented set_solver exception.
        self.ts_control.dt_first = 0.0001/(24*60*60)
        self.ts_control.dt_min = 1e-15
        self.ts_control.dt_mult = 2
        self.ts_control.dt_max = 2/(24*60*60)
        self.ts_control.runtime = 1/24/60  # This runtime will be used when CI test is conducted without the main file

        # # For injection at a constant WHP
        # self.linear_solver.set_sim_params(first_ts=0.0001/(24*60*60), mult_ts=2, max_ts=0.1/(24*60*60),  tol_linear=1e-4,
        #                      it_linear=10,
        #
        #                     )

        # # For injection at a constant total mass rate
        # # Use 0.001 as the first time-step size because 0.0001 did not converge
        # self.linear_solver.set_sim_params(first_ts=0.001/(24*60*60), mult_ts=2, max_ts=2/(24*60*60),  tol_linear=1e-4,
        #                      it_linear=10,
        #
        #                     )

        self.timer.node["initialization"].stop()

    def set_solver(self):
        # Linear-solver settings live on self.linear_solver (the LinearSolverSpec).
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-3, max_iterations=10,
            chop=ChopSpec(mode='local'),
            coupled_well_res_norm_method=2)
        self.linear_solver.spec.tolerance = 1e-4
        self.linear_solver.spec.max_iterations = 10

    def set_reservoir(self):
        (nr, nz) = (1000, 1)
        (dr, dz) = (0.01, 50)

        poro = np.ones((nr, nz)) * 0.21
        permr = np.ones((nr, nz)) * 100

        permz = permr

        self.well_1_ID = 0.1016
        self.reservoir = StructRadialReservoir(self.timer, nr=nr, nz=nz, dr=dr, dz=dz, poro=poro.flatten(order='F'),
                                               permr=permr.flatten(order='F'), permz=permz.flatten(order='F'),
                                               R0=self.well_1_ID / 2, R1=1000, logspace=True, rcond=259.2, hcap=5250,
                                               depth=2025)  # depth is the depth of the centroid of the top reservoir cell
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
        from dartsflash.libflash import EoS
        from dartsflash.components import CompData
        from dartsflash.mixtures import DARTSFlash, Mixture
        components_names = ['CO2']
        phases_names = ['G', 'L']   # G is the gaseous-CO2 phase and L is the liquid-CO2 phase
        comp_data = CompData(components_names, setprops=True)
        epsilon = self.zero / 10

        """ Define state specification and initialize physics object """
        ph = True
        state_spec = PhysicsBase.StateSpecification.PH if ph else PhysicsBase.StateSpecification.PT
        # state_spec=PH for 1-comp thermal → axes [p, h]
        self.physics = EoSPhysics(components_names, phases_names, self.timer, state_spec=state_spec,
                                  axes_step=[0.05, 0.035],  # p [bar], h
                                  axes_origin=[1.0, 150.0],
                                  epsilon_z=epsilon)

        """ PropertyContainer object and correlations """
        property_container = PropertyContainer(phases_names, components_names, Mw=comp_data.Mw, eps_z=epsilon,
                                               temperature=None, rock_comp=0)

        """ Define flash """
        mixture = Mixture(comp_data)
        mixture.set_vl_eos("PR", root_order=[EoS.MAX, EoS.MIN])
        mixture.init_flash(flash_type=DARTSFlash.FlashType.PHFlash if ph else DARTSFlash.FlashType.PTFlash)

        self.physics.set_mixture(mixture)
        property_container.flash_ev = self.physics.get_flash_ev()

        """ Define phase properties """
        pr = mixture.eos["PR"]
        property_container.density_ev = dict([('G', EoSDensity(eos=pr, root_flag=EoS.RootFlag.MAX)),
                                              ('L', EoSDensity(eos=pr, root_flag=EoS.RootFlag.MIN)),
        # property_container.density_ev = dict([('G', self.physics.get_density_ev_from_flash(phase_idx=0)),
        #                                       ('L', self.physics.get_density_ev_from_flash(phase_idx=1)),
                                              ])
        property_container.enthalpy_ev = dict([('G', EoSEnthalpy(eos=pr, root_flag=EoS.RootFlag.MAX)),
                                               ('L', EoSEnthalpy(eos=pr, root_flag=EoS.RootFlag.MIN)),
        # property_container.enthalpy_ev = dict([('G', self.physics.get_enthalpy_ev_from_flash(phase_idx=0)),
        #                                        ('L', self.physics.get_enthalpy_ev_from_flash(phase_idx=1)),
                                               ])
        property_container.viscosity_ev = dict([('G', Fenghour1998()),
                                                ('L', Fenghour1998()),
                                                ])

        # diff = 8.64e-6
        # property_container.diffusion_ev = dict([('G', ConstFunc(np.ones(len(components_names)) * diff)),
        #                                         ('L', ConstFunc(np.ones(len(components_names)) * diff)),
        #                                         ('aqueous', ConstFunc(np.ones(len(components_names)) * diff * 1e-3))])

        property_container.conductivity_ev = dict([('G', ConstFunc(3.5)),
                                                   ('L', ConstFunc(7.)),
                                                   ])

        self.sw_init_res = 0
        swc = self.sw_init_res
        property_container.rel_perm_ev = dict([('G', PhaseRelPerm("gas", swc=swc, sgr=swc, n=1.5)),
                                               ('L', PhaseRelPerm("oil", swc=swc, sgr=swc, n=1.5)),
                                               ])

        property_container.IFT_ev = IFT_multicomponent_MCM(components_names)

        """ Add property region """
        self.physics.add_property_region(property_container)

        property_container.output_props = {}
        property_container.output_props['temperature'] = lambda: property_container.temperature
        for j, ph in enumerate(phases_names):
            property_container.output_props['s' + ph] = lambda jj=j: property_container.sat[jj]
            property_container.output_props['rho' + ph] = lambda jj=j: property_container.dens[jj]
            property_container.output_props['miu' + ph] = lambda jj=j: property_container.mu[jj]
            for i, comp in enumerate(components_names):
                property_container.output_props[f'x{comp}_in_{ph}_mass'] = lambda jj=j, ii=i: property_container.x_mass[jj, ii]

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

        initial_conditions_dict = {'phases_names': ['G'], 'phases_compositions': [[1.]],
                                   'pipe_intervals': [[0, well_1_geometry.pipe_length]]}  # 0 is the beginning of the pipe and pipe_intervals are TVD

        well_1_initial_conditions = LinearAmbientTemperature(well_1_name, well_1_geometry, self.physics,
                                                             pipe_head_pressure, pipe_head_temperature, temp_grad,
                                                             pipe_head_segment_index, initial_conditions_dict, verbose)

        #%% Add source/sink terms
        inj_segment_idx = 0
        inflow_or_outflow = "inflow"
        target_inj_rate = 58895.98  # in kmol/day
        ramp_up_period = 3 / (24 * 60)  # in day

        inj_phase_comp = np.array([1.])
        # There is no difference if the phase used in the following line for evaluating enthalpy is either G
        # or L because for both the same EoSs are used.
        inj_phase_name = "G"
        injected_fluid_pressure = 60.
        injected_fluid_temperature = 10 + 273.15
        inj_fluid_props = {"composition": inj_phase_comp, "phase_name": inj_phase_name,
                           "pressure": injected_fluid_pressure,"temperature": injected_fluid_temperature}

        ramp_up_rate = RampUpRate(well_1_name, well_1_geometry, self.physics, self.ts_control.dt_first, inj_segment_idx,
                                  inflow_or_outflow, target_inj_rate, ramp_up_period, inj_fluid_props,
                                  verbose=verbose)
        # The following dict will be used in set_rhs_flux and pipe velocity evaluation
        source_sinks = {"RampUpRate1": ramp_up_rate}

        # %% Store well props
        self.wells = {'I1': Pipe('I1', well_1_geometry, self.physics, self.reservoir, well_1_initial_conditions,
                                 source_sinks=source_sinks,
                                 verbose=verbose)}

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
        Mw_avg = np.sum(self.physics.property_containers[0].Mw * inj_comp)
        inj_fluid_molar_potential_energy = inj_fluid_specific_potential_energy * Mw_avg
        inj_fluid_energy = inj_fluid_molar_enthalpy + inj_fluid_molar_potential_energy

        inj_energy_rate = inj_rate * inj_fluid_energy

        inj_rates = np.append(component_rate, inj_energy_rate)

        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        well_head_start_idx = (self.reservoir.mesh.n_res_blocks + inj_segment_idx) * self.physics.n_vars
        rhs_flux[well_head_start_idx:well_head_start_idx+self.physics.n_vars:] = - inj_rates

        return rhs_flux

    def set_well_controls(self):
        # When using well controls, make sure all the unnecessary sources/sinks from the pipe are removed and the
        # function set_rhs_flux is commented out.
        inj_composition = []
        w = self.reservoir.wells[0]

        # # Constant WHP
        # self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
        #                                is_inj=True, target=60.0, inj_composition=inj_composition, inj_temp=283.15)

        # # Constant injection mass rate of gaseous phase
        # # Don't inject at a constant liquid rate because it fails readily. The reason is that there is no liquid available in
        # # the wellhead cell. There are methods to overcome this later, e.g., use a high initial pressure for the wellhead cell to
        # # have liquid CO2 available in it from the beginning.
        # target_inj_rate = 1 * 24 * 3600  # in kg/day
        # self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE, phase_name="G",
        #                                is_inj=True, target=target_inj_rate, inj_composition=inj_composition, inj_temp=283.15)

        # # Constant total injection mass rate
        # target_inj_rate = 5 * 24 * 3600  # in kg/day
        # self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE,
        #                                is_inj=True, target=target_inj_rate, inj_composition=inj_composition, inj_temp=283.15)
