import numpy as np

from darts.models.cicd_model import CICDModel
from darts.engines import sim_params, ms_well, value_vector
from darts.nonlinear_solvers import NewtonSolver, ChopSpec

from darts.reservoirs.struct_radial_reservoir import StructRadialReservoir

from darts.physics.base.physics import PhysicsBase
from darts.physics.eos_physics import EoSPhysics
from darts.physics.base.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.set_initial_conditions import SingleAmbientTemperature
from darts.pipes.ramp_up_rate import RampUpRate
from darts.pipes.pipe import Pipe
from darts.pipes.interfacial_tension import IFT_multicomponent_MCM


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

        # NOTE: set_sim_params stays in __init__ (not moved to set_solver): set_wells()
        # builds RampUpRate from self.data_ts.dt_first and runs during init() before
        # reset()/set_solver(). dfm_well is the documented set_solver exception.
        if self.linear_solver is None:
            from darts.linear_solvers import LinearSolver
            self.linear_solver = LinearSolver(model=self)
        self.linear_solver.set_sim_params(first_ts=0.0001/(24*60*60), mult_ts=2, max_ts=2/(24*60*60),
                            runtime = 100 / 60 / 60 / 24,  # This runtime will be used when CI test is conducted without the main file
                            )

        self.timer.node["initialization"].stop()

    def set_solver(self):
        # Linear-solver settings live on self.linear_solver (the LinearSolverSpec).
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-3, max_iterations=10,
            chop=ChopSpec(mode='local'),
            coupled_well_res_norm_method=2)
        if self.linear_solver is None:
            from darts.linear_solvers import LinearSolver
            self.linear_solver = LinearSolver(model=self)
        self.linear_solver.spec.tolerance = 1e-4
        self.linear_solver.spec.max_iterations = 10

    def set_reservoir(self):
        (nr, nz) = (2, 1)
        (dr, dz) = (5, 50)

        poro = np.ones((nr, nz)) * 0.2
        permr = np.ones((nr, nz)) * 200

        permz = permr

        self.well_1_ID = 0.1
        self.reservoir = StructRadialReservoir(self.timer, nr=nr, nz=nz, dr=dr, dz=dz, poro=poro.flatten(order='F'),
                                               permr=permr.flatten(order='F'), permz=permz.flatten(order='F'),
                                               R0=self.well_1_ID / 2, R1=10, logspace=True,
                                               depth=975)  # depth is the depth of the centroid of the top reservoir cell
        self.reservoir.boundary_volumes['yz_minus'] = 1e20

        return

    def set_initial_conditions(self):
        p_init_res = 102.66245   # from the pressure of the perforated segment of the wellbore
        z_CO2 = self.zero   # from the composition of the perforated segment of the wellbore

        input_distribution = {self.physics.vars[0]: p_init_res,
                              self.physics.vars[1]: z_CO2,
                              }
        self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, input_distribution=input_distribution)

        for well in self.reservoir.wells:
            well.init_state = value_vector(self.wells[well.name].initial_conditions.initial_conditions_vector)

        return

    def set_physics(self):
        from dartsflash.libflash import EoS, NegativeFlash
        from dartsflash.components import CompData
        from dartsflash.mixtures import DARTSFlash, Mixture
        components_names = ['CO2', 'H2O']
        phases_names = ['G', 'L']   # G is the CO2-rich phase and L is the aqueous phase
        comp_data = CompData(components_names, setprops=True)
        epsilon = self.zero / 10

        """ Define state specification and initialize physics object """
        state_spec = PhysicsBase.StateSpecification.P
        self.physics = EoSPhysics(components_names, phases_names, self.timer, state_spec=state_spec,
                                  axes_step=[0.05, 1e-4], axes_origin=[1., 0.], epsilon_z=epsilon)

        """ Define flash """
        mixture = Mixture(comp_data)

        mixture.set_vl_eos("PR", hybrid_aq_eos_name="Aq",
                            root_order=[EoS.STABLE])
        mixture.set_aq_eos(aq_eos_name="Aq", )

        mixture.init_flash(flash_type=DARTSFlash.FlashType.NegativeFlash,
                           eos_order=["PR", "Aq"], nf_initial_guess=[NegativeFlash.Ki.Henry_VA])
        self.physics.set_mixture(mixture)

        """ Add property region """
        """ PropertyContainer object and correlations """
        system_temperature = 40 + 273.15

        property_container = PropertyContainer(phases_names, components_names, Mw=comp_data.Mw, eps_z=epsilon,
                                               temperature=system_temperature, rock_comp=0)

        self.physics.add_property_region(property_container)

        property_container.flash_ev = self.physics.get_flash_ev()

        property_container.density_ev = dict([('G', EoSDensity(eos=mixture.eos["PR"])),
                                              ('L', Garcia2001(components_names)),
                                              ])

        property_container.viscosity_ev = dict([('G', Fenghour1998()),
                                                ('L', Islam2012(components_names)),
                                                ])

        property_container.rel_perm_ev = dict([('G', PhaseRelPerm("gas", swc=0, sgr=0, n=1)),
                                               ('L', PhaseRelPerm("oil", swc=0, sgr=0, n=1))])

        property_container.IFT_ev = IFT_multicomponent_MCM(components_names)


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
        well_1_segments_lengths = 50 * np.ones(20)  # From top to bottom of the wellbore
        well_1_ID = self.well_1_ID
        well_1_inclination_angle = 0.  # in degrees relative to the vertical direction
        well_1_wall_roughness = 2.5e-5
        verbose = True
        well_1_geometry = PipeGeometry(well_1_name, well_1_segments_lengths, well_1_ID, well_1_inclination_angle,
                                       well_1_wall_roughness, verbose)

        #%% Set initial conditions in the pipe using SingleAmbientTemperature
        ambient_temperature = self.physics.property_containers[0].temperature
        pipe_head_pressure = 10  # bar
        pipe_head_segment_index = 0  # index starts from zero

        initial_conditions_dict = {'phases_names': ['L'], 'phases_compositions': [[1e-10, 1 - 1e-10]],
                                   'pipe_intervals': [[0, well_1_geometry.pipe_length]]}  # 0 is the beginning of the pipe and pipe_intervals are TVD

        well_1_initial_conditions = SingleAmbientTemperature(well_1_name, well_1_geometry, self.physics,
                                                             ambient_temperature, pipe_head_pressure,
                                                             pipe_head_segment_index, initial_conditions_dict, verbose)

        #%% Add source/sink terms
        inj_segment_idx = 0
        inflow_or_outflow = "inflow"
        target_inj_rate = 58895.98 / 15  # in kmol/day
        ramp_up_period = 0.0

        # Use zero water in the injected fluid in order to avoid sustained accumulation of water in the bottom-hole
        inj_phase_comp = np.array([1 - self.zero, self.zero])
        inj_fluid_props = {"composition": inj_phase_comp}

        ramp_up_rate = RampUpRate(well_1_name, well_1_geometry, self.physics, self.data_ts.dt_first, inj_segment_idx,
                                  inflow_or_outflow, target_inj_rate, ramp_up_period, inj_fluid_props,
                                  verbose=verbose)
        # The following dict will be used in set_rhs_flux and pipe velocity evaluation
        source_sinks = {"RampUpRate1": ramp_up_rate}

        # %% Store well props
        self.wells = {'I1': Pipe('I1', well_1_geometry, self.physics, self.reservoir, well_1_initial_conditions,
                                 source_sinks=source_sinks, verbose=verbose)}

        self.reservoir.add_well(well_1_name, well_1_ms_type, well_geometry=well_1_geometry)

        # Well with a single perforation
        well_1_perforated_segment = well_1_geometry.num_segments

        # Reservoir cell sizes for the Peaceman model
        self.reservoir.discretizer.len_cell_xdir[0, 0, 0] = 50.0
        self.reservoir.discretizer.len_cell_ydir[0, 0, 0] = 50.0
        self.reservoir.discretizer.len_cell_zdir[0, 0, 0] = 50.0
        well_index = 0.0  # Zero well index since perforation is treated with a well injectivity/productivity index instead
        well_index = 65.54393  # For the variable injectivity, which is equivalent to 1e5 kg/day/bar
        self.reservoir.add_perforation(well_1_name, res_cell_idx=(1, 1, 1), well_seg_idx=well_1_perforated_segment, well_index=well_index,
                                       well_diameter=well_1_geometry.pipe_ID, with_peaceman_for_coupled_well_reservoir=True)

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        inj_comp = self.wells["I1"].source_sinks["RampUpRate1"].inj_fluid_props["composition"]

        # Get updated ramp-up injection rate (rate is updated in pipe.py)
        inj_rate = self.wells["I1"].source_sinks["RampUpRate1"].current_rate

        component_rate = inj_rate * inj_comp

        inj_segment_idx = self.wells["I1"].source_sinks["RampUpRate1"].segment_idx

        inj_rates = component_rate

        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        well_head_start_idx = (self.reservoir.mesh.n_res_blocks + inj_segment_idx) * self.physics.n_vars
        rhs_flux[well_head_start_idx:well_head_start_idx + self.physics.n_vars:] = - inj_rates

        return rhs_flux
