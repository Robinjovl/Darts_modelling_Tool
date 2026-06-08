import numpy as np

from darts.models.cicd_model import CICDModel
from darts.engines import sim_params, ms_well, value_vector, well_control_iface

from darts.reservoirs.struct_radial_reservoir import StructRadialReservoir

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.set_initial_conditions import LinearAmbientTemperature
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

        self.set_sim_params(first_ts=0.0001/(24*60*60), mult_ts=2, max_ts=2/(24*60*60), tol_newton=1e-3, tol_linear=1e-4,
                            it_newton=10, it_linear=10,
                            newton_type=sim_params.newton_local_chop,
                            coupled_well_res_norm_method=2,
                            runtime=10 / 24 / 60,   # This runtime will be used when CI test is conducted without the main file
                            )

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        (nr, nz) = (2, 1)
        (dr, dz) = (5, 50)

        poro = np.ones((nr, nz)) * 0.2
        permr = np.ones((nr, nz)) * 115

        permz = permr

        self.well_1_ID = 0.1
        self.reservoir = StructRadialReservoir(self.timer, nr=nr, nz=nz, dr=dr, dz=dz, poro=poro.flatten(order='F'),
                                               permr=permr.flatten(order='F'), permz=permz.flatten(order='F'),
                                               R0=self.well_1_ID / 2, R1=10, logspace=True, rcond=181.44, hcap=2200,
                                               depth=975)  # depth is the depth of the centroid of the top reservoir cell
        self.reservoir.boundary_volumes['yz_minus'] = 1e20

        return

    def set_initial_conditions(self):
        p_init_res = 5.88812   # from the pressure of the perforated segment of the wellbore
        T_init_res = 321.90000   # from the temperature of the perforated segment of the wellbore

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
        phases_names = ['G', 'L']
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
        property_container.density_ev = dict([('G', EoSDensity(eos=pr, Mw=comp_data.Mw, root_flag=EoS.RootFlag.MAX)),
                                              ('L', EoSDensity(eos=pr, Mw=comp_data.Mw, root_flag=EoS.RootFlag.MIN)),
                                              ])
        property_container.enthalpy_ev = dict([('G', EoSEnthalpy(eos=pr, root_flag=EoS.RootFlag.MAX)),
                                               ('L', EoSEnthalpy(eos=pr, root_flag=EoS.RootFlag.MIN)),
                                               ])
        property_container.viscosity_ev = dict([('G', Fenghour1998()),
                                                ('L', Fenghour1998()),
                                                ])

        property_container.conductivity_ev = dict([('G', ConstFunc(10.)),
                                                   ('L', ConstFunc(7.)),
                                                   ])

        property_container.rel_perm_ev = dict([('G', PhaseRelPerm("gas", swc=0.0, sgr=0.0, n=1.0)),
                                               ('L', PhaseRelPerm("oil", swc=0.0, sgr=0.0, n=1.0))])

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
        well_1_segments_lengths = 50 * np.ones(20)  # From top to bottom of the wellbore
        well_1_ID = self.well_1_ID
        well_1_inclination_angle = 0.  # in degrees relative to the vertical direction
        well_1_wall_roughness = 2.5e-5
        verbose = True
        well_1_geometry = PipeGeometry(well_1_name, well_1_segments_lengths, well_1_ID, well_1_inclination_angle,
                                       well_1_wall_roughness, verbose)

        # %% Set initial conditions in the pipe using LinearAmbientTemperature
        pipe_head_pressure = 5.0  # bar
        pipe_head_temperature = 298.15  # Kelvin
        temp_grad = 0.025  # deg C/meter
        pipe_head_segment_index = 0  # index starts from zero

        initial_conditions_dict = {'phases_names': ['G'], 'phases_compositions': [[1]],
                                   'pipe_intervals': [[0, well_1_geometry.pipe_length]]}  # 0 is the beginning of the pipe and pipe_intervals are TVD

        well_1_initial_conditions = LinearAmbientTemperature(well_1_name, well_1_geometry, self.physics,
                                                             pipe_head_pressure, pipe_head_temperature, temp_grad,
                                                             pipe_head_segment_index, initial_conditions_dict, verbose)

        # %% Store well props
        self.wells = {'I1': Pipe('I1', well_1_geometry, self.physics, self.reservoir, well_1_initial_conditions,
                                 verbose=verbose)}

        self.reservoir.add_well(well_1_name, well_1_ms_type, well_geometry=well_1_geometry)

        # Well with a single perforation
        well_1_perforated_segment = well_1_geometry.num_segments

        # Reservoir cell sizes for the Peaceman model
        self.reservoir.discretizer.len_cell_xdir[0, 0, 0] = 50.0
        self.reservoir.discretizer.len_cell_ydir[0, 0, 0] = 50.0
        self.reservoir.discretizer.len_cell_zdir[0, 0, 0] = 50.0
        well_index = 0.0  # Zero well index since perforation is treated with a well injectivity/productivity index instead
        self.reservoir.add_perforation(well_1_name, res_cell_idx=(1, 1, 1), well_seg_idx=well_1_perforated_segment,
                                       well_diameter=well_1_geometry.pipe_ID, with_peaceman_for_coupled_well_reservoir=True,
                                       well_index=well_index)

    def set_well_controls(self):
        inj_composition = []
        w = self.reservoir.wells[0]

        # Constant injection mass rate of gaseous phase
        target_inj_rate = 2 * 24 * 3600  # in kg/day
        self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE, phase_name="G",
                                       is_inj=True, target=target_inj_rate, inj_composition=inj_composition, inj_temp=293.15)
