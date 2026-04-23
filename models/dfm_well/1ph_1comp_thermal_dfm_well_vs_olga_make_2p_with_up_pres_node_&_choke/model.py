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
from darts.pipes.upstream_mass_node import UpstreamMassNode
from darts.pipes.upstream_pressure_node_with_choke import (
    UpstreamPressureNodeWithChoke,
)
from darts.pipes.pipe import Pipe
from darts.pipes.interfacial_tension import IFT_multicomponent_MCM
from darts.pipes.viz.plot_live import DartsModelWithLivePlots

# class Model(DartsModelWithLivePlots):
class Model(CICDModel):
    def __init__(
        self,
        inlet_boundary_kind: str = "upstream_mass_node",
        inlet_choke_valve_geometry: str = "ORIFICE",
        inlet_choke_equilibrium_model: str = "FROZEN",
        inlet_choke_diameter: float = None,
        inlet_choke_discharge_coefficient: float = 0.84,
        inlet_choke_opening: float = 1.0,
        inlet_choke_flow_coefficient: float = 1.0,
        inlet_choke_gas_liquid_sizing_ratio: float = 26.8465,
        inlet_choke_thermal_phase_equilibrium: bool = False,
        inlet_choke_recovery: str = "OFF",
        inlet_choke_recovery_tuning: float = 1.0,
        inlet_choke_slip_model: str = "NOSLIP",
    ):
        # Call base class constructor
        super().__init__()

        inlet_boundary_kind = inlet_boundary_kind.lower()
        if inlet_boundary_kind not in ("upstream_mass_node", "pressure_node_choke"):
            raise ValueError(
                "inlet_boundary_kind must be either 'upstream_mass_node' or 'pressure_node_choke'."
            )
        self.inlet_boundary_kind = inlet_boundary_kind
        self.inlet_choke_valve_geometry = inlet_choke_valve_geometry
        self.inlet_choke_equilibrium_model = inlet_choke_equilibrium_model
        self.inlet_choke_diameter = inlet_choke_diameter
        self.inlet_choke_discharge_coefficient = inlet_choke_discharge_coefficient
        self.inlet_choke_opening = inlet_choke_opening
        self.inlet_choke_flow_coefficient = inlet_choke_flow_coefficient
        self.inlet_choke_gas_liquid_sizing_ratio = (
            inlet_choke_gas_liquid_sizing_ratio
        )
        self.inlet_choke_thermal_phase_equilibrium = (
            inlet_choke_thermal_phase_equilibrium
        )
        self.inlet_choke_recovery = inlet_choke_recovery
        self.inlet_choke_recovery_tuning = inlet_choke_recovery_tuning
        self.inlet_choke_slip_model = inlet_choke_slip_model

        # self.live_plot_config.enable_well_res_profiles = True
        # self.live_plot_config.plot_till_this_res_cell = 0
        # self.live_plot_config.every_newton_iter = False

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

    def _get_inlet_molar_enthalpy(self, mass_node: UpstreamMassNode) -> float:
        return getattr(
            mass_node,
            "current_discharge_molar_enthalpy",
            mass_node.inj_fluid_props["molar_enthalpy"],
        )

    def set_initial_conditions(self):
        p_init_res = 5.88812   # from the pressure of the perforated segment of the wellbore
        # p_init_res = 79.79812  # from the pressure of the perforated segment of the wellbore
        T_init_res = 321.90000   # from the temperature of the perforated segment of the wellbore

        input_distribution = {self.physics.vars[0]: p_init_res,
                              "temperature": T_init_res,
                              }
        self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, input_distribution=input_distribution)

        for well in self.reservoir.wells:
            # self.wells[well.name].initial_conditions.initial_conditions_vector[:2:] = [70.90588379, -12520.9]
            well.init_state = value_vector(self.wells[well.name].initial_conditions.initial_conditions_vector)

        # self.reservoir.mesh.volume[self.reservoir.wells[0].well_head_idx] = 1e20
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

        #%% Add source/sink terms
        inj_segment_idx = 0
        target_inj_rate = 58895.98 / 30  # in kmol/day
        ramp_up_period = 0.0  # in day

        inj_phase_comp = np.array([1.])
        inj_phase_name = "L"
        injected_fluid_pressure = 60.0
        injected_fluid_temperature = 10 + 273.15

        if self.inlet_boundary_kind == "upstream_mass_node":
            ramp_up_rate = UpstreamMassNode(
                well_1_name,
                well_1_geometry,
                self.physics,
                self.data_ts.dt_first,
                inj_segment_idx,
                target_inj_rate,
                ramp_up_period,
                inj_phase_comp,
                injected_fluid_pressure,
                injected_fluid_temperature,
                inj_phase_name,
                verbose=verbose,
            )
        else:
            ramp_up_rate = UpstreamPressureNodeWithChoke(
                well_1_name,
                well_1_geometry,
                self.reservoir,
                self.physics,
                self.data_ts.dt_first,
                inj_segment_idx,
                target_inj_rate,
                ramp_up_period,
                inj_phase_comp,
                injected_fluid_pressure,
                injected_fluid_temperature,
                inj_phase_name,
                valve_geometry=self.inlet_choke_valve_geometry,
                equilibrium_model=self.inlet_choke_equilibrium_model,
                diameter=self.inlet_choke_diameter,
                discharge_coefficient=self.inlet_choke_discharge_coefficient,
                opening=self.inlet_choke_opening,
                flow_coefficient=self.inlet_choke_flow_coefficient,
                gas_liquid_sizing_ratio=self.inlet_choke_gas_liquid_sizing_ratio,
                thermal_phase_equilibrium=self.inlet_choke_thermal_phase_equilibrium,
                recovery=self.inlet_choke_recovery,
                recovery_tuning=self.inlet_choke_recovery_tuning,
                slip_model=self.inlet_choke_slip_model,
                initial_downstream_pressure=pipe_head_pressure,
                verbose=verbose,
            )
        # The following dict will be used in set_rhs_flux and pipe velocity evaluation
        source_sinks = {"UpstreamMassNode1": ramp_up_rate}

        # %% Store well props
        self.wells = {'I1': Pipe('I1', well_1_geometry, self.physics, self.reservoir, well_1_initial_conditions,
                                 source_sinks=source_sinks,
                                 verbose=verbose,
                                 )}

        self.reservoir.add_well(well_1_name, well_1_ms_type, well_geometry=well_1_geometry)

        # Well with a single perforation
        well_1_perforated_segment = well_1_geometry.num_segments

        self.reservoir.add_perforation(well_1_name, res_cell_idx=(1, 1, 1), well_seg_idx=well_1_perforated_segment,
                                       well_diameter=well_1_geometry.pipe_ID,
                                       pi=1e5,
                                       pi_type=ms_well.PI_Type.MASS,
                                       )

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        mass_node = self.wells["I1"].source_sinks["UpstreamMassNode1"]
        inj_segment_idx = mass_node.segment_idx
        specific_potential_energy = self.reservoir.mesh.cell_spe[
            self.reservoir.mesh.n_res_blocks + inj_segment_idx
        ]

        inj_rate = mass_node.current_rate
        component_rate = inj_rate * mass_node.composition
        inj_h = self._get_inlet_molar_enthalpy(mass_node)
        mw_avg = float(
            np.sum(self.physics.property_containers[0].Mw * mass_node.composition)
        )
        molar_energy = inj_h + specific_potential_energy * mw_avg
        inj_rates = np.append(component_rate, inj_rate * molar_energy)

        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        well_head_start_idx = (self.reservoir.mesh.n_res_blocks + inj_segment_idx) * self.physics.n_vars
        rhs_flux[well_head_start_idx:well_head_start_idx+self.physics.n_vars:] = - inj_rates

        return rhs_flux

    # def set_well_controls(self):
    #     inj_composition = []
    #     w = self.reservoir.wells[0]
    #
    #     # Constant injection mass rate of gaseous phase
    #     target_inj_rate = 2 * 24 * 3600  # in kg/day
    #     self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE, phase_name="G",
    #                                    is_inj=True, target=target_inj_rate, inj_composition=inj_composition, inj_temp=293.15)
