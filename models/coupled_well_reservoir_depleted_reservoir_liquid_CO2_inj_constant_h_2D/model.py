from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params, ms_well
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import Flash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, EoS
from dartsflash.components import CompData

from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.set_initial_conditions import LinearAmbientTemperature
from darts.pipes.pipe import Pipe
from darts.pipes.add_lateral_heat_exchange import WellLateralHeatTransfer
from darts.pipes.interfacial_tension import IFT_multicomponent_MCM
import darts.pipes.library as library
from darts.pipes.units import *

class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.zero = 1e-10
        self.set_physics()

        self.set_sim_params(first_ts=0.0001/(24*60*60), mult_ts=2, max_ts=2/(24*60*60), tol_newton=1e-13, tol_linear=1e-4,
                            it_newton=10, it_linear=10, newton_type=sim_params.newton_local_chop)

        self.timer.node["initialization"].stop()

        # calculate the state of the reservoir for the following p_init_res, sw_init_res, and zCO2_init_res
        p_init_res = 25   # from the pressure of the perforated segment of the wellbore
        T_init_res = 326.90   # from the temperature of the perforated segment of the wellbore

        self.initial_values = {self.physics.vars[0]: p_init_res,
                               self.physics.vars[1]: self.zero * 10,
                               self.physics.vars[2]: T_init_res,
                               }

    def set_reservoir(self):
        nx = 41
        ny = 41
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=1, dx=0.5, dy=0.5, dz=50,
                                         permx=200, permy=200, permz=200, poro=0.2, depth=1975)
        self.reservoir.boundary_volumes = {'xy_minus': None, 'xy_plus': None,
                                           'yz_minus': 1e9, 'yz_plus': 1e9,
                                           'xz_minus': 1e9, 'xz_plus': 1e9}
        return

    def set_physics(self):
        components_names = ['CO2', 'C1']
        phases_names = ['gas', 'LCO2']
        comp_data = CompData(components_names, setprops=True)

        flash_params = FlashParams(comp_data)
        pr = CubicEoS(comp_data, CubicEoS.PR)

        flash_params.add_eos("PR", pr)
        flash_params.eos_order = ["PR"]
        flash_params.eos_params["PR"].root_order = [EoS.MAX, EoS.MIN]

        params = flash_params.eos_params["PR"]
        params.initial_guesses = [i for i in range(comp_data.nc)]
        params.stability_tol = 1e-8
        params.stability_switch_tol = 1e-10
        params.stability_max_iter = 50
        params.use_gmix = False
        flash_params.comp_tol = 1e-2

        self.physics = Compositional(components_names, phases_names, self.timer, thermal=True, n_points=10000,
                                     min_p=1, max_p=500, min_z=self.zero / 10, max_z=1 - self.zero / 10,
                                     min_t=150, max_t=500)

        """ PropertyContainer object and correlations """
        property_container = PropertyContainer(phases_names, components_names, Mw=comp_data.Mw, min_z=self.zero / 10,
                                               temperature=None, rock_comp=0)

        property_container.flash_ev = Flash(flash_params)

        property_container.density_ev = dict([('gas', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                              ('LCO2', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                              ])
        property_container.viscosity_ev = dict([('gas', Fenghour1998()),
                                                ('LCO2', Fenghour1998()),
                                                ])

        diff = 8.64e-6
        property_container.diffusion_ev = dict([('gas', ConstFunc(np.ones(len(components_names)) * diff)),
                                                ('LCO2', ConstFunc(np.ones(len(components_names)) * diff)),
                                                ])

        property_container.enthalpy_ev = dict([('gas', EoSEnthalpy(eos=pr)),
                                               ('LCO2', EoSEnthalpy(eos=pr)),
                                               ])

        property_container.conductivity_ev = dict([('gas', ConstFunc(10.)),
                                                   ('LCO2', ConstFunc(10.)),
                                                   ])

        self.sw_init_res = 0.25
        swc = self.sw_init_res
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas", swc=swc, sgr=swc, n=1.5)),
                                               ('LCO2', PhaseRelPerm("oil", swc=swc, sgr=swc, n=1.5)),
                                               ])

        property_container.IFT_ev = IFT_multicomponent_MCM(components_names)

        """ Add property region """
        self.physics.add_property_region(property_container)

        property_container.output_props = {}
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
        well_1_segments_lengths = 50 * np.ones(40)  # From top to bottom of the wellbore
        well_1_ID = 0.1
        well_1_inclination_angle = 0.  # in degrees relative to the vertical direction
        well_1_wall_roughness = 2.5e-5
        verbose = True
        well_1_geometry = PipeGeometry(well_1_name, well_1_segments_lengths, well_1_ID, well_1_inclination_angle,
                                       well_1_wall_roughness, verbose)

        #%% Set initial conditions in the pipe using SingleAmbientTemperature
        pipe_head_pressure = 21.986738  # bar
        pipe_head_temperature = 5 + 273.15  # Kelvin
        temp_grad = 0.025  # deg C/meter
        pipe_head_segment_index = 0  # index starts from zero

        initial_conditions_dict = {'phases_names': ['gas'], 'phases_compositions': [[self.zero, 1 - self.zero]],
                                   'pipe_intervals': [[0, well_1_geometry.pipe_length]]}  # 0 is the beginning of the pipe and pipe_intervals are TVD

        well_1_initial_conditions = LinearAmbientTemperature(well_1_name, well_1_geometry, self.physics,
                                                             pipe_head_pressure, pipe_head_temperature, temp_grad,
                                                             pipe_head_segment_index, initial_conditions_dict)

        # %% Store well props
        self.wells = {'I1': Pipe('I1', well_1_geometry, self.physics, well_1_initial_conditions)}

        self.reservoir.add_well(well_1_name, well_1_ms_type, well_geometry=well_1_geometry)

        # Well with a single perforation
        middle_cell_x = int(np.ceil(self.reservoir.nx / 2))
        middle_cell_y = int(np.ceil(self.reservoir.ny / 2))
        self.reservoir.add_perforation(well_1_name, res_cell_idx=(middle_cell_x, middle_cell_y, 1), well_seg_idx=well_1_geometry.num_segments, well_ID=well_1_geometry.pipe_ID)

        # Add lateral heat exchange
        # Import rock data from the library
        c_rock = library.mats_thermal_props['Rock']['c']
        K_rock = library.mats_thermal_props['Rock']['K']
        rho_rock = library.mats_thermal_props['Rock']['rho']
        earth_thermal_props = {'T': well_1_initial_conditions.temp_init_segments, 'c': c_rock, 'K': K_rock,
                               'rho': rho_rock}
        pipe_wall_thickness = 5e-3   # Thickness of the outermost layer of the wellbore in meters
        outermost_layer_OD = well_1_geometry.pipe_ID + 2 * pipe_wall_thickness
        # Set a constant overall heat transfer coefficient (Ui)
        Ui = 0.2 * BTU() / (ft() ** 2 * hour() * Fahrenheit())  # Unit: BTU / (ft2 * hr * F)]  or  W / (m2 * C)
        well_1_lateral_heat_transfer = WellLateralHeatTransfer(well_1_name, well_1_geometry, earth_thermal_props,
                                                               outermost_layer_OD, Ui, time_function_name="Chiu&Thakur",
                                                               verbose=verbose)
        self.wells["I1"].lateral_heat_flux = well_1_lateral_heat_transfer

    def set_well_controls(self):
        # The following dict will be used in set_rhs_flux and PipeVelocityEvaluator
        inj_segment_idx = 0
        inj_rate = 58895.98   # kmol/day
        inj_comp = np.array([0.9825, 1 - 0.9825])
        self.wells["I1"].source_props = {"segment_idx_source": inj_segment_idx, "target_rate_source": inj_rate,
                                         "comp_source": inj_comp}

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        inj_segment_idx = self.wells["I1"].source_props["segment_idx_source"]

        # Calc ramp-up injection rate
        ramp_up_period = 4 / (24 * 60)   # 4 minutes
        inj_rate = self.calc_ramp_up_rate(self.wells["I1"].source_props["target_rate_source"], ramp_up_period, t)
        # Update injection rate in source_props
        self.wells["I1"].source_props["rate_source"] = inj_rate

        inj_comp = self.wells["I1"].source_props["comp_source"]
        inj_flux = inj_rate * inj_comp

        injected_fluid_pressure = 55
        injected_fluid_temperature = (15 + 273.15) * Kelvin()
        injected_fluid_mole_fractions = inj_comp

        injected_fluid_specific_enthalpy = self.physics.property_containers[0].enthalpy_ev['gas'].evaluate(
            injected_fluid_pressure,
            injected_fluid_temperature,
            injected_fluid_mole_fractions)  # Constant injection specific enthalpy
        injected_heat_rate = inj_rate * injected_fluid_specific_enthalpy
        inj_flux = np.append(inj_flux, injected_heat_rate)

        well_head_start_idx = (self.reservoir.mesh.n_res_blocks + inj_segment_idx) * self.physics.n_vars
        rhs_flux[well_head_start_idx:well_head_start_idx+self.physics.n_vars:] = - inj_flux   # inflow (e.g., injection) becomes minus for rhs

        return rhs_flux

    def calc_ramp_up_rate(self, target_rate, ramp_up_period, simulation_time) -> float:

        if simulation_time == 0:
            rate = (self.params.first_ts / ramp_up_period) * target_rate
        elif simulation_time < ramp_up_period:
            rate = ((simulation_time + self.params.first_ts) / ramp_up_period) * target_rate
        else:
            rate = target_rate

        return rate
