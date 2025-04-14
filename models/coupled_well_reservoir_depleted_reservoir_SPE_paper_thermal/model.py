from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params, ms_well
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
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

        # If 0.1 is chosen as the max time step size, no stationary points will be observed, and the max NR iterations will be 3.
        self.set_sim_params(first_ts=0.0001/(24*60*60), mult_ts=2, max_ts=1/(24*60*60), tol_newton=1e-4, tol_linear=1e-4,
                            it_newton=50, it_linear=50, newton_type=sim_params.newton_local_chop)

        self.timer.node["initialization"].stop()

        # calculate the state of the reservoir for the following p_init_res, sw_init_res, and zCO2_init_res
        p_init_res = 23.771887   # from the pressure of the perforated segment of the wellbore
        T_init_res = 371.90   # from the temperature of the perforated segment of the wellbore

        sw_init_res = 0.25
        zCO2_init_res = self.zero
        zC1_range = np.linspace(self.zero, 1 - self.zero, 10000)
        for zC1 in zC1_range:
            state = [p_init_res, zCO2_init_res, zC1, T_init_res]
            self.physics.property_containers[0].compute_saturation_full(state)
            if self.physics.property_containers[0].sat[self.physics.phases.index("aqueous")] < sw_init_res:
                break

        self.initial_values = {self.physics.vars[0]: state[0],
                               self.physics.vars[1]: state[1],
                               self.physics.vars[2]: state[2],
                               self.physics.vars[3]: state[3]
                               }

    def set_reservoir(self):
        (nr, nz) = (50, 5)
        (dr, dz) = (5, 50)
        if 1:
            case = 'depleted_gas_reservoir'
            poro = np.ones((nr, nz)) * 0.2
            permr = np.ones((nr, nz)) * 20

            permz = permr / 10

        else:
            case = 'Porthos'
            poro = np.ones((nr, nz)) * 0.075
            permr = np.ones((nr, nz)) * 0.29
            # upper detfurth
            permr[:, :16] = 12.6
            # hardegsen
            poro[:, :10] = 0.09
            permr[:, :10] = 24
            # hardegsen high perm
            poro[:, :8] = 0.2
            permr[:, :8] = 240
            # hardegsen
            poro[:, :6] = 0.09
            permr[:, :6] = 24
            # caprock
            poro[:, :4] = 0.01
            permr[:, :4] = 0.01

            permz = permr / 10

        self.set_reservoir_radial(nr=nr, dr=dr, nz=nz, dz=dz, poro=poro.flatten(order='F'), permr=permr.flatten(order='F'), permz=permz.flatten(order='F'))

        return

    def set_reservoir_radial(self, nr, dr, nz, dz, poro, permr, permz):
        self.seg_ratio = 1
        if 1:
            from nearwellbore import RadialStruct
            self.reservoir = RadialStruct(self.timer, nr=nr, nz=nz, dr=dr, dz=dz, permr=permr, permz=permz, poro=poro,
                                          R1=1000, logspace=True, depth=2850)    # depth is the depth of the top exterface of the reservoir

        else:
            from nearwellbore import RadialUnstruct
            self.reservoir = RadialUnstruct()

        return

    def set_physics(self):
        """Physical properties"""
        components_names = ['CO2', 'C1', 'H2O']
        phases_names = ['gas', 'aqueous']
        comp_data = CompData(components_names, setprops=True)

        ceos = CubicEoS(comp_data, CubicEoS.PR)
        aq = AQEoS(comp_data, {AQEoS.water: AQEoS.Jager2003, AQEoS.solute: AQEoS.Ziabakhsh2012})

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        flash_params.add_eos("CEOS", ceos)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["CEOS", "AQ"]

        # Flash-related parameters
        flash_params.split_tol = 1e-14

        # system_temperature = 35 + 273.15

        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases_names, components_name=components_names, Mw=comp_data.Mw,
                                               temperature=None, rock_comp=0, min_z=self.zero / 10)

        property_container.flash_ev = NegativeFlash(flash_params, ["CEOS", "AQ"], [InitialGuess.Henry_VA])
        property_container.density_ev = dict([('gas', EoSDensity(ceos, comp_data.Mw)),
                                              ('aqueous', Garcia2001(components_names))])
        property_container.enthalpy_ev = dict([('gas', EoSEnthalpy(ceos)),
                                               ('aqueous', EoSEnthalpy(aq))])
        property_container.viscosity_ev = dict([('gas', Fenghour1998()),
                                                ('aqueous', Islam2012(components_names))])
        property_container.conductivity_ev = dict([('gas', ConstFunc(10.)),
                                                   ('aqueous', ConstFunc(180.)), ])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas", swc=0.25, sgr=0.0, n=1.5)),
                                               ('aqueous', PhaseRelPerm("oil", swc=0.25, sgr=0.0, n=4))])
        property_container.IFT_ev = IFT_multicomponent_MCM(components_names)

        """ Activate physics """
        self.physics = Compositional(components_names, phases_names, self.timer, thermal=True,
                                     n_points=1000, min_p=1, max_p=500, min_z=self.zero/10, max_z=1-self.zero/10,
                                     min_t=200, max_t=500)
        self.physics.add_property_region(property_container)

        property_container.output_props = {"sat_CO2/C1_rich_phase": lambda: self.physics.property_containers[0].sat[0],
                                           "sat_aqueous_phase": lambda: self.physics.property_containers[0].sat[1],
                                           "mole_fraction_CO2__in_CO2/C1_rich_phase": lambda: self.physics.property_containers[0].x[0, 0],
                                           "mole_fraction_CO2__in_aqueous_phase": lambda: self.physics.property_containers[0].x[1, 0],
                                           "mole_fraction_CH4__in_CO2/C1_rich_phase": lambda: self.physics.property_containers[0].x[0, 1],
                                           "mole_fraction_CH4__in_aqueous_phase": lambda: self.physics.property_containers[0].x[1, 1],
                                           "mole_fraction_H2O__in_CO2/C1_rich_phase": lambda: self.physics.property_containers[0].x[0, 2],
                                           "mole_fraction_H2O__in_aqueous_phase": lambda: self.physics.property_containers[0].x[1, 2],
                                           "rho_CO2/C1_rich_phase": lambda: self.physics.property_containers[0].dens[0],
                                           "rho_aqueous_phase": lambda: self.physics.property_containers[0].dens[1],
                                           "miu_CO2/C1_rich_phase": lambda: self.physics.property_containers[0].mu[0],
                                           "miu_aqueous_phase": lambda: self.physics.property_containers[0].mu[1],
                                           "enthalpy_CO2/C1_rich_phase": lambda: self.physics.property_containers[0].enthalpy[0],
                                           "enthalpy_aqueous_phase": lambda: self.physics.property_containers[0].enthalpy[1]}

        return

    def set_wells(self):
        """================================================= Well 1 ================================================="""
        well_1_name = "I1"
        well_1_ms_type = ms_well.MS_Type.DFM
        # Lengths of the well segments are specified here.
        # The lengths of the well segments in front of the reservoir must be equal to the height of the reservoir cells.
        well_1_segments_lengths = 50 * np.ones(60)  # From top to bottom of the wellbore
        well_1_ID = 0.1
        well_1_inclination_angle = 0  # in degrees relative to the vertical direction
        well_1_wall_roughness = 2.5e-5
        verbose = True
        well_1_geometry = PipeGeometry(well_1_name, well_1_segments_lengths, well_1_ID, well_1_inclination_angle,
                                       well_1_wall_roughness, verbose)

        #%% Set initial conditions in the pipe using SingleAmbientTemperature
        pipe_head_pressure = 20  # bar
        pipe_head_temperature = 25 + 273.15  # Kelvin
        temp_grad = 0.025  # deg C/meter
        pipe_head_segment_index = 0  # index starts from zero

        initial_fluid_conditions = {'phases_names': ['gas'], 'phases_compositions': [[self.zero, 1 - 2 * self.zero, self.zero]],
                                    'pipe_intervals': [[0, well_1_geometry.pipe_length]]}  # 0 is the beginning of the pipe

        well_1_initial_conditions = LinearAmbientTemperature(well_1_name, well_1_geometry, self.physics.property_containers[0],
                                                             pipe_head_pressure, pipe_head_temperature, temp_grad,
                                                             pipe_head_segment_index, initial_fluid_conditions)

        # %% Store well props
        self.wells = {'I1': Pipe('I1', well_1_geometry, self.physics, well_1_initial_conditions)}

        self.reservoir.add_well(well_1_name, well_1_ms_type, well_geometry=well_1_geometry)

        # Well with a single perforation
        self.reservoir.add_perforation(well_1_name, res_cell_idx=(1, 1, 3), well_seg_idx=60, well_ID=well_1_geometry.pipe_ID)

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
        inj_comp = np.array([1.0 - 2 * self.zero, self.zero, self.zero])
        self.wells["I1"].source_props = {"segment_idx_source": inj_segment_idx, "rate_source": inj_rate, "comp_source": inj_comp}

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        inj_segment_idx = self.wells["I1"].source_props["segment_idx_source"]

        # Calc ramp-up injection rate
        ramp_up_period = 4 / (24 * 60)   # 4 minutes
        inj_rate = self.calc_ramp_up_rate(self.wells["I1"].source_props["rate_source"], ramp_up_period, t)

        inj_comp = self.wells["I1"].source_props["comp_source"]
        inj_flux = inj_rate * inj_comp

        injected_fluid_pressure = 20
        injected_fluid_temperature = (35 + 273.15) * Kelvin()
        injected_fluid_mole_fractions = inj_comp

        # injected_fluid_specific_enthalpy = self.physics.property_containers[0].enthalpy_ev['gas'].evaluate(
        #     injected_fluid_pressure,
        #     injected_fluid_temperature,
        #     injected_fluid_mole_fractions)  # Constant injection specific enthalpy
        injected_fluid_specific_enthalpy = - 2000
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

    def plot(self, output_properties: list, fig=None, lims: dict = None, i: int = -1):
        output_data = self.output_properties()

        tot_props = self.physics.vars + self.physics.property_operators[0].props_name

        # loop for adding units to the titles of axes
        output_idxs = {}
        new_lims = {}
        for prop in output_properties:
            if prop == 'pressure':
                axis_name = 'pressure, bar'
            elif prop == 'temperature':
                axis_name = 'temperature, K'
            else:
                axis_name = prop
            output_idxs[axis_name] = tot_props.index(prop)
            new_lims[axis_name] = lims[prop]

        return self.reservoir.plot(output_idxs, output_data, fig=fig, lims=new_lims)

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

        timestep, output_data = self.output_properties(output_properties=output_properties, timestep=ith_step)

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

