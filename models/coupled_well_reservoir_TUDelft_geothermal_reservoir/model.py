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

from darts.wells.define_pipe_geometry import PipeGeometry
from darts.wells.set_initial_conditions import LinearAmbientTemperature
from darts.wells.check_initial_conditions import check_initial_conditions
from darts.wells.add_lateral_heat_exchange import WellLateralHeatTransfer
from darts.wells.interfacial_tension import IFT_multicomponent_MCM
import darts.wells.library as library
from darts.wells.units import *

class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.zero = 1e-10
        self.set_physics()

        self.set_sim_params(first_ts=0.0001/(24*60*60), mult_ts=2, max_ts=1/(24*60*60), tol_newton=1e-4, tol_linear=1e-4,
                            it_newton=50, it_linear=50, newton_type=sim_params.newton_local_chop)

        self.timer.node["initialization"].stop()

        # calculate the state of the reservoir for the following p_init_res, sw_init_res, and zCO2_init_res
        p_init_res = 219.011190   # from the pressure of the perforated segment of the wellbore
        T_init_res = 351.90   # from the temperature of the perforated segment of the wellbore

        self.initial_values = {self.physics.vars[0]: p_init_res,
                               self.physics.vars[1]: self.zero,
                               self.physics.vars[2]: 0.01,
                               self.physics.vars[3]: T_init_res
                               }

    def set_reservoir(self):
        (nr, nz) = (50, 5)
        (dr, dz) = (5, 50)
        if 1:
            case = 'depleted_gas_reservoir'
            poro = np.ones((nr, nz)) * 0.2
            permr = np.ones((nr, nz)) * 2000

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
                                          R1=1000, logspace=True, depth=2050)    # depth is the depth of the top exterface of the reservoir

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
        well_1__ms_type = ms_well.MS_Type.DFM
        # Lengths of the well segments are specified here.
        # The lengths of the well segments in front of the reservoir must be equal to the height of the reservoir cells.
        well_1_segments_lengths = 50 * np.ones(44)  # From bottom to top of the wellbore
        well_1_ID = 0.1
        well_1_inclination_angle = 0  # in degrees relative to the vertical direction
        well_1_wall_roughness = 2.5e-5
        verbose = True
        well_1_geometry = PipeGeometry(well_1_name, well_1_segments_lengths, well_1_ID, well_1_inclination_angle,
                                       well_1_wall_roughness, verbose)
        # This stored geometry information is only used for post-processing (plotting)
        self.wells_geometry = {well_1_name: well_1_geometry}

        #%% Set initial conditions in the pipe using SingleAmbientTemperature
        pipe_head_pressure = 10  # bar
        pipe_head_temperature = 25 + 273.15  # Kelvin
        temp_grad = 0.025  # deg C/meter
        pipe_head_segment_index = well_1_geometry.num_segments - 1  # index starts from zero

        # Wellhead conditions because of the constant rate control
        # zero = self.physics.axes_min[1]
        # well_head_segment_phase = 'gas'
        # well_head_segment_composition = [1.0 - 2 * zero*10, zero*10, zero*10]
        # well_head_segment_interval = [well_1_geometry.pipe_length - 50, well_1_geometry.pipe_length]
        initial_fluid_conditions = {'phases_names': ['aqueous'], 'phases_compositions': [[self.zero, self.zero, 1 - 2 * self.zero]],
                                    'pipe_intervals': [[0, well_1_geometry.pipe_length]]}  # 0 is the beginning of the pipe

        well_1_initial_conditions = LinearAmbientTemperature(well_1_name, well_1_geometry, self.physics.property_containers[0],
                                                             pipe_head_pressure, pipe_head_temperature, temp_grad,
                                                             pipe_head_segment_index, initial_fluid_conditions)

        # %% Put initial conditions in wells_initial_conditions
        initial_CO2_mole_fraction = [initial_fluid_conditions['phases_compositions'][0][0]] * well_1_geometry.num_segments
        initial_C1_mole_fraction = [initial_fluid_conditions['phases_compositions'][0][1]] * well_1_geometry.num_segments

        self.wells_initial_conditions = {'initial_pressure': well_1_initial_conditions.p_init_segments,
                                         'initial_CO2_mole_fraction': initial_CO2_mole_fraction,
                                         'initial_C1_mole_fraction': initial_C1_mole_fraction,
                                         'initial_temperature': well_1_initial_conditions.temp_init_segments}
        check_initial_conditions(self.wells_initial_conditions, self.physics.property_containers[0].components_name,
                                 not self.physics.property_containers[0].thermal)

        self.reservoir.add_well(well_1_name, well_1__ms_type, well_geometry=well_1_geometry, physics=self.physics, darts_model=self)

        # Well with single perforation
        self.reservoir.add_perforation(well_1_name, res_cell_idx=(1, 1, 3), well_seg_idx=44, well_geometry=well_1_geometry)

        # Add lateral heat exchange
        # Import rock data from the library
        c_rock = library.mats_thermal_props['Rock']['c']
        K_rock = library.mats_thermal_props['Rock']['K']
        rho_rock = library.mats_thermal_props['Rock']['rho']
        earth_thermal_props = {'T': well_1_initial_conditions.temp_init_segments[::-1], 'c': c_rock, 'K': K_rock,
                               'rho': rho_rock}
        pipe_wall_thickness = 5e-3   # Thickness of the outermost layer of the wellbore in meters
        outermost_layer_OD = well_1_geometry.pipe_ID + 2 * pipe_wall_thickness
        # Set a constant overall heat transfer coefficient (Ui)
        Ui = 0.2 * BTU() / (ft() ** 2 * hour() * Fahrenheit())  # Unit: BTU / (ft2 * hr * F)]  or  W / (m2 * C)
        well_1_lateral_heat_transfer = WellLateralHeatTransfer(well_1_name, well_1_geometry, earth_thermal_props,
                                                               outermost_layer_OD, Ui, time_function_name="Chiu&Thakur",
                                                               verbose=verbose)
        self.reservoir.wells_lateral_heat_flux = {well_1_name: well_1_lateral_heat_transfer}

    def set_well_controls(self):
        # The following dict will be used in set_rhs_flux and PipeVelocityEvaluator
        prod_segment_idx = 0
        prod_rate = - 471829./6   # kmol/day
        self.source_props = {"segment_idx_source": prod_segment_idx, "rate_source": prod_rate, "comp_source": 0.}

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        prod_segment_idx = self.source_props["segment_idx_source"]
        wellhead_idx = prod_segment_idx + self.reservoir.mesh.n_res_blocks
        prod_rate = self.source_props["rate_source"]
        wellhead_state = self.physics.engine.X[wellhead_idx * self.physics.n_vars:wellhead_idx * self.physics.n_vars + self.physics.n_vars]
        wellhead_pres = wellhead_state[0]
        wellhead_comp = list(wellhead_state[1:-1]) + [1 - sum(wellhead_state[1:-1])]
        wellhead_temp = wellhead_state[-1]
        prod_fluid_specific_enthalpy = self.physics.property_containers[0].enthalpy_ev['gas'].evaluate(wellhead_pres, wellhead_temp, wellhead_comp)

        prod_flux = prod_rate * np.array(wellhead_comp)

        # injected_fluid_pressure = 20
        # injected_fluid_temperature = (35 + 273.15) * Kelvin()
        # injected_fluid_mole_fractions = prod_comp

        # injected_fluid_specific_enthalpy = self.physics.property_containers[0].enthalpy_ev['gas'].evaluate(
        #     injected_fluid_pressure,
        #     injected_fluid_temperature,
        #     injected_fluid_mole_fractions)  # Constant injection specific enthalpy
        # injected_fluid_specific_enthalpy = - 1000
        prod_heat_rate = prod_rate * prod_fluid_specific_enthalpy
        prod_flux = np.append(prod_flux, prod_heat_rate)

        well_head_start_idx = wellhead_idx * self.physics.n_vars
        rhs_flux[well_head_start_idx:well_head_start_idx+self.physics.n_vars:] = - prod_flux   # inflow (e.g., injection) becomes minus for rhs

        return rhs_flux

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

