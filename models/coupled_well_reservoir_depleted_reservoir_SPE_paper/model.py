from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData

from darts.wells.define_pipe_geometry import PipeGeometry
from darts.wells.set_initial_conditions import SingleAmbientTemperature
from darts.wells.check_initial_conditions import check_initial_conditions
from darts.wells.interfacial_tension import IFT_multicomponent_MCM
from darts.wells.units import *

class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        self.set_sim_params(first_ts=0.0001/(24*60*60), mult_ts=2, max_ts=6000/(24*60*60), tol_newton=1e-2, tol_linear=1e-3,   # increase the time step size from 20 to 30 sec
                            it_newton=50, it_linear=50, newton_type=sim_params.newton_local_chop)

        self.timer.node["initialization"].stop()
        zero = 1e-5

        # find saturation corresponding with composition
        p_init_res = 22.658649
        z_range = np.linspace(zero, 1 - zero, 10000)
        for z in z_range:
            state = [p_init_res, zero, z]
            sat =self.physics.property_containers[0].compute_saturation_full(state)
            if self.physics.property_containers[0].sat[self.physics.phases.index("liquid")] < 0.25:
                break

        self.initial_values = {self.physics.vars[0]: state[0],
                               self.physics.vars[1]: state[1],
                               self.physics.vars[2]: state[2]
                               }
        # self.initial_values = {self.physics.vars[0]: 60.915767,
        #                        self.physics.vars[1]: zero,
        #                        }

    def set_reservoir(self):
        (nr, nz) = (50, 5)
        if 1:
            case = 'depleted_gas_reservoir'
            poro = np.ones((nr, nz)) * 0.2
            perm = np.ones((nr, nz)) * 20

        else:
            case = 'Porthos'
            poro = np.ones((nr, nz)) * 0.075
            perm = np.ones((nr, nz)) * 0.29
            # upper detfurth
            perm[:, :16] = 12.6
            # hardegsen
            poro[:, :10] = 0.09
            perm[:, :10] = 24
            # hardegsen high perm
            poro[:, :8] = 0.2
            perm[:, :8] = 240
            # hardegsen
            poro[:, :6] = 0.09
            perm[:, :6] = 24
            # caprock
            poro[:, :4] = 0.01
            perm[:, :4] = 0.01

        self.set_reservoir_radial(nr=nr, dr=5, nz=nz, dz=50, poro=poro.flatten(order='F'), perm=perm.flatten(order='F'))

        return

    def set_reservoir_radial(self, nr, dr, nz, dz, poro, perm):
        self.seg_ratio = 1
        if 1:
            from nearwellbore import RadialStruct
            self.reservoir = RadialStruct(self.timer, nr=nr, nz=nz, dr=dr, dz=dz, permr=perm, permz=perm/10, poro=poro,
                                          R1=1000, logspace=True, boundary_volume=4.3e10, depth=1775)

        else:
            from nearwellbore import RadialUnstruct
            self.reservoir = RadialUnstruct()

        # self.reservoir.boundary_volumes['xy_plus'] = 1e8
        # self.reservoir.boundary_volumes['xy_minus'] = 1e8

        return

    def set_wells(self):
        """================================================= Well 1 ================================================="""
        well_1_name = "I1"
        well_1_type = "ms_well"
        # Lengths of the well segments are specified here.
        # The lengths of the well segments in front of the reservoir must be equal to the height of the reservoir cells.
        # well_1_segments_lengths = np.concatenate(([50], 50 * np.ones(20), [50]))  # From bottom to top of the wellbore
        # well_1_segments_lengths = 10 * np.ones(2)
        well_1_segments_lengths = 50 * np.ones(40)
        well_1_ID = 0.1
        well_1_inclination_angle = 0  # in degrees relative to the vertical direction
        well_1_wall_roughness = 2.5e-5
        verbose = True
        well_1_geometry = PipeGeometry(well_1_name, well_1_segments_lengths, well_1_ID, well_1_inclination_angle,
                                       well_1_wall_roughness, verbose)

        # %% Set initial conditions in the pipe using SingleAmbientTemperature
        system_temperature = self.physics.property_containers[0].temperature
        pipe_head_pressure = 20   # bar
        pipe_head_segment_index = well_1_geometry.num_segments - 1  # index starts from zero

        # Wellhead conditions because of the constant rate control
        # zero = self.physics.axes_min[1]
        # well_head_segment_phase = 'gas'
        # well_head_segment_composition = [1.0 - 2 * zero*10, zero*10, zero*10]
        # well_head_segment_interval = [well_1_geometry.pipe_length - 50, well_1_geometry.pipe_length]
        initial_fluid_conditions = {'phases_names': ['gas'], 'phases_compositions': [[1e-5, 1 - 2 * 1e-5, 1e-5]],
                                    'pipe_intervals': [[0, well_1_geometry.pipe_length]]}  # 0 is the beginning of the pipe
        # initial_fluid_conditions = {'phases_names': ['liquid'], 'phases_compositions': [[1e-5, 1 - 1e-5]],
        #                             'pipe_intervals': [[0, well_1_geometry.pipe_length]]}  # 0 is the beginning of the pipe

        well_1_initial_conditions = SingleAmbientTemperature(well_1_name, well_1_geometry, self.physics.property_containers[0], system_temperature,
                                                             pipe_head_pressure, pipe_head_segment_index,
                                                             initial_fluid_conditions, verbose)

        # %% Put initial conditions in wells_initial_conditions
        initial_CO2_mole_fraction = [initial_fluid_conditions['phases_compositions'][0][0]] * well_1_geometry.num_segments
        initial_C1_mole_fraction = [initial_fluid_conditions['phases_compositions'][0][1]] * well_1_geometry.num_segments

        self.wells_initial_conditions = {'initial_pressure': well_1_initial_conditions.p_init_segments,
                                         'initial_CO2_mole_fraction': initial_CO2_mole_fraction,
                                         'initial_C1_mole_fraction': initial_C1_mole_fraction}
        # self.wells_initial_conditions = {'initial_pressure': well_1_initial_conditions.p_init_segments,
        #                                  'initial_CO2_mole_fraction': initial_CO2_mole_fraction}
        check_initial_conditions(self.wells_initial_conditions, self.physics.property_containers[0].components_name,
                                 not self.physics.property_containers[0].thermal)

        # self.reservoir.add_well(well_1_name, well_1_type, well_geometry=well_1_geometry, physics=self.physics, darts_model = self)
        # reservoir_middle_cell_index = int(self.reservoir.nx / 2)
        # self.reservoir.add_perforation(well_1_name, perforated_reservoir_cell_index=(1, 1, 1), perforated_segment_index=2, well_geometry=well_1_geometry, ms_well=True)
        # self.reservoir.add_perforation(well_1_name, perforated_reservoir_cell_index=(1, 1, 2), perforated_segment_index=3, well_geometry=well_1_geometry, ms_well=True)

        self.reservoir.add_well(well_1_name, well_1_type, well_geometry=well_1_geometry, physics=self.physics, darts_model = self)
        self.reservoir.add_perforation(well_1_name, cell_index=(1, 1, 3), well_geometry=well_1_geometry)
        # self.reservoir.add_perforation(well_1_name, perforated_reservoir_cell_index=(1, 1, 2), perforated_segment_index=well_1_geometry.num_segments - self.reservoir.nz + 4, well_geometry=well_1_geometry, ms_well=True)
        # self.reservoir.add_perforation(well_1_name, perforated_reservoir_cell_index=(1, 1, 5), perforated_segment_index=well_1_geometry.num_segments - self.reservoir.nz + 5, well_geometry=well_1_geometry, ms_well=True)
        # self.reservoir.add_perforation(well_1_name, perforated_reservoir_cell_index=(1, 1, 5), perforated_segment_index=well_1_geometry.num_segments - self.reservoir.nz + 5, well_geometry=well_1_geometry, ms_well=True)
        # self.reservoir.add_perforation(well_1_name, perforated_reservoir_cell_index=(1, 1, 5), perforated_segment_index=well_1_geometry.num_segments - self.reservoir.nz + 5, well_geometry=well_1_geometry, ms_well=True)
        # for k in range(3, self.reservoir.nz):
        #     self.reservoir.add_perforation(well_1_name, perforated_reservoir_cell_index=(1, 1, k + 1), perforated_segment_index=well_1_geometry.num_segments - self.reservoir.nz + k + 1, well_geometry=well_1_geometry, ms_well=True)

        """================================================= Well 2 ================================================="""
        # well_2_name = "P1"
        # well_2_type = "basic_well"
        # well_2_ID = 0.1
        # self.reservoir.add_well(well_2_name, well_2_type, well_ID=well_2_ID)
        # self.reservoir.add_perforation(well_2_name, cell_index=(1, 1, 1), well_ID=well_2_ID)

        """================================================= Well 3 ================================================="""
        # well_3_name = "P2"
        # well_3_type = "basic_well"
        # well_3_ID = 0.1
        # self.reservoir.add_well(well_3_name, well_3_type, well_ID=well_3_ID)
        # self.reservoir.add_perforation(well_3_name, cell_index=(self.reservoir.nx, 1, 1), well_ID=well_3_ID)

    def set_physics(self):
        # """Physical properties"""
        zero = 1e-5
        components_names = ['CO2', 'C1', 'H2O']
        phases_names = ['gas', 'liquid']
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

        system_temperature = 35 + 273.15

        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases_names, components_name=components_names, Mw=comp_data.Mw,
                                               temperature=system_temperature, rock_comp=0, min_z=zero / 10)

        property_container.flash_ev = NegativeFlash(flash_params, ["CEOS", "AQ"], [InitialGuess.Henry_VA])
        property_container.density_ev = dict([('gas', EoSDensity(ceos, comp_data.Mw)),
                                              ('liquid', Garcia2001(components_names))])
        property_container.viscosity_ev = dict([('gas', Fenghour1998()),
                                                ('liquid', Islam2012(components_names))])

        # """Physical properties"""
        # zero = 1e-5
        # # Create property containers:
        # components_names = ['CO2', 'C1', 'H2O']
        # # components_names = ['CO2', 'H2O']
        # phases_names = ['gas', 'liquid']
        # thermal = 0
        # system_temperature = 35 + 273.15
        # Mw = [44.0098, 16.04288, 18.0152]
        # # Mw = [44.0098, 18.0152]
        #
        # property_container = PropertyContainer(phases_name=phases_names, components_name=components_names,
        #                                        Mw=Mw, min_z=zero / 10, temperature=system_temperature)
        #
        # """ properties correlations """
        # property_container.flash_ev = ConstantK(len(components_names), [4, 2, 1e-1], zero)
        # # property_container.flash_ev = ConstantK(len(components_names), [4, 1e-1], zero)
        # property_container.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
        #                                       ('liquid', DensityBasic(compr=1e-5, dens0=600))])
        # property_container.viscosity_ev = dict([('gas', ConstFunc(0.05)),
        #                                         ('liquid', ConstFunc(0.5))])

        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas", swc=0.25, sgr=0.0, n=1.5)),
                                               ('liquid', PhaseRelPerm("oil", swc=0.25, sgr=0.0, n=4))])
        property_container.IFT_ev = IFT_multicomponent_MCM(components_names)

        """ Activate physics """
        self.physics = Compositional(components_names, phases_names, self.timer,
                                     n_points=200, min_p=1, max_p=400, min_z=zero/10, max_z=1-zero/10)
        self.physics.add_property_region(property_container)

        property_container.output_props = {"sat_CO2_rich_phase": lambda: self.physics.property_containers[0].sat[0],
                                           "sat_aqueous_phase": lambda: self.physics.property_containers[0].sat[1],
                                           "mole_fraction_CO2__in_CO2_rich_phase": lambda: self.physics.property_containers[0].x[0, 0],
                                           "mole_fraction_CO2__in_aqueous_phase": lambda: self.physics.property_containers[0].x[1, 0],
                                           "mole_fraction_CH4__in_CO2_rich_phase": lambda: self.physics.property_containers[0].x[0, 1],
                                           "mole_fraction_CH4__in_aqueous_phase": lambda: self.physics.property_containers[0].x[1, 1],
                                           "mole_fraction_H2O__in_CO2_rich_phase": lambda: self.physics.property_containers[0].x[0, 2],
                                           "mole_fraction_H2O__in_aqueous_phase": lambda: self.physics.property_containers[0].x[1, 2],
                                           "rho_CO2_rich_phase": lambda: self.physics.property_containers[0].dens[0],
                                           "rho_aqueous_phase": lambda: self.physics.property_containers[0].dens[1],
                                           "miu_CO2_rich_phase": lambda: self.physics.property_containers[0].mu[0],
                                           "miu_aqueous_phase": lambda: self.physics.property_containers[0].mu[1]}

        return

    def set_well_controls(self):
        inj_stream = [1e-5, 1e-5]
        # inj_stream = [1e-5]
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                # If the injected fluid composition changes, the momentum bc in pipe_velocity_evaluator.py should get updated.
                # 58895.98 kmol/day = 30 kg/s
                w.control = self.physics.new_rate_inj(0, inj_stream, 0)   # inj rate in kmol/day
            # else:
            #     w.control = self.physics.new_bhp_prod(self.initial_values['pressure'])

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        inj_comp = np.array([1.0 - 2 * 1e-5, 1e-5, 1e-5])
        # inj_comp = np.array([1.0 - 1e-5, 1e-5])
        inj_rate = 58895.98   # kmol/day
        inj_flux = inj_rate * inj_comp
        well_head_start_idx = self.reservoir.mesh.n_res_blocks * self.physics.n_vars
        rhs_flux[well_head_start_idx:well_head_start_idx+self.physics.n_vars:] = - inj_flux   # inflow (e.g., injection) becomes minus for rhs
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

