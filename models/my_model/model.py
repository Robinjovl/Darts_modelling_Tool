from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic

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

        self.set_sim_params(first_ts=0.0001/(24*60*60), mult_ts=2, max_ts=2/(24*60*60), tol_newton=1e-2, tol_linear=1e-3,
                            it_newton=10, it_linear=50, newton_type=sim_params.newton_local_chop)

        self.timer.node["initialization"].stop()
        zero = 1e-8
        self.initial_values = {self.physics.vars[0]: 5.271563,
                               self.physics.vars[1]: zero,
                               self.physics.vars[2]: zero
                               }

    def set_reservoir(self):
        nx = 3
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=1, dy=1, dz=1,
                                         permx=100, permy=100, permz=100, poro=0.2, depth=1000)
        return

    def set_wells(self):
        """================================================= Well 1 ================================================="""
        well_1_name = "I1"
        well_1_type = "ms_well"
        # Lengths of the well segments are specified here.
        # The lengths of the well segments in front of the reservoir must be equal to the height of the reservoir cells.
        well_1_segments_lengths = np.concatenate(([1], 2 * np.ones(2), [1]))  # From bottom to top of the wellbore
        well_1_ID = 0.1
        well_1_inclination_angle = 0  # in degrees relative to the vertical direction
        well_1_wall_roughness = 2.5e-5
        verbose = True
        well_1_geometry = PipeGeometry(well_1_name, well_1_segments_lengths, well_1_ID, well_1_inclination_angle,
                                       well_1_wall_roughness, verbose)

        # %% Set initial conditions in the pipe using SingleAmbientTemperature
        system_temperature = self.physics.property_containers[0].temperature
        pipe_head_pressure = 5   # bar
        pipe_head_segment_index = well_1_geometry.num_segments - 1  # index starts from zero

        # Wellhead conditions because of the constant rate control
        zero = self.physics.axes_min[1]
        well_head_segment_phase = 'gas'
        well_head_segment_composition = [1.0 - 2 * zero*10, zero*10, zero*10]
        well_head_segment_interval = [well_1_geometry.pipe_length -1, well_1_geometry.pipe_length]
        initial_fluid_conditions = {'phases_names': ['liquid', well_head_segment_phase], 'phases_compositions': [[1e-5, 1e-5, 1 - 2 * 1e-5], well_head_segment_composition],
                                    'pipe_intervals': [[0, well_1_geometry.pipe_length - 1], well_head_segment_interval]}  # 0 is the beginning of the pipe

        well_1_initial_conditions = SingleAmbientTemperature(well_1_name, well_1_geometry, self.physics.property_containers[0], system_temperature,
                                                             pipe_head_pressure, pipe_head_segment_index,
                                                             initial_fluid_conditions, verbose)

        # %% Put initial conditions in wells_initial_conditions
        initial_CO2_mole_fraction = np.concatenate(([initial_fluid_conditions['phases_compositions'][0][0]] * (well_1_geometry.num_segments - 1), [well_head_segment_composition[0]]))
        initial_C1_mole_fraction = np.concatenate(([initial_fluid_conditions['phases_compositions'][0][1]] * (well_1_geometry.num_segments - 1), [well_head_segment_composition[1]]))

        self.wells_initial_conditions = {'initial_pressure': well_1_initial_conditions.p_init_segments,
                                         'initial_CO2_mole_fraction': initial_CO2_mole_fraction,
                                         'initial_C1_mole_fraction': initial_C1_mole_fraction}
        check_initial_conditions(self.wells_initial_conditions, self.physics.property_containers[0].components_name,
                                 not self.physics.property_containers[0].thermal)

        self.reservoir.add_well(well_1_name, well_1_type, well_geometry=well_1_geometry, physics=self.physics, darts_model = self)
        self.reservoir.add_perforation(well_1_name, cell_index=(1, 1, 1), well_geometry=well_1_geometry)

        """================================================= Well 2 ================================================="""
        well_2_name = "P1"
        well_2_type = "basic_well"
        well_2_ID = 0.1
        self.reservoir.add_well(well_2_name, well_2_type, well_ID=well_2_ID)
        self.reservoir.add_perforation(well_2_name, cell_index=(self.reservoir.nx, 1, 1), well_ID=well_2_ID)

    def set_physics(self):
        """Physical properties"""
        zero = 1e-8
        # Create property containers:
        components_names = ['CO2', 'C1', 'H2O']
        phases_names = ['gas', 'liquid']
        thermal = 0
        system_temperature = 35 + 273.15
        Mw = [44.0098, 16.04288, 18.0152]

        property_container = PropertyContainer(phases_name=phases_names, components_name=components_names,
                                               Mw=Mw, min_z=zero / 10, temperature=system_temperature)

        """ properties correlations """
        property_container.flash_ev = ConstantK(len(components_names), [4, 2, 1e-1], zero)
        property_container.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                                              ('liquid', DensityBasic(compr=1e-5, dens0=600))])
        property_container.viscosity_ev = dict([('gas', ConstFunc(0.05)),
                                                ('liquid', ConstFunc(0.5))])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                               ('liquid', PhaseRelPerm("oil"))])
        property_container.IFT_ev = IFT_multicomponent_MCM(components_names)

        """ Activate physics """
        self.physics = Compositional(components_names, phases_names, self.timer,
                                     n_points=200, min_p=1, max_p=300, min_z=zero/10, max_z=1-zero/10)
        self.physics.add_property_region(property_container)

        return

    def set_well_controls(self):
        zero = self.physics.axes_min[1]
        inj_stream = [1.0 - 2 * zero*10, zero*10]
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                # If the injected fluid composition changes, the momentum bc in pipe_velocity_evaluator.py should get updated.
                # 58895.98 kmol/day = 30 kg/s
                w.control = self.physics.new_rate_inj(58895.98/3, inj_stream, 0)   # inj rate in kmol/day
            else:
                w.control = self.physics.new_bhp_prod(5.271563)
