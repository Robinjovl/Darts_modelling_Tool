from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm
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

        self.set_reservoir_radial()

        self.set_physics()

        self.set_sim_params(first_ts=0.0001/(24*60*60), mult_ts=2, max_ts=6000/(24*60*60), tol_newton=1e-2, tol_linear=1e-3,
                            it_newton=50, it_linear=50, newton_type=sim_params.newton_local_chop)

        self.timer.node["initialization"].stop()
        zero = 1e-5
        self.initial_values = {self.physics.vars[0]: 5.894002,
                               self.physics.vars[1]: zero,
                               self.physics.vars[2]: 0.1
                               }

    def set_reservoir(self):
        nx = 1000
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=0.1, dy=50, dz=50,
                                         permx=100, permy=100, permz=100, poro=0.2, depth=975)
        self.reservoir.boundary_volumes = {'xy_minus': None, 'xy_plus': None,
                                           'yz_minus': 1e9, 'yz_plus': 1e9,
                                           'xz_minus': None, 'xz_plus': None}
        return

    def set_reservoir_radial(self):
        from radial_grid import RadialStruct

        (nr, nz) = (100, 11)
        (dr, dz) = (5, 50)

        self.nr = nr

        poro = np.ones((nr, nz)) * 0.001
        perm = np.ones((nr, nz)) * 0.001
        poro[:, 1:10] = 0.2
        perm[:, 1:10] = 100

        top_depth = 700

        poro = poro.flatten(order='F')
        perm = perm.flatten(order='F')

        self.reservoir = RadialStruct(self.timer, nr=nr, nz=nz, dr=dr, dz=dz, permr=perm, permz=perm / 10, poro=poro,
                                      R1=300, logspace=True, boundary_volume=1e6, depth=top_depth)

        self.reservoir.boundary_volumes['xy_plus'] = 1e8
        self.reservoir.boundary_volumes['xy_minus'] = 1e8
        return

    def set_wells(self):
        """================================================= Well 1 ================================================="""
        well_1_name = "I1"
        well_1_type = "ms_well"
        # Lengths of the well segments are specified here.
        # The lengths of the well segments in front of the reservoir must be equal to the height of the reservoir cells.
        well_1_segments_lengths = 50 * np.ones(20)  # From bottom to top of the wellbore
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

        initial_fluid_conditions = {'phases_names': ['gas'], 'phases_compositions': [[1 - 2 * 1e-5, 1e-5, 1e-5]],
                                    'pipe_intervals': [[0, well_1_geometry.pipe_length]]}  # 0 is the beginning of the pipe

        well_1_initial_conditions = SingleAmbientTemperature(well_1_name, well_1_geometry, self.physics.property_containers[0], system_temperature,
                                                             pipe_head_pressure, pipe_head_segment_index,
                                                             initial_fluid_conditions, verbose)

        # %% Put initial conditions in wells_initial_conditions
        initial_CO2_mole_fraction = [initial_fluid_conditions['phases_compositions'][0][0]] * well_1_geometry.num_segments
        initial_C1_mole_fraction = [initial_fluid_conditions['phases_compositions'][0][1]] * well_1_geometry.num_segments

        self.wells_initial_conditions = {'initial_pressure': well_1_initial_conditions.p_init_segments,
                                         'initial_CO2_mole_fraction': initial_CO2_mole_fraction,
                                         'initial_C1_mole_fraction': initial_C1_mole_fraction}
        check_initial_conditions(self.wells_initial_conditions, self.physics.property_containers[0].components_name,
                                 not self.physics.property_containers[0].thermal)

        self.reservoir.add_well(well_1_name, well_1_type, well_geometry=well_1_geometry, physics=self.physics, darts_model=self)
        self.reservoir.add_perforation(well_1_name, res_cell_idx=(1, 1, 6), well_seg_idx=20, well_geometry=well_1_geometry)

    def set_physics(self):
        """Physical properties"""
        zero = 1e-5
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

        system_temperature = 35 + 273.15

        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases_names, components_name=components_names, Mw=comp_data.Mw,
                                               temperature=system_temperature, rock_comp=0, min_z=zero / 10)

        property_container.flash_ev = NegativeFlash(flash_params, ["CEOS", "AQ"], [InitialGuess.Henry_VA])
        property_container.density_ev = dict([('gas', EoSDensity(ceos, comp_data.Mw)),
                                              ('aqueous', Garcia2001(components_names))])
        property_container.viscosity_ev = dict([('gas', Fenghour1998()),
                                                ('aqueous', Islam2012(components_names))])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                               ('aqueous', PhaseRelPerm("oil"))])
        property_container.IFT_ev = IFT_multicomponent_MCM(components_names)

        """ Activate physics """
        self.physics = Compositional(components_names, phases_names, self.timer,
                                     n_points=200, min_p=1, max_p=300, min_z=zero/10, max_z=1-zero/10)
        self.physics.add_property_region(property_container)

        property_container.output_props = {"sat_CO2/C1_rich_phase": lambda: self.physics.property_containers[0].sat[0],
                                           "mole_fraction_CO2_in_CO2/C1_rich_phase": lambda: self.physics.property_containers[0].x[0,0],
                                           "mole_fraction_CO2_in_aqueous_phase": lambda: self.physics.property_containers[0].x[1,0],
                                           "rho_CO2/C1_rich_phase": lambda: self.physics.property_containers[0].dens[0],
                                           "rho_aqueous_phase": lambda: self.physics.property_containers[0].dens[1],
                                           "miu_CO2/C1_rich_phase": lambda: self.physics.property_containers[0].mu[0],
                                           "miu_aqueous_phase": lambda: self.physics.property_containers[0].mu[1]}

        return

    def set_well_controls(self):
        # If the injected fluid composition changes, the momentum bc in pipe_velocity_evaluator.py should get updated.
        # 58895.98 kmol/day = 30 kg/s
        inj_stream = [1e-5, 1e-5]
        self.reservoir.wells[0].control = self.physics.new_rate_inj(0, inj_stream, 0)  # inj rate in kmol/day

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        inj_comp = np.array([1.0 - 2 * 1e-5, 1e-5, 1e-5])
        # inj_comp = np.array([1.0 - 1e-5, 1e-5])
        inj_rate = 58895.98/3   # kmol/day
        inj_flux = inj_rate * inj_comp
        well_head_start_idx = self.reservoir.mesh.n_res_blocks * self.physics.n_vars
        rhs_flux[well_head_start_idx:well_head_start_idx+self.physics.n_vars:] = - inj_flux   # inflow (e.g., injection) becomes minus for rhs
        return rhs_flux
