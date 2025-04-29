from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params
from darts.engines import ms_well

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

from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.set_initial_conditions import SingleAmbientTemperature
from darts.pipes.pipe import Pipe
from darts.pipes.interfacial_tension import IFT_multicomponent_MCM
from darts.pipes.units import *

class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        self.set_sim_params(first_ts=0.0001/(24*60*60), mult_ts=2, max_ts=3000/(24*60*60), tol_newton=1e-2, tol_linear=1e-3,
                            it_newton=50, it_linear=50, newton_type=sim_params.newton_local_chop)

        self.timer.node["initialization"].stop()
        zero = 1e-5
        self.initial_values = {self.physics.vars[0]: 5.332344,
                               self.physics.vars[1]: zero,
                               self.physics.vars[2]: zero
                               }

    def set_reservoir(self):
        nx = 1000
        depth = np.concatenate((925 * np.ones(nx), 975 * np.ones(nx)))
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=2, dx=0.1, dy=50, dz=50,
                                         permx=100, permy=100, permz=100, poro=0.2, depth=depth)
        self.reservoir.boundary_volumes = {'xy_minus': None, 'xy_plus': None,
                                           'yz_minus': 1e9, 'yz_plus': 1e9,
                                           'xz_minus': None, 'xz_plus': None}
        return

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

        system_temperature = 10 + 273.15

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
        well_1_segments_lengths = 50 * np.ones(20)  # From top to bottom of the wellbore
        well_1_ID = 0.1
        well_1_inclination_angle = 0.  # in degrees relative to the vertical direction
        well_1_wall_roughness = 2.5e-5
        verbose = True
        well_1_geometry = PipeGeometry(well_1_name, well_1_segments_lengths, well_1_ID, well_1_inclination_angle,
                                       well_1_wall_roughness, verbose)

        # %% Set initial conditions in the pipe using SingleAmbientTemperature
        system_temperature = self.physics.property_containers[0].temperature
        pipe_head_pressure = 5   # bar
        pipe_head_segment_index = 0  # index starts from zero

        # Wellhead conditions because of the constant rate control
        # zero = self.physics.axes_min[1]
        # well_head_segment_phase = 'gas'
        # well_head_segment_composition = [1.0 - 2 * zero*10, zero*10, zero*10]
        # well_head_segment_interval = [0, well_1_geometry.segments_lengths[0]]
        initial_conditions_dict = {'phases_names': ['gas'], 'phases_compositions': [[1e-5, 1 - 2 * 1e-5, 1e-5]],
                                    'pipe_intervals': [[0, well_1_geometry.pipe_length]]}  # 0 is the beginning of the pipe and pipe_intervals are TVD

        well_1_initial_conditions = SingleAmbientTemperature(well_1_name, well_1_geometry, self.physics, system_temperature,
                                                             pipe_head_pressure, pipe_head_segment_index,
                                                             initial_conditions_dict, verbose)

        # %% Store well props
        self.wells = {'I1': Pipe('I1', well_1_geometry, self.physics, well_1_initial_conditions)}

        self.reservoir.add_well(well_1_name, well_1_ms_type, well_geometry=well_1_geometry)

        # Well with two perforations
        reservoir_middle_cell_index = int(self.reservoir.nx / 2)
        self.reservoir.add_perforation(well_1_name, res_cell_idx=(reservoir_middle_cell_index, 1, 1), well_seg_idx=19, well_ID=well_1_geometry.pipe_ID)
        self.reservoir.add_perforation(well_1_name, res_cell_idx=(reservoir_middle_cell_index, 1, 2), well_seg_idx=20, well_ID=well_1_geometry.pipe_ID)

        """================================================= Well 2 ================================================="""
        # well_2_name = "P1"
        # well_2_ms_type = ms_well.MS_Type.EPM
        # well_2_ID = 0.1
        # self.reservoir.add_well(well_2_name, well_2_ms_type, well_ID=well_2_ID)
        # self.reservoir.add_perforation(well_2_name, cell_index=(1, 1, 1), well_ID=well_2_ID)

        """================================================= Well 3 ================================================="""
        # well_3_name = "P2"
        # well_3_ms_type = ms_well.MS_Type.EPM
        # well_3_ID = 0.1
        # self.reservoir.add_well(well_3_name, well_3_ms_type, well_ID=well_3_ID)
        # self.reservoir.add_perforation(well_3_name, cell_index=(self.reservoir.nx, 1, 1), well_ID=well_3_ID)

    def set_well_controls(self):
        # The following dict will be used in set_rhs_flux and PipeVelocityEvaluator
        inj_segment_idx = 0
        inj_rate = 58895.98/3   # kmol/day
        inj_comp = np.array([1.0 - 2 * 1e-5, 1e-5, 1e-5])
        self.wells["I1"].source_props = {"segment_idx_source": inj_segment_idx, "rate_source": inj_rate, "comp_source": inj_comp}

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        inj_segment_idx = self.wells["I1"].source_props["segment_idx_source"]
        inj_rate = self.wells["I1"].source_props["rate_source"]
        inj_comp = self.wells["I1"].source_props["comp_source"]
        inj_flux = inj_rate * inj_comp
        well_head_start_idx = (self.reservoir.mesh.n_res_blocks + inj_segment_idx) * self.physics.n_vars
        rhs_flux[well_head_start_idx:well_head_start_idx+self.physics.n_vars:] = - inj_flux   # inflow (e.g., injection) becomes minus for rhs

        return rhs_flux
