from darts.reservoirs.struct_radial_reservoir import StructRadialReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import value_vector, sim_params, ms_well
from darts.nonlinear_solvers import NewtonSolver
import numpy as np

from darts.physics.base.physics import PhysicsBase
from darts.physics.base.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic

from darts.pipes.interfacial_tension import IFT_multicomponent_MCM
from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.set_initial_conditions import SingleAmbientTemperature
from darts.pipes.pipe import Pipe


class Model(CICDModel):
    def __init__(self):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.zero = 1e-13
        self.set_physics()

        self.nonlinear_solver = NewtonSolver(tolerance=1e-3, coupled_well_res_norm_method=2)
        self.set_sim_params(first_ts=0.0001, mult_ts=2, max_ts=0.2, runtime=300, tol_linear=1e-6)

        self.timer.node["initialization"].stop()

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 89.22660,
                              self.physics.vars[1]: self.ini[0],
                              }
        self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                       input_distribution=input_distribution)

        for well in self.reservoir.wells:
            well.init_state = value_vector(self.wells[well.name].initial_conditions.initial_conditions_vector)

        return

    def set_reservoir(self):
        (nr, nz) = (1000, 1)
        (dr, dz) = (0.01, 100)

        poro = np.ones((nr, nz)) * 0.21
        permr = np.ones((nr, nz)) * 100

        permz = permr
        self.well_1_ID = 0.5
        self.reservoir = StructRadialReservoir(self.timer, nr=nr, nz=nz, dr=dr, dz=dz, poro=poro.flatten(order='F'),
                                               permr=permr.flatten(order='F'), permz=permz.flatten(order='F'),
                                               R0=self.well_1_ID / 2, R1=1000, logspace=True, rcond=216, hcap=1890,
                                               depth=450)  # depth is the depth of the centroid of the reservoir top cell

        self.reservoir.boundary_volumes['yz_plus'] = 1e20

        return

    def set_physics(self):
        """Physical properties"""
        epsilon = 1e-14
        components = ["w", "g"]
        phases = ["L", "G"]

        self.inj = value_vector([self.zero])
        self.ini = value_vector([1 - self.zero])

        property_container = ModelProperties(phases_name=phases, components_name=components, eps_z=epsilon)

        property_container.density_ev = dict([('L', DensityBasic(compr=0, dens0=1000)),
                                              ('G', DensityBasic(compr=0, dens0=1000))])
        property_container.viscosity_ev = dict([('L', ConstFunc(0.2)),
                                                # ('G', ConstFunc(0.2))])
                                                ('G', ConstFunc(0.1))])
        property_container.rel_perm_ev = dict([('L', PhaseRelPerm("wat", n=1)),
                                               ('G', PhaseRelPerm("oil", n=1))])

        property_container.IFT_ev = ConstFunc(0.01)

        # create physics
        thermal = False
        state_spec = PhysicsBase.StateSpecification.PT if thermal else PhysicsBase.StateSpecification.P
        self.physics = PhysicsBase(components, phases, self.timer, state_spec=state_spec,
                                     axes_step=[2.5, 2.5e-3], axes_origin=[0., 0.], epsilon_z=epsilon,
                                     extrapolation_flag=True)
        """ Add property region """
        self.physics.add_property_region(property_container)

        property_container.output_props = {}
        property_container.output_props['temperature'] = lambda: property_container.temperature
        for j, ph in enumerate(phases):
            property_container.output_props['s' + ph] = lambda jj=j: property_container.sat[jj]
            property_container.output_props['rho' + ph] = lambda jj=j: property_container.dens[jj]
            property_container.output_props['miu' + ph] = lambda jj=j: property_container.mu[jj]
            for i, comp in enumerate(components):
                property_container.output_props[f'x{comp}_in_{ph}_mass'] = lambda jj=j, ii=i: property_container.x_mass[jj, ii]

        return

    def set_wells(self):
        """================================================= Well 1 ================================================="""
        well_1_name = "I1"
        well_1_ms_type = ms_well.MS_Type.DFM
        # Lengths of the well segments are specified here.
        # The lengths of the well segments in front of the reservoir must be equal to the height of the reservoir cells.
        well_1_segments_lengths = 100 * np.ones(5)  # From top to bottom of the wellbore
        well_1_ID = self.well_1_ID
        well_1_inclination_angle = 0.  # in degrees relative to the vertical direction
        well_1_wall_roughness = 2.032e-5
        verbose = True
        well_1_geometry = PipeGeometry(well_1_name, well_1_segments_lengths, well_1_ID, well_1_inclination_angle,
                                       well_1_wall_roughness, verbose)

        #%% Set initial conditions in the pipe using SingleAmbientTemperature
        pipe_head_pressure = 50  # bar
        ambient_temperature = self.physics.property_containers[0].temperature
        pipe_head_segment_index = 0  # index starts from zero

        initial_conditions_dict = {'phases_names': ['L'], 'phases_compositions': [[1 - self.zero, self.zero]],
                                   'pipe_intervals': [[0, well_1_geometry.pipe_length]]}  # 0 is the beginning of the pipe and pipe_intervals are TVD

        well_1_initial_conditions = SingleAmbientTemperature(well_1_name, well_1_geometry, self.physics, ambient_temperature,
                                                             pipe_head_pressure, pipe_head_segment_index, initial_conditions_dict)

        # %% Store well props
        self.wells = {'I1': Pipe('I1', well_1_geometry, self.physics, self.reservoir, well_1_initial_conditions,
                                 enable_drift_velocity=False,
                                 enable_profile_parameter=False,
                                 )}

        self.reservoir.add_well(well_1_name, well_1_ms_type, well_geometry=well_1_geometry)

        # Well with a single perforation
        well_1_perforated_segment = well_1_geometry.num_segments
        self.reservoir.add_perforation(well_1_name, res_cell_idx=(1, 1, 1), well_seg_idx=well_1_perforated_segment, well_diameter=well_1_geometry.pipe_ID)

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        inj_segment_idx = 0
        inj_rates = np.array([100, 100])

        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        well_head_start_idx = (self.reservoir.mesh.n_res_blocks + inj_segment_idx) * self.physics.n_vars
        rhs_flux[well_head_start_idx:well_head_start_idx+self.physics.n_vars:] = - inj_rates

        return rhs_flux

class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, eps_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, eps_z=eps_z, temperature=1.)

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        self.pressure = vec_state_as_np[0]
        self.temperature = vec_state_as_np[-1] if self.thermal else self.temperature

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        self.ph = np.array([0, 1], dtype=np.intp)

        for j in self.ph:
            # molar weight of mixture
            M = np.sum(self.x[j, :] * self.Mw)
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(self.pressure)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]

        self.nu = zc
        self.compute_saturation(self.ph)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc[j] = 0

        return
