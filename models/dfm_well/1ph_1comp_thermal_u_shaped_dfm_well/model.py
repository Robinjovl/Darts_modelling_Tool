from pathlib import Path

import numpy as np
from darts.engines import ms_well, value_vector

from darts.linear_solvers import SuperLUSolverSpec
from darts.models.darts_model import DartsModel
from darts.nonlinear_solvers import ChopSpec, NewtonSolver
from darts.physics.base.physics import PhysicsBase
from darts.physics.base.property_container import PropertyContainer
from darts.physics.eos_physics import EoSPhysics
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from darts.physics.properties.viscosity import Fenghour1998
from darts.pipes.define_pipe_geometry import PETREL_PipeGeometry
from darts.pipes.interfacial_tension import IFT_multicomponent_MCM
from darts.pipes.pipe import Pipe
from darts.pipes.ramp_up_rate import RampUpRate
from darts.pipes.set_initial_conditions import LinearAmbientTemperature
from darts.reservoirs.struct_radial_reservoir import StructRadialReservoir


class Model(DartsModel):
    """
    Non-isothermal CO2 flow through a standalone U-shaped DFM well.

    Fluid enters the first segment through a ramped source. The final segment
    has a very large volume and therefore approximates a constant-pressure
    production wellhead over the simulated period. There are no reservoir
    perforations or lateral heat-exchange connections.
    """

    def __init__(self):
        super().__init__()

        self.timer.node["initialization"].start()

        self.well_name = "U1"
        self.well_ID = 0.1
        self.producer_boundary_volume = 1e20  # m3
        self.set_reservoir()
        self.reservoir.grav_acceleration_for_spe = 9.80665
        self.zero = 1.0e-10
        self.set_physics()

        # RampUpRate uses this value while set_wells() runs during init().
        self.ts_control.dt_first = 0.0001 / (24 * 60 * 60)
        self.ts_control.dt_min = 1.0e-15
        self.ts_control.dt_mult = 2.0
        self.ts_control.dt_max = 2.0 / (24 * 60 * 60)
        self.ts_control.runtime = 30.0 / (24 * 60 * 60)

        self.timer.node["initialization"].stop()

    def set_solver(self):
        super().set_solver()
        self.nonlinear_solver = NewtonSolver(tolerance=1.0e-3, max_iterations=12,
                                             chop=ChopSpec(mode="local"), coupled_well_res_norm_method=2)
        self.linear_solver.spec = SuperLUSolverSpec(tolerance=1.0e-4, max_iterations=20)

    def set_reservoir(self):
        # The reservoir is required by DartsModel but remains disconnected from
        # this standalone well because no perforations are added.
        porosity = np.full(2, 0.2)
        permeability = np.full(2, 100.0)
        self.reservoir = StructRadialReservoir(self.timer, nr=2, nz=1, dr=5.0, dz=50.0, poro=porosity,
                                               permr=permeability, permz=permeability, R0=self.well_ID / 2,
                                               R1=10.0, logspace=True, rcond=181.44, hcap=2200.0, depth=0.0)

    def set_physics(self):
        from dartsflash.components import CompData
        from dartsflash.libflash import EoS
        from dartsflash.mixtures import DARTSFlash, Mixture

        component_names = ["CO2"]
        phase_names = ["G", "L"]
        component_data = CompData(component_names, setprops=True)
        epsilon = self.zero / 10

        self.physics = EoSPhysics(component_names, phase_names, self.timer,
                                  state_spec=PhysicsBase.StateSpecification.PH,
                                  axes_step=[0.05, 0.035], axes_origin=[1.0, 150.0], epsilon_z=epsilon)

        property_container = PropertyContainer(phase_names, component_names, Mw=component_data.Mw,
                                               eps_z=epsilon, temperature=None, rock_comp=0)

        mixture = Mixture(component_data)
        mixture.set_vl_eos(vl_eos_name="PR", root_order=[EoS.MAX, EoS.MIN])
        peng_robinson = mixture.eos["PR"]
        mixture.init_flash(flash_type=DARTSFlash.FlashType.PHFlash)
        self.physics.set_mixture(mixture)

        property_container.flash_ev = self.physics.get_flash_ev()
        property_container.density_ev = {
            "G": EoSDensity(eos=peng_robinson, root_flag=EoS.RootFlag.MAX),
            "L": EoSDensity(eos=peng_robinson, root_flag=EoS.RootFlag.MIN)
        }
        property_container.enthalpy_ev = {
            "G": EoSEnthalpy(eos=peng_robinson, root_flag=EoS.RootFlag.MAX),
            "L": EoSEnthalpy(eos=peng_robinson, root_flag=EoS.RootFlag.MIN)
        }
        property_container.viscosity_ev = {
            "G": Fenghour1998(),
            "L": Fenghour1998()
        }
        property_container.conductivity_ev = {
            "G": ConstFunc(10.0),
            "L": ConstFunc(7.0)
        }
        property_container.rel_perm_ev = {
            "G": PhaseRelPerm("gas", swc=0.0, sgr=0.0, n=1.0),
            "L": PhaseRelPerm("oil", swc=0.0, sgr=0.0, n=1.0)
        }
        property_container.IFT_ev = IFT_multicomponent_MCM(component_names)
        self.physics.add_property_region(property_container)

        property_container.output_props = {"temperature": lambda: property_container.temperature}
        for phase_index, phase_name in enumerate(phase_names):
            property_container.output_props[f"s{phase_name}"] = lambda index=phase_index: property_container.sat[index]
            property_container.output_props[f"rho{phase_name}"] = lambda index=phase_index: property_container.dens[index]
            property_container.output_props[f"mu{phase_name}"] = lambda index=phase_index: property_container.mu[index]

    def set_wells(self):
        trajectory_path = Path(__file__).with_name("u_shaped_trajectory.txt")
        geometry = PETREL_PipeGeometry(self.well_name, str(trajectory_path), num_segments=60,
                                       pipe_ID=self.well_ID, wall_roughness=2.5e-5,
                                       verbose=self.verbose > 0)

        # Increase segment volume without changing the length used by the momentum
        # equation. This makes the last segment an approximate fixed-pressure receiving boundary.
        geometry.segment_volumes[-1] = self.producer_boundary_volume

        initial_conditions_dict = {
            "phases_names": ["G"],
            "phases_compositions": [[1.0]],
            "pipe_intervals": [[float(np.min(geometry.TVD_faces)), float(np.max(geometry.TVD_faces))]]
        }
        initial_conditions = LinearAmbientTemperature(self.well_name, geometry, self.physics,
                                                      pipe_head_pressure=5.0, pipe_head_temperature=298.15,
                                                      temp_grad=0.025, pipe_head_segment_index=0,
                                                      initial_conditions_dict=initial_conditions_dict,
                                                      verbose=self.verbose > 0)

        injection_props = {"composition": np.array([1.0]), "phase_name": "G",
                           "pressure": 60.0, "temperature": 283.15}
        injection = RampUpRate(self.well_name, geometry, self.physics, self.ts_control.dt_first,
                               segment_idx=0, inflow_or_outflow="inflow",
                               target_molar_rate=float(58895.98 / 15),
                               ramp_up_period=float(2.0 / (24 * 60 * 60)),
                               inj_fluid_props=injection_props, verbose=self.verbose > 0)

        pipe = Pipe(self.well_name, geometry, self.physics, self.reservoir, initial_conditions,
                    source_sinks={"injection": injection}, drift_flux_model="tang_2019",
                    verbose=self.verbose > 0)
        self.wells = {self.well_name: pipe}
        self.reservoir.add_well(self.well_name, ms_well.MS_Type.DFM, well_geometry=geometry)

    def set_initial_conditions(self):
        self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh, input_distribution={"pressure": 5.0, "temperature": 298.15})
        for well in self.reservoir.wells:
            well.init_state = value_vector(self.wells[well.name].initial_conditions.initial_conditions_vector)

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        injection = self.wells[self.well_name].source_sinks["injection"]
        composition = injection.inj_fluid_props["composition"]
        component_rate = injection.current_rate * composition

        well_block_idx = self.reservoir.mesh.n_res_blocks + injection.segment_idx
        specific_potential_energy = self.reservoir.mesh.cell_spe[well_block_idx]
        average_molar_mass = np.sum(self.physics.property_containers[0].Mw * composition)
        molar_energy = injection.inj_fluid_props["molar_enthalpy"] + specific_potential_energy * average_molar_mass
        rates = np.append(component_rate, injection.current_rate * molar_energy)

        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        first_well_block = well_block_idx * self.physics.n_vars
        rhs_flux[first_well_block : first_well_block + self.physics.n_vars] = -rates
        return rhs_flux
