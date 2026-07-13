from dataclasses import dataclass

import numpy as np
from darts.engines import ms_well, sim_params, value_vector, well_control_iface
from darts.models.darts_model import DartsModel
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.enthalpy import EnthalpyBasic
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.interfacial_tension import IFT_multicomponent_MCM
from darts.pipes.pipe import Pipe
from darts.pipes.ramp_up_rate import RampUpRate
from darts.pipes.set_initial_conditions import LinearAmbientTemperature
from darts.reservoirs.struct_reservoir import StructReservoir
from dartsflash.components import CompData
from dartsflash.libflash import EoS
from dartsflash.mixtures import DARTSFlash, VLAq


@dataclass
class AquiferCO2InjectionConfig:
    p_init: float | None = None
    t_reservoir: float = 356.15
    z_co2_init: float = 1e-4

    nx: int = 20
    ny: int = 10
    nz: int = 1
    dx: float = 30.0
    dy: float = 30.0
    dz: float = 10.0
    permx: float = 800.0
    permy: float = 800.0
    permz: float = 80.0
    poro: float = 0.2
    start_z: float = 2000.0

    injector_i: int | None = None
    injector_j: int | None = None
    injector_k: int = 1
    producer_i: int | None = None
    producer_j: int | None = None
    producer_k: int = 1

    well_depth: float = 2000.0
    well_segments: int = 40
    well_diameter: float = 0.1524
    well_temp_grad: float = 0.03

    injection_source_pressure: float = 60.5
    injection_source_temperature: float = 313.15
    target_gas_mass_rate_kg_day: float = 1e6
    producer_whp_bar: float = 1.0
    injection_ramp_up_period: float = 0.01 / (24.0 * 60.0)

    first_ts: float = 1e-7
    mult_ts: float = 2.0
    max_ts: float = 1e-4
    runtime: float = 0.05
    tol_newton: float = 1e-3
    tol_linear: float = 1e-4
    it_newton: int = 12
    it_linear: int = 60

class Model(DartsModel):
    def __init__(self, config: AquiferCO2InjectionConfig | None = None):
        super().__init__()

        self.config = config or AquiferCO2InjectionConfig()
        self.zero = 1e-8
        self.components = ["CO2", "H2O"]
        self.phases = ["G", "L"]
        self.wells = {}

        self.timer.node["initialization"].start()
        self.set_reservoir()
        self.reservoir.grav_acceleration_for_spe = 9.80665
        self.set_physics()
        self.set_sim_params(first_ts=self.config.first_ts, mult_ts=self.config.mult_ts, max_ts=self.config.max_ts,
                            runtime=self.config.runtime, tol_newton=self.config.tol_newton,
                            tol_linear=self.config.tol_linear, it_newton=self.config.it_newton,
                            it_linear=self.config.it_linear, newton_type=sim_params.newton_local_chop,
                            coupled_well_res_norm_method=2, well_rate_ctrl_absolute_residual_scale=1.0,
                            well_rate_ctrl_relative_residual_scale=1e-5)
        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        cfg = self.config
        self.reservoir = StructReservoir(self.timer, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy, dz=cfg.dz,
                                         permx=cfg.permx, permy=cfg.permy, permz=cfg.permz, poro=cfg.poro,
                                         depth=None, start_z=cfg.start_z, hcap=2200.0, rcond=2.1 * 86.4)
        self.reservoir.boundary_volumes["yz_minus"] = 1e20
        self.reservoir.boundary_volumes["yz_plus"] = 1e20
        self.reservoir.boundary_volumes["xz_minus"] = 1e20
        self.reservoir.boundary_volumes["xz_plus"] = 1e20

    def injector_cell(self) -> tuple[int, int, int]:
        cfg = self.config
        i = cfg.injector_i or cfg.nx // 4 + 1
        j = cfg.injector_j or cfg.ny // 2 + 1
        k = cfg.injector_k

        for name, value, upper in (
            ("injector_i", i, cfg.nx),
            ("injector_j", j, cfg.ny),
            ("injector_k", k, cfg.nz),
        ):
            if not 1 <= value <= upper:
                raise ValueError(f"{name}={value} is outside the reservoir grid.")
        return i, j, k

    def producer_cell(self) -> tuple[int, int, int]:
        cfg = self.config
        i = cfg.producer_i or cfg.nx - cfg.nx // 4
        j = cfg.producer_j or cfg.ny // 2 + 1
        k = cfg.producer_k

        for name, value, upper in (
            ("producer_i", i, cfg.nx),
            ("producer_j", j, cfg.ny),
            ("producer_k", k, cfg.nz),
        ):
            if not 1 <= value <= upper:
                raise ValueError(f"{name}={value} is outside the reservoir grid.")
        return i, j, k

    def set_physics(self):
        epsilon = 1e-9
        comp_data = CompData(self.components, setprops=True)

        pc = PropertyContainer(phases_name=self.phases, components_name=self.components, Mw=comp_data.Mw,
                               eps_z=epsilon, temperature=None)
        pc.flash_ev = VLAq(comp_data, hybrid=True)
        pc.flash_ev.set_vl_eos("PR", root_order=[EoS.STABLE], trial_comps=[i for i in range(len(self.components))],
                               stability_tol=1e-20, switch_tol=1e-2, max_iter=50, use_gmix=False)
        pc.flash_ev.set_aq_eos("Aq", stability_tol=1e-20, max_iter=10, use_gmix=True)
        pc.flash_ev.init_flash(flash_type=DARTSFlash.FlashType.PTFlash, eos_order=["VL", "Aq"], t_min=270.0,
                               t_max=430.0, t_init=300.0)

        pr = pc.flash_ev.eos["VL"]
        aq = pc.flash_ev.eos["Aq"]
        pc.density_ev = {"G": EoSDensity(eos=pr, Mw=comp_data.Mw), "L": Garcia2001(self.components)}
        pc.viscosity_ev = {"G": Fenghour1998(), "L": Islam2012(self.components)}
        pc.rel_perm_ev = {"G": PhaseRelPerm("gas", swc=0.20, sgr=0.00, kre=0.95, n=5),
                          "L": PhaseRelPerm("oil", swc=0.20, sgr=0.00, kre=1.0, n=6)}
        pc.enthalpy_ev = {"G": EoSEnthalpy(eos=pr), "L": EoSEnthalpy(eos=aq)}
        pc.conductivity_ev = {"G": ConstFunc(6.0), "L": ConstFunc(60.0)}
        pc.rock_energy_ev = EnthalpyBasic(hcap=1.0)
        pc.IFT_ev = IFT_multicomponent_MCM(self.components)

        pc.output_props = {"temperature": lambda: pc.temperature}
        for phase_idx, phase in enumerate(self.phases):
            pc.output_props[f"s{phase}"] = lambda idx=phase_idx: pc.sat[idx]
            pc.output_props[f"rho{phase}"] = lambda idx=phase_idx: pc.dens[idx]
            pc.output_props[f"mu{phase}"] = lambda idx=phase_idx: pc.mu[idx]
            for comp_idx, comp in enumerate(self.components):
                pc.output_props[f"x{comp}_in_{phase}_mass"] = lambda ph=phase_idx, comp_i=comp_idx: pc.x_mass[ph, comp_i]

        self.physics = Compositional(self.components, self.phases, self.timer,
                                     state_spec=Compositional.StateSpecification.PT, n_points=2000, min_p=0.1,
                                     max_p=400.0, min_z=0.0, max_z=1.0, epsilon_z=epsilon, min_t=273.15,
                                     max_t=473.15, extrapolation_flag=True)
        self.physics.add_property_region(pc)

    def set_wells(self):
        cfg = self.config
        upper_segment_length = cfg.well_depth / cfg.well_segments
        segments_lengths = np.concatenate([upper_segment_length * np.ones(cfg.well_segments), np.array([cfg.dz])])

        injector_geometry = PipeGeometry("I1", segments_lengths, cfg.well_diameter, inclination_angle=0.0)
        producer_geometry = PipeGeometry("P1", segments_lengths, cfg.well_diameter, inclination_angle=0.0)
        injector_perforated_segment = injector_geometry.num_segments - 2
        producer_perforated_segment = producer_geometry.num_segments - 2
        self._set_initial_pressure_from_producer_whp(producer_geometry, producer_perforated_segment)
        injector_initial_conditions = self._make_equilibrated_initial_well_conditions("I1", injector_geometry, injector_perforated_segment)
        producer_initial_conditions = self._make_equilibrated_initial_well_conditions("P1", producer_geometry, producer_perforated_segment,
                                                                                     "L", np.array([cfg.z_co2_init, 1.0 - cfg.z_co2_init]))

        self.wells = {
            "I1": Pipe("I1", injector_geometry, self.physics, self.reservoir, injector_initial_conditions,
                       source_sinks=self._make_injection_source_sinks("I1", injector_geometry, 0)),
            "P1": Pipe("P1", producer_geometry, self.physics, self.reservoir, producer_initial_conditions),
        }

        self.reservoir.add_well("I1", ms_well.MS_Type.DFM, well_geometry=injector_geometry)
        self.reservoir.add_perforation("I1", res_cell_idx=self.injector_cell(),
                                       well_seg_idx=injector_geometry.num_segments,
                                       well_diameter=injector_geometry.pipe_ID, well_indexD=None,
                                       with_peaceman_for_dfm_well=True)
        self.reservoir.add_well("P1", ms_well.MS_Type.DFM, well_geometry=producer_geometry)
        self.reservoir.add_perforation("P1", res_cell_idx=self.producer_cell(),
                                       well_seg_idx=producer_geometry.num_segments,
                                       well_diameter=producer_geometry.pipe_ID, well_indexD=None,
                                       with_peaceman_for_dfm_well=True)

    def _producer_wellhead_temperature(self, geometry: PipeGeometry, perforated_segment_local: int) -> float:
        perforation_depth = geometry.TVD_segments[perforated_segment_local] - geometry.TVD_segments[0]
        return float(self.config.t_reservoir - self.config.well_temp_grad * perforation_depth)

    def _set_initial_pressure_from_producer_whp(self, geometry: PipeGeometry, perforated_segment_local: int) -> None:
        cfg = self.config
        initial_conditions = LinearAmbientTemperature(
            pipe_name="P1", pipe_geom=geometry, physics=self.physics, pipe_head_pressure=cfg.producer_whp_bar,
            pipe_head_temperature=self._producer_wellhead_temperature(geometry, perforated_segment_local),
            temp_grad=cfg.well_temp_grad, pipe_head_segment_index=0,
            initial_conditions_dict={
                "phases_names": ["L"],
                "phases_compositions": [[cfg.z_co2_init, 1.0 - cfg.z_co2_init]],
                "pipe_intervals": [[0.0, geometry.pipe_length]],
            }
        )
        states = np.asarray(initial_conditions.initial_conditions_vector, dtype=float).reshape(-1, self.physics.n_vars)
        cfg.p_init = float(states[perforated_segment_local, 0])
        self.producer_initial_target_whp = float(cfg.producer_whp_bar)
        self.derived_initial_reservoir_pressure = float(cfg.p_init)

    def _make_equilibrated_initial_well_conditions(self, well_name: str, geometry: PipeGeometry, perforated_segment_local: int,
                                                   phase_name: str = "G", phase_composition: np.ndarray | None = None) -> LinearAmbientTemperature:
        cfg = self.config
        initial_phase_composition = np.array([1.0 - self.zero, self.zero]) if phase_composition is None else np.asarray(phase_composition)

        def make_initial_conditions(reference_pressure: float, reference_temperature: float):
            return LinearAmbientTemperature(
                pipe_name=well_name, pipe_geom=geometry, physics=self.physics, pipe_head_pressure=reference_pressure,
                pipe_head_temperature=reference_temperature, temp_grad=cfg.well_temp_grad,
                pipe_head_segment_index=geometry.num_segments - 1,
                initial_conditions_dict={
                    "phases_names": [phase_name],
                    "phases_compositions": [initial_phase_composition.tolist()],
                    "pipe_intervals": [[0.0, geometry.pipe_length]],
                }
            )

        reference_pressure = cfg.p_init
        reference_temperature = cfg.t_reservoir
        initial_conditions = make_initial_conditions(reference_pressure, reference_temperature)
        for _ in range(8):
            states = np.asarray(initial_conditions.initial_conditions_vector, dtype=float).reshape(-1, self.physics.n_vars)
            pressure_error = states[perforated_segment_local, 0] - cfg.p_init
            temperature_error = states[perforated_segment_local, 2] - cfg.t_reservoir
            if abs(pressure_error) < 1e-10 and abs(temperature_error) < 1e-10:
                break
            reference_pressure -= pressure_error
            reference_temperature -= temperature_error
            initial_conditions = make_initial_conditions(reference_pressure, reference_temperature)

        setattr(self, f"{well_name.lower()}_initial_well_reference_pressure", float(reference_pressure))
        setattr(self, f"{well_name.lower()}_initial_well_reference_temperature", float(reference_temperature))
        return initial_conditions

    def set_initial_conditions(self):
        cfg = self.config
        input_distribution = {self.physics.vars[0]: cfg.p_init, self.physics.vars[1]: cfg.z_co2_init,
                              self.physics.vars[2]: cfg.t_reservoir}
        self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, input_distribution=input_distribution)

        for well in self.reservoir.wells:
            well.init_state = value_vector(self.wells[well.name].initial_conditions.initial_conditions_vector)

    def set_well_controls(self):
        self.set_producer_whp_control(self.config.producer_whp_bar)

    def set_producer_whp_control(self, target_whp_bar: float):
        producer = self.reservoir.get_well("P1")
        self.physics.set_well_controls(wctrl=producer.control, control_type=well_control_iface.BHP,
                                       is_inj=False, target=float(target_whp_bar))

    def _injection_source_composition(self) -> np.ndarray:
        return np.array([1.0 - self.zero, self.zero], dtype=float)

    def _target_gas_molar_rate_kmol_day(self) -> float:
        pc = self.physics.property_containers[0]
        inj_comp = self._injection_source_composition()
        mw_avg = float(np.dot(np.asarray(pc.Mw[: pc.nc_fl]), inj_comp[: pc.nc_fl]))
        return float(self.config.target_gas_mass_rate_kg_day / mw_avg)

    def _make_injection_source_sinks(self, well_name: str, well_geometry: PipeGeometry,
                                     segment_idx: int) -> dict[str, RampUpRate]:
        cfg = self.config
        inj_fluid_props = {
            "composition": self._injection_source_composition(),
            "phase_name": "G",
            "pressure": float(cfg.injection_source_pressure),
            "temperature": float(cfg.injection_source_temperature),
        }
        return {
            "IsenthalpicInjection": RampUpRate(well_name, well_geometry, self.physics, float(cfg.first_ts),
                                               int(segment_idx), "inflow", self._target_gas_molar_rate_kmol_day(),
                                               float(cfg.injection_ramp_up_period), inj_fluid_props)
        }

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        source = self.wells["I1"].source_sinks["IsenthalpicInjection"]
        if t is not None:
            source.update_current_molar_rate(float(t))

        inj_comp = source.inj_fluid_props["composition"]
        inj_rate = source.current_rate
        component_rate = inj_rate * inj_comp

        well = self.reservoir.get_well("I1")
        global_segment_idx = int(well.well_head_idx) + int(source.segment_idx)
        mw_avg = np.sum(self.physics.property_containers[0].Mw * inj_comp)
        inj_fluid_energy = source.inj_fluid_props["molar_enthalpy"] + self.reservoir.mesh.cell_spe[global_segment_idx] * mw_avg
        inj_energy_rate = inj_rate * inj_fluid_energy

        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        block_start = global_segment_idx * self.physics.n_vars
        rhs_flux[block_start : block_start + self.physics.n_vars] = -np.append(component_rate, inj_energy_rate)
        return rhs_flux

    def initial_well_state_table(self, well_name: str) -> list[dict[str, float]]:
        pipe = self.wells[well_name]
        states = np.asarray(pipe.initial_conditions.initial_conditions_vector, dtype=float).reshape(-1, self.physics.n_vars)
        pc = self.physics.property_containers[0]

        rows = []
        for segment, state in enumerate(states):
            pc.evaluate(state)
            rows.append({"well": well_name, "segment": segment, "pressure_bar": float(state[0]), "z_co2": float(state[1]),
                         "temperature_K": float(pc.temperature), "sG": float(pc.sat[0]), "sL": float(pc.sat[1])})
        return rows
