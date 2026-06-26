from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from darts.engines import ms_well, sim_params, value_vector, well_control_iface
from dartsflash.components import CompData
from dartsflash.libflash import EoS, InitialGuess
from dartsflash.mixtures import DARTSFlash, VLAq
from precondition_reservoir_state import (
    DEFAULT_OUTPUT_DIR as STAGE1_OUTPUT_DIR,
)
from precondition_reservoir_state import (
    ReservoirOnlyPreconditionModel,
    Stage1PreconditionConfig,
    build_config,
)

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

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STAGE1_STATE_PATH = STAGE1_OUTPUT_DIR / "final_reservoir_state_for_dfm.npz"
DEFAULT_STAGE2_OUTPUT_DIR = SCRIPT_DIR / "output_stage2_dfm_thermosiphon"


@dataclass
class Stage2DFMThermosiphonConfig:
    stage1: Stage1PreconditionConfig = field(default_factory=build_config)
    stage1_state_path: Path | str = DEFAULT_STAGE1_STATE_PATH

    output_dir: Path | str = DEFAULT_STAGE2_OUTPUT_DIR
    platform: str = "cpu"

    well_depth: float = 2000.0
    well_segments: int = 40
    well_diameter: float = 0.1524

    injector_bottom_pressure: float = 203.3036
    injector_head_temperature: float = 15.0 + 273.15
    injector_temp_grad: float = 0.034
    injector_initial_phase_name: str = "G"
    injector_initial_phase_composition: tuple[float, float] = (0.999, 0.001)

    producer_bottom_pressure: float = 199.5136
    producer_head_temperature: float = 15.0 + 273.15
    # producer_head_pressure: float = 6.825
    producer_head_pressure: float = 71.426
    producer_temp_grad: float = 0.034
    producer_initial_phase_name: str = "L"
    producer_initial_phase_composition: tuple[float, float] = (0.001, 0.009)
    # producer_initial_phase_name:str = " G"
    # producer_phase_composition: tuple[float, float] = (0.999, 0.001)
    # producer_initial_phase_composition: tuple[float, float] = (0.048, 0.952)

    injector_source_pressure: float = 60.5
    injector_source_temperature: float = 22.0 + 273.15
    injector_target_gas_mass_rate_kg_day: float = 1e5
    injector_ramp_up_period: float = 0.01 / (24.0 * 60.0)

    enable_producer_whp_control: bool = True

    first_ts: float = 1e-7
    mult_ts: float = 2.0
    max_ts: float = 1e-4
    runtime: float = 1.0
    tol_newton: float = 1e-3
    tol_linear: float = 1e-4
    it_newton: int = 12
    it_linear: int = 60

    def __post_init__(self):
        self.stage1_state_path = self._resolve_case_path(self.stage1_state_path)
        self.output_dir = self._resolve_case_path(self.output_dir)

    @staticmethod
    def _resolve_case_path(path: Path | str) -> Path:
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = SCRIPT_DIR / path
        return path.resolve()


class Model(ReservoirOnlyPreconditionModel):
    def __init__(self, config: Stage2DFMThermosiphonConfig | None = None):
        self.stage2_config = config or Stage2DFMThermosiphonConfig()
        super().__init__(self.stage2_config.stage1)
        self.reservoir.grav_acceleration_for_spe = 9.80665
        self.stage1_state = None
        self.anchor_states = {}
        self.wells = {}

        cfg = self.stage2_config
        self.set_sim_params(
            first_ts=cfg.first_ts,
            mult_ts=cfg.mult_ts,
            max_ts=cfg.max_ts,
            runtime=cfg.runtime,
            tol_newton=cfg.tol_newton,
            tol_linear=cfg.tol_linear,
            it_newton=cfg.it_newton,
            it_linear=cfg.it_linear,
            newton_type=sim_params.newton_local_chop,
            coupled_well_res_norm_method=2,
            well_rate_ctrl_absolute_residual_scale=1.0,
            well_rate_ctrl_relative_residual_scale=1e-5,
        )

    def set_physics(self):
        epsilon = 1e-9
        components = ["CO2", "H2O"]
        comp_data = CompData(self.components, setprops=True)

        pc = PropertyContainer(
            phases_name=self.phases,
            components_name=self.components,
            Mw=comp_data.Mw,
            eps_z=epsilon,
            temperature=None,
        )
        pc.flash_ev = VLAq(comp_data, hybrid=True)

        pc.flash_ev.set_vl_eos(
            "PR",
            root_order=[EoS.STABLE],
            trial_comps=[i for i in range(len(components))],
            stability_tol=1e-20,
            switch_tol=1e-2,
            max_iter=50,
            use_gmix=False,
        )
        pc.flash_ev.set_aq_eos(
            "Aq",
            stability_tol=1e-20,
            max_iter=10,
            use_gmix=True,
        )
        pc.flash_ev.init_flash(
            flash_type=DARTSFlash.FlashType.PTFlash,
            eos_order=["VL", "Aq"],
            t_min=270.0,
            t_max=430.0,
            t_init=300.0,
        )
        # pc.flash_ev.init_flash(
        #     flash_type=DARTSFlash.FlashType.NegativeFlash,
        #     eos_order=["VL", "Aq"],
        #     nf_initial_guess=[InitialGuess.Henry_VA],
        # )
        pr = pc.flash_ev.eos["VL"]
        aq = pc.flash_ev.eos["Aq"]

        pc.density_ev = {
            "G": EoSDensity(eos=pr, Mw=comp_data.Mw),
            "L": Garcia2001(self.components),
        }
        # pc.viscosity_ev = {
        #     "G": ConstFunc(0.05),
        #     "L": ConstFunc(0.5),
        # }

        pc.viscosity_ev = {
            "G": Fenghour1998(),
            "L": Islam2012(components),
        }
        pc.rel_perm_ev = {
            "G": PhaseRelPerm("gas", swc=0.20, sgr=0.00, kre=0.95, n=5),
            "L": PhaseRelPerm("oil", swc=0.20, sgr=0.00, kre=1.0, n=6),
        }

        pc.enthalpy_ev = {
            "G": EoSEnthalpy(eos=pr),
            "L": EoSEnthalpy(eos=aq),
        }

        # pc.enthalpy_ev = {
        #     "G": EnthalpyBasic(hcap=37),
        #     "L": EnthalpyBasic(hcap=75.3),
        # } # test numerical stability

        pc.conductivity_ev = {
            "G": ConstFunc(6),
            "L": ConstFunc(60),
        }
        pc.rock_energy_ev = EnthalpyBasic(hcap=1.0)
        pc.IFT_ev = IFT_multicomponent_MCM(self.components)
        pc.output_props = {"temperature": lambda: pc.temperature}

        for phase_idx, phase in enumerate(self.phases):
            pc.output_props[f"s{phase}"] = lambda idx=phase_idx: pc.sat[idx]
            pc.output_props[f"rho{phase}"] = lambda idx=phase_idx: pc.dens[idx]
            pc.output_props[f"mu{phase}"] = lambda idx=phase_idx: pc.mu[idx]
            for comp_idx, comp in enumerate(self.components):
                pc.output_props[f"x{comp}_in_{phase}_mass"] = (
                    lambda ph=phase_idx, comp_i=comp_idx: pc.x_mass[ph, comp_i]
                )

        self.physics = Compositional(
            self.components,
            self.phases,
            self.timer,
            state_spec=Compositional.StateSpecification.PT,
            n_points=200,
            min_p=1.0,
            max_p=400.0,
            min_z=0.0,
            max_z=1.0,
            epsilon_z=epsilon,
            min_t=273.15,
            max_t=473.15,
            extrapolation_flag=True,
        )
        self.physics.add_property_region(pc)

    def set_wells(self):
        cfg = self.stage2_config
        self.wells = {}
        injector_lgr_k = int(self.lgr_patches["inj_lgr"].fine_shape[2])
        self._add_dfm_well(
            well_name="I1",
            lgr_name="inj_lgr",
            lgr_cell_ij=self._well_lgr_cell_ij(),
            lgr_k=injector_lgr_k,
            completion_segments=injector_lgr_k,
            pipe_head_pressure=cfg.injector_bottom_pressure,
            pipe_head_temperature=cfg.injector_head_temperature,
            temp_grad=cfg.injector_temp_grad,
            initial_phase_name=[cfg.injector_initial_phase_name],
            initial_phase_composition=[list(cfg.injector_initial_phase_composition)],
            source_sinks=True,
        )
        self._add_dfm_well(
            well_name="P1",
            lgr_name="prod_lgr",
            lgr_cell_ij=self._well_lgr_cell_ij(),
            lgr_k=1,
            completion_segments=1,
            pipe_head_pressure=cfg.producer_bottom_pressure,
            pipe_head_temperature=cfg.producer_head_temperature,
            temp_grad=cfg.producer_temp_grad,
            initial_phase_name=[cfg.producer_initial_phase_name],
            initial_phase_composition=[list(cfg.producer_initial_phase_composition)],
            source_sinks=False,
        )

    def set_initial_conditions(self):
        self._ensure_stage1_state_loaded()
        state = self.stage1_state["states"]
        input_distribution = {
            self.physics.vars[var_idx]: state[:, var_idx]
            for var_idx in range(self.physics.n_vars)
        }
        self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh,
            input_distribution=input_distribution,
        )

        for well in self.reservoir.wells:
            well.init_state = value_vector(
                self.wells[well.name].initial_conditions.initial_conditions_vector
            )
        # self.reservoir.mesh.volume[self.reservoir.wells[1].well_head_idx] = 1e20
        # self.reservoir.wells[1].init_state[0] = 1

    def set_well_controls(self):
        cfg = self.stage2_config
        if not cfg.enable_producer_whp_control:
            return

        producer = self.reservoir.get_well("P1")
        self.physics.set_well_controls(
            wctrl=producer.control,
            control_type=well_control_iface.BHP,
            is_inj=False,
            target=float(cfg.producer_head_pressure),
        )

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
        inj_fluid_molar_potential_energy = (
            self.reservoir.mesh.cell_spe[global_segment_idx] * mw_avg
        )
        inj_fluid_energy = (
            source.inj_fluid_props["molar_enthalpy"] + inj_fluid_molar_potential_energy
        )
        inj_energy_rate = inj_rate * inj_fluid_energy

        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        block_start = global_segment_idx * self.physics.n_vars
        rhs_flux[block_start : block_start + self.physics.n_vars] = -np.append(
            component_rate,
            inj_energy_rate,
        )
        return rhs_flux

    def _add_dfm_well(
        self,
        well_name: str,
        lgr_name: str,
        lgr_cell_ij: tuple[int, int],
        lgr_k: int,
        completion_segments: int,
        pipe_head_pressure: float,
        pipe_head_temperature: float,
        temp_grad: float,
        initial_phase_name: list[str],
        initial_phase_composition: list[list],
        source_sinks: bool,
    ):
        cfg = self.stage2_config
        upper_segment_length = cfg.well_depth / cfg.well_segments
        max_lgr_layers = int(self.lgr_patches[lgr_name].fine_shape[2])
        if not 1 <= int(lgr_k) <= max_lgr_layers:
            raise ValueError(
                f"{well_name} lgr_k={lgr_k} is outside the valid LGR layer range "
                f"[1, {max_lgr_layers}] for {lgr_name}."
            )
        if int(completion_segments) < int(lgr_k):
            raise ValueError(
                f"{well_name} completion_segments={completion_segments} cannot reach "
                f"perforation layer lgr_k={lgr_k}."
            )
        completion_segment_length = cfg.stage1.dz / cfg.stage1.lgr_refine[2]
        segments_lengths = np.concatenate(
            [
                upper_segment_length * np.ones(cfg.well_segments),
                completion_segment_length * np.ones(int(completion_segments)),
            ]
        )

        well_geometry = PipeGeometry(
            well_name,
            segments_lengths,
            cfg.well_diameter,
            inclination_angle=0.0,
        )
        if well_name == "I1":
            initial_conditions = LinearAmbientTemperature(
                pipe_name=well_name,
                pipe_geom=well_geometry,
                physics=self.physics,
                pipe_head_pressure=pipe_head_pressure,
                pipe_head_temperature=pipe_head_temperature,
                temp_grad=temp_grad,
                pipe_head_segment_index=well_geometry.num_segments-1,
                initial_conditions_dict={
                    "phases_names": initial_phase_name,
                    "phases_compositions": initial_phase_composition,
                    "pipe_intervals": [
                        [0.0, well_geometry.pipe_length],
                    ],
                },
            )

        else:
            initial_conditions = LinearAmbientTemperature(
                pipe_name=well_name,
                pipe_geom=well_geometry,
                physics=self.physics,
                pipe_head_pressure=pipe_head_pressure,
                pipe_head_temperature=pipe_head_temperature,
                temp_grad=temp_grad,
                pipe_head_segment_index=well_geometry.num_segments-1,
                initial_conditions_dict={
                    "phases_names": initial_phase_name,
                    "phases_compositions": initial_phase_composition,
                    "pipe_intervals": [
                        [0.0, well_geometry.pipe_length],
                    ]
                }
            )

        well_source_sinks = None
        if source_sinks:
            well_source_sinks = self._make_injection_source_sinks(
                well_name,
                well_geometry,
            )
        self.wells[well_name] = Pipe(
            well_name,
            well_geometry,
            self.physics,
            self.reservoir,
            initial_conditions,
            source_sinks=well_source_sinks,
        )

        self.reservoir.add_well(
            well_name,
            ms_well.MS_Type.DFM,
            well_geometry=well_geometry,
        )
        lgr_i, lgr_j = lgr_cell_ij
        self.reservoir.add_perforation(
            well_name,
            lgr_name=lgr_name,
            lgr_cell_idx=(lgr_i, lgr_j, lgr_k),
            well_seg_idx=well_geometry.num_segments,
            well_diameter=well_geometry.pipe_ID,
            well_indexD=None,
            with_peaceman_for_coupled_well_reservoir=True,
        )

    def _make_injection_source_sinks(
        self,
        well_name: str,
        well_geometry: PipeGeometry,
    ) -> dict[str, RampUpRate]:
        cfg = self.stage2_config
        inj_comp = np.asarray(cfg.injector_initial_phase_composition, dtype=float)
        inj_fluid_props = {
            "composition": inj_comp,
            "phase_name": "G",
            "pressure": float(cfg.injector_source_pressure),
            "temperature": float(cfg.injector_source_temperature),
        }
        return {
            "IsenthalpicInjection": RampUpRate(
                well_name,
                well_geometry,
                self.physics,
                float(cfg.first_ts),
                0,
                "inflow",
                self._target_gas_molar_rate_kmol_day(),
                float(cfg.injector_ramp_up_period),
                inj_fluid_props,
            )
        }

    def _target_gas_molar_rate_kmol_day(self) -> float:
        cfg = self.stage2_config
        pc = self.physics.property_containers[0]
        inj_comp = np.asarray(cfg.injector_initial_phase_composition, dtype=float)
        mw_avg = float(np.dot(np.asarray(pc.Mw[: pc.nc_fl]), inj_comp[: pc.nc_fl]))
        return float(cfg.injector_target_gas_mass_rate_kg_day / mw_avg)

    def _ensure_stage1_state_loaded(self) -> None:
        if self.stage1_state is not None:
            return
        self.stage1_state = self._load_stage1_state()
        self.anchor_states = self._stage2_anchor_states()

    def _stage2_anchor_states(self) -> dict[str, dict[str, float]]:
        state = self.stage1_state["states"]
        satg = self.stage1_state["satG"]
        lgr_i, lgr_j = self._well_lgr_cell_ij()
        anchors = {
            "I1": ("inj_lgr", int(self.lgr_patches["inj_lgr"].fine_shape[2])),
            "P1": ("prod_lgr", 1),
        }
        rows = {}
        for well_name, (lgr_name, lgr_k) in anchors.items():
            cell_idx = self.reservoir.resolve_cell_index(
                res_cell_idx=None,
                lgr_name=lgr_name,
                lgr_cell_idx=(lgr_i, lgr_j, lgr_k),
            )
            rows[well_name] = {
                "cell_index": int(cell_idx),
                "lgr_name": lgr_name,
                "lgr_k": int(lgr_k),
                "pressure_bar": float(state[cell_idx, 0]),
                "z_co2": float(state[cell_idx, 1]),
                "temperature_K": float(state[cell_idx, 2]),
                "satG": float(satg[cell_idx]),
            }
        return rows

    def _load_stage1_state(self) -> dict[str, np.ndarray]:
        path = Path(self.stage2_config.stage1_state_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Stage 1 state file was not found: {path}. "
                "Run precondition_reservoir_state.py first or update "
                "Stage2DFMThermosiphonConfig.stage1_state_path."
            )

        with np.load(path, allow_pickle=True) as data:
            state = {key: np.asarray(data[key]) for key in data.files}

        variable_names = [str(name) for name in state["variable_names"]]
        if variable_names != self.physics.vars:
            raise ValueError(
                "Stage 1 variable order does not match Stage 2 physics: "
                f"{variable_names} != {self.physics.vars}"
            )

        n_res = self.reservoir.mesh.n_res_blocks
        expected_shape = (n_res, self.physics.n_vars)
        if tuple(state["states"].shape) != expected_shape:
            raise ValueError(
                f"Stage 1 states have shape {state['states'].shape}, "
                f"expected {expected_shape}."
            )

        current_centers = np.asarray(
            [cell.center for cell in self.reservoir.cells[:n_res]],
            dtype=float,
        )
        if not np.allclose(state["cell_centers"], current_centers, atol=1e-8):
            raise ValueError(
                "Stage 1 cell centers do not match the Stage 2 reservoir mesh. "
                "Use the same grid, LGR, and boundary setup in both stages."
            )

        return state
