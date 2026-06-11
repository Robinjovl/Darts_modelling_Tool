from dataclasses import dataclass

import numpy as np
from darts.engines import (
    ms_well,
    sim_params,
    value_vector,
    well_control_iface,
)
from dartsflash.components import CompData
from dartsflash.libflash import EoS
from dartsflash.mixtures import DARTSFlash, VLAq

from darts.models.cicd_model import CICDModel
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
from darts.reservoirs.struct_reservoir_with_lgr import LGRPatch, StructReservoirWithLGR


@dataclass
class Stage1ThermosiphonRateConfig:
    p_init: float = 243.71
    t_reservoir: float = 356.15
    sG_init_target: float | None = 0.5
    z_co2_init: float = 1e-8

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
    lgr_refine: tuple[int, int, int] = (5, 5, 1)
    injector_parent_i: int | None = None
    producer_parent_i: int | None = None
    well_parent_j: int | None = None

    well_depth: float = 2000.0
    well_segments: int = 40
    well_diameter: float = 0.1524
    initial_bhp: float = 243.71
    initial_whp: float = 100
    producer_whp: float = 1
    target_gas_mass_rate_kg_day: float = 1e6
    injection_source_pressure: float = 60.5
    injection_source_temperature: float = 22 + 273.15
    injection_ramp_up_period: float = 0.01 / (24 * 60)
    surface_pressure_loss: float = 0.0
    injector_head_temperature: float = 15 + 273.15
    producer_head_temperature: float = 15 + 273.15
    injector_temp_grad: float = 0.034
    producer_temp_grad: float = 0.034

    first_ts: float = 1e-7
    max_ts: float = 1e-4
    runtime: float = 0.01


class Model(CICDModel):
    def __init__(self, config: Stage1ThermosiphonRateConfig | None = None):
        super().__init__()
        self.config = config or Stage1ThermosiphonRateConfig()
        self.zero = 1e-8
        self.components = ["CO2", "H2O"]
        self.phases = ["G", "L"]
        self.wells = {}
        self._resolved_z_co2_init = None

        self.timer.node["initialization"].start()
        self.set_reservoir()
        self.reservoir.grav_acceleration_for_spe = 9.80665
        self.set_physics()
        self.set_sim_params(
            first_ts=self.config.first_ts,
            mult_ts=2,
            max_ts=self.config.max_ts,
            runtime=self.config.runtime,
            tol_newton=1e-3,
            tol_linear=1e-4,
            it_newton=12,
            it_linear=60,
            newton_type=sim_params.newton_local_chop,
            coupled_well_res_norm_method=2,
            well_rate_ctrl_absolute_residual_scale=1.0,
            well_rate_ctrl_relative_residual_scale=1e-5,
        )
        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        cfg = self.config
        parent = StructReservoir(
            self.timer,
            nx=cfg.nx,
            ny=cfg.ny,
            nz=cfg.nz,
            dx=cfg.dx,
            dy=cfg.dy,
            dz=cfg.dz,
            permx=cfg.permx,
            permy=cfg.permy,
            permz=cfg.permz,
            poro=cfg.poro,
            depth=None,
            start_z=2000.0,
            hcap=2200.0,
            rcond=2.1 * 86.4,
        )
        parent.boundary_volumes["yz_minus"] = 1e20
        parent.boundary_volumes["yz_plus"] = 1e20
        parent.boundary_volumes["xz_minus"] = 1e20
        parent.boundary_volumes["xz_plus"] = 1e20
        injector_i, producer_i, well_j = self._well_parent_cell_indices()
        self.lgr_patches = {
            "inj_lgr": LGRPatch(
                "inj_lgr",
                (injector_i, injector_i),
                (well_j, well_j),
                (1, 1),
                cfg.lgr_refine,
            ),
            "prod_lgr": LGRPatch(
                "prod_lgr",
                (producer_i, producer_i),
                (well_j, well_j),
                (1, 1),
                cfg.lgr_refine,
            ),
        }
        lgrs = [
            self.lgr_patches["inj_lgr"],
            self.lgr_patches["prod_lgr"],
        ]
        self.reservoir = StructReservoirWithLGR(
            self.timer,
            parent,
            lgrs,
            lgr_coarse_fine_tran_mode="flow_based",
        )

    def _well_parent_cell_indices(self) -> tuple[int, int, int]:
        cfg = self.config
        injector_i = cfg.injector_parent_i or cfg.nx // 4 + 1
        producer_i = cfg.producer_parent_i or cfg.nx - cfg.nx // 4
        well_j = cfg.well_parent_j or cfg.ny // 2 + 1

        for name, value, upper in (
            ("injector_parent_i", injector_i, cfg.nx),
            ("producer_parent_i", producer_i, cfg.nx),
            ("well_parent_j", well_j, cfg.ny),
        ):
            if not 1 <= value <= upper:
                raise ValueError(f"{name}={value} is outside the parent grid.")

        if injector_i == producer_i:
            raise ValueError("Injector and producer parent cells must be different.")

        return injector_i, producer_i, well_j

    def _well_lgr_cell_ij(self) -> tuple[int, int]:
        refine_i, refine_j, _ = self.config.lgr_refine
        return refine_i // 2 + 1, refine_j // 2 + 1

    def set_physics(self):
        epsilon = 1e-9
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
            trial_comps=[i for i in range(len(self.components))],
            stability_tol=1e-20,
            switch_tol=1e-2,
            max_iter=50,
            use_gmix=False,
        )
        pc.flash_ev.set_aq_eos("Aq", stability_tol=1e-20, max_iter=10, use_gmix=True)
        pc.flash_ev.init_flash(
            flash_type=DARTSFlash.FlashType.PTFlash,
            eos_order=["VL", "Aq"],
            t_min=270.0,
            t_max=430.0,
            t_init=300.0,
        )
        pr = pc.flash_ev.eos["VL"]
        aq = pc.flash_ev.eos["Aq"]

        pc.density_ev = {
            "G": EoSDensity(eos=pr, Mw=comp_data.Mw),
            "L": Garcia2001(self.components),
        }
        pc.viscosity_ev = {
            "G": Fenghour1998(),
            "L": Islam2012(self.components),
        }
        pc.rel_perm_ev = {
            "G": PhaseRelPerm("gas", swc=0.20, sgr=0.00, kre=0.95, n=5),
            "L": PhaseRelPerm("oil", swc=0.20, sgr=0.00, kre=1.0, n=6),
        }
        pc.enthalpy_ev = {
            "G": EoSEnthalpy(eos=pr),
            "L": EoSEnthalpy(eos=aq),
        }
        pc.conductivity_ev = {
            "G": ConstFunc(6),
            "L": ConstFunc(60),
        }
        pc.rock_energy_ev = EnthalpyBasic(hcap=1.0)
        pc.IFT_ev = IFT_multicomponent_MCM(self.components)

        pc.output_props = {"temperature": lambda: pc.temperature}
        for j, ph in enumerate(self.phases):
            pc.output_props["s" + ph] = lambda jj=j: pc.sat[jj]
            pc.output_props["rho" + ph] = lambda jj=j: pc.dens[jj]
            pc.output_props["miu" + ph] = lambda jj=j: pc.mu[jj]
            for i, comp in enumerate(self.components):
                pc.output_props[f"x{comp}_in_{ph}_mass"] = lambda jj=j, ii=i: pc.x_mass[
                    jj, ii
                ]

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
        cfg = self.config
        self.wells = {}
        self._add_dfm_well(
            well_name="I1",
            lgr_name="inj_lgr",
            lgr_cell_ij=self._well_lgr_cell_ij(),
            pipe_head_temperature=cfg.injector_head_temperature,
            pipe_head_pressure=cfg.initial_bhp,
            temp_grad=cfg.injector_temp_grad,
            initial_phase_name=["G"],
            initial_phase_composition=[
                [1 - self.zero, self.zero],
            ],
        )
        self._add_dfm_well(
            well_name="P1",
            lgr_name="prod_lgr",
            lgr_cell_ij=self._well_lgr_cell_ij(),
            pipe_head_temperature=cfg.producer_head_temperature,
            pipe_head_pressure=cfg.initial_whp,
            temp_grad=cfg.producer_temp_grad,
            initial_phase_name=["G"],
            initial_phase_composition=[
                [1 - self.zero, self.zero],
            ],
        )


    def _add_dfm_well(
        self,
        well_name: str,
        lgr_name: str,
        lgr_cell_ij: tuple[int, int],
        pipe_head_pressure: float,
        pipe_head_temperature: float,
        temp_grad: float,
        initial_phase_name: list[str],
        initial_phase_composition: list[list],
    ):
        cfg = self.config
        upper_segment_length = cfg.well_depth / cfg.well_segments
        completion_layers = int(self.lgr_patches[lgr_name].fine_shape[2])
        completion_segment_length = cfg.dz / cfg.lgr_refine[2]
        segments_lengths = np.concatenate(
            [
                upper_segment_length * np.ones(cfg.well_segments),
                completion_segment_length * np.ones(completion_layers),
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
                pipe_head_segment_index=well_geometry.num_segments - 1,
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
                pipe_head_segment_index=0,
                initial_conditions_dict={
                    "phases_names": initial_phase_name,
                    "phases_compositions": initial_phase_composition,
                    "pipe_intervals": [
                        [0.0, well_geometry.pipe_length],
                    ],
                },
            )

        well_source_sinks = self._make_injection_source_sinks(well_name, well_geometry)
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
        first_completion_segment = well_geometry.num_segments - completion_layers + 1
        for lgr_k in range(1, completion_layers + 1):
            self.reservoir.add_perforation(
                well_name,
                lgr_name=lgr_name,
                lgr_cell_idx=(lgr_i, lgr_j, lgr_k),
                well_seg_idx=first_completion_segment + lgr_k - 1,
                well_diameter=well_geometry.pipe_ID,
                well_indexD=None,
                with_peaceman_for_coupled_well_reservoir=True,
            )

    def set_initial_conditions(self):
        cfg = self.config
        input_distribution = {
            self.physics.vars[0]: cfg.p_init,
            self.physics.vars[1]: self.resolved_z_co2_init(),
            self.physics.vars[2]: cfg.t_reservoir,
        }
        self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh,
            input_distribution=input_distribution,
        )

        for well in self.reservoir.wells:
            well.init_state = value_vector(
                self.wells[well.name].initial_conditions.initial_conditions_vector
            )
        self.reservoir.mesh.volume[self.reservoir.wells[1].well_head_idx] = 1e20
        self.reservoir.wells[1].init_state[0] = 1

    def resolved_z_co2_init(self):
        if self._resolved_z_co2_init is not None:
            return self._resolved_z_co2_init

        if self.config.sG_init_target is None:
            self._resolved_z_co2_init = self.config.z_co2_init
        else:
            self._resolved_z_co2_init = self._z_co2_for_target_sg(
                self.config.sG_init_target
            )
        return self._resolved_z_co2_init

    def _evaluate_sg_for_z(self, z_co2):
        pc = self.physics.property_containers[0]
        state = value_vector([self.config.p_init, z_co2, self.config.t_reservoir])
        pc.evaluate(state)
        return float(pc.sat[0])

    def _z_co2_for_target_sg(self, target_sg):
        if not 0.0 <= target_sg <= 1.0:
            raise ValueError("sG_init_target must be between 0 and 1")

        z_grid = np.linspace(self.zero, 1.0 - self.zero, 401)
        sg_grid = np.array([self._evaluate_sg_for_z(float(z)) for z in z_grid])

        best_idx = int(np.argmin(np.abs(sg_grid - target_sg)))
        best_z = float(z_grid[best_idx])
        best_sg = float(sg_grid[best_idx])

        bracket = None
        for i in range(len(z_grid) - 1):
            sg_l = sg_grid[i] - target_sg
            sg_r = sg_grid[i + 1] - target_sg
            if sg_l == 0:
                self.actual_sG_init_from_target = float(sg_grid[i])
                return float(z_grid[i])
            if sg_l * sg_r <= 0:
                bracket = (float(z_grid[i]), float(z_grid[i + 1]))
                break

        if bracket is None:
            print(
                "WARNING: requested sG_init_target="
                f"{target_sg:g} is not reachable at P={self.config.p_init:g} bar, "
                f"T={self.config.t_reservoir:g} K. Closest flash result is "
                f"sG={best_sg:g} at z_CO2={best_z:g}."
            )
            self.actual_sG_init_from_target = best_sg
            return best_z

        z_l, z_r = bracket
        for _ in range(60):
            z_m = 0.5 * (z_l + z_r)
            sg_l = self._evaluate_sg_for_z(z_l) - target_sg
            sg_m = self._evaluate_sg_for_z(z_m) - target_sg
            if abs(sg_m) < 1e-8:
                break
            if sg_l * sg_m <= 0:
                z_r = z_m
            else:
                z_l = z_m

        z = 0.5 * (z_l + z_r)
        self.actual_sG_init_from_target = self._evaluate_sg_for_z(z)
        return float(z)

    def _injection_source_composition(self):
        return np.array([1.0 - self.zero, self.zero], dtype=float)

    def _target_gas_molar_rate_kmol_day(self):
        pc = self.physics.property_containers[0]
        inj_comp = self._injection_source_composition()
        mw_avg = float(np.dot(np.asarray(pc.Mw[: pc.nc_fl]), inj_comp[: pc.nc_fl]))
        return float(self.config.target_gas_mass_rate_kg_day / mw_avg)

    def _make_injection_source_sinks(self, well_name, well_geometry):
        if well_name != "I1":
            return None

        cfg = self.config
        inj_fluid_props = {
            "composition": self._injection_source_composition(),
            "phase_name": "G",
            "pressure": float(cfg.injection_source_pressure),
            "temperature": float(cfg.injection_source_temperature),
        }
        return {
            "IsenthalpicInjection": RampUpRate(
                well_name,
                well_geometry,
                self.physics,
                float(self.data_ts.dt_first),
                0,
                "inflow",
                self._target_gas_molar_rate_kmol_day(),
                float(cfg.injection_ramp_up_period),
                inj_fluid_props,
            )
        }

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        source = self.wells["I1"].source_sinks["IsenthalpicInjection"]
        if t is not None:
            source.update_current_molar_rate(float(t))

        inj_comp = source.inj_fluid_props["composition"]
        inj_rate = source.current_rate
        component_rate = inj_rate * inj_comp

        inj_segment_idx = source.segment_idx
        well = next(w for w in self.reservoir.wells if w.name == "I1")
        global_segment_idx = well.well_head_idx + inj_segment_idx

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
            component_rate, inj_energy_rate
        )
        return rhs_flux

    def set_well_controls(self):
        cfg = self.config

        for well in self.reservoir.wells:
            if well.name == "P1":
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=cfg.producer_whp,
                )
