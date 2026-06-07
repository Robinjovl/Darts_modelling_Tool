from dataclasses import dataclass

import numpy as np
from darts.engines import (
    ms_well,
    sim_params,
    timer_node,
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
from darts.pipes.set_initial_conditions import LinearAmbientTemperature
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.struct_reservoir_with_lgr import LGRPatch, StructReservoirWithLGR


@dataclass
class Stage1ThermosiphonConfig:
    p_init: float = 200.0
    t_reservoir: float = 356.15
    sG_init_target: float | None = 0.5
    z_co2_init: float = 1e-8

    nx: int = 20
    ny: int = 1
    nz: int = 10
    dx: float = 30.0
    dy: float = 30.0
    dz: float = 10.0
    depth: float = 2005.0
    permx: float = 800.0
    permy: float = 800.0
    permz: float = 80.0
    poro: float = 0.2
    lgr_refine: tuple[int, int, int] = (5, 5, 1)

    well_depth: float = 2000.0
    well_segments: int = 40
    well_diameter: float = 0.1524
    equal_whp: float = 100
    surface_pressure_loss: float = 0.0
    injector_head_temperature: float = 22 + 273.15
    producer_head_temperature: float = 83 + 273.15
    injector_temp_grad: float = 0.034
    producer_temp_grad: float = 0.0

    first_ts: float = 1e-7
    max_ts: float = 1e-4
    runtime: float = 0.01


class Model(CICDModel):
    def __init__(self, config: Stage1ThermosiphonConfig | None = None):
        super().__init__()
        self.config = config or Stage1ThermosiphonConfig()
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
        lgrs = [
            LGRPatch("inj_lgr", (5, 5), (1, 1), (1, 10), cfg.lgr_refine),
            LGRPatch("prod_lgr", (15, 15), (1, 1), (1, 10), cfg.lgr_refine),
        ]
        self.reservoir = StructReservoirWithLGR(
            self.timer,
            parent,
            lgrs,
            lgr_coarse_fine_tran_mode="flow_based",
        )

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
                pc.output_props[f"x{comp}_in_{ph}_mass"] = (
                    lambda jj=j, ii=i: pc.x_mass[jj, ii]
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
        cfg = self.config
        self.wells = {}
        self._add_dfm_well(
            well_name="I1",
            lgr_name="inj_lgr",
            lgr_cell_ij=(3, 3),
            pipe_head_temperature=cfg.injector_head_temperature,
            pipe_head_pressure=cfg.equal_whp,
            temp_grad=cfg.injector_temp_grad,
            initial_phase_name="G",
            initial_phase_composition=[0.99, 0.01],
        )
        self._add_dfm_well(
            well_name="P1",
            lgr_name="prod_lgr",
            lgr_cell_ij=(3, 3),
            pipe_head_temperature=cfg.producer_head_temperature,
            pipe_head_pressure=cfg.equal_whp,
            temp_grad=cfg.producer_temp_grad,
            initial_phase_name="G",
            initial_phase_composition=[0.99, 0.01],
        )

    def _add_dfm_well(
        self,
        well_name: str,
        lgr_name: str,
        lgr_cell_ij: tuple[int, int],
        pipe_head_pressure: float,
        pipe_head_temperature: float,
        temp_grad: float,
        initial_phase_name: str,
        initial_phase_composition: list[float],
    ):
        cfg = self.config
        upper_segment_length = cfg.well_depth / cfg.well_segments
        completion_layers = int(self.reservoir.lgr_cell_maps[lgr_name].shape[2])
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
                pipe_head_segment_index=0,
                initial_conditions_dict={
                    "phases_names": [initial_phase_name],
                    "phases_compositions": [initial_phase_composition],
                    "pipe_intervals": [[0.0, well_geometry.pipe_length]],
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
                    "phases_names": [initial_phase_name],
                    "phases_compositions":  [initial_phase_composition],
                    "pipe_intervals": [[0.0, well_geometry.pipe_length]],
                },
            )
        self.wells[well_name] = Pipe(
            well_name,
            well_geometry,
            self.physics,
            self.reservoir,
            initial_conditions,
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

    def set_well_controls(self):
        cfg = self.config
        injector_whp = 100.0
        producer_whp = 100.0
        injection_composition = [1.0 - self.zero]

        for well in self.reservoir.wells:
            if well.name == "I1":
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.BHP,
                    is_inj=True,
                    target=injector_whp,
                    phase_name="G",
                    inj_composition=injection_composition,
                    inj_temp=22+273.15,
                )
            elif well.name == "P1":
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=producer_whp,
                )

    def initial_phase_state(self):
        pc = self.physics.property_containers[0]
        pc.evaluate(
            value_vector(
                [
                    self.config.p_init,
                    self.resolved_z_co2_init(),
                    self.config.t_reservoir,
                ]
            )
        )
        return {
            "sG_init_target": (
                np.nan
                if self.config.sG_init_target is None
                else float(self.config.sG_init_target)
            ),
            "z_co2_init_resolved": float(self.resolved_z_co2_init()),
            "sG_init": float(pc.sat[0]),
            "sL_init": float(pc.sat[1]),
            "rhoG_init": float(pc.dens[0]),
            "rhoL_init": float(pc.dens[1]),
            "muG_init": float(pc.mu[0]),
            "muL_init": float(pc.mu[1]),
        }
