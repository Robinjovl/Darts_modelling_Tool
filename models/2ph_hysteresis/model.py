from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from darts.engines import value_vector
from darts.models.darts_model import DartsModel
from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from darts.physics.properties.flash import ConstantK
from darts.physics.properties.hysteresis import (
    KilloughCapillaryPressureTable,
    KilloughRelPermCorey,
)
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.reservoirs.struct_reservoir import StructReservoir
from dartsflash.components import CompData
from dartsflash.libflash import AQEoS, CubicEoS, FlashParams


@dataclass
class Corey:
    Pc_drainage_section: str
    Pc_imbibition_section: str
    nowetting_d: str
    nowetting_i: str
    wetting_type: str
    nw: float
    ng: float
    swc: float
    sgc: float
    krwe: float
    krge: float
    labda: float
    p_entry: float
    pcmax: float
    c2: float
    sgrmax: float
    a: float


def default_corey_regions() -> dict[int, Corey]:
    base = dict(
        sgrmax=0.40,
        Pc_drainage_section="Pc_drainage",
        Pc_imbibition_section="Pc_imbibition",
        nowetting_d="nonwetting_drainage_kr",
        nowetting_i="nonwetting_imbibition_kr",
        wetting_type="wetting_kr",
        nw=2.0,
        ng=1.5,
        swc=0.2,
        sgc=0.1,
        krwe=1.0,
        krge=0.8,
        labda=2.0,
        p_entry=2.0,
        pcmax=30.0,
        c2=1.5,
        a=0.8,
    )
    return {0: Corey(**base)}


class Model(DartsModel):
    def __init__(self, hys: bool = True):
        super().__init__()
        self.hys = hys
        self.prod = True
        self.rate_rhs = True
        self.zero = 1e-12
        self.components = ["H2O", "CO2"]
        self.temperature = 338.15
        self.producer_bhp = 250.0
        self.injection_rate = 6.734006734006734e-05 * 3600.0 * 24.0
        self.well_centers = {"I1": [0.0, 0.0, 0.5]}
        self.lookup_file = str(Path(__file__).with_name("LookupTable.txt"))
        self.corey = default_corey_regions()
        self.initial_z_h2o = 1.0 - self.zero

    def setup_case(
        self,
        nx: int = 100,
        n_points: int = 1000,
        temperature: float = 338.15,
        zero: float = 1e-12,
        producer_bhp: float = 250.0,
        injection_rate: float = 6.734006734006734e-05 * 3600.0 * 24.0,
        initial_z_h2o: float | None = None,
        injection_stream: dict[str, float] | None = None,
        components: list[str] | None = None,
        corey_regions: dict[int, Corey] | None = None,
        logscale: bool = False,
    ) -> None:
        self.zero = zero
        self.temperature = temperature
        self.producer_bhp = producer_bhp
        self.injection_rate = injection_rate
        self.components = list(components or ["H2O", "CO2"])
        self.corey = corey_regions or default_corey_regions()
        self.initial_z_h2o = (
            float(initial_z_h2o)
            if initial_z_h2o is not None
            else 1.0 - zero
        )

        self.set_reservoir(
            nx=nx,
            logscale=logscale,
        )
        self.set_physics(
            corey_regions=self.corey,
            n_points=n_points,
            components=self.components,
            temperature=temperature,
        )
        self.inj_stream = self.build_injection_stream(
            injection_stream=injection_stream,
            zero=zero,
        )
        self.inj_rate = [injection_rate, 0.0]
        self.p_prod = producer_bhp

    def build_injection_stream(
        self,
        injection_stream: dict[str, float] | None,
        zero: float,
    ) -> dict[int, list[float]]:
        if injection_stream is None:
            injection_stream = {"H2O": zero, "CO2": 1.0 - zero}

        z_h2o = float(injection_stream["H2O"])
        return {
            0: [z_h2o],
            1: [1.0 - z_h2o],
        }

    def set_reservoir(
        self,
        nx: int = 100,
        logscale: bool = False,
    ) -> None:
        ny = 1
        nz = 1
        n_blocks = nx * ny * nz
        length = 100.0

        if logscale:
            x_edges = np.logspace(-4, np.log10(length), nx + 1)
        else:
            x_edges = np.linspace(0.0, length, nx + 1)

        dx = x_edges[1:] - x_edges[:-1]
        dy = np.ones(nx)
        dz = 1.0
        depth = np.zeros(n_blocks)
        actnum = np.ones((nx, ny, nz))
        op_num = np.zeros(n_blocks, dtype=int)

        for i in range(nx):
            global_idx = i
            depth[global_idx] = 50.0
            op_num[global_idx] = 0

        self.reservoir = StructReservoir(
            self.timer,
            nx,
            ny,
            nz,
            dx=dx,
            dy=dy,
            dz=dz,
            permx=40,
            permy=40,
            permz=40,
            hcap=0,
            rcond=0,
            poro=0.2,
            op_num=op_num,
            depth=depth,
            actnum=actnum,
        )
        self.op_num = np.asarray(op_num, dtype=int)

    def set_wells(self, verbose: bool = False) -> None:
        del verbose
        self.well_cells = []
        for center in self.well_centers.values():
            self.well_cells.append(self.reservoir.find_cell_index(center))

        if self.prod:
            self.reservoir.add_well("P1")
            self.reservoir.add_perforation(
                "P1",
                res_cell_idx=(self.reservoir.nx, 1, self.reservoir.nz),
                well_index=10000,
                well_indexD=0,
            )

    def set_physics(
        self,
        corey_regions: dict[int, Corey],
        n_points: int,
        components: list[str],
        temperature: float,
        zero: float | None = None,
        lookup_file: str | None = None,
    ) -> None:
        zero = self.zero if zero is None else zero
        lookup_file = lookup_file or str(Path(__file__).with_name("LookupTable.txt"))
        self.lookup_file = lookup_file
        phases = ["Aq", "V"]
        comp_data = CompData(components, setprops=True)

        pr = CubicEoS(comp_data, CubicEoS.PR)
        aq = AQEoS(
            comp_data,
            {
                AQEoS.water: AQEoS.Jager2003,
                AQEoS.solute: AQEoS.Ziabakhsh2012,
            },
        )

        flash_params = FlashParams(comp_data)
        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["AQ", "PR"]

        history_kwargs = {}
        if self.hys:
            history_kwargs = {
                "history_labels": ["sg_max"],
                "history_axes_min": [0.0],
                "history_axes_max": [1.0],
                "history_n_axes_points": [n_points],
                "history_defaults": {"sg_max": 0.0},
                "hysteresis_enabled": True,
            }

        self.physics = Compositional(
            components,
            phases,
            self.timer,
            n_points,
            min_p=1,
            max_p=500,
            min_z=zero / 10.0,
            max_z=1.0 - zero / 10.0,
            epsilon_z=zero / 10.0,
            state_spec=Compositional.StateSpecification.P,
            cache=False,
            axes_min=[1.0, zero / 10.0],
            axes_max=[500.0, 1.0 - zero / 10.0],
            n_axes_points=[n_points, n_points],
            **history_kwargs,
        )

        for region, params in corey_regions.items():
            property_container = PropertyContainer(
                phases_name=phases,
                components_name=components,
                Mw=comp_data.Mw,
                temperature=temperature,
                rock_comp=0,
                eps_z=zero / 10.0,
                history_labels=self.physics.history_labels,
            )
            property_container.flash_ev = ConstantK(
                len(components),
                [1.0 / 0.011406373765724964, 1.0 / 67.29641667035624],
                zero,
            )
            property_container.density_ev = {
                "V": EoSDensity(pr, comp_data.Mw),
                "Aq": Garcia2001(components),
            }
            property_container.viscosity_ev = {
                "V": Fenghour1998(),
                "Aq": Islam2012(components),
            }
            property_container.rel_perm_ev = {
                "V": KilloughRelPermCorey(params, "gas"),
                "Aq": KilloughRelPermCorey(params, "water"),
            }
            property_container.capillary_pressure_ev = {
                "V": KilloughCapillaryPressureTable(
                    params,
                    "gas",
                    lookup_file=lookup_file,
                ),
                "Aq": KilloughCapillaryPressureTable(
                    params,
                    "Aq",
                    lookup_file=lookup_file,
                ),
            }
            property_container.enthalpy_ev = {
                "V": EoSEnthalpy(pr),
                "Aq": EoSEnthalpy(aq),
            }
            property_container.conductivity_ev = {
                "V": ConstFunc(0.0),
                "Aq": ConstFunc(0.0),
            }
            property_container.output_props = {
                "sat_Aq": lambda ii=region: self.physics.property_containers[ii].sat[0],
                "sat_V": lambda ii=region: self.physics.property_containers[ii].sat[1],
                "dens_Aq": lambda ii=region: self.physics.property_containers[ii].dens[0],
                "dens_V": lambda ii=region: self.physics.property_containers[ii].dens[1],
                "mu_Aq": lambda ii=region: self.physics.property_containers[ii].mu[0],
                "mu_V": lambda ii=region: self.physics.property_containers[ii].mu[1],
                "x_Aq_H2O": lambda ii=region: self.physics.property_containers[ii].x[0, 0],
                "x_Aq_CO2": lambda ii=region: self.physics.property_containers[ii].x[0, 1],
                "x_V_H2O": lambda ii=region: self.physics.property_containers[ii].x[1, 0],
                "x_V_CO2": lambda ii=region: self.physics.property_containers[ii].x[1, 1],
                "kr_Aq": lambda ii=region: self.physics.property_containers[ii].kr[0],
                "kr_V": lambda ii=region: self.physics.property_containers[ii].kr[1],
                "pc_Aq": lambda ii=region: self.physics.property_containers[ii].pc[0],
                "pc_V": lambda ii=region: self.physics.property_containers[ii].pc[1],
            }
            self.physics.add_property_region(property_container, region)

    def set_initial_conditions(self):
        pressure = self.producer_bhp * np.ones(self.reservoir.mesh.n_res_blocks)
        z_h2o = self.initial_z_h2o * np.ones(self.reservoir.mesh.n_res_blocks)
        return self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh,
            input_distribution={
                self.physics.vars[0]: pressure,
                self.physics.vars[1]: z_h2o,
            },
        )

    def set_well_controls(self, rate=None) -> None:
        del rate
        from darts.engines import well_control_iface

        for well in self.reservoir.wells:
            self.physics.set_well_controls(
                wctrl=well.control,
                control_type=well_control_iface.BHP,
                is_inj=False,
                target=self.p_prod,
            )

    def set_rhs_flux(self, t: float | None = None) -> np.ndarray:
        del t
        n_vars = self.physics.n_vars
        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * n_vars)

        if not self.rate_rhs:
            return rhs_flux

        m_co2 = 44.01
        m_h2o = 18.0
        x = np.asarray(self.physics.engine.X)
        sg_max = None
        if "sg_max" in getattr(self.physics, "history_labels", []):
            sg_max = self.physics.get_engine_history_array(
                "sg_max",
                n_blocks=self.reservoir.mesh.n_blocks,
            )

        for well_cell in self.well_cells:
            pressure = x[well_cell * n_vars]
            co2_idx = well_cell * n_vars + 1
            h2o_idx = well_cell * n_vars
            state_values = [pressure] + self.inj_stream[0]
            if sg_max is not None:
                state_values.append(sg_max[well_cell])
            state = value_vector(state_values)

            region = int(self.op_num[well_cell])
            property_container = self.physics.property_containers[region]
            property_container.evaluate(state)

            n_co2 = self.inj_rate[0] / m_co2
            n_h2o = self.inj_rate[1] / m_h2o
            rhs_flux[co2_idx] -= n_co2
            rhs_flux[h2o_idx] -= n_h2o

        return rhs_flux

    def update_history_fields_after_timestep(self) -> None:
        if not self.hys or "sg_max" not in getattr(self.physics, "history_labels", []):
            return

        n_res_blocks = self.reservoir.mesh.n_res_blocks
        sg = np.asarray(self.get_engine_output_properties(["sat_V"])["sat_V"], dtype=float)
        sg_max = np.array(
            self.physics.get_engine_history_array(
                "sg_max",
                n_blocks=self.reservoir.mesh.n_blocks,
            ),
            copy=True,
        )

        for region, corey in self.corey.items():
            block_idx = np.where(self.op_num[:n_res_blocks] == region)[0]
            c_land = 1.0 / corey.sgrmax - 1.0 / (1.0 - corey.swc)
            for block in block_idx:
                if sg[block] >= sg_max[block]:
                    sg_max[block] = sg[block]
                    continue

                sgr = sg_max[block] / (1.0 + c_land * sg_max[block])
                if sg[block] < sgr:
                    sg_max[block] = np.clip(
                        sg[block] / (1.0 - c_land * sg[block]),
                        0.0,
                        1.0,
                    )

        self.physics.set_engine_history_array(
            "sg_max",
            sg_max,
            n_blocks=self.reservoir.mesh.n_blocks,
        )

    def run_hysteresis(
        self,
        days: float,
        save_well_data: bool = True,
        save_solution_data: bool = True,
        verbose: bool = True,
    ):
        return super().run(
            days=days,
            save_well_data=save_well_data,
            save_reservoir_data=save_solution_data,
            verbose=verbose,
        )

    def pore_volume(self) -> float:
        volume = np.asarray(self.reservoir.mesh.volume, copy=False)
        poro = np.asarray(self.reservoir.mesh.poro, copy=False)
        return float(np.sum(volume * poro))

    def vtk_output_properties(self) -> list[str]:
        output_properties = list(self.physics.property_containers[0].output_props.keys())
        output_properties.extend(self.physics.vars)
        if self.hys:
            output_properties.append("sg_max")
        return output_properties
