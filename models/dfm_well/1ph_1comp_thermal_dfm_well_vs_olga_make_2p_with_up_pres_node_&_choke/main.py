"""
Injection of pure liquid CO2 using a choke into a well containing gaseous CO2 using a standalone well model to
compare its results with those in OLGA for a thermal two-phase scenario.
"""

import os

import numpy as np
import pandas as pd

from darts.engines import redirect_darts_output
from darts.pipes.save_results import save_dfm_well_props
from darts.pipes.viz.plot_heat_map_pcolormesh import plot_heat_map_pcolormesh

from model import Model


RUNTIME_DAY = 10 / 24 / 60
RATE_TOL_KG_S = 1e-4
MAX_CHOKE_SIZE_ITERS = 4
OLGA_EQUILIBRIUM_MODEL = "FROZEN"
OLGA_RECOVERY = "ON"
OLGA_CD = 0.84
OLGA_CF = 26.8465
OLGA_CR = 1.0
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(
    MODEL_DIR,
    "profiles_60bar_10C_PI_1e5_PH_1kgpersec.xlsx",
)


def build_model(choke_diameter_m: float = None) -> Model:
    coupled_model = Model(
        inlet_boundary_kind="pressure_node_choke",
        inlet_choke_equilibrium_model=OLGA_EQUILIBRIUM_MODEL,
        inlet_choke_diameter=choke_diameter_m,
        inlet_choke_recovery=OLGA_RECOVERY,
        inlet_choke_discharge_coefficient=OLGA_CD,
        inlet_choke_gas_liquid_sizing_ratio=OLGA_CF,
        inlet_choke_recovery_tuning=OLGA_CR,
    )
    coupled_model.reservoir.grav_acceleration_for_spe = 9.80665
    return coupled_model


def run_model(
    choke_diameter_m: float = None,
    save_visual_outputs: bool = False,
) -> Model:
    coupled_model = build_model(choke_diameter_m=choke_diameter_m)
    coupled_model.init()
    coupled_model.set_output()

    if save_visual_outputs:
        output_props = coupled_model.physics.vars + coupled_model.output.properties
        coupled_model.output.well_output_to_vtp(
            ith_step=0,
            output_properties=output_props,
        )

    coupled_model.run(RUNTIME_DAY)

    if save_visual_outputs:
        output_props = coupled_model.physics.vars + coupled_model.output.properties
        coupled_model.output.well_output_to_vtp(
            ith_step=1,
            output_properties=output_props,
        )

    return coupled_model


def size_choke_for_target_steady_rate(
    rate_tol_kg_s: float = RATE_TOL_KG_S,
    max_iters: int = MAX_CHOKE_SIZE_ITERS,
) -> tuple[float, list[dict]]:
    choke_diameter_m = None
    history = []

    for iter_idx in range(1, max_iters + 1):
        coupled_model = run_model(choke_diameter_m=choke_diameter_m)
        node = coupled_model.wells["I1"].source_sinks["UpstreamMassNode1"]

        entry = {
            "iter": iter_idx,
            "diameter_m": float(node.diameter),
            "target_rate_kg_s": float(node.target_mass_rate_kg_s),
            "final_rate_kg_s": float(node.last_mass_rate_kg_s),
            "downstream_pressure_bar": float(node.last_downstream_pressure),
        }
        history.append(entry)

        print(
            "Choke sizing iteration "
            f"{iter_idx}: diameter={entry['diameter_m']:.9f} m, "
            f"target_rate={entry['target_rate_kg_s']:.6f} kg/s, "
            f"final_rate={entry['final_rate_kg_s']:.6f} kg/s, "
            f"wellhead_pressure={entry['downstream_pressure_bar']:.6f} bar"
        )

        if abs(entry["final_rate_kg_s"] - entry["target_rate_kg_s"]) <= rate_tol_kg_s:
            return entry["diameter_m"], history

        choke_diameter_m = node.suggest_diameter_for_target_rate()

    return history[-1]["diameter_m"], history


def compare_final_profiles(coupled_model: Model) -> dict:
    save_dfm_well_props("I1", coupled_model)

    darts_profile_path = os.path.join(
        coupled_model.output.output_folder,
        "dfm_well_props_I1.pkl",
    )
    darts_df = pd.read_pickle(darts_profile_path)
    olga_df = pd.read_excel(EXCEL_PATH)

    num_segments = coupled_model.reservoir.wells[0].num_segments
    darts_final = darts_df.tail(num_segments).iloc[:-1].reset_index(drop=True)

    if len(darts_final) != len(olga_df):
        raise ValueError(
            f"Expected {len(olga_df)} comparison rows from Excel, got {len(darts_final)} DARTS rows."
        )

    return {
        "p_rmse_bar": float(
            np.sqrt(
                np.mean(
                    (darts_final["Pressure"].to_numpy() - olga_df["OLGA_p"].to_numpy())
                    ** 2
                )
            )
        ),
        "T_rmse_C": float(
            np.sqrt(
                np.mean(
                    (
                        darts_final["Temperature"].to_numpy()
                        - 273.15
                        - olga_df["OLGA_T"].to_numpy()
                    )
                    ** 2
                )
            )
        ),
        "sL_rmse": float(
            np.sqrt(
                np.mean(
                    (darts_final["sL"].to_numpy() - olga_df["OLGA_sL"].to_numpy())
                    ** 2
                )
            )
        ),
    }


redirect_darts_output("run.log")

final_diameter_m, _diameter_history = size_choke_for_target_steady_rate()
coupled_model = run_model(
    choke_diameter_m=final_diameter_m,
    save_visual_outputs=True,
)
comparison = compare_final_profiles(coupled_model)

coupled_model.print_timers()
plot_heat_map_pcolormesh("I1", coupled_model, show_plot=False)

final_node = coupled_model.wells["I1"].source_sinks["UpstreamMassNode1"]
print(
    "Final steady-state choke sizing: "
    f"diameter={final_node.diameter:.9f} m, "
    f"target_rate={final_node.target_mass_rate_kg_s:.6f} kg/s, "
    f"final_rate={final_node.last_mass_rate_kg_s:.6f} kg/s"
)
print(
    "Configured choke inputs: "
    f"EQUILIBRIUMMODEL={final_node.equilibrium_model}, "
    f"CD={final_node.discharge_coefficient:.4f}, "
    f"CF={final_node.gas_liquid_sizing_ratio:.4f}, "
    f"CR={final_node.recovery_tuning:.4f}, "
    f"RECOVERY={final_node.recovery}"
)
print(
    "Comparison against OLGA (top 19 segments only): "
    f"p_rmse={comparison['p_rmse_bar']:.6f} bar, "
    f"T_rmse={comparison['T_rmse_C']:.6f} C, "
    f"sL_rmse={comparison['sL_rmse']:.6f}"
)
