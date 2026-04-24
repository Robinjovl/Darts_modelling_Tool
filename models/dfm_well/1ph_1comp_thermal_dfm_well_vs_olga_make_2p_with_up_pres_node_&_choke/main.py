"""
Source-free top-segment-state case for comparing the DARTS well against OLGA.
"""

import os

import numpy as np
import pandas as pd

from darts.engines import redirect_darts_output
from darts.pipes.save_results import save_dfm_well_props
from darts.pipes.viz.plot_heat_map_pcolormesh import plot_heat_map_pcolormesh

from model import Model


RUNTIME_DAY = 10 / 24 / 60
TOP_SEGMENT_VOLUME_MULTIPLIER = 1.0e6
BOTTOM_BOUNDARY_MODE = "olga_linear_ipr"
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(
    MODEL_DIR,
    "profiles_60bar_10C_PI_1e5_PH_1kgpersec.xlsx",
)


def load_olga_top_state() -> dict:
    olga_df = pd.read_excel(EXCEL_PATH)
    top_row = olga_df.iloc[0]
    return {
        "pressure": float(top_row["OLGA_p"]),
        "temperature": float(top_row["OLGA_T"]) + 273.15,
        "liquid_holdup": float(top_row["OLGA_sL"]),
    }


def build_model() -> Model:
    top_state = load_olga_top_state()
    coupled_model = Model(
        inlet_boundary_kind="upstream_mass_node",
        use_inlet_source_term=False,
        forced_top_state_pressure=top_state["pressure"],
        forced_top_state_temperature=top_state["temperature"],
        forced_top_state_liquid_holdup=top_state["liquid_holdup"],
        top_segment_volume_multiplier=TOP_SEGMENT_VOLUME_MULTIPLIER,
        bottom_boundary_mode=BOTTOM_BOUNDARY_MODE,
    )
    coupled_model.reservoir.grav_acceleration_for_spe = 9.80665
    return coupled_model


def run_model(save_visual_outputs: bool = False) -> Model:
    coupled_model = build_model()
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


def compare_final_profiles(coupled_model: Model) -> dict:
    save_dfm_well_props("I1", coupled_model)

    darts_profile_path = os.path.join(
        coupled_model.output.output_folder,
        "dfm_well_props_I1.pkl",
    )
    darts_df = pd.read_pickle(darts_profile_path)
    olga_df = pd.read_excel(EXCEL_PATH)

    num_segments = coupled_model.reservoir.wells[0].num_segments
    darts_final = darts_df.tail(num_segments).reset_index(drop=True)

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
                    (darts_final["sL"].to_numpy() - olga_df["OLGA_sL"].to_numpy()) ** 2
                )
            )
        ),
        "comparison_label": "all 20 segments",
        "num_compared_segments": int(len(darts_final)),
    }


redirect_darts_output("run.log")

coupled_model = run_model(save_visual_outputs=True)
comparison = compare_final_profiles(coupled_model)

coupled_model.print_timers()
plot_heat_map_pcolormesh("I1", coupled_model, show_plot=False)

forced_state = coupled_model.forced_top_segment_state
print(
    "Forced top-segment state: "
    f"p={forced_state['pressure']:.6f} bar, "
    f"T={forced_state['temperature'] - 273.15:.6f} C, "
    f"sL={forced_state['liquid_holdup']:.6f}, "
    f"h={forced_state['molar_enthalpy']:.6f} kJ/kmol, "
    f"rho_mix={forced_state['mixture_density']:.6f} kg/m3, "
    f"top_volume_multiplier={TOP_SEGMENT_VOLUME_MULTIPLIER:.1f}, "
    f"inlet_source_term=OFF, bottom_boundary_mode={BOTTOM_BOUNDARY_MODE}"
)
print(
    f"Comparison against OLGA ({comparison['comparison_label']}): "
    f"p_rmse={comparison['p_rmse_bar']:.6f} bar, "
    f"T_rmse={comparison['T_rmse_C']:.6f} C, "
    f"sL_rmse={comparison['sL_rmse']:.6f}"
)
