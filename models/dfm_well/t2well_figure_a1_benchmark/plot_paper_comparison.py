from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASE_DIR = Path(__file__).resolve().parent
COMPARISON_DIR = CASE_DIR / "paper_comparison"
SECONDS_PER_DAY = 24 * 60 * 60
REFERENCE_PROPERTY_COLUMNS = {
    "Pressure": "pressure_pa",
    "Gas Saturation": "sG",
    "Gas Phase Velocity": "vG_m_s",
    "Drift Velocity": "drift_velocity_m_s",
}


def _final_profile(model):
    props = pd.read_pickle(model.output_folder + "/dfm_well_props_I1.pkl")
    num_segments = model.wells["I1"].geometry.num_segments
    final = props.iloc[-num_segments:].copy()

    geometry = model.wells["I1"].geometry
    final["depth_m"] = geometry.TVD_segments
    area = geometry.pipe_internal_A
    final["vG_m_s"] = model.mass_rate_co2_kg_s / (final["rhoG"] * final["sG"] * area)
    final["vL_m_s"] = model.mass_rate_h2o_kg_s / (final["rhoL"] * final["sL"] * area)

    # The first segment is the artificial top pressure block.
    return final.iloc[1:].reset_index(drop=True)


def _make_summary(model, profile):
    p_top = float(profile["pressure"].iloc[0])
    p_bottom = float(profile["pressure"].iloc[-1])
    sg_top = float(profile["sG"].iloc[0])
    sg_bottom = float(profile["sG"].iloc[-1])
    vg_top = float(profile["vG_m_s"].iloc[0])
    vg_bottom = float(profile["vG_m_s"].iloc[-1])

    rows = [
        ("paper_well_length_m", model.well_length_m),
        ("paper_grid_resolution_m", model.physical_segment_length_m),
        ("paper_temperature_C", model.temperature_k - 273.15),
        ("paper_top_pressure_Pa", model.top_pressure_bar * 1.0e5),
        ("paper_wall_roughness_m", model.wall_roughness_m),
        ("paper_co2_mass_rate_kg_s", model.mass_rate_co2_kg_s),
        ("paper_h2o_mass_rate_kg_s", model.mass_rate_h2o_kg_s),
        ("paper_total_mass_flux_kg_m2_s", 50.0),
        ("paper_steady_time_s", 0.456869e9),
        ("darts_final_time_s", model.physics.engine.t * SECONDS_PER_DAY),
        ("darts_top_physical_pressure_Pa", p_top * 1.0e5),
        ("darts_bottom_pressure_Pa", p_bottom * 1.0e5),
        ("darts_top_physical_gas_saturation", sg_top),
        ("darts_bottom_gas_saturation", sg_bottom),
        ("darts_top_gas_velocity_m_s", vg_top),
        ("darts_near_bottom_gas_velocity_m_s", vg_bottom),
    ]
    summary = pd.DataFrame(rows, columns=["metric", "value"])
    summary.to_csv(COMPARISON_DIR / "summary.csv", index=False)


def _load_reference_profiles():
    long_reference_path = CASE_DIR / "digitized_profiles_long.csv"
    if long_reference_path.exists():
        reference_long = pd.read_csv(long_reference_path)
        reference_long["quantity"] = reference_long["property"].map(
            REFERENCE_PROPERTY_COLUMNS
        )
        return reference_long.dropna(subset=["quantity"]), True

    reference_path = CASE_DIR / "paper_reference" / "figure_a1_digitized.csv"
    if not reference_path.exists():
        return None, False

    reference = pd.read_csv(reference_path)
    reference_long = reference.melt(
        id_vars="depth_m",
        value_vars=["pressure_pa", "sG", "vG_m_s", "drift_velocity_m_s"],
        var_name="quantity",
        value_name="value",
    )
    reference_long["solution"] = "T2Well Figure A1"
    return reference_long, False


def plot_comparison(model):
    COMPARISON_DIR.mkdir(exist_ok=True)
    profile = _final_profile(model)
    profile.to_csv(COMPARISON_DIR / "darts_final_profile.csv", index=False)
    _make_summary(model, profile)

    reference_long, has_long_reference = _load_reference_profiles()

    profile_plot = profile.copy()
    profile_plot["pressure_pa"] = profile_plot["pressure"] * 1.0e5
    profile_plot["drift_velocity_m_s"] = profile_plot["vG_m_s"] - (
        profile_plot["sG"] * profile_plot["vG_m_s"]
        + profile_plot["sL"] * profile_plot["vL_m_s"]
    )

    if reference_long is not None:
        metric_rows = []
        for column in ["pressure_pa", "sG", "vG_m_s", "drift_velocity_m_s"]:
            for solution in reference_long["solution"].unique():
                reference_subset = reference_long[
                    (reference_long["quantity"] == column)
                    & (reference_long["solution"] == solution)
                ]
                darts_interp = np.interp(
                    reference_subset["depth_m"],
                    profile_plot["depth_m"],
                    profile_plot[column],
                )
                diff = darts_interp - reference_subset["value"]
                metric_rows.append(
                    {
                        "reference_solution": solution,
                        "quantity": column,
                        "rmse": float(np.sqrt(np.mean(diff**2))),
                        "max_abs_error": float(np.max(np.abs(diff))),
                    }
                )
        pd.DataFrame(metric_rows).to_csv(
            COMPARISON_DIR / "darts_vs_paper_metrics.csv", index=False
        )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), sharey=True)
    axes = axes.ravel()

    panels = [
        ("pressure_pa", "Pressure (Pa)", "(a)"),
        ("sG", "Gas saturation", "(b)"),
        ("vG_m_s", "Gas phase velocity (m/s)", "(c)"),
        ("drift_velocity_m_s", "Drift velocity (m/s)", "(d)"),
    ]
    reference_styles = {
        "Analytical": {"color": "#d62728", "lw": 2.0, "ls": "-"},
        "T2Well": {"color": "#2ca02c", "lw": 2.0, "ls": "--"},
        "T2Well Figure A1": {"color": "#d62728", "lw": 2.0, "ls": "-"},
    }
    for ax, (column, xlabel, label) in zip(axes, panels):
        if reference_long is not None:
            for solution in reference_long["solution"].unique():
                reference_subset = reference_long[
                    (reference_long["quantity"] == column)
                    & (reference_long["solution"] == solution)
                ]
                style = reference_styles.get(
                    solution, {"color": "0.25", "lw": 1.8, "ls": "-"}
                )
                label_solution = solution if has_long_reference else "T2Well Figure A1"
                ax.plot(
                    reference_subset["value"],
                    reference_subset["depth_m"],
                    label=label_solution,
                    **style,
                )
        ax.plot(
            profile_plot[column],
            profile_plot["depth_m"],
            color="#1f77b4",
            lw=1.8,
            marker="o",
            markersize=2.5,
            markevery=5,
            label="DARTS-well",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Depth (m)")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_ylim(model.well_length_m, 0.0)
        ax.grid(True, color="0.88", linewidth=0.7)
        ax.legend(frameon=False)

    fig.suptitle("T2Well Figure A1 benchmark: DARTS-well final profiles", y=0.99)
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "darts_figure_a1_profiles.png", bbox_inches="tight")
    fig.savefig(COMPARISON_DIR / "darts_figure_a1_profiles.pdf", bbox_inches="tight")
    plt.close(fig)
