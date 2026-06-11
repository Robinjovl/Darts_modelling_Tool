from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CASE_DIR = Path(__file__).resolve().parent
COMPARISON_DIR = CASE_DIR / "paper_comparison"
REFERENCE_PRESSURE_OFFSET_PA = 1.0e5
REFERENCE_PROPERTY_COLUMNS = {
    "Pressure": "pressure_bar",
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
    final = final.iloc[1:].reset_index(drop=True)
    return final


def _load_reference_profiles():
    long_reference_path = CASE_DIR / "digitized_profiles_long.csv"
    if long_reference_path.exists():
        reference_long = pd.read_csv(long_reference_path)
        reference_long["quantity"] = reference_long["property"].map(
            REFERENCE_PROPERTY_COLUMNS
        )
        pressure_rows = reference_long["quantity"] == "pressure_bar"
        reference_long.loc[pressure_rows, "value"] = (
            reference_long.loc[pressure_rows, "value"] + REFERENCE_PRESSURE_OFFSET_PA
        ) / 1.0e5
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
    pressure_rows = reference_long["quantity"] == "pressure_pa"
    reference_long.loc[pressure_rows, "quantity"] = "pressure_bar"
    reference_long.loc[pressure_rows, "value"] = (
        reference_long.loc[pressure_rows, "value"] + REFERENCE_PRESSURE_OFFSET_PA
    ) / 1.0e5
    return reference_long, False


def plot_comparison(model):
    COMPARISON_DIR.mkdir(exist_ok=True)
    profile = _final_profile(model)
    reference_long, has_long_reference = _load_reference_profiles()

    profile_plot = profile.copy()
    profile_plot["pressure_bar"] = profile_plot["pressure"]
    profile_plot["drift_velocity_m_s"] = profile_plot["vG_m_s"] - (
        profile_plot["sG"] * profile_plot["vG_m_s"]
        + profile_plot["sL"] * profile_plot["vL_m_s"]
    )

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8), sharey=True)
    axes = axes.ravel()

    panels = [
        ("pressure_bar", "Pressure [bar]", "(a)"),
        ("sG", "Gas saturation [-]", "(b)"),
        ("vG_m_s", "Gas phase velocity (m/s)", "(c)"),
        ("drift_velocity_m_s", "Drift velocity (m/s)", "(d)"),
    ]
    reference_styles = {
        "Analytical": {"color": "#000000", "lw": 2.4, "ls": "-"},
        "T2Well": {"color": "#0072B2", "lw": 2.4, "ls": "--"},
        "T2Well Figure A1": {
            "color": "#000000",
            "lw": 2.4,
            "ls": "-",
        },
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
                    solid_capstyle="round",
                    dash_capstyle="round",
                    **style,
                )
        ax.plot(
            profile_plot[column],
            profile_plot["depth_m"],
            color="#D55E00",
            lw=2.4,
            label="DARTS-well",
            solid_capstyle="round",
            zorder=5,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Depth (m)")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_ylim(model.well_length_m, 0.0)
        ax.grid(True, color="0.86", linewidth=0.6)
        ax.tick_params(direction="in", top=True, right=True, length=4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="0.75",
        framealpha=0.95,
        handlelength=3.2,
        columnspacing=1.8,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), h_pad=1.6, w_pad=1.6)
    fig.savefig(COMPARISON_DIR / "darts_figure_a1_profiles.png", bbox_inches="tight")
    fig.savefig(COMPARISON_DIR / "darts_figure_a1_profiles.pdf", bbox_inches="tight")
    plt.close(fig)
