"""
The two functions
    plot_well_1d_reservoir_line_graphs_for_reported_times
    and
    plot_well_1d_reservoir_line_graphs_for_scenarios
in this file can be used for plotting the profile of a property both in the wellbore and reservoir.
plot_well_1d_reservoir_line_graphs_for_reported_times can be used to plot a property for reported times of a particular
scenario, but it cannot be used if we want to plot well-reservoir profiles of different scenarios and compare the
results in the same axes. To do this, you can use plot_well_1d_reservoir_line_graphs_for_scenarios.

The file all_solutions.csv which contains the reservoir data at reported time steps needs to be generated from the
corresponding vtk files by using the Python interface of ParaView and stored in the output folder.

Note:
    This functino can be used only for 1D reservoirs.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from darts.models.darts_model import DartsModel
from darts.tools.hdf5_tools import load_hdf5_to_dict


def plot_well_1d_reservoir_line_graphs_for_reported_times(
    coupled_model: DartsModel,
    report_time_labels: list,
    reported_times: list,
    *,
    prop_name: str,
    legend_loc: str = 'best',
):
    """
    Plot well-reservoir property profiles at certain reported times for a scenario.
    Note: It can be used only for 1D reservoirs.

    :param coupled_model: An instance of DartsModel
    :type coupled_model: DartsModel
    :param report_time_labels: List of labels of reported times
    :type report_time_labels: list
    :param reported_times: List of reported times [day]
    :type reported_times: list
    :param prop_name: Name of the property to plot
    :type prop_name: str
    :param legend_loc: Legend location
    :type legend_loc: str
    """
    output_folder_path = coupled_model.output_folder

    dx = coupled_model.reservoir.global_data['dx']
    assert dx.ndim == 3 and dx.shape[1:] == (1, 1), f"Expected (*,1,1), got {dx.shape}"

    assert len(report_time_labels) == len(reported_times), (
        "Number of report step labels must be equal to number of report step times!"
    )

    assert prop_name in [
        "pressure",
        "temperature",
        "sL",
        "rhoG",
        "rhoL",
        "miuG",
        "miuL",
    ]
    if prop_name == "pressure":
        prop_name_in_well_output = "Pressure"
        prop_name_in_reservoir_output = "pressure"
        xlabel = "Pressure [bar]"
    elif prop_name == "temperature":
        prop_name_in_well_output = "Temperature"
        prop_name_in_reservoir_output = "temperature"
        xlabel = "Temperature [\u00b0C]"
    elif prop_name == "sL":
        prop_name_in_well_output = "sL"
        prop_name_in_reservoir_output = "sat_LCO2"
        xlabel = "Liquid volume fraction [-]"
    elif prop_name == "rhoG":
        prop_name_in_well_output = "rhoG"
        prop_name_in_reservoir_output = "rho_gas"
        xlabel = r"Gas density [kg/m$^3$]"
    elif prop_name == "rhoL":
        prop_name_in_well_output = "rhoL"
        prop_name_in_reservoir_output = "rho_LCO2"
        xlabel = r"Liquid density [kg/m$^3$]"
    elif prop_name == "miuG":
        prop_name_in_well_output = "miuG"
        prop_name_in_reservoir_output = "miu_gas"
        xlabel = "Gas viscosity [cP]"
    elif prop_name == "miuL":
        prop_name_in_well_output = "miuL"
        prop_name_in_reservoir_output = "miu_LCO2"
        xlabel = "Liquid viscosity [cP]"

    # This line gets the geometry object of the first well (by insertion order) from the wells_geometry dictionary
    # and assigns it to well_geom.
    well_geom = next(iter(coupled_model.wells.values())).geometry

    # Get well depth array and number of segments
    segment_depths = well_geom.z
    num_segments = well_geom.num_segments

    # Get reservoir radial distance array: strictly positive, non-zero start
    reservoir_dx = dx.flatten()
    reservoir_radial_distance = np.cumsum(reservoir_dx)

    # report_step_labels = [
    #     "Initial conditions",
    #     "30 seconds",
    #     "1 minute",
    #     "2 minutes",
    #     "3 minutes",
    #     "5 minutes",
    #     "10 minutes",
    #     "20 minutes",
    #     "30 seconds",
    #     "50 seconds",
    #     "1 hour",
    # ]
    #
    # report_step_times = [0,
    #                      0.5 / 24 / 60,  # 30 seconds
    #                      0.5 / 24 / 60,  # 1 minute
    #                      1 / 24 / 30 - 1 / 24 / 60,  # 2 minute
    #                      1 / 24 / 20 - 1 / 24 / 30,  # 3 minute
    #                      1 / 24 / 12 - 1 / 24 / 20,  # 5 minute
    #                      1 / 24 / 6 - 1 / 24 / 12,  # 10 minute
    #                      1 / 24 / 3 - 1 / 24 / 6,  # 20 minute
    #                      1 / 24 / 2 - 1 / 24 / 3,  # 30 minute
    #                      1 / 24 / 6 * 5 - 1 / 24 / 2,  # 50 minute
    #                      1 / 24 - 1 / 24 / 6 * 5,  # 1 hour
    #                      ]

    # Well HDF5 file is used here to get the time step sizes
    h5_well_file_path = coupled_model.well_filepath
    h5_well_dict = load_hdf5_to_dict(h5_well_file_path)
    simulated_time = h5_well_dict["dynamic"]["time"]

    report_indices = [
        np.where(np.isclose(simulated_time, a))[0][0] for a in reported_times
    ]

    # Load primary vars and phase props for well
    primary_vars_and_phase_props_file_path = os.path.join(
        coupled_model.output.output_folder, "well_primary_vars_and_phase_props.pkl"
    )
    well_data_frame = pd.read_pickle(primary_vars_and_phase_props_file_path)

    # Fast load; infer low-memory dtypes
    all_solutions_csv_path = os.path.join(output_folder_path, "all_solutions.csv")
    reservoir_data_frame = pd.read_csv(all_solutions_csv_path, low_memory=False)

    # Ensure stable ordering
    tvals = np.array(sorted(reservoir_data_frame["Timestep"].unique()))
    num_res_cells = coupled_model.reservoir.nx

    # Make an index to pivot quickly
    df_idx = reservoir_data_frame.set_index(["Timestep", "CellID"]).sort_index()

    def to_matrix(prop_name):
        """
        Return a (num_report_times, num_res_cells) matrix for the given property name
        """
        full_idx = pd.MultiIndex.from_product(
            [tvals, sorted(reservoir_data_frame["CellID"].unique())],
            names=["Timestep", "CellID"],
        )
        s = df_idx[prop_name].reindex(full_idx)
        matrix_full = s.values.reshape((len(tvals), -1))
        reservoir_prop_matrix = matrix_full[
            :, :num_res_cells
        ]  # keep only the first num_res_cells

        return reservoir_prop_matrix

    reservoir_prop_matrix = to_matrix(prop_name_in_reservoir_output)
    if prop_name == "temperature":
        reservoir_prop_matrix -= 273.15

    # Generate a colormap for the report steps (big jumps for the first time steps, then smaller)
    n = len(report_time_labels)
    cmap = mpl.colormaps['jet']
    alpha = 10.0  # larger => more contrast early, flatter later
    t = np.arange(n) / (n - 1 if n > 1 else 1)
    t_nonlin = np.log1p(alpha * t) / np.log1p(alpha)
    colors = [cmap(v) for v in t_nonlin]

    # Define markers and line styles
    markers = ['o', 's', 'd', '^', 'v', 'x', '*']
    linestyles = ['-', '--', '-.', ':']

    # Start plotting
    fig, ax = plt.subplots(figsize=(6, 5))
    y_r, offset, rmin, rmax = stacked_y_axis_linear_log(
        ax,
        segment_depths,
        reservoir_radial_distance,
        gap_ratio=0.06,
        n_ticks_well=5,
        add_minor=True,
    )

    for idx, report_index in enumerate(report_indices):
        well_prop_profile = well_data_frame[prop_name_in_well_output][
            report_index * num_segments : (report_index + 1) * num_segments
        ]
        if prop_name == "temperature":
            well_prop_profile -= 273.15

        color = colors[idx]  # Assign color from the colormap
        marker = markers[idx % len(markers)]  # Cycle through markers
        linestyle = linestyles[idx % len(linestyles)]  # Cycle through line styles

        ax.plot(
            well_prop_profile,
            segment_depths,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0,  # match linewidth
            markersize=4.0,  # match markersize
            color=color,
            label=report_time_labels[idx],
        )

        reservoir_prop_profile = reservoir_prop_matrix[idx, :]
        ax.plot(
            reservoir_prop_profile,
            y_r(reservoir_radial_distance),
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0,  # match linewidth
            markersize=4.0,  # match markersize
            color=color,
        )

    ax.set_ylim(top=0)

    ax.set_xlabel(xlabel)

    # ax.legend(loc="best", frameon=False)

    ax.tick_params(
        axis='y',
        which='both',
        labelleft=True,
        labelright=False,
        left=True,
        right=False,
        pad=3,  # <-- increase this to push labels farther from ticks
        direction='in',
    )  # optional: ticks point outward
    ax.spines['right'].set_visible(True)

    # Add subtle grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)

    plt.legend(
        fontsize=6,
        loc=legend_loc,
        title="Report steps",
        title_fontsize=10,
    ).get_frame().set_edgecolor('black')  # Optional: Add a border
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_folder_path, f"{prop_name}_well_reservoir_profile.pdf")
    )
    fig.savefig(
        os.path.join(output_folder_path, f"{prop_name}_well_reservoir_profile.png")
    )
    plt.show()


def plot_well_1d_reservoir_line_graphs_for_scenarios(
    primary_vars_and_phase_props_file_address: str,
    h5_well_data: dict,
    coupled_model: DartsModel,
    report_time_labels: list,
    reported_times: list,
    *,
    prop_name: str,
    ax,
    legend_label: str,
    legend_title: str,
    color: str,
    marker: str,
    linestyle: str,
    legend_loc: str = 'best',
):
    """
    This function is used to plot well-reservoir property profiles at certain reported times for different scenarios.
    It can be used only for 1D reservoirs.

    :param primary_vars_and_phase_props_file_address: Address of the pickle file in which primary variables and phase
    properties of well segments are stored
    :type primary_vars_and_phase_props_file_address: str
    :param h5_well_data: HDF5 file containing well solution
    :type h5_well_data: dict
    :param coupled_model: An instance of DartsModel
    :type coupled_model: DartsModel
    :param report_time_labels: List of labels of reported times
    :type report_time_labels: list
    :param reported_times: List of reported times [day]
    :type reported_times: list
    :param prop_name: Name of the property to plot
    :type prop_name: str
    :param ax: Axes to plot on. This is needed to plot all the plots of different scenarios in the same axes.
    :type ax: matplotlib.axes.Axes
    :param legend_label: Label of the plot for the legend
    :type legend_label: str
    :param color: Color of the plot
    :type color: str
    :param marker: Marker of the plot
    :type marker: str
    :param linestyle: Linestyle of the plot
    :type linestyle: str
    :param legend_loc: Legend location
    :type legend_loc: str
    """
    output_folder = coupled_model.output_folder

    dx = coupled_model.reservoir.global_data['dx']
    assert dx.ndim == 3 and dx.shape[1:] == (1, 1), f"Expected (*,1,1), got {dx.shape}"

    assert len(report_time_labels) == len(reported_times), (
        "Number of report step labels must be equal to number of report step times!"
    )

    assert prop_name in ["pressure", "temperature", "sL"]
    if prop_name == "pressure":
        prop_name_in_well_output = "Pressure"
        prop_name_in_reservoir_output = "pressure"
        xlabel = "Pressure [bar]"
    elif prop_name == "temperature":
        prop_name_in_well_output = "Temperature"
        prop_name_in_reservoir_output = "temperature"
        xlabel = "Temperature [\u00b0C]"
    elif prop_name == "sL":
        prop_name_in_well_output = "sL"
        prop_name_in_reservoir_output = "sat_LCO2"
        xlabel = "Liquid volume fraction [-]"

    # This line gets the geometry object of the first well (by insertion order) from the wells_geometry dictionary
    # and assigns it to well_geom.
    well_geom = next(iter(coupled_model.wells.values())).geometry

    # Get well depth array and number of segments
    segment_depths = well_geom.z
    num_segments = well_geom.num_segments

    # Get reservoir radial distance array: strictly positive, non-zero start
    reservoir_dx = dx.flatten()
    reservoir_radial_distance = np.cumsum(reservoir_dx)

    # report_step_labels = [
    #     "Initial conditions",
    #     "30 seconds",
    #     "1 minute",
    #     "2 minutes",
    #     "3 minutes",
    #     "5 minutes",
    #     "10 minutes",
    #     "20 minutes",
    #     "30 seconds",
    #     "50 seconds",
    #     "1 hour",
    # ]
    #
    # report_step_times = [0,
    #                      0.5 / 24 / 60,  # 30 seconds
    #                      0.5 / 24 / 60,  # 1 minute
    #                      1 / 24 / 30 - 1 / 24 / 60,  # 2 minute
    #                      1 / 24 / 20 - 1 / 24 / 30,  # 3 minute
    #                      1 / 24 / 12 - 1 / 24 / 20,  # 5 minute
    #                      1 / 24 / 6 - 1 / 24 / 12,  # 10 minute
    #                      1 / 24 / 3 - 1 / 24 / 6,  # 20 minute
    #                      1 / 24 / 2 - 1 / 24 / 3,  # 30 minute
    #                      1 / 24 / 6 * 5 - 1 / 24 / 2,  # 50 minute
    #                      1 / 24 - 1 / 24 / 6 * 5,  # 1 hour
    #                      ]

    simulated_time = h5_well_data["dynamic"]["time"]

    report_indices = [
        np.where(np.isclose(simulated_time, a))[0][0] for a in reported_times
    ]

    # Load primary vars and phase props for well
    well_data_frame = pd.read_pickle(primary_vars_and_phase_props_file_address)

    # Fast load; infer low-memory dtypes
    all_solutions_csv_path = os.path.join(output_folder, "all_solutions.csv")
    reservoir_data_frame = pd.read_csv(all_solutions_csv_path, low_memory=False)

    # Ensure stable ordering
    tvals = np.array(sorted(reservoir_data_frame["Timestep"].unique()))
    num_res_cells = coupled_model.reservoir.nx

    # Make an index to pivot quickly
    df_idx = reservoir_data_frame.set_index(["Timestep", "CellID"]).sort_index()

    def to_matrix(prop_name):
        """
        Return a (num_report_times, num_res_cells) matrix for the given property name
        """
        full_idx = pd.MultiIndex.from_product(
            [tvals, sorted(reservoir_data_frame["CellID"].unique())],
            names=["Timestep", "CellID"],
        )
        s = df_idx[prop_name].reindex(full_idx)
        matrix_full = s.values.reshape((len(tvals), -1))
        reservoir_prop_matrix = matrix_full[
            :, :num_res_cells
        ]  # keep only the first num_res_cells

        return reservoir_prop_matrix

    reservoir_prop_matrix = to_matrix(prop_name_in_reservoir_output)
    if prop_name == "temperature":
        reservoir_prop_matrix -= 273.15

    # Start plotting
    # fig, ax = plt.subplots(figsize=(6, 5))
    y_r, offset, rmin, rmax = stacked_y_axis_linear_log(
        ax,
        segment_depths,
        reservoir_radial_distance,
        gap_ratio=0.06,
        n_ticks_well=5,
        add_minor=True,
    )

    for idx, report_index in enumerate(report_indices):
        well_prop_profile = well_data_frame[prop_name_in_well_output][
            report_index * num_segments : (report_index + 1) * num_segments
        ]
        if prop_name == "temperature":
            well_prop_profile -= 273.15

        ax.plot(
            well_prop_profile,
            segment_depths,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0,  # match linewidth
            markersize=4.0,  # match markersize
            color=color,
            label=legend_label,
        )

        reservoir_prop_profile = reservoir_prop_matrix[idx, :]
        ax.plot(
            reservoir_prop_profile,
            y_r(reservoir_radial_distance),
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0,  # match linewidth
            markersize=4.0,  # match markersize
            color=color,
        )

    ax.set_ylim(top=0)

    ax.set_xlabel(xlabel)

    # ax.legend(loc="best", frameon=False)

    ax.tick_params(
        axis='y',
        which='both',
        labelleft=True,
        labelright=False,
        left=True,
        right=False,
        pad=3,  # <-- increase this to push labels farther from ticks
        direction='in',
    )  # optional: ticks point outward
    ax.spines['right'].set_visible(True)

    # Add subtle grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)

    plt.legend(
        fontsize=8,
        loc=legend_loc,
        title=legend_title,
        title_fontsize=10,
    ).get_frame().set_edgecolor('black')  # Optional: Add a border


def stacked_y_axis_linear_log(
    ax,
    segment_depths,
    reservoir_radial_distance,
    *,
    gap_ratio=0.04,
    res_height=None,
    n_ticks_well=5,
    add_minor=True,
):
    """
    Build a single y-axis that is linear for the well part and logarithmic for the
    reservoir part. Returns a mapper y_r(r) for the reservoir part.

    :param ax: The desired matplotlib axes
    :type ax: matplotlib.axes.Axes
    :param segment_depths: The array of the depths of the segment centroids
    :type segment_depths: array-like
    :param reservoir_radial_distance: The array of the radial distance of the reservoir cells (strictly positive)
    :type reservoir_radial_distance: array-like
    :param gap_ratio: Visual gap between the well and reservoir parts as a fraction of z-span
    :type gap_ratio: float
    :param res_height: Pixel-height of the reservoir part in data units;
                       default equals z-span for visual balance
    :type res_height: float or None
    :param n_ticks_well: Number of ticks on the well (linear) part
    :type n_ticks_well: int
    :param add_minor: Add log minor ticks in reservoir part
    :type add_minor: bool
    """
    z = np.asarray(segment_depths, float)
    r = np.asarray(reservoir_radial_distance, float)

    z0, z1 = float(z[0]), float(z[-1])
    z_span = abs(z1 - z0)
    gap = gap_ratio * z_span
    H_res = z_span if res_height is None else float(res_height)
    offset = z1 + gap  # start of reservoir segment in the stacked axis

    # --- reservoir log domain (must be > 0 and have spread) ---
    rmin, rmax = float(np.min(r)), float(np.max(r))
    if not (rmin > 0 and rmax > rmin):
        raise ValueError(
            "Reservoir radii must satisfy 0 < rmin < rmax for log scaling."
        )

    log_base = 10
    logb = lambda x: np.log(x) / np.log(log_base)
    Lmin, Lmax = logb(rmin), logb(rmax)

    def y_r(rv):
        """Map reservoir radii to stacked y coordinates."""
        rv = np.clip(np.asarray(rv, float), rmin, rmax)
        return offset + H_res * (logb(rv) - Lmin) / (Lmax - Lmin)

    # --- ticks: well (linear) ---
    ticks_w = np.linspace(z0, z1, n_ticks_well)
    labels_w = [f"{t:g}" for t in ticks_w]

    # --- ticks: reservoir (log) ---
    k0, k1 = int(np.floor(Lmin)), int(np.ceil(Lmax))
    decades = log_base ** np.arange(k0, k1 + 1, dtype=float)
    decades = decades[(decades >= rmin) & (decades <= rmax)]

    # always include rmin (even if not a decade)
    major_vals = np.unique(np.r_[rmin, decades])
    major_y = y_r(major_vals)

    def is_decade(val):
        k = round(logb(val))
        return np.isclose(val, (log_base**k), rtol=0, atol=1e-12)

    def format_log_label(v):
        if is_decade(v) and not np.isclose(v, rmin):  # show decades as 10^{k}
            k = int(round(logb(v)))
            base_text = (
                r"e"
                if np.isclose(log_base, np.e)
                else str(int(log_base) if float(log_base).is_integer() else log_base)
            )
            return rf"$\mathrm{{{base_text}}}^{{{k}}}$"
        return f"{v:g}"  # keep rmin as the actual number

    major_labels = [format_log_label(v) for v in major_vals]

    if add_minor:
        minors = []
        for k in range(k0, k1):
            base = log_base**k
            minors.extend(base * np.arange(2, log_base, dtype=float))  # 2..base-1
        minors = np.array(minors, float)
        minors = minors[(minors > rmin) & (minors < rmax)]
        all_ticks = np.r_[ticks_w, major_y, y_r(minors)]
        all_labels = labels_w + major_labels + [''] * len(minors)
        ax.set_yticks(all_ticks)
        ax.set_yticklabels(all_labels)
    else:
        ax.set_yticks(np.r_[ticks_w, major_y])
        ax.set_yticklabels(labels_w + major_labels)

    # limits & look
    ax.set_ylim(z0, offset + H_res)
    ax.invert_yaxis()

    # divider & side labels
    ax.axhline(offset - gap * 0.5, ls='--', lw=1)

    # --- helper to convert a y data value to axes coordinates (0..1) ---
    def ydata_to_axes(yval):
        return ax.transAxes.inverted().transform(ax.transData.transform((0, yval)))[1]

    # y positions for labels: midpoint of each segment (well is linear; reservoir use geometric mean in log)
    y_well_mid_data = 0.5 * (z0 + z1)
    y_res_mid_data = y_r(np.sqrt(rmin * rmax))

    LEFT_OFFSET = -0.08  # move labels further left; make more negative to push farther
    # fig.subplots_adjust(left=0.22)

    ax.text(
        LEFT_OFFSET,
        ydata_to_axes(y_well_mid_data),
        r"$z_{\mathrm{well}}\,[\mathrm{m}]$ (linear)",
        transform=ax.transAxes,
        rotation=90,
        va='center',
        ha='right',
        clip_on=False,
    )

    ax.text(
        LEFT_OFFSET,
        ydata_to_axes(y_res_mid_data),
        r"$r_\mathrm{reservoir}\,[\mathrm{m}]$ (logarithmic)",
        transform=ax.transAxes,
        rotation=90,
        va='center',
        ha='right',
        clip_on=False,
    )

    return y_r, offset, rmin, rmax
