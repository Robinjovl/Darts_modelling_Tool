"""
The function plot_well_1d_reservoir_line_graphs in this file can be used for plotting the profile of a property
both in the wellbore and reservoir.

The file all_solutions.csv which contains the reservoir data at reported time steps needs to be generated from the
corresponding vtk files by using the Python interface of ParaView.

Note:
    This functino can be used only for 1D reservoirs.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from darts.models.darts_model import DartsModel


def plot_well_1d_reservoir_line_graphs(
    primary_vars_and_phase_props_file_address: str,
    h5_well_data: dict,
    coupled_model: DartsModel,
    report_step_labels: list,
    report_step_times: list,
    *,
    property_name: str,
    legend_loc: str = 'upper right',
):
    """
    This function is used to plot well-reservoir property profiles at certain time steps.

    :param primary_vars_and_phase_props_file_address: Address of the pickle file in which primary variables and phase
    properties of well segments are stored
    :type primary_vars_and_phase_props_file_address: str
    :param h5_well_data: HDF5 file containing well solution
    :type h5_well_data: dict
    :param coupled_model: An instance of DartsModel
    :type coupled_model: DartsModel
    :param report_step_labels: List of labels of reported steps
    :type report_step_labels: list
    :param report_step_times: List of times of reported steps
    :type report_step_times: list
    :param property_name: Name of the property to plot
    :type property_name: str
    :param legend_loc: Legend location
    :type legend_loc: str
    """
    dx = coupled_model.reservoir.global_data['dx']
    assert dx.ndim == 3 and dx.shape[1:] == (1, 1), f"Expected (*,1,1), got {dx.shape}"

    assert len(report_step_labels) == len(report_step_times), (
        "Number of report step labels must be equal to number of report step times!"
    )

    assert property_name in ["pressure", "temperature", "sL"]
    if property_name == "pressure":
        prop_name_in_well_output = "Pressure"
        prop_name_in_reservoir_output = "pressure"
        xlabel = "Pressure [bar]"
    elif property_name == "temperature":
        prop_name_in_well_output = "Temperature"
        prop_name_in_reservoir_output = "temperature"
        xlabel = "Temperature [\u00b0C]"
    elif property_name == "sL":
        prop_name_in_well_output = "sL"
        prop_name_in_reservoir_output = "sat_LCO2"
        xlabel = "Liquid volume fraction [-]"

    # This line gets the geometry object of the first well (by insertion order) from the wells_geometry dictionary
    # and assigns it to well_geom.
    well_geom = next(iter(coupled_model.wells.values())).geometry

    # Get well depth array and number of segments
    segments_depths = well_geom.z
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

    simulation_time = h5_well_data["dynamic"]["time"]

    report_step_times = np.cumsum(report_step_times)
    report_indices = [
        np.where(np.isclose(simulation_time, a))[0][0] for a in report_step_times
    ]

    # Load primary vars and phase props for well
    well_data_frame = pd.read_pickle(primary_vars_and_phase_props_file_address)

    # Fast load; infer low-memory dtypes
    reservoir_data_frame = pd.read_csv("all_solutions.csv", low_memory=False)

    # Ensure stable ordering
    tvals = np.array(sorted(reservoir_data_frame["Timestep"].unique()))
    num_res_cells = coupled_model.reservoir.nx

    # Make an index to pivot quickly
    df_idx = reservoir_data_frame.set_index(["Timestep", "CellID"]).sort_index()

    def to_matrix(column):
        """
        Return a (nT, num_res_cells) matrix for the given column (trimmed).
        """
        full_idx = pd.MultiIndex.from_product(
            [tvals, sorted(reservoir_data_frame["CellID"].unique())],
            names=["Timestep", "CellID"],
        )
        s = df_idx[column].reindex(full_idx)
        matrix_full = s.values.reshape(len(tvals), -1)  # all cells
        return matrix_full[:, :num_res_cells]  # keep only the first num_res_cells

    reservoir_property_matrix = to_matrix(prop_name_in_reservoir_output)
    if property_name == "temperature":
        reservoir_property_matrix -= 273.15

    # Generate a colormap for the report steps
    cmap = mpl.colormaps['jet']
    colors = [cmap(i / len(report_step_labels)) for i in range(len(report_step_labels))]

    # Define markers and line styles
    markers = ['o', 's', 'd', '^', 'v', 'x', '*']
    linestyles = ['-', '--', '-.', ':']

    # --- Build the plot ---
    fig, ax = plt.subplots(figsize=(6, 5))  # single compact panel
    y_r, offset, rmin, rmax = stacked_y_axis_linear_log(
        ax,
        segments_depths,
        reservoir_radial_distance,
        gap_ratio=0.05,
        n_ticks_well=5,
        add_minor=True,
    )

    for idx, report_index in enumerate(report_indices):
        well_prop_profile = well_data_frame[prop_name_in_well_output][
            report_index * num_segments : (report_index + 1) * num_segments
        ]
        if property_name == "temperature":
            well_prop_profile -= 273.15

        color = colors[idx]  # Assign color from the colormap
        marker = markers[idx % len(markers)]  # Cycle through markers
        linestyle = linestyles[idx % len(linestyles)]  # Cycle through line styles

        ax.plot(
            well_prop_profile,
            segments_depths,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0,  # match linewidth
            markersize=4.0,  # match markersize
            color=color,
            label=report_step_labels[idx],
        )

        reservoir_prop_profile = reservoir_property_matrix[idx, :]
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

    # (optional) subtle grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)

    plt.legend(
        fontsize=7,
        loc=legend_loc,
        title="Report steps",
        title_fontsize=10,
    ).get_frame().set_edgecolor('black')  # Optional: Add a border
    fig.tight_layout()
    fig.savefig(
        f"{property_name}_well_reservoir_profile.pdf", dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        f"{property_name}_well_reservoir_profile.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def stacked_y_axis_linear_log(
    ax,
    segments_depths,
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
    :param segments_depths: The array of the depths of the segment centroids
    :type segments_depths: array-like
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
    z = np.asarray(segments_depths, float)
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
        rf"$r_{{\mathrm{{reservoir}}}}\,[\mathrm{{m}}]$ (log$_{{{log_base}}}$)",
        transform=ax.transAxes,
        rotation=90,
        va='center',
        ha='right',
        clip_on=False,
    )

    return y_r, offset, rmin, rmax
