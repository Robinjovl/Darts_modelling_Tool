from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from darts.tools.hdf5_tools import load_hdf5_to_dict


@dataclass(frozen=True)
class ScenarioProfile:
    """
    Input folder and legend label for one plotted scenario.

    :param path: Path to the scenario output folder.
    :param label: Legend label used for the scenario.
    """

    path: str
    label: str


def plot_well_segment_property_vs_time(
    scenarios: Sequence[ScenarioProfile | tuple[str, str]],
    well_name: str,
    property_key: str,
    segment_index: int,
    output_path: str,
    *,
    num_segments: int | None = None,
    min_time_step_idx: int = 0,
    well_data_filename: str = "well_data.h5",
    well_props_filename_template: str = "dfm_well_props_{well_name}.pkl",
    time_unit: str = "seconds",
    property_offset: float = 0.0,
    y_label: str | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    y_tick_increment: float | None = None,
    log_x: bool = True,
    legend_loc: str = "best",
    show_plot: bool = True,
    figure_size: tuple[float, float] = (8, 5),
    marker_size: float = 5.0,
    line_width: float = 2.0,
) -> str:
    """
    Plot one well segment property over time for multiple output folders.

    Each scenario output folder must contain ``well_data_filename`` and the
    pickle file resolved from ``well_props_filename_template``.

    :param scenarios: Output folders and legend labels.
    :param well_name: Name used in the well property file template.
    :param property_key: DataFrame column to plot on the y-axis.
    :param segment_index: Zero-based segment index to plot. Negative values
                          count from the bottom, so -1 plots the bottom segment.
    :param output_path: Path for the saved figure.
    :param num_segments: Number of well segments. If not available, it is
                         inferred from the property rows and time steps.
    :param min_time_step_idx: Number of leading time steps to skip.
    :param well_data_filename: HDF5 filename inside each output folder.
    :param well_props_filename_template: Pickle filename template. It may contain
                                         {well_name}.
    :param time_unit: One of "days", "hours", "minutes", or "seconds".
    :param property_offset: Offset added to property values, for unit conversions.
    :param y_label: Optional y-axis label. Defaults to property_key.
    :param y_min: Optional y-axis minimum.
    :param y_max: Optional y-axis maximum.
    :param y_tick_increment: Optional y tick spacing.
    :param log_x: If True, use a logarithmic x-axis.
    :param legend_loc: Matplotlib legend location.
    :param show_plot: If True, display the figure.
    :param figure_size: Matplotlib figure size.
    :param marker_size: Marker size.
    :param line_width: Line width.
    :return: Saved output path.
    """
    scenarios = _normalise_scenarios(scenarios)
    _validate_scenarios(scenarios)
    if min_time_step_idx < 0:
        raise ValueError("min_time_step_idx must be non-negative.")
    if num_segments is not None and num_segments <= 0:
        raise ValueError("num_segments must be positive when specified.")

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=figure_size)

    x_max = None
    for idx, scenario in enumerate(scenarios):
        output_folder = scenario.path
        well_data_file_path = os.path.join(output_folder, well_data_filename)
        h5_well_data = load_hdf5_to_dict(well_data_file_path)
        simulated_time = _convert_time(
            np.asarray(h5_well_data["dynamic"]["time"], dtype=float), time_unit
        )

        well_props_filename = well_props_filename_template.format(well_name=well_name)
        well_props_file_path = os.path.join(output_folder, well_props_filename)
        data_frame = pd.read_pickle(well_props_file_path)

        scenario_num_segments = _infer_num_segments(
            data_frame,
            property_key,
            simulated_time,
            num_segments,
        )
        resolved_segment_index = _resolve_segment_index(
            segment_index,
            scenario_num_segments,
        )
        if (
            resolved_segment_index < 0
            or resolved_segment_index >= scenario_num_segments
        ):
            raise ValueError(
                f"segment_index {segment_index} is outside the available "
                f"{scenario_num_segments} segments for '{scenario.label}'."
            )

        property_time_series = _get_segment_time_series(
            data_frame,
            property_key,
            scenario_num_segments,
            resolved_segment_index,
        )
        if len(property_time_series) != len(simulated_time):
            raise ValueError(
                f"Scenario '{scenario.label}' has {len(property_time_series)} "
                f"property time values and {len(simulated_time)} HDF5 time values."
            )

        x_values, y_values = _select_time_window(
            simulated_time,
            property_time_series + property_offset,
            min_time_step_idx,
            log_x,
        )
        if len(x_values) == 0:
            raise ValueError(
                f"Scenario '{scenario.label}' has no plottable time values after "
                "applying min_time_step_idx and log-axis filtering."
            )
        x_max = max(x_max or x_values[0], float(np.max(x_values)))

        plotter = ax.semilogx if log_x else ax.plot
        plotter(
            x_values,
            y_values,
            linestyle="",
            marker=_MARKERS[idx % len(_MARKERS)],
            linewidth=line_width,
            markersize=marker_size,
            label=scenario.label,
        )

    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    if y_tick_increment is not None:
        lower, upper = ax.get_ylim()
        tick_start = y_min if y_min is not None else lower
        tick_stop = y_max if y_max is not None else upper
        ax.set_yticks(
            np.arange(tick_start, tick_stop + y_tick_increment, y_tick_increment)
        )

    ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.3)
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.3)
    ax.set_xlabel(_get_time_axis_label(time_unit), labelpad=6)
    ax.set_ylabel(y_label or property_key, labelpad=6)
    _add_legend(ax, legend_loc)

    if log_x and x_max is not None:
        lower, _upper = ax.get_xlim()
        ax.set_xlim(lower, x_max)

    fig.tight_layout()
    _save_figure(fig, output_path)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return output_path


_MARKERS = ("o", "s", "D", "^", "v", "None")


def _normalise_scenarios(
    scenarios: Sequence[ScenarioProfile | tuple[str, str]],
) -> list[ScenarioProfile]:
    normalised = []
    for scenario in scenarios:
        if isinstance(scenario, ScenarioProfile):
            normalised.append(scenario)
        else:
            path, label = scenario
            normalised.append(ScenarioProfile(path, label))
    return normalised


def _validate_scenarios(scenarios: Sequence[ScenarioProfile]) -> None:
    if len(scenarios) == 0:
        raise ValueError("At least one scenario must be provided.")


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.0,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _add_legend(ax: Axes, legend_loc: str) -> None:
    leg = ax.legend(frameon=False, loc=legend_loc, handlelength=3)
    if leg.get_title() is not None:
        leg.get_title().set_fontsize(12)


def _infer_num_segments(
    data_frame: pd.DataFrame,
    property_key: str,
    simulated_time: np.ndarray,
    num_segments: int | None,
) -> int:
    if property_key not in data_frame.columns:
        raise ValueError(f"Column '{property_key}' was not found in input data.")
    if num_segments is not None:
        return num_segments

    num_times = len(simulated_time)
    num_values = len(data_frame[property_key])
    if num_times == 0 or num_values % num_times != 0:
        raise ValueError(
            "Could not infer num_segments from property rows and HDF5 time values. "
            "Pass num_segments explicitly."
        )
    return num_values // num_times


def _get_segment_time_series(
    data_frame: pd.DataFrame,
    property_key: str,
    num_segments: int,
    segment_index: int,
) -> np.ndarray:
    values = data_frame[property_key].iloc[segment_index::num_segments]
    return values.to_numpy(dtype=float)


def _resolve_segment_index(segment_index: int, num_segments: int) -> int:
    if segment_index < 0:
        return num_segments + segment_index
    return segment_index


def _convert_time(time_days: np.ndarray, time_unit: str) -> np.ndarray:
    scale_by_unit = {
        "days": 1.0,
        "hours": 24.0,
        "minutes": 24.0 * 60.0,
        "seconds": 24.0 * 60.0 * 60.0,
    }
    try:
        return time_days * scale_by_unit[time_unit]
    except KeyError as exc:
        raise ValueError(
            "time_unit must be one of 'days', 'hours', 'minutes', or 'seconds'."
        ) from exc


def _get_time_axis_label(time_unit: str) -> str:
    label_by_unit = {
        "days": "Simulated time [day]",
        "hours": "Simulated time [hour]",
        "minutes": "Simulated time [minute]",
        "seconds": "Simulated time [second]",
    }
    return label_by_unit[time_unit]


def _select_time_window(
    x_values: np.ndarray,
    y_values: np.ndarray,
    min_time_step_idx: int,
    log_x: bool,
) -> tuple[np.ndarray, np.ndarray]:
    selected_x = x_values[min_time_step_idx:]
    selected_y = y_values[min_time_step_idx:]
    if log_x:
        positive_time = selected_x > 0.0
        selected_x = selected_x[positive_time]
        selected_y = selected_y[positive_time]
    return selected_x, selected_y


def _save_figure(fig: Figure, output_path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path)
