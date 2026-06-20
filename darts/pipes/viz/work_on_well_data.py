from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator


@dataclass(frozen=True)
class ScenarioProfile:
    """
    Input pickle file and legend label for one plotted scenario.

    :param path: Path to the scenario pickle file.
    :param label: Legend label used for the scenario.
    """

    path: str
    label: str


def plot_well_property_profiles_from_pickles(
    scenarios: Sequence[ScenarioProfile | tuple[str, str]],
    property_key: str,
    output_path: str,
    *,
    num_segments: int,
    profile_index: int = -1,
    segment_positions: Sequence[float] | None = None,
    segment_length: float | None = None,
    property_offset: float = 0.0,
    x_label: str | None = None,
    y_label: str | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    y_tick_increment: float | None = None,
    top_x_axis: bool = True,
    legend_loc: str = "best",
    show_plot: bool = True,
    figure_size: tuple[float, float] = (8, 5),
    marker_size: float = 4.0,
    line_width: float = 2.0,
) -> str:
    """
    Plot a stacked well profile from pickled DataFrames for multiple scenarios.

    :param scenarios: Pickle paths and legend labels.
    :param property_key: DataFrame column to plot on the x-axis.
    :param output_path: Path for the saved figure.
    :param num_segments: Number of well segments in each stacked profile.
    :param profile_index: Profile index in the stacked data. The default -1 plots
                          the final complete profile.
    :param segment_positions: Optional explicit y-axis segment positions.
    :param segment_length: Optional constant segment length used to build segment
                           center positions.
    :param property_offset: Offset added to property values, for unit conversions.
    :param x_label: Optional x-axis label. Defaults to property_key.
    :param y_label: Optional y-axis label. Defaults to segment index or segment MD.
    :param x_min: Optional x-axis minimum.
    :param x_max: Optional x-axis maximum.
    :param y_min: Optional y-axis minimum.
    :param y_max: Optional y-axis maximum.
    :param y_tick_increment: Optional y tick spacing.
    :param top_x_axis: If True, place x-axis ticks and label at the top.
    :param legend_loc: Matplotlib legend location.
    :param show_plot: If True, display the figure.
    :param figure_size: Matplotlib figure size.
    :param marker_size: Marker size.
    :param line_width: Line width.
    :return: Saved output path.
    """
    scenarios = _normalise_scenarios(scenarios)
    _validate_scenarios(scenarios)
    if num_segments <= 0:
        raise ValueError("num_segments must be positive.")

    y_values = _get_segment_positions(num_segments, segment_positions, segment_length)

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=figure_size)

    for idx, scenario in enumerate(scenarios):
        data_frame = pd.read_pickle(scenario.path)
        x_values = (
            _get_stacked_profile(data_frame, property_key, num_segments, profile_index)
            + property_offset
        )
        ax.plot(
            x_values,
            y_values,
            linestyle=_LINESTYLES[idx % len(_LINESTYLES)],
            marker=_MARKERS[idx % len(_MARKERS)],
            linewidth=line_width,
            markersize=marker_size,
            label=scenario.label,
        )

    if x_min is not None or x_max is not None:
        ax.set_xlim(x_min, x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    if y_tick_increment is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_increment))

    ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.3)
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.3)
    ax.set_xlabel(x_label or property_key, labelpad=6)
    ax.set_ylabel(y_label or _get_segment_axis_label(segment_length), labelpad=6)
    ax.invert_yaxis()

    if top_x_axis:
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

    _add_legend(ax, legend_loc)

    fig.tight_layout()
    _save_figure(fig, output_path)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return output_path


_LINESTYLES = ("-", "--", "-.", ":", "-", "--")
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


def _get_segment_positions(
    num_segments: int,
    segment_positions: Sequence[float] | None,
    segment_length: float | None,
) -> np.ndarray:
    if segment_positions is not None:
        values = np.asarray(segment_positions, dtype=float)
        if len(values) != num_segments:
            raise ValueError(
                "segment_positions must contain exactly num_segments values."
            )
        return values
    if segment_length is not None:
        if segment_length <= 0.0:
            raise ValueError("segment_length must be positive.")
        return (np.arange(num_segments, dtype=float) + 0.5) * segment_length
    return np.arange(num_segments, dtype=float)


def _get_segment_axis_label(segment_length: float | None) -> str:
    if segment_length is None:
        return "Segment index [-]"
    return "Segment MD [m]"


def _get_stacked_profile(
    data_frame: pd.DataFrame,
    property_key: str,
    num_segments: int,
    profile_index: int,
) -> np.ndarray:
    if property_key not in data_frame.columns:
        raise ValueError(f"Column '{property_key}' was not found in input data.")
    series = data_frame[property_key]
    if len(series) % num_segments != 0:
        raise ValueError(
            f"Column '{property_key}' has {len(series)} rows, which is not an "
            f"integer multiple of num_segments={num_segments}."
        )
    num_profiles = len(series) // num_segments
    selected_profile_index = profile_index
    if selected_profile_index < 0:
        selected_profile_index = num_profiles + selected_profile_index
    if selected_profile_index < 0 or selected_profile_index >= num_profiles:
        raise ValueError(
            f"profile_index {profile_index} is outside the available "
            f"{num_profiles} profiles."
        )

    start = selected_profile_index * num_segments
    stop = start + num_segments
    return series.iloc[start:stop].to_numpy(dtype=float)


def _save_figure(fig: Figure, output_path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path)
