from __future__ import annotations

import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter

from darts.models.darts_model import DartsModel
from darts.tools.hdf5_tools import load_hdf5_to_dict


@dataclass(frozen=True)
class WellProfilePlotSpec:
    """
    Description of one well-property profile plot.

    :param property_key: DataFrame column to plot.
    :param label: Optional x-axis label.
    :param output_name: Optional saved-file stem.
    :param component_index: Optional component index for vector-valued columns.
    :param offset: Offset added to values, for unit conversions.
    """

    property_key: str
    label: str | None = None
    output_name: str | None = None
    component_index: int | None = None
    offset: float = 0.0


def plot_well_property_profiles(
    well_name: str,
    coupled_model: DartsModel,
    reported_times: Sequence[float] | None = None,
    *,
    report_time_labels: Sequence[str] | None = None,
    properties: Sequence[str | WellProfilePlotSpec] | None = None,
    property_labels: dict[str, str] | None = None,
    property_offsets: dict[str, float] | None = None,
    output_dir: str | None = None,
    save_as: str | Sequence[str] | None = "pdf",
    show_plot: bool = True,
    cmap_name: str = "jet",
    figure_size: tuple[float, float] = (10, 6),
    marker_size: float = 4.0,
    line_width: float = 1.5,
    time_atol: float = 1e-10,
    time_rtol: float = 1e-8,
) -> dict[str, list[str]]:
    """
    Plot configurable property profiles for a DFM well.

    If reported_times is omitted, every available time in the well HDF5 file is
    plotted. If properties is omitted, scalar columns and numeric components of
    vector-valued columns in the well property DataFrame are plotted.

    :param well_name: Name of the well to plot.
    :param coupled_model: An initialized DartsModel with output files available.
    :param reported_times: Simulation times to plot, in days.
    :param report_time_labels: Optional legend labels for reported_times.
    :param properties: Optional property names or WellProfilePlotSpec objects.
                       Vector components can be requested as "column:index".
    :param property_labels: Optional x-axis labels keyed by property name or
                            "column:index" component key.
    :param property_offsets: Optional value offsets keyed by property name or
                             "column:index" component key.
    :param output_dir: Optional output directory. Defaults to a subfolder in the
                       model output folder.
    :param save_as: File extension or iterable of extensions to save. If None,
                    figures are not saved.
    :param show_plot: If True, display each figure.
    :param cmap_name: Matplotlib colormap name for report-step colors.
    :param time_atol: Absolute tolerance for matching requested times.
    :param time_rtol: Relative tolerance for matching requested times.
    :return: Dictionary mapping each plotted property key to saved figure paths.
    """
    well_props_file_path = os.path.join(
        coupled_model.output_folder, f"dfm_well_props_{well_name}.pkl"
    )
    data_frame = pd.read_pickle(well_props_file_path)

    h5_well_dict = load_hdf5_to_dict(coupled_model.well_filepath)
    simulated_time = np.asarray(h5_well_dict["dynamic"]["time"], dtype=float)
    report_indices, report_time_labels = _get_report_steps(
        simulated_time,
        reported_times,
        report_time_labels,
        time_atol,
        time_rtol,
    )

    well_geom = coupled_model.wells[well_name].geometry
    num_segments = well_geom.num_segments
    segment_depths = np.asarray(well_geom.z, dtype=float)

    plot_specs = _get_plot_specs(
        data_frame,
        properties,
        property_labels or {},
        property_offsets or {},
    )
    if len(plot_specs) == 0:
        raise ValueError("No plottable well properties were found.")

    output_dir = _prepare_output_dir(coupled_model, well_name, output_dir)
    saved_files = {}
    for spec in plot_specs:
        fig, ax = _plot_profile(
            data_frame,
            spec,
            report_indices,
            report_time_labels,
            num_segments,
            segment_depths,
            cmap_name,
            figure_size,
            marker_size,
            line_width,
        )
        saved_files[_get_spec_key(spec)] = _save_profile_figure(
            fig,
            output_dir,
            spec,
            save_as,
        )
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return saved_files


def _get_report_steps(
    simulated_time: np.ndarray,
    reported_times: Sequence[float] | None,
    report_time_labels: Sequence[str] | None,
    time_atol: float,
    time_rtol: float,
) -> tuple[list[int], list[str]]:
    if reported_times is None:
        report_indices = list(range(len(simulated_time)))
        selected_times = simulated_time
    else:
        selected_times = np.asarray(reported_times, dtype=float)
        if len(selected_times) == 0:
            raise ValueError("reported_times must contain at least one time value.")
        report_indices = []
        for requested_time in selected_times:
            matches = np.where(
                np.isclose(
                    simulated_time,
                    requested_time,
                    atol=time_atol,
                    rtol=time_rtol,
                )
            )[0]
            if len(matches) == 0:
                nearest_idx = int(np.argmin(np.abs(simulated_time - requested_time)))
                raise ValueError(
                    f"Requested time {requested_time:g} day was not found in "
                    f"{os.path.basename('well_data.h5')}. Nearest available time "
                    f"is {simulated_time[nearest_idx]:g} day."
                )
            report_indices.append(int(matches[0]))

    if report_time_labels is None:
        labels = [_format_time_label(time) for time in selected_times]
    else:
        labels = list(report_time_labels)
        if len(labels) != len(report_indices):
            raise ValueError(
                "report_time_labels must have the same length as reported_times."
            )

    return report_indices, labels


def _get_plot_specs(
    data_frame: pd.DataFrame,
    properties: Sequence[str | WellProfilePlotSpec] | None,
    property_labels: dict[str, str],
    property_offsets: dict[str, float],
) -> list[WellProfilePlotSpec]:
    if properties is None:
        return _infer_plot_specs(data_frame, property_labels, property_offsets)

    specs = []
    for property_spec in properties:
        if isinstance(property_spec, WellProfilePlotSpec):
            specs.append(property_spec)
            continue

        property_key, component_index = _parse_property_name(property_spec)
        spec_key = _make_spec_key(property_key, component_index)
        specs.append(
            WellProfilePlotSpec(
                property_key=property_key,
                label=property_labels.get(spec_key, property_labels.get(property_key)),
                output_name=make_safe_filename(spec_key),
                component_index=component_index,
                offset=property_offsets.get(
                    spec_key,
                    property_offsets.get(property_key, 0.0),
                ),
            )
        )

    _validate_plot_specs(data_frame, specs)
    return specs


def _infer_plot_specs(
    data_frame: pd.DataFrame,
    property_labels: dict[str, str],
    property_offsets: dict[str, float],
) -> list[WellProfilePlotSpec]:
    specs = []
    for column in data_frame.columns:
        sample = _get_first_valid_value(data_frame[column])
        if sample is None:
            continue
        sample_array = np.asarray(sample)
        if sample_array.ndim == 0:
            specs.append(
                _make_plot_spec(column, None, property_labels, property_offsets)
            )
        elif sample_array.ndim == 1:
            for component_index in range(len(sample_array)):
                specs.append(
                    _make_plot_spec(
                        column,
                        component_index,
                        property_labels,
                        property_offsets,
                    )
                )

    return [
        spec
        for spec in specs
        if _is_numeric_profile(data_frame, spec.property_key, spec.component_index)
    ]


def _make_plot_spec(
    property_key: str,
    component_index: int | None,
    property_labels: dict[str, str],
    property_offsets: dict[str, float],
) -> WellProfilePlotSpec:
    spec_key = _make_spec_key(property_key, component_index)
    return WellProfilePlotSpec(
        property_key=property_key,
        label=property_labels.get(spec_key, property_labels.get(property_key)),
        output_name=make_safe_filename(spec_key),
        component_index=component_index,
        offset=property_offsets.get(spec_key, property_offsets.get(property_key, 0.0)),
    )


def _validate_plot_specs(
    data_frame: pd.DataFrame,
    specs: Sequence[WellProfilePlotSpec],
) -> None:
    for spec in specs:
        if spec.property_key not in data_frame.columns:
            raise ValueError(
                f"Column '{spec.property_key}' was not found in well property data."
            )
        if not _is_numeric_profile(data_frame, spec.property_key, spec.component_index):
            raise ValueError(
                f"Column '{_get_spec_key(spec)}' is not numeric and cannot be plotted."
            )


def _plot_profile(
    data_frame: pd.DataFrame,
    spec: WellProfilePlotSpec,
    report_indices: Sequence[int],
    report_time_labels: Sequence[str],
    num_segments: int,
    segment_depths: np.ndarray,
    cmap_name: str,
    figure_size: tuple[float, float],
    marker_size: float,
    line_width: float,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figure_size)
    colors = _get_report_colors(len(report_indices), cmap_name)
    markers = ("o", "s", "d", "^", "v", "x", "*")
    line_styles = ("-", "--", "-.", ":")

    for idx, report_index in enumerate(report_indices):
        profile = _get_profile_values(data_frame, spec, report_index, num_segments)
        ax.plot(
            profile,
            segment_depths,
            marker=markers[idx % len(markers)],
            linestyle=line_styles[idx % len(line_styles)],
            color=colors[idx],
            label=report_time_labels[idx],
            markersize=marker_size,
            linewidth=line_width,
        )

    ax.set_xlabel(_get_axis_label(spec), fontsize=16, labelpad=10)
    ax.set_ylabel("Well segment depth [m]", fontsize=16, labelpad=10)
    ax.tick_params(axis="both", labelsize=12)
    ax.invert_yaxis()
    ax.grid(linestyle="--")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.legend(
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.05, 1.05),
        ncol=1,
        title="Report steps",
        title_fontsize=12,
    ).get_frame().set_edgecolor("black")
    ax.legend_.set_frame_on(True)
    fig.tight_layout()
    return fig, ax


def _get_profile_values(
    data_frame: pd.DataFrame,
    spec: WellProfilePlotSpec,
    report_index: int,
    num_segments: int,
) -> np.ndarray:
    start = report_index * num_segments
    stop = start + num_segments
    values = data_frame[spec.property_key].iloc[start:stop]
    if len(values) != num_segments:
        raise ValueError(
            f"Column '{spec.property_key}' has no complete profile at time "
            f"index {report_index}."
        )

    if spec.component_index is None:
        profile = values.to_numpy(dtype=float)
    else:
        profile = np.asarray(
            [np.asarray(value)[spec.component_index] for value in values],
            dtype=float,
        )
    return profile + spec.offset


def _prepare_output_dir(
    coupled_model: DartsModel,
    well_name: str,
    output_dir: str | None,
) -> str:
    remove_existing = output_dir is None
    if output_dir is None:
        output_dir = os.path.join(
            coupled_model.output_folder,
            f"well_prop_profiles_{well_name}",
        )

    output_dir = os.path.normpath(output_dir)
    if remove_existing and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_profile_figure(
    fig: Figure,
    output_dir: str,
    spec: WellProfilePlotSpec,
    save_as: str | Sequence[str] | None,
) -> list[str]:
    if save_as is None:
        return []
    if isinstance(save_as, str):
        extensions = (save_as,)
    else:
        extensions = tuple(save_as)

    output_name = spec.output_name or make_safe_filename(_get_spec_key(spec))
    saved_files = []
    for extension in extensions:
        extension = extension.lstrip(".")
        file_path = os.path.join(output_dir, f"{output_name}_profiles.{extension}")
        fig.savefig(file_path)
        saved_files.append(file_path)
    return saved_files


def _parse_property_name(property_name: str) -> tuple[str, int | None]:
    if ":" not in property_name:
        return property_name, None

    property_key, component_index = property_name.rsplit(":", 1)
    try:
        return property_key, int(component_index)
    except ValueError as exc:
        raise ValueError(
            "Vector-valued properties must be requested as 'column:index'."
        ) from exc


def _get_first_valid_value(series: pd.Series) -> object | None:
    values = series.dropna()
    if values.empty:
        return None
    return values.iloc[0]


def _is_numeric_profile(
    data_frame: pd.DataFrame,
    property_key: str,
    component_index: int | None,
) -> bool:
    sample = _get_first_valid_value(data_frame[property_key])
    if sample is None:
        return False
    try:
        if component_index is None:
            float(sample)
        else:
            float(np.asarray(sample)[component_index])
    except (IndexError, TypeError, ValueError):
        return False
    return True


def _get_report_colors(
    num_reports: int, cmap_name: str
) -> list[tuple[float, float, float, float]]:
    cmap = mpl.colormaps[cmap_name]
    if num_reports == 1:
        return [cmap(0.0)]
    return [cmap(value) for value in np.linspace(0.0, 1.0, num_reports)]


def _get_axis_label(spec: WellProfilePlotSpec) -> str:
    if spec.label is not None:
        return spec.label
    return _get_spec_key(spec)


def _get_spec_key(spec: WellProfilePlotSpec) -> str:
    return _make_spec_key(spec.property_key, spec.component_index)


def _make_spec_key(property_key: str, component_index: int | None) -> str:
    if component_index is None:
        return property_key
    return f"{property_key}:{component_index}"


def _format_time_label(time_days: float) -> str:
    seconds = time_days * 24.0 * 60.0 * 60.0
    if np.isclose(seconds, 0.0):
        return "0 s"
    if seconds < 60.0:
        return f"{seconds:g} s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:g} min"
    hours = minutes / 60.0
    if hours < 24.0:
        return f"{hours:g} h"
    return f"{time_days:g} d"


def make_safe_filename(value: str) -> str:
    """
    Convert free-form text to a filesystem-safe filename stem.

    :param value: Input value.
    :return: Filename-safe value.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
