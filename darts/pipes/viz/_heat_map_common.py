"""Shared building blocks of the well heat-map plotters.

:func:`darts.pipes.viz.plot_heat_map_contourf.plot_heat_map_contourf` and
:func:`darts.pipes.viz.plot_heat_map_pcolormesh.plot_heat_map_pcolormesh` draw
the same set of well properties and differ only in the matplotlib call used to
paint a single frame. Everything they share lives here:

* :func:`prepare_heat_maps` loads the stored well properties, builds the axes,
  and returns the list of properties to plot,
* :data:`HeatMapProperty` describes one plotted property declaratively (which
  column of the well-property data frame it reads, how it is labelled, and how
  its values are prepared),
* :func:`render_heat_maps` walks that list, extracts the matrix of each
  property, and delegates the actual drawing to the caller's ``draw`` callback.

The property table is derived from the property container, so the phase-suffixed
column names (``sG``, ``rhoL``, ``xCO2_in_G_mass``, ...) written by
:func:`darts.pipes.save_results.save_dfm_well_props` follow
``property_container.phases_name`` instead of being hard-coded.
"""

import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from darts.tools.hdf5_tools import load_hdf5_to_dict

#: Seconds per day, used to convert the stored per-day rates and velocities.
SECONDS_PER_DAY = 24 * 60 * 60

#: Size of every heat-map figure.
FIGURE_SIZE = (12, 6)

#: Rows of a property matrix are either well segments or segment interfaces.
SEGMENTS = "segments"
INTERFACES = "interfaces"

#: Masking rules applied to a property matrix before it is drawn.
MASK_ZERO = "zero"  # hide values equal to zero (unfilled phase properties)
MASK_INVALID = "invalid"  # hide non-finite values (undefined phase rates)

#: Long names of the phases whose short names are used in the column names.
PHASE_DISPLAY_NAMES = {"G": "Gas", "L": "Liquid"}

#: Phase rate types written by :func:`darts.pipes.save_results.save_dfm_well_props`.
PHASE_RATE_SPECS = (
    ("molar", "molar", "kmol/s"),
    ("mass", "mass", "kg/s"),
    ("volumetric", "volumetric", "m$^3$/s"),
)


@dataclass(frozen=True)
class HeatMapProperty:
    """
    Declarative description of one property drawn as a heat map.

    :param column: Name of the well-property data frame column holding the values
    :param file_name: Name of the saved figure, without the counter and the extension
    :param cbar_label: Label of the colorbar, including the unit
    :param title: Title of the figure
    :param grid: Whether the values live on the segments or on the interfaces
    :param component_index: Index to pick from a vector-valued column (e.g. "z")
    :param divisor: Value the raw column is divided by (unit conversion)
    :param shift: Value subtracted from the raw column (unit conversion)
    :param mask: :data:`MASK_ZERO`, :data:`MASK_INVALID`, or None
    :param unit_range_fallback: Ignore NaNs when taking the value range and fall
        back to [0, 1] for a constant profile
    :param vlim: Fixed (vmin, vmax) of the color scale, used by pcolormesh only
    :param n_levels_key: Key of the number of contour levels in the bins mapping
    :param n_cmap_key: Key of the number of colormap bins in the bins mapping
    :param log_x: Whether a logarithmic x-axis is applied to this property
    :param cbar_label_mesh: Colorbar label of the pcolormesh renderer, when it
        historically differs from the one of the contourf renderer
    """

    column: str
    file_name: str
    cbar_label: str
    title: str
    grid: str = SEGMENTS
    component_index: int | None = None
    divisor: float = 1.0
    shift: float = 0.0
    mask: str | None = None
    unit_range_fallback: bool = False
    vlim: tuple[float, float] | None = None
    n_levels_key: str = "p"
    n_cmap_key: str = "p"
    log_x: bool = True
    cbar_label_mesh: str | None = None

    @property
    def mesh_cbar_label(self) -> str:
        """Colorbar label to use in the pcolormesh renderer."""
        return self.cbar_label if self.cbar_label_mesh is None else self.cbar_label_mesh


@dataclass
class HeatMapAxes:
    """Everything the renderers need that does not depend on the property."""

    main_dir: str
    data_frame: pd.DataFrame
    x: Sequence
    x_label: str
    y_segments: Sequence
    y_segments_label: str
    y_interfaces: Sequence
    y_interfaces_label: str
    num_segments: int
    num_interfaces: int
    time_step_idx_range: range
    num_selected_ts: int
    y_axis: str
    cmap_color: str
    font_size: float
    with_title: bool
    show_plot: bool
    save_as: str
    n_cmap_bins: dict[str, int] = field(default_factory=dict)


def get_contour_levels(min_value, max_value, num_bins):
    """
    Return strictly increasing contour levels, expanding equal bounds for
    constant profiles because contourf cannot plot non-increasing levels.
    """
    min_value = float(np.ma.filled(min_value, np.nan))
    max_value = float(np.ma.filled(max_value, np.nan))

    if not np.isfinite(min_value) or not np.isfinite(max_value):
        min_value, max_value = 0.0, 1.0
    elif min_value == max_value:
        delta = 1.0 if min_value == 0 else abs(min_value) * 0.05
        min_value -= delta
        max_value += delta

    return np.linspace(min_value, max_value, num_bins + 1)


def phase_display_name(phase_name: str) -> str:
    """Long name of a phase, falling back to the short name used in the columns."""
    return PHASE_DISPLAY_NAMES.get(phase_name, phase_name)


def mobile_phase_names(property_container, n_mobile_phases: int) -> list[str]:
    """
    Short names of the mobile phases, as used in the stored column names.

    :func:`darts.pipes.save_results.save_dfm_well_props` writes the ``s``,
    ``rho``, ``mu`` and ``x<component>_in_<phase>_mass`` columns keyed on
    ``property_container.phases_name``, so the phase suffixes are taken from
    there. The historical two-phase (G, L) and three-phase (G, L_a, L_b) lists
    are used only as a fallback for containers whose fluid phases do not
    describe the mobile phases of the well.
    """
    names = list(getattr(property_container, "phases_name", None) or [])
    np_fl = getattr(property_container, "np_fl", None)
    if isinstance(np_fl, int) and np_fl >= 0:
        names = names[:np_fl]
    if len(names) == n_mobile_phases:
        return names
    return ["G", "L_a", "L_b"] if n_mobile_phases == 3 else ["G", "L"]


def build_property_specs(
    property_container, n_mobile_phases: int, columns
) -> list[HeatMapProperty]:
    """
    Build the table of properties to plot, in the order they are plotted.

    :param property_container: The property container of the physics
    :param n_mobile_phases: Number of mobile phases of the well
    :param columns: Columns available in the well-property data frame
    :returns: The list of property specifications
    """
    pc = property_container
    columns = set(columns)
    components_names = list(pc.components_name)
    fluid_components_names = components_names[: pc.nc_fl]
    phases = mobile_phase_names(pc, n_mobile_phases)

    specs: list[HeatMapProperty] = []

    # Pressure
    if "pressure" in columns:
        specs.append(
            HeatMapProperty(
                column="pressure",
                file_name="Pressure",
                cbar_label="Pressure [bar]",
                title="Pressure profile along the wellbore over time",
                n_levels_key="p",
                n_cmap_key="p",
            )
        )

    # Overall mole fractions, one figure per component
    if "z" in columns:
        for comp_idx, comp_name in enumerate(components_names):
            specs.append(
                HeatMapProperty(
                    column="z",
                    component_index=comp_idx,
                    file_name=f"{comp_name} overall mole fraction",
                    cbar_label=f"{comp_name} overall mole fraction [-]",
                    title=(
                        f"Profile of overall mole fraction of {comp_name} "
                        "along the wellbore over time"
                    ),
                    unit_range_fallback=True,
                    n_levels_key="comp",
                    n_cmap_key="comp",
                )
            )

    # Temperature
    if pc.thermal and "temperature" in columns:
        specs.append(
            HeatMapProperty(
                column="temperature",
                file_name="Temperature",
                cbar_label="Temperature [°C]",
                title="Temperature profile along the wellbore over time",
                shift=273.15,
                n_levels_key="t",
                n_cmap_key="t",
            )
        )

    # Volume fractions. With two mobile phases only the first (gas) volume
    # fraction is plotted, the other one being its complement.
    volume_fraction_phases = phases if n_mobile_phases == 3 else phases[:1]
    for phase_idx, phase_name in enumerate(volume_fraction_phases):
        column = f"s{phase_name}"
        if column not in columns:
            continue
        display = phase_display_name(phase_name)
        specs.append(
            HeatMapProperty(
                column=column,
                file_name=f"{display} volume fraction",
                cbar_label=f"{display} volume fraction [-]",
                title=f"{display} volume fraction profile along the wellbore over time",
                # Only the first phase is drawn on a fixed [0, 1] color scale.
                vlim=(0.0, 1.0) if phase_idx == 0 else None,
                n_levels_key="s",
                n_cmap_key="s",
            )
        )

    # Component mass fractions in the phases
    for phase_name in phases:
        for comp_name in fluid_components_names:
            column = f"x{comp_name}_in_{phase_name}_mass"
            if column not in columns:
                continue
            specs.append(
                HeatMapProperty(
                    column=column,
                    file_name=f"{comp_name} mass fraction in the {phase_name} phase",
                    cbar_label=f"{comp_name} mass fraction in {phase_name} phase [-]",
                    cbar_label_mesh=(
                        f"{comp_name} mass fraction in the {phase_name} phase [-]"
                    ),
                    title=(
                        f"Profile of {comp_name} mass fraction in the {phase_name} "
                        "phase along the wellbore over time"
                    ),
                    unit_range_fallback=True,
                    vlim=(0.0, 1.0),
                    n_levels_key="comp",
                    n_cmap_key="comp",
                )
            )

    # Phase densities
    for phase_name in phases:
        column = f"rho{phase_name}"
        if column not in columns:
            continue
        display = phase_display_name(phase_name)
        specs.append(
            HeatMapProperty(
                column=column,
                file_name=f"{display} density",
                cbar_label=f"{display} density [kg/m$^3$]",
                title=f"{display} density profile along the wellbore over time",
                mask=MASK_ZERO,
                n_levels_key="rho",
                n_cmap_key="rho",
            )
        )

    # Phase viscosities
    for phase_idx, phase_name in enumerate(phases):
        column = f"mu{phase_name}"
        if column not in columns:
            continue
        display = phase_display_name(phase_name)
        # The first (gas) phase historically labels the unit in text mode and
        # the remaining phases in math mode; kept so the figures stay comparable.
        unit = "cP" if phase_idx == 0 else "$cP$"
        specs.append(
            HeatMapProperty(
                column=column,
                file_name=f"{display} viscosity",
                cbar_label=f"{display} viscosity [{unit}]",
                title=f"{display} viscosity profile along the wellbore over time",
                mask=MASK_ZERO,
                n_levels_key="mu",
                # The colormap of the viscosity is binned like the density one.
                n_cmap_key="rho",
            )
        )

    # Phase velocities. save_dfm_well_props always writes the two interface
    # velocities of the pipe model under the fixed names vG and vL, so these
    # columns are not derived from the phase list.
    for column, display in (("vG", "Gas"), ("vL", "Liquid")):
        if column not in columns:
            continue
        specs.append(
            HeatMapProperty(
                column=column,
                file_name=f"{display} velocity",
                cbar_label=f"{display} velocity [m/s]",
                title=f"{display} velocity profile along the wellbore over time",
                grid=INTERFACES,
                divisor=SECONDS_PER_DAY,
                mask=MASK_ZERO,
                n_levels_key="v",
                # The colormap of the velocity is binned like the density one.
                n_cmap_key="rho",
                # The velocity figures keep a linear x-axis.
                log_x=False,
            )
        )

    # Phase rates
    for rate_type, rate_label, unit in PHASE_RATE_SPECS:
        for phase_name in pc.phases_name:
            column = f"phase_{rate_type}_rate_{phase_name}"
            if column not in columns:
                continue
            display = phase_display_name(phase_name)
            specs.append(
                HeatMapProperty(
                    column=column,
                    file_name=f"{display} {rate_label} rate",
                    cbar_label=f"{display} {rate_label} rate [{unit}]",
                    title=(
                        f"{display} {rate_label} rate profile "
                        "along the wellbore over time"
                    ),
                    grid=INTERFACES,
                    divisor=SECONDS_PER_DAY,
                    mask=MASK_INVALID,
                    n_levels_key="v",
                    n_cmap_key="v",
                )
            )

    return specs


def prepare_heat_maps(
    well_name: str,
    coupled_model,
    output_folder_name: str,
    min_ts_idx: int = 0,
    max_ts_idx: int | None = None,
    x_axis: str = "simulated_time",
    y_axis: str = "segments_MD",
    cmap_color: str = "jet",
    save_as: str = "pdf",
    show_plot: bool = True,
    font_size: float = 14,
    with_title: bool = True,
    n_cmap_bins: dict[str, int] | None = None,
) -> tuple[HeatMapAxes, list[HeatMapProperty]]:
    """
    Reset the output directory, load the stored well properties, build the axes,
    and collect the properties to plot.

    :param well_name: Name of the well the properties of which will be plotted
    :param coupled_model: An instance of DartsModel
    :param output_folder_name: Name of the sub-folder the figures are saved in
    :param min_ts_idx: First time step index shown on the heat maps
    :param max_ts_idx: Time step index the heat maps are shown until. If not
        specified, the heat maps are shown till the last time step.
    :param x_axis: "simulated_time" or "time_step_index"
    :param y_axis: "segments_MD" or "segments_TVD" or "segment_index"
    :param cmap_color: Name of the colormap
    :param save_as: The extension of the image files that will be saved
    :param show_plot: Whether or not to show the plot
    :param font_size: Size of the fonts
    :param with_title: If you want the figures to have a title or not
    :param n_cmap_bins: Number of colorbar and colormap bins, per property group
    :returns: The axes shared by all figures and the properties to plot
    """
    main_dir = os.path.join(coupled_model.output_folder, output_folder_name)

    # Reset directory
    if os.path.exists(main_dir):
        shutil.rmtree(main_dir)
    os.makedirs(main_dir)

    # Well HDF5 file is used here to get the time step sizes
    h5_well_file_path = coupled_model.well_filepath
    h5_well_dict = load_hdf5_to_dict(h5_well_file_path)

    # Get well geometry info
    well_geom = coupled_model.wells[well_name].geometry
    num_segments = well_geom.num_segments
    num_interfaces = well_geom.num_interfaces
    n_mobile_phases = coupled_model.wells[well_name].n_mobile_phases

    # Get physics info
    pc = coupled_model.physics.property_containers[0]

    # Load primary vars and phase props
    well_props_file_path = os.path.join(
        coupled_model.output_folder, f"dfm_well_props_{well_name}.pkl"
    )
    data_frame = pd.read_pickle(well_props_file_path)

    num_ts = int(
        len(data_frame["sG"]) / num_segments
    )  # Initial conditions of sG is not stored.
    if max_ts_idx is None:
        max_ts_idx = num_ts
    assert max_ts_idx <= num_ts, (
        f"max_ts_idx is larger than the total number of time steps, which is {num_ts}!"
    )
    assert min_ts_idx < num_ts, (
        f"min_ts_idx is equal to or larger than the total number of time steps, which is {num_ts}!"
    )
    assert min_ts_idx < max_ts_idx, (
        f"min_ts_idx is equal to or larger than max_ts_idx, which is {max_ts_idx}!"
    )

    time_step_idx_range = range(min_ts_idx, max_ts_idx)

    if x_axis == "time_step_index":
        x = time_step_idx_range
        x_label = "Time step [-]"
    elif x_axis == "simulated_time":
        # Convert days to seconds over the user-specified index range. The
        # chained multiplication is kept as it is because it does not round
        # exactly like a multiplication by 86400.
        x = h5_well_dict["dynamic"]["time"][min_ts_idx:max_ts_idx] * 24 * 60 * 60
        x_label = "Simulated time [second]"

    if y_axis == "segment_index":
        y_segments = range(num_segments)
        y_segments_label = "Segment index [-]"
        y_interfaces = range(num_interfaces)
        y_interfaces_label = "Interface index [-]"
    elif y_axis == "segments_MD":
        y_segments = well_geom.z
        y_segments_label = "Segment MD [meter]"
        y_interfaces = well_geom.z_interfaces
        y_interfaces_label = "Interface MD [meter]"
    elif y_axis == "segments_TVD":
        y_segments = well_geom.TVD_segments
        y_segments_label = "Segment TVD [meter]"
        y_interfaces = well_geom.TVD_interfaces
        y_interfaces_label = "Interface TVD [meter]"

    axes = HeatMapAxes(
        main_dir=main_dir,
        data_frame=data_frame,
        x=x,
        x_label=x_label,
        y_segments=y_segments,
        y_segments_label=y_segments_label,
        y_interfaces=y_interfaces,
        y_interfaces_label=y_interfaces_label,
        num_segments=num_segments,
        num_interfaces=num_interfaces,
        time_step_idx_range=time_step_idx_range,
        num_selected_ts=len(time_step_idx_range),
        y_axis=y_axis,
        cmap_color=cmap_color,
        font_size=font_size,
        with_title=with_title,
        show_plot=show_plot,
        save_as=save_as,
        n_cmap_bins=dict(n_cmap_bins or {}),
    )

    specs = build_property_specs(pc, n_mobile_phases, data_frame.columns)

    return axes, specs


def extract_matrix(axes: HeatMapAxes, spec: HeatMapProperty):
    """
    Build the (rows x time steps) matrix of one property, applying its unit
    conversion and its masking rule.
    """
    num_rows = axes.num_interfaces if spec.grid == INTERFACES else axes.num_segments
    matrix = np.zeros((num_rows, axes.num_selected_ts))

    column = axes.data_frame[spec.column]
    for ts_idx, ts_counter in enumerate(axes.time_step_idx_range):
        chunk = column[
            ts_counter * axes.num_segments : (ts_counter + 1) * axes.num_segments
        ]
        if spec.component_index is None:
            values = chunk.to_numpy(dtype=float)
        else:
            values = np.array(
                [entry[spec.component_index] for entry in chunk.tolist()], dtype=float
            )
        if spec.grid == INTERFACES:
            # The interface values are stored padded to the number of segments
            values = values[:-1]
        matrix[:, ts_idx] = values / spec.divisor - spec.shift

    if spec.mask == MASK_ZERO:
        # Hide the values equal to zero
        matrix = np.ma.masked_where(matrix == 0, matrix)
    elif spec.mask == MASK_INVALID:
        matrix = np.ma.masked_invalid(matrix)

    return matrix


def value_range(matrix, spec: HeatMapProperty) -> tuple[float, float]:
    """
    Lower and upper bound of the color scale of one property matrix.

    NaNs are ignored where the property allows them, and a constant profile then
    falls back to the [0, 1] range.
    """
    if spec.unit_range_fallback:
        min_value, max_value = np.nanmin(matrix), np.nanmax(matrix)
        if min_value == max_value:
            return 0.0, 1.0
        return min_value, max_value
    return np.min(matrix), np.max(matrix)


def render_heat_maps(
    axes: HeatMapAxes,
    specs: Sequence[HeatMapProperty],
    draw: Callable,
) -> None:
    """
    Draw and save one figure per property.

    :param axes: The axes shared by all figures, from :func:`prepare_heat_maps`
    :param specs: The properties to plot, from :func:`prepare_heat_maps`
    :param draw: Renderer callback ``draw(fig, ax, spec, matrix, y_values)``
        painting one frame and its colorbar. Returning False skips the figure.
    """
    # rc_context keeps any rcParam touched while rendering out of the global state
    with plt.rc_context():
        for figure_counter, spec in enumerate(specs):
            matrix = extract_matrix(axes, spec)
            if spec.grid == INTERFACES:
                y_values, y_label = axes.y_interfaces, axes.y_interfaces_label
            else:
                y_values, y_label = axes.y_segments, axes.y_segments_label

            fig, ax = plt.subplots(figsize=FIGURE_SIZE)

            if draw(fig, ax, spec, matrix, y_values) is False:
                plt.close(fig)
                continue

            # Reverse the y-axis
            ax.invert_yaxis()

            # Add axes labels
            ax.set_xlabel(axes.x_label, fontsize=axes.font_size)
            ax.set_ylabel(y_label, fontsize=axes.font_size)

            # Set the font size of tick labels
            ax.tick_params(axis="both", labelsize=axes.font_size)

            # Add title
            if axes.with_title:
                ax.set_title(spec.title, fontsize=axes.font_size, fontweight="bold")

            fig.tight_layout()
            file_address = os.path.join(
                axes.main_dir,
                f"{figure_counter}- {spec.file_name}.{axes.save_as}",
            )
            fig.savefig(file_address)
            if axes.show_plot:
                plt.show()

            plt.close(fig)
