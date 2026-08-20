from matplotlib.ticker import MultipleLocator

from darts.models.darts_model import DartsModel
from darts.pipes.viz._heat_map_common import prepare_heat_maps, render_heat_maps


def plot_heat_map_pcolormesh(
    well_name: str,
    coupled_model: DartsModel,
    max_ts_idx: int = None,
    x_axis: str = "simulated_time",
    y_axis: str = "segments_MD",
    cmap_color: str = "jet",
    save_as: str = "pdf",
    show_plot: bool = True,
    font_size: float = 14,
    with_title: bool = True,
):
    """
    Plot property profiles over time using pcolormesh for the specified well

    :param well_name: Name of the well the properties of which will be plotted
    :type well_name: str
    :param coupled_model: An instance of DartsModel
    :type coupled_model: DartsModel
    :param max_ts_idx: If specified, the heat map will be shown until the specified maximum time step index. If not
    specified, the heat map will be shown for all the time steps.
    :type max_ts_idx: int
    :param x_axis: "simulated_time" or "time_step_index"
    :type x_axis: str
    :param y_axis: "segments_MD" or "segments_TVD" or "segment_index"
    :type y_axis: str
    :param save_as: The extension of the image files that will be saved
    :type save_as: str
    :param show_plot: Whether or not to show the plot
    :type show_plot: bool
    :param font_size: Size of the fonts
    :type font_size: float
    :param with_title: If you want the figure to have a title or not
    :type with_title: bool
    """
    axes, specs = prepare_heat_maps(
        well_name=well_name,
        coupled_model=coupled_model,
        output_folder_name=f'heat_maps_pcolormesh_{well_name}',
        max_ts_idx=max_ts_idx,
        x_axis=x_axis,
        y_axis=y_axis,
        cmap_color=cmap_color,
        save_as=save_as,
        show_plot=show_plot,
        font_size=font_size,
        with_title=with_title,
    )

    def draw(fig, ax, spec, matrix, y_values):
        """Draw one property as a pseudo-color mesh."""
        vmin, vmax = spec.vlim if spec.vlim is not None else (None, None)

        # Create the heatmap
        cax = ax.pcolormesh(
            axes.x,
            y_values,
            matrix,
            cmap=cmap_color,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )

        # Set the y-axis ticks
        if y_axis == "segment_index":
            ax.yaxis.set_major_locator(MultipleLocator(1))

        # Add a colorbar to show the property values
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(spec.mesh_cbar_label, fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        return True

    render_heat_maps(axes, specs, draw)
