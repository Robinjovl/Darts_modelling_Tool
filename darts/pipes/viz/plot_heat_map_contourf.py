import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator

from darts.models.darts_model import DartsModel
from darts.pipes.viz._heat_map_common import (
    MASK_INVALID,
    get_contour_levels,
    prepare_heat_maps,
    render_heat_maps,
    value_range,
)


def plot_heat_map_contourf(
    well_name: str,
    coupled_model: DartsModel,
    min_ts_idx: int = 0,
    max_ts_idx: int = None,
    x_axis: str = "simulated_time",
    y_axis: str = "segments_MD",
    cmap_color: str = "jet",
    save_as: str = 'pdf',
    show_plot: bool = True,
    y_axis_tick_interval=50.0,
    n_cmap_bins_p: int = 10,
    n_cmap_bins_comp: int = 10,
    n_cmap_bins_t: int = 10,
    n_cmap_bins_s: int = 10,
    n_cmap_bins_rho: int = 10,
    n_cmap_bins_mu: int = 10,
    n_cmap_bins_v: int = 10,
    font_size: float = 14,
    with_title: bool = True,
    with_logarithmic_x_axis: bool = False,
):
    """
    Plot property profiles over time using contourf for the specified well

    :param well_name: Name of the well the properties of which will be plotted
    :type well_name: str
    :param coupled_model: An instance of DartsModel
    :type coupled_model: DartsModel
    :param min_ts_idx: If specified, the heat map will be shown from the specified minimum time step index. If not
    specified, the heat map will be shown from the zeroth time step.
    :type min_ts_idx: int
    :param max_ts_idx: If specified, the heat map will be shown until the specified maximum time step index. If not
    specified, the heat map will be shown till the last time step.
    :type max_ts_idx: int
    :param x_axis: "simulated_time" or "time_step_index"
    :type x_axis: str
    :param y_axis: "segments_MD" or "segments_TVD" or "segment_index"
    :param save_as: The extension of the image files that will be saved
    :type save_as: str
    :param show_plot: Whether or not to show the plot
    :type show_plot: bool
    :param y_axis_tick_interval: The interval of the ticks of the y axis.
    :type y_axis_tick_interval: float
    :param n_cmap_bins_p: Number of bins of the colorbar and colormap of pressure
    :type n_cmap_bins_p: int
    :param n_cmap_bins_comp: Number of bins of the colorbar and colormap of composition
    :type n_cmap_bins_comp: int
    :param n_cmap_bins_t: Number of bins of the colorbar and colormap of temperature
    :type n_cmap_bins_t: int
    :param n_cmap_bins_s: Number of bins of the colorbar and colormap of volume fraction
    :type n_cmap_bins_s: int
    :param n_cmap_bins_rho: Number of bins of the colorbar and colormap of density
    :type n_cmap_bins_rho: int
    :param n_cmap_bins_mu: Number of bins of the colorbar and colormap of viscosity
    :type n_cmap_bins_mu: int
    :param n_cmap_bins_v: Number of bins of the colorbar and colormap of velocity
    :type n_cmap_bins_v: int
    :param font_size: Size of the fonts
    :type font_size: float
    :param with_title: If you want the figure to have a title or not
    :type with_title: bool
    :param with_logarithmic_x_axis: Whether or not to have the logarithmic x-axis
    :type with_logarithmic_x_axis: bool
    """
    axes, specs = prepare_heat_maps(
        well_name=well_name,
        coupled_model=coupled_model,
        output_folder_name=f'heat_maps_contourf_{well_name}',
        min_ts_idx=min_ts_idx,
        max_ts_idx=max_ts_idx,
        x_axis=x_axis,
        y_axis=y_axis,
        cmap_color=cmap_color,
        save_as=save_as,
        show_plot=show_plot,
        font_size=font_size,
        with_title=with_title,
        n_cmap_bins={
            "p": n_cmap_bins_p,
            "comp": n_cmap_bins_comp,
            "t": n_cmap_bins_t,
            "s": n_cmap_bins_s,
            "rho": n_cmap_bins_rho,
            "mu": n_cmap_bins_mu,
            "v": n_cmap_bins_v,
        },
    )

    def draw(fig, ax, spec, matrix, y_values):
        """Draw one property as filled contours overlaid with contour lines."""
        if spec.mask == MASK_INVALID and matrix.count() == 0:
            # Nothing to contour: the property is undefined over the whole well
            return False

        # Create a discrete colorbar and colormap
        min_value, max_value = value_range(matrix, spec)
        levels = get_contour_levels(
            min_value, max_value, axes.n_cmap_bins[spec.n_levels_key]
        )
        cmap = plt.get_cmap(cmap_color, axes.n_cmap_bins[spec.n_cmap_key])
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Region-based fill
        cf = ax.contourf(
            axes.x,
            y_values,
            matrix,
            levels=levels,
            cmap=cmap,
            norm=norm,
            # extend='both'  # if you want arrows at the ends
        )

        # Overlay the exact same contour lines
        ax.contour(axes.x, y_values, matrix, levels=levels, colors='k', linewidths=0.7)

        # Create the colorbar
        cbar = fig.colorbar(
            cf,
            ax=ax,
            boundaries=levels,
            ticks=levels,
            spacing='proportional',
        )
        cbar.set_label(spec.cbar_label, fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)  # Set tick font size of the colorbar

        # Set the y-axis ticks
        ax.yaxis.set_major_locator(MultipleLocator(y_axis_tick_interval))

        # Make the x-axis logarithmic
        if with_logarithmic_x_axis and spec.log_x:
            ax.set_xscale('log')

        return True

    render_heat_maps(axes, specs, draw)
