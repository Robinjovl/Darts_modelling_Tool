import matplotlib.pyplot as plt
import numpy as np


def init_live_plots(
    live_plot_store: dict,
    physics,
    reservoir,
    wells=None,
):
    # At the moment, the function is supported for the PH formulation with a single component.
    if (
        not (physics.state_spec == physics.StateSpecification.PH)
        or not physics.n_vars == 2
    ):
        raise Exception(
            "Plotting the live PH diagram is supported for the PH formulation and a single component!"
        )

    """
    Initialize live plots
    """
    plt.ion()

    """ Start initializing the figure containing axes for the properties of the Newton solver """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)

    ax0 = axes[0]
    ax1 = axes[1]

    # Axes for number of Newton iterations
    (line0,) = ax0.plot(
        [],
        [],
        linestyle='-',
        linewidth=2,
        marker='o',
        markersize=6,
        color='red',
        markerfacecolor='red',
        markeredgecolor='red',
    )
    # ax0.set_xscale("log")
    ax0.set_xlabel("Time [days]")
    ax0.set_ylabel("Number of Newton iterations [-]")

    # Axes for time step size
    (line1,) = ax1.plot(
        [],
        [],
        linestyle='-',
        linewidth=2,
        marker='o',
        markersize=6,
        color='red',
        markerfacecolor='red',
        markeredgecolor='red',
    )
    # ax1.set_xscale("log")
    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("Time step size [days]")

    fig.show()

    live_plot_store["solver_fig"] = {
        "fig": fig,
        "axes": axes,
        "lines": [line0, line1],
    }
    """ End initializing the figure containing axes for the properties of the Newton solver """

    """ Start initializing the figure containing a pair of axes for the PH diagram """
    fig, axes = plt.subplots(figsize=(10, 6), constrained_layout=True)

    p_idx = physics.vars.index("pressure")
    h_idx = physics.vars.index("enthalpy")

    # Get the bounds of the OBL domain
    p_bounds = (physics.PT_axes_min[p_idx], physics.PT_axes_max[p_idx])
    h_bounds = (physics.axes_min[h_idx], physics.axes_max[h_idx])

    # Resolution of the PH diagram
    n_p, n_h = physics.n_axes_points[p_idx], physics.n_axes_points[h_idx]

    p_range = np.linspace(p_bounds[0], p_bounds[1], n_p)
    h_range = np.linspace(h_bounds[0], h_bounds[1], n_h)

    # Calculate the property matrix
    prop_matrix = np.empty((n_p, n_h))
    for idx_p, p in enumerate(p_range):
        for idx_h, h in enumerate(h_range):
            state_ph = [p, h]
            physics.property_containers[0].evaluate(state_ph)
            prop_matrix[idx_p, idx_h] = physics.property_containers[0].temperature

    n_cmap_bins = 50
    levels = np.linspace(np.nanmin(prop_matrix), np.nanmax(prop_matrix), n_cmap_bins)

    # Filled contour (colored areas)
    cax = axes.contourf(h_range, p_range, prop_matrix, levels=levels, cmap='jet')

    # Contour lines at the same levels
    contours = axes.contour(
        h_range, p_range, prop_matrix, levels=levels, colors='black', linewidths=0.5
    )

    # Label each contour line with its property value
    axes.clabel(contours, fmt='%1.1f', inline=True, fontsize=7)

    axes.set_xlim(h_bounds[0], h_bounds[1])
    axes.set_ylim(p_bounds[0], p_bounds[1])
    axes.set_xlabel('Specific enthalpy [kJ/kmole]')
    axes.set_ylabel('Pressure [bar]')

    # Add a colorbar
    cbar = fig.colorbar(cax, ax=axes)
    cbar.set_label("Temperature [K]")

    # Create a plot for adding bottom-hole points (states) to the PH diagram
    (line,) = axes.plot(
        [],
        [],
        linestyle='-',
        linewidth=2,
        marker='o',
        markersize=6,
        color='red',
        markerfacecolor='red',
        markeredgecolor='red',
        label='Bottom-hole state',
    )

    fig.show()

    # Update the figure store
    live_plot_store["ph_fig"] = {
        "fig": fig,
        "axes": axes,
        "lines": [line],
    }
    """ Stop initializing the figure containing a pair of axes for the PH diagram """

    return live_plot_store


def update_live_plots(
    live_plot_store: dict,
    physics,
    reservoir,
    time: float,
    iter_counter: int,
    time_step_size: float,
    wells=None,
):
    """
    Update live plots

    :param time: Current time [days]
    :type time: float
    :param iter_counter: Newton-Raphson iteration counter
    :type iter_counter: int
    """
    """ Start updating the figure containing axes for the properties of the Newton solver """
    fig = live_plot_store["solver_fig"]["fig"]
    axes = live_plot_store["solver_fig"]["axes"]
    lines = live_plot_store["solver_fig"]["lines"]

    # lines[0].set_data(self.time, self.n_newton_iters)
    # axes[0].relim()
    # axes[0].autoscale_view()

    # lines[1].set_data(self.time, self.time_step_size)
    # axes[1].relim()
    # axes[1].autoscale_view()

    # Update iteration counter vs time
    x = list(lines[0].get_xdata())
    y = list(lines[0].get_ydata())
    x.append(time)
    y.append(iter_counter)

    lines[0].set_data(x, y)
    axes[0].relim()
    axes[0].autoscale_view()

    # Update time-step size vs time
    x = list(lines[1].get_xdata())
    y = list(lines[1].get_ydata())
    x.append(time)
    y.append(time_step_size)

    lines[1].set_data(x, y)
    axes[1].relim()
    axes[1].autoscale_view()

    # Refresh display
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    """ End updating the figure containing axes for the properties of the Newton solver """

    """ Start updating the figure containing a pair of axes for the PH diagram """
    fig = live_plot_store["ph_fig"]["fig"]
    axes = live_plot_store["ph_fig"]["axes"]
    lines = live_plot_store["ph_fig"]["lines"]

    # Update state of the desired block on the PH diagram
    assert 0 <= live_plot_store["tracked_block_idx"] < reservoir.mesh.n_blocks, (
        "The specified tracked_block_idx is out of range!"
    )
    block_idx = live_plot_store["tracked_block_idx"]
    X_np = np.asarray(physics.engine.X).reshape(-1, physics.n_vars)
    p_idx = physics.vars.index("pressure")
    h_idx = physics.vars.index("enthalpy")
    p = X_np[block_idx, p_idx]
    h = X_np[block_idx, h_idx]

    x = list(lines[0].get_xdata())
    y = list(lines[0].get_ydata())
    x.append(h)
    y.append(p)

    lines[0].set_data(x, y)

    axes.relim()
    axes.autoscale_view()

    # Update the figure title
    fig.suptitle(
        live_plot_store["title_template"].format(time=time, iter_counter=iter_counter)
    )

    # Refresh display
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    """ Stop updating the figure containing a pair of axes for the PH diagram """

    return live_plot_store
