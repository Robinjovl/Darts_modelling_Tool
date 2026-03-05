import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def init_live_plots(
    live_plot_store: dict,
    physics,
    reservoir,
    has_dfm_well: bool,
    wells: dict = None,
):
    """
    Initialize the live plot

    :param live_plot_store: A store to save figures and parameters needed for live plotting
    :type live_plot_store: dict
    :param physics: Physics object
    :param reservoir: Reservoir object
    :param has_dfm_well: Whether or not the reservoir has a DFM well
    :type has_dfm_well: bool
    :param wells: Dictionary of well objects if DFM wells exist
    :type wells: dict
    """
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
    # Use integers for the labels of the y-axis
    ax0.yaxis.set_major_locator(MaxNLocator(integer=True))

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
    # PH diagram is only supported for the PH formulation (at the moment with a single component).
    if physics.state_spec == physics.StateSpecification.PH:
        if not physics.n_vars == 2:
            raise Exception(
                "Plotting the live PH diagram is supported for the PH formulation and a single component!"
            )

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
        levels = np.linspace(
            np.nanmin(prop_matrix), np.nanmax(prop_matrix), n_cmap_bins
        )

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

    """ Start initializing the figure containing axes for profiles of wellbore and 1D reservoir properties """
    if has_dfm_well and physics.state_spec == physics.StateSpecification.PH:
        n_columns = 8
        fig, axes = plt.subplots(2, n_columns, figsize=(22, 7), constrained_layout=True)

        # Well props (on 0th row)
        well_axes = [axes[0, i] for i in range(n_columns)]
        well_lines = []
        # Reservoir props (on 1st row)
        res_axes = [axes[1, i] for i in range(n_columns)]
        res_lines = []

        # Axes for wellbore pressure
        (line,) = well_axes[0].plot(
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
        well_lines.append(line)

        well_axes[0].set_xlabel("Pressure [bar]")
        well_axes[0].set_ylabel("Segment index [-]")
        well_axes[0].set_title("** Pressure **")
        well_axes[0].invert_yaxis()

        # Axes for wellbore temperature
        (line,) = well_axes[1].plot(
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
        well_lines.append(line)

        well_axes[1].set_xlabel(r"Temperature [$^\circ$C]")
        well_axes[1].set_ylabel("Segment index [-]")
        well_axes[1].set_title("** Temperature **")
        well_axes[1].invert_yaxis()

        # Axes for wellbore gas volume fraction
        (line,) = well_axes[2].plot(
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
        well_lines.append(line)

        well_axes[2].set_xlabel("Gas volume fraction [-]")
        well_axes[2].set_ylabel("Segment index [-]")
        well_axes[2].set_title("** Gas volume fraction **")
        well_axes[2].invert_yaxis()

        # Axes for wellbore liquid volume fraction
        (line,) = well_axes[3].plot(
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
        well_lines.append(line)

        well_axes[3].set_xlabel("Liquid volume fraction [-]")
        well_axes[3].set_ylabel("Segment index [-]")
        well_axes[3].set_title("** Liquid volume fraction **")
        well_axes[3].invert_yaxis()

        # Axes for wellbore gas density
        (line,) = well_axes[4].plot(
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
        well_lines.append(line)

        well_axes[4].set_xlabel(r"Gas density [kg/m$^3$]")
        well_axes[4].set_ylabel("Segment index [-]")
        well_axes[4].set_title("** Gas density **")
        well_axes[4].invert_yaxis()

        # Axes for wellbore liquid density
        (line,) = well_axes[5].plot(
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
        well_lines.append(line)

        well_axes[5].set_xlabel(r"Liquid density [kg/m$^3$]")
        well_axes[5].set_ylabel("Segment index [-]")
        well_axes[5].set_title("** Liquid density **")
        well_axes[5].invert_yaxis()

        # Axes for wellbore gas viscosity
        (line,) = well_axes[6].plot(
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
        well_lines.append(line)

        well_axes[6].set_xlabel("Gas viscosity [cP]")
        well_axes[6].set_ylabel("Segment index [-]")
        well_axes[6].set_title("** Gas viscosity **")
        well_axes[6].invert_yaxis()

        # Axes for wellbore liquid viscosity
        (line,) = well_axes[7].plot(
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
        well_lines.append(line)

        well_axes[7].set_xlabel("Liquid viscosity [cP]")
        well_axes[7].set_ylabel("Segment index [-]")
        well_axes[7].set_title("** Liquid viscosity **")
        well_axes[7].invert_yaxis()

        # Axes for reservoir pressure
        (line,) = res_axes[0].plot(
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
        res_lines.append(line)

        res_axes[0].set_xscale("log")
        res_axes[0].set_xlabel("Reservoir radial distance [m]")
        res_axes[0].set_ylabel("Pressure [bar]")

        # Axes for reservoir temperature
        (line,) = res_axes[1].plot(
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
        res_lines.append(line)

        res_axes[1].set_xscale("log")
        res_axes[1].set_xlabel("Reservoir radial distance [m]")
        res_axes[1].set_ylabel("Temperature [$^\circ$C]")

        # Axes for reservoir gas volume fraction
        (line,) = res_axes[2].plot(
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
        res_lines.append(line)

        res_axes[2].set_xscale("log")
        res_axes[2].set_xlabel("Reservoir radial distance [m]")
        res_axes[2].set_ylabel("Gas volume fraction [-]")

        # Axes for reservoir liquid volume fraction
        (line,) = res_axes[3].plot(
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
        res_lines.append(line)

        res_axes[3].set_xscale("log")
        res_axes[3].set_xlabel("Reservoir radial distance [m]")
        res_axes[3].set_ylabel("Liquid volume fraction [-]")

        # Axes for reservoir gas density
        (line,) = res_axes[4].plot(
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
        res_lines.append(line)

        res_axes[4].set_xscale("log")
        res_axes[4].set_xlabel("Reservoir radial distance [m]")
        res_axes[4].set_ylabel(r"Gas density [kg/m$^3$]")

        # Axes for reservoir liquid density
        (line,) = res_axes[5].plot(
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
        res_lines.append(line)

        res_axes[5].set_xscale("log")
        res_axes[5].set_xlabel("Reservoir radial distance [m]")
        res_axes[5].set_ylabel(r"Liquid density [kg/m$^3$]")

        # Axes for reservoir gas viscosity
        (line,) = res_axes[6].plot(
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
        res_lines.append(line)

        res_axes[6].set_xscale("log")
        res_axes[6].set_xlabel("Reservoir radial distance [m]")
        res_axes[6].set_ylabel("Gas viscosity [cP]")

        # Axes for reservoir liquid viscosity
        (line,) = res_axes[7].plot(
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
        res_lines.append(line)

        res_axes[7].set_xscale("log")
        res_axes[7].set_xlabel("Reservoir radial distance [m]")
        res_axes[7].set_ylabel("Liquid viscosity [cP]")

        fig.show()

        live_plot_store["well_fig"] = {
            "fig": fig,
            "axes": axes,
            "lines": well_lines + res_lines,
        }
    """ Stop initializing the figure containing axes for profiles of wellbore and 1D reservoir properties """

    # Template of the figure title
    live_plot_store["title_template"] = (
        "Time: {time:.4e} \nNR iteration counter: {iter_counter}"
    )

    return live_plot_store


def update_live_plots(
    live_plot_store: dict,
    physics,
    reservoir,
    time: float,
    iter_counter: int,
    time_step_size: float,
    has_dfm_well: bool,
    wells: dict = None,
):
    """
    Update the live plot

    :param live_plot_store: A store to save figures and parameters needed for live plotting
    :type live_plot_store: dict
    :param physics: Physics object
    :param reservoir: Reservoir object
    :param time: Current time [day]
    :type time: float
    :param iter_counter: Newton-Raphson iteration counter [-]
    :type iter_counter: int
    :param time_step_size: Size of the current time step [day]
    :type time_step_size: float
    :param has_dfm_well: Whether or not the reservoir has a DFM well
    :type has_dfm_well: bool
    :param wells: Dictionary of well objects if DFM wells exist
    :type wells: dict
    """
    """ Start updating the figure containing axes for the properties of the Newton solver """
    fig = live_plot_store["solver_fig"]["fig"]
    axes = live_plot_store["solver_fig"]["axes"]
    lines = live_plot_store["solver_fig"]["lines"]

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
    if physics.state_spec == physics.StateSpecification.PH:
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
            live_plot_store["title_template"].format(
                time=time, iter_counter=iter_counter
            )
        )

        # Refresh display
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    """ Stop updating the figure containing a pair of axes for the PH diagram """

    """ Start updating the figure containing axes for profiles of wellbore and 1D reservoir properties """
    if (
        has_dfm_well
        and reservoir.ndims == 1
        and physics.state_spec == physics.StateSpecification.PH
    ):
        fig = live_plot_store["well_fig"]["fig"]
        axes = live_plot_store["well_fig"]["axes"]
        lines = live_plot_store["well_fig"]["lines"]

        i_start_well = reservoir.wells[0].well_head_idx
        i_end_well = reservoir.wells[0].well_bottom_idx
        p_idx = physics.vars.index('pressure')
        h_idx = physics.vars.index('enthalpy')
        X_np = np.asarray(physics.engine.X).reshape(-1, physics.n_vars)
        p_well = X_np[i_start_well : i_end_well + 1, p_idx]
        _h_well = X_np[i_start_well : i_end_well + 1, h_idx]

        # Get the property container to evaluate phase props
        pc = physics.property_containers[0]
        n_segments = wells['I1'].geometry.num_segments
        n_res_blocks = reservoir.mesh.n_res_blocks
        n_vars = physics.n_vars
        T_well = np.zeros(n_segments)
        for i in range(n_res_blocks, n_res_blocks + n_segments):
            state = np.asarray(physics.engine.X)[i * n_vars : (i + 1) * n_vars]
            pc.evaluate(state)
            if physics.thermal:
                pc.evaluate_thermal(state)
            T_well[i - n_res_blocks] = pc.temperature - 273.15

        till_this_res_cell = live_plot_store[
            "plot_till_this_res_cell"
        ]  # Plot till this reservoir cell index
        assert till_this_res_cell <= n_res_blocks
        p_res = X_np[:i_start_well, p_idx][:till_this_res_cell]
        x_res = reservoir.global_data['dx'].reshape(-1)[:till_this_res_cell]

        # Calculate reservoir phase props
        nc = physics.nc
        # Preallocate phase props arrays
        T_res = np.zeros(till_this_res_cell)
        sG_res = np.zeros(till_this_res_cell)
        rhoG_res = np.zeros(till_this_res_cell)
        rhoL_res = np.zeros(till_this_res_cell)
        miuG_res = np.zeros(till_this_res_cell)
        miuL_res = np.zeros(till_this_res_cell)
        xG_mass_res = np.zeros((till_this_res_cell, nc))
        xL_mass_res = np.zeros((till_this_res_cell, nc))

        for i in range(till_this_res_cell):
            state = np.asarray(physics.engine.X)[i * n_vars : (i + 1) * n_vars]
            pc.evaluate(state)
            if physics.thermal:
                pc.evaluate_thermal(state)
            T_res[i] = pc.temperature - 273.15
            sG_res[i] = pc.sat[0]
            rhoG_res[i] = pc.dens[0]
            rhoL_res[i] = pc.dens[1]
            miuG_res[i] = pc.mu[0]
            miuL_res[i] = pc.mu[1]
            x_mass0 = np.zeros((pc.nph, nc))
            for j in pc.ph:
                x_mass0[j, :] = (pc.x[j, :] * pc.Mw) / sum(pc.x[j, :] * pc.Mw)
            xG_mass_res[i, :], xL_mass_res[i, :] = x_mass0[0, :], x_mass0[1, :]

        [
            xG_mass_well,
            xL_mass_well,
            sG_well,
            rhoG_well,
            rhoL_well,
            miuG_well,
            miuL_well,
        ] = wells['I1'].iter_phases_props

        # Well props
        n_columns = 8
        well_profiles = [
            p_well,
            T_well,
            sG_well,
            1 - sG_well,
            rhoG_well,
            rhoL_well,
            miuG_well,
            miuL_well,
        ]
        for i in range(n_columns):
            lines[i].set_data(well_profiles[i], np.arange(n_segments))
            axes[0, i].relim()
            axes[0, i].autoscale_view()

        # Reservoir props
        res_profiles = [
            p_res,
            T_res,
            sG_res,
            1 - sG_res,
            rhoG_res,
            rhoL_res,
            miuG_res,
            miuL_res,
        ]
        for i in range(n_columns):
            lines[i + n_columns].set_data(x_res, res_profiles[i])
            axes[1, i].relim()
            axes[1, i].autoscale_view()

        # Update the figure title
        fig.suptitle(
            live_plot_store["title_template"].format(
                time=time, iter_counter=iter_counter
            )
        )

        # Refresh display
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    """ Stop updating the figure containing axes for profiles of wellbore and 1D reservoir properties """

    return live_plot_store
