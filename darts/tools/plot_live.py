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
        fig, axes = plt.subplots(2, 8, figsize=(22, 7), constrained_layout=True)

        # Well props
        ax0 = axes[0, 0]
        ax1 = axes[0, 1]
        ax2 = axes[0, 2]
        ax3 = axes[0, 3]
        ax4 = axes[0, 4]
        ax5 = axes[0, 5]
        ax6 = axes[0, 6]
        ax7 = axes[0, 7]
        # Reservoir props
        ax8 = axes[1, 0]
        ax9 = axes[1, 1]
        ax10 = axes[1, 2]
        ax11 = axes[1, 3]
        ax12 = axes[1, 4]
        ax13 = axes[1, 5]
        ax14 = axes[1, 6]
        ax15 = axes[1, 7]

        # Axes for wellbore pressure
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

        ax0.set_xlabel("Pressure [bar]")
        ax0.set_ylabel("Segment index [-]")
        ax0.set_title("** Pressure **")
        ax0.invert_yaxis()

        # Axes for wellbore temperature
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

        ax1.set_xlabel(r"Temperature [$^\circ$C]")
        ax1.set_ylabel("Segment index [-]")
        ax1.set_title("** Temperature **")
        ax1.invert_yaxis()

        # Axes for wellbore gas volume fraction
        (line2,) = ax2.plot(
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

        ax2.set_xlabel("Gas volume fraction [-]")
        ax2.set_ylabel("Segment index [-]")
        ax2.set_title("** Gas volume fraction **")
        ax2.invert_yaxis()

        # Axes for wellbore liquid volume fraction
        (line3,) = ax3.plot(
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

        ax3.set_xlabel("Liquid volume fraction [-]")
        ax3.set_ylabel("Segment index [-]")
        ax3.set_title("** Liquid volume fraction **")
        ax3.invert_yaxis()

        # Axes for wellbore gas density
        (line4,) = ax4.plot(
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

        ax4.set_xlabel(r"Gas density [kg/m$^3$]")
        ax4.set_ylabel("Segment index [-]")
        ax4.set_title("** Gas density **")
        ax4.invert_yaxis()

        # Axes for wellbore liquid density
        (line5,) = ax5.plot(
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

        ax5.set_xlabel(r"Liquid density [kg/m$^3$]")
        ax5.set_ylabel("Segment index [-]")
        ax5.set_title("** Liquid density **")
        ax5.invert_yaxis()

        # Axes for wellbore gas viscosity
        (line6,) = ax6.plot(
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

        ax6.set_xlabel("Gas viscosity [cP]")
        ax6.set_ylabel("Segment index [-]")
        ax6.set_title("** Gas viscosity **")
        ax6.invert_yaxis()

        # Axes for wellbore liquid viscosity
        (line7,) = ax7.plot(
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

        ax7.set_xlabel("Liquid viscosity [cP]")
        ax7.set_ylabel("Segment index [-]")
        ax7.set_title("** Liquid viscosity **")
        ax7.invert_yaxis()

        # Axes for reservoir pressure
        (line8,) = ax8.plot(
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

        ax8.set_xscale("log")
        ax8.set_xlabel("Reservoir radial distance [m]")
        ax8.set_ylabel("Pressure [bar]")

        # Axes for reservoir temperature
        (line9,) = ax9.plot(
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

        ax9.set_xscale("log")
        ax9.set_xlabel("Reservoir radial distance [m]")
        ax9.set_ylabel("Temperature [$^\circ$C]")

        # Axes for reservoir gas volume fraction
        (line10,) = ax10.plot(
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

        ax10.set_xscale("log")
        ax10.set_xlabel("Reservoir radial distance [m]")
        ax10.set_ylabel("Gas volume fraction [-]")

        # Axes for reservoir liquid volume fraction
        (line11,) = ax11.plot(
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

        ax11.set_xscale("log")
        ax11.set_xlabel("Reservoir radial distance [m]")
        ax11.set_ylabel("Liquid volume fraction [-]")

        # Axes for reservoir gas density
        (line12,) = ax12.plot(
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

        ax12.set_xscale("log")
        ax12.set_xlabel("Reservoir radial distance [m]")
        ax12.set_ylabel(r"Gas density [kg/m$^3$]")

        # Axes for reservoir liquid density
        (line13,) = ax13.plot(
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

        ax13.set_xscale("log")
        ax13.set_xlabel("Reservoir radial distance [m]")
        ax13.set_ylabel(r"Liquid density [kg/m$^3$]")

        # Axes for reservoir gas viscosity
        (line14,) = ax14.plot(
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

        ax14.set_xscale("log")
        ax14.set_xlabel("Reservoir radial distance [m]")
        ax14.set_ylabel("Gas viscosity [cP]")

        # Axes for reservoir liquid viscosity
        (line15,) = ax15.plot(
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

        ax15.set_xscale("log")
        ax15.set_xlabel("Reservoir radial distance [m]")
        ax15.set_ylabel("Liquid viscosity [cP]")

        fig.show()

        lines = [
            line0,
            line1,
            line2,
            line3,
            line4,
            line5,
            line6,
            line7,
            line8,
            line9,
            line10,
            line11,
            line12,
            line13,
            line14,
            line15,
        ]

        live_plot_store["well_fig"] = {
            "fig": fig,
            "axes": axes,
            "lines": lines,
        }
    """ Stop initializing the figure containing axes for profiles of wellbore and 1D reservoir properties """

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
        lines[0].set_data(p_well, np.arange(n_segments))
        axes[0, 0].relim()
        axes[0, 0].autoscale_view()

        lines[1].set_data(T_well, np.arange(n_segments))
        axes[0, 1].relim()
        axes[0, 1].autoscale_view()

        lines[2].set_data(sG_well, np.arange(n_segments))
        axes[0, 2].relim()
        axes[0, 2].autoscale_view()

        lines[3].set_data(1 - sG_well, np.arange(n_segments))
        axes[0, 3].relim()
        axes[0, 3].autoscale_view()

        lines[4].set_data(rhoG_well, np.arange(n_segments))
        axes[0, 4].relim()
        axes[0, 4].autoscale_view()

        lines[5].set_data(rhoL_well, np.arange(n_segments))
        axes[0, 5].relim()
        axes[0, 5].autoscale_view()

        lines[6].set_data(miuG_well, np.arange(n_segments))
        axes[0, 6].relim()
        axes[0, 6].autoscale_view()

        lines[7].set_data(miuL_well, np.arange(n_segments))
        axes[0, 7].relim()
        axes[0, 7].autoscale_view()

        # Reservoir props
        lines[8].set_data(x_res, p_res)
        axes[1, 0].relim()
        axes[1, 0].autoscale_view()

        lines[9].set_data(x_res, T_res)
        axes[1, 1].relim()
        axes[1, 1].autoscale_view()

        lines[10].set_data(x_res, sG_res)
        axes[1, 2].relim()
        axes[1, 2].autoscale_view()

        lines[11].set_data(x_res, 1 - sG_res)
        axes[1, 3].relim()
        axes[1, 3].autoscale_view()

        lines[12].set_data(x_res, rhoG_res)
        axes[1, 4].relim()
        axes[1, 4].autoscale_view()

        lines[13].set_data(x_res, rhoL_res)
        axes[1, 5].relim()
        axes[1, 5].autoscale_view()

        lines[14].set_data(x_res, miuG_res)
        axes[1, 6].relim()
        axes[1, 6].autoscale_view()

        lines[15].set_data(x_res, miuL_res)
        axes[1, 7].relim()
        axes[1, 7].autoscale_view()

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
