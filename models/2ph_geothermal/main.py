import numpy as np
import pandas as pd
import os
from datetime import datetime

from model import Model
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt
from darts.tools.vtk_io import write_lines_vtp
from darts.physics.base.operator_evaluator import PropertyOperators as props

def plot_sol(n):
    Xn = np.array(n.physics.engine.X, copy=False)
    nc = n.property_container.nc + n.thermal
    nb = n.reservoir.mesh.n_res_blocks
    P = Xn[0:nb * nc:nc]
    z = np.ones((nc, nb))
    phi = np.ones(nb)
    sat_ev = props(n.property_container)
    prop = np.zeros(2*n.property_container.nph)

    plt.figure(num=1, figsize=(12, 8), dpi=100)
    for i in range(nc-1):
        z[i][:] = Xn[i + 1:nb * nc:nc]
        z[-1][:] -= z[i][:]

    for i in range(nb):
        state = Xn[i*nc:(i+1)*nc]
        sat_ev.evaluate(state, prop)
        density_tot = np.sum(prop[0:3] * prop[3:6])
        phi[i] -= prop[2]  # (z[-1, i] * density_tot / prop[-1])

    for i in range(3):
        plt.subplot(330 + (i + 1))
        plt.plot(z[i]/(1-z[3]))
        plt.title('Composition' + str(i + 1), y=1)

    i = 3
    plt.subplot(330 + (i + 1))
    plt.plot(phi)
    plt.title('Porosity', y=1)

    i = 4
    plt.subplot(330 + (i + 1))
    plt.plot(P)
    plt.title('Pressure', y=1)

    plt.show()

if __name__ == '__main__':

    redirect_darts_output('run.log')


    n = Model()
    # n.params.linear_type = n.params.linear_solver_t.cpu_superlu
    n.init()
    output_folder = os.path.join(
        'output', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    print(f'Writing simulation output to {output_folder}')
    n.set_output(output_folder=output_folder)
    time_data_filename = n.output_folder + "/darts_time_data.pkl"

    if True:
        simulation_years = 60.0
        report_step = 365.25
        simulation_end_days = simulation_years * 365.25
        while n.physics.engine.t < simulation_end_days - 1e-8:
            step = min(report_step, simulation_end_days - n.physics.engine.t)
            n.set_well_controls()
            n.run(step, save_reservoir_data=True)
        # n.reservoir.wells[0].control = n.physics.new_bhp_inj(100, 3*[n.zero])
        # n.run_python(300, restart_dt=1e-3)
        n.print_timers()
        n.print_stat()

        # compute and save well time data
        time_data_dict = n.output.store_well_time_data(save_output_files=True)
        time_data_df = pd.DataFrame.from_dict(time_data_dict)
        time_data_df['time_years'] = time_data_df['time'] / 365.25
        bhp_plot_data_df = time_data_df.iloc[1:]
        time_figure, time_axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for row, well in enumerate(n.reservoir.wells):
            time_axes[row, 0].plot(
                bhp_plot_data_df['time_years'],
                bhp_plot_data_df[f'well_{well.name}_BHP'],
            )
            time_axes[row, 1].plot(
                time_data_df['time_years'],
                time_data_df[f'well_{well.name}_BHT'],
            )
            time_axes[row, 0].set_title(f'{well.name} bottom-hole pressure')
            time_axes[row, 1].set_title(f'{well.name} bottom-hole temperature')
        for axis in time_axes[:, 0]:
            axis.set_ylabel('Pressure [bar]')
        for axis in time_axes[:, 1]:
            axis.set_ylabel('Temperature [K]')
        for axis in time_axes[-1, :]:
            axis.set_xlabel('Time [years]')
        for axis in time_axes.flat:
            axis.grid(True, alpha=0.3)
        time_figure.tight_layout()
        for plot_name in ('out.png', 'borehole_pressure_temperature.png'):
            time_figure.savefig(
                os.path.join(n.output_folder, plot_name),
                dpi=150,
            )
        plt.close(time_figure)

    else:
        # n.load_restart_data()
        n.load_restart_data('output/solution.h5')
        time_data = pd.read_pickle(time_data_filename)


    if True:
        Xn = np.array(n.physics.engine.X, copy=False)
        nc = n.physics.nc + n.physics.thermal
        nb = n.reservoir.mesh.n_res_blocks

        pressure = Xn[0:nb * nc:nc]
        temperature = Xn[1:nb * nc:nc]
        nx, ny, nz = n.reservoir.nx, n.reservoir.ny, n.reservoir.nz
        global_to_local = np.asarray(n.reservoir.discretizer.global_to_local)
        pressure_global = np.full(nx * ny * nz, np.nan)
        active_global = np.flatnonzero(global_to_local >= 0)
        pressure_global[active_global] = pressure[global_to_local[active_global]]
        pressure_grid = pressure_global.reshape((nx, ny, nz), order='F')
        temperature_global = np.full(nx * ny * nz, np.nan)
        temperature_global[active_global] = temperature[global_to_local[active_global]]
        temperature_grid = temperature_global.reshape((nx, ny, nz), order='F')
        well_y_positions = [cell[1] - 1 for cells in n.well_paths.values() for cell in cells]
        y_slice = int(np.clip(round(np.mean(well_y_positions)), 0, ny - 1))
        pressure_map_layer = 33
        reservoir_layer = pressure_map_layer - 1
        pressure_plan = pressure_grid[:, :, reservoir_layer].T
        temperature_plan = temperature_grid[:, :, reservoir_layer].T
        pressure_change_plan = pressure_plan - 200.0
        temperature_change_plan = temperature_plan - 350.0
        pressure_section = pressure_grid[:, y_slice, :].T
        temperature_section = temperature_grid[:, y_slice, :].T
        pressure_change_section = pressure_section - 200.0
        temperature_difference_section = temperature_section - 350.0
        plan_well_markers = {
            well_name: (
                np.mean([cell[0] for cell in cells]) - 1,
                np.mean([cell[1] for cell in cells]) - 1,
            )
            for well_name, cells in n.well_paths.items()
            if cells
        }
        section_well_markers = {
            well_name: (
                np.mean([cell[0] for cell in cells]) - 1,
                np.mean([cell[2] for cell in cells]) - 1,
            )
            for well_name, cells in n.well_paths.items()
            if cells
        }

        pressure_limit = max(
            np.nanmax(np.abs(pressure_change_plan)),
            np.nanmax(np.abs(pressure_change_section)),
            1e-6,
        )
        temperature_limit = max(
            np.nanmax(np.abs(temperature_change_plan)),
            np.nanmax(np.abs(temperature_difference_section)),
            1e-6,
        )
        fig, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)
        pressure_plot = axes[0, 0].imshow(
            pressure_plan, origin='lower', aspect='auto'
        )
        axes[0, 0].set_title(f'Pressure at layer {pressure_map_layer} [bar]')
        axes[0, 0].set_xlabel('X cell')
        axes[0, 0].set_ylabel('Y cell')
        fig.colorbar(pressure_plot, ax=axes[0, 0], label='Pressure [bar]')
        pressure_change_plot = axes[0, 1].imshow(
            pressure_change_plan,
            origin='lower',
            aspect='auto',
            cmap='coolwarm',
            vmin=-pressure_limit,
            vmax=pressure_limit,
        )
        axes[0, 1].set_title(f'Pressure change at layer {pressure_map_layer} [bar]')
        axes[0, 1].set_xlabel('X cell')
        axes[0, 1].set_ylabel('Y cell')
        fig.colorbar(pressure_change_plot, ax=axes[0, 1], label='Change from 200 bar')
        temperature_plot = axes[0, 2].imshow(
            temperature_change_plan,
            origin='lower',
            aspect='auto',
            cmap='coolwarm',
            vmin=-temperature_limit,
            vmax=temperature_limit,
        )
        axes[0, 2].set_title(f'Temperature change at layer {pressure_map_layer} [K]')
        axes[0, 2].set_xlabel('X cell')
        axes[0, 2].set_ylabel('Y cell')
        fig.colorbar(temperature_plot, ax=axes[0, 2], label='Change from 350 K')

        section_images = (
            (pressure_section, 'Pressure'),
            (pressure_change_section, 'Pressure change'),
            (temperature_difference_section, 'Temperature change'),
        )
        for column, (section, title) in enumerate(section_images):
            image = axes[1, column].imshow(
                section,
                origin='upper',
                aspect='auto',
                cmap='coolwarm' if column else None,
                vmin=(-pressure_limit if column == 1 else -temperature_limit)
                if column
                else None,
                vmax=(pressure_limit if column == 1 else temperature_limit)
                if column
                else None,
            )
            axes[1, column].set_title(
                f'{title} at Y section {y_slice + 1}'
                + (' [K]' if column == 2 else ' [bar]' if column else ' [bar]')
            )
            axes[1, column].set_xlabel('X cell')
            axes[1, column].set_ylabel('Layer, top to bottom')
            fig.colorbar(image, ax=axes[1, column])

        for axis in axes[0, :]:
            for well_name, (x_cell, y_cell) in plan_well_markers.items():
                axis.plot(x_cell, y_cell, 'wo', markeredgecolor='black')
                axis.text(x_cell + 1, y_cell + 1, well_name, color='black', weight='bold')
        for axis in axes[1, :]:
            axis.axhline(n.burden_layers - 0.5, color='white', linestyle='--')
            axis.axhline(
                n.burden_layers + n.reservoir_nz - 0.5,
                color='white',
                linestyle='--',
            )
            for well_name, (x_cell, y_cell) in section_well_markers.items():
                axis.plot(x_cell, y_cell, 'wo', markeredgecolor='black')
                axis.text(
                    x_cell + 1,
                    y_cell + 1,
                    well_name,
                    color='black',
                    weight='bold',
                )
        fig.savefig(os.path.join(n.output_folder, 'pressure_map.png'), dpi=150)
        plt.close(fig)

        vtk_times, vtk_data = n.output.output_properties(n.sol_filepath)
        annual_targets = np.arange(1, int(simulation_years) + 1) * 365.25
        annual_indices = np.array(
            [np.argmin(np.abs(vtk_times - target)) for target in annual_targets],
            dtype=int,
        )
        annual_data = {
            name: np.asarray(values)[annual_indices]
            for name, values in vtk_data.items()
        }
        pressure_name = n.physics.vars[0]
        annual_data['pressure_change_x10'] = 10.0 * (
            annual_data[pressure_name] - 200.0
        )
        n.output.output_to_vtk(
            output_data=[vtk_times[annual_indices], annual_data]
        )

        vtk_dir = os.path.join(n.output_folder, 'vtk_files')
        os.makedirs(vtk_dir, exist_ok=True)
        for well_name, cells in n.well_paths.items():
            nodes = np.array([
                ((cell[0] - 0.5) * n.reservoir.global_data['dx'][cell[0] - 1, cell[1] - 1, cell[2] - 1],
                 (cell[1] - 0.5) * n.reservoir.global_data['dy'][cell[0] - 1, cell[1] - 1, cell[2] - 1],
                 -cell[2])
                for cell in cells
            ])
            if len(nodes) > 1:
                write_lines_vtp(
                    os.path.join(vtk_dir, f'{well_name}.vtp'),
                    nodes,
                )

    else:
        #plot_sol(n)
        n.print_and_plot('sim_data')

    # n.compare_well_rates(time_data_filename)

#z_c10 = Xn[nc-1:n.reservoir.nb*nc:nc]

# rho_aq = n.property_container.density_ev['wat'].evaluate(P, z_co2)
# Sg = np.zeros(n.reservoir.nb)
#
# for i in range (n.reservoir.nb):
#     x_list = Xn[i*nc:(i+1)*nc]
#     state = value_vector(x_list)
#     Sg[i] = n.properties(state)

# """ start plots """
# plt.figure(num=1, figsize=(12, 8), dpi=100)
# """ sg and x """
# plt.subplot(211)
# plt.plot(z1)
# #plt.imshow(np.reshape(T, (220, 60)).T)
#
# plt.title('First composition', y=1)
#
# plt.subplot(212)
# plt.plot(P)
# #plt.imshow(np.reshape(P, (220, 60)).T)
# plt.title('Pressure', y=1)

# """ sg and x """
# plt.subplot(223)
# plt.plot(P)
# plt.title('Pressure', y=1)
#
# plt.subplot(224)
# plt.plot(T)
# plt.title('Gas saturation', y=1)


# time_data1 = pd.DataFrame.from_dict(n.physics.engine.time_data)
# from darts.tools.plot_darts import *
# writer = pd.ExcelWriter('time_data.xlsx')
# plot_phase_rate_darts('I1', time_data1, 'wat')
#
#
# plt.show()

# from darts.tools.plot_darts import *
# time_data1 = pd.DataFrame.from_dict(n.physics.engine.time_data)
# for i, w in enumerate(n.reservoir.wells):
#     # plot oil rate
#     ax1 = plot_oil_rate_darts(w.name, time_data1, color='b')
#     ax1.tick_params(labelsize=14)
#     ax1.set_xlabel('Days', fontsize=14)
#
#     ax3 = plot_gas_rate_darts(w.name, time_data1, color='b')
#     ax3.tick_params(labelsize=14)
#     ax3.set_xlabel('Days', fontsize=14)
#
# plt.show()
