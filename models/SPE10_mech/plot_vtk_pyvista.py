import pyvista as pv
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_vtk_pyvista(output_dir, contour=False, tstep_to_plot=-1):
    '''
    Plot VTK results using PyVista.
    saves 2D plots - xz slice - of specified arrays (vertic displ and stress) from the last timestep.
    '''
    
    if 'sawcut' in output_dir or '2rocks' in output_dir: # contours help to see that u_z is the same along X-axes in the inclined hex mesh
        contour = True
    
    #filename = os.path.join(output_dir, 'vtk', 'solution.pvd')
    #output_dir_plots = os.path.join(os.path.dirname(os.path.dirname(filename)), 'plots')
    filename = os.path.join(output_dir, 'solution.pvd')
    output_dir_plots = os.path.join(os.path.dirname(filename), 'plots_timestep_last')
    os.makedirs(output_dir_plots, exist_ok=True)

    # Get reader and check available timesteps
    reader = pv.get_reader(filename)
    days2sec = 86400
    t_steps = np.asarray(reader.time_values) * days2sec
    print("Available timesteps (sec):", t_steps[:5], '...', t_steps[-5:])

    # Load the data for the asked timestep
    reader.set_active_time_value(reader.time_values[tstep_to_plot])
    mesh = reader.read()

    # Check if mesh is MultiBlock
    if isinstance(mesh, pv.MultiBlock):
        print(f"MultiBlock with {len(mesh)} blocks")
        # Access the first block
        block = mesh[0]
    else:
        block = mesh

    print("Available cell data keys:", list(block.cell_data.keys()))
    print("Available point data keys:", list(block.point_data.keys()))

    # Define plot configurations: (array name, is tensor?, array name for plot (filename), add contour?)
    plot_config_list = []
    component_index = None

    arr_name = "ux"; tensor = False; arr_name_plot = 'u_x,m';
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index))
    
    arr_name = "uy"; tensor = False; arr_name_plot = 'u_y,m';
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index))

    arr_name = "uz"; tensor = False; arr_name_plot = 'u_z,m';
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index))
    
    arr_name = "temperature"; tensor = False; arr_name_plot = 'temperature,K';
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index))
    
    arr_name = "pressure"; tensor = False; arr_name_plot = 'pressure,bars'; 
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index))
    
    arr_name = "delta_temperature"; tensor = False; arr_name_plot = 'delta_temperature,K';
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index))
    
    arr_name = "delta_pressure"; tensor = False; arr_name_plot = 'delta_pressure,bars'; 
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index))

    arr_name = 'delta_eff_stress'; tensor = True; component_index = 2; arr_name_plot = 'delta_eff_stress_ZZ,bars';
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index))

    arr_name = 'delta_eff_stress'; tensor = True; component_index = 0; arr_name_plot = 'delta_eff_stress_XX,bars';
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index))

    arr_name = 'delta_eff_stress'; tensor = True; component_index = 1; arr_name_plot = 'delta_eff_stress_YY,bars';
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index))

    arr_name = 'delta_tot_stress'; tensor = True; component_index = 2; arr_name_plot = 'delta_tot_stress_ZZ,bars';
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index))

    arr_name = 'delta_tot_stress'; tensor = True; component_index = 0; arr_name_plot = 'delta_tot_stress_XX,bars';
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index))

    arr_name = 'delta_tot_stress'; tensor = True; component_index = 1; arr_name_plot = 'delta_tot_stress_YY,bars';
    plot_config_list.append((arr_name, tensor, arr_name_plot, contour, component_index))

    rsv_xy_plot_bnd = 5000. # m.

    for plot_config in plot_config_list:
        arr_name, tensor, arr_name_plot, contour, component_index = plot_config

        if arr_name not in block.array_names: # skip temperature if not thermal model
            print('Warning: ', arr_name, 'not found in point data')
            continue
        
        print('Plotting: ', arr_name)
        block.set_active_scalars(None)
        if not tensor:
            block.set_active_scalars(arr_name, preference='point')
        else:
            block.set_active_tensors(arr_name, preference='point')

        # Create a slice
        slice_plane = block.slice(normal='y')
        #slice_plane = block  # no slice (plot in 3D)
        y_bnd = max(abs(block.bounds[2]), abs(block.bounds[3]))
        slice_plane = slice_plane.clip_box(
            bounds=[-rsv_xy_plot_bnd, rsv_xy_plot_bnd, -y_bnd, y_bnd,
                    block.bounds[4], block.bounds[5]], invert=False)

        if tensor:
            slice_plane[arr_name_plot] = slice_plane[arr_name][:,component_index]
        else:
            slice_plane[arr_name_plot] = slice_plane[arr_name]

        plotter = pv.Plotter(off_screen=True) # save without showing the GUI window

        # to plot with the same color if values are almost the same everywhere
        plot_rel_diff_threshold = 0.001
        values = np.array(slice_plane[arr_name_plot])
        min_val = values.min()
        max_val = values.max()
        rel_diff = np.fabs(max_val - min_val) #/ max(np.fabs(min_val), np.fabs(max_val)) 
        if rel_diff < plot_rel_diff_threshold:
            slice_plane[arr_name_plot][:] = min_val

        #n_levels = 100
        #cmap = plt.get_cmap("viridis", n_levels)

        plotter.add_mesh(slice_plane, show_edges=False,
                         scalar_bar_args={'vertical': True, 'position_y': 0.25, 'height': 0.5,
                                          'title': ''})

        if contour:
            slice_plane_points = slice_plane.cell_data_to_point_data()
            # Generate contour surfaces
            contours = slice_plane_points.contour(isosurfaces=20, scalars=arr_name_plot)
            # Plot result
            if contours.n_cells > 0:  # skip empty plots
                plotter.add_mesh(contours, cmap="viridis", opacity=1, color="black")
                plotter.add_mesh(slice_plane_points.outline(), color="black")

        #arrows = slice_plane.glyph(orient="stress_vec", factor=0.05)
        #plotter.add_mesh(arrows, color="black")
        xmin_blk = -rsv_xy_plot_bnd 
        xmax_blk = rsv_xy_plot_bnd 
        #ymin_blk = -rsv_xy_plot_bnd
        #ymax_blk = rsv_xy_plot_bnd 
        zmin_blk = block.bounds[4] 
        zmax_blk = block.bounds[5]
        y_slice = block.center[1]
        # horizontal reference lines at z=2000 and z=2400
        for z_ref in [2000., 2400.]:
            plotter.add_mesh(pv.Line(pointa=(xmin_blk, y_slice, z_ref),
                                     pointb=(xmax_blk, y_slice, z_ref)),
                             color='white', line_width=0.5)
        # vertical lines for injection (x=550) and production (x=-550) wells
        for x_well, z2, z1, clr, lw in [(550., 2400., -0., 'cyan', 2), \
                                    (-550., 2400., -0., 'red', 2), \
                                    (250., 5000., 0., 'black', 1)]:
            plotter.add_mesh(pv.Line(pointa=(x_well, y_slice, z1),
                                     pointb=(x_well, y_slice, z2)),
                             color=clr, line_width=lw)
        plotter.view_xz()
        plotter.camera.up = (0, 0, -1)   # invert Z axis
        plotter.camera.position = (plotter.camera.position[0],
                                   plotter.camera.position[1] * -1,
                                   plotter.camera.position[2])  # invert X axis
        #plotter.reset_camera(bounds=[xmin_blk, xmax_blk, ymin_blk, ymax_blk, zmin_blk, zmax_blk])
        plotter.camera.zoom(1.1)
        #n_xlabels = int(round((xmax_blk - xmin_blk) / 1000.)) + 1
        plotter.show_bounds(grid=False, location='outer', ticks='outside',
                            xtitle='X, m.', ytitle='', ztitle='Z, m.',
                            show_yaxis=False, n_xlabels=9, n_zlabels=6,
                            font_size=12, fmt='%d')
        #plotter.add_text(arr_name_plot, position='upper_edge', font_size=8)  # title
        plotter.show(screenshot=os.path.join(output_dir_plots, arr_name_plot + "_slice.png"))
        plotter.close()
        
        # Contour plot using matplotlib #################################################
        # Define desired resolution and bounds
        res_x, res_z = 1000, 1000
        xmin, xmax = slice_plane.points[:, 0].min(), slice_plane.points[:, 0].max()
        zmin, zmax = slice_plane.points[:, 2].min(), slice_plane.points[:, 2].max()
        y_mean = slice_plane.points[:, 1].mean()
        # Create a "Template" Grid (UniformGrid / ImageData) structured XY plane (at Y=0)
        grid = pv.ImageData(
            dimensions=(res_x, 1, res_z),
            spacing=((xmax - xmin)/(res_x-1), y_mean, (zmax - zmin)/(res_z-1)),
            origin=(xmin, y_mean, zmin)
        )
        
        # Sample the data from your original 'block' or 'slice'
        # interpolate values from the slice onto struct grid
        structured_resample = grid.sample(slice_plane)
        
        # reshape to 2D
        if tensor:
            values_2d = structured_resample[arr_name][:,component_index].reshape(res_x, res_z)
        else:
            values_2d = structured_resample[arr_name].reshape(res_x, res_z)
        
        # Get the X and Z coordinates as 2D arrays (matching the values)
        x_coords = structured_resample.points[:, 0].reshape(res_x, res_z)
        z_coords = structured_resample.points[:, 2].reshape(res_x, res_z)
        
        # plot contours (don't look nice, so commented)
        if False:
            plt.figure(figsize=(8, 4))
            plt.contourf(x_coords, z_coords, values_2d, levels=30, cmap='viridis')
            plt.colorbar()
            for z_ref in [2000., 2400.]:
                plt.axhline(y=z_ref, color='white', linestyle='--', linewidth=1)
            for x_well, clr, lw in [(550., 'cyan', 1), (-550., 'red', 1), (250., 'black', 1)]:
                plt.axvline(x=x_well, color=clr, linestyle='--', linewidth=lw)
            plt.xlabel("X Axis")
            plt.ylabel("Depth, m.")
            plt.ylim(zmin_blk, zmax_blk)
            plt.gca().invert_yaxis()
            plot_suffix = "_contour.png"
            plt.savefig(os.path.join(output_dir_plots, arr_name_plot + plot_suffix))
            plt.close()
        
        # plot 1D #################################################################################
        sample_resolution = 15  # number of point along Z for plotting
        # select a few evenly-spaced timestep indices (always include the 1-th and the last)
        n_t = len(reader.time_values)
        n_t_plot = 5
        t_indices_1d = sorted(set(
            [1] + list(np.linspace(0, n_t - 1, n_t_plot, dtype=int)) + [n_t - 1]))
        
        if 'stress' in arr_name:
            t_indices_1d = [tstep_to_plot]
        
        # Define line endpoints (x, y fixed; z varies)
        points_xy = [[50, 50, 'center'], [500, 500, 'right']] # XY
        z1, z2 = 0.0, 5000.   # vertical extent
        for x0, y0, name in points_xy:
            p0 = (x0, y0, z1)
            p1 = (x0, y0, z2)
            plt.figure(figsize=(6, 6))
            for t_idx in t_indices_1d:
                reader.set_active_time_value(reader.time_values[t_idx])
                mesh_t = reader.read()
                block_t = mesh_t[0] if isinstance(mesh_t, pv.MultiBlock) else mesh_t
                sampled = block_t.sample_over_line(pointa=p0, pointb=p1, resolution=sample_resolution)
                z = sampled.points[:, 2]
                values = sampled.point_data[arr_name]
                t_days = reader.time_values[t_idx]
                t_label = f't={t_days:.0f} d'
                if len(values.shape) > 1:  # tensor: plot all components, label by component+time
                    for kk, comp in enumerate(['XX', 'YY', 'ZZ', 'YZ', 'XZ', 'XY']):
                        plt.plot(values[:, kk], z, "-o", markersize=2, label=f'{comp} {t_label}')
                else:
                    plt.plot(values, z, "-o", markersize=2, label=t_label)
            arr_name_plot_1d = arr_name if len(values.shape) > 1 else arr_name_plot
            plt.xlabel(arr_name_plot_1d)
            plt.ylabel("Depth, m.")
            plt.ylim(zmin_blk, zmax_blk)
            plt.gca().invert_yaxis()
            plt.title(f"Vertical profile of {arr_name_plot_1d} at x={x0}, y={y0}")
            plt.grid(True)
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth=0.8)
            plt.grid(which='minor', linestyle=':', linewidth=0.5)
            plt.tight_layout()
            plt.legend(fontsize=7)
            plt.savefig(os.path.join(output_dir_plots, arr_name_plot_1d + '_vertic_line_' + name + '.png'))
            plt.close()
    print('Plotting from VTK is completed for', output_dir)
        ##################################################################################

if __name__ == "__main__":
    contour = False
    
    output_dir = os.path.join('results', 'sol_cpp_single_phase_inj_16_16_15')
    #output_dir = os.path.join('results', 'sol_cpp_single_phase_doublet_16_16_15')
    #output_dir = os.path.join('results', 'sol_cpp_single_phase_thermal_doublet_16_16_15')
    
    #output_dir = os.path.join('results', 'sol_cpp_single_phase_inj_34_34_57')
    #output_dir = os.path.join('results', 'sol_cpp_single_phase_thermal_inj_34_34_57')
    
    #output_dir = os.path.join('results', 'sol_cpp_single_phase_inj_34_34_66')
    #output_dir = os.path.join('results', 'sol_cpp_single_phase_thermal_doublet_34_34_66')
    
    #output_dir = os.path.join('results', 'sol_cpp_single_phase_thermal_doublet_42_42_90')
    #output_dir = r'\\wsl.localhost\Ubuntu-24.04\root\projects\open-darts_dev_debug\models\SPE10_mech\results\sol_cpp_single_phase_inj_42_42_66'
    
    plot_vtk_pyvista(output_dir, contour=contour)
