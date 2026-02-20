import pyvista as pv
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_vtk_pyvista(output_dir, contour=False):
    '''
    Plot VTK results using PyVista.
    saves 2D plots - xz slice - of specified arrays (vertic displ and stress) from the last timestep.
    '''
    
    if 'sawcut' in output_dir or '2rocks' in output_dir: # contours help to see that u_z is the same along X-axes in the inclined hex mesh
        contour = True
    
    #filename = os.path.join(output_dir, 'vtk', 'solution.pvd')
    #output_dir_plots = os.path.join(os.path.dirname(os.path.dirname(filename)), 'plots')
    filename = os.path.join(output_dir, 'solution.pvd')
    output_dir_plots = os.path.join(os.path.dirname(filename), 'plots')

    # Get reader and check available timesteps
    reader = pv.get_reader(filename)
    days2sec = 86400
    t_steps = np.asarray(reader.time_values) * days2sec
    print("Available timesteps (sec):", t_steps[:5], '...', t_steps[-5:])

    # Load the last timestep
    reader.set_active_time_value(reader.time_values[-1])
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
    
    arr_name = "delta_temperature"; tensor = False; arr_name_plot = 'temperature,K';
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

    for plot_config in plot_config_list:
        arr_name, tensor, arr_name_plot, contour, component_index = plot_config

        if arr_name not in block.array_names: # skip temperature if not thermal model
            print('Warning: ', arr_name, 'not found in point data')
            continue
        block.set_active_scalars(None)
        if not tensor:
            block.set_active_scalars(arr_name, preference='point')
        else:
            block.set_active_tensors(arr_name, preference='point')

        # Create a slice
        slice_plane = block.slice(normal='y')
        #slice_plane = block  # no slice (plot in 3D)

        # Define the bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
        # Use large values for Y and Z if you only want to limit X
        #bounds = [-2000.0, 2000.0, -2000.0, 2000.0, 1000., 4000.]
        #slice_plane = slice_plane.clip_box(bounds=bounds, invert=False)

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

        plotter.add_mesh(slice_plane, show_edges=False)#, cmap=cmap, show_scalar_bar=True)

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
        plotter.view_xz()
        plotter.show_axes()
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
        
        # plot 
        plt.figure(figsize=(8, 4))
        plt.contourf(x_coords, z_coords, values_2d, levels=30, cmap='viridis')
        plt.colorbar()
        plt.xlabel("X Axis")
        plt.ylabel("Z Axis")
        plt.axis('equal')
        plt.savefig(os.path.join(output_dir_plots, arr_name_plot + "_contour.png"))
        plt.close()
        
        # plot 1D #################################################################################
        sample_resolution = 15
        # Define line endpoints (x, y fixed; z varies)
        points_xy = [[50, 50, 'center'], [500, 500, 'right']] # XY
        z1, z2 = 0.0, 5000.   # vertical extent
        for x0, y0, name in points_xy:
            p0 = (x0, y0, z1)
            p1 =  (x0, y0, z2)
            sampled = block.sample_over_line(pointa=p0, pointb=p1, resolution=sample_resolution)
            z = sampled.points[:, 2]      # vertical coordinate
            values = sampled.point_data[arr_name]
            plt.figure(figsize=(6,6))
            if len(values.shape) > 1:# a tensor
                for kk, label in enumerate(['XX', 'YY', 'ZZ', 'YZ', 'XZ', 'XY']):
                    plt.plot(values[:, kk], z, "-o", markersize=2, label=label)
                arr_name_plot = arr_name # without xx and ZZ as all components are plotted 
            else:
                plt.plot(values, z, "-o", markersize=2)
            #
            plt.xlabel(arr_name_plot)
            plt.ylabel("Height (z), m.")
            plt.title(f"Vertical profile of {arr_name_plot} at x={x0}, y={y0}")
            #
            plt.grid(True)
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth=0.8)
            plt.grid(which='minor', linestyle=':', linewidth=0.5) 
            #
            plt.tight_layout()
            plt.legend()
            plt.savefig(os.path.join(output_dir_plots, arr_name_plot + '_vertic_line_' + name + '.png'))
            plt.close()
        ##################################################################################

if __name__ == "__main__":
    contour = False
    output_dir = os.path.join('results', 'sol_cpp_single_phase_inj_34_34_57')
    plot_vtk_pyvista(output_dir, contour=contour)
