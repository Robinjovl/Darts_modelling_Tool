import numpy as np
import os
from main import read_vtk
from scipy.interpolate import griddata as gd
from matplotlib import pyplot as plt

vtk_directory = 'sol_mixed_well_slip_weakening'
prop_names = ['strain', 'v_y', 'strain_rate']


output_directory = os.path.join(vtk_directory, 'plots')
os.makedirs(output_directory, exist_ok=True)

def add_tstep_range_plot(tstep_plot_dict, name : str, timestep_start : int, timestep_end : int, timestep_stride : int):
    tstep_plot = np.arange(timestep_start, timestep_end, timestep_stride, dtype=np.int32)
    tstep_plot_dict[name] = tstep_plot

tstep_plot_dict = dict()

#add_tstep_range_plot(tstep_plot_dict, name='all_step10', timestep_start = 1, timestep_end = 1500, timestep_stride = 10)
#add_tstep_range_plot(tstep_plot_dict, name='all_step1', timestep_start = 1, timestep_end = 1500, timestep_stride = 1)
#add_tstep_range_plot(tstep_plot_dict, name='before_slip', timestep_start = 1,   timestep_end = 370,  timestep_stride = 1)
add_tstep_range_plot(tstep_plot_dict, name='during_slip', timestep_start = 350, timestep_end = 1000,  timestep_stride = 1)
#add_tstep_range_plot(tstep_plot_dict, name='after_slip',  timestep_start = 700, timestep_end = 1500, timestep_stride = 1)
#add_tstep_range_plot(tstep_plot_dict, name='after_slip2',  timestep_start = 1300, timestep_end = 1500, timestep_stride = 1)

def plot_contour(fname_prefix, array_dict, output_folder, points_x=None, points_y=None, layer=0):
    # plot contours XY plane, 1 layer by z
    for arr_name, arr in array_dict.items():
        if len(arr.shape) == 3:
            arr_layer = arr[:, :, layer]
        else:
            arr_layer = arr
        if points_x is None or points_y is None:
            cs = plt.contourf(arr_layer, levels=10)
        else:
            if False:
                cs = plt.contourf(points_x, points_y, arr_layer, levels=10)
            else:
                point_size = 0.0001
                x1, y1 = np.meshgrid(points_x, points_y)
                cs = plt.scatter(x1, y1, c=arr_layer, s=point_size, cmap='plasma')
        plt.colorbar(cs)
        #plt.gca().set_aspect('equal')
        #plt.tight_layout()
        plt.xlabel('Timestep index')
        plt.ylabel('Coordinate(m)')
        plt.title(arr_name)
        plt.savefig(os.path.join(output_folder, fname_prefix + arr_name + '.png'))
        plt.close()

def plot_strain(tstep_plot_name, tstep_plot, plot_1d=False):
    fname_prefix = tstep_plot_name + '_'

    if True: # set to False if run for unfinished simulation, where there is no .pvd (it stored in the end of simulation)
        import pyvista as pv
        pvd_filename = os.path.join(vtk_directory, 'solution.pvd')
        reader = pv.get_reader(pvd_filename)
        t_steps = np.asarray(reader.time_values)[tstep_plot]
        print("Timesteps for plotting:", t_steps[:5], '...', t_steps[-5:])

    vtk_files = []
    for ti in tstep_plot:
        vtk_files.append(os.path.join(vtk_directory, 'solution' + str(ti) + '.vtu'))

    # Define points along the well and at the surface
    y_range_well = np.arange(2000., -2000., -0.2)
    n_points_well = y_range_well.size
    points_well = np.zeros((n_points_well, 3))
    points_well[:,0] = 0.
    points_well[:,1] = y_range_well
    points_well[:,2] = 250.

    x_range_surface = np.arange(-2000., 2000., 100.)
    n_points_surface = x_range_surface.size
    points_surface = np.zeros((n_points_surface, 3))
    points_surface[:,0] = x_range_surface
    points_surface[:,1] = 2200. # 50m depth from surface
    points_surface[:,2] = 250.

    strain_yy_well_list = []
    strain_rate_yy_well_list = []
    strain_xx_surface_list = []
    v_y_well_list = []
    for k, filename in enumerate(vtk_files):
        centroids, props, __, __ = read_vtk(filename=filename, props=prop_names)

        # Voight notation
        strain_xx = props['strain'][0][:, 0]
        strain_yy = props['strain'][0][:, 1]
        #strain_zz = props['strain'][0][:, 2]

        if 'strain_rate' in props.keys():
            strain_rate_xx = props['strain_rate'][0][:, 0]
            strain_rate_yy = props['strain_rate'][0][:, 1]
        else:
            strain_rate_xx = np.zeros_like(strain_xx)
            strain_rate_yy = np.zeros_like(strain_yy)

        if 'v_y' not in props.keys():
            v_y = np.zeros_like(strain_yy)
        else:
            v_y = props['v_y'][0]

        # interpolate the solution (arr) from cell centers to given set of points
        strain_yy_well = gd((centroids[:, 0], centroids[:, 1], centroids[:, 2]), \
                                         strain_yy,
                            (points_well[:,0], points_well[:,1], points_well[:,2]),
                            method='nearest')

        strain_rate_yy_well = gd((centroids[:, 0], centroids[:, 1], centroids[:, 2]), \
                                         strain_rate_yy,
                            (points_well[:,0], points_well[:,1], points_well[:,2]),
                            method='nearest')

        v_y_well = gd((centroids[:, 0], centroids[:, 1], centroids[:, 2]), \
                                         v_y,
                            (points_well[:,0], points_well[:,1], points_well[:,2]),
                            method='nearest')

        strain_xx_surface = gd((centroids[:, 0], centroids[:, 1], centroids[:, 2]), \
                                         strain_xx,
                            (points_surface[:,0], points_surface[:,1], points_surface[:,2]),
                            method='nearest')

        strain_yy_well_list.append(strain_yy_well)
        strain_rate_yy_well_list.append(strain_rate_yy_well)
        strain_xx_surface_list.append(strain_xx_surface)

        v_y_well_list.append(v_y_well)

    if plot_1d:
        # Plot strain_yy along the well
        for k, strain_yy_well in enumerate(strain_yy_well_list):
            plt.plot(y_range_well, strain_yy_well, label='tstep_'+str(tstep_plot[k]))#, marker='.')
        plt.xlabel('Y-coordinate, m.')
        plt.title('Strain YY change along the well')
        plt.ylabel('Strain YY change')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_directory, fname_prefix + 'strain_yy_well.png'))
        #plt.show()
        plt.close()

        # Plot strain_yy along the well
        for k, strain_rate_yy_well in enumerate(strain_rate_yy_well_list):
            plt.plot(y_range_well, strain_rate_yy_well, label='tstep_'+str(tstep_plot[k]))#, marker='.')
        plt.xlabel('Y-coordinate, m.')
        plt.title('Strain rate YY change along the well')
        plt.ylabel('Strain rate YY change')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_directory, fname_prefix + 'strain_rate_yy_well.png'))
        #plt.show()
        plt.close()

        # Plot strain_xx along the surface
        for k, strain_xx_surface in enumerate(strain_xx_surface_list):
            plt.plot(x_range_surface, strain_xx_surface, label='tstep_'+str(tstep_plot[k]))#, marker='.')
        plt.xlabel('X-coordinate, m.')
        plt.title('Strain XX change along the line at the surface')
        plt.ylabel('Strain XX change')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_directory,  fname_prefix + 'strain_xx_surface.png'))
        #plt.show()
        plt.close()

        # Plot strain_yy along the well
        for k, v_y_well in enumerate(v_y_well_list):
            plt.plot(y_range_well, v_y_well, label='tstep_'+str(tstep_plot[k]))#, marker='.')
        plt.xlabel('Y-coordinate, m.')
        plt.title('Velocity Y along the well')
        plt.ylabel('Velocity Y')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_directory,  fname_prefix + 'velocity_y_well.png'))
        #plt.show()
        plt.close()

    ############ contour
    if False:
        # surface
        n_timesteps = len(strain_xx_surface_list)
        n_points = strain_xx_surface_list[0].size
        strain_2d = np.zeros((n_points, n_timesteps))
        for k in range(n_timesteps):
            strain_2d[:, k] = strain_xx_surface_list[k]
        array_dict = {'Strain_XX_change_surface_contour': strain_2d}
        plot_contour(fname_prefix, array_dict, points_x=tstep_plot, points_y=x_range_surface, output_folder=output_directory)
    # well - strain change
    n_timesteps = len(strain_yy_well_list)
    n_points = strain_yy_well_list[0].size
    strain_2d = np.zeros((n_points, n_timesteps))
    for k in range(n_timesteps):
        strain_2d[:, k] = strain_yy_well_list[k]
    array_dict = {'Strain_YY_change_well_contour': strain_2d}
    plot_contour(fname_prefix, array_dict, points_x=tstep_plot, points_y=y_range_well, output_folder=output_directory)
    # well rsv part - strain change
    y_range_well_rsv = (-350 < y_range_well) & (y_range_well < 350)
    array_dict = {'Strain_YY_change_well_contour_rsv': strain_2d[y_range_well_rsv, :]}
    plot_contour(fname_prefix, array_dict, points_x=tstep_plot, points_y=y_range_well[y_range_well_rsv], output_folder=output_directory)
    # well - strain rate
    n_timesteps = len(strain_rate_yy_well_list)
    n_points = strain_rate_yy_well_list[0].size
    strain_2d = np.zeros((n_points, n_timesteps))
    for k in range(n_timesteps):
        strain_2d[:, k] = strain_rate_yy_well_list[k]
    array_dict = {'Strain_rate_YY_change_well_contour': strain_2d}
    plot_contour(fname_prefix, array_dict, points_x=tstep_plot, points_y=y_range_well, output_folder=output_directory)
    # well rsv part - strain rate change
    array_dict = {'Strain_rate_YY_change_well_contour_rsv': strain_2d[y_range_well_rsv, :]}
    plot_contour(fname_prefix, array_dict, points_x=tstep_plot, points_y=y_range_well[y_range_well_rsv], output_folder=output_directory)

    # well - velocity
    if False:
        n_timesteps = len(v_y_well_list)
        n_points = v_y_well_list[0].size
        strain_2d = np.zeros((n_points, n_timesteps))
        for k in range(n_timesteps):
            strain_2d[:, k] = v_y_well_list[k]
        array_dict = {'Velocity_Y_well_contour': strain_2d}
        plot_contour(fname_prefix, array_dict, points_x=tstep_plot, points_y=y_range_well, output_folder=output_directory)

    ############# fault
    if False:
        vtk_files_fault = []
        for ti in tstep_plot:
            vtk_files_fault.append(os.path.join(vtk_directory, 'solution_fault' + str(ti) + '.vtu'))

        slip_max_list = []
        tstep_list = []
        for k, filename in enumerate(vtk_files_fault):
            centroids, props, __, __ = read_vtk(filename=filename, props=['f_local'])
            slip_y = props['f_local'][0][:, 1] # Y
            slip_max_list.append(slip_y.max())
            tstep_list.append(tstep_plot[k])

        # Plot slip max over time
        plt.plot(tstep_list, slip_max_list)
        plt.xlabel('Timestep index')
        plt.title('Max. slip (Y), mm.')
        plt.ylabel('Slip')
        plt.grid()
        plt.savefig(os.path.join(output_directory,  fname_prefix + 'fault_slip.png'))
        #plt.show()
        plt.close()


if __name__ == '__main__':
    if True: # 1d and 2d plots
        for tstep_plot_name in tstep_plot_dict.keys():
            plot_strain(tstep_plot_name, tstep_plot_dict[tstep_plot_name])
    else:
        # save animated video
        labels = ['DARTS']
        from main import plot_profiles
        plot_profiles(data_folder=output_directory, labels=labels, analytics=None, animate=True)
