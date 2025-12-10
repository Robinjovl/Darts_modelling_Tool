import numpy as np
import os
from main import read_vtk
from scipy.interpolate import griddata as gd
from matplotlib import pyplot as plt

output_directory = 'sol_mixed_well_slip_weakening'

#
timestep_start = 1
timestep_end = 1500
timestep_stride = 100

# before the fault slip
#timestep_start = 1
#timestep_end = 370
#timestep_stride = 30

# during the fault slip
#timestep_start = 350
#timestep_end = 700
#timestep_stride = 50

# after the fault slip
#timestep_start = 700
#timestep_end = 1500
#timestep_stride = 100

tstep_plot = np.arange(timestep_start, timestep_end, timestep_stride)

def plot_contour(array_dict, output_folder, points_x=None, points_y=None, layer=0):
    # plot contours XY plane, 1 layer by z
    for arr_name, arr in array_dict.items():
        if len(arr.shape) == 3:
            arr_layer = arr[:, :, layer]
        else:
            arr_layer = arr
        if points_x is None or points_y is None:
            cs = plt.contourf(arr_layer, levels=10)
        else:
            cs = plt.contourf(points_x, points_y, arr_layer, levels=10)
        plt.colorbar(cs)
        #plt.gca().set_aspect('equal')
        #plt.tight_layout()
        plt.xlabel('Timestep index')
        plt.ylabel('Coordinate(m)')
        plt.title(arr_name)
        plt.savefig(os.path.join(output_folder, arr_name + '.png'))
        plt.close()

def plot_strain():
    vtk_files = []
    for ti in tstep_plot:
        vtk_files.append(os.path.join(output_directory, 'solution' + str(ti) + '.vtu'))

    # Define points along the well and at the surface
    y_range_well = np.arange(2000., -2000., -100.)
    n_points = y_range_well.size
    points_well = np.zeros((n_points, 3))
    points_well[:,0] = 500.
    points_well[:,1] = y_range_well
    points_well[:,2] = 250.

    x_range_surface = np.arange(-2000., 2000., 100.)
    points_surface = np.zeros((n_points, 3))
    points_surface[:,0] = x_range_surface
    points_surface[:,1] = 2200. # 50m depth from surface
    points_surface[:,2] = 250.

    strain_yy_well_list = []
    strain_xx_surface_list = []
    for k, filename in enumerate(vtk_files):
        centroids, props, __, __ = read_vtk(filename=filename, props=['strain'])

        strain_xx = props['strain'][0][:, 0]
        strain_yy = props['strain'][0][:, 1]
        #strain_zz = props['strain'][0][:, 2]

        # interpolate the solution (arr) from cell centers to given set of points
        strain_yy_well = gd((centroids[:, 0], centroids[:, 1], centroids[:, 2]), \
                                         strain_yy,
                            (points_well[:,0], points_well[:,1], points_well[:,2]),
                            method='nearest')

        strain_xx_surface = gd((centroids[:, 0], centroids[:, 1], centroids[:, 2]), \
                                         strain_xx,
                            (points_surface[:,0], points_surface[:,1], points_surface[:,2]),
                            method='nearest')

        strain_yy_well_list.append(strain_yy_well)
        strain_xx_surface_list.append(strain_xx_surface)

    # Plot strain_yy along the well
    for k, strain_yy_well in enumerate(strain_yy_well_list):
        plt.plot(y_range_well, strain_yy_well, label='tstep_'+str(tstep_plot[k]))#, marker='.')
    plt.xlabel('Y-coordinate, m.')
    plt.title('Strain YY change along the well')
    plt.ylabel('Strain YY change')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_directory, 'strain_yy_well.png'))
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
    plt.savefig(os.path.join(output_directory, 'strain_xx_surface.png'))
    #plt.show()
    plt.close()

    ############ contour
    # surface
    n_timesteps = len(strain_xx_surface_list)
    n_points = strain_xx_surface_list[0].size
    strain_2d = np.zeros((n_points, n_timesteps))
    for k in range(n_timesteps):
        strain_2d[:, k] = strain_xx_surface_list[k]
    array_dict = {'Strain_XX_change_surface_contour': strain_2d}
    plot_contour(array_dict, points_x=tstep_plot, points_y=x_range_surface, output_folder=output_directory)
    # well
    n_timesteps = len(strain_yy_well_list)
    n_points = strain_yy_well_list[0].size
    strain_2d = np.zeros((n_points, n_timesteps))
    for k in range(n_timesteps):
        strain_2d[:, k] = strain_yy_well_list[k]
    array_dict = {'Strain_ZZ_change_well_contour': strain_2d}
    plot_contour(array_dict, points_x=tstep_plot, points_y=y_range_well, output_folder=output_directory)
    # well rsv part
    y_range_well_rsv = (-350 < y_range_well) & (y_range_well < 350)
    array_dict = {'Strain_ZZ_change_well_contour_rsv': strain_2d[y_range_well_rsv, :]}
    plot_contour(array_dict, points_x=tstep_plot, points_y=y_range_well[y_range_well_rsv], output_folder=output_directory)

    ############# fault
    if False:
        vtk_files_fault = []
        for ti in tstep_plot:
            vtk_files_fault.append(os.path.join(output_directory, 'solution_fault' + str(ti) + '.vtu'))

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
        plt.savefig(os.path.join(output_directory, 'fault_slip.png'))
        #plt.show()
        plt.close()


if __name__ == '__main__':
    if True:
        plot_strain()
    else:
        labels = ['DARTS']
        from main import plot_profiles
        plot_profiles(data_folder=output_directory, labels=labels, analytics=None, animate=True)
