import os
import numpy as np
import meshio

# add path to import
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)  # 1 level up
balmatt_dir = os.path.join(os.path.join(parent_dir, 'vito'), 'darts_model_balmatt')
sys.path.insert(0, balmatt_dir)

def calc_frac_aper_by_stress(input_data, rotation=0):
    '''
    rotation - ? - stress angle, degrees
    save to .npy files
    #return the array of final apertures for all fractures
    '''
    fname = '_' + input_data['mesh_prefix'] + '_' + str(input_data['char_len']) + '.msh'
    mesh_file = os.path.join('meshes_' + input_data['case_name'], input_data['case_name'] + fname)

    initial_frac_aper = input_data['frac_aper']
    sigma_c = 100  # [MPa] typically for Dinantian rock 70-250 MPa (Entela Kane)

    #function to get an angle and output apertures
    mesh_data = meshio.read(mesh_file) #read the mesh file
    num_frac = mesh_data.cells[0].data.shape[0] #number of fractures
    act_frac_sys = np.zeros((num_frac, 4)) #create an array to store the fracture system
    for ii in range(num_frac): #loop over the fractures
        ith_line = mesh_data.points[mesh_data.cells[0].data[ii][:2]]
        act_frac_sys[ii, :2] = ith_line[0, :2]
        act_frac_sys[ii, 2:] = ith_line[1, :2]

    def calc_aperture(sigma_n, sigma_c, e_0_in=None): #function to calculate the aperture
        #BARTON-BANDIS MODEL parameters (all in Stephans paper)
        JRC = 7.225
        JCS = 17.5
        if e_0_in is None:
            e_0 = JRC * (0.2 * sigma_c / JCS - 0.1) / 5.
        else:
            e_0 = e_0_in
        # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JB012657
        v_m = -0.1032 - 0.0074 * JRC + 1.135 * (JCS / e_0) ** -0.251
        K_ni = -7.15 + 1.75 * JRC + 0.02 * JCS / e_0
        aper = e_0 - 1. / (1 / v_m + K_ni / sigma_n)
        return aper # return the final aperture in mm.

    def get_normal_stress_on_fault(x0, y0, x1, y1, frac_angle):
        stress_tensor_orig = np.zeros((2,2))
        stress_tensor_orig[0, 0] = input_data['Sh_max']
        stress_tensor_orig[1, 1] = input_data['Sh_min']

        # rotate principal stress tensor to fracture coordinate system (SHmax || frac)
        a = (input_data['SHmax_azimuth'] - frac_angle)/ 180 * np.pi
        rot_1 = np.array([[np.cos(a), np.sin(a)],[-np.sin(a), np.cos(a)]])
        rot_2 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        stress_tensor = rot_1 @ stress_tensor_orig @ rot_2

        stress_n = stress_tensor[0,0] # Sh_min new is normal to the fault

        # normal vector to vertical fault
        #ux, uy, uz = u = [x1 - x0, y1 - y0, 0]  # first vector
        #vx, vy, vz = v = [x2 - x0, y2 - y0, 0]  # sec vector
        #cross = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]  # cross product
        #fault_normal = np.array(cross)
        #fault_normal /= np.linalg.norm(self.normal)

        #stress_n = stress_tensor @ fault_normal @ fault_normal
        #stress_t = np.linalg.norm(
        #    (np.eye(3, 3) - np.tensordot(fault_normal, fault_normal_T, axes=0))
        #    @ stress_tensor @ fault_normal)
        return stress_n #, stress_t]

    epsilon = 1e-4 #small number to avoid division by zero
    dx = act_frac_sys[:, 0] - act_frac_sys[:, 2] + epsilon * np.random.rand(num_frac) #calculate the x and y components of the fracture
    dy = act_frac_sys[:, 1] - act_frac_sys[:, 3] + epsilon * np.random.rand(num_frac)
    #rotation = 0
    frac_angles = np.arctan(dy / dx) * 180 / np.pi + rotation + epsilon * np.random.rand(num_frac) #calculate the angle of the fracture
    #sigma_n = (sigma_H + sigma_h) / 2 + (sigma_H + sigma_h) / 2 * np.cos(angles * np.pi / 180 * 2) #calculate the normal stress
    #sigma_n = (input_data['Sh_max'] + input_data['Sh_min']) / 2 * (1 + np.cos(frac_angles * 2))
    sigma_n = []
    for fi in range(frac_angles.size):
        sigma_n.append(get_normal_stress_on_fault(act_frac_sys[:, 0], act_frac_sys[:, 1], act_frac_sys[:, 2], act_frac_sys[:, 3], frac_angles[fi]))
    sigma_n = np.array(sigma_n)
    factor_aper = 1 #factor to increase the aperture, if needed to compensate for unknowns in the model
    fracture_aper = calc_aperture(sigma_n, sigma_c) * 1e-3 * factor_aper #calculate the aperture, convert to [m]
    fracture_aper[fracture_aper < 1e-6] = 1e-6 #set the minimum aperture
    fracture_aper[fracture_aper > 1e-2] = 1e-2 #set the maximum aperture

    fracture_aper = np.array(fracture_aper)

    np.save('frac_tips.npy', act_frac_sys, allow_pickle=True)
    np.save('frac_aper.npy', [frac_angles, sigma_n, fracture_aper], allow_pickle=True)

    return fracture_aper  #return the array of final apertures for all fractures

def plot_frac_aper():
    frac_data_raw = np.load('frac_tips.npy', allow_pickle=True)
    [frac_angles, sigma_n, fracture_aper] = np.load('frac_aper.npy', allow_pickle=True)
    #frac_angles += 90
    # 2D geometry plot (wells and fractures)
    import matplotlib.pyplot as plt
    from matplotlib import cm
    plt.gca().set_aspect('equal')
    aper_max = fracture_aper.max()
    fracture_aper_mean = fracture_aper.mean()
    for i in range(frac_data_raw.shape[0]):
        clr='0.7' # more closed
        if fracture_aper[i] > fracture_aper_mean: # more opened
            clr='m'
        plt.plot(np.append(frac_data_raw[i, 0], frac_data_raw[i, 2]),
                 np.append(frac_data_raw[i, 1], frac_data_raw[i, 3]),
                 color=clr)
                 #c=cm.hot(fracture_aper[i]/aper_max))

    wells_inj = input_data['inj_well_coords']
    for i in range(len(wells_inj)):
        plt.plot(wells_inj[i][0], wells_inj[i][1], 'o', color='b', label='inj well'+str(i))
    wells_prod = input_data['prod_well_coords']
    for i in range(len(wells_prod)):
        plt.plot(wells_prod[i][0], wells_prod[i][1], 'o', color='r', label='prod well'+str(i))

    print('frac_angles', 'min=', frac_angles.min(), 'max=', frac_angles.max())
    print('sigma_n', 'min=', sigma_n.min(), 'max=', sigma_n.max())
    print('fracture_aper,mm', 'min=', fracture_aper.min()*1e+3, 'max=', fracture_aper.max()*1e+3)

    frac_tips_x = np.append(frac_data_raw[:, 0], frac_data_raw[:, 2])
    frac_tips_y = np.append(frac_data_raw[:, 1], frac_data_raw[:, 3])

    # plot the principal stress direction
    x1 = frac_tips_x.min()
    y1 = frac_tips_y.min()
    x2 = frac_tips_x.max()
    y2 = frac_tips_y.max()
    #dx = x2 - x1
    dy = y2 - y1
    alpha = input_data['SHmax_azimuth']
    # to avoid tan(alpha) = inf
    if np.abs(alpha - 0) > 1 and np.abs(alpha - 180) > 1:
        dx = dy / np.tan(np.radians(alpha))
    else:
        dx = 0.
    xc = (x1 + x2) * 0.5
    yc = (y1 + y2) * 0.5
    scale = 0.2
    plt.plot(np.append(xc - dx * scale, xc + dx * scale), np.append(yc - dy * scale, yc + dy * scale), '--', color='b', label='SHmax_azimuth')

    plt.xlabel('X, m.')
    plt.ylabel('Y, m.')
    #plt.legend()
    plt.savefig('fracture_aper_2d.png',dpi=600)

    # plot fracture apertures vs fracture angles
    plt.figure()
    frac_angles_plot = frac_angles  # to the same system as initial stress angle
    frac_angles_plot[frac_angles_plot < alpha] += 180  # for better plot (to have max in the middle)
    plt.scatter(frac_angles_plot, fracture_aper*1000) # to mm.
    plt.xlabel('frac angle, degrees')
    plt.ylabel('frac aperture, mm')
    plt.savefig('fracture_aper_scatter.png', dpi=600)

if __name__ == '__main__':
    #from examples.case_3 import *
    from balmatt import *
    frac_apertures = calc_frac_aper_by_stress(input_data)
    plot_frac_aper()
