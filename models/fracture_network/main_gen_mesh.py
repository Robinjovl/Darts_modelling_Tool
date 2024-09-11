import numpy as np
from multiprocessing import freeze_support
from darts.tools.fracture_network.preprocessing_code import frac_preprocessing
import os
from datetime import datetime
import shutil

def rotate_input(input_data, frac_data_raw):
    rot_angle_degrees = 90 - input_data['SHmax_azimuth']
    rot_angle = np.radians(rot_angle_degrees)
    rot_matrix = [[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]]
    def rotate_point(x, y, rot_matrix):
        x_new = x * rot_matrix[0][0] + y * rot_matrix[0][1]
        y_new = x * rot_matrix[1][0] + y * rot_matrix[1][1]
        return [x_new, y_new]

    # rotate fracture points
    frac_data_raw_out = []
    for f_tips in frac_data_raw:
        x1, y1, x2, y2 = f_tips
        frac_data_raw_out.append(rotate_point(x1, y1, rot_matrix) + rotate_point(x2, y2, rot_matrix))

    # rotate well coordinates
    prod_wells_out = []
    for pw in input_data['prod_well_coords']:
        prod_wells_out.append(rotate_point(pw[0], pw[1], rot_matrix) + [pw[2]])
    inj_wells_out = []
    for iw in input_data['inj_well_coords']:
        inj_wells_out.append(rotate_point(iw[0], iw[1], rot_matrix) + [iw[2]])

    #TODO rotate input_data['box_data'] ?

    # update input_data
    frac_data_raw[:] = frac_data_raw_out
    input_data['prod_well_coords'] = prod_wells_out
    input_data['inj_well_coords'] = inj_wells_out

def generate_mesh(input_data):
    print('case', input_data['case_name'])
    output_dir = 'meshes_' + input_data['case_name']
    if not 'balmatt' in input_data['case_name']: # simple test case
        frac_data_raw = np.genfromtxt(input_data['frac_file'])
    else:
        f = open(input_data['frac_file'], 'r')
        data = f.readlines()
        f.close()

        fault_coords = []
        for line in data[1:]:
            s = line.split()
            fault_coords.append([float(s[1]), float(s[2]), float(s[3]), float(s[4])])

        frac_data_raw = np.array(fault_coords)

    # rename output dir if exists
    if os.path.exists(output_dir):
        ren_fname = output_dir + '_prev'
        if os.path.exists(ren_fname):
            shutil.rmtree(ren_fname)
        os.renames(output_dir, ren_fname)
    os.makedirs(output_dir)

    #rotate_input(input_data, frac_data_raw)

    # 2D geometry plot (wells and fractures)
    import matplotlib.pyplot as plt
    plt.gca().set_aspect('equal')
    for i in range(frac_data_raw.shape[0]):
        plt.plot(np.append(frac_data_raw[i, 0], frac_data_raw[i, 2]),
                 np.append(frac_data_raw[i, 1], frac_data_raw[i, 3]))
    wells_inj = input_data['inj_well_coords']
    for i in range(len(wells_inj)):
        plt.plot(wells_inj[i][0], wells_inj[i][1], 'o', color='b', label='inj well')
    wells_prod = input_data['prod_well_coords']
    for i in range(len(wells_prod)):
        plt.plot(wells_prod[i][0], wells_prod[i][1], 'o', color='r', label='prod well')
    plt.xlabel('X, m.')
    plt.ylabel('Y, m.')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'grid_2d.png'), dpi=600)

    # Input parameters for cleaning procedure
    angle_tol_straighten = 7.5  # tolerance for straightening fracture segments [degrees]
    merge_threshold = 0.86  # tolerance for merging nodes in algebraic constraint, values on interval [0.5, 0.86] [-]
    angle_tol_remove_segm = np.arctan(0.35) * 180 / np.pi   # tolerance for removing accute intersections, values on interval [15, 25] [degrees]
    decimals = 7  # in order to remove duplicates we need to have fixed number of decimals
    mesh_raw = True#False  # need gmsh installed and callable from command line in order to mesh!!!
    num_partition_x = 4  # number of partitions for parallel implementation of intersection finding algorithm
    num_partition_y = 4  # " ... "

    frac_preprocessing(frac_data_raw, char_len=input_data['char_len'], output_dir=output_dir, filename_base=input_data['case_name'], merge_threshold=merge_threshold, z_top=input_data['z_top'],
                       height_res=input_data['height_res'], angle_tol_small_intersect=angle_tol_remove_segm, apertures_raw=None, box_data=input_data['box_data'], margin=input_data['margin'],
                       mesh_clean=input_data['mesh_clean'], mesh_raw=mesh_raw, angle_tol_straighten=angle_tol_straighten, straighten_after_cln=True, decimals=decimals,
                       tolerance_zero=1e-10, tolerance_intersect=1e-10, calc_intersections_before=False, calc_intersections_after=False,
                       num_partition_x=num_partition_x, num_partition_y=num_partition_y, partition_fractures_in_segms=True, matrix_perm=1, correct_aperture=False,
                       small_angle_iter=2, char_len_mult=1, char_len_boundary=input_data['char_len_boundary'], main_algo_iters=1,
                       wells=None,#input_data['inj_well_coords']+input_data['prod_well_coords'],
                       char_len_well=input_data['char_len_well'], input_data=input_data)


if __name__ == "__main__":
    freeze_support()

    t1 = datetime.now()
    print(t1)

    input_data = set_input_data('case_1')
    generate_mesh(input_data)

    t2 = datetime.now()
    print((t2-t1).total_seconds())