import numpy as np
from multiprocessing import freeze_support
from darts.fracture_network.preprocessing_code import frac_preprocessing
import os
from datetime import datetime

# add balmatt path to import
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)  # 1 level up
balmatt_dir = os.path.join(os.path.join(parent_dir, 'vito'), 'darts_model_balmatt')
sys.path.insert(0, balmatt_dir)

# add examples path to import
examples_dir = os.path.join(current_dir, 'examples')
sys.path.insert(0, examples_dir)

# specify the case to run
from examples.case_1 import *
#from examples.case_2 import *
#from examples.case_3 import *

output_dir = 'meshes_' + input_data['case_name']

def main():
    print('case', input_data['case_name'])
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

    # clean dir
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)

    # create dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
                       height_res=input_data['height_res'], angle_tol_small_intersect=angle_tol_remove_segm, apertures_raw=None, box_data=None, margin=input_data['margin'],
                       mesh_clean=input_data['mesh_clean'], mesh_raw=mesh_raw, angle_tol_straighten=angle_tol_straighten, straighten_after_cln=True, decimals=decimals,
                       tolerance_zero=1e-10, tolerance_intersect=1e-10, calc_intersections_before=False, calc_intersections_after=False,
                       num_partition_x=num_partition_x, num_partition_y=num_partition_y, partition_fractures_in_segms=True, matrix_perm=1, correct_aperture=False,
                       small_angle_iter=2, char_len_mult=1, char_len_boundary=input_data['char_len_boundary'], main_algo_iters=1,
                       wells=None, #input_data['inj_well_coords']+input_data['prod_well_coords'],
                       char_len_well=input_data['char_len_well'], input_data=input_data)


if __name__ == "__main__":
    freeze_support()

    t1 = datetime.now()
    print(t1)

    main()

    t2 = datetime.now()
    print((t2-t1).total_seconds())