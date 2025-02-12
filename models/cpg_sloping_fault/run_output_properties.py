import os, sys
import numpy as np

from model_geothermal import ModelGeothermal
from model_deadoil import ModelDeadOil



def run_output_prop(physics_type, case, out_dir, redirect_log, platform):
    if physics_type == 'geothermal':
        m = ModelGeothermal()
    elif physics_type == 'deadoil':
        m = ModelDeadOil()
    else:
        print('Error: wrong physics specified:', physics_type)
        exit(1)

    m.physics_type = physics_type

    m.set_input_data(case=case)
    m.init_reservoir()
    m.init(output_folder=out_dir, platform=platform, restart=True)

    # geomech #############################################
    from geomechanics import calc_geomech
    from reservoir_geomech import CPG_Reservoir_proxy

    export_vtk = True
    compute_stress = False

    #geomech_mode = 'none'
    #geomech_mode = 'surface'  # compute geomech at z=0
    #geomech_mode = 'layer_5'
    geomech_mode = 'cell_centers'

    rsv_geomech = CPG_Reservoir_proxy(m)

    #timesteps, cell_id, X, var_names = m.read_specific_data(path)

    for ith_step in range(len(m.idata.sim.time_steps)):
        arr_geomech = calc_geomech(m=m, rsv=rsv_geomech, ith_step=ith_step, mode=geomech_mode, compute_stress=compute_stress, compute_displs=True)
        ########################################################

        if export_vtk and geomech_mode == 'cell_centers':
            m.out = dict()
            for k in arr_geomech.keys(): # reshape 1D geomech output to 2D since output_to_vtk requires that
                m.out[k] = arr_geomech[k].reshape(1, arr_geomech[k].shape[0])

        if export_vtk:
            for ith_step in range(len(m.idata.sim.time_steps)):
                m.output_to_vtk(ith_step=ith_step)

##########################################################################################################

if __name__ == '__main__':
    platform = 'cpu'
    if os.getenv('TEST_GPU') != None and os.getenv('TEST_GPU') == '1':
            platform = 'gpu'

    physics_list = []
    #physics_list += ['geothermal']
    physics_list += ['ccs']
    #physics_list += ['deadoil']

    cases_list = []
    cases_list += ['generate_5x3x4']
    #cases_list += ['generate_51x51x1']
    #cases_list += ['generate_100x100x100']
    #cases_list += ['case_40x40x10']
    #cases_list += ['brugge']

    well_controls = []
    well_controls += ['wrate']
    #well_controls += ['wbhp']
    #well_controls += ['wperiodic']

    for physics_type in physics_list:
        for case_geom in cases_list:
            for wctrl in well_controls:
                if physics_type == 'deadoil' and wctrl == 'wrate':
                    continue
                case = case_geom + '_' + wctrl
                out_dir = 'results_' + physics_type + '_' + case
                run_output_prop(physics_type=physics_type,case=case, out_dir=out_dir,redirect_log=False,platform=platform)

