# Section of the Python code where we import all dependencies on third party Python modules/libaries or our own
# libraries (exposed C++ code to Python, i.e. darts.engines && darts.physics)
from model import Model
import numpy as np
import meshio
from darts.engines import redirect_darts_output
from darts.models.opt.opt_module_settings import model_modifier_aggregator, flux_multiplier_modifier

from numpy import linalg
import pandas as pd
from scipy.optimize import minimize
import sys
import os
import shutil
from matplotlib import pyplot as plt

from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
import math
import pickle
import multiprocessing
import time
import platform

import random
import scipy.stats as stats
from scipy import linalg
import scipy
import meshio
import time

try:
    # if compiled with OpenMP, set to run with 1 thread, as MPFA tests are not working in the multithread version yet
    from darts.engines import set_num_threads

    set_num_threads(1)
except:
    pass

# Run a realization to generate "true" data (otherwise it will be read from pickle file, if it exists)
generate_true_data = True
plot_and_check = generate_true_data  # plot and check the "true" model (observation data); plot and update kernel size, EM data, etc.
true_realization = ""


T = 3000
training_time = 2000
report_step = 500.0


discr_type = 'mpfa'

# mesh_file = "meshes/transfinite.msh"
# mesh_file = 'meshes/transfinite_coarse.msh'
mesh_file = "meshes/transfinite_very_coarse.msh"



opt_algorithm = 'L-BFGS-B'
# opt_algorithm = 'SLSQP'

training_model = True  # switch off to compare and plot the optimized results and un-optimized result
optimization = False  # switch off to compare the adjoint and numerical gradient
apply_adjoint_method = True  # switch off to apply numerical method

add_prod_rate_to_objfun = True
add_inj_rate_to_objfun = False
add_BHP_to_objfun = False
add_well_tempr_to_objfun = True
add_temperature_to_objfun = True
add_customized_op_to_objfun = False

# if switch on, the customized operator should be defined in "set_op_list", e.g. temperature
customize_new_operator = add_temperature_to_objfun



maxiter = 100
# maxiter = 0  # a single initial run for re-scaling the misfit term in objective function (i.e. weight settings)


if generate_true_data:
    starting_time = time.time()

    true_model = Model(discr_type=discr_type, mesh_file=mesh_file, T=T, report_step=report_step,
                       customize_new_operator=customize_new_operator)
    true_model.init(discr_type=discr_type)


    # if discr_type == "mpfa" and 1:
    #     n_fm = true_model.get_n_flux_multiplier()
    #     print("number of fm: %s" % n_fm)
    #
    #     cell_m_mpfa = np.array(true_model.reservoir.mesh.block_m, copy=True)
    #     cell_p_mpfa = np.array(true_model.reservoir.mesh.block_p, copy=True)
    #     tran = np.array(true_model.reservoir.mesh.tran, copy=True)
    #     conn_mpfa = np.column_stack((cell_m_mpfa, cell_p_mpfa))
    #     np.savetxt("conn_C++.txt", conn_mpfa, fmt='%d')
    #     np.savetxt("tran_C++.txt", tran)
    #
    #     n_blocks = true_model.reservoir.mesh.n_blocks
    #     print("number of blocks: %s" % n_blocks)
    #     cell_m_one_way = []
    #     cell_p_one_way = []
    #     conn_index_to_one_way = []
    #     idx_one_way = 0
    #     for idx, cm in enumerate(cell_m_mpfa):
    #         cp = cell_p_mpfa[idx]
    #         # if cm < cp and cp < n_blocks:
    #         if cm < cp:
    #             cell_m_one_way.append(cm)
    #             cell_p_one_way.append(cp)
    #             conn_index_to_one_way.append(idx_one_way)
    #             idx_one_way += 1
    #         elif cm > cp:
    #             cp_indices = np.where(np.array(cell_m_one_way) == cp)[0]  # note here we find cp from cm array
    #             cm_indices = np.where(np.array(cell_p_one_way) == cm)[0]  # note here we find cm from cp array
    #
    #             # Find the intersection set of cp_indices and cm_indices
    #             intersection_indices = np.intersect1d(cp_indices, cm_indices)
    #
    #             conn_index_to_one_way.append(intersection_indices[0])
    #         else:
    #             conn_index_to_one_way.append(-999)
    #
    #
    #     conn_on_way = np.column_stack((np.array(cell_m_one_way), np.array(cell_p_one_way)))
    #     np.savetxt("conn_C++_one_way.txt", conn_on_way, fmt='%d')
    #     np.savetxt("conn_idx_one_way.txt", np.array(conn_index_to_one_way), fmt='%d')



    true_model.run(export_to_vtk=True)


    true_model.print_timers()
    true_model.print_stat()

    ending_time = time.time()
    print('Total elapsed %s sec' % (ending_time - starting_time))

    # Save realization output as a true data
    time_data = pd.DataFrame.from_dict(true_model.physics.engine.time_data)
    time_data.to_pickle("_TRUE_%s_darts_time_data_%s_%sdays.pkl" % (true_realization, T, report_step))
    time_data_report = pd.DataFrame.from_dict(true_model.physics.engine.time_data_report)
    time_data_report.to_pickle("_TRUE_%s_darts_time_data_report_%s_%sdays.pkl" % (true_realization, T, report_step))

    if customize_new_operator:
        # time-lapse temperature data
        time_data_customized = np.array(true_model.physics.engine.time_data_customized)[:, 0:true_model.reservoir.mesh.n_res_blocks]
        time_data_report_customized = np.array(true_model.physics.engine.time_data_report_customized)[:, 0:true_model.reservoir.mesh.n_res_blocks]
        np.savetxt('_TRUE_%s_darts_time_data_customized_%s_%sdays.txt' % (true_realization, T, report_step), np.array(time_data_customized))
        np.savetxt('_TRUE_%s_darts_time_data_report_customized_%s_%sdays.txt' % (true_realization, T, report_step), np.array(time_data_report_customized))
else:
    time_data = pd.read_pickle("_TRUE_%s_darts_time_data_%s_%sdays.pkl" % (true_realization, T, report_step))
    time_data_report = pd.read_pickle("_TRUE_%s_darts_time_data_report_%s_%sdays.pkl" % (true_realization, T, report_step))
    if customize_new_operator:
        time_data_customized = np.loadtxt('_TRUE_%s_darts_time_data_customized_%s_%sdays.txt' % (true_realization, T, report_step))
        time_data_report_customized = np.loadtxt('_TRUE_%s_darts_time_data_report_customized_%s_%sdays.txt' % (true_realization, T, report_step))



# time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
# # np.save("benchmark.npy", time_data['PROD001 : wat rate (m3/day)'].to_numpy())
# diff = np.sum(time_data['PROD001 : wat rate (m3/day)'].to_numpy() - np.load("benchmark.npy"))
# print(diff)


# plot and check the TRUE results
if 0:
    # plot well curves--------------------------------------------------------------------------------------------------
    df_data = time_data.copy()
    df_data_report = time_data_report.copy()

    doublet_idx = 1
    time_arr = time_data['time'].to_numpy()
    time_report_arr = time_data_report['time'].to_numpy()

    plt.subplot(2, 4, 1)
    plt.plot(time_arr, time_data['I%s : temperature (K)' % doublet_idx].to_numpy() - 273.15)
    plt.title('temperature (C)')
    plt.ylabel('I%s' % doublet_idx)
    plt.subplot(2, 4, 2)
    plt.plot(time_arr, time_data['I%s : BHP (bar)' % doublet_idx].to_numpy())
    plt.title('BHP (bar)')
    plt.subplot(2, 4, 3)
    plt.plot(time_arr, time_data['I%s : wat rate (m3/day)' % doublet_idx].to_numpy())
    plt.title('water rate (m3/day)')
    plt.subplot(2, 4, 4)
    plt.plot(time_report_arr, time_data_report['I%s : wat  acc volume (m3)' % doublet_idx].to_numpy())
    plt.title('water acc volume (m3)')

    plt.subplot(2, 4, 5)
    plt.plot(time_arr, time_data['P%s : temperature (K)' % doublet_idx].to_numpy() - 273.15)
    plt.ylabel('P%s' % doublet_idx)
    plt.subplot(2, 4, 6)
    plt.plot(time_arr, time_data['P%s : BHP (bar)' % doublet_idx].to_numpy())
    plt.subplot(2, 4, 7)
    plt.plot(time_arr, time_data['P%s : wat rate (m3/day)' % doublet_idx].to_numpy())
    plt.subplot(2, 4, 8)
    plt.plot(time_report_arr, time_data_report['P%s : wat  acc volume (m3)' % doublet_idx].to_numpy())

    plt.show()


# --------------------------------------------------------------------------------------------------------------
# -----------------------------------THE PREPERATION OF OBSERVATION DATA ---------------------------------------
# --------------------------------------------------------------------------------------------------------------
# prepare training time index, prediction time index, observation data based on synthetic data
# if the observation data is based on the field data, please save them as the pickle file as the sane name
# and data structure of "time_data_report" and "time_data"
Training_report = time_data_report[time_data_report['time'] == training_time].index[0] + 1
truth_df_report = pd.DataFrame.from_dict(time_data_report)[:Training_report]

Training_sim = time_data[time_data['time'] == training_time].index[0] + 1
truth_df_sim = pd.DataFrame.from_dict(time_data)[:Training_sim]

prediction_time = T - training_time

# Record Prediction data
truth_df_sim_pred = pd.DataFrame.from_dict(time_data)
truth_df_report_pred = pd.DataFrame.from_dict(time_data_report)

# BHP or Temperature observation data collection
# BHP stored only in time_data
BT_data = time_data.copy()
t_report = time_data_report['time']
flag = time_data['time'].isin(t_report)

for iii in range(flag.size):
    idx = flag.size - iii - 1
    if flag[idx]:
        pass
    else:
        BT_data.drop([idx], axis=0, inplace=True)

truth_df_BT_report = pd.DataFrame.from_dict(BT_data)[:Training_report]

# customized result data
if customize_new_operator:
    truth_df_customized_report = time_data_report_customized[:Training_report]



# ---------------------------------------------------------------------------------------------------------------
# ----------------------------------------------HISTORY MATCHING-------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

proxy_model = Model(discr_type=discr_type, mesh_file=mesh_file, T=training_time, report_step=report_step,
                    customize_new_operator=customize_new_operator)

if training_model:
    redirect_darts_output('')


proxy_model.init(discr_type=discr_type)


# scale_function_value = 1e-5
scale_function_value = 1e-9


# Set optimization parameters------------------------------------------------------------------
opt_function_tolerance = 1e-7
tol = opt_function_tolerance * scale_function_value
# eps = 1e-9
eps = 1e-7


n_fm = proxy_model.get_n_flux_multiplier()

# n_well_index = 0
# for well in proxy_model.reservoir.wells:
#     n_well_index += len(well.perforations)
#
# n_fm_res = n_fm - n_well_index


# Create modifier-------------------------------------------------------------------------------
model_modifier = model_modifier_aggregator()

# flux multiplier modifier-----------------------
# flux multiplier includes both the reservoir flux multiplier and well index multiplier
model_modifier.append(flux_multiplier_modifier())
norm_tran = 1
model_modifier.modifiers[-1].norms = norm_tran

# ---------------------------------------------------------------------------------------------------------------
# # Set optimization parameters ( Initial guess, bounds, observation data, objective function)
x0_initial = model_modifier.get_x0(proxy_model)
eps_bound = 1e-4
np.random.seed(1)
noise = np.random.uniform(-0.001, 0.001, n_fm)


x0 = x0_initial + noise
bounds = [(0.0001, 5)] * n_fm

for b_idx, b in enumerate(bounds):
    if x0[b_idx] < b[0]:
        x0[b_idx] = b[0] + eps_bound
    if x0[b_idx] > b[1]:
        x0[b_idx] = b[1] - eps_bound


# optimization option settings-----------------------------------------------------------------------------------
proxy_model.scale_function_value = scale_function_value
proxy_model.modifier = model_modifier
proxy_model.n_fm = n_fm

# production well-------------------------------------
prod_well_name = []
for i, w in enumerate(proxy_model.reservoir.wells):
    if "P" in w.name:
        prod_well_name.append(w.name)


prod_phase_name = ["wat"]  # disordered phase names

proxy_model.objfun_prod_phase_rate = add_prod_rate_to_objfun
proxy_model.prod_well_name = prod_well_name
proxy_model.prod_phase_name = prod_phase_name

prod_coef = 1  # overall weights
ww = np.size(prod_well_name)  # well dimension
pp = np.size(prod_phase_name)  # phase dimension
tt = Training_report  # time dimension
proxy_model.prod_weights = prod_coef * np.ones((ww, pp, tt))  # all 1 by default, or customize your own weights matrix here


# injection well------------------------------------
inj_well_name = []
for i, w in enumerate(proxy_model.reservoir.wells):
    if "I" in w.name:
        inj_well_name.append(w.name)

inj_phase_name = ["wat"]

proxy_model.objfun_inj_phase_rate = add_inj_rate_to_objfun
proxy_model.inj_well_name = inj_well_name
proxy_model.inj_phase_name = inj_phase_name

inj_coef = 1  # overall weights
ww = np.size(inj_well_name)  # well dimension
pp = np.size(inj_phase_name)  # phase dimension
tt = Training_report  # time dimension
proxy_model.inj_weights = inj_coef * np.ones((ww, pp, tt))  # all 1 by default, or customize your own weights matrix here

# BHP----------------------------------------------
BHP_well_name = []
for i, w in enumerate(proxy_model.reservoir.wells):
    BHP_well_name.append(w.name)
    # if "I" in w.name:
    #     BHP_well_name.append(w.name)

proxy_model.objfun_BHP = add_BHP_to_objfun
proxy_model.BHP_well_name = BHP_well_name

BHP_coef = 1
ww = np.size(BHP_well_name)  # well dimension
tt = Training_report  # time dimension
proxy_model.BHP_weights = BHP_coef * np.ones((ww, tt))  # all 1 by default, or customize your own weights matrix here

# well temperature----------------------------------
well_tempr_name = []
for i, w in enumerate(proxy_model.reservoir.wells):
    if "P" in w.name:
        well_tempr_name.append(w.name)

proxy_model.objfun_well_tempr = add_well_tempr_to_objfun
proxy_model.well_tempr_name = well_tempr_name

well_tempr_coef = 1
ww = np.size(well_tempr_name)  # well dimension
tt = Training_report  # time dimension
proxy_model.well_tempr_weights = well_tempr_coef * np.ones((ww, tt))  # all 1 by default, or customize your own weights matrix here

# temperature distribution---------------------------
proxy_model.objfun_temperature = add_temperature_to_objfun

if customize_new_operator:
    tempr_coef = 1
    proxy_model.temperature_weights = tempr_coef * np.ones(
        (Training_report, np.size(time_data_report_customized, 1)))


# customized operator---------------------------------
proxy_model.objfun_customized_op = add_customized_op_to_objfun

if customize_new_operator:
    op_coef = 1
    proxy_model.customized_op_weights = op_coef * np.ones(
        (Training_report, np.size(time_data_report_customized, 1)))


# activate optimization options--------------------------------
proxy_model.activate_opt_options()

# rate control input--------------------------------
proxy_model.report_data = truth_df_report
proxy_model.sim_data = truth_df_sim

# eps if for numerical gradient
proxy_model.eps = eps

proxy_model.set_objfun(proxy_model.objfun_assembly)
# proxy_model.set_objfun(proxy_model.objfun_dirac_separate_ensemble_based)

# get the measurement time point array "t_Q"
proxy_model.set_observation_data_report(truth_df_report)

# set your synthetic data or field data here-----------------------
proxy_model.set_observation_data(truth_df_report)
proxy_model.BHP_report_data = truth_df_BT_report
proxy_model.well_tempr_report_data = truth_df_BT_report
if customize_new_operator:
    if add_temperature_to_objfun:
        proxy_model.temperature_report_data = truth_df_customized_report

    if add_customized_op_to_objfun:
        proxy_model.customized_op_report_data = truth_df_customized_report


# history matching ---------------------------------------------------------------------
job_id = 'base_case'
proxy_model.job_id = job_id  # will be used to save %s_Optimized_parameters_best.pkl

print("\n")
print("The number of control variables: %s" % np.size(x0))
print("\n")

starting_time = time.time()

if training_model:
    if optimization:
        if apply_adjoint_method:

            obj_func = proxy_model.make_opt_step_adjoint_method
            grad_func = proxy_model.grad_adjoint_method_mpfa_all

            if opt_algorithm == 'SLSQP':
                opt_adjoint = minimize(obj_func, x0, method='SLSQP', jac=grad_func, bounds=bounds,
                                       options={'maxiter': maxiter, 'ftol': tol, 'iprint': 100, 'disp': True})

            elif opt_algorithm == 'L-BFGS-B':
                opt_adjoint = minimize(obj_func, x0, method='L-BFGS-B', jac=grad_func, bounds=bounds,
                                       options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-15, 'iprint': 101,
                                                'disp': True, 'maxcor': 50})
            print("\r")
            print("\r" + str(opt_adjoint.message))
            print("\r")
            print("\r")
            print('Objective function value is %f' % (opt_adjoint.fun / scale_function_value))
            print("\r")

            ending_time = time.time()
            print('Total elapsed %s sec' % (ending_time - starting_time))

            filename = '%s_Optimized_parameters.pkl' % job_id
            with open(filename, "wb") as fp:
                pickle.dump([opt_adjoint.x, proxy_model.modifier.mod_x_idx, proxy_model.modifier],
                            fp, pickle.HIGHEST_PROTOCOL)

    else:  # compare the adjoint and numerical gradient
        obj_func = proxy_model.make_opt_step_adjoint_method
        grad_func = proxy_model.grad_adjoint_method_mpfa_all

        opt_adjoint = minimize(obj_func, x0, method='SLSQP', jac=grad_func, bounds=bounds,
                               options={'maxiter': 0, 'ftol': tol, 'iprint': 100, 'disp': True})

        opt_num = minimize(obj_func, x0, method='SLSQP', bounds=bounds,
                           options={'maxiter': 0, 'ftol': tol, 'iprint': 100, 'disp': True, 'eps': eps})


        adjoint_gradient = array(opt_adjoint.jac)

        proxy_model.print_timers()
        proxy_model.print_stat()



        numerical_gradient = array(opt_num.jac)
        # np.save("_num_grad.npy", numerical_gradient)

        # numerical_gradient = np.load("_num_grad.npy")

        c = dot(adjoint_gradient, numerical_gradient) / norm(adjoint_gradient) / norm(numerical_gradient)
        angle = arccos(clip(c, -1, 1))

        print("\r")
        # print('Objective function value is %f' % (opt_num.fun / scale_function_value))
        print("\r" + 'The numerical gradients are:')
        print(numerical_gradient)

        print("\r")
        # print('Objective function value is %f' % (opt_adjoint.fun / scale_function_value))
        print("\r" + 'The adjoint gradients are:')
        print(adjoint_gradient)

        print("\r")
        print("\r")


        # so far we compare the adjoint gradient with pkl file. This might be improved in the future
        with open('perf_gradient.pkl', 'rb') as f:
            perf_gradient = pickle.load(f)

        c = dot(adjoint_gradient, perf_gradient) / norm(adjoint_gradient) / norm(perf_gradient)
        angle = arccos(clip(c, -1, 1))

        angle_grade = angle * 180 / math.pi
        print('The angle is %f' % angle_grade)
