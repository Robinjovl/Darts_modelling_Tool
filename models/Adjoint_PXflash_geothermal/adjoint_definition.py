import numpy as np
from darts.engines import value_vector, redirect_darts_output
from darts.models.opt.opt_module_settings import model_modifier_aggregator, transmissibility_modifier, well_index_modifier
from model_definition import Model

import time
import pandas as pd
from scipy.optimize import minimize
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
import math
import pickle



# Run a realization to generate "true" data (otherwise it will be read from pickle file, if it exists)
generate_true_data = True
plot_and_check = False  # plot and check the "true" model (observation data); plot and update kernel size, EM data, etc.
true_realization = "real"
iapws_physics = False



# year = 4
# T = 360 * year  # days
# training_time = year / 2 * 360
# report_step = 30  # days

T = 5  # days
training_time = 3
report_step = 1  # days

perm = 300  # mD
poro = 0.2

opt_algorithm = 'L-BFGS-B'
# opt_algorithm = 'SLSQP'

training_model = True  # switch off to compare and plot the optimized results and un-optimized result
optimization = False  # switch off to compare the adjoint and numerical gradient
apply_adjoint_method = True  # switch off to apply numerical method

add_prod_rate_to_objfun = True
add_inj_rate_to_objfun = False
add_BHP_to_objfun = False
add_well_tempr_to_objfun = False
add_temperature_to_objfun = False
add_customized_op_to_objfun = False

# if switch on, the customized operator should be defined in "set_op_list", e.g. temperature
# customize_new_operator = add_temperature_to_objfun
if add_temperature_to_objfun or add_customized_op_to_objfun:
    customize_new_operator = True
else:
    customize_new_operator = False


time_data = 0
time_data_report = 0
time_data_customized = 0
time_data_report_customized = 0


def prepare_synthetic_observation_data():
    # --------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------TRUE-MODEL----------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

    if generate_true_data:
        true_model = Model(T, report_step=report_step, perm=perm, poro=poro, iapws_physics=iapws_physics)
        true_model.init()
        true_model.set_output()
        true_model.run(export_to_vtk=False)
        true_model.print_timers()
        true_model.print_stat()

        # Save realization output as a true data
        time_data = pd.DataFrame.from_dict(true_model.physics.engine.time_data)
        time_data.to_pickle("_TRUE_%s_darts_time_data_%s_%sdays.pkl" % (true_realization, T, report_step))
        time_data_report = pd.DataFrame.from_dict(true_model.physics.engine.time_data_report)
        time_data_report.to_pickle(
            "_TRUE_%s_darts_time_data_report_%s_%sdays.pkl" % (true_realization, T, report_step))

        if customize_new_operator:
            # time-lapse temperature data
            time_data_customized = np.array(true_model.physics.engine.time_data_customized)[:, 0:true_model.reservoir.mesh.n_res_blocks]
            time_data_report_customized = np.array(true_model.physics.engine.time_data_report_customized)[:, 0:true_model.reservoir.mesh.n_res_blocks]
            np.savetxt('_TRUE_%s_darts_time_data_customized_%s_%sdays.txt' % (true_realization, T, report_step), np.array(time_data_customized))
            np.savetxt('_TRUE_%s_darts_time_data_report_customized_%s_%sdays.txt' % (true_realization, T, report_step), np.array(time_data_report_customized))

        print('The time data results are saved!!!')


def read_observation_data():
    global time_data, time_data_report, time_data_customized, time_data_report_customized
    time_data = pd.read_pickle("_TRUE_%s_darts_time_data_%s_%sdays.pkl" % (true_realization, T, report_step))
    time_data_report = pd.read_pickle("_TRUE_%s_darts_time_data_report_%s_%sdays.pkl" % (true_realization, T, report_step))
    if customize_new_operator:
        time_data_customized = np.loadtxt('_TRUE_%s_darts_time_data_customized_%s_%sdays.txt' % (true_realization, T, report_step))
        time_data_report_customized = np.loadtxt('_TRUE_%s_darts_time_data_report_customized_%s_%sdays.txt' % (true_realization, T, report_step))


def process_adjoint(history_matching=False):
    optimization = history_matching
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


    maxiter = 100

    # ---------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------HISTORY MATCHING-------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------


    proxy_model = Model(T=training_time, report_step=report_step, perm=perm, poro=poro, iapws_physics=iapws_physics)

    if training_model:
        redirect_darts_output('')


    proxy_model.init()
    proxy_model.set_output(save_initial=False, )


    scale_function_value = 1e-5
    # scale_function_value = 1e-9


    # Set optimization parameters------------------------------------------------------------------
    opt_function_tolerance = 1e-7
    tol = opt_function_tolerance * scale_function_value
    # eps = 1e-9
    eps = 1e-5

    t = value_vector([])
    t_D = value_vector([])
    proxy_model.reservoir.mesh.get_res_tran(t, t_D)
    n_T_res = np.size(t)

    # n_T_res = np.size(proxy_model.reservoir.tran)
    n_well_index = 0
    for well in proxy_model.reservoir.wells:
        n_well_index += len(well.perforations)
    n_T = n_T_res + n_well_index

    # Create modifier-------------------------------------------------------------------------------
    # IMPORTANT! When applying adjoint method, the first and the second modifier have to be transmissibility_modifier()
    # and well_index_modifier(), respectively. The nonlinear modifiers (e.g., corey_params_modifier_region) should follow
    # behind them.
    model_modifier = model_modifier_aggregator()


    # trans modifier-----------------------
    model_modifier.append(transmissibility_modifier())
    norm_tran = 10000
    # bounds += [(1e-9, 1)] * n_T_res
    model_modifier.modifiers[-1].norms = norm_tran

    # well index modifier------------------
    model_modifier.append(well_index_modifier())  # Number of entries = 30
    norm_WI = 1000
    # bounds += [(1e-9, 1)] * n_well_index
    model_modifier.modifiers[-1].norms = norm_WI

    # ---------------------------------------------------------------------------------------------------------------
    # # Set optimization parameters ( Initial guess, bounds, observation data, objective function)
    x0_initial = model_modifier.get_x0(proxy_model)

    lb = np.min(t) / model_modifier.modifiers[0].norms
    ub = np.max(t) / model_modifier.modifiers[0].norms

    eps_bound = 1e-4

    x0_trans = np.linspace(lb * 1.1, ub * 0.9, n_T_res)
    x0_WI = np.linspace(0.01, 0.5, n_well_index)
    x = np.concatenate((x0_trans, x0_WI))

    x0_trans = np.random.uniform(lb + eps_bound, ub - eps_bound, n_T_res)
    x0_WI = np.random.uniform(0 + eps_bound, 1 - eps_bound, n_well_index)
    x = np.concatenate((x0_trans, x0_WI))


    seed = 0
    np.random.seed(seed)
    noise_trans = (np.random.uniform(lb, ub, n_T_res) - (ub - lb) / 2) * 0.02
    noise_WI = (np.random.uniform(0, 1, n_well_index) - 0.5) * 0.02
    noise = np.concatenate((noise_trans, noise_WI))

    x0_trans = np.ones(n_T_res) * (lb + ub) / 2
    x0_WI = np.ones(n_well_index) * (1e-7 + 1) / 2
    x_uniform = np.concatenate((x0_trans, x0_WI))

    bounds = [(lb, ub)] * n_T_res + [(1e-7, 1)] * n_well_index
    x_idx = [0, n_T_res, n_T]

    # x0 = x
    # x0 = x0_initial
    # x0 = x0_initial * (1 + 0.2 * np.random.randn(np.size(x0_initial)))
    x0 = x0_initial + noise
    # x0 = x_uniform

    for b_idx, b in enumerate(bounds):
        if x0[b_idx] < b[0]:
            x0[b_idx] = b[0] + eps_bound
        if x0[b_idx] > b[1]:
            x0[b_idx] = b[1] - eps_bound



    # optimization option settings-----------------------------------------------------------------------------------
    proxy_model.scale_function_value = scale_function_value
    proxy_model.modifier = model_modifier
    proxy_model.x_idx = x_idx

    # production well-------------------------------------
    prod_well_name = []
    for i, w in enumerate(proxy_model.reservoir.wells):
        if "P" in w.name:
            prod_well_name.append(w.name)

    # prod_well_name = ['P1', 'P2']
    prod_well_name = ['P2', 'P1']  # disordered well names

    # prod_phase_name = [proxy_model.physics.phases[0]]  # water phase
    # prod_phase_name = [proxy_model.physics.phases[1]]  # oil phase
    # prod_phase_name = proxy_model.physics.phases  # all phases
    # prod_phase_name = ['wat', 'gas']  # disordered phase names
    prod_phase_name = ['water', 'steam']


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
    inj_phase_name = [proxy_model.physics.phases[0]]

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
                grad_func = proxy_model.grad_adjoint_method_all

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

                if 1:
                    from matplotlib import pyplot as plt
                    wn = "P1"
                    # plot and check the TRUE results-------------------------------------------------------------------
                    df_data = time_data.copy()
                    df_data_report = time_data_report.copy()

                    time_arr = time_data['time'].to_numpy()
                    time_report_arr = time_data_report['time'].to_numpy()

                    ax1 = plt.subplot(1, 1, 1)
                    ax1.plot(time_report_arr, time_data_report['%s : oil rate (m3/day)' % wn].to_numpy(), color="red")


                    proxy_model.T = training_time + prediction_time
                    # re-run unoptimized parameters---------------------------------------------------------------------
                    # ************************************************************************************************
                    print('\n')
                    print(
                        '--------------------------------------Re-run unoptimized model-------------------------------')
                    print('\n')
                    proxy_model.set_modifier_and_du_dT_and_x_idx(model_modifier, x_idx)
                    model_modifier.set_x_by_du_dT(proxy_model, x0)

                    proxy_model.set_initial_conditions()
                    proxy_model.set_boundary_conditions()
                    proxy_model.set_op_list()
                    proxy_model.reset()

                    proxy_model.run()
                    response_initial = proxy_model.physics.engine.time_data_report
                    unopt_df_report = pd.DataFrame.from_dict(response_initial)
                    unopt_df_report.to_pickle('time_data_report_unopt_%s.pkl' % job_id)
                    unopt_df = pd.DataFrame.from_dict(proxy_model.physics.engine.time_data)
                    unopt_df.to_pickle('time_data_unopt_%s.pkl' % job_id)
                    if customize_new_operator:
                        unopt_tempr_pred = np.array(proxy_model.physics.engine.time_data_report_customized)[:,
                                           0:proxy_model.reservoir.mesh.n_res_blocks]
                        np.save('Tempr_distr_unopt_report_%s.npy' % job_id, unopt_tempr_pred)


                    time_report_unopt = unopt_df_report['time'].to_numpy()
                    ax1.plot(time_report_unopt, unopt_df_report['%s : oil rate (m3/day)' % wn].to_numpy(), color="silver")

                    # re-run optimized parameters-----------------------------------------------------------------------
                    # ************************************************************************************************
                    print('\n')
                    print('--------------------------------------Re-run optimized model-------------------------------')
                    print('\n')

                    proxy_model.reset()

                    # here you can choose either Optimized_parameters_best.pkl or Optimized_parameters.pkl to re-run it
                    # Optimized_parameters_best.pkl is for saving the temporary result in case the history matching fails
                    with open('%s_Optimized_parameters_best.pkl' % job_id, "rb") as fp:
                        pickl_result = pickle.load(fp)
                        u = pickl_result[0]
                        proxy_model.set_modifier_and_du_dT_and_x_idx(model_modifier, x_idx)
                        proxy_model.modifier.set_x_by_du_dT(proxy_model, u)

                    proxy_model.set_initial_conditions()
                    proxy_model.set_boundary_conditions()
                    proxy_model.set_op_list()
                    proxy_model.reset()

                    proxy_model.run()
                    response_optimized = proxy_model.physics.engine.time_data_report
                    opt_df_report_pred = pd.DataFrame.from_dict(response_optimized)
                    opt_df_report_pred.to_pickle('time_data_report_opt_%s.pkl' % job_id)
                    opt_df_pred = pd.DataFrame.from_dict(proxy_model.physics.engine.time_data)
                    opt_df_pred.to_pickle('time_data_opt_%s.pkl' % job_id)
                    if customize_new_operator:
                        opt_tempr_pred = np.array(proxy_model.physics.engine.time_data_report_customized)[:,
                                         0:proxy_model.reservoir.mesh.n_res_blocks]
                        tempr_time_data = np.array(proxy_model.physics.engine.time_data_customized)[:,
                                          0:proxy_model.reservoir.mesh.n_res_blocks]
                        np.save('Tempr_distr_opt_report_%s.npy' % job_id, opt_tempr_pred)

                    time_report_opt = opt_df_report_pred['time'].to_numpy()
                    ax1.plot(time_report_opt, opt_df_report_pred['%s : oil rate (m3/day)' % wn].to_numpy(), color="blue")
                    plt.show()

        else:  # compare the adjoint and numerical gradient
            obj_func = proxy_model.make_opt_step_adjoint_method
            grad_func = proxy_model.grad_adjoint_method_all

            opt_num = minimize(obj_func, x0, method='SLSQP', bounds=bounds,
                               options={'maxiter': 0, 'ftol': tol, 'iprint': 100, 'disp': True, 'eps': eps})

            opt_adjoint = minimize(obj_func, x0, method='SLSQP', jac=grad_func, bounds=bounds,
                                   options={'maxiter': 0, 'ftol': tol, 'iprint': 100, 'disp': True})

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
            print('The angle is %f' % (angle * 180 / math.pi))

            print("\r")

            angle_grade = angle * 180 / math.pi

            if angle_grade >= 0 and angle_grade < 5:
                return 0
            else:
                return 1
