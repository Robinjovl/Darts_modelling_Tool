import time
import math
import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime
from darts.engines import *

from numpy import linalg

import multiprocessing
from pickle import Pickler
import sys
import pickle


from darts.engines import value_vector
import os.path as osp

from typing import List
sq_norm = lambda x: np.inner(x, x)


class OptModuleSettings:
    def __init__(self):
        self.terminated_runs = 0
        self.objfun_norm = 1000
        # self.objfun_norm = 1e6
        self.opt_step_time = 0
        self.obj_time = 0
        self.sim_time = 0
        self.n_opt_steps = 0
        self.save_unfinished_runs = False

        self.x_log = []
        self.prepare_obs_data = True
        self.prepare_input_data = True
        self.scale_modified_grad = 1
        self.norm_grad_old = 10000
        self.ad_grad_time = 0
        self.n_grad_calc = 0
        self.forward_temp_time = 0
        self.n_forward_temp = 0
        self.eps = 1e-7
        self.nonlinear_grad_time = 0
        self.result_list = []
        self.task_id = ''
        # self.fval_temp = 10000
        self.regularization = False
        self.re_parameterized_PCA = False
        self.read_Cm_inv_vector = False
        self.Cv_path = ""
        self.ksi = 1
        self.space_dim = 0
        self.x_ave = 1
        self.grad_convert_coeff = 1
        self.phi = 1
        self.norm_ksi = 1
        self.C_m_inv = 0
        self.alpha = 1
        self.ksi_ref = 1
        self.regularization_diagonal = False
        self.x_ref = 1
        self.Cm_inv_diagonal = 0

        self.objfun_prod_phase_rate = True
        self.prod_well_name = []
        self.prod_phase_name = []
        self.prod_rate_measurement_error = 0.0
        self.prod_weights = []
        self.prod_error_lower_bound = 5
        self.prod_error_upper_bound = 10000000

        self.objfun_inj_phase_rate = False
        self.inj_well_name = []
        self.inj_phase_name = []
        self.inj_rate_measurement_error = 0.0
        self.inj_weights = []
        self.inj_error_lower_bound = 5
        self.inj_error_upper_bound = 10000000

        self.objfun_BHP = False
        self.BHP_well_name = []
        self.BHP_report_data = pd.DataFrame()
        self.BHP_measurement_error = 0.0
        self.BHP_weights = []
        self.BHP_error_lower_bound = 0.001
        self.BHP_error_upper_bound = 10000000

        self.objfun_well_tempr = False
        self.well_tempr_name = []
        self.well_tempr_report_data = pd.DataFrame()
        self.well_tempr_measurement_error = 0.0
        self.well_tempr_weights = []
        self.well_tempr_error_lower_bound = 1
        self.well_tempr_error_upper_bound = 10000000

        self.objfun_temperature = False
        self.temperature_report_data = []
        self.temperature_measurement_error = 0.0
        self.temperature_weights = []
        self.temperature_error_lower_bound = 1
        self.temperature_error_upper_bound = 10000000
        self.base_noise_temperature_EM = []
        self.temperature_EM_list = []
        self.temperature_EM_cov_mat_inv = []
        
        self.objfun_customized_op = False
        self.customized_op_report_data = []
        self.customized_op_measurement_error = 0.0
        self.customized_op_weights = []
        self.customized_op_error_lower_bound = 1
        self.customized_op_error_upper_bound = 10000000
        

        self.objfun_saturation = False
        self.phase_relative_density = []

        self.opt_phase_rate = True
        self.scale_function_value = 1e-10
        self.modifier = ''
        self.x_idx = []

        self.forward_temp_result = 0
        self.previous_forward_result = 0
        self.heuristic_rate_control = False

        self.job_id = ''
        self.objfunval = 100000000
        self.objfun_all = []
        self.fval_list = []
        self.misfit_watch = False
        self.misfit_value = 0.0
        self.misfit_watch_list = []

        # hinge loss for binary classification or threshold problem
        self.threshold = []
        self.binary_array = []
        self.save_error = False
        self.label = ''

        # MPFA
        self.n_fm = 0


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------  Adjoint method - Xiaoming Tian------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
    def set_modifier_and_du_dT_and_x_idx(self, modifier, x_idx: List[int]):
        '''
        The settings of modifier, x_idx, and col_idx
        :param modifier: model modifier
        :param x_idx: list of the index of the control variables: [0, n_T_res, n_T_res + n_WI]
        '''
        # self.du_dT = du_dT
        self.x_idx = x_idx
        self.col_idx_original = []

        self.col_idx_original = list(range(self.x_idx[-1]))
        self.col_idx = self.col_idx_original

        # sorting the perforation index for adjoint gradient
        # `self.col_idx` will be passed into C++ `col_dT_du`
        self.perforation_idx = []
        for i, w in enumerate(self.reservoir.wells):
            for p, per in enumerate(w.perforations):
                self.perforation_idx.append(per[1])

        self.sort_perforation_idx = np.zeros(np.size(self.perforation_idx))
        for i in range(np.size(self.perforation_idx)):
            for j in range(np.size(self.perforation_idx)):
                if self.perforation_idx[j] < self.perforation_idx[i]:
                    self.sort_perforation_idx[i] += 1

        temp_idx = self.sort_perforation_idx.astype(int)
        n_T_res = np.size(self.col_idx) - np.size(temp_idx)
        col_idx_well = self.col_idx[np.size(self.col_idx) - np.size(temp_idx):]
        for i, j in enumerate(temp_idx):
            self.col_idx[n_T_res + i] = col_idx_well[j]

        self.modifier = modifier

    def make_opt_step_adjoint_method(self, x: np.array, *args) -> float:
        '''
        Objective function definition
        :param x: model control variables
        :param *args: extra argument. This is usually kept empty
        :return: objective function value
        '''
        # print(args[0])
        self.x_temp = x

        self.opt_step_time -= time.time()

        # 1. Update model
        self.modifier.set_x_by_du_dT(self, x)
        # self.modifier.set_x(self, x)

        self.engine.opt_history_matching = True
        if type(self.modifier.modifiers[0]) == flux_multiplier_modifier:  # for MPFA
            self.engine.is_mp = True

        # 2. Reset
        self.set_initial_conditions()
        self.set_boundary_conditions()
        self.set_op_list()
        self.reset()

        self.sim_time -= time.time()
        # 3. Run
        if args:
            args = args[0]
            self.engine.clear_previous_adjoint_assembly()
            self.run(start_opt=args[0],stop_opt=args[1])
        else:
            self.engine.clear_previous_adjoint_assembly()
            self.run(export_to_vtk=False)
            # self.run_python()opt_history_matching
        # self.run()
        self.sim_time += time.time()

        self.obj_time -= time.time()
        # 4. Return objective
        if args:
            obj = self.objfun(start_opt=args[0],stop_opt=args[1])
        else:
            obj = self.objfun()
            # print(obj)
        self.obj_time += time.time()

        # 5. If simulation has not finished, rerun it to save the logs.
        if self.save_unfinished_runs:
            if (obj == 1000):
                log_fname = 'terminated_run_%d' % self.terminated_runs

                with open(log_fname + '.x', 'w') as log:
                    log.write('Problem occurred with: \n')
                    log.write(np.array_str(x))

                np.save(log_fname, x)

                redirect_darts_output(log_fname + '.log')
                self.terminated_runs += 1
                self.modifier.set_x_by_du_dT(self, x)
                self.reset()
                self.engine.clear_previous_adjoint_assembly()
                self.run()
                redirect_darts_output('')

        self.opt_step_time += time.time()
        self.n_opt_steps += 1

        print('\r Run %d: %f s/forward_obj' % (self.n_opt_steps, self.opt_step_time / self.n_opt_steps), end='', flush=True)

        return obj


    def make_opt_step_adjoint_method_PCA(self, x: np.array, *args) -> float:
        '''
        Objective function defintion for dimension reduction using PCA
        :param x: model control variables in reduced-dimension space
        :param *args: extra argument. This is usually kept empty
        :return: objective function value
        '''
        self.x_temp = x

        # convert the control variables from the reduced-dimension space to the original space
        u_trans = self.phi.dot(x[0:self.space_dim] * self.norm_ksi) + self.x_ave
        u = np.concatenate((u_trans, x[self.space_dim:]))
        u[u < 0] = np.mean(u)  # the correction of control variables in case of smaller than 0
        # u[u < 0] = 0.00001

        self.opt_step_time -= time.time()

        # 1. Update model
        self.modifier.set_x_by_du_dT(self, u)
        # self.modifier.set_x(self, x)

        self.engine.opt_history_matching = True

        # 2. Reset
        self.set_initial_conditions()
        self.set_boundary_conditions()
        self.set_op_list()
        self.reset()

        self.sim_time -= time.time()
        # 3. Run
        if args:
            args = args[0]
            self.engine.clear_previous_adjoint_assembly()
            self.run(start_opt=args[0],stop_opt=args[1])
        else:
            self.engine.clear_previous_adjoint_assembly()
            self.run(export_to_vtk=False)
            # self.run_python()opt_history_matching
        # self.run()
        self.sim_time += time.time()

        self.obj_time -= time.time()
        # 4. Return objective
        if args:
            obj = self.objfun(start_opt=args[0],stop_opt=args[1])
        else:
            obj = self.objfun()
            # print(obj)
        self.obj_time += time.time()

        # 5. If simulation has not finished, rerun it to save the logs
        if self.save_unfinished_runs:
            if (obj == 1000):
                log_fname = 'terminated_run_%d' % self.terminated_runs

                with open(log_fname + '.x', 'w') as log:
                    log.write('Problem occurred with: \n')
                    log.write(np.array_str(u))

                np.save(log_fname, u)

                redirect_darts_output(log_fname + '.log')
                self.terminated_runs += 1
                self.modifier.set_x_by_du_dT(self, u)
                self.reset()
                self.engine.clear_previous_adjoint_assembly()
                self.run()
                redirect_darts_output('')

        self.opt_step_time += time.time()
        self.n_opt_steps += 1

        print('\r Run %d: %f s/forward_obj' % (self.n_opt_steps, self.opt_step_time / self.n_opt_steps), end='', flush=True)

        return obj


    def make_opt_single_step(self, x: np.array, *args) -> float:
        '''
        Objective function defintion for a single forward simulation run. This is for re-scaling the misfit terms using different weights
        :param x: model control variables
        :param *args: extra argument. This is usually kept empty
        :return: objective function value
        '''
        # print(args[0])
        self.x_temp = x

        self.opt_step_time -= time.time()

        # 1. Update model
        self.modifier.set_x_by_du_dT(self, x)
        # self.modifier.set_x(self, x)

        # self.engine.opt_history_matching = True

        # 2. Reset
        self.set_initial_conditions()
        self.set_boundary_conditions()
        self.set_op_list()
        self.reset()

        self.sim_time -= time.time()
        if args:
            args = args[0]
            self.run(start_opt=args[0],stop_opt=args[1])
        else:
            self.run()
            # self.run_python()
        # self.run()
        self.sim_time += time.time()

        self.obj_time -= time.time()
        # 4. Return objective
        if args:
            obj = self.objfun(start_opt=args[0],stop_opt=args[1])
        else:
            obj = self.objfun()
            # print(obj)
        self.obj_time += time.time()

        # 5. If simulation has not finished, rerun it to save the logs
        if self.save_unfinished_runs:
            if (obj == 1000):
                log_fname = 'terminated_run_%d' % self.terminated_runs

                with open(log_fname + '.x', 'w') as log:
                    log.write('Problem occurred with: \n')
                    log.write(np.array_str(x))

                np.save(log_fname, x)

                redirect_darts_output(log_fname + '.log')
                self.terminated_runs += 1
                self.modifier.set_x_by_du_dT(self, x)
                self.reset()
                self.run()
                redirect_darts_output('')

        self.opt_step_time += time.time()
        self.n_opt_steps += 1

        print('\r Run %d: %f s/forward_obj' % (self.n_opt_steps, self.opt_step_time / self.n_opt_steps), end='', flush=True)

        return obj


    def input_data_assembly(self):
        '''
        Assemblying of the input observation data
        '''
        # replace these data with your field data if they are available

        self.inj_water_rate = pd.read_pickle("darts_time_data_report_training_period.pkl")
        self.prod_oil_rate = pd.read_pickle("darts_time_data_report_training_period.pkl")
        # self.inj_water_rate = self.previous_forward_result
        # self.prod_oil_rate = self.previous_forward_result
        time_arr = np.array(self.inj_water_rate['time'])
        time_arr = np.concatenate(([0], time_arr))
        self.time_step_arr = time_arr[1:] - time_arr[0:-1]

    def make_single_forward_simulation(self, x: np.array, *args) -> int:
        '''
        The preparation of the last time results for generating heuristic rate control
        :param x: model control variables
        :param *args: extra argument. This is usually kept empty
        :return: 0
        '''
        # print(args[0])
        self.x_temp = x

        self.forward_temp_time -= time.time()

        # 1. Update model----------------------
        self.modifier.set_x_by_du_dT(self, x)
        # self.modifier.set_x(self, x)

        # self.set_boundary_conditions()
        for i, w in enumerate(self.reservoir.wells):
            if "I" in w.name:
                w.control = self.physics.new_rate_water_inj(0)  # will be set some new value next
                w.constraint = self.physics.new_bhp_water_inj(190)  # set this BHP constraint in reasonable range
            else:
                w.control = self.physics.new_rate_oil_prod(0)  # will be set some new value next
                w.constraint = self.physics.new_bhp_prod(20)  # set this BHP constraint in reasonable range

        # 2. Reset------------------------------
        self.reset()

        # 3. Run-------------------------------
        # only need to prepare input data once
        if self.prepare_input_data:
            self.input_data_assembly()
            self.prepare_input_data = False

        # this is where we actually apply rate control as input (e.g., your field data or synthetic data),
        # though we will soon convert these rate information to BHP information
        for ts in range(0, len(self.prod_oil_rate)):
            for w in self.reservoir.wells:
                if 'I' in w.name:
                    col = w.name + ' : water rate (m3/day)'
                    if type(w.control) == rate_inj_well_control:
                        c = w.control
                    else:
                        c = w.constraint
                    if self.inj_water_rate[col][ts] != -1:
                        c.target_rate = self.inj_water_rate[col][ts]
                    else:
                        c.target_rate = 0
                else:
                    col = w.name + ' : oil rate (m3/day)'
                    if type(w.control) == rate_prod_well_control:
                        c = w.control
                    else:
                        c = w.constraint
                    if self.prod_oil_rate[col][ts] != -1:
                        c.target_rate = self.prod_oil_rate[col][ts]
                    else:
                        c.target_rate = 0

            self.engine.run(self.time_step_arr[ts])
            self.engine.report()


        self.forward_temp_time += time.time()
        self.n_forward_temp += 1

        print(', %f s/forward_temp' % (self.forward_temp_time / self.n_forward_temp), end='', flush=True)

        self.forward_temp_result = pd.DataFrame.from_dict(self.engine.time_data)

        return 0


    def set_observation_data_string(self, well_name: List[str], component_index: List[int], phase_index: List[int],
                                    phase_name: List[str], unit: str, opt_comp_rate: str, opt_phase_rate: str):
        '''
        Legacy code for the settings of the history matcing. This will be deprecated.
        '''
        self.prod_well_name = well_name
        self.component_index = component_index
        self.phase_index = phase_index
        self.prod_phase_name = phase_name
        self.unit = unit
        self.opt_comp_rate = opt_comp_rate
        self.opt_phase_rate = opt_phase_rate

        self.engine.prod_well_name = well_name
        # self.engine.component_index = index_vector(component_index)
        # self.engine.phase_index = index_vector(phase_index)
        self.engine.prod_phase_name = phase_name
        self.engine.unit = unit
        self.engine.optimize_component_rate = opt_comp_rate
        self.engine.optimize_phase_rate = opt_phase_rate

    def set_optimization_scale_parameter(self, scale_fun_val: float):
        '''
        Legacy code for the settings of some scaling factors. This will be deprecated.
        '''
        self.scale_function_value = scale_fun_val
        self.engine.scale_function_value = scale_fun_val

    def activate_opt_options(self):
        '''
        The activation of some history matching settings, including well names, phase names, customized operators, scaling factors, etc.
        '''
        self.engine.objfun_prod_phase_rate = self.objfun_prod_phase_rate
        self.engine.prod_well_name = self.prod_well_name
        self.engine.prod_phase_name = self.prod_phase_name

        self.engine.objfun_inj_phase_rate = self.objfun_inj_phase_rate
        self.engine.inj_well_name = self.inj_well_name
        self.engine.inj_phase_name = self.inj_phase_name

        self.engine.objfun_BHP = self.objfun_BHP
        self.engine.BHP_well_name = self.BHP_well_name

        self.engine.objfun_well_tempr = self.objfun_well_tempr
        self.engine.well_tempr_name = self.well_tempr_name

        self.engine.objfun_temperature = self.objfun_temperature

        self.engine.objfun_customized_op = self.objfun_customized_op

        self.engine.objfun_saturation = self.objfun_saturation


        self.engine.scale_function_value = self.scale_function_value

        if type(self.modifier.modifiers[0]) == flux_multiplier_modifier:  # for MPFA
            self.col_idx = list(range(self.n_fm))
        else:
            # sorting the perforation index for adjoint gradient
            # `self.col_idx` will be passed into C++ `col_dT_du`
            self.col_idx_original = []
            self.col_idx_original = list(range(self.x_idx[-1]))
            self.col_idx = self.col_idx_original

            self.perforation_idx = []
            for i, w in enumerate(self.reservoir.wells):
                for p, per in enumerate(w.perforations):
                    self.perforation_idx.append(per[1])

            self.sort_perforation_idx = np.zeros(np.size(self.perforation_idx))
            for i in range(np.size(self.perforation_idx)):
                for j in range(np.size(self.perforation_idx)):
                    if self.perforation_idx[j] < self.perforation_idx[i]:
                        self.sort_perforation_idx[i] += 1

            temp_idx = self.sort_perforation_idx.astype(int)
            n_T_res = np.size(self.col_idx) - np.size(temp_idx)
            col_idx_well = self.col_idx[np.size(self.col_idx) - np.size(temp_idx):]
            for i, j in enumerate(temp_idx):
                self.col_idx[n_T_res + i] = col_idx_well[j]

    def set_objfun(self, objfun):
        '''
        The settings of the function name of objective function
        :param objfun: function name of objective function
        '''
        self.objfun = objfun

    def set_observation_data_report(self, data: pd.DataFrame):
        '''
        The settings of the measurement time (report time) point array "t_Q" for generating Dirac function
        :param data: observation data
        '''
        assert (type(data) == pd.core.frame.DataFrame)
        self.observation_data_report = data.set_index('time', drop=False)
        self.observation_last_date_report = data['time'][len(data['time']) - 1]
        self.t_Q = self.observation_data_report['time']

    def set_observation_data(self, data: pd.DataFrame):
        '''
        The settings of the simulation time point array "t_sim" for generating Dirac function
        :param data: observation data based on simulation time steps
        '''
        # verify pandas format
        assert (type(data) == pd.core.frame.DataFrame)
        self.observation_data = data.set_index('time', drop=False)
        self.observation_last_date = data['time'][len(data['time']) - 1]

    def observation_data_assembly(self):
        '''
        The assembly of the observation data for further use
        The observation data here can be synthetic data from high-resolution model
        or if you have field data, input your data as "self.observation_data"
        replace these code below with your field data, e.g. phase rate, BHP, saturation, etc.
        '''
        # ------------------------------------------------------------------------
        # ----------------------observation data assembly-------------------------
        # ------------------------------------------------------------------------

        # setting the random seeds for generating the noise to the observation data
        try:
            np.random.seed(int(self.task_id))
        except:
            np.random.seed(0)

        # add production phase rate data in objective function-----------------
        if self.objfun_prod_phase_rate:
            if np.size(np.array(self.phase_relative_density)) == 0:
                self.phase_relative_density = np.ones(np.size(self.prod_phase_name))

            self.engine.phase_relative_density = value_vector(self.phase_relative_density)

            self.std_gaussian_noise_prod = []
            # self.Q_temp = []
            self.Q_list_temp = []
            # Q_w_o_total = []
            self.prod_cov_mat_inv = []

            for n, well in enumerate(self.prod_well_name):
                # phase rate
                if self.opt_phase_rate:
                    gaussian_noise_list = []
                    rate_list = []
                    std_dev_list = []

                    # for i in range(self.physics.n_phases):
                    for i in range(np.size(self.prod_phase_name)):
                        gaussian_noise_list.append(0)
                        rate_list.append(0)
                        std_dev_list.append(0)

                    for p, phase in enumerate(self.prod_phase_name):
                        rate_string = well + " : " + phase + " rate (m3/day)"
                        rate_serie = self.observation_data.get(rate_string)

                        gaussian_noise_list[p] = np.random.randn(rate_serie.size)

                        if np.min(self.prod_rate_measurement_error * gaussian_noise_list[p]) < -1:
                            print('The added Gaussian noise changes the sign of rate!')

                        prod_rate_value = np.array(rate_serie.values)
                        if self.prod_rate_measurement_error != 0:
                            prod_rate_value[prod_rate_value < self.prod_error_lower_bound] = self.prod_error_lower_bound
                            prod_rate_value[prod_rate_value > self.prod_error_upper_bound] = self.prod_error_upper_bound

                        # rate_list[p] = rate_serie.values * self.phase_relative_density[p] \
                        #                * (1 + self.prod_rate_measurement_error * gaussian_noise_list[p])
                        rate_list[p] = rate_serie.values * self.phase_relative_density[p] \
                                       + prod_rate_value * self.prod_rate_measurement_error * gaussian_noise_list[p]


                        if self.prod_rate_measurement_error == 0:
                            std_dev_list[p] = np.ones(rate_serie.size)
                        else:
                            std_dev_list[p] = prod_rate_value * self.phase_relative_density[p] * self.prod_rate_measurement_error
                            # std_dev_list[p] = rate_serie.values * self.phase_relative_density[p] * self.prod_rate_measurement_error + 1e-6

                self.std_gaussian_noise_prod.append(gaussian_noise_list)
                self.Q_list_temp.append(rate_list)
                self.prod_cov_mat_inv.append(1 / np.array(std_dev_list))

        # # replace inf with 1
        # cov_temp = np.array(self.prod_cov_mat_inv)
        # cov_temp[np.isinf(cov_temp)] = 1
        # cc=cov_temp[4]

        # add injection phase rate data in objective function-----------------
        if self.objfun_inj_phase_rate:
            self.std_gaussian_noise_inj = []
            self.Q_inj_list_temp = []
            self.inj_cov_mat_inv = []

            for n, well in enumerate(self.inj_well_name):
                # phase rate
                gaussian_noise_list = []
                rate_list = []
                std_dev_list = []
                # for i in range(self.physics.n_phases):
                for i in range(np.size(self.inj_phase_name)):
                    gaussian_noise_list.append(0)
                    rate_list.append(0)
                    std_dev_list.append(0)

                for p, phase in enumerate(self.inj_phase_name):
                    rate_string = well + " : " + phase + " rate (m3/day)"
                    rate_serie = self.observation_data.get(rate_string)

                    gaussian_noise_list[p] = np.random.randn(rate_serie.size)

                    if np.min(self.inj_rate_measurement_error * gaussian_noise_list[p]) < -1:
                        print('The added Gaussian noise changes the sign of rate!')

                    inj_rate_value = np.array(rate_serie.values)
                    if self.inj_rate_measurement_error != 0:
                        inj_rate_value[inj_rate_value < self.inj_error_lower_bound] = self.inj_error_lower_bound
                        inj_rate_value[inj_rate_value > self.inj_error_upper_bound] = self.inj_error_upper_bound

                    # rate_list[p] = rate_serie.values * (1 + self.inj_rate_measurement_error * gaussian_noise_list[p])
                    rate_list[p] = rate_serie.values + inj_rate_value * self.inj_rate_measurement_error * gaussian_noise_list[p]

                    if self.inj_rate_measurement_error == 0:
                        std_dev_list[p] = np.ones(rate_serie.size)
                    else:
                        std_dev_list[p] = inj_rate_value * self.inj_rate_measurement_error

                self.std_gaussian_noise_inj.append(gaussian_noise_list)
                self.Q_inj_list_temp.append(rate_list)
                self.inj_cov_mat_inv.append(1 / np.array(std_dev_list))

        # add BHP data in objective function-----------------------
        if self.objfun_BHP:
            self.std_gaussian_noise_BHP = []
            self.BHP_list_temp = []
            self.BHP_cov_mat_inv = []

            for n, well in enumerate(self.BHP_well_name):
                std_dev_list = []
                BHP_string = well + " : BHP (bar)"
                BHP_serie = self.BHP_report_data.get(BHP_string)

                gaussian_noise_list = np.random.randn(BHP_serie.size)

                if np.min(self.BHP_measurement_error * gaussian_noise_list) < -1:
                    print('The added Gaussian noise changes the sign of BHP!')

                BHP_value = np.array(BHP_serie.values)
                if self.BHP_measurement_error != 0:
                    BHP_value[BHP_value < self.BHP_error_lower_bound] = self.BHP_error_lower_bound
                    BHP_value[BHP_value > self.BHP_error_upper_bound] = self.BHP_error_upper_bound

                # BHP_list = BHP_serie.values * (1 + self.BHP_measurement_error * gaussian_noise_list)
                BHP_list = BHP_serie.values + BHP_value * self.BHP_measurement_error * gaussian_noise_list

                if self.BHP_measurement_error == 0:
                    std_dev_list = np.ones(BHP_serie.size)
                else:
                    std_dev_list = BHP_value * self.BHP_measurement_error

                self.std_gaussian_noise_BHP.append(gaussian_noise_list)
                self.BHP_list_temp.append(BHP_list)
                self.BHP_cov_mat_inv.append(1 / np.array(std_dev_list))

        # add well temperature data in objective function-----------------------
        if self.objfun_well_tempr:
            self.std_gaussian_noise_well_tempr = []
            self.well_tempr_list_temp = []
            self.well_tempr_cov_mat_inv = []

            for n, well in enumerate(self.well_tempr_name):
                std_dev_list = []
                well_tempr_string = well + " : temperature (K)"
                well_tempr_serie = self.well_tempr_report_data.get(well_tempr_string)

                gaussian_noise_list = np.random.randn(well_tempr_serie.size)

                if np.min(self.well_tempr_measurement_error * gaussian_noise_list) < -1:
                    print('The added Gaussian noise changes the sign of well temperature!')

                well_tempr_value = np.array(well_tempr_serie.values)
                if self.well_tempr_measurement_error != 0:
                    well_tempr_value[well_tempr_value < self.well_tempr_error_lower_bound] = self.well_tempr_error_lower_bound
                    well_tempr_value[well_tempr_value > self.well_tempr_error_upper_bound] = self.well_tempr_error_upper_bound

                # well_tempr_list = well_tempr_serie.values * (1 + self.well_tempr_measurement_error * gaussian_noise_list)
                well_tempr_list = well_tempr_serie.values + well_tempr_value * self.well_tempr_measurement_error * gaussian_noise_list

                if self.well_tempr_measurement_error == 0:
                    std_dev_list = np.ones(well_tempr_serie.size)
                else:
                    std_dev_list = well_tempr_value * self.well_tempr_measurement_error

                self.std_gaussian_noise_well_tempr.append(gaussian_noise_list)
                self.well_tempr_list_temp.append(well_tempr_list)
                self.well_tempr_cov_mat_inv.append(1 / np.array(std_dev_list))

        # add temperature data in objective function-----------------------
        if self.objfun_temperature:
            gaussian_noise_list = np.random.randn(np.size(self.temperature_report_data, 0), np.size(self.temperature_report_data, 1))

            tempr_value = np.array(self.temperature_report_data)
            if self.temperature_measurement_error != 0:
                tempr_value[tempr_value < self.temperature_error_lower_bound] = self.temperature_error_lower_bound
                tempr_value[tempr_value > self.temperature_error_upper_bound] = self.temperature_error_upper_bound

            if self.temperature_measurement_error == 0:
                std_dev_list = np.ones(np.shape(self.temperature_report_data))
            else:
                std_dev_list = tempr_value * self.temperature_measurement_error

            self.std_gaussian_noise_temperature = gaussian_noise_list
            self.temperature_list = self.temperature_report_data + tempr_value * self.temperature_measurement_error * gaussian_noise_list
            self.temperature_cov_mat_inv = 1 / np.array(std_dev_list)

            # override the lists with EM data if they exist
            if len(self.temperature_EM_list) > 0:
                self.std_gaussian_noise_temperature = self.base_noise_temperature_EM
                self.temperature_list = self.temperature_EM_list
                self.temperature_cov_mat_inv = self.temperature_EM_cov_mat_inv
            
        # add customized operator in objective funciton-----------------------
        if self.objfun_customized_op:
            gaussian_noise_list = np.random.randn(np.size(self.customized_op_report_data, 0), np.size(self.customized_op_report_data, 1))
            
            op_value = np.array(self.customized_op_report_data)
            if self.customized_op_measurement_error != 0:
                op_value[op_value < self.customized_op_error_lower_bound] = self.customized_op_error_lower_bound
                op_value[op_value > self.customized_op_error_upper_bound] = self.customized_op_error_upper_bound

            if self.customized_op_measurement_error == 0:
                std_dev_list = np.ones(np.shape(self.customized_op_report_data))
            else:
                std_dev_list = op_value * self.customized_op_measurement_error

            self.std_gaussian_noise_customized_op = gaussian_noise_list
            self.customized_op_list = self.customized_op_report_data + op_value * self.customized_op_measurement_error * gaussian_noise_list
            self.customized_op_cov_mat_inv = 1 / np.array(std_dev_list)
            


        # add saturation data in objective function-----------------
        if self.objfun_saturation:
            pass


        if self.misfit_watch:  # IMPORTANT!!!  You need to specify which misfit term is going to be watched below
            # gaussian_noise_list = np.random.randn(np.size(self.temperature_report_data, 0), np.size(self.temperature_report_data, 1))
            # 
            # tempr_value = np.array(self.temperature_report_data)
            # if self.temperature_measurement_error != 0:
            #     tempr_value[tempr_value < self.temperature_error_lower_bound] = self.temperature_error_lower_bound
            #     tempr_value[tempr_value > self.temperature_error_upper_bound] = self.temperature_error_upper_bound
            # 
            # if self.temperature_measurement_error == 0:
            #     std_dev_list = np.ones(np.shape(self.temperature_report_data))
            # else:
            #     std_dev_list = tempr_value * self.temperature_measurement_error
            # 
            # self.std_gaussian_noise_temperature = gaussian_noise_list
            # self.temperature_list = self.temperature_report_data + tempr_value * self.temperature_measurement_error * gaussian_noise_list
            # self.temperature_cov_mat_inv = 1 / np.array(std_dev_list)
            # 
            # # override the lists with EM data if they exist
            # if len(self.temperature_EM_list) > 0:
            #     self.std_gaussian_noise_temperature = self.base_noise_temperature_EM
            #     self.temperature_list = self.temperature_EM_list
            #     self.temperature_cov_mat_inv = self.temperature_EM_cov_mat_inv
                
                
            self.std_gaussian_noise_well_tempr = []
            self.well_tempr_list_temp = []
            self.well_tempr_cov_mat_inv = []

            for n, well in enumerate(self.well_tempr_name):
                std_dev_list = []
                well_tempr_string = well + " : temperature (K)"
                well_tempr_serie = self.well_tempr_report_data.get(well_tempr_string)

                gaussian_noise_list = np.random.randn(well_tempr_serie.size)

                if np.min(self.well_tempr_measurement_error * gaussian_noise_list) < -1:
                    print('The added Gaussian noise changes the sign of well temperature!')

                well_tempr_value = np.array(well_tempr_serie.values)
                if self.well_tempr_measurement_error != 0:
                    well_tempr_value[well_tempr_value < self.well_tempr_error_lower_bound] = self.well_tempr_error_lower_bound
                    well_tempr_value[well_tempr_value > self.well_tempr_error_upper_bound] = self.well_tempr_error_upper_bound

                # well_tempr_list = well_tempr_serie.values * (1 + self.well_tempr_measurement_error * gaussian_noise_list)
                well_tempr_list = well_tempr_serie.values + well_tempr_value * self.well_tempr_measurement_error * gaussian_noise_list

                if self.well_tempr_measurement_error == 0:
                    std_dev_list = np.ones(well_tempr_serie.size)
                else:
                    std_dev_list = well_tempr_value * self.well_tempr_measurement_error

                self.std_gaussian_noise_well_tempr.append(gaussian_noise_list)
                self.well_tempr_list_temp.append(well_tempr_list)
                self.well_tempr_cov_mat_inv.append(1 / np.array(std_dev_list))



        # dirac measurement funtion-----------------
        if np.size(self.observation_data['time']) == np.size(self.observation_data_report['time']):  # report
            if self.objfun_prod_phase_rate:
                self.obs_Q = np.array(self.Q_list_temp)
            else:
                self.obs_Q = 0

            if self.objfun_inj_phase_rate:
                self.obs_Q_inj = np.array(self.Q_inj_list_temp)
            else:
                self.obs_Q_inj = 0

            if self.objfun_BHP:
                self.obs_BHP = np.array(self.BHP_list_temp)
            else:
                self.obs_BHP = 0

            if self.objfun_well_tempr:
                self.obs_well_tempr = np.array(self.well_tempr_list_temp)
            else:
                self.obs_well_tempr = 0

            if self.objfun_temperature:
                self.obs_TEMPERATURE = np.array(self.temperature_list)
            else:
                self.obs_TEMPERATURE = 0
                
            if self.objfun_customized_op:
                self.obs_CUSTOMIZED_OP = np.array(self.customized_op_list)
            else:
                self.obs_CUSTOMIZED_OP = 0

            if self.objfun_saturation:
                pass
            else:
                self.obs_saturation = 0

        else:  # sim
            t_sim = self.observation_data['time']
            self.dirac_vec = np.zeros(np.size(t_sim))
            for i, ts in enumerate(t_sim):
                for tr in self.t_Q:
                    if ts == tr:
                        self.dirac_vec[i] = 1

            self.engine.dirac_vec = value_vector(self.dirac_vec)


            if self.objfun_prod_phase_rate:
                self.obs_Q = np.array(self.Q_list_temp)[:, self.dirac_vec.astype('bool')]
            else:
                self.obs_Q = 0

            if self.objfun_inj_phase_rate:
                self.obs_Q_inj = np.array(self.Q_inj_list_temp)[:, self.dirac_vec.astype('bool')]
            else:
                self.obs_Q_inj = 0

            if self.objfun_BHP:
                self.obs_BHP = np.array(self.BHP_list_temp)[:, self.dirac_vec.astype('bool')]
            else:
                self.obs_BHP = 0

            if self.objfun_well_tempr:
                self.obs_well_tempr = np.array(self.well_tempr_list_temp)[:, self.dirac_vec.astype('bool')]
            else:
                self.obs_well_tempr = 0

            if self.objfun_temperature:
                self.obs_TEMPERATURE = np.array(self.temperature_list)[:, self.dirac_vec.astype('bool')]
            else:
                self.obs_TEMPERATURE = 0
                
            if self.objfun_customized_op:
                self.obs_CUSTOMIZED_OP = np.array(self.customized_op_list)[:, self.dirac_vec.astype('bool')]
            else:
                self.obs_CUSTOMIZED_OP = 0

            if self.objfun_saturation:
                pass
            else:
                self.obs_saturation = 0



        if self.misfit_watch:  # IMPORTANT!!!  You need to specify which misfit term is going to be watched below
            # self.obs_TEMPERATURE = np.array(self.temperature_list)

            self.obs_well_tempr = np.array(self.well_tempr_list_temp)


        # -------------------------------------------------------------------------------------------------------------
        # --------set the observation data, the inverse of covariance matrix, and the weights matrix into C++ engine---
        # -------------------------------------------------------------------------------------------------------------
        # cov_mat_inv here is actually the inverse of standard deviation (i.e. 1/sigma), but this will be converted to
        # (1/sigma)^2 in the Python and C++ adjoint engine

        # production phase rate ------------------------
        if self.objfun_prod_phase_rate:
            if np.size(self.prod_weights) == 0:
                self.prod_weights = np.ones(np.shape(self.obs_Q))

            # phase rate
            if self.opt_phase_rate:
                # self.engine.clear_Q()
                # self.engine.prod_phase_name = self.prod_phase_name

                for w in range(np.size(self.obs_Q, 0)):
                    self.engine.clear_Q_p()
                    self.engine.clear_cov_prod_p()
                    self.engine.clear_prod_wei_p()
                    for p in range(np.size(self.obs_Q, 1)):
                        self.engine.add_value_to_Q_p(value_vector(self.obs_Q[w][p]))
                        self.engine.add_value_to_cov_prod_p(value_vector(self.prod_cov_mat_inv[w][p]))
                        self.engine.add_value_to_prod_wei_p(value_vector(self.prod_weights[w][p]))
                    self.engine.push_back_to_Q_all()
                    self.engine.push_back_to_cov_prod_all()
                    self.engine.push_back_to_prod_wei_all()

        # injection phase rate ------------------------
        if self.objfun_inj_phase_rate:
            if np.size(self.inj_weights) == 0:
                self.inj_weights = np.ones(np.shape(self.obs_Q_inj))

            for w in range(np.size(self.obs_Q_inj, 0)):
                self.engine.clear_Q_inj_p()
                self.engine.clear_cov_inj_p()
                self.engine.clear_inj_wei_p()
                for p in range(np.size(self.obs_Q_inj, 1)):
                    self.engine.add_value_to_Q_inj_p(value_vector(self.obs_Q_inj[w][p]))
                    self.engine.add_value_to_cov_inj_p(value_vector(self.inj_cov_mat_inv[w][p]))
                    self.engine.add_value_to_inj_wei_p(value_vector(self.inj_weights[w][p]))
                self.engine.push_back_to_Q_inj_all()
                self.engine.push_back_to_cov_inj_all()
                self.engine.push_back_to_inj_wei_all()

        # BHP ------------------------------------------------
        if self.objfun_BHP:
            if np.size(self.BHP_weights) == 0:
                # self.BHP_weights = np.ones(np.shape(self.BHP_weights))
                self.BHP_weights = np.ones(np.shape(self.obs_BHP))

            for w in range(np.size(self.obs_BHP, 0)):
                self.engine.push_back_to_BHP_all(value_vector(self.obs_BHP[w]))
                self.engine.push_back_to_cov_BHP_all(value_vector(self.BHP_cov_mat_inv[w]))
                self.engine.push_back_to_BHP_wei_all(value_vector(self.BHP_weights[w]))

        # well temperature ------------------------------------
        if self.objfun_well_tempr:
            if np.size(self.well_tempr_weights) == 0:
                self.well_tempr_weights = np.ones(np.shape(self.obs_well_tempr))

            for w in range(np.size(self.obs_well_tempr, 0)):
                self.engine.push_back_to_well_tempr_all(value_vector(self.obs_well_tempr[w]))
                self.engine.push_back_to_cov_well_tempr_all(value_vector(self.well_tempr_cov_mat_inv[w]))
                self.engine.push_back_to_well_tempr_wei_all(value_vector(self.well_tempr_weights[w]))


        # temperature -----------------------------------------
        if self.objfun_temperature:
            if np.size(self.temperature_weights) == 0:
                self.temperature_weights = np.ones(np.shape(self.obs_TEMPERATURE))

            for t in range(np.size(self.obs_TEMPERATURE, 0)):
                self.engine.push_back_to_temperature_all(value_vector(self.obs_TEMPERATURE[t]))
                self.engine.push_back_to_cov_temperature_all(value_vector(self.temperature_cov_mat_inv[t]))
                self.engine.push_back_to_temperature_wei_all(value_vector(self.temperature_weights[t]))
                
        # customized operator -----------------------------------------
        if self.objfun_customized_op:
            if np.size(self.customized_op_weights) == 0:
                self.customized_op_weights = np.ones(np.shape(self.obs_CUSTOMIZED_OP))

            for t in range(np.size(self.obs_CUSTOMIZED_OP, 0)):
                self.engine.push_back_to_customized_op_all(value_vector(self.obs_CUSTOMIZED_OP[t]))
                self.engine.push_back_to_cov_customized_op_all(value_vector(self.customized_op_cov_mat_inv[t]))
                self.engine.push_back_to_customized_op_wei_all(value_vector(self.customized_op_weights[t]))

            if len(self.binary_array) > 0:
                self.engine.threshold = self.threshold
                for t in range(np.size(self.binary_array, 0)):
                    self.engine.push_back_to_binary_all(value_vector(self.binary_array[t]))

        # saturation-----------------------------------
        if self.objfun_saturation:
            pass

        # ------------------------------------------------------------------------
        # -----------------------covariance matrix--------------------------------
        # ------------------------------------------------------------------------
        # self.cov_mat_inv_diagnal = []
        # for i in range(np.size(self.cov_mat_inv, 0)):
        #     self.cov_mat_inv_diagnal.append(self.cov_mat_inv[i][i])
        # self.engine.cov_mat_inv = value_vector(self.cov_mat_inv_diagnal)



    def objfun_assembly(self) -> float:
        '''
        The assembly of the objective function
        The observation data is assembled by `self.observation_data_assembly()`
        This function is first getting the model response, and then compute the objective function value
        '''

        # only need to prepare observation data once
        if self.prepare_obs_data:
            self.observation_data_assembly()
            self.prepare_obs_data = False

        # get the model response
        response = pd.DataFrame.from_dict(self.engine.time_data)
        try:
            t_sim = response['time']
        except KeyError:
            print('Forward simulation failed!')
            sys.exit()

        self.dirac_vec = np.zeros(np.size(t_sim))
        for i, ts in enumerate(t_sim):
            for tr in self.t_Q:
                if ts == tr:
                    self.dirac_vec[i] = 1

        self.engine.dirac_vec = value_vector(self.dirac_vec)

        self.objfun_dict = {}
        self.objfun_list = []

        self.fval = 0
        # add production phase rate data in objective function----------------------------------------
        if self.objfun_prod_phase_rate:
            q_separate = []

            for n, well in enumerate(self.prod_well_name):
                q_w_o = 0

                # phase rate
                if self.opt_phase_rate:
                    rate_list = []
                    # for i in range(self.physics.n_phases):
                    for i in range(np.size(self.prod_phase_name)):
                        rate_list.append(0)

                    for p, phase in enumerate(self.prod_phase_name):
                        rate_string = well + " : " + phase + " rate (m3/day)"
                        rate_list[p] = -response.get(rate_string).values * self.phase_relative_density[p]
                        if phase == 'oil':
                            q_w_o = -response.get(rate_string).values

                q_separate.append(rate_list)

            # compute fval in a backward direction to make it similar with adjoint backward integration in C++ engine
            TotStep = np.size(t_sim)
            size_Q = np.size(self.obs_Q, 2)

            # idx_sim_ts is the index of the simulation timestep. It corresponds to time_data
            # idx_obs_ts is the index of the observation timestep. It corresponds to time_data_report
            # the conflict between idx_sim_ts and idx_obs_ts is solved by Dirac function
            # see eq.(17), Tian et al. 2015  https://doi.org/10.1016/j.petrol.2021.109911
            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_Q - 1

            q_Q = []
            q_Q.append((np.array(q_separate)[:, :, idx_sim_ts] - np.array(self.obs_Q)[:, :, idx_obs_ts])
                       * self.dirac_vec[idx_sim_ts]
                       * np.array(self.prod_cov_mat_inv)[:, :, idx_obs_ts]
                       * np.array(self.prod_weights[:, :, idx_obs_ts]) ** 0.5)

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1

                q_Q.append((np.array(q_separate)[:, :, idx_sim_ts] - np.array(self.obs_Q)[:, :, idx_obs_ts])
                           * self.dirac_vec[idx_sim_ts]
                           * np.array(self.prod_cov_mat_inv)[:, :, idx_obs_ts]
                           * np.array(self.prod_weights[:, :, idx_obs_ts]) ** 0.5)

            q_Q_clean = np.array(q_Q)[np.flipud(self.dirac_vec).astype('bool'), :].T

            q_Q_temp = []
            for q_Q_well in q_Q_clean:
                # q_Q_temp.append(q_Q_well * q_Q_well * self.cov_mat_inv_diagnal)
                q_Q_temp.append(q_Q_well * q_Q_well)
            self.fval += np.sum(q_Q_temp)
            # self.fval += np.sum(q_Q_clean ** 2)
            self.objfun_dict['prod_phase_rate'] = np.sum(q_Q_temp)
            self.objfun_list.append(np.sum(q_Q_temp))


        # add injection phase rate data in objective function----------------------------
        if self.objfun_inj_phase_rate:
            q_inj_separate = []

            for n, well in enumerate(self.inj_well_name):
                # phase rate
                rate_list = []
                # for i in range(self.physics.n_phases):
                for i in range(np.size(self.inj_phase_name)):
                    rate_list.append(0)

                for p, phase in enumerate(self.inj_phase_name):
                    rate_string = well + " : " + phase + " rate (m3/day)"
                    rate_list[p] = response.get(rate_string).values

                q_inj_separate.append(rate_list)

            # compute fval in a backward direction to make it similar with adjoint backward integration in C++ engine
            TotStep = np.size(t_sim)
            size_Q_inj = np.size(self.obs_Q_inj, 2)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_Q_inj - 1

            q_Q_inj = []
            q_Q_inj.append((np.array(q_inj_separate)[:, :, idx_sim_ts] - np.array(self.obs_Q_inj)[:, :, idx_obs_ts])
                           * self.dirac_vec[idx_sim_ts]
                           * np.array(self.inj_cov_mat_inv)[:, :, idx_obs_ts]
                           * np.array(self.inj_weights[:, :, idx_obs_ts]) ** 0.5)

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1
                q_Q_inj.append((np.array(q_inj_separate)[:, :, idx_sim_ts] - np.array(self.obs_Q_inj)[:, :, idx_obs_ts])
                               * self.dirac_vec[idx_sim_ts]
                               * np.array(self.inj_cov_mat_inv)[:, :, idx_obs_ts]
                               * np.array(self.inj_weights[:, :, idx_obs_ts]) ** 0.5)

            q_Q_inj_clean = np.array(q_Q_inj)[np.flipud(self.dirac_vec).astype('bool'), :].T

            q_Q_inj_temp = []
            for q_Q_inj_well in q_Q_inj_clean:
                # q_Q_inj_temp.append(q_Q_inj_well * q_Q_inj_well * self.cov_mat_inv_diagnal)
                q_Q_inj_temp.append(q_Q_inj_well * q_Q_inj_well)
            self.fval += np.sum(q_Q_inj_temp)
            # self.fval += np.sum(q_Q_inj_clean ** 2)
            self.objfun_dict['inj_phase_rate'] = np.sum(q_Q_inj_temp)
            self.objfun_list.append(np.sum(q_Q_inj_temp))


        # add BHP data in objective function---------------------------------------------
        if self.objfun_BHP:
            bhp_separate = []

            for n, well in enumerate(self.BHP_well_name):
                BHP_string = well + " : BHP (bar)"
                bhp_separate.append(response.get(BHP_string).values)

            # compute fval in a backward direction to make it similar with adjoint backward integration in C++ engine
            TotStep = np.size(t_sim)
            size_BHP = np.size(self.obs_BHP, 1)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_BHP - 1

            bhp_BHP = []
            bhp_BHP.append((np.array(bhp_separate)[:, idx_sim_ts] - np.array(self.obs_BHP)[:, idx_obs_ts])
                           * self.dirac_vec[idx_sim_ts]
                           * np.array(self.BHP_cov_mat_inv)[:, idx_obs_ts]
                           * np.array(self.BHP_weights[:, idx_obs_ts]) ** 0.5)

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1
                bhp_BHP.append((np.array(bhp_separate)[:, idx_sim_ts] - np.array(self.obs_BHP)[:, idx_obs_ts])
                               * self.dirac_vec[idx_sim_ts]
                               * np.array(self.BHP_cov_mat_inv)[:, idx_obs_ts]
                               * np.array(self.BHP_weights[:, idx_obs_ts]) ** 0.5)

            bhp_BHP_clean = np.array(bhp_BHP)[np.flipud(self.dirac_vec).astype('bool'), :].T

            bhp_BHP_temp = []
            for bhp_BHP_well in bhp_BHP_clean:
                # bhp_BHP_temp.append(bhp_BHP_well * bhp_BHP_well * self.cov_mat_inv_diagnal)
                bhp_BHP_temp.append(bhp_BHP_well * bhp_BHP_well)
            self.fval += np.sum(bhp_BHP_temp)
            # self.fval += np.sum(bhp_BHP_clean ** 2)
            self.objfun_dict['BHP'] = np.sum(bhp_BHP_temp)
            self.objfun_list.append(np.sum(bhp_BHP_temp))


        # add well temperature in objective function---------------------------------------------
        if self.objfun_well_tempr:
            well_tempr_separate = []

            for n, well in enumerate(self.well_tempr_name):
                well_tempr_string = well + " : temperature (K)"
                well_tempr_separate.append(response.get(well_tempr_string).values)

            # compute fval in a backward direction to make it similar with adjoint backward integration in C++ engine
            TotStep = np.size(t_sim)
            size_well_tempr = np.size(self.obs_well_tempr, 1)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_well_tempr - 1

            wt_WT = []
            wt_WT.append((np.array(well_tempr_separate)[:, idx_sim_ts] - np.array(self.obs_well_tempr)[:, idx_obs_ts])
                           * self.dirac_vec[idx_sim_ts]
                           * np.array(self.well_tempr_cov_mat_inv)[:, idx_obs_ts]
                           * np.array(self.well_tempr_weights[:, idx_obs_ts]) ** 0.5)

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1
                wt_WT.append((np.array(well_tempr_separate)[:, idx_sim_ts] - np.array(self.obs_well_tempr)[:, idx_obs_ts])
                               * self.dirac_vec[idx_sim_ts]
                               * np.array(self.well_tempr_cov_mat_inv)[:, idx_obs_ts]
                               * np.array(self.well_tempr_weights[:, idx_obs_ts]) ** 0.5)

            wt_WT_clean = np.array(wt_WT)[np.flipud(self.dirac_vec).astype('bool'), :].T

            wt_WT_temp = []
            for wt_WT_well in wt_WT_clean:
                wt_WT_temp.append(wt_WT_well * wt_WT_well)
            self.fval += np.sum(wt_WT_temp)
            self.objfun_dict['well_tempr'] = np.sum(wt_WT_temp)
            self.objfun_list.append(np.sum(wt_WT_temp))


        # add temperature data in objective function---------------------------------------------
        if self.objfun_temperature:
            # cc=np.array(self.engine.time_data_customized)
            temperature_separate = np.array(self.engine.time_data_customized)[:, 0:self.reservoir.mesh.n_res_blocks]

            # compute fval in a backward direction to make it similar with adjoint backward integration in C++ engine
            TotStep = np.size(t_sim)
            size_temperature = np.size(self.obs_TEMPERATURE, 0)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_temperature - 1

            tempr_TEMPR = []
            tempr_TEMPR.append((np.array(temperature_separate)[idx_sim_ts, :] - np.array(self.obs_TEMPERATURE)[idx_obs_ts, :])
                               * self.dirac_vec[idx_sim_ts]
                               * self.temperature_cov_mat_inv[idx_obs_ts, :]
                               * self.temperature_weights[idx_obs_ts, :] ** 0.5)

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1
                tempr_TEMPR.append((np.array(temperature_separate)[idx_sim_ts, :] - np.array(self.obs_TEMPERATURE)[idx_obs_ts, :])
                                   * self.dirac_vec[idx_sim_ts]
                                   * self.temperature_cov_mat_inv[idx_obs_ts, :]
                                   * self.temperature_weights[idx_obs_ts, :] ** 0.5)

            tempr_TEMPR_clean = np.array(tempr_TEMPR)[np.flipud(self.dirac_vec).astype('bool'), :].T

            tempr_TEMPR_temp = []
            for tempr_TEMPR_t in tempr_TEMPR_clean:
                tempr_TEMPR_temp.append(tempr_TEMPR_t * tempr_TEMPR_t)
            self.fval += np.sum(tempr_TEMPR_temp)
            # self.fval += np.sum(tempr_TEMPR_clean ** 2)
            self.objfun_dict['tempr_distr'] = np.sum(tempr_TEMPR_temp)
            self.objfun_list.append(np.sum(tempr_TEMPR_temp))


        # add customized operator data in objective function---------------------------------------------
        if self.objfun_customized_op:
            if len(self.binary_array) == 0:
                # cc=np.array(self.engine.time_data_customized)
                customized_op_separate = np.array(self.engine.time_data_customized)[:, 0:self.reservoir.mesh.n_res_blocks]

                # compute fval in a backward direction to make it similar with adjoint backward integration in C++ engine
                TotStep = np.size(t_sim)
                size_customized_op = np.size(self.obs_CUSTOMIZED_OP, 0)

                idx_sim_ts = TotStep - 1
                idx_obs_ts = size_customized_op - 1

                op_OP = []
                op_OP.append((np.array(customized_op_separate)[idx_sim_ts, :] - np.array(self.obs_CUSTOMIZED_OP)[idx_obs_ts, :])
                                   * self.dirac_vec[idx_sim_ts]
                                   * self.customized_op_cov_mat_inv[idx_obs_ts, :]
                                   * self.customized_op_weights[idx_obs_ts, :] ** 0.5)

                for idx_sim_ts in range(TotStep - 2, -1, -1):
                    if self.dirac_vec[idx_sim_ts] == 1:
                        idx_obs_ts -= 1
                    op_OP.append((np.array(customized_op_separate)[idx_sim_ts, :] - np.array(self.obs_CUSTOMIZED_OP)[idx_obs_ts, :])
                                       * self.dirac_vec[idx_sim_ts]
                                       * self.customized_op_cov_mat_inv[idx_obs_ts, :]
                                       * self.customized_op_weights[idx_obs_ts, :] ** 0.5)

                op_OP_clean = np.array(op_OP)[np.flipud(self.dirac_vec).astype('bool'), :].T

                op_OP_temp = []
                for op_OP_t in op_OP_clean:
                    op_OP_temp.append(op_OP_t * op_OP_t)
                self.fval += np.sum(op_OP_temp)
                # self.fval += np.sum(tempr_TEMPR_clean ** 2)
                self.objfun_dict['customized_op'] = np.sum(op_OP_temp)
                self.objfun_list.append(np.sum(op_OP_temp))

            else:
                # hinge loss function---------------------------------
                # cc=np.array(self.engine.time_data_customized)
                customized_op_separate = np.array(self.engine.time_data_customized)[:, 0:self.reservoir.mesh.n_res_blocks]

                # compute fval in a backward direction to make it similar with adjoint backward integration in C++ engine
                TotStep = np.size(t_sim)
                size_customized_op = np.size(self.obs_CUSTOMIZED_OP, 0)

                idx_sim_ts = TotStep - 1
                idx_obs_ts = size_customized_op - 1

                op_OP = []

                op_OP.append((np.array(customized_op_separate)[idx_sim_ts, :] - np.array(self.obs_CUSTOMIZED_OP)[idx_obs_ts, :])
                             * self.dirac_vec[idx_sim_ts]
                             * self.customized_op_cov_mat_inv[idx_obs_ts, :]
                             * self.customized_op_weights[idx_obs_ts, :] ** 0.5)

                hinge_coeff = 1000 * np.ones(self.obs_CUSTOMIZED_OP.shape)

                cus_op = np.array(customized_op_separate)[idx_sim_ts, :]
                for b, co in enumerate(cus_op):
                    if self.binary_array[idx_obs_ts][b] == 1:
                        if co > self.threshold:
                            hinge_coeff[idx_obs_ts, b] = 0
                        else:
                            hinge_coeff[idx_obs_ts, b] = 1
                    else:
                        if co < self.threshold:
                            hinge_coeff[idx_obs_ts, b] = 0
                        else:
                            hinge_coeff[idx_obs_ts, b] = 1

                for idx_sim_ts in range(TotStep - 2, -1, -1):
                    if self.dirac_vec[idx_sim_ts] == 1:
                        idx_obs_ts -= 1
                        cus_op = np.array(customized_op_separate)[idx_sim_ts, :]
                        for b, co in enumerate(cus_op):
                            if self.binary_array[idx_obs_ts][b] == 1:
                                if co > self.threshold:
                                    hinge_coeff[idx_obs_ts, b] = 0
                                else:
                                    hinge_coeff[idx_obs_ts, b] = 1
                            else:
                                if co < self.threshold:
                                    hinge_coeff[idx_obs_ts, b] = 0
                                else:
                                    hinge_coeff[idx_obs_ts, b] = 1

                    op_OP.append((np.array(customized_op_separate)[idx_sim_ts, :] - np.array(self.obs_CUSTOMIZED_OP)[idx_obs_ts, :])
                                 * self.dirac_vec[idx_sim_ts]
                                 * self.customized_op_cov_mat_inv[idx_obs_ts, :]
                                 * self.customized_op_weights[idx_obs_ts, :] ** 0.5)

                op_OP_clean = np.array(op_OP)[np.flipud(self.dirac_vec).astype('bool'), :].T

                op_OP_temp = []
                for idx_b, op_OP_t in enumerate(op_OP_clean):
                    op_OP_temp.append(op_OP_t * op_OP_t * hinge_coeff[:, idx_b])
                self.fval += np.sum(op_OP_temp)
                # self.fval += np.sum(tempr_TEMPR_clean ** 2)
                self.objfun_dict['customized_op'] = np.sum(op_OP_temp)
                self.objfun_list.append(np.sum(op_OP_temp))
                if self.save_error:
                    np.save('%s_%s.npy' % (self.label, self.job_id), np.array(op_OP_temp))


        # add saturation data in objective function---------------------------------------------
        if self.objfun_saturation:
            sat_SAT = 0.0  #todo......
            self.fval += sat_SAT



        if self.misfit_watch:  # IMPORTANT!!!  You need to specify which misfit term is going to be watched below
            # # cc=np.array(self.engine.time_data_customized)
            # temperature_separate = np.array(self.engine.time_data_customized)[:, 0:self.reservoir.mesh.n_res_blocks]
            # 
            # # compute fval in a backward direction to make it similar with adjoint backward integration in C++ engine
            # TotStep = np.size(t_sim)
            # size_temperature = np.size(self.obs_TEMPERATURE, 0)
            # 
            # idx_sim_ts = TotStep - 1
            # idx_obs_ts = size_temperature - 1
            # 
            # tempr_TEMPR = []
            # tempr_TEMPR.append(
            #     (np.array(temperature_separate)[idx_sim_ts, :] - np.array(self.obs_TEMPERATURE)[idx_obs_ts, :])
            #     * self.dirac_vec[idx_sim_ts]
            #     * self.temperature_cov_mat_inv[idx_obs_ts, :]
            #     * self.temperature_weights[idx_obs_ts, :] ** 0.5)
            # 
            # for idx_sim_ts in range(TotStep - 2, -1, -1):
            #     if self.dirac_vec[idx_sim_ts] == 1:
            #         idx_obs_ts -= 1
            #     tempr_TEMPR.append(
            #         (np.array(temperature_separate)[idx_sim_ts, :] - np.array(self.obs_TEMPERATURE)[idx_obs_ts, :])
            #         * self.dirac_vec[idx_sim_ts]
            #         * self.temperature_cov_mat_inv[idx_obs_ts, :]
            #         * self.temperature_weights[idx_obs_ts, :] ** 0.5)
            # 
            # tempr_TEMPR_clean = np.array(tempr_TEMPR)[np.flipud(self.dirac_vec).astype('bool'), :].T
            # 
            # tempr_TEMPR_temp = []
            # for tempr_TEMPR_t in tempr_TEMPR_clean:
            #     tempr_TEMPR_temp.append(tempr_TEMPR_t * tempr_TEMPR_t)
            # self.misfit_value = np.sum(tempr_TEMPR_temp)



            well_tempr_separate = []

            for n, well in enumerate(self.well_tempr_name):
                well_tempr_string = well + " : temperature (K)"
                well_tempr_separate.append(response.get(well_tempr_string).values)

            # compute fval in a backward direction to make it similar with adjoint backward integration in C++ engine
            TotStep = np.size(t_sim)
            size_well_tempr = np.size(self.obs_well_tempr, 1)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_well_tempr - 1

            wt_WT = []
            wt_WT.append((np.array(well_tempr_separate)[:, idx_sim_ts] - np.array(self.obs_well_tempr)[:, idx_obs_ts])
                         * self.dirac_vec[idx_sim_ts]
                         * np.array(self.well_tempr_cov_mat_inv)[:, idx_obs_ts]
                         * np.array(self.well_tempr_weights[:, idx_obs_ts]) ** 0.5)

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1
                wt_WT.append(
                    (np.array(well_tempr_separate)[:, idx_sim_ts] - np.array(self.obs_well_tempr)[:, idx_obs_ts])
                    * self.dirac_vec[idx_sim_ts]
                    * np.array(self.well_tempr_cov_mat_inv)[:, idx_obs_ts]
                    * np.array(self.well_tempr_weights[:, idx_obs_ts]) ** 0.5)

            wt_WT_clean = np.array(wt_WT)[np.flipud(self.dirac_vec).astype('bool'), :].T

            wt_WT_temp = []
            for wt_WT_well in wt_WT_clean:
                wt_WT_temp.append(wt_WT_well * wt_WT_well)
            self.misfit_value = np.sum(wt_WT_temp)




        self.objfun_all.append(self.objfun_list)
        self.fval_temp = self.fval * self.scale_function_value


        # Regularization term--------------------------------------
        self.response_diff = self.fval_temp

        if self.regularization:
            if self.re_parameterized_PCA:
                self.ksi = self.x_temp[0:self.space_dim] * self.norm_ksi
                
                # R = self.alpha * self.ksi.dot(self.ksi.transpose())
                
                ksi_diff = self.ksi - self.ksi_ref * self.norm_ksi
                R = self.alpha * ksi_diff.dot(ksi_diff.transpose())

                print('misfit: %s' % self.fval_temp)
                print('R: %s' % R)
                self.fval_temp = self.fval_temp + R
            else:
                u = self.x_temp[0:np.size(self.x_ref)] * self.modifier.modifiers[0].norms
                self.x_diff = np.array(u) - np.array(self.x_ref)
                if self.read_Cm_inv_vector:
                    R = 0
                    for idx_Cv in range(np.size(self.x_ref)):
                        R = R + np.sum(self.x_diff * np.load(self.Cv_path + "/%s_Cv.npy" % idx_Cv)) * self.x_diff[idx_Cv]
                    R = R * self.alpha
                else:
                    R = self.alpha * np.array([self.x_diff]).dot(self.C_m_inv).dot(np.array([self.x_diff]).transpose())
                # self.fval_temp = self.fval_temp + R[0][0]

                print('misfit: %s' % self.fval_temp)
                print('R: %s' % abs(float(R)))
                self.fval_temp = self.fval_temp + abs(float(R))


        if self.regularization_diagonal:  # there are only elements on the diagonal of the covariance matrix
            u = self.x_temp[0:np.size(self.x_ref)] * self.modifier.modifiers[0].norms
            # u = self.x_temp[0:np.size(self.x_ref)]

            self.x_diff = np.array([u]) - np.array([self.x_ref])
            R = self.alpha * np.sum((self.x_diff ** 2) * self.Cm_inv_diagonal)

            print('misfit: %s' % self.fval_temp)
            print('R: %s' % R)
            self.fval_temp = self.fval_temp + R


        print(' fval: %s' % self.fval_temp)
        return self.fval_temp




    def fval_nonlinear_FDM(self, x_eps: np.array) -> float:
        '''
        The preparation of the model response based on Finite Diffirence Method (FDM)
        :param x_esp: the control variables with perturbation epsilon when computing numerical gradient based on FDM
        :return: objective function value
        '''

        # print("start calculate the objective function values")
        # calculate the gradient of nonlinear modifiers using finite difference method
        # 1. Update model
        # self.modifier.set_x(self, x)
        self.modifier.set_x_by_du_dT(self, x_eps)

        # 2. Reset
        self.set_boundary_conditions()
        self.reset()
        self.run()

        fval = self.objfun()

        return fval


    def collect_result(self, result: float):
        self.result_list.append(result)




    def grad_adjoint_method_all(self, x: np.array) -> np.array:
        '''
        The preparation of the adjoint gradient for transmissibility and well index
        :param x: control variables
        :return: the adjoint gradient for transmissibility and well index
        '''
        # --------------------------gradients for transmissibility and well index--------------------------------------
        self.ad_grad_time -= time.time()

        grad_linear = np.zeros(self.x_idx[2])

        self.engine.n_control_vars = np.size(x)
        self.engine.col_dT_du = index_vector(self.col_idx)

        grad_linear = np.zeros(np.size(x))

        try:
            self.engine.calc_adjoint_gradient_dirac_all()  # call the adjoint calculation function from C++

            temp_grad = np.array(self.engine.derivatives, copy=False)  # get the adjoint gradinet from C++

            grad_linear = self.modifier.set_grad(temp_grad, self.x_idx)  # set the gradient to modifiers
        except IndexError:
            print("Adjoint fails! Discard this iteration step!")
            grad_linear = self.grad_old

        self.grad_old = grad_linear

        self.ad_grad_time += time.time()
        self.n_grad_calc += 1
        print(', %f s/lin_adj' % (self.ad_grad_time / self.n_grad_calc), end='', flush=True)

        # ---calculate the gradient of nonlinear parameters using finite difference method-----------------------------
        self.nonlinear_grad_time -= time.time()
        x_old = x
        fval_old = self.fval_temp
        grad_nonlinear = []
        if np.size(self.modifier.mod_x_idx) > 3:
            for i in range(self.modifier.mod_x_idx[2], self.modifier.mod_x_idx[-1]):
                x_eps = x_old.copy()  # copy the array, otherwise it passes the address, and this array will be changed
                x_eps[i] += self.eps

                fval = self.fval_nonlinear_FDM(x_eps)
                grad_nonlinear.append((fval - fval_old) / self.eps)

        self.nonlinear_grad_time += time.time()
        print(', %f s/nlin_num' % (self.nonlinear_grad_time / self.n_grad_calc), end='', flush=True)

        GRAD = np.concatenate((grad_linear, np.array(grad_nonlinear)))

        # gradient for regularization term----------------------------------------
        if self.regularization:
            R_vector = np.zeros(np.size(self.x_ref))
            for idx in range(np.size(self.x_ref)):
                if self.read_Cm_inv_vector:
                    R_vector[idx] = 2 * np.sum(self.x_diff * np.load(self.Cv_path + "/%s_Cv.npy" % idx)) * self.modifier.modifiers[0].norms
                else:
                    R_vector[idx] = 2 * np.sum(self.x_diff * self.C_m_inv[idx, :]) * self.modifier.modifiers[0].norms
            GRAD[0:np.size(self.x_ref)] = GRAD[0:np.size(self.x_ref)] + self.alpha * R_vector

        if self.regularization_diagonal:
            R_vector = self.alpha * 2 * self.x_diff * self.Cm_inv_diagonal * self.modifier.modifiers[0].norms
            # R_vector = self.alpha * 2 * self.x_diff * self.Cm_inv_diagonal
            GRAD[0:np.size(self.x_diff)] = GRAD[0:np.size(self.x_diff)] + R_vector

        self.grad_previous = GRAD

        # save the best optimized result and some history matching logs
        self.fval_list.append(self.fval_temp)
        if self.fval_temp < self.objfunval:
            filename = '%s_Optimized_parameters_best.pkl' % self.job_id
            with open(filename, "wb") as fp:
                # pickle.dump([self.x_temp, self.modifier.mod_x_idx, self.modifier], fp, pickle.HIGHEST_PROTOCOL)
                pickle.dump([self.x_temp, self.modifier.mod_x_idx, self.objfun_all, self.fval_temp, self.fval_list], fp, pickle.HIGHEST_PROTOCOL)
            self.objfunval = self.fval_temp


        # for heuristic rate control--------------------------
        # self.previous_forward_result = pd.DataFrame.from_dict(self.engine.time_data)

        if self.heuristic_rate_control:
            self.make_single_forward_simulation(x)  # convert rate control to BHP control

        return GRAD

    def grad_adjoint_method_mpfa_all(self, x: np.array) -> np.array:
        '''
        The preparation of the adjoint gradient for transmissibility and well index
        :param x: control variables
        :return: the adjoint gradient for transmissibility and well index
        '''
        # --------------------------gradients for transmissibility and well index--------------------------------------
        self.ad_grad_time -= time.time()

        grad_linear = np.zeros(self.n_fm)

        self.engine.n_control_vars = np.size(x)
        self.engine.col_dT_du = index_vector(self.col_idx)

        grad_linear = np.zeros(np.size(x))

        try:
            self.engine.calc_adjoint_gradient_dirac_all()  # call the adjoint calculation function from C++

            temp_grad = np.array(self.engine.derivatives, copy=True)  # get the adjoint gradinet from C++

            grad_linear = self.modifier.set_grad(temp_grad, self.x_idx)  # set the gradient to modifiers
        except IndexError:
            print("Adjoint fails! Discard this iteration step!")
            grad_linear = self.grad_old

        self.grad_old = grad_linear

        self.ad_grad_time += time.time()
        self.n_grad_calc += 1
        print(', %f s/lin_adj' % (self.ad_grad_time / self.n_grad_calc), end='', flush=True)

        # ---calculate the gradient of nonlinear parameters using finite difference method-----------------------------
        self.nonlinear_grad_time -= time.time()
        x_old = x
        fval_old = self.fval_temp
        grad_nonlinear = []
        if np.size(self.modifier.mod_x_idx) > 3:
            for i in range(self.modifier.mod_x_idx[2], self.modifier.mod_x_idx[-1]):
                x_eps = x_old.copy()  # copy the array, otherwise it passes the address, and this array will be changed
                x_eps[i] += self.eps

                fval = self.fval_nonlinear_FDM(x_eps)
                grad_nonlinear.append((fval - fval_old) / self.eps)

        self.nonlinear_grad_time += time.time()
        print(', %f s/nlin_num' % (self.nonlinear_grad_time / self.n_grad_calc), end='', flush=True)

        GRAD = np.concatenate((grad_linear, np.array(grad_nonlinear)))

        # gradient for regularization term----------------------------------------
        if self.regularization:
            R_vector = np.zeros(np.size(self.x_ref))
            for idx in range(np.size(self.x_ref)):
                if self.read_Cm_inv_vector:
                    R_vector[idx] = 2 * np.sum(self.x_diff * np.load(self.Cv_path + "/%s_Cv.npy" % idx)) * self.modifier.modifiers[0].norms
                else:
                    R_vector[idx] = 2 * np.sum(self.x_diff * self.C_m_inv[idx, :]) * self.modifier.modifiers[0].norms
            GRAD[0:np.size(self.x_ref)] = GRAD[0:np.size(self.x_ref)] + self.alpha * R_vector

        if self.regularization_diagonal:
            R_vector = self.alpha * 2 * self.x_diff * self.Cm_inv_diagonal * self.modifier.modifiers[0].norms
            # R_vector = self.alpha * 2 * self.x_diff * self.Cm_inv_diagonal
            GRAD[0:np.size(self.x_diff)] = GRAD[0:np.size(self.x_diff)] + R_vector

        self.grad_previous = GRAD

        # save the best optimized result and some history matching logs
        self.fval_list.append(self.fval_temp)
        if self.fval_temp < self.objfunval:
            filename = '%s_Optimized_parameters_best.pkl' % self.job_id
            with open(filename, "wb") as fp:
                # pickle.dump([self.x_temp, self.modifier.mod_x_idx, self.modifier], fp, pickle.HIGHEST_PROTOCOL)
                pickle.dump([self.x_temp, self.modifier.mod_x_idx, self.objfun_all, self.fval_temp, self.fval_list], fp, pickle.HIGHEST_PROTOCOL)
            self.objfunval = self.fval_temp


        # for heuristic rate control--------------------------
        # self.previous_forward_result = pd.DataFrame.from_dict(self.physics.engine.time_data)

        if self.heuristic_rate_control:
            self.make_single_forward_simulation(x)  # convert rate control to BHP control

        return GRAD

    def grad_adjoint_method_all_PCA(self, x: np.array) -> np.array:
        '''
        The preparation of the adjoint gradient for transmissibility and well index in reduced-dimension space using PCA
        :param x: control variables in reduced-dimension space
        :return: the adjoint gradient for transmissibility and well index in reduced-dimension space
        '''
        # --------------------------gradients for transmissibility and well index----------------
        self.ad_grad_time -= time.time()

        u_trans = self.phi.dot(x[0:self.space_dim] * self.norm_ksi) + self.x_ave
        u = np.concatenate((u_trans, x[self.space_dim:]))

        grad_linear = np.zeros(self.x_idx[2])

        self.engine.n_control_vars = np.size(u)
        self.engine.col_dT_du = index_vector(self.col_idx)

        grad_linear = np.zeros(np.size(u))

        try:
            self.engine.calc_adjoint_gradient_dirac_all()  # call the adjoint calculation function from C++

            temp_grad = np.array(self.engine.derivatives, copy=False)  # get the adjoint gradinet from C++

            grad_linear = self.modifier.set_grad(temp_grad, self.x_idx)  # set the gradient to modifiers
        except IndexError:
            print("Adjoint fails! Discard this iteration step!")
            grad_linear = self.grad_old

        grad_ksi_trans = self.grad_convert_coeff.dot(grad_linear[0:np.size(self.x_ave)]) * self.norm_ksi
        grad_WI = grad_linear[np.size(self.x_ave):]
        grad_linear = np.concatenate((grad_ksi_trans, grad_WI))

        self.grad_old = grad_linear

        self.ad_grad_time += time.time()
        self.n_grad_calc += 1
        print(', %f s/lin_adj' % (self.ad_grad_time / self.n_grad_calc), end='', flush=True)

        # ---calculate the gradient of nonlinear parameters using finite difference method--------
        self.nonlinear_grad_time -= time.time()
        x_old = u
        fval_old = self.fval_temp
        grad_nonlinear = []
        if np.size(self.modifier.mod_x_idx) > 3:
            for i in range(self.modifier.mod_x_idx[2], self.modifier.mod_x_idx[-1]):
                x_eps = x_old.copy()  # copy the array, otherwise it passes the address, and this array will be changed
                x_eps[i] += self.eps

                fval = self.fval_nonlinear_FDM(x_eps)
                grad_nonlinear.append((fval - fval_old) / self.eps)

        self.nonlinear_grad_time += time.time()
        print(', %f s/nlin_num' % (self.nonlinear_grad_time / self.n_grad_calc), end='', flush=True)

        GRAD = np.concatenate((grad_linear, np.array(grad_nonlinear)))

        # gradient for regularization term----------------------------------------
        if self.re_parameterized_PCA:
            self.ksi = self.x_temp[0:self.space_dim] * self.norm_ksi
            
            # GRAD[0:np.size(self.ksi)] = GRAD[0:np.size(self.ksi)] + self.alpha * 2 * self.ksi

            ksi_diff = self.ksi - self.ksi_ref * self.norm_ksi
            GRAD[0:np.size(self.ksi)] = GRAD[0:np.size(self.ksi)] + self.alpha * 2 * ksi_diff


        self.grad_previous = GRAD

        # save the best optimized result and some history matching logs
        self.fval_list.append(self.fval_temp)
        self.misfit_watch_list.append(self.misfit_value)
        if self.fval_temp < self.objfunval:
            filename = '%s_Optimized_parameters_best.pkl' % self.job_id
            with open(filename, "wb") as fp:
                # pickle.dump([self.x_temp, self.modifier.mod_x_idx, self.modifier], fp, pickle.HIGHEST_PROTOCOL)
                pickle.dump([self.x_temp, self.modifier.mod_x_idx, self.objfun_all, self.fval_temp, self.fval_list, self.misfit_watch_list], fp, pickle.HIGHEST_PROTOCOL)
            self.objfunval = self.fval_temp

        # for heuristic rate control--------------------------
        # self.previous_forward_result = pd.DataFrame.from_dict(self.engine.time_data)

        if self.heuristic_rate_control:
            self.make_single_forward_simulation(u)  # convert rate control to BHP control

        return GRAD




    def calculate_overall_rate_error_dirac_all(self, truth_df: pd.DataFrame, opt_df: pd.DataFrame, train_lenght: pd.DataFrame,
                                               truth_df_BT=0, truth_TEMPR=0, opt_tempr=0, truth_OP=0, opt_op=0):
        '''
        The calculation of the error between the history matching results and the true data considering noise covariance matrix and the weights.
        It is suggested to carefully review and customize this function for your specific purpose instead of directly using this function
        :param truth_df: time_data_report of true data
        :param opt_df: time_data_report of the optimized results
        :param train_lenght: index of the last training time step
        :param truth_df_BT: modified time_data (containing BHP and temperature data) that corresponds to the report time steps (i.e. observation time steps)
        :param truth_TEMPR: true time-elapse temperature data that corresponds to the report time steps (i.e. observation time steps)
        :param opt_tempr: oprimized time-elapse temperature data that corresponds to the report time steps (i.e. observation time steps)
        :param truth_OP: true time-elapse customized operature data that corresponds to the report time steps (i.e. observation time steps)
        :param opt_op: optimized time-elapse customized operature data that corresponds to the report time steps (i.e. observation time steps)
        '''
        # production phase rate data ----------------------------------------------------------------------------------
        prod_clean_training = 0
        PROD_L2_training = 0
        prod_clean_forcast = 0
        PROD_L2_forcast = 0
        prod_clean_overall = 0
        PROD_L2_overall = 0
        if self.objfun_prod_phase_rate:
            if np.size(np.array(self.phase_relative_density)) == 0:
                self.phase_relative_density = np.ones(np.size(self.prod_phase_name))

            q_separate = []
            Q_separate = []
            prod_cov_mat_inv = []

            for n, well in enumerate(self.prod_well_name):
                # phase rate
                if self.opt_phase_rate:
                    gaussian_noise_list = []
                    rate_list = []
                    rate_list_Q = []
                    std_dev_list = []
                    # for i in range(self.physics.n_phases):
                    for i in range(np.size(self.prod_phase_name)):
                        gaussian_noise_list.append(0)
                        rate_list.append(0)
                        rate_list_Q.append(0)
                        std_dev_list.append(0)

                    for p, phase in enumerate(self.prod_phase_name):
                        rate_string = well + " : " + phase + " rate (m3/day)"
                        rate_serie_q = -opt_df.get(rate_string)  # set the production phase as positive sign
                        rate_serie_Q = truth_df.get(rate_string)

                        gaussian_noise_list[p] = np.random.randn(rate_serie_Q.size)
                        random_training_period = self.std_gaussian_noise_prod[n][p]
                        gaussian_noise_list[p][0:np.size(random_training_period)] = random_training_period

                        rate_list[p] = rate_serie_q.values * self.phase_relative_density[p]

                        rate_value_Q = np.array(rate_serie_Q.values)
                        if self.prod_rate_measurement_error != 0:
                            rate_value_Q[rate_value_Q < self.prod_error_lower_bound] = self.prod_error_lower_bound
                            rate_value_Q[rate_value_Q > self.prod_error_upper_bound] = self.prod_error_upper_bound

                        # rate_list_Q[p] = rate_serie_Q.values * self.phase_relative_density[p] \
                        #                  * (1 + self.prod_rate_measurement_error * gaussian_noise_list[p])

                        rate_list_Q[p] = rate_serie_Q.values * self.phase_relative_density[p] \
                                         + rate_value_Q * self.prod_rate_measurement_error * gaussian_noise_list[p]

                        if self.prod_rate_measurement_error == 0:
                            std_dev_list[p] = np.ones(rate_serie_Q.size)
                        else:
                            std_dev_list[p] = rate_value_Q * self.phase_relative_density[p] * self.prod_rate_measurement_error

                q_separate.append(rate_list)
                Q_separate.append(rate_list_Q)
                prod_cov_mat_inv.append(1 / np.array(std_dev_list))

            q_separate = np.array(q_separate)
            Q_separate = np.array(Q_separate)
            prod_cov_mat_inv = np.array(prod_cov_mat_inv)

            prod_weights = np.ones(np.shape(Q_separate))
            prod_weights[:, :, 0:train_lenght] = self.prod_weights
            for pw in range(train_lenght, np.size(Q_separate, -1)):
                prod_weights[:, :, pw] = self.prod_weights[:, :, -1]

            t_sim = opt_df['time']
            t_Q = truth_df['time']
            self.dirac_vec = np.zeros(np.size(t_sim))
            for i, ts in enumerate(t_sim):
                for tr in t_Q:
                    if ts==tr:
                        self.dirac_vec[i] = 1


            TotStep = np.size(t_sim)
            size_Q = np.size(Q_separate, 2)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_Q - 1
            q_Q = []

            q_Q.append((np.array(q_separate)[:, :, idx_sim_ts] - np.array(Q_separate)[:, :, idx_obs_ts])
                       * self.dirac_vec[idx_sim_ts]
                       * np.array(prod_cov_mat_inv)[:, :, idx_obs_ts]
                       * np.array(prod_weights)[:, :, idx_obs_ts] ** 0.5)

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1

                q_Q.append((np.array(q_separate)[:, :, idx_sim_ts] - np.array(Q_separate)[:, :, idx_obs_ts])
                           * self.dirac_vec[idx_sim_ts]
                           * np.array(prod_cov_mat_inv)[:, :, idx_obs_ts]
                           * np.array(prod_weights)[:, :, idx_obs_ts] ** 0.5)

            q_Q_clean = np.array(q_Q)[np.flipud(self.dirac_vec).astype('bool'), :].T


            prod_clean_training = np.sum(q_Q_clean[:, :, 0:train_lenght]**2)
            PROD_L2_training = np.sum((np.array(Q_separate)[:, :, 0:train_lenght]
                                       * np.array(prod_cov_mat_inv)[:, :, 0:train_lenght]
                                       * np.array(prod_weights)[:, :, 0:train_lenght] ** 0.5
                                       )**2)

            prod_clean_forcast = np.sum(q_Q_clean[:, :, train_lenght:]**2)
            PROD_L2_forcast = np.sum((np.array(Q_separate)[:, :, train_lenght:]
                                      * np.array(prod_cov_mat_inv)[:, :, train_lenght:]
                                      * np.array(prod_weights)[:, :, train_lenght:] ** 0.5
                                      )**2)

            prod_clean_overall = np.sum(q_Q_clean**2)
            PROD_L2_overall = np.sum((np.array(Q_separate)
                                      * np.array(prod_cov_mat_inv)
                                      * np.array(prod_weights) ** 0.5
                                      )**2)

            err_training_prod = prod_clean_training / PROD_L2_training * 100
            err_forcast_prod = prod_clean_forcast / PROD_L2_forcast * 100
            err_overall_prod = prod_clean_overall / PROD_L2_overall * 100

            print('\n')
            print('Production Phase---Relative Error Training: ', err_training_prod)
            print('Production Phase---Relative Error Forecast: ', err_forcast_prod)
            print('Production Phase---Relative Error Overall: ', err_overall_prod)

        # injection phase rate data ----------------------------------------------------------------------------------
        inj_clean_training = 0
        INJ_L2_training = 0
        inj_clean_forcast = 0
        INJ_L2_forcast = 0
        inj_clean_overall = 0
        INJ_L2_overall = 0
        if self.objfun_inj_phase_rate:
            q_inj_separate = []
            Q_inj_separate = []
            inj_cov_mat_inv = []

            for n, well in enumerate(self.inj_well_name):
                gaussian_noise_list = []
                rate_list_inj_q = []
                rate_list_inj_Q = []
                std_dev_list = []

                for i in range(np.size(self.inj_phase_name)):
                    gaussian_noise_list.append(0)
                    rate_list_inj_q.append(0)
                    rate_list_inj_Q.append(0)
                    std_dev_list.append(0)

                for p, phase in enumerate(self.inj_phase_name):
                    rate_string = well + " : " + phase + " rate (m3/day)"
                    rate_serie_inj_q = opt_df.get(rate_string)
                    rate_serie_inj_Q = truth_df.get(rate_string)

                    gaussian_noise_list[p] = np.random.randn(rate_serie_Q.size)
                    random_training_period = self.std_gaussian_noise_inj[n][p]
                    gaussian_noise_list[p][0:np.size(random_training_period)] = random_training_period

                    rate_list_inj_q[p] = rate_serie_inj_q.values

                    rate_value_Q_inj = np.array(rate_serie_inj_Q.values)
                    if self.inj_rate_measurement_error != 0:
                        rate_value_Q_inj[rate_value_Q_inj < self.inj_error_lower_bound] = self.inj_error_lower_bound
                        rate_value_Q_inj[rate_value_Q_inj > self.inj_error_upper_bound] = self.inj_error_upper_bound

                    # rate_list_inj_Q[p] = rate_serie_inj_Q.values * (1 + self.inj_rate_measurement_error * gaussian_noise_list[p])
                    rate_list_inj_Q[p] = rate_serie_inj_Q.values + rate_value_Q_inj * self.inj_rate_measurement_error * gaussian_noise_list[p]

                    if self.inj_rate_measurement_error == 0:
                        std_dev_list[p] = np.ones(rate_serie_inj_Q.size)
                    else:
                        std_dev_list[p] = rate_value_Q_inj * self.inj_rate_measurement_error

                q_inj_separate.append(rate_list_inj_q)
                Q_inj_separate.append(rate_list_inj_Q)
                inj_cov_mat_inv.append(1 / np.array(std_dev_list))

            q_inj_separate = np.array(q_inj_separate)
            Q_inj_separate = np.array(Q_inj_separate)
            inj_cov_mat_inv = np.array(inj_cov_mat_inv)

            inj_weights = np.ones(np.shape(Q_inj_separate))
            inj_weights[:, :, 0:train_lenght] = self.inj_weights
            for iw in range(train_lenght, np.size(Q_inj_separate, -1)):
                inj_weights[:, :, iw] = self.inj_weights[:, :, -1]

            t_sim = opt_df['time']
            t_Q = truth_df['time']
            self.dirac_vec = np.zeros(np.size(t_sim))
            for i, ts in enumerate(t_sim):
                for tr in t_Q:
                    if ts==tr:
                        self.dirac_vec[i] = 1

            TotStep = np.size(t_sim)
            size_inj_Q = np.size(Q_inj_separate, 2)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_inj_Q - 1
            q_Q_inj = []

            q_Q_inj.append((np.array(q_inj_separate)[:, :, idx_sim_ts] - np.array(Q_inj_separate)[:, :, idx_obs_ts])
                           * self.dirac_vec[idx_sim_ts]
                           * np.array(inj_cov_mat_inv)[:, :, idx_obs_ts]
                           * np.array(inj_weights)[:, :, idx_obs_ts] ** 0.5)

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1

                q_Q_inj.append((np.array(q_inj_separate)[:, :, idx_sim_ts] - np.array(Q_inj_separate)[:, :, idx_obs_ts])
                               * self.dirac_vec[idx_sim_ts]
                               * np.array(inj_cov_mat_inv)[:, :, idx_obs_ts]
                               * np.array(inj_weights)[:, :, idx_obs_ts] ** 0.5)

            q_Q_inj_clean = np.array(q_Q_inj)[np.flipud(self.dirac_vec).astype('bool'), :].T


            inj_clean_training = np.sum(q_Q_inj_clean[:, :, 0:train_lenght]**2)
            INJ_L2_training = np.sum((np.array(Q_inj_separate)[:, :, 0:train_lenght]
                                      * np.array(inj_cov_mat_inv)[:, :, 0:train_lenght]
                                      * np.array(inj_weights)[:, :, 0:train_lenght] ** 0.5
                                      )**2)

            inj_clean_forcast = np.sum(q_Q_inj_clean[:, :, train_lenght:]**2)
            INJ_L2_forcast = np.sum((np.array(Q_inj_separate)[:, :, train_lenght:]
                                     * np.array(inj_cov_mat_inv)[:, :, train_lenght:]
                                     * np.array(inj_weights)[:, :, train_lenght:] ** 0.5
                                     )**2)

            inj_clean_overall = np.sum(q_Q_inj_clean**2)
            INJ_L2_overall = np.sum((np.array(Q_inj_separate)
                                     * np.array(inj_cov_mat_inv)
                                     * np.array(inj_weights)
                                     )**2)

            err_training_inj = inj_clean_training / INJ_L2_training * 100
            err_forcast_inj = inj_clean_forcast / INJ_L2_forcast * 100
            err_overall_inj = inj_clean_overall / INJ_L2_overall * 100


            print('\n')
            print('Injction Phase---Relative Error Training: ', err_training_inj)
            print('Injction Phase---Relative Error Forecast: ', err_forcast_inj)
            print('Injction Phase---Relative Error Overall: ', err_overall_inj)

        # BHP data --------------------------------------------------------------------------------------------------
        bhp_clean_training = 0
        BHP_L2_training = 0
        bhp_clean_forcast = 0
        BHP_L2_forcast = 0
        bhp_clean_overall = 0
        BHP_L2_overall = 0
        if self.objfun_BHP:
            bhp_separate = []
            BHP_separate = []
            BHP_cov_mat_inv = []

            for n, well in enumerate(self.BHP_well_name):
                std_dev_list = []
                BHP_string = well + " : BHP (bar)"
                bhp_serie = opt_df.get(BHP_string)
                BHP_serie = truth_df_BT.get(BHP_string)

                gaussian_noise_list = np.random.randn(BHP_serie.size)
                random_training_period = self.std_gaussian_noise_BHP[n]
                gaussian_noise_list[0:np.size(random_training_period)] = random_training_period

                BHP_value = np.array(BHP_serie.values)
                if self.BHP_measurement_error != 0:
                    BHP_value[BHP_value < self.BHP_error_lower_bound] = self.BHP_error_lower_bound
                    BHP_value[BHP_value > self.BHP_error_upper_bound] = self.BHP_error_upper_bound

                if self.BHP_measurement_error == 0:
                    std_dev_list = np.ones(BHP_serie.size)
                else:
                    std_dev_list = BHP_value * self.BHP_measurement_error

                bhp_separate.append(bhp_serie.values)
                BHP_separate.append(BHP_serie.values + BHP_value * self.BHP_measurement_error * gaussian_noise_list)
                BHP_cov_mat_inv.append(1 / np.array(std_dev_list))

            bhp_separate = np.array(bhp_separate)
            BHP_separate = np.array(BHP_separate)
            BHP_cov_mat_inv = np.array(BHP_cov_mat_inv)

            BHP_weights = np.ones(np.shape(BHP_separate))
            BHP_weights[:, 0:train_lenght] = self.BHP_weights
            for bw in range(train_lenght, np.size(BHP_separate, -1)):
                BHP_weights[:, bw] = self.BHP_weights[:, -1]

            t_sim = opt_df['time']
            t_Q = truth_df_BT['time']
            self.dirac_vec = np.zeros(np.size(t_sim))
            for i, ts in enumerate(t_sim):
                for tr in t_Q:
                    if ts == tr:
                        self.dirac_vec[i] = 1

            TotStep = np.size(t_sim)
            size_BHP = np.size(BHP_separate, 1)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_BHP - 1

            bhp_BHP = []
            bhp_BHP.append((np.array(bhp_separate)[:, idx_sim_ts] - np.array(BHP_separate)[:, idx_obs_ts])
                           * self.dirac_vec[idx_sim_ts]
                           * np.array(BHP_cov_mat_inv)[:, idx_obs_ts]
                           * np.array(BHP_weights)[:, idx_obs_ts] ** 0.5)

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1
                bhp_BHP.append((np.array(bhp_separate)[:, idx_sim_ts] - np.array(BHP_separate)[:, idx_obs_ts])
                               * self.dirac_vec[idx_sim_ts]
                               * np.array(BHP_cov_mat_inv)[:, idx_obs_ts]
                               * np.array(BHP_weights)[:, idx_obs_ts] ** 0.5)

            bhp_BHP_clean = np.array(bhp_BHP)[np.flipud(self.dirac_vec).astype('bool'), :].T


            bhp_clean_training = sq_norm(bhp_BHP_clean[:, 0:train_lenght])
            BHP_L2_training = sq_norm((np.array(BHP_separate)[:, 0:train_lenght]
                                      * np.array(BHP_cov_mat_inv)[:, 0:train_lenght]
                                      * np.array(BHP_weights)[:, 0:train_lenght] ** 0.5
                                      ))

            bhp_clean_forcast = sq_norm(bhp_BHP_clean[:, train_lenght:])
            BHP_L2_forcast = sq_norm((np.array(BHP_separate)[:, train_lenght:]
                                     * np.array(BHP_cov_mat_inv)[:, train_lenght:]
                                     * np.array(BHP_weights)[:, train_lenght:] ** 0.5
                                     ))

            bhp_clean_overall = sq_norm(bhp_BHP_clean)
            BHP_L2_overall = sq_norm((np.array(BHP_separate)
                                     * np.array(BHP_cov_mat_inv)
                                     * np.array(BHP_weights) ** 0.5
                                     ))

            err_training_BHP = bhp_clean_training / BHP_L2_training * 100
            err_forcast_BHP = bhp_clean_forcast / BHP_L2_forcast * 100
            err_overall_BHP = bhp_clean_overall / BHP_L2_overall * 100

            print('\n')
            print('BHP---Relative Error Training: ', err_training_BHP)
            print('BHP---Relative Error Forecast: ', err_forcast_BHP)
            print('BHP---Relative Error Overall: ', err_overall_BHP)

        # well temperature data ---------------------------------------------------------------------------------------
        wt_clean_training = 0
        WT_L2_training = 0
        wt_clean_forcast = 0
        WT_L2_forcast = 0
        wt_clean_overall = 0
        WT_L2_overall = 0
        if self.objfun_well_tempr:
            wt_separate = []
            WT_separate = []
            well_tempr_cov_mat_inv = []

            for n, well in enumerate(self.well_tempr_name):
                std_dev_list = []
                well_tempr_string = well + " : temperature (K)"
                wt_serie = opt_df.get(well_tempr_string)
                WT_serie = truth_df_BT.get(well_tempr_string)

                gaussian_noise_list = np.random.randn(WT_serie.size)
                random_training_period = self.std_gaussian_noise_well_tempr[n]
                gaussian_noise_list[0:np.size(random_training_period)] = random_training_period

                WT_value = np.array(WT_serie.values)
                if self.well_tempr_measurement_error != 0:
                    WT_value[WT_value < self.well_tempr_error_lower_bound] = self.well_tempr_error_lower_bound
                    WT_value[WT_value > self.well_tempr_error_upper_bound] = self.well_tempr_error_upper_bound

                if self.well_tempr_measurement_error == 0:
                    std_dev_list = np.ones(WT_serie.size)
                else:
                    std_dev_list = WT_value * self.well_tempr_measurement_error

                wt_separate.append(wt_serie.values)
                WT_separate.append(WT_serie.values + WT_value * self.well_tempr_measurement_error * gaussian_noise_list)
                well_tempr_cov_mat_inv.append(1/ np.array(std_dev_list))

            wt_separate = np.array(wt_separate)
            WT_separate = np.array(WT_separate)
            well_tempr_cov_mat_inv = np.array(well_tempr_cov_mat_inv)

            WT_weights = np.ones(np.shape(WT_separate))
            WT_weights[:, 0:train_lenght] = self.well_tempr_weights
            for wtw in range(train_lenght, np.size(WT_separate, -1)):
                WT_weights[:, wtw] = self.well_tempr_weights[:, -1]

            t_sim = opt_df['time']
            t_Q = truth_df_BT['time']
            self.dirac_vec = np.zeros(np.size(t_sim))
            for i, ts in enumerate(t_sim):
                for tr in t_Q:
                    if ts == tr:
                        self.dirac_vec[i] = 1

            TotStep = np.size(t_sim)
            size_WT = np.size(WT_separate, 1)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_WT - 1

            wt_WT = []
            wt_WT.append((np.array(wt_separate)[:, idx_sim_ts] - np.array(WT_separate)[:, idx_obs_ts])
                           * self.dirac_vec[idx_sim_ts]
                           * np.array(well_tempr_cov_mat_inv)[:, idx_obs_ts]
                           * np.array(WT_weights)[:, idx_obs_ts] ** 0.5)

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1
                wt_WT.append((np.array(wt_separate)[:, idx_sim_ts] - np.array(WT_separate)[:, idx_obs_ts])
                               * self.dirac_vec[idx_sim_ts]
                               * np.array(well_tempr_cov_mat_inv)[:, idx_obs_ts]
                               * np.array(WT_weights)[:, idx_obs_ts] ** 0.5)

            wt_WT_clean = np.array(wt_WT)[np.flipud(self.dirac_vec).astype('bool'), :].T


            wt_clean_training = sq_norm(wt_WT_clean[:, 0:train_lenght])
            WT_L2_training = sq_norm((np.array(WT_separate)[:, 0:train_lenght]
                                      * np.array(well_tempr_cov_mat_inv)[:, 0:train_lenght]
                                      * np.array(WT_weights)[:, 0:train_lenght] ** 0.5
                                      ))

            wt_clean_forcast = sq_norm(wt_WT_clean[:, train_lenght:])
            WT_L2_forcast = sq_norm((np.array(WT_separate)[:, train_lenght:]
                                     * np.array(well_tempr_cov_mat_inv)[:, train_lenght:]
                                     * np.array(WT_weights)[:, train_lenght:] ** 0.5
                                     ))

            wt_clean_overall = sq_norm(wt_WT_clean)
            WT_L2_overall = sq_norm((np.array(WT_separate)
                                     * np.array(well_tempr_cov_mat_inv)
                                     * np.array(WT_weights) ** 0.5
                                     ))

            err_training_well_temper = wt_clean_training / WT_L2_training * 100
            err_forcast_well_temper = wt_clean_forcast / WT_L2_forcast * 100
            err_overall_well_temper = wt_clean_overall / WT_L2_overall * 100

            print('\n')
            print('well_temper---Relative Error Training: ', err_training_well_temper)
            print('well_temper---Relative Error Forecast: ', err_forcast_well_temper)
            print('well_temper---Relative Error Overall: ', err_overall_well_temper)

        # temperature data --------------------------------------------------------------------------------------------
        temperature_clean_training = 0
        TEMPERATURE_L2_training = 0
        temperature_clean_forcast = 0
        TEMPERATURE_L2_forcast = 0
        temperature_clean_overall = 0
        TEMPERATURE_L2_overall = 0
        if self.objfun_temperature:
            gaussian_noise_list = np.random.randn(np.size(truth_TEMPR, 0), np.size(truth_TEMPR, 1))
            random_training_period = self.std_gaussian_noise_temperature
            gaussian_noise_list[0:np.size(random_training_period, 0), :]

            tempr_value = np.array(truth_TEMPR)
            if self.temperature_measurement_error != 0:
                tempr_value[tempr_value < self.temperature_error_lower_bound] = self.temperature_error_lower_bound
                tempr_value[tempr_value > self.temperature_error_upper_bound] = self.temperature_error_upper_bound

            if self.temperature_measurement_error == 0:
                std_dev_list = np.ones(np.shape(truth_TEMPR))
            else:
                std_dev_list = tempr_value * self.temperature_measurement_error

            temperature_separate = opt_tempr
            TEMPERATURE_separate = truth_TEMPR + tempr_value * self.temperature_measurement_error * gaussian_noise_list
            TEMPERATURE_cov_mat_inv = 1 / np.array(std_dev_list)

            temperature_weights = np.ones(np.shape(TEMPERATURE_separate))
            temperature_weights[0:train_lenght, :] = self.temperature_weights
            for tw in range(train_lenght, np.size(temperature_weights, 0)):
                temperature_weights[tw, :] = self.temperature_weights[-1, :]

            t_sim = opt_df['time']
            t_Q = truth_df['time']
            self.dirac_vec = np.zeros(np.size(t_sim))
            for i, ts in enumerate(t_sim):
                for tr in t_Q:
                    if ts == tr:
                        self.dirac_vec[i] = 1

            TotStep = np.size(t_sim)
            size_TEMPR = np.size(TEMPERATURE_separate, 0)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_TEMPR - 1

            tempr_TEMPR = []
            tempr_TEMPR.append((np.array(temperature_separate)[idx_sim_ts, :] - np.array(TEMPERATURE_separate)[idx_obs_ts, :])
                               * self.dirac_vec[idx_sim_ts]
                               * TEMPERATURE_cov_mat_inv[idx_obs_ts, :]
                               * temperature_weights[idx_obs_ts, :] ** 0.5)

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1
                tempr_TEMPR.append((np.array(temperature_separate)[idx_sim_ts, :] - np.array(TEMPERATURE_separate)[idx_obs_ts, :])
                                   * self.dirac_vec[idx_sim_ts]
                                   * TEMPERATURE_cov_mat_inv[idx_obs_ts, :]
                                   * temperature_weights[idx_obs_ts, :] ** 0.5)

            tempr_TEMPR_clean = np.array(tempr_TEMPR)[np.flipud(self.dirac_vec).astype('bool'), :].T


            temperature_clean_training = sq_norm(tempr_TEMPR_clean[0:train_lenght, :])
            TEMPERATURE_L2_training = sq_norm((np.array(TEMPERATURE_separate)[0:train_lenght, :]
                                              * np.array(TEMPERATURE_cov_mat_inv)[0:train_lenght, :]
                                              * np.array(temperature_weights)[0:train_lenght, :] ** 0.5
                                              ))

            temperature_clean_forcast = sq_norm(tempr_TEMPR_clean[train_lenght:, :])
            TEMPERATURE_L2_forcast = sq_norm((np.array(TEMPERATURE_separate)[train_lenght:, :]
                                             * np.array(TEMPERATURE_cov_mat_inv)[train_lenght:, :]
                                             * np.array(temperature_weights)[train_lenght:, :] ** 0.5
                                             ))

            temperature_clean_overall = sq_norm(tempr_TEMPR_clean)
            TEMPERATURE_L2_overall = sq_norm((np.array(TEMPERATURE_separate)
                                             * np.array(TEMPERATURE_cov_mat_inv)
                                             * np.array(temperature_weights) ** 0.5
                                             ))

            err_training_tempr = temperature_clean_training / TEMPERATURE_L2_training * 100
            err_forcast_tempr = temperature_clean_forcast / TEMPERATURE_L2_forcast * 100
            err_overall_tempr = temperature_clean_overall / TEMPERATURE_L2_overall * 100
            print('\n')
            print('Temperature---Relative Error Training: ', err_training_tempr)
            print('Temperature---Relative Error Forecast: ', err_forcast_tempr)
            print('Temperature---Relative Error Overall: ', err_overall_tempr)

        # customized operator data --------------------------------------------------------------------------------------------
        customized_op_clean_training = 0
        CUSTOMIZED_OP_L2_training = 0
        customized_op_clean_forcast = 0
        CUSTOMIZED_OP_L2_forcast = 0
        customized_op_clean_overall = 0
        CUSTOMIZED_OP_L2_overall = 0
        if self.objfun_customized_op:
            gaussian_noise_list = np.random.randn(np.size(truth_OP, 0), np.size(truth_OP, 1))
            random_training_period = self.std_gaussian_noise_customized_op
            gaussian_noise_list[0:np.size(random_training_period, 0), :]

            op_value = np.array(truth_OP)
            if self.customized_op_measurement_error != 0:
                op_value[op_value < self.customized_op_error_lower_bound] = self.customized_op_error_lower_bound
                op_value[op_value > self.customized_op_error_upper_bound] = self.customized_op_error_upper_bound

            if self.customized_op_measurement_error == 0:
                std_dev_list = np.ones(np.shape(truth_OP))
            else:
                std_dev_list = op_value * self.customized_op_measurement_error

            customized_op_separate = opt_op
            CUSTOMIZED_OP_separate = truth_OP + op_value * self.customized_op_measurement_error * gaussian_noise_list
            CUSTOMIZED_OP_cov_mat_inv = 1 / np.array(std_dev_list)

            customized_op_weights = np.ones(np.shape(CUSTOMIZED_OP_separate))
            customized_op_weights[0:train_lenght, :] = self.customized_op_weights
            for tw in range(train_lenght, np.size(customized_op_weights, 0)):
                customized_op_weights[tw, :] = self.customized_op_weights[-1, :]

            t_sim = opt_df['time']
            t_Q = truth_df['time']
            self.dirac_vec = np.zeros(np.size(t_sim))
            for i, ts in enumerate(t_sim):
                for tr in t_Q:
                    if ts == tr:
                        self.dirac_vec[i] = 1

            TotStep = np.size(t_sim)
            size_OP = np.size(CUSTOMIZED_OP_separate, 0)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_OP - 1

            op_OP = []
            op_OP.append((np.array(customized_op_separate)[idx_sim_ts, :] - np.array(CUSTOMIZED_OP_separate)[idx_obs_ts, :])
                               * self.dirac_vec[idx_sim_ts]
                               * CUSTOMIZED_OP_cov_mat_inv[idx_obs_ts, :]
                               * customized_op_weights[idx_obs_ts, :] ** 0.5)

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1
                op_OP.append((np.array(customized_op_separate)[idx_sim_ts, :] - np.array(CUSTOMIZED_OP_separate)[idx_obs_ts, :])
                                   * self.dirac_vec[idx_sim_ts]
                                   * CUSTOMIZED_OP_cov_mat_inv[idx_obs_ts, :]
                                   * customized_op_weights[idx_obs_ts, :] ** 0.5)

            op_OP_clean = np.array(op_OP)[np.flipud(self.dirac_vec).astype('bool'), :].T

            customized_op_clean_training = sq_norm(op_OP_clean[0:train_lenght, :])
            CUSTOMIZED_OP_L2_training = sq_norm((np.array(CUSTOMIZED_OP_separate)[0:train_lenght, :]
                                              * np.array(CUSTOMIZED_OP_cov_mat_inv)[0:train_lenght, :]
                                              * np.array(customized_op_weights)[0:train_lenght, :] ** 0.5
                                              ))

            customized_op_clean_forcast = sq_norm(op_OP_clean[train_lenght:, :])
            CUSTOMIZED_OP_L2_forcast = sq_norm((np.array(CUSTOMIZED_OP_separate)[train_lenght:, :]
                                             * np.array(CUSTOMIZED_OP_cov_mat_inv)[train_lenght:, :]
                                             * np.array(customized_op_weights)[train_lenght:, :] ** 0.5
                                             ))

            customized_op_clean_overall = sq_norm(op_OP_clean)
            CUSTOMIZED_OP_L2_overall = sq_norm((np.array(CUSTOMIZED_OP_separate)
                                             * np.array(CUSTOMIZED_OP_cov_mat_inv)
                                             * np.array(customized_op_weights) ** 0.5
                                             ))

            err_training_op = customized_op_clean_training / CUSTOMIZED_OP_L2_training * 100
            err_forcast_op = customized_op_clean_forcast / CUSTOMIZED_OP_L2_forcast * 100
            err_overall_op = customized_op_clean_overall / CUSTOMIZED_OP_L2_overall * 100
            print('\n')
            print('Temperature---Relative Error Training: ', err_training_op)
            print('Temperature---Relative Error Forecast: ', err_forcast_op)
            print('Temperature---Relative Error Overall: ', err_overall_op)


        # all data including prod, inj, BHP, well_tempr, tempr, etc.--------------------------------------------------
        err_training_total = (prod_clean_training + inj_clean_training + bhp_clean_training + wt_clean_training + temperature_clean_training + customized_op_clean_training) \
                             / (PROD_L2_training + INJ_L2_training + BHP_L2_training + WT_L2_training + TEMPERATURE_L2_training + CUSTOMIZED_OP_L2_training) * 100

        err_forcast_total = (prod_clean_forcast + inj_clean_forcast + bhp_clean_forcast + wt_clean_forcast + temperature_clean_forcast + customized_op_clean_forcast) \
                            / (PROD_L2_forcast + INJ_L2_forcast + BHP_L2_forcast + WT_L2_forcast + TEMPERATURE_L2_forcast + CUSTOMIZED_OP_L2_forcast) * 100

        err_overall_total = (prod_clean_overall + inj_clean_overall + bhp_clean_overall + wt_clean_overall + temperature_clean_overall + customized_op_clean_overall) \
                            / (PROD_L2_overall + INJ_L2_overall + BHP_L2_overall + WT_L2_overall + TEMPERATURE_L2_overall + CUSTOMIZED_OP_L2_overall) * 100
        print('\n')
        print('Total---Relative Error Training: ', err_training_total)
        print('Total---Relative Error Forecast: ', err_forcast_total)
        print('Total---Relative Error Overall: ', err_overall_total)




    def calculate_error_without_noise_weight(self, truth_df: pd.DataFrame, opt_df: pd.DataFrame, train_lenght: pd.DataFrame,
                                             truth_df_BT=0, truth_TEMPR=0, opt_tempr=0, truth_OP=0, opt_op=0):
        '''
        The calculation of the error between the history matching results and the true data without considering noise covariance matrix and the weights
        It is suggested to carefully review and customize this function for your specific purpose instead of directly using this function
        :param truth_df: time_data_report of true data
        :param opt_df: time_data_report of the optimized results
        :param train_lenght: index of the last training time step
        :param truth_df_BT: modified time_data (containing BHP and temperature data) that corresponds to the report time steps (i.e. observation time steps)
        :param truth_TEMPR: true time-elapse temperature data that corresponds to the report time steps (i.e. observation time steps)
        :param opt_tempr: oprimized time-elapse temperature data that corresponds to the report time steps (i.e. observation time steps)
        :param truth_OP: true time-elapse customized operature data that corresponds to the report time steps (i.e. observation time steps)
        :param opt_op: optimized time-elapse customized operature data that corresponds to the report time steps (i.e. observation time steps)
        '''
        # production phase rate data ----------------------------------------------------------------------------------
        prod_clean_training = 0
        PROD_L2_training = 0
        prod_clean_forcast = 0
        PROD_L2_forcast = 0
        prod_clean_overall = 0
        PROD_L2_overall = 0
        if self.objfun_prod_phase_rate:
            if np.size(np.array(self.phase_relative_density)) == 0:
                self.phase_relative_density = np.ones(np.size(self.prod_phase_name))

            q_separate = []
            Q_separate = []

            for n, well in enumerate(self.prod_well_name):
                # phase rate
                if self.opt_phase_rate:
                    rate_list = []
                    rate_list_Q = []
                    # for i in range(self.physics.n_phases):
                    for i in range(np.size(self.prod_phase_name)):
                        rate_list.append(0)
                        rate_list_Q.append(0)

                    for p, phase in enumerate(self.prod_phase_name):
                        rate_string = well + " : " + phase + " rate (m3/day)"
                        rate_serie_q = -opt_df.get(rate_string)  # set the production phase as positive sign
                        rate_serie_Q = truth_df.get(rate_string)

                        rate_list[p] = rate_serie_q.values * self.phase_relative_density[p]
                        rate_list_Q[p] = rate_serie_Q.values \
                                         * self.phase_relative_density[p]

                q_separate.append(rate_list)
                Q_separate.append(rate_list_Q)

            q_separate = np.array(q_separate)
            Q_separate = np.array(Q_separate)

            t_sim = opt_df['time']
            t_Q = truth_df['time']
            self.dirac_vec = np.zeros(np.size(t_sim))
            for i, ts in enumerate(t_sim):
                for tr in t_Q:
                    if ts==tr:
                        self.dirac_vec[i] = 1


            TotStep = np.size(t_sim)
            size_Q = np.size(Q_separate, 2)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_Q - 1
            q_Q = []

            q_Q.append((np.array(q_separate)[:, :, idx_sim_ts] - np.array(Q_separate)[:, :, idx_obs_ts])
                       * self.dirac_vec[idx_sim_ts]
                       )

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1

                q_Q.append((np.array(q_separate)[:, :, idx_sim_ts] - np.array(Q_separate)[:, :, idx_obs_ts])
                           * self.dirac_vec[idx_sim_ts]
                           )

            q_Q_clean = np.array(q_Q)[np.flipud(self.dirac_vec).astype('bool'), :].T


            prod_clean_training = np.sum(q_Q_clean[:, :, 0:train_lenght]**2)
            PROD_L2_training = np.sum((np.array(Q_separate)[:, :, 0:train_lenght]
                                      )**2)

            prod_clean_forcast = np.sum(q_Q_clean[:, :, train_lenght:]**2)
            PROD_L2_forcast = np.sum((np.array(Q_separate)[:, :, train_lenght:]
                                      )**2)

            prod_clean_overall = np.sum(q_Q_clean**2)
            PROD_L2_overall = np.sum((np.array(Q_separate)
                                      )**2)

            err_training_prod = prod_clean_training / PROD_L2_training * 100
            err_forcast_prod = prod_clean_forcast / PROD_L2_forcast * 100
            err_overall_prod = prod_clean_overall / PROD_L2_overall * 100

            print('\n')
            print('Production Phase---Relative Error Training: ', err_training_prod)
            print('Production Phase---Relative Error Forecast: ', err_forcast_prod)
            print('Production Phase---Relative Error Overall: ', err_overall_prod)

        # injection phase rate data ----------------------------------------------------------------------------------
        inj_clean_training = 0
        INJ_L2_training = 0
        inj_clean_forcast = 0
        INJ_L2_forcast = 0
        inj_clean_overall = 0
        INJ_L2_overall = 0
        if self.objfun_inj_phase_rate:
            q_inj_separate = []
            Q_inj_separate = []

            for n, well in enumerate(self.inj_well_name):
                rate_list_inj_q = []
                rate_list_inj_Q = []

                for i in range(np.size(self.inj_phase_name)):
                    rate_list_inj_q.append(0)
                    rate_list_inj_Q.append(0)

                for p, phase in enumerate(self.inj_phase_name):
                    rate_string = well + " : " + phase + " rate (m3/day)"
                    rate_serie_inj_q = opt_df.get(rate_string)
                    rate_serie_inj_Q = truth_df.get(rate_string)

                    rate_list_inj_q[p] = rate_serie_inj_q.values
                    rate_list_inj_Q[p] = rate_serie_inj_Q.values

                q_inj_separate.append(rate_list_inj_q)
                Q_inj_separate.append(rate_list_inj_Q)

            q_inj_separate = np.array(q_inj_separate)
            Q_inj_separate = np.array(Q_inj_separate)

            t_sim = opt_df['time']
            t_Q = truth_df['time']
            self.dirac_vec = np.zeros(np.size(t_sim))
            for i, ts in enumerate(t_sim):
                for tr in t_Q:
                    if ts==tr:
                        self.dirac_vec[i] = 1

            TotStep = np.size(t_sim)
            size_inj_Q = np.size(Q_inj_separate, 2)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_inj_Q - 1
            q_Q_inj = []

            q_Q_inj.append((np.array(q_inj_separate)[:, :, idx_sim_ts] - np.array(Q_inj_separate)[:, :, idx_obs_ts])
                           * self.dirac_vec[idx_sim_ts]
                           )

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1

                q_Q_inj.append((np.array(q_inj_separate)[:, :, idx_sim_ts] - np.array(Q_inj_separate)[:, :, idx_obs_ts])
                               * self.dirac_vec[idx_sim_ts]
                               )

            q_Q_inj_clean = np.array(q_Q_inj)[np.flipud(self.dirac_vec).astype('bool'), :].T


            inj_clean_training = np.sum(q_Q_inj_clean[:, :, 0:train_lenght]**2)
            INJ_L2_training = np.sum((np.array(Q_inj_separate)[:, :, 0:train_lenght]
                                      )**2)

            inj_clean_forcast = np.sum(q_Q_inj_clean[:, :, train_lenght:]**2)
            INJ_L2_forcast = np.sum((np.array(Q_inj_separate)[:, :, train_lenght:]
                                     )**2)

            inj_clean_overall = np.sum(q_Q_inj_clean**2)
            INJ_L2_overall = np.sum((np.array(Q_inj_separate)
                                     )**2)

            err_training_inj = inj_clean_training / INJ_L2_training * 100
            err_forcast_inj = inj_clean_forcast / INJ_L2_forcast * 100
            err_overall_inj = inj_clean_overall / INJ_L2_overall * 100


            print('\n')
            print('Injction Phase---Relative Error Training: ', err_training_inj)
            print('Injction Phase---Relative Error Forecast: ', err_forcast_inj)
            print('Injction Phase---Relative Error Overall: ', err_overall_inj)

        # BHP data --------------------------------------------------------------------------------------------------
        bhp_clean_training = 0
        BHP_L2_training = 0
        bhp_clean_forcast = 0
        BHP_L2_forcast = 0
        bhp_clean_overall = 0
        BHP_L2_overall = 0
        if self.objfun_BHP:
            bhp_separate = []
            BHP_separate = []

            for n, well in enumerate(self.BHP_well_name):
                BHP_string = well + " : BHP (bar)"
                bhp_serie = opt_df.get(BHP_string)
                BHP_serie = truth_df_BT.get(BHP_string)

                bhp_separate.append(bhp_serie.values)
                BHP_separate.append(BHP_serie.values)

            bhp_separate = np.array(bhp_separate)
            BHP_separate = np.array(BHP_separate)

            t_sim = opt_df['time']
            t_Q = truth_df_BT['time']
            self.dirac_vec = np.zeros(np.size(t_sim))
            for i, ts in enumerate(t_sim):
                for tr in t_Q:
                    if ts == tr:
                        self.dirac_vec[i] = 1

            TotStep = np.size(t_sim)
            size_BHP = np.size(BHP_separate, 1)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_BHP - 1

            bhp_BHP = []
            bhp_BHP.append((np.array(bhp_separate)[:, idx_sim_ts] - np.array(BHP_separate)[:, idx_obs_ts])
                           * self.dirac_vec[idx_sim_ts]
                           )

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1
                bhp_BHP.append((np.array(bhp_separate)[:, idx_sim_ts] - np.array(BHP_separate)[:, idx_obs_ts])
                               * self.dirac_vec[idx_sim_ts]
                               )

            bhp_BHP_clean = np.array(bhp_BHP)[np.flipud(self.dirac_vec).astype('bool'), :].T


            bhp_clean_training = sq_norm(bhp_BHP_clean[:, 0:train_lenght])
            BHP_L2_training = sq_norm((np.array(BHP_separate)[:, 0:train_lenght]
                                      ))

            bhp_clean_forcast = sq_norm(bhp_BHP_clean[:, train_lenght:])
            BHP_L2_forcast = sq_norm((np.array(BHP_separate)[:, train_lenght:]
                                     ))

            bhp_clean_overall = sq_norm(bhp_BHP_clean)
            BHP_L2_overall = sq_norm((np.array(BHP_separate)
                                     ))

            err_training_BHP = bhp_clean_training / BHP_L2_training * 100
            err_forcast_BHP = bhp_clean_forcast / BHP_L2_forcast * 100
            err_overall_BHP = bhp_clean_overall / BHP_L2_overall * 100

            print('\n')
            print('BHP---Relative Error Training: ', err_training_BHP)
            print('BHP---Relative Error Forecast: ', err_forcast_BHP)
            print('BHP---Relative Error Overall: ', err_overall_BHP)

        # well temperature --------------------------------------------------------------------------------------------
        wt_clean_training = 0
        WT_L2_training = 0
        wt_clean_forcast = 0
        WT_L2_forcast = 0
        wt_clean_overall = 0
        WT_L2_overall = 0
        if self.objfun_well_tempr:
            wt_separate = []
            WT_separate = []

            for n, well in enumerate(self.well_tempr_name):
                well_tempr_string = well + " : temperature (K)"
                wt_serie = opt_df.get(well_tempr_string)
                WT_serie = truth_df_BT.get(well_tempr_string)

                wt_separate.append(wt_serie.values)
                WT_separate.append(WT_serie.values)

            wt_separate = np.array(wt_separate)
            WT_separate = np.array(WT_separate)

            t_sim = opt_df['time']
            t_Q = truth_df_BT['time']
            self.dirac_vec = np.zeros(np.size(t_sim))
            for i, ts in enumerate(t_sim):
                for tr in t_Q:
                    if ts == tr:
                        self.dirac_vec[i] = 1

            TotStep = np.size(t_sim)
            size_WT = np.size(WT_separate, 1)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_WT - 1

            wt_WT = []
            wt_WT.append((np.array(wt_separate)[:, idx_sim_ts] - np.array(WT_separate)[:, idx_obs_ts])
                           * self.dirac_vec[idx_sim_ts]
                           )

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1
                wt_WT.append((np.array(wt_separate)[:, idx_sim_ts] - np.array(WT_separate)[:, idx_obs_ts])
                               * self.dirac_vec[idx_sim_ts]
                               )

            wt_WT_clean = np.array(wt_WT)[np.flipud(self.dirac_vec).astype('bool'), :].T


            wt_clean_training = sq_norm(wt_WT_clean[:, 0:train_lenght])
            WT_L2_training = sq_norm((np.array(WT_separate)[:, 0:train_lenght]
                                      ))

            wt_clean_forcast = sq_norm(wt_WT_clean[:, train_lenght:])
            WT_L2_forcast = sq_norm((np.array(WT_separate)[:, train_lenght:]
                                     ))

            wt_clean_overall = sq_norm(wt_WT_clean)
            WT_L2_overall = sq_norm((np.array(WT_separate)
                                     ))

            err_training_well_tempr = wt_clean_training / WT_L2_training * 100
            err_forcast_well_tempr = wt_clean_forcast / WT_L2_forcast * 100
            err_overall_well_tempr = wt_clean_overall / WT_L2_overall * 100

            print('\n')
            print('well_tempr---Relative Error Training: ', err_training_well_tempr)
            print('well_tempr---Relative Error Forecast: ', err_forcast_well_tempr)
            print('well_tempr---Relative Error Overall: ', err_overall_well_tempr)

        # temperature data --------------------------------------------------------------------------------------------
        temperature_clean_training = 0
        TEMPERATURE_L2_training = 0
        temperature_clean_forcast = 0
        TEMPERATURE_L2_forcast = 0
        temperature_clean_overall = 0
        TEMPERATURE_L2_overall = 0
        if self.objfun_temperature:

            temperature_separate = opt_tempr
            TEMPERATURE_separate = truth_TEMPR

            t_sim = opt_df['time']
            t_Q = truth_df['time']
            self.dirac_vec = np.zeros(np.size(t_sim))
            for i, ts in enumerate(t_sim):
                for tr in t_Q:
                    if ts == tr:
                        self.dirac_vec[i] = 1

            TotStep = np.size(t_sim)
            size_TEMPR = np.size(TEMPERATURE_separate, 0)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_TEMPR - 1

            tempr_TEMPR = []
            tempr_TEMPR.append((np.array(temperature_separate)[idx_sim_ts, :] - np.array(TEMPERATURE_separate)[idx_obs_ts, :])
                               * self.dirac_vec[idx_sim_ts]
                               )

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1
                tempr_TEMPR.append((np.array(temperature_separate)[idx_sim_ts, :] - np.array(TEMPERATURE_separate)[idx_obs_ts, :])
                                   * self.dirac_vec[idx_sim_ts]
                                   )

            tempr_TEMPR_clean = np.array(tempr_TEMPR)[np.flipud(self.dirac_vec).astype('bool'), :].T


            temperature_clean_training = sq_norm(tempr_TEMPR_clean[0:train_lenght, :])
            TEMPERATURE_L2_training = sq_norm((np.array(TEMPERATURE_separate)[0:train_lenght, :]
                                              ))

            temperature_clean_forcast = sq_norm(tempr_TEMPR_clean[train_lenght:, :])
            TEMPERATURE_L2_forcast = sq_norm((np.array(TEMPERATURE_separate)[train_lenght:, :]
                                             ))

            temperature_clean_overall = sq_norm(tempr_TEMPR_clean)
            TEMPERATURE_L2_overall = sq_norm((np.array(TEMPERATURE_separate)
                                             ))

            err_training_tempr = temperature_clean_training / TEMPERATURE_L2_training * 100
            err_forcast_tempr = temperature_clean_forcast / TEMPERATURE_L2_forcast * 100
            err_overall_tempr = temperature_clean_overall / TEMPERATURE_L2_overall * 100
            print('\n')
            print('Temperature---Relative Error Training: ', err_training_tempr)
            print('Temperature---Relative Error Forecast: ', err_forcast_tempr)
            print('Temperature---Relative Error Overall: ', err_overall_tempr)


        # all data including prod, inj, BHP, well_tempr, tempr, etc.--------------------------------------------------
        err_training_total = (prod_clean_training + inj_clean_training + bhp_clean_training + wt_clean_training + temperature_clean_training) \
                             / (PROD_L2_training + INJ_L2_training + BHP_L2_training + WT_L2_training + TEMPERATURE_L2_training) * 100

        err_forcast_total = (prod_clean_forcast + inj_clean_forcast + bhp_clean_forcast + wt_clean_forcast + temperature_clean_forcast) \
                            / (PROD_L2_forcast + INJ_L2_forcast + BHP_L2_forcast + WT_L2_forcast + TEMPERATURE_L2_forcast) * 100

        err_overall_total = (prod_clean_overall + inj_clean_overall + bhp_clean_overall + wt_clean_overall + temperature_clean_overall) \
                            / (PROD_L2_overall + INJ_L2_overall + BHP_L2_overall + WT_L2_overall + TEMPERATURE_L2_overall) * 100
        print('\n')
        print('Total---Relative Error Training: ', err_training_total)
        print('Total---Relative Error Forecast: ', err_forcast_total)
        print('Total---Relative Error Overall: ', err_overall_total)

        # customized operator data --------------------------------------------------------------------------------------------
        customized_op_clean_training = 0
        CUSTOMIZED_OP_L2_training = 0
        customized_op_clean_forcast = 0
        CUSTOMIZED_OP_L2_forcast = 0
        customized_op_clean_overall = 0
        CUSTOMIZED_OP_L2_overall = 0
        if self.objfun_customized_op:

            customized_op_separate = opt_op
            CUSTOMIZED_OP_separate = truth_OP

            t_sim = opt_df['time']
            t_Q = truth_df['time']
            self.dirac_vec = np.zeros(np.size(t_sim))
            for i, ts in enumerate(t_sim):
                for tr in t_Q:
                    if ts == tr:
                        self.dirac_vec[i] = 1

            TotStep = np.size(t_sim)
            size_OP = np.size(CUSTOMIZED_OP_separate, 0)

            idx_sim_ts = TotStep - 1
            idx_obs_ts = size_OP - 1

            op_OP = []
            op_OP.append((np.array(customized_op_separate)[idx_sim_ts, :] - np.array(CUSTOMIZED_OP_separate)[idx_obs_ts, :])
                               * self.dirac_vec[idx_sim_ts]
                               )

            for idx_sim_ts in range(TotStep - 2, -1, -1):
                if self.dirac_vec[idx_sim_ts] == 1:
                    idx_obs_ts -= 1
                op_OP.append((np.array(customized_op_separate)[idx_sim_ts, :] - np.array(CUSTOMIZED_OP_separate)[idx_obs_ts, :])
                                   * self.dirac_vec[idx_sim_ts]
                                   )

            op_OP_clean = np.array(op_OP)[np.flipud(self.dirac_vec).astype('bool'), :].T

            customized_op_clean_training = sq_norm(op_OP_clean[0:train_lenght, :])
            CUSTOMIZED_OP_L2_training = sq_norm((np.array(CUSTOMIZED_OP_separate)[0:train_lenght, :]
                                             ))

            customized_op_clean_forcast = sq_norm(op_OP_clean[train_lenght:, :])
            CUSTOMIZED_OP_L2_forcast = sq_norm((np.array(CUSTOMIZED_OP_separate)[train_lenght:, :]
                                            ))

            customized_op_clean_overall = sq_norm(op_OP_clean)
            CUSTOMIZED_OP_L2_overall = sq_norm((np.array(CUSTOMIZED_OP_separate)
                                             ))

            err_training_op = customized_op_clean_training / CUSTOMIZED_OP_L2_training * 100
            err_forcast_op = customized_op_clean_forcast / CUSTOMIZED_OP_L2_forcast * 100
            err_overall_op = customized_op_clean_overall / CUSTOMIZED_OP_L2_overall * 100
            print('\n')
            print('Temperature---Relative Error Training: ', err_training_op)
            print('Temperature---Relative Error Forecast: ', err_forcast_op)
            print('Temperature---Relative Error Overall: ', err_overall_op)

        # all data including prod, inj, BHP, well_tempr, tempr, etc.--------------------------------------------------
        err_training_total = (prod_clean_training + inj_clean_training + bhp_clean_training + wt_clean_training + temperature_clean_training +  customized_op_clean_training) \
                             / (PROD_L2_training + INJ_L2_training + BHP_L2_training + WT_L2_training + TEMPERATURE_L2_training + CUSTOMIZED_OP_L2_training) * 100

        err_forcast_total = (prod_clean_forcast + inj_clean_forcast + bhp_clean_forcast + wt_clean_forcast + temperature_clean_forcast + customized_op_clean_forcast) \
                            / (PROD_L2_forcast + INJ_L2_forcast + BHP_L2_forcast + WT_L2_forcast + TEMPERATURE_L2_forcast + CUSTOMIZED_OP_L2_forcast) * 100

        err_overall_total = (prod_clean_overall + inj_clean_overall + bhp_clean_overall + wt_clean_overall + temperature_clean_overall + customized_op_clean_overall) \
                            / (PROD_L2_overall + INJ_L2_overall + BHP_L2_overall + WT_L2_overall + TEMPERATURE_L2_overall + CUSTOMIZED_OP_L2_overall) * 100
        print('\n')
        print('Total---Relative Error Training: ', err_training_total)
        print('Total---Relative Error Forecast: ', err_forcast_total)
        print('Total---Relative Error Overall: ', err_overall_total)




class model_modifier_aggregator:
    '''
    The class of modifier definition
    This class includes the functions of modifier initialization, getting the initial guess, setting the updated modifers, etc.
    '''

    def __init__(self):
        # list with modifiers
        self.modifiers = []

        # array of unknowns to dump for restart
        self.x = 0

        # array of indexes of x for each modifier in aggregated x array
        self.mod_x_idx = np.zeros(1, int)

    def append(self, modifier):
        self.modifiers.append(modifier)

    def get_x0(self, model) -> np.array:
        '''
        The function of getting the initial guess of the modifier
        :param model: DARTS reservoir model
        :return: initial guess of the modifer
        '''
        x0 = np.zeros(0)
        for m in self.modifiers:
            x0 = np.append(x0, m.get_x0(model))
            self.mod_x_idx = np.append(self.mod_x_idx, len(x0))
        return x0

    def get_random_x0(self, model) -> np.array:
        b = np.array(self.get_bounds(model))
        low = b[:, 0]
        high = b[:, 1]
        x = np.random.rand(low.size) * (high - low) + low
        return np.array(x)

    def get_random_x0_in_neighbourhood(self, model, rel_distance=0.1) -> np.array:
        b = np.array(self.get_bounds_around_x0_neighbourhood(model, rel_distance))
        low = b[:, 0]
        high = b[:, 1]
        x = np.random.rand(low.size) * (high - low) + low
        return np.array(x)

    def get_bounds_around_x0_neighbourhood(self, model, rel_distance=0.1) -> List[tuple]:
        x0 = self.get_x0(model)
        l = list(x0 * (1 - rel_distance))
        h = list(x0 * (1 + rel_distance))
        bounds = list(zip(l, h))
        return bounds

    def get_bounds(self, model) -> List[tuple]:
        '''
        The function of getting the bound of the modifier
        :param model: DARTS reservoir model
        :return: bound of the modifer
        '''
        bounds = []
        for m in self.modifiers:
            bounds.extend(m.get_bounds(model))
        return bounds

    def get_x(self, model) -> np.array:
        x = np.array([])
        for i, m in enumerate(self.modifiers):
            x = np.append(x, m.get_x(model))
        return x

    def set_x(self, model, x: np.array):
        '''
        The function of setting the updated modifier
        :param model: DARTS reservoir model
        :param x: updated control variables
        '''
        self.x = x
        for i, m in enumerate(self.modifiers):
            m.set_x(model, x[self.mod_x_idx[i]: self.mod_x_idx[i + 1]])

    def load_restart_data(self, model, filename='restart_mm_agg.pkl'):
        if osp.exists(filename):
            with open(filename, "rb") as fp:
                [self.x, self.mod_x_idx] = pickle.load(fp)
                self.set_x(model, self.x)

    def save_restart_data(self, filename='restart_mm_agg.pkl'):
        with open(filename, "wb") as fp:
            pickle.dump([self.x, self.mod_x_idx], fp, pickle.HIGHEST_PROTOCOL)



    def save_opt_model_params(self, filename='optimized_model_params.pkl'):
        with open(filename, "wb") as fp:
            pickle.dump([self.x, self.mod_x_idx, self.modifiers], fp, pickle.HIGHEST_PROTOCOL)

    def set_grad(self, grad_original: np.array, x_idx: List[int]) -> np.array:
        '''
        The function of setting the gradients
        :param grad_original: original gradient array
        :param x_idx: list of the index of the control variables, including linear gradients (e.g. transmissibility and well index) and nonlinear gradients
        :return: gradients
        '''
        grad = np.zeros(0)

        for i, m in enumerate(self.modifiers):
            if type(m) == transmissibility_modifier or type(m) == well_index_modifier:  # linear modifiers
                left_idx = x_idx[i]
                right_idx = x_idx[i+1]
                temp_grad = m.set_grad(grad_original[left_idx: right_idx])
                grad = np.append(grad, temp_grad)
            if type(m) == flux_multiplier_modifier:
                temp_grad = m.set_grad(grad_original)
                grad = np.append(grad, temp_grad)


        return grad

    def set_x_by_du_dT(self, model, x: np.array):
        '''
        The function of setting the updated modifier by correct index
        :param model: DARTS reservoir model
        :param x: updated control variables
        '''
        self.x = x
        if type(self.modifiers[0]) == flux_multiplier_modifier:  # for MPFA
            x_linear = x[:self.mod_x_idx[1]]

            for i, m in enumerate(self.modifiers):
                if type(self.modifiers[0]) == flux_multiplier_modifier:  # linear modifiers
                    m.set_x(model, x_linear[self.mod_x_idx[i]: self.mod_x_idx[i + 1]])
                else:  # nonlinear modifiers
                    m.set_x(model, x[self.mod_x_idx[i]: self.mod_x_idx[i + 1]])
        else:
            # prepare x_linear by du_dT
            if type(self.modifiers[0]) == transmissibility_modifier and type(self.modifiers[1]) == well_index_modifier:
                # x_temp = x[:self.mod_x_idx[2]]
                # x_linear = np.dot(x_temp, model.du_dT)
                x_linear = x[:self.mod_x_idx[2]]

            for i, m in enumerate(self.modifiers):
                if type(m) == transmissibility_modifier or type(m) == well_index_modifier:  # linear modifiers
                    m.set_x(model, x_linear[self.mod_x_idx[i]: self.mod_x_idx[i + 1]])
                else:  # nonlinear modifiers
                    m.set_x(model, x[self.mod_x_idx[i]: self.mod_x_idx[i + 1]])



    def load_restart_data_by_du_dT(self, model, filename='restart_mm_agg.pkl'):
        if osp.exists(filename):
            with open(filename, "rb") as fp:
                [self.x, self.mod_x_idx] = pickle.load(fp)[0:2]
                self.set_x_by_du_dT(model, self.x)

    def get_x0_in_bounds(self, model) -> np.array:
        b = np.array(self.get_bounds(model))
        low = b[:, 0]
        high = b[:, 1]
        # x = np.random.rand(low.size) * (high - low) + low
        x = 0.5 * (high - low) + low
        return np.array(x)


class transmissibility_modifier:
    '''
    The class of transmissibility modifier
    This class includes the functions of modifier initialization, getting the initial guess, setting the updated modifers, etc.
    '''
    def __init__(self):
        # self.norms = 1000
        self.norms = 50000

    def get_x0(self, model) -> np.array:
        '''
        The function of getting the initial guess of transmissibility modifier
        :param model: DARTS reservoir model
        :return: initial guess of transmissibility modifier
        '''
        t = value_vector([])
        t_D = value_vector([])

        model.reservoir.mesh.get_res_tran(t, t_D)
        self.t = t

        return np.array(t) / self.norms

    def get_bounds(self, model, mult=10) -> List[tuple]:
        '''
        The function of getting the bound of transmissibility modifier
        :param model: DARTS reservoir model
        :return: bound of transmissibility modifer
        '''
        bound = list()
        for i in range(0, len(self.t)):
            bound += [(0.00002, self.t[i] * mult / self.norms)]

        # bound = [(1e-8, 100)] * len(self.t)
        self.bound = bound
        return bound

    def set_x(self, model, x: np.array):
        '''
        The function of setting the updated transmissibility
        :param model: DARTS reservoir model
        :param x: updated transmissibility
        '''
        tran = x * self.norms
        tranD = tran
        model.reservoir.mesh.set_res_tran(value_vector(tran), value_vector(tranD))

    def set_grad(self, grad_original: np.array) -> np.array:
        '''
        The function of setting the gradients of transmissibility
        :param grad_original: original gradient array
        :return: gradients of transmissibility
        '''
        return grad_original * self.norms

class flux_multiplier_modifier:
    '''
    The class of flux multiplier modifier
    This class includes the functions of modifier initialization, getting the initial guess, setting the updated modifers, etc.
    '''
    def __init__(self):
        self.norms = 1

    def get_x0(self, model) -> np.array:
        '''
        The function of getting the initial guess of flux multiplier modifier
        :param model: DARTS reservoir model
        :return: initial guess of flux multiplier modifier
        '''
        n_fm = model.get_n_flux_multiplier()

        # n_well_index = 0
        # for well in model.reservoir.wells:
        #     n_well_index += len(well.perforations)
        # n_fm_res = n_fm - n_well_index
        # self.fm = np.ones(n_fm_res)

        self.fm = np.ones(n_fm)

        return self.fm / self.norms

    def get_bounds(self, model, mult=10) -> List[tuple]:
        '''
        The function of getting the bound of flux multiplier modifier
        :param model: DARTS reservoir model
        :return: bound of flux multiplier modifer
        '''
        bound = list()
        for i in range(0, len(self.fm)):
            bound += [(0.00002, self.fm[i] * mult / self.norms)]

        self.bound = bound
        return bound

    def set_x(self, model, x: np.array):
        '''
        The function of setting the updated flux multiplier
        :param model: DARTS reservoir model
        :param x: updated flux multiplier
        '''
        fm = x * self.norms

        # always keep the fm between well head and well body equals to 1
        fm_full = np.ones(np.size(fm) + len(model.reservoir.wells))
        fm_full[0:np.size(fm)] = fm

        model.engine.flux_multiplier = value_vector(fm_full)

    def set_grad(self, grad_original: np.array) -> np.array:
        '''
        The function of setting the gradients of flux multiplier
        :param grad_original: original gradient array
        :return: gradients of flux multiplier
        '''
        return grad_original * self.norms

class transmissibility_fracture_modifier:
    '''
    The legacy code of Mark Khait
    The class of facture transmissibility modifier
    '''
    def __init__(self, nr_frac_frac_con):
        self.norms = 5000000
        self.nr_frac_frac_con = nr_frac_frac_con


    def get_x0(self, model) -> np.array:

        t = value_vector([])
        t_D = value_vector([])

        model.reservoir.mesh.get_res_tran(t, t_D)
        self.t = t
        self.t_D = t_D

        return np.array(t[0:self.nr_frac_frac_con]) / self.norms

    def get_bounds(self, model, mult=10) -> List[tuple]:
        bound = list()
        for i in range(0, len(self.t[0:self.nr_frac_frac_con])):
            bound += [(0.00002, self.t[i] * mult / self.norms)]
        # bound = [(0.0001, 1)] * len(self.t)
        return bound

    def set_x(self, model, x: np.array):
        tran = self.t
        tran[0:self.nr_frac_frac_con] = value_vector(x * self.norms)
        tranD = self.t_D

        model.reservoir.mesh.set_res_tran(value_vector(tran), value_vector(tranD))


class well_index_modifier:
    '''
    The class of well index modifier
    This class includes the functions of modifier initialization, getting the initial guess, setting the updated modifers, etc.
    '''
    def __init__(self):
        # self.norms = 100
        self.norms = 1000

    def get_x0(self, model) -> np.array:
        '''
        The function of getting the initial guess of well index modifier
        :param model: DARTS reservoir model
        :return: initial guess of well index modifier
        '''
        well_index = value_vector([])
        model.reservoir.mesh.get_wells_tran(well_index)
        self.wi = well_index

        return np.array(well_index) / self.norms

    def get_bounds(self, model, mult=10) -> List[tuple]:
        '''
        The function of getting the bound of well index modifier
        :param model: DARTS reservoir model
        :return: bound of well index modifer
        '''
        bound = list()
        # for i in range(0, len(self.wi)):
        #     bound += [(0.00002, self.wi[i] * mult / self.norms)]
        bound = [(0.00001, 5)] * len(self.wi)

        return bound

    def set_x(self, model, x: np.array):
        '''
        The function of setting the updated well index
        :param model: DARTS reservoir model
        :param x: updated well index
        '''
        well_index = x * self.norms
        model.reservoir.mesh.set_wells_tran(value_vector(well_index))

    def set_grad(self, grad_original: np.array) -> np.array:
        '''
        The function of setting the gradients of well index
        :param grad_original: original gradient array
        :return: gradients of well index
        '''
        return grad_original * self.norms




#-----------------------------------------------------------------------------------------------------------------------
# Legacy code for multiprocessing, will be depricated
#-----------------------------------------------------------------------------------------------------------------------
def grad_linear_adjoint_method(model, x):
    return model.grad_linear_adjoint_method(x)

def fval_nonlinear_FDM(model, x):
    return model.fval_nonlinear_FDM(x)
