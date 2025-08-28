#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>
#include <functional>  // adjoint method -- function 'bind1st'
#ifdef __GNUC__
#include <cxxabi.h>
#endif

#include "engine_base.h"


#ifdef OPENDARTS_LINEAR_SOLVERS
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/linsolv_iface.hpp"
#else
#include "csr_matrix.h"
#include "linsolv_iface.h"
#endif // OPENDARTS_LINEAR_SOLVERS

#ifdef OPENDARTS_LINEAR_SOLVERS
using namespace opendarts::auxiliary;
using namespace opendarts::linear_solvers;
#endif // OPENDARTS_LINEAR_SOLVERS  

int engine_base::print_header()
{
	std::cout << "Engine: \t" << engine_name << "\n";
#ifdef _OPENMP
	std::cout << "OpenMP threads: \t" << omp_get_max_threads() << std::endl;
#endif
	//  std::cout << "\tResolution: \t" << acc_flux_op_set->axis_points[0] << std::endl;

	return 0;
}

int engine_base::init_jacobian_structure(csr_matrix_base *jacobian)
{
	const char n_vars = get_n_vars();

	// init Jacobian structure
	index_t *rows_ptr = jacobian->get_rows_ptr();
	index_t *diag_ind = jacobian->get_diag_ind();
	index_t *cols_ind = jacobian->get_cols_ind();
	index_t *row_thread_starts = jacobian->get_row_thread_starts();

	index_t n_blocks = mesh->n_blocks;
	index_t n_conns = mesh->n_conns;
	std::vector<index_t> &block_m = mesh->block_m;
	std::vector<index_t> &block_p = mesh->block_p;

#ifdef _OPENMP
#pragma omp parallel
	{
		int id, nt;
		index_t local, start, end;

		id = omp_get_thread_num();
		nt = omp_get_num_threads();
		start = row_thread_starts[id];
		end = row_thread_starts[id + 1];
		// 'first touch' rows_ptr, cols_ind, diag_ind

		numa_set(rows_ptr, 0, start, end);
		// since the length of rows_ptr is n_blocks+1, take care of the last entry (using last thread)
		if (id == nt - 1)
		{
			rows_ptr[n_blocks] = 0;
		}
		numa_set(diag_ind, -1, start, end);
	}
#else
	rows_ptr[0] = 0;
	memset(diag_ind, -1, n_blocks * sizeof(index_t)); // t_long <-----> index_t
#endif
	// now we have to split the work into two loops
	// 1. Find out rows_ptr
	index_t j = 0, k = 0;
	for (index_t i = 0; i < n_blocks; i++)
	{
		rows_ptr[i + 1] = rows_ptr[i];
		for (; j < n_conns && block_m[j] == i; j++)
		{
			rows_ptr[i + 1]++;
			if (diag_ind[i] < 0 && block_p[j] > i)
			{
				diag_ind[i] = k++;
				rows_ptr[i + 1]++;
			}
			k++;
		}
		if (diag_ind[i] < 0)
		{
			diag_ind[i] = k++;
			rows_ptr[i + 1]++;
		}
	}

	// Now we know rows_ptr and can 'first touch' cols_ind
#ifdef _OPENMP
#pragma omp parallel
	{
		int id, nt;
		index_t local, start, end;

		id = omp_get_thread_num();
		start = row_thread_starts[id];
		end = row_thread_starts[id + 1];

		start = rows_ptr[start];
		end = rows_ptr[end];
		numa_set(cols_ind, 0, start, end);
	}
#endif

	// 2. Write cols_ind
	j = 0, k = 0;
	for (index_t i = 0; i < n_blocks; i++)
	{
		for (; j < n_conns && block_m[j] == i; j++)
		{
			if (diag_ind[i] == k)
			{
				cols_ind[k++] = i;
			}
			cols_ind[k++] = block_p[j];
		}

		if (diag_ind[i] == k)
		{
			cols_ind[k++] = i;
		}
	}

	//Jacobian.write_matrix_to_file ("jac_struct.csr");

	//cpr_prec.init (&Jacobian, 0, 0);
	return 0;
}



int
engine_base::init_adjoint_structure(csr_matrix_base* init_adjoint)
{
	const char n_vars = get_n_vars();

	index_t n_blocks = mesh->n_blocks;
	index_t n_conns = mesh->n_conns;
	std::vector <index_t>& block_m = mesh->block_m;
	std::vector <index_t>& block_p = mesh->block_p;

	//csr_matrix_base* ad_temp = 0;
	//ad_temp = new csr_matrix<1>;
	//ad_temp->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
	//index_t* rows = ad_temp->get_rows_ptr();
	//index_t* diag_ind = ad_temp->get_diag_ind();
	//index_t* cols = ad_temp->get_cols_ind();
	//index_t* row_thread_starts = ad_temp->get_row_thread_starts();

	//index_t temp = n_blocks * 1000;
	//index_t* rows = new index_t[temp];
	//index_t* diag_ind = new index_t[temp];
	//index_t* cols = new index_t[temp];
	//index_t* row_thread_starts = 0;

	//index_t* rows = new index_t[n_blocks];
	//index_t* diag_ind = new index_t[n_blocks];
	//index_t* cols = new index_t[n_conns + n_blocks];
	//index_t* row_thread_starts = 0;

	std::vector <index_t> rows(n_blocks + 1, 0); // plus 1 because CSR requires row pointer start from 0 to n_blocks
	//index_t* diag_ind = new index_t[n_blocks];
	std::vector <index_t> diag_ind(n_blocks, -1);
	std::vector <index_t> cols(n_conns + n_blocks, 0);
	std::vector <index_t> row_thread_starts(1, 0);


//#ifdef _OPENMP
//#pragma omp parallel
//	{
//		int id, nt;
//		index_t local, start, end;
//
//		id = omp_get_thread_num();
//		nt = omp_get_num_threads();
//		start = row_thread_starts[id];
//		end = row_thread_starts[id + 1];
//		// 'first touch' rows_ptr, cols_ind, diag_ind
//
//
//
//		//numa_set(rows_ptr, 0, start, end);
//		//// since the length of rows_ptr is n_blocks+1, take care of the last entry (using last thread)
//		//if (id == nt - 1)
//		//{
//		//	rows_ptr[n_blocks] = rows_ptr[n_blocks];
//		//}
//		//numa_set(diag_ind, -1, start, end);
//
//
//
//		numa_set(&rows, 0, start, end);
//		// since the length of rows_ptr is n_blocks+1, take care of the last entry (using last thread)
//		if (id == nt - 1)
//		{
//			rows[n_blocks] = rows[n_blocks];
//		}
//		numa_set(&diag_ind, -1, start, end);
//	}
//#else
//	rows[0] = 0;
//	//memset(diag_ind, -1, n_blocks * sizeof(index_t)); // t_long <-----> index_t
//#endif


	// now we have to split the work into two loops
	// 1. Find out rows_ptr
	index_t j = 0, k = 0;
	for (index_t i = 0; i < n_blocks; i++)
	{
		rows[i + 1] = rows[i];
		for (; j < n_conns && block_m[j] == i; j++)
		{
			rows[i + 1]++;
			if (diag_ind[i] < 0 && block_p[j] > i)
			{
				diag_ind[i] = k++;
				rows[i + 1]++;
			}
			k++;
		}
		if (diag_ind[i] < 0)
		{
			diag_ind[i] = k++;
			rows[i + 1]++;
		}
	}

//	// Now we know rows_ptr and can 'first touch' cols_ind
//#ifdef _OPENMP
//#pragma omp parallel
//	{
//		int id, nt;
//		index_t local, start, end;
//
//		id = omp_get_thread_num();
//		start = row_thread_starts[id];
//		end = row_thread_starts[id + 1];
//
//
//		//start = rows_ptr[start];
//		//end = rows_ptr[end];
//		//numa_set(cols_ind, 0, start, end);
//
//
//		start = rows[start];
//		end = rows[end];
//		numa_set(&cols, 0, start, end);
//	}
//#endif



	// 2. Write cols_ind
	j = 0, k = 0;
	for (index_t i = 0; i < n_blocks; i++)
	{
		for (; j < n_conns && block_m[j] == i; j++)
		{
			if (diag_ind[i] == k)
			{
				cols[k++] = i;
			}
			cols[k++] = block_p[j];
		}

		if (diag_ind[i] == k)
		{
			cols[k++] = i;
		}
	}


	if (init_adjoint->is_square) // dg_dx_n
	{
		index_t* ad_rows = init_adjoint->get_rows_ptr();
		index_t* ad_diag_ind = init_adjoint->get_diag_ind();
		index_t* ad_cols = init_adjoint->get_cols_ind();
		index_t* ad_row_thread_starts = init_adjoint->get_row_thread_starts();


		//index_t n_value = mesh->n_conns + mesh->n_blocks;
		//std::memcpy(ad_rows, rows, (n_blocks + 1) * sizeof(ad_rows));
		//std::memcpy(ad_diag_ind, diag_ind, n_blocks * sizeof(ad_diag_ind));
		//std::memcpy(ad_cols, cols, n_value * sizeof(ad_cols));


		index_t n_value = mesh->n_conns + mesh->n_blocks;
        //memcpy(ad_cols, &cols[0], n_value * sizeof(ad_cols)); // still have some problems when using memcpy
        memcpy(ad_cols, &cols[0], n_value * sizeof(index_t)); 
		//for (index_t i = 0; i < n_value; i++)
		//{
		//	ad_cols[i] = cols[i];
		//}


        memcpy(ad_rows, &rows[0], (n_blocks + 1) * sizeof(index_t));
        memcpy(ad_diag_ind, &diag_ind[0], n_blocks * sizeof(index_t));
		//for (index_t i = 0; i < n_blocks; i++)
		//{
		//	ad_rows[i] = rows[i];
		//	ad_diag_ind[i] = diag_ind[i];
		//}
		//ad_rows[n_blocks] = rows[n_blocks];


	}
	else // dg_dT_general
	{

		index_t n_element;
		index_t NC = n_vars;
		std::vector <index_t>& conn_index_to_one_way = mesh->conn_index_to_one_way;
		index_t tran_idx = 0;
		// init adjoint structure
		index_t* rows_ptr_general = init_adjoint->get_rows_ptr();
		//index_t* diag_ind_general = init_adjoint->get_diag_ind();
		index_t* cols_ind_general = init_adjoint->get_cols_ind();
		//index_t* row_thread_starts_general = init_adjoint->get_row_thread_starts();

		rows_ptr_general[0] = 0;

		// now we have to split the work into two loops
		// 1. Find out rows_ptr_general
		index_t count = 0;
		j = 0; k = 0; n_element = 0;
		for (index_t i = 0; i < n_blocks; i++)
		{
			n_element = rows[i + 1] - rows[i] - 1;

			for (index_t m = 0; m < n_vars; m++)
			{
				rows_ptr_general[k + 1] = rows_ptr_general[k];
				rows_ptr_general[k + 1] += n_element;
				k++;
			}
		}

		// 2. Write cols_ind
		k = 0;
		index_t N_element = 0;
		index_t k_count = 0;
		index_t cc;
		std::vector<index_t> temp_conn_one_way;
		std::vector<index_t> temp_num;
		for (index_t i = 0; i < n_blocks; ++i)
		{
			n_element = rows[i + 1] - rows[i] - 1;

			// index of first entry for block i in CSR cols array
			index_t csr_idx_start = rows[i];
			// index of last entry for block i in CSR cols array
			index_t csr_idx_end = rows[i + 1];
			// index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
			index_t conn_idx = csr_idx_start - i;


			N_element = rows[i + 1] - rows[i] - 1;
			temp_conn_one_way.clear();
			temp_num.clear();
			for (index_t m = 0; m < N_element; m++)
			{
				temp_conn_one_way.push_back(conn_index_to_one_way[conn_idx + m]);
				temp_num.push_back(0);
			}

			for (index_t m = 0; m < N_element; m++)
			{
				for (index_t com : temp_conn_one_way)
				{
					if (com < temp_conn_one_way[m])
						temp_num[m] += 1;
				}
			}

			k_count = 0;

			for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++)
			{
				j = cols[csr_idx];
				// skip diagonal
				if (i == j)
				{
					continue;
				}

				tran_idx = conn_index_to_one_way[conn_idx];

				for (uint8_t c = 0; c < NC; c++)
				{
					cc = count + c * N_element + temp_num[k_count];
					cols_ind_general[cc] = tran_idx;
				}
				k_count++;


				conn_idx++;
			}


			count += NC * N_element;
		}
	}


	//delete[]rows;
	//delete[]diag_ind;
	//delete[]cols;
	//delete[]row_thread_starts;

	//_ASSERT(_CrtCheckMemory());
	return 0;
}





int
engine_base::calc_adjoint_gradient_dirac_all()
{
	derivatives.resize(n_control_vars);
	std::vector<value_t> deriv_old(n_control_vars, 0);

	std::vector<value_t> lambda_n(mesh->n_blocks * n_vars, 0);
	std::vector<value_t> lambda_temp(mesh->n_blocks * n_vars, 0);

	std::vector<value_t> zeros_1(n_interfaces - wells.size(), 0);
	std::vector<value_t> Temp_1(n_interfaces - wells.size(), 0);

	std::vector<value_t> zeros_2_3(n_control_vars, 0);
	std::vector<value_t> Temp_2(n_control_vars, 0);
	std::vector<value_t> Temp_3(n_control_vars, 0);

	std::vector<value_t> zeros_4(mesh->n_blocks * n_vars, 0);
	std::vector<value_t> Temp_4(mesh->n_blocks * n_vars, 0);

	//std::vector<value_t> gradient(n_interfaces, 0);
	std::vector<value_t> gradient_u;
	std::vector<value_t> Zeros_1(n_interfaces, 0);
	std::vector<value_t> temp_1(n_interfaces, 0);

	std::vector<value_t> Zeros_2_3(n_control_vars, 0);
	std::vector<value_t> temp_2(n_control_vars, 0);
	std::vector<value_t> temp_3(n_control_vars, 0);

	std::vector<value_t>* X_state;

    // idx_sim_ts is the index of the simulation timestep. It corresponds to time_data
    // idx_obs_ts is the index of the observation timestep. It corresponds to time_data_report
    // the conflict between idx_sim_ts and idx_obs_ts is solved by Dirac function
    // See eq.(17), Tian et al. 2015  https://doi.org/10.1016/j.petrol.2021.109911
	index_t idx_sim_ts = 0, idx_obs_ts = 0, idx_well = 0;
	index_t TotStep = dt_t.size();

	index_t size_obs = 0;
	std::vector<value_t> miu = dt_t;
	if (objfun_prod_phase_rate)
	{
		size_obs = Q_all[0][0].size();
	}
	if (objfun_inj_phase_rate)
	{
		size_obs = Q_inj_all[0][0].size();
	}
	if (objfun_BHP)
	{
		size_obs = BHP_all[0].size();
	}
	if (objfun_well_tempr)
	{
		size_obs = well_tempr_all[0].size();
	}
	if (objfun_temperature)
	{
		size_obs = temperature_all.size();
	}
	if (objfun_customized_op)
	{
		size_obs = customized_op_all.size();
	}


    vec_3d q;
    std::vector<std::vector<value_t>> q_p;
    if (objfun_prod_phase_rate)
    {

        for (std::string well_ : prod_well_name)
        {
            std::vector<value_t> q_temp(TotStep, 0);
            std::vector<value_t> rate;
            q_p.clear();

            for (std::string opt_phase : prod_phase_name)
            {
                rate = time_data.at(well_ + " : " + opt_phase + " rate (m3/day)");

                //std::transform(rate.begin(), rate.end(), rate.begin(), std::bind1st(std::multiplies<double>(), -1));
                std::transform(rate.begin(), rate.end(), rate.begin(), [](double val) { return -1 * val; });  // adding minus sign here to make sure the production rate is positive
                q_p.push_back(rate);
            }

            q.push_back(q_p);
        }

    }

	vec_3d q_inj;
	std::vector<std::vector<value_t>> q_inj_p;
	if (objfun_inj_phase_rate)
	{

		for (std::string well_ : inj_well_name)
		{
			std::vector<value_t> rate_inj;
			q_inj_p.clear();

            for (std::string opt_phase : inj_phase_name)
            {
                rate_inj = time_data.at(well_ + " : " + opt_phase + " rate (m3/day)");

                //std::transform(rate.begin(), rate.end(), rate.begin(), std::bind1st(std::multiplies<double>(), -1));
                q_inj_p.push_back(rate_inj);
            }

			q_inj.push_back(q_inj_p);
		}
	}

	std::vector<std::vector<value_t>> bhp;
	if (objfun_BHP)
	{
		for (std::string well_ : BHP_well_name)
		{
			bhp.push_back(time_data.at(well_ + " : BHP (bar)"));
		}
	}

	std::vector<std::vector<value_t>> well_tempr;
	if (objfun_well_tempr)
	{
		for (std::string well_ : well_tempr_name)
		{
			well_tempr.push_back(time_data.at(well_ + " : temperature (K)"));
		}
	}

	std::vector<std::vector<value_t>> temperature;
	if (objfun_temperature)
	{
		/*index_t size_temp_vec = time_data_customized.size();
		for (index_t t = 0; t < size_temp_vec; t++)
		{
			temperature.push_back(time_data_customized[t]);
		}*/
        temperature = time_data_customized;
	}

	std::vector<std::vector<value_t>> customized_op;
	if (objfun_customized_op)
	{
		/*index_t size_vec = time_data_customized.size();
		for (index_t t = 0; t < size_vec; t++)
		{
			customized_op.push_back(time_data_customized[t]);
		}*/
        customized_op = time_data_customized;
	}


    // dT_du is a transformation matrix that converts dJ_dT to dJ_du, that is dJ_du = dJ_dT * dT_du
    // therefore the elements in dT_du is either 1 or 0
    // this means that in CSR matrix of dT_du, all values is 1. What really matters is the column index `cols`
	value_t* values = dT_du->get_values();
	index_t* rows = dT_du->get_rows_ptr();
	index_t* cols = dT_du->get_cols_ind();

	index_t n_row = n_interfaces - wells.size();
	for (index_t i = 0; i < n_row; i++)
	{
		rows[i] = i;
		cols[i] = col_dT_du[i];  // col_dT_du is initialized from `opt_module_settings.py`
		values[i] = 1;
	}
	rows[n_row] = n_row;


	derivatives = Temp_2;
	//gradient = Temp_2;




	std::vector<value_t> gradient(n_interfaces - well_head_idx_collection.size(), 0);
	// the last timestep--------------------------------------------------------------------------------------------------------
	// See eq.(19), Tian et al. 2015  https://doi.org/10.1016/j.petrol.2021.109911
	// note that the the notation x and X in this code means the omega in the above paper

	idx_sim_ts = TotStep - 1;
	X = X_t[idx_sim_ts];
	dt = dt_t[idx_sim_ts];

	// recover the well definition from forward simulation, e.g. control and constraints
	idx_well = 0;
	for (ms_well* w : wells)
	{
		w = &(well_control_arr[idx_sim_ts][idx_well]);
		idx_well++;
	}


	// evaluate all operators and their derivatives
	if (is_mp)
	{
		Xop_mp = Xop_t[idx_sim_ts];
		X_state = &Xop_mp;
	}
	else
	{
		X_state = &X;
	}
	for (index_t r = 0; r < acc_flux_op_set_list.size(); r++)
	{
		index_t result = acc_flux_op_set_list[r]->evaluate_with_derivatives(*X_state, block_idxs[r], op_vals_arr, op_ders_arr);
		if (result < 0)
			return 0;
	}


	idx_obs_ts = size_obs - 1;


	// assemble dg_dx (jacobian), RHS is a dummy vector here. See eq.(19), Tian et al. 2015  https://doi.org/10.1016/j.petrol.2021.109911
	assemble_jacobian_array(dt, X, Jacobian, RHS);

	// prepare dj_dx
	prepare_dj_dx(q, q_inj, bhp, well_tempr, temperature, customized_op, idx_sim_ts, idx_obs_ts);

	// assemble dg_dx_n, dg_dT, dj_dx, while dg_dx_n is a dummy matrix here. Because there is no dg_dx_n in eq.(19), Tian et al. 2015  https://doi.org/10.1016/j.petrol.2021.109911
	adjoint_gradient_assembly(dt, X, dg_dx_n, RHS);




	linear_solver_ad->init(dg_dx_T, params->max_i_linear, params->tolerance_linear);

	// solve lambda and adjoint gradient at the last time step
    timer->node["linear solver for adjoint method - setup"].start();
	linear_solver_ad->setup(dg_dx_T);
    timer->node["linear solver for adjoint method - setup"].stop();

    timer->node["linear solver for adjoint method - solve"].start();
	linear_solver_ad->solve(&Temp_dj_dx[0], &lambda_temp[0]);
    timer->node["linear solver for adjoint method - solve"].stop();



	lambda_n = lambda_temp;

	(static_cast<csr_matrix<1>*>(dg_dT_general))->matrix_vector_product_t(&lambda_temp[0], &temp_1[0]);
	index_t well_numeration = 0;
	for (index_t wh : well_head_tran_idx_collection)
	{
		temp_1.erase(temp_1.begin() + (wh - well_numeration));
		well_numeration++;
	}

	std::transform(gradient.begin(), gradient.end(), temp_1.begin(), gradient.begin(), std::plus<double>());
	std::transform(gradient.begin(), gradient.end(), Temp_dj_du.begin(), gradient.begin(), std::plus<double>());
	temp_1.resize(n_interfaces);




	// the rest of timesteps-----------------------------------------------------------------------------------------------------------
	// See eq.(18), Tian et al. 2015  https://doi.org/10.1016/j.petrol.2021.109911
	// note that the the notation x and X in this code means the omega in the above paper
	for (idx_sim_ts = TotStep - 2; idx_sim_ts >= 0; idx_sim_ts--)
	{
		X = X_t[idx_sim_ts];
		dt = dt_t[idx_sim_ts];

		// recover the well definition from forward simulation, e.g. control and constraints
		idx_well = 0;
		for (ms_well* w : wells)
		{
			w = &(well_control_arr[idx_sim_ts][idx_well]);
			idx_well++;
		}


		// evaluate all operators and their derivatives
		if (is_mp)
		{
			Xop_mp = Xop_t[idx_sim_ts];
			X_state = &Xop_mp;
		}
		else
		{
			X_state = &X;
		}
		for (index_t r = 0; r < acc_flux_op_set_list.size(); r++)
		{
			index_t result = acc_flux_op_set_list[r]->evaluate_with_derivatives(*X_state, block_idxs[r], op_vals_arr, op_ders_arr);
			if (result < 0)
				return 0;
		}


		if (dirac_vec[idx_sim_ts] == 1)
		{
			idx_obs_ts -= 1;
		}


		// assemble dg_dx (jacobian)
		assemble_jacobian_array(dt, X, Jacobian, RHS);

		// prepare dj_dx
		prepare_dj_dx(q, q_inj, bhp, well_tempr, temperature, customized_op, idx_sim_ts, idx_obs_ts);

		// assemble dg_dx_n, dg_dT, dj_dx
		adjoint_gradient_assembly(dt, X, dg_dx_n, RHS);



		Temp_4 = zeros_4;
		(static_cast<csr_matrix<1>*>(dg_dx_n))->matrix_vector_product_t(&lambda_n[0], &Temp_4[0]);

        // Temp_dj_dx is already moved to right hand side by adding a minus sign in the function "prepare_dj_dx". That is why here it is "std::minus"
		std::transform(Temp_dj_dx.begin(), Temp_dj_dx.end(), Temp_4.begin(), Temp_dj_dx.begin(), std::minus<double>());

        timer->node["linear solver for adjoint method - setup"].start();
		linear_solver_ad->setup(dg_dx_T);
        timer->node["linear solver for adjoint method - setup"].stop();

        timer->node["linear solver for adjoint method - solve"].start();
		linear_solver_ad->solve(&Temp_dj_dx[0], &lambda_temp[0]);
        timer->node["linear solver for adjoint method - solve"].stop();



		lambda_n = lambda_temp;



		temp_1 = Zeros_1;
		(static_cast<csr_matrix<1>*>(dg_dT_general))->matrix_vector_product_t(&lambda_temp[0], &temp_1[0]);
		well_numeration = 0;
		for (index_t wh : well_head_tran_idx_collection)
		{
			temp_1.erase(temp_1.begin() + (wh - well_numeration));
			well_numeration++;
		}

		std::transform(gradient.begin(), gradient.end(), temp_1.begin(), gradient.begin(), std::plus<double>());
		std::transform(gradient.begin(), gradient.end(), Temp_dj_du.begin(), gradient.begin(), std::plus<double>());

		temp_1.resize(n_interfaces);

	}

	gradient_u = Zeros_2_3;
	(static_cast<csr_matrix<1>*>(dT_du))->matrix_vector_product_t(&gradient[0], &gradient_u[0]);

	std::transform(derivatives.begin(), derivatives.end(), gradient_u.begin(), derivatives.begin(), std::plus<double>());



	deriv_old = derivatives;

	return 0;

};




int
engine_base::prepare_dj_dx(vec_3d q, vec_3d q_inj,
	std::vector<std::vector<value_t>> bhp, std::vector<std::vector<value_t>> well_tempr, 
	std::vector<std::vector<value_t>> temperature, std::vector<std::vector<value_t>> customized_op, 
	index_t idx_sim_ts, index_t idx_obs_ts)
{

	std::vector<std::vector<value_t>>  q_Q;
	std::vector<value_t> rate_diff;
	q_Q.clear();
	index_t ww = 0, idx_well = 0;


    if (objfun_prod_phase_rate)
    {
        ww = 0;
        for (std::string well_ : prod_well_name)
        {
            rate_diff.clear();

            index_t p = 0;      // the index of the phase in observation data set
            for (std::string opt_phase : prod_phase_name)
            {
                //rate_diff.push_back(2 * (q[ww][p][idx_sim_ts] - Q_all[ww][p][idx_obs_ts]) * dirac_vec[idx_sim_ts] * scale_function_value * cov_mat_inv[idx_obs_ts]);
                rate_diff.push_back(2 * (q[ww][p][idx_sim_ts] - Q_all[ww][p][idx_obs_ts])
                    * dirac_vec[idx_sim_ts] * scale_function_value
                    * cov_mat_inv_prod_all[ww][p][idx_obs_ts] * cov_mat_inv_prod_all[ww][p][idx_obs_ts]
                    * prod_weights_all[ww][p][idx_obs_ts]);

                p++;
            }
            q_Q.push_back(rate_diff);
            ww++;
        }
    }


	std::vector<std::vector<value_t>>  q_inj_Q;
	std::vector<value_t> rate_inj_diff;
	q_inj_Q.clear();

	if (objfun_inj_phase_rate)
	{
        ww = 0;
		for (std::string well_ : inj_well_name)
		{
			rate_inj_diff.clear();

            index_t p = 0;      // the index of the phase in observation data set
            for (std::string opt_phase : inj_phase_name)
            {
                //rate_inj_diff.push_back(2 * (q_inj[ww][p][idx_sim_ts] - Q_inj_all[ww][p][idx_obs_ts]) * dirac_vec[idx_sim_ts] * scale_function_value * cov_mat_inv[idx_obs_ts]);
                rate_inj_diff.push_back(2 * (q_inj[ww][p][idx_sim_ts] - Q_inj_all[ww][p][idx_obs_ts])
                    * dirac_vec[idx_sim_ts] * scale_function_value
                    * cov_mat_inv_inj_all[ww][p][idx_obs_ts] * cov_mat_inv_inj_all[ww][p][idx_obs_ts]
                    * inj_weights_all[ww][p][idx_obs_ts]);

                p++;
            }

			q_inj_Q.push_back(rate_inj_diff);
			ww++;
		}
	}



	std::vector<value_t> bhp_BHP;

	if (objfun_BHP)
	{
        ww = 0;
		for (std::string well_ : BHP_well_name)
		{
			bhp_BHP.push_back(2 * (bhp[ww][idx_sim_ts] - BHP_all[ww][idx_obs_ts])
				* dirac_vec[idx_sim_ts] * scale_function_value
				* cov_mat_inv_BHP_all[ww][idx_obs_ts] * cov_mat_inv_BHP_all[ww][idx_obs_ts]
				* BHP_weights_all[ww][idx_obs_ts]);
			ww++;
		}
	}



	std::vector<value_t> wt_WT;

	if (objfun_well_tempr)
	{
        ww = 0;
		for (std::string well_ : well_tempr_name)
		{
			wt_WT.push_back(2 * (well_tempr[ww][idx_sim_ts] - well_tempr_all[ww][idx_obs_ts])
				* dirac_vec[idx_sim_ts] * scale_function_value
				* cov_mat_inv_well_tempr_all[ww][idx_obs_ts] * cov_mat_inv_well_tempr_all[ww][idx_obs_ts]
				* well_tempr_weights_all[ww][idx_obs_ts]);
			ww++;
		}
	}



	std::vector<value_t> tempr_TEMPR;
	if (objfun_temperature)
	{
		for (index_t b = 0; b < mesh->n_res_blocks; b++)
		{
			tempr_TEMPR.push_back(2 * (temperature[idx_sim_ts][b] - temperature_all[idx_obs_ts][b])
				* dirac_vec[idx_sim_ts] * scale_function_value
				* cov_mat_inv_temperature_all[idx_obs_ts][b] * cov_mat_inv_temperature_all[idx_obs_ts][b]
				* temperature_weights_all[idx_obs_ts][b]);
		}
	}


	std::vector<value_t> op_OP;
	if (objfun_customized_op)
	{
		double hinge_coeff = 1;

		for (index_t b = 0; b < mesh->n_res_blocks; b++)
		{
			if ((binary_all.size() > 0) && (dirac_vec[idx_sim_ts] > 0))  // i.e. dirac_vec[idx_sim_ts] == 1 when we are at reporting time (i.e. observation time)
			{
				if (binary_all[idx_obs_ts][b] > 0)  // binary_all[idx_obs_ts][b] == 1
				{
					if (customized_op[idx_sim_ts][b] > threshold)
					{
						hinge_coeff = 0;
					}
					else
					{
						hinge_coeff = 1;
					}
				}
				else
				{
					if (customized_op[idx_sim_ts][b] < threshold)
					{
						hinge_coeff = 0;
					}
					else
					{
						hinge_coeff = 1;
					}
				}
			}
			else
			{
				hinge_coeff = 1;  // we don't do anything to op_OP in this case
			}


			op_OP.push_back(2 * (customized_op[idx_sim_ts][b] - customized_op_all[idx_obs_ts][b])
				* dirac_vec[idx_sim_ts] * scale_function_value
				* cov_mat_inv_customized_op_all[idx_obs_ts][b] * cov_mat_inv_customized_op_all[idx_obs_ts][b]
				* customized_op_weights_all[idx_obs_ts][b]
				* hinge_coeff);
		}
	}




	std::vector<value_t> sub1(n_vars * mesh->n_blocks, 0);
	std::vector<value_t> sub2(n_interfaces, 0);

	Temp_dj_dx.clear();
	Temp_dj_du.clear();

	Temp_dj_dx = sub1;
	Temp_dj_du = sub2;


    
    if (objfun_prod_phase_rate)
    {
        index_t upstream_idx;
        ww = 0;
        ms_well* w;
        for (std::string well_ : prod_well_name)
        {
            //for (ms_well* well : wells)
            //{
            //    if (well->name == well_)
            //    {
            //        w = well;
            //    }
            //}

			// recover the well definition from forward simulation, e.g. control and constraints
			idx_well = 0;
			for (ms_well* well : wells)
			{
				if (well->name == well_)
				{
					w = &(well_control_arr[idx_sim_ts][idx_well]);
				}
				idx_well++;
			}

            // find upstream state
            value_t p_diff = X[w->well_head_idx * w->n_block_size + w->P_VAR] - X[w->well_body_idx * w->n_block_size + w->P_VAR];
            if (p_diff > 0)
                upstream_idx = w->well_head_idx; // injector
            else
                upstream_idx = w->well_body_idx; // producer

            //index_t nc = n_vars;
            //index_t n_ops = 2 * nc;

            std::vector<value_t> state;
            std::vector<index_t> block_idx = { 0 };
            std::vector<value_t> rates;
            std::vector<value_t> rates_derivs;

            //rates.resize(w->n_phases);
            //rates_derivs.resize(w->n_phases * n_vars);


			index_t n_ops_well = w->control.get_well_n_ops();
			index_t n_vars_well = w->control.get_well_n_vars();

			rates.resize(n_ops_well);
			rates_derivs.resize(n_ops_well * n_vars_well);

            state.assign(X.begin() + upstream_idx * w->n_block_size + w->P_VAR, X.begin() + upstream_idx * w->n_block_size + w->P_VAR + n_vars_well);
            w->rate_etor_ad->evaluate_with_derivatives(state, block_idx, rates, rates_derivs);




            //uint8_t c = component_index[0];
            double ders_term, vals_term;
            for (uint8_t v = 0; v < n_vars_well; v++)
            {
                ders_term = 0.0;
                vals_term = 0.0;

                index_t p_idx = observation_rate_type * w->n_phases;  // by default it is volumetric rate
                for (std::string phase : w->phase_names)
                {
                    index_t p = 0;  // the index of the phase in observation data set
                    for (std::string opt_phase : prod_phase_name)
                    {
                        if (opt_phase == phase)
                        {
                            // adding minus sign on "q_Q" to move Temp_dj_dx to the right hand side of eq.(18) and eq.(19), Tian et al. 2015  https://doi.org/10.1016/j.petrol.2021.109911
                            ders_term += rates_derivs[p_idx * n_vars_well + v] * p_diff * w->segment_transmissibility * (-q_Q[ww][p]);
                            vals_term += rates[p_idx] * w->segment_transmissibility * (-q_Q[ww][p]);
                        }
                        p++;
                    }
                    p_idx++;
                }



                // corresponding to ms_well::check_constraints
				if (w->control.get_well_control_type() == well_control_iface::BHP)  // BHP control
                {
                    Temp_dj_dx[upstream_idx * n_vars_well + v] += -ders_term;
                    if (v == 0)  // derivatives w.r.t. pressure
                    {
                        Temp_dj_dx[w->well_body_idx * n_vars_well + v] += vals_term;
                    }
                }
                else  // rate control
                {
                    //;  // all zero

                    Temp_dj_dx[upstream_idx * n_vars_well + v] += -ders_term;
                    if (v == 0)  // derivatives w.r.t. pressure
                    {
                        Temp_dj_dx[w->well_body_idx * n_vars_well + v] += vals_term;
                        Temp_dj_dx[w->well_head_idx * n_vars_well + v] += -vals_term; // add extra term on well head
                    }
                }


            }
            ww++;
        }
    }





	
	if (objfun_inj_phase_rate)
	{
		index_t upstream_idx;
        ww = 0;
        ms_well* w;
        for (std::string well_ : inj_well_name)
        {
            //for (ms_well* well : wells)
            //{
            //    if (well->name == well_)
            //    {
            //        w = well;
            //    }
            //}

			// recover the well definition from forward simulation, e.g. control and constraints
			idx_well = 0;
			for (ms_well* well : wells)
			{
				if (well->name == well_)
				{
					w = &(well_control_arr[idx_sim_ts][idx_well]);
				}
				idx_well++;
			}

			// find upstream state
			value_t p_diff = X[w->well_head_idx * w->n_block_size + w->P_VAR] - X[w->well_body_idx * w->n_block_size + w->P_VAR];
			if (p_diff > 0)
				upstream_idx = w->well_head_idx; // injector
			else
				upstream_idx = w->well_body_idx; // producer

			//index_t nc = n_vars;
			//index_t n_ops = 2 * nc;

			std::vector<value_t> state;
			std::vector<index_t> block_idx = { 0 };
			std::vector<value_t> rates;
			std::vector<value_t> rates_derivs;

			//rates.resize(w->n_phases);
			//rates_derivs.resize(w->n_phases * n_vars);

			index_t n_ops_well = w->control.get_well_n_ops();
			index_t n_vars_well = w->control.get_well_n_vars();

			rates.resize(n_ops_well);
			rates_derivs.resize(n_ops_well* n_vars_well);

			state.assign(X.begin() + upstream_idx * w->n_block_size + w->P_VAR, X.begin() + upstream_idx * w->n_block_size + w->P_VAR + n_vars_well);
			w->rate_etor_ad->evaluate_with_derivatives(state, block_idx, rates, rates_derivs);




			//uint8_t c = component_index[0];
			double ders_term, vals_term;
			for (uint8_t v = 0; v < n_vars_well; v++)
			{
				ders_term = 0.0;
				vals_term = 0.0;

				index_t p_idx = observation_rate_type * w->n_phases;  // by default it is volumetric rate
				for (std::string phase : w->phase_names)
				{
                    index_t p = 0;  // the index of the phase in observation data set
					for (std::string opt_phase : inj_phase_name)
					{
						if (opt_phase == phase)
						{
                            // adding minus sign on "q_inj_Q" to move Temp_dj_dx to the right hand side of eq.(18) and eq.(19), Tian et al. 2015  https://doi.org/10.1016/j.petrol.2021.109911
							ders_term += rates_derivs[p_idx * n_vars_well + v] * p_diff * w->segment_transmissibility * (-q_inj_Q[ww][p]);
							vals_term += rates[p_idx] * w->segment_transmissibility * (-q_inj_Q[ww][p]);
						}
                        p++;
					}
                    p_idx++;
				}

				if (w->control.get_well_control_type() == well_control_iface::BHP)  // BHP control
				{
					Temp_dj_dx[upstream_idx * n_vars_well + v] += ders_term;
					if (v == 0)  // derivatives w.r.t. pressure
					{
						Temp_dj_dx[w->well_body_idx * n_vars_well + v] += -vals_term;
					}
				}
				else  // rate control
				{
					//;  // all zero

					Temp_dj_dx[upstream_idx * n_vars_well + v] += ders_term;
					if (v == 0)  // derivatives w.r.t. pressure
					{
						Temp_dj_dx[w->well_body_idx * n_vars_well + v] += -vals_term;
						Temp_dj_dx[w->well_head_idx * n_vars_well + v] += vals_term;
					}
				}
			}
            ww++;
        }
	}


	
	if (objfun_BHP)
	{
        index_t upstream_idx;
        ww = 0;
        ms_well* w;
		for (std::string well_ : BHP_well_name)
		{
            //for (ms_well* well : wells)
            //{
            //    if (well->name == well_)
            //    {
            //        w = well;
            //    }
            //}

			// recover the well definition from forward simulation, e.g. control and constraints
			idx_well = 0;
			for (ms_well* well : wells)
			{
				if (well->name == well_)
				{
					w = &(well_control_arr[idx_sim_ts][idx_well]);
				}
				idx_well++;
			}

            // adding minus sign on "bhp_BHP" to move Temp_dj_dx to the right hand side of eq.(18) and eq.(19), Tian et al. 2015  https://doi.org/10.1016/j.petrol.2021.109911
			// corresponding to ms_well::check_constraints
			if (w->control.get_well_control_type() == well_control_iface::BHP)  // BHP control
			{
				index_t v = 0;  // derivatives w.r.t. pressure
				Temp_dj_dx[w->well_head_idx * n_vars + v] += 0 * (-bhp_BHP[ww]);
				//Temp_dj_dx[w->well_head_idx * n_vars + v] += 1 * (-bhp_BHP[ww]);
			}
			else  // rate control
			{
				index_t v = 0;  // derivatives w.r.t. pressure
				Temp_dj_dx[w->well_head_idx * n_vars + v] += 1 * (-bhp_BHP[ww]);

			}
			ww++;
		}
	}



	
	if (objfun_well_tempr)
	{
        index_t upstream_idx;
        ww = 0;
        ms_well* w;
		for (std::string well_ : well_tempr_name)
		{
            for (ms_well* well : wells)
            {
                if (well->name == well_)
                {
                    w = well;
                }
            }

			// find upstream state
			value_t p_diff = X[w->well_head_idx * w->n_block_size + w->P_VAR] - X[w->well_body_idx * w->n_block_size + w->P_VAR];
			if (p_diff > 0)
				upstream_idx = w->well_head_idx; // injector
			else
				upstream_idx = w->well_body_idx; // producer




			if (w->thermal)  // for thermal super_engine_cpu and super_engine_mp_cpu
			{
				index_t v = n_vars - 1;  // derivatives w.r.t. temperature
				Temp_dj_dx[upstream_idx * n_vars + v] += 1 * (-wt_WT[ww]);
			}
			else 
			{
				//index_t nc = n_vars;
				//index_t n_ops = 2 * nc;

				std::vector<value_t> state;
				std::vector<index_t> block_idx = { 0 };
				std::vector<value_t> rates;
				std::vector<value_t> rates_derivs;

				rates.resize(w->n_phases);
				rates_derivs.resize(w->n_phases * n_vars);

				state.assign(X.begin() + upstream_idx * w->n_block_size + w->P_VAR, X.begin() + upstream_idx * w->n_block_size + w->P_VAR + n_vars);
				w->rate_etor_ad->evaluate_with_derivatives(state, block_idx, rates, rates_derivs);


				double ders_term, vals_term;
				for (uint8_t v = 0; v < n_vars; v++)
				{
					ders_term = 0.0;
					vals_term = 0.0;

					index_t p_idx = 0;  // the index of the phase in DARTS model definition
					for (std::string phase : w->phase_names)
					{
						if (phase == "temperature")
						{
							// adding minus sign on "wt_WT" to move Temp_dj_dx to the right hand side of eq.(18) and eq.(19), Tian et al. 2015  https://doi.org/10.1016/j.petrol.2021.109911
							ders_term += rates_derivs[p_idx * n_vars + v] * (-wt_WT[ww]);
							//vals_term += rates[p] * (-wt_WT[ww]);
						}
						p_idx++;
					}

					Temp_dj_dx[upstream_idx * n_vars + v] += ders_term;
				}
			}

			ww++;
		}
	}




	if (objfun_temperature)
	{
		// evaluate the customized operators and their derivatives
		if (customize_operator)
		{
			index_t r = idx_customized_operator;
			index_t result = acc_flux_op_set_list[r]->evaluate_with_derivatives(X, customize_block_idxs[r], op_vals_arr_customized, op_ders_arr_customized);
			if (result < 0)
				return 0;
		}


		for (index_t b = 0; b < mesh->n_res_blocks; b++)
		{
			for (index_t v = 0; v < n_vars; v++)
			{
                // adding minus sign on "tempr_TEMPR" to move Temp_dj_dx to the right hand side of eq.(18) and eq.(19), Tian et al. 2015  https://doi.org/10.1016/j.petrol.2021.109911
				Temp_dj_dx[b * n_vars + v] += op_ders_arr_customized[b * n_vars + v] * (-tempr_TEMPR[b]);
			}
		}
	}



	if (objfun_customized_op)
	{
		// evaluate the customized operators and their derivatives
		if (customize_operator)
		{
			index_t r = idx_customized_operator;
			index_t result = acc_flux_op_set_list[r]->evaluate_with_derivatives(X, customize_block_idxs[r], op_vals_arr_customized, op_ders_arr_customized);
			if (result < 0)
				return 0;
		}


		for (index_t b = 0; b < mesh->n_res_blocks; b++)
		{
			for (index_t v = 0; v < n_vars; v++)
			{
                // adding minus sign on "op_OP" to move Temp_dj_dx to the right hand side of eq.(18) and eq.(19), Tian et al. 2015  https://doi.org/10.1016/j.petrol.2021.109911
				Temp_dj_dx[b * n_vars + v] += op_ders_arr_customized[b * n_vars + v] * (-op_OP[b]);
			}
		}
	}


	return 0;
};

int engine_base::print_timestep(value_t time, value_t deltat)
{
	double estimate;
	int hour, min, sec;
	char buffer[1024];
	char buffer2[1024];
	char line[] = "-------------------------------------------------------------------------------------------------------------\n";

	estimate = timer->get_timer();
	hour = estimate / 3600;
	estimate -= hour * 3600;
	min = estimate / 60;
	estimate -= min * 60;
	sec = estimate;

	sprintf(buffer, "T = %g, DT = %g, NI = %d, LI = %d, RES = %.1e (%.1e), CFL=%.3lf (ELAPSED %02d:%02d:%02d",
			time, deltat, n_newton_last_dt, n_linear_last_dt, newton_residual_last_dt, well_residual_last_dt, CFL_max, hour, min, sec);
	if ((dt * params->mult_ts > params->max_ts || full_step_timer.timer) && t < stop_time)
	{
		if (!full_step_timer.timer)
		{
			full_step_timer.start();
			t_full_step = t;
		}
		else
		{
			estimate = full_step_timer.get_timer() / (t - t_full_step) * (stop_time - t);
			hour = estimate / 3600;
			estimate -= hour * 3600;
			min = estimate / 60;
			estimate -= min * 60;
			sec = estimate;
			sprintf(buffer2, "%s, REMAINING %02d:%02d:%02d", buffer, hour, min, sec);
			sprintf(buffer, "%s", buffer2);
		}
	}
	sprintf(buffer2, "%s %s )\n%s", line, buffer, line);
	std::cout << buffer2 << std::flush;

	return 0;
}

/*!
 @details

	This procedure prints all statistics of simultion run

 @note The statistics includes the following items:
  - total timesteps n_timesteps_total,
  - wasted timesteps n_timesteps_wasted,
  - total number of Neton iterations n_newton_total,
  - number of wated Newton iterations n_newton_wasted,
  - number of linear iterations n_linear_total,
  - number of waterd linear iterations n_linear_wasted
  - extended OBL statistics
  */

int engine_base::print_stat()
{
	index_t r_code = 0;
	char buffer[10240];

	const char n_ops = get_n_ops();

	r_code += sprintf(buffer, "\n");
	r_code += sprintf(buffer + r_code, "Total steps %d (%d) newton %d (%d) linear %d (%d)\n", stat.n_timesteps_total,
					  stat.n_timesteps_wasted, stat.n_newton_total, stat.n_newton_wasted, stat.n_linear_total, stat.n_linear_wasted);

	r_code += sprintf(buffer + r_code, "---OBL Statistics---\n");
	r_code += sprintf(buffer + r_code, "Number of operators: %d\n", n_ops);

	r_code += sprintf(buffer + r_code, "Number of points: %d\n", acc_flux_op_set_list[0]->get_axis_n_points(0));
	r_code += sprintf(buffer + r_code, "Number of interpolations: %lu \n", acc_flux_op_set_list[0]->get_n_interpolations());
	r_code += sprintf(buffer + r_code, "Number of points generated: %lu (%.3f%%)\n", acc_flux_op_set_list[0]->get_n_points_used(), (acc_flux_op_set_list[0]->get_n_points_used() * 100.0 / acc_flux_op_set_list[0]->get_n_points_total()));
	//r_code += sprintf(buffer + r_code, "Number of hypercubes used: %lu (%.3f%%)\n", acc_flux_op_set_list[0]->get_n_hypercubes_used(), (acc_flux_op_set_list[0]->get_n_hypercubes_used() * 100.0 / acc_flux_op_set_list[0]->get_n_hypercubes_total()));
	/*
	r_code += sprintf (buffer + r_code, "OMIPS: %.4lf \n", acc_flux_op_set->get_n_interpolations() / interpolation_timer / 1000000);
	r_code += sprintf (buffer + r_code, "Parse me: %d %d %d %.4lf %.4lf %.4lf %.4lf  %d %d %d %d %d %d %.4lf\n", n_timesteps_total, n_newton_total, n_linear_total, total_timer, assemble_timer, \
	linear_setup_timer, linear_solve_timer, (2 * nc), acc_flux_op_set->get_resolution(), acc_flux_op_set->get_n_interpolations(),  \
	acc_flux_op_set->get_n_points_used(), acc_flux_op_set->get_n_hypercubes_used(), acc_flux_op_set->get_n_hypercubes_total(), \
	acc_flux_op_set->get_n_interpolations() / interpolation_timer / 1000000);
	*/
	std::cout << buffer << std::flush;

	std::string timer_stat;
	timer->print("", timer_stat);
	std::cout << timer_stat << std::flush;

	return 0;
}

/*!
 @details

	This method adds averaged and accumulated phase rates over the period since the last report call till current moment
	(possibly spannin over several timesteps) to time_data_report array.

  */

int engine_base::report()
{
	double t_prev_report = 0;
	int average_start_ts = 0;

	// if that`s true then there is nothing to average
	if (time_data.empty())
		return -1;

	// get the time of previous report if any
	if (!time_data_report.empty())
	{
		t_prev_report = time_data_report["time"].back();
	}

	// find the ts correspondent to the first date after the previous report (go backwards)
	for (int ts = time_data["time"].size() - 1; ts >= 0; ts--)
	{
		if (time_data["time"][ts] <= t_prev_report)
		{
			average_start_ts = ts + 1;
			break;
		}
	}

	// still nothing to average
	if (average_start_ts == time_data["time"].size())
		return -1;

	//adjoint method
	if (opt_history_matching)
	{
		X_t_report.push_back(X);
		dt_t_report.push_back(t - t_prev_report);
		t_t_report.push_back(t);
		//Jacobian->write_matrix_to_file("jac_forward_simulation.csr");
		//write_vector_to_file("rhs_forward_simulation.rhs", RHS);
	}

	// evaluate the customized operators and their derivatives -- adjoint method
	if (customize_operator)
	{
		index_t r = idx_customized_operator;
		index_t result = acc_flux_op_set_list[r]->evaluate_with_derivatives(X, customize_block_idxs[r], op_vals_arr_customized, op_ders_arr_customized);
		if (result < 0)
			return 0;

		time_data_report_customized.push_back(op_vals_arr_customized);
	}

	for (auto c : time_data)
	{
		std::string name;
		std::vector<value_t> data;
		value_t total_volume = 0;
		std::tie(name, data) = c;
		std::string search_str = "rate (m3/day)";

		// look only for phase volumetric rates
		if (name.find(search_str) != std::string::npos)
		{
			std::string acc_vol = name.substr(0, name.size() - search_str.size()) + " acc volume (m3)";
			std::string vol = name.substr(0, name.size() - search_str.size()) + " volume (m3)";

			// sum over the computational timesteps
			for (int ts = average_start_ts; ts < time_data["time"].size(); ts++)
			{
				if (ts == 0)
					total_volume += data[ts] * time_data["time"][ts];
				else
					total_volume += data[ts] * (time_data["time"][ts] - time_data["time"][ts - 1]);
			}
			// only positive values
			total_volume = fabs(total_volume);
			time_data_report[name].push_back(total_volume / (t - t_prev_report));
			time_data_report[vol].push_back(total_volume);
			if (time_data_report[acc_vol].size() > 0)
			{
				time_data_report[acc_vol].push_back(total_volume + time_data_report[acc_vol].back());
			}
			else
			{
				time_data_report[acc_vol].push_back(total_volume);
			}
		}
		else
		{
			for (int ts = time_data["time"].size()-1; ts < time_data["time"].size(); ts++)
			{
				time_data_report[name].push_back(data[ts]);
			}
		}
	}

	return 0;
}

void engine_base::average_operator(std::vector<value_t> &av_op)
{
	for (int c = 0; c < n_vars; c++)
	{
		av_op[c] = 0;
	}
	for (int i = 0; i < mesh->n_res_blocks; i++)
	{
		for (int c = 0; c < n_vars; c++)
		{
			av_op[c] += op_vals_arr[i * n_ops + c];
		}
	}
	for (int c = 0; c < n_vars; c++)
	{
		av_op[c] /= mesh->n_res_blocks;
	}
}

double
engine_base::calc_newton_residual_L1()
{
	double residual = 0;
	std::vector<value_t> res(n_vars, 0);
	std::vector<value_t> norm(n_vars, 0);

	for (int i = 0; i < mesh->n_res_blocks; i++)
	{
		for (int c = 0; c < n_vars; c++)
		{
			res[c] += RHS[i * n_vars + c];
			norm[c] += PV[i] * op_vals_arr[i * n_ops + c];
		}
	}
	for (int c = 0; c < n_vars; c++)
	{
		residual = std::max(residual, fabs(res[c] / norm[c]));
	}

	return residual;
}

double
engine_base::calc_newton_residual_L2()
{
	double residual = 0;
	std::vector<value_t> res(n_vars, 0);
	std::vector<value_t> norm(n_vars, 0);

	for (int i = 0; i < mesh->n_res_blocks; i++)
	{
		for (int c = 0; c < n_vars; c++)
		{
			res[c] += RHS[i * n_vars + c] * RHS[i * n_vars + c];
			norm[c] += (PV[i] * op_vals_arr[i * n_ops + c]) * (PV[i] * op_vals_arr[i * n_ops + c]);
		}
	}
	for (int c = 0; c < n_vars; c++)
	{
		residual = std::max(residual, sqrt(res[c] / norm[c]));
	}

	return residual;
}

double
engine_base::calc_newton_residual_Linf()
{
	double residual = 0, norm;
	// Linf norm

	for (int i = 0; i < mesh->n_res_blocks; i++)
	{
		for (int c = 0; c < n_vars; c++)
		{
			norm = PV[i] * op_vals_arr[i * n_ops + c];
			if (norm > 1e-3)
				residual = std::max(residual, fabs(RHS[i * n_vars + c] / norm));
		}
	}

	return residual;
}

double
engine_base::calc_well_residual_L1()
{
	double residual = 0;
	std::vector<value_t> res(n_vars, 0);
	std::vector<value_t> norm(n_vars, 0);

	std::vector<value_t> av_op(n_vars, 0);
	average_operator(av_op);

	for (ms_well *w : wells)
	{
		// first sum up RHS for well segments which have perforations
		int nperf = w->perforations.size();
		for (int ip = 0; ip < nperf; ip++)
		{
			for (int v = 0; v < n_vars; v++)
			{
				index_t i_w, i_r;
				value_t wi, wid;
				std::tie(i_w, i_r, wi, wid) = w->perforations[ip];

				res[v] += fabs(RHS[(w->well_body_idx + i_w) * n_vars + v]);
				norm[v] += (PV[w->well_body_idx + i_w] * av_op[v]);
			}
		}
		// and then add RHS for well control equations
		for (int v = 0; v < n_vars; v++)
		{
			// well constraints should not be normalized, so pre-multiply it by norm
			res[v] += fabs(RHS[w->well_head_idx * n_vars + v]) * PV[w->well_body_idx] * av_op[v];
		}
	}

	for (int v = 0; v < n_vars; v++)
	{
		residual = std::max(residual, fabs(res[v] / norm[v]));
	}

	return residual;
}

double
engine_base::calc_well_residual_L2()
{
	double residual = 0;
	std::vector<value_t> res(n_vars, 0);
	std::vector<value_t> norm(n_vars, 0);

	std::vector<value_t> av_op(n_vars, 0);
	average_operator(av_op);

	for (ms_well *w : wells)
	{
		// first sum up RHS for well segments which have perforations
		int nperf = w->perforations.size();
		for (int ip = 0; ip < nperf; ip++)
		{
			for (int v = 0; v < n_vars; v++)
			{
				index_t i_w, i_r;
				value_t wi, wid;
				std::tie(i_w, i_r, wi, wid) = w->perforations[ip];

				res[v] += RHS[(w->well_body_idx + i_w) * n_vars + v] * RHS[(w->well_body_idx + i_w) * n_vars + v];
				norm[v] += PV[w->well_body_idx + i_w] * av_op[v] * PV[w->well_body_idx + i_w] * av_op[v];
			}
		}
		// and then add RHS for well control equations
		for (int v = 0; v < n_vars; v++)
		{
			// well constraints should not be normalized, so pre-multiply by norm
			res[v] += RHS[w->well_head_idx * n_vars + v] * RHS[w->well_head_idx * n_vars + v] * PV[w->well_body_idx] * av_op[v] * PV[w->well_body_idx] * av_op[v];
		}
	}

	for (int v = 0; v < n_vars; v++)
	{
		residual = std::max(residual, sqrt(res[v] / norm[v]));
	}

	return residual;
}

double
engine_base::calc_well_residual_Linf()
{
	double residual = 0, res;

	std::vector<value_t> av_op(n_vars, 0);
	average_operator(av_op);

	for (ms_well *w : wells)
	{
		int nperf = w->perforations.size();
		for (int ip = 0; ip < nperf; ip++)
			for (int v = 0; v < n_vars; v++)
			{
				index_t i_w, i_r;
				value_t wi, wid;
				std::tie(i_w, i_r, wi, wid) = w->perforations[ip];

				res = fabs(RHS[(w->well_body_idx + i_w) * n_vars + v] / (PV[w->well_body_idx + i_w] * av_op[v]));
				residual = std::max(residual, res);
			}

		for (int v = 0; v < n_vars; v++)
		{
			res = fabs(RHS[w->well_head_idx * n_vars + v]);
			residual = std::max(residual, res);
		}
	}

	return residual;
}

double
engine_base::calc_newton_residual()
{

	switch (params->nonlinear_norm_type)
	{
	case sim_params::L1:
	{
		return calc_newton_residual_L1();
	}
	case sim_params::L2:
	{
		return calc_newton_residual_L2();
	}
	case sim_params::LINF:
	{
		return calc_newton_residual_Linf();
	}
	default:
	{
		return calc_newton_residual_L2();
	}
	}
}

double
engine_base::calc_well_residual()
{

	switch (params->nonlinear_norm_type)
	{
	case sim_params::L1:
	{
		return calc_well_residual_L1();
	}
	case sim_params::L2:
	{
		return calc_well_residual_L2();
	}
	case sim_params::LINF:
	{
		return calc_well_residual_Linf();
	}
	default:
	{
		return calc_well_residual_L2();
	}
	}
}

//#define NORMAL_ZC //If you want to use logtransform of zc, i.e. X = [P, log(z1), ..., log(znc-1)] instead of [P, z1, ..., znc-1], comment this line!

int engine_base::apply_newton_update(value_t dt)
{
	timer->node["newton update"].node["composition correction"].start();
	if (nc > 1)
	{
	  if (params->log_transform == 1)
	  {
		apply_composition_correction_new(X, dX);
	  }
	  else
	  {
		apply_composition_correction(X, dX);
	  }
	}
	timer->node["newton update"].node["composition correction"].stop();

	if (params->newton_type == sim_params::NEWTON_GLOBAL_CHOP)
	{
	  if (n_solid > 0)
	  {
		apply_local_chop_correction_with_solid(X, dX);
	  }
	  else
	  {
		if (params->log_transform == 1)
		{
		  apply_global_chop_correction_new(X, dX);
		}
		else
		{
		  apply_global_chop_correction(X, dX);
		}
	  }
	}
	// apply local chop only if number of components is 2 and more
	else if (params->newton_type == sim_params::NEWTON_LOCAL_CHOP && nc > 1)
	{
	  if (n_solid > 0)
	  {
		apply_local_chop_correction_with_solid(X, dX);
	  }
	  else
	  {
		if (params->log_transform == 1)
		{
		  apply_local_chop_correction_new(X, dX);
		}
		else
		{
		  apply_local_chop_correction(X, dX);
		}
	  }
	}

	// apply only if interpolation is used for derivatives
	// make decision based on only the first region
	if (op_axis_min[0].size() > 0)
		apply_obl_axis_local_correction(X, dX);

	// make newton update
	auto newton_update_coefficient_copy = this->newton_update_coefficient;
	std::transform(X.begin(), X.end(), dX.begin(), X.begin(), 
	  [newton_update_coefficient_copy](double x, double dx) {
		return x - newton_update_coefficient_copy * dx;
	  });
	this->newton_update_coefficient = 1.0;

	return 0;
}

void engine_base::apply_composition_correction(std::vector<value_t>& X, std::vector<value_t>& dX)
{
	double sum_z, new_z;
	index_t nb = mesh->n_blocks;
	bool z_corrected;
	index_t n_solid_corrected = 0, n_fluid_corrected = 0;

	for (index_t i = 0; i < nb; i++)
	{
		/* ---- check solid compositions ---- */
		sum_z = 0;
		z_corrected = false;
		for (char c = 0; c < n_solid; c++)
		{
			new_z = X[i * n_vars + z_var + c] - dX[i * n_vars + z_var + c];

			if (new_z < min_zc)
			{
			  new_z = min_zc;
			  z_corrected = true;
			}
			else if (new_z > 1 - min_zc)
			{
			  new_z = 1 - min_zc;
			  z_corrected = true;
			}
			sum_z += new_z;
		}
		// check the last composition
		new_z = 1 - sum_z;
		if (new_z < min_zc)
		{
		  new_z = min_zc;
		  z_corrected = true;
		}
		sum_z += new_z;
		// correction
		if (z_corrected)
		{
		  // normalize compositions and set appropriate update
		  for (char c = 0; c < n_solid; c++)
		  {
			new_z = X[i * n_vars + z_var + c] - dX[i * n_vars + z_var + c];

			new_z = std::max(min_zc, new_z);
			new_z = std::min(1 - min_zc, new_z);

			new_z = new_z / sum_z;
			dX[i * n_vars + z_var + c] = X[i * n_vars + z_var + c] - new_z;
		  }
		  n_solid_corrected++;
		}
		/* ---- end check solid compositions ---- */

		/* ---- check fluid compositions ---- */ 		
		sum_z = 0;
		z_corrected = false;
		for (char c = n_solid; c < nc - 1; c++)
		{
			new_z = X[i * n_vars + z_var + c] - dX[i * n_vars + z_var + c];
			if (new_z < min_zc)
			{
				new_z = min_zc;
				z_corrected = true;
			}
			else if (new_z > 1 - min_zc)
			{
				new_z = 1 - min_zc;
				z_corrected = true;
			}
			sum_z += new_z;
		}
		// check the last composition
		new_z = 1 - sum_z;
		if (new_z < min_zc)
		{
			new_z = min_zc;
			z_corrected = true;
		}
		sum_z += new_z;
		// correction
		if (z_corrected)
		{
			// normalize compositions and set appropriate update
			for (char c = n_solid; c < nc - 1; c++)
			{
				new_z = X[i * n_vars + z_var + c] - dX[i * n_vars + z_var + c];

				new_z = std::max(min_zc, new_z);
				new_z = std::min(1 - min_zc, new_z);

				new_z = new_z / sum_z;
				dX[i * n_vars + z_var + c] = X[i * n_vars + z_var + c] - new_z;
			}
			n_fluid_corrected++;
		}
		/* ---- end check fluid compositions ---- */
	}
	if (n_solid_corrected || n_fluid_corrected)
		std::cout << "Composition correction applied to solid in " << n_solid_corrected << 
		  " block(s), to fluid in " << n_fluid_corrected << " block(s)" << std::endl;
}

void engine_base::apply_composition_correction_(std::vector<value_t> &X, std::vector<value_t> &dX)
{
	double sum_z, new_z, last_z, neg_z;
	index_t nb = mesh->n_blocks;
	index_t n_corrected = 0, c_min;

	for (index_t i = 0; i < nb; i++)
	{
		last_z = 1;
		neg_z = min_zc;
		c_min = -1;
		for (char c = 0; c < nc - 1; c++)
		{
			new_z = X[i * n_vars + z_var + c] - dX[i * n_vars + z_var + c];
			last_z -= new_z; // keep track of last component
			// find smallest component < min_z
			if (new_z < neg_z)
			{
				neg_z = new_z;
				c_min = c;
			}
		}

		if (last_z < neg_z) // check if last component is the smallest < min_z
		{
			double last_dz = 0;
			for (char c = 0; c < nc - 1; c++)
				last_dz += dX[i * n_vars + z_var + c]; // find update for the last component
			last_z -= last_dz; // find old_z = new_z - dX for last component

			if (last_dz != 0)
			{
			  // compute fraction of update to be at min_zc
			  double frac = (min_zc - last_z) / (last_dz);
			  for (char c = 0; c < nc - 1; c++)
				dX[i * n_vars + z_var + c] *= frac;
			  n_corrected++;
			}
		}
		else if (c_min >= 0)
		{
			// compute fraction of update to be at min_zc 
			double frac = -(min_zc - X[i * n_vars + z_var + c_min]) / (dX[i * n_vars + z_var + c_min]);
			if (dX[i * n_vars + z_var + c_min]  != 0)
			{
			  // correct update to be at min_zc for the smallest component
			  for (char c = 0; c < nc - 1; c++)
				dX[i * n_vars + z_var + c] *= frac;
			  n_corrected++;
			}
		}
	}

	if (n_corrected)
		std::cout << "Composition correction applied in " << n_corrected << " block(s)" << std::endl;
}

void engine_base::apply_composition_correction_new(std::vector<value_t> &X, std::vector<value_t> &dX)
{
	double sum_z, new_z, temp_sum, min_count;
	std::vector<value_t> check_vec;
	index_t nb = mesh->n_blocks;
	bool z_corrected;
	index_t n_corrected = 0;

	// Check if solving for the log-transform or regular composition:
	if (params->log_transform == 0)
	{
		// No log-transform is applied to nonlinear unknowns (compositions only), proceed normally:
		for (index_t i = 0; i < nb; i++)
		{
			sum_z = 0;
			temp_sum = 0;		  // sum of any composition not set to z_min
			min_count = 0;		  // number of times a composition is set to z_min
			check_vec.resize(nc); // vector that holds 0 for z_c > z_min && 1 for z_c = z_min
			z_corrected = false;

			// check all but one composition in grid block
			for (index_t c = 0; c < nc - 1; c++)
			{
				new_z = X[i * n_vars + z_var + c] - dX[i * n_vars + z_var + c];

				if (new_z < min_zc)
				{
					//new_z = min_zc * (1 + min_zc);  //TODO: check if this update is consistent!
					new_z = min_zc; //TODO: check if this update is consistent!
					z_corrected = true;
					check_vec[c] = 1;
					min_count += 1;
				}
				else if (new_z > max_zc)
				{
					new_z = max_zc;
					z_corrected = true;
					temp_sum += new_z;
				}
				else
				{
					temp_sum += new_z;
				}
				sum_z += new_z;
			}

			// check the last composition
			new_z = 1 - sum_z;
			if (new_z < min_zc)
			{
				//new_z = min_zc * (1 + min_zc);  //TODO: check if this update is consistent!
				new_z = min_zc;
				z_corrected = true;
				check_vec[nc - 1] = 1;
				min_count += 1;
			}
			else
			{
				temp_sum += new_z;
			}
			sum_z += new_z;

			if (z_corrected)
			{
				// normalize compositions and set appropriate update
				for (index_t c = 0; c < nc - 1; c++)
				{
					new_z = X[i * n_vars + z_var + c] - dX[i * n_vars + z_var + c];

					//new_z = std::max(min_zc * (1 + min_zc), new_z);  //TODO: check if this update is consistent!
					new_z = std::max(min_zc, new_z);
					new_z = std::min(max_zc, new_z);

					if (check_vec[c] != 1)
					{
						//new_z = new_z / temp_sum * (1 - min_count * min_zc * (1 + min_zc));
						new_z = new_z / temp_sum * (1 - min_count * min_zc);
					}

					dX[i * n_vars + z_var + c] = X[i * n_vars + z_var + c] - new_z;
				}
				n_corrected++;
			}
			check_vec.clear();
		}
	}
	else if (params->log_transform == 1)
	{
		// Log-transform is applied to nonlinear unknowns (compositions only), transform back composition exp(log(zc)) to apply correction:
		for (index_t i = 0; i < nb; i++)
		{
			sum_z = 0;
			temp_sum = 0;		  // sum of any composition not set to z_min
			min_count = 0;		  // number of times a composition is set to z_min
			check_vec.resize(nc); // vector that holds 0 for z_c > z_min && 1 for z_c = z_min
			z_corrected = false;

			// check all but one composition in grid block
			for (char c = 0; c < nc - 1; c++)
			{
				new_z = exp(X[i * n_vars + z_var + c] - dX[i * n_vars + z_var + c]); //log based composition

				if (new_z < min_zc)
				{
					//new_z = min_zc * (1 + min_zc);  //TODO: check if this update is consistent!
					new_z = min_zc;
					z_corrected = true;
					check_vec[c] = 1;
					min_count += 1;
				}
				else if (new_z > max_zc)
				{
					new_z = max_zc;
					z_corrected = true;
					temp_sum += new_z;
				}
				else
				{
					temp_sum += new_z;
				}
				sum_z += new_z;
			}

			// check the last composition
			new_z = 1 - sum_z;
			if (new_z < min_zc)
			{
				//new_z = min_zc * (1 + min_zc);  //TODO: check if this update is consistent!
				new_z = min_zc;
				z_corrected = true;
				check_vec[nc - 1] = 1;
				min_count += 1;
			}
			else
			{
				temp_sum += new_z;
			}
			sum_z += new_z;

			if (z_corrected)
			{
				// normalize compositions and set appropriate update
				for (char c = 0; c < nc - 1; c++)
				{
					new_z = exp(X[i * n_vars + z_var + c] - dX[i * n_vars + z_var + c]); //log based composition

					//new_z = std::max(min_zc * (1 + min_zc), new_z);  //TODO: check if this update is consistent!
					new_z = std::max(min_zc, new_z);
					new_z = std::min(max_zc, new_z);

					if (check_vec[c] != 1)
					{
						//new_z = new_z / temp_sum * (1 - min_count * min_zc * (1 + min_zc));
						new_z = new_z / temp_sum * (1 - min_count * min_zc);
					}

					dX[i * n_vars + z_var + c] = log(exp(X[i * n_vars + z_var + c]) / new_z); //log based composition
				}
				n_corrected++;
			}
			check_vec.clear();
		}
	}

	if (n_corrected)
		std::cout << "Composition correction applied in " << n_corrected << " block(s)" << std::endl;
}

void engine_base::apply_global_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX)
{
	double max_ratio = 0;
	index_t n_vars_total = X.size();

	for (index_t i = 0; i < n_vars_total; i++)
	{
		if (fabs(X[i]) > 1e-4)
		{
			double ratio = fabs(dX[i]) / fabs(X[i]);
			max_ratio = (max_ratio < ratio) ? ratio : max_ratio;
		}
	}

	if (max_ratio > params->newton_params[0])
	{
		std::cout << "Apply global chop with max changes = " << max_ratio << "\n";
		for (size_t i = 0; i < n_vars_total; i++)
			dX[i] *= params->newton_params[0] / max_ratio;
	}
}

void engine_base::apply_global_chop_correction_new(std::vector<value_t> &X, std::vector<value_t> &dX)
{
	double max_ratio = 0;
	index_t n_vars_total = X.size();
	double temp_zc = 0;
	double temp_dz = 0;

	if (params->log_transform == 0)
	{
		for (index_t i = 0; i < n_vars_total; i++)
		{
			if (fabs(X[i]) > 1e-4)
			{
				double ratio;
				ratio = fabs(dX[i]) / fabs(X[i]);
				max_ratio = (max_ratio < ratio) ? ratio : max_ratio;
			}
		}

		if (max_ratio > params->newton_params[0])
		{
			std::cout << "Apply global chop with max changes = " << max_ratio << "\n";
			for (size_t i = 0; i < n_vars_total; i++)
			{
				dX[i] *= params->newton_params[0] / max_ratio;
			}
		}
	}
	else if (params->log_transform == 1)
	{
		for (index_t i = 0; i < n_vars_total; i++)
		{
			if (fabs(X[i]) > 1e-4)
			{
				double ratio;
				if (i % nc == 0)
				{
					ratio = fabs(dX[i]) / fabs(X[i]);
				}
				else
				{
					temp_zc = exp(X[i] - dX[i]);
					temp_dz = exp(X[i]) - temp_zc;
					ratio = fabs(temp_dz) / fabs(exp(X[i]));
				}
				max_ratio = (max_ratio < ratio) ? ratio : max_ratio;
			}
		}

		if (max_ratio > params->newton_params[0])
		{
			std::cout << "Apply global chop with max changes = " << max_ratio << "\n";
			for (size_t i = 0; i < n_vars_total; i++)
			{
				dX[i] *= params->newton_params[0] / max_ratio; //log based composition
			}
		}
	}
}

void engine_base::apply_local_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX)
{
	value_t max_dx = params->newton_params[0];
	value_t ratio, dx;
	index_t n_corrected = 0;

	for (int i = 0; i < mesh->n_blocks; i++)
	{
		ratio = 1.0;
		old_z[nc - 1] = 1.0;
		new_z[nc - 1] = 1.0;
		for (int j = 0; j < nc - 1; j++)
		{
			old_z[j] = X[i * n_vars + j + z_var];
			old_z[nc - 1] -= old_z[j];
			new_z[j] = old_z[j] - dX[i * n_vars + j + z_var];
			new_z[nc - 1] -= new_z[j];
		}

		for (int j = 0; j < nc; j++)
		{
			dx = fabs(new_z[j] - old_z[j]);
			if (dx > 0.0001) // if update is not too small
			{
				ratio = std::min<value_t>(ratio, max_dx / dx); // update the ratio
			}
		}

		if (ratio < 1.0) // perform chopping if ratio is below 1.0
		{
			n_corrected++;
			for (int j = z_var; j < z_var + nc - 1; j++)
			{
				dX[i * n_vars + j] *= ratio;
			}
		}
	}
	if (n_corrected)
		std::cout << "Local chop applied in " << n_corrected << " block(s)" << std::endl;
}

void engine_base::apply_local_chop_correction_with_solid(std::vector<value_t> &X, std::vector<value_t> &dX)
{
	value_t max_dx = params->newton_params[0];
	value_t ratio, dx;
	index_t n_corrected = 0;
	uint8_t nc_fl = nc - n_solid;

	for (int i = 0; i < mesh->n_blocks; i++)
	{
		ratio = 1.0;
		old_z_fl[nc_fl - 1] = 1.0;
		new_z_fl[nc_fl - 1] = 1.0;
		for (int j = 0; j < nc_fl - 1; j++)
		{
			old_z_fl[j] = X[i * n_vars + j + z_var + n_solid];
			old_z_fl[nc_fl - 1] -= old_z_fl[j];
			new_z_fl[j] = old_z_fl[j] - dX[i * n_vars + j + z_var + n_solid];
			new_z_fl[nc_fl - 1] -= new_z_fl[j];
		}

		for (int j = 0; j < nc_fl; j++)
		{
			dx = fabs(new_z_fl[j] - old_z_fl[j]);
			if (dx > 0.0001) // if update is not too small
			{
				ratio = std::min<value_t>(ratio, max_dx / dx); // update the ratio
			}
		}

		if (ratio < 1.0) // perform chopping if ratio is below 1.0
		{
			n_corrected++;
			for (int j = z_var + n_solid; j < z_var + nc - 1; j++)
			{
				dX[i * n_vars + j] *= ratio;
			}
		}
	}
	if (n_corrected)
		std::cout << "Local chop applied in " << n_corrected << " block(s)" << std::endl;
}

void engine_base::apply_local_chop_correction_new(std::vector<value_t> &X, std::vector<value_t> &dX)
{
	value_t max_dx = params->newton_params[0];
	value_t ratio, dx;
	index_t n_corrected = 0;

	if (params->log_transform == 0)
	{
		for (int i = 0; i < mesh->n_blocks; i++)
		{
			ratio = 1.0;
			old_z[nc - 1] = 1.0;
			new_z[nc - 1] = 1.0;
			for (int j = 0; j < nc - 1; j++)
			{
				old_z[j] = X[i * n_vars + j + z_var];
				old_z[nc - 1] -= old_z[j];

				new_z[j] = old_z[j] - dX[i * n_vars + j + z_var];
				new_z[nc - 1] -= new_z[j];
			}

			for (int j = 0; j < nc; j++)
			{
				dx = fabs(new_z[j] - old_z[j]);
				if (dx > 0.0001) // if update is not too small
				{
					ratio = std::min<value_t>(ratio, max_dx / dx); // update the ratio
				}
			}

			if (ratio < 1.0) // perform chopping if ratio is below 1.0
			{
				n_corrected++;
				for (int j = z_var; j < z_var + nc - 1; j++)
				{
					dX[i * n_vars + j] *= ratio;
				}
			}
		}
	}
	else if (params->log_transform == 1)
	{
		std::cout << "!!!Using local chop for log-transform of variables is not tested properly, proceed with caution!!!" << std::endl;
		for (int i = 0; i < mesh->n_blocks; i++)
		{
			ratio = 1.0;
			old_z[nc - 1] = 1.0;
			new_z[nc - 1] = 1.0;
			for (int j = 0; j < nc - 1; j++)
			{
				old_z[j] = exp(X[i * n_vars + j + z_var]); //log based composition
				old_z[nc - 1] -= old_z[j];

				new_z[j] = exp(log(old_z[j]) - dX[i * n_vars + j + z_var]); //log based composition
				new_z[nc - 1] -= new_z[j];
			}

			for (int j = 0; j < nc; j++)
			{
				dx = fabs(new_z[j] - old_z[j]);
				if (dx > 0.0001) // if update is not too small
				{
					ratio = std::min<value_t>(ratio, max_dx / dx); // update the ratio
				}
			}

			if (ratio < 1.0) // perform chopping if ratio is below 1.0
			{
				n_corrected++;
				for (int j = z_var; j < z_var + nc - 1; j++)
				{
					dX[i * n_vars + j] *= log(exp(dX[i * n_vars + j]) * ratio); //log based composition
				}
			}
		}
	}
	if (n_corrected)
		std::cout << "Local chop applied in " << n_corrected << " block(s)" << std::endl;
}

void engine_base::apply_obl_axis_local_correction(std::vector<value_t> &X, std::vector<value_t> &dX)
{
	double max_ratio = 0;
	index_t n_vars_total = X.size();
	index_t n_obl_fixes = 0;
	double eps = 1e-15;

	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		for (index_t v = 0; v < n_vars; v++)
		{
			// make sure state values are strictly inside (min;max) interval by using eps
			// otherwise index issues with interpolation can be caused
			value_t axis_min = op_axis_min[mesh->op_num[i]][v] + eps;
			value_t axis_max = op_axis_max[mesh->op_num[i]][v] - eps;
			value_t new_x = X[i * n_vars + v] - dX[i * n_vars + v];
			if (new_x > axis_max)
			{
				dX[i * n_vars + v] = X[i * n_vars + v] - axis_max;
				// output only for the first time
				if (n_obl_fixes == 0)
				{
					std::cout << "OBL axis correction: block " << i << " variable " << v << " shoots over axis limit of " << axis_max << " to " << new_x << std::endl;
				}
				n_obl_fixes++;
			}
			else if (new_x < axis_min)
			{
				dX[i * n_vars + v] = X[i * n_vars + v] - axis_min;
				// output only for the first time
				if (n_obl_fixes == 0)
				{
					std::cout << "OBL axis correction: block " << i << " variable " << v << " shoots under axis limit of " << axis_min << " to " << new_x << std::endl;
				}
				n_obl_fixes++;
			}
		}
	}

	if (n_obl_fixes > 0)
	{
		std::cout << "OBL axis correction applied " << n_obl_fixes << " time(s) \n";
	}
}

int engine_base::test_assembly(int n_times, int kernel_number, int dump_jacobian_rhs)
{
	// timestep does not matter
	double deltat = 1;
	timer->node["jacobian assembly"].timer = 0;
	timer->node["jacobian assembly"].node["kernel"].timer = 0;
	timer->node["jacobian assembly"].node["interpolation"].timer = 0;

	// switch constraints if needed

	for (ms_well *w : wells)
	{
		w->check_constraints(deltat, X);
	}
	timer->node["jacobian assembly"].start();
	// evaluate all operators and their derivatives
	timer->node["jacobian assembly"].node["interpolation"].start();

	for (int i = 0; i < n_times; i++)
	{
		for (auto r = 0; r < acc_flux_op_set_list.size(); r++)
		{
			int result = acc_flux_op_set_list[r]->evaluate_with_derivatives(X, block_idxs[r], op_vals_arr, op_ders_arr);
			if (result < 0)
				return 0;
		}
	}
	timer->node["jacobian assembly"].node["interpolation"].stop();
	timer->node["jacobian assembly"].node["kernel"].start();

	for (int i = 0; i < n_times; i++)
	{
		// assemble jacobian

		assemble_jacobian_array(deltat, X, Jacobian, RHS);
	}
	timer->node["jacobian assembly"].node["kernel"].stop();
	timer->node["jacobian assembly"].stop();

	if (dump_jacobian_rhs)
	{
		char filename[1024];
#ifdef __GNUC__
		int status = -4;
		char *res = abi::__cxa_demangle(typeid(*this).name(), NULL, NULL, &status);
#else
		const char *res = typeid(*this).name();
#endif
		sprintf(filename, "%s_%d_jac.csr", res, kernel_number);
		Jacobian->write_matrix_to_file(filename);
		sprintf(filename, "%s_%d_rhs.vec", res, kernel_number);
		write_vector_to_file(filename, RHS);
	}

	printf("Average assembly %d: %e sec, interpolation %e sec, kernel %e\n", kernel_number, timer->node["jacobian assembly"].get_timer() / n_times,
		   timer->node["jacobian assembly"].node["interpolation"].get_timer() / n_times,
		   timer->node["jacobian assembly"].node["kernel"].get_timer() / n_times);
	//printf ("Average assembly kernel: %e sec\n", timer->node["test_assembly"].get_timer_gpu() / n_times);
	return 0;
}

int engine_base::test_spmv(int n_times, int kernel_number, int dump_result)
{
	timer->node["test_spmv"].timer = 0;
	timer->node["test_spmv"].start();
	for (int i = 0; i < n_times; i++)
	{
		Jacobian->matrix_vector_product(&X[0], &RHS[0]);
	}
	timer->node["test_spmv"].stop();
	printf("Average SPMV kernel: %e sec\n", timer->node["test_spmv"].get_timer() / n_times);
	return 0;
}

int engine_base::assemble_linear_system(value_t deltat)
{
	// switch constraints if needed
	timer->node["jacobian assembly"].start();
	for (ms_well *w : wells)
	{
		w->check_constraints(deltat, X);
	}

	// evaluate all operators and their derivatives
	timer->node["jacobian assembly"].node["interpolation"].start();

	for (int r = 0; r < acc_flux_op_set_list.size(); r++)
	{
		int result = acc_flux_op_set_list[r]->evaluate_with_derivatives(X, block_idxs[r], op_vals_arr, op_ders_arr);
		if (result < 0)
			return 0;
	}

	timer->node["jacobian assembly"].node["interpolation"].stop();

	// assemble jacobian
	assemble_jacobian_array(deltat, X, Jacobian, RHS);

	//Jacobian->write_matrix_to_file("jac_tpfa.txt");

#ifdef WITH_GPU
	if (params->linear_type >= sim_params::GPU_GMRES_CPR_AMG)
	{
		timer->node["jacobian assembly"].node["send_to_device"].start();
		Jacobian->copy_values_to_device();
		timer->node["jacobian assembly"].node["send_to_device"].stop();
	}
#endif

	timer->node["jacobian assembly"].stop();
	return 0;
}

int engine_base::solve_linear_equation()
{
	int r_code;
	char buffer[1024];
	linear_solver_error_last_dt = 0;
	timer->node["linear solver setup"].start();
	r_code = linear_solver->setup(Jacobian);
	timer->node["linear solver setup"].stop();

	if (r_code)
	{
		sprintf(buffer, "ERROR: Linear solver setup returned %d \n", r_code);
		std::cout << buffer << std::flush;
		// use class property to save error state from linear solver
		// this way it will work for both C++ and python newton loop
		//Jacobian->write_matrix_to_file("jac_linear_setup_fail.csr");
		linear_solver_error_last_dt = 1;
		return linear_solver_error_last_dt;
	}

	timer->node["linear solver solve"].start();
	r_code = linear_solver->solve(&RHS[0], &dX[0]);
	timer->node["linear solver solve"].stop();

	if (print_linear_system) //changed this to write jacobian to file!
	{
	  const std::string matrix_filename = "jac_nc_dar_" + std::to_string(output_counter) + ".csr";
#ifdef OPENDARTS_LINEAR_SOLVERS
	  Jacobian->export_matrix_to_file(matrix_filename, opendarts::linear_solvers::sparse_matrix_export_format::csr);
#else
	  Jacobian->write_matrix_to_file_mm(matrix_filename.c_str());
#endif
	  //Jacobian->write_matrix_to_file(("jac_nc_dar_" + std::to_string(output_counter) + ".csr").c_str());
	  write_vector_to_file("jac_nc_dar_" + std::to_string(output_counter) + ".rhs", RHS);
	  write_vector_to_file("jac_nc_dar_" + std::to_string(output_counter) + ".sol", dX);
	  output_counter++;
	}

	if (r_code)
	{
		sprintf(buffer, "ERROR: Linear solver solve returned %d \n", r_code);
		std::cout << buffer << std::flush;
		// use class property to save error state from linear solver
		// this way it will work for both C++ and python newton loop
		linear_solver_error_last_dt = 2;
		return linear_solver_error_last_dt;
	}
	else
	{
		sprintf(buffer, "\t #%d (%.4e, %.4e): lin %d (%.1e)\n", n_newton_last_dt + 1, newton_residual_last_dt,
				well_residual_last_dt, linear_solver->get_n_iters(), linear_solver->get_residual());
		std::cout << buffer << std::flush;
		n_linear_last_dt += linear_solver->get_n_iters();
	}
	return 0;
}

int engine_base::post_newtonloop(value_t deltat, value_t time)
{
	int converged = 0;
	char buffer[1024];
	double well_tolerance_coefficient = 1e2;

	if (linear_solver_error_last_dt == 1) // linear solver setup failed
	{
		sprintf(buffer, "FAILED TO CONVERGE WITH DT = %.3lf (linear solver setup failed) \n", deltat);
	}
	else if (linear_solver_error_last_dt == 2) // linear solver solve failed
	{
		sprintf(buffer, "FAILED TO CONVERGE WITH DT = %.3lf (linear solver solve failed) \n", deltat);
	}
	else if (newton_residual_last_dt >= params->tolerance_newton) // no reservoir convergence reached
	{
		sprintf(buffer, "FAILED TO CONVERGE WITH DT = %.3lf (newton residual reservoir) \n", deltat);
	}
	else if (well_residual_last_dt > well_tolerance_coefficient * params->tolerance_newton) // no well convergence reached
	{
		sprintf(buffer, "FAILED TO CONVERGE WITH DT = %.3lf (newton residual wells) \n", deltat);
	}
	else
	{
		converged = 1;
	}

	if (!converged)
	{
		stat.n_newton_wasted += n_newton_last_dt;
		stat.n_linear_wasted += n_linear_last_dt;
		stat.n_timesteps_wasted++;
		converged = 0;

		X = Xn;

		std::cout << buffer << std::flush;
	}
	else //convergence reached
	{
		stat.n_newton_total += n_newton_last_dt;
		stat.n_linear_total += n_linear_last_dt;
		stat.n_timesteps_total++;
		converged = 1;

		//adjoint method
		if (opt_history_matching == false)
		{
			print_timestep(time + deltat, deltat);
		}

		time_data["time"].push_back(time + deltat);

		for (ms_well *w : wells)
		{
			w->calc_rates(X, op_vals_arr, time_data);
		}

		// calculate FIPS
		FIPS.assign(nc, 0);
		for (index_t i = 0; i < mesh->n_res_blocks; i++)
		{
			for (uint8_t c = 0; c < nc; c++)
			{
				// assuming ACC_OP is 0
				FIPS[c] += PV[i] * op_vals_arr[i * n_ops + 0 + c];
			}
		}

		for (uint8_t c = 0; c < nc; c++)
		{
			time_data["FIPS c " + std::to_string(c) + " (kmol)"].push_back(FIPS[c]);
		}

		Xn = X;
		op_vals_arr_n = op_vals_arr;
		t += dt;



		// adjoint method
		if (opt_history_matching)
		{
			X_t.push_back(X);
			//dt_t.push_back(dt);
			//t_t.push_back(t);
			dt_t.push_back(deltat);
			t_t.push_back(time + deltat);
			if (is_mp)
			{
				Xop_t.push_back(Xop_mp);
			}
			//Jacobian->write_matrix_to_file("jac_forward_simulation.csr");
			//write_vector_to_file("rhs_forward_simulation.rhs", RHS);



			//char buff[100];
			//snprintf(buff, sizeof(buff), "Jacobian_fw_%d.txt", X_t.size()-1);
			//Jacobian->write_matrix_to_file(buff);


			// save the well definition from forward simulation, e.g. control and constraints
			std::vector<ms_well> well_arr;
			for (ms_well* w : wells)
			{
				well_arr.push_back(*(w));
			}

			well_control_arr.push_back(well_arr);

		}

		// evaluate the customized operators and their derivatives
		if (customize_operator)
		{
			index_t r = idx_customized_operator;
			index_t result = acc_flux_op_set_list[r]->evaluate_with_derivatives(X, customize_block_idxs[r], op_vals_arr_customized, op_ders_arr_customized);
			if (result < 0)
				return 0;

			time_data_customized.push_back(op_vals_arr_customized);
		}



	}
	return converged;
}
