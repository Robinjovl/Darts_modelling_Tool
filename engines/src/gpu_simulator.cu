#include <algorithm>
#include <time.h>

#include <iostream>
#include <fstream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>


#include "gpu_simulator.cu.h"
#include "conn_mesh.h"
#include "interp_table.h"
#include "csr_ilu_prec.h"
#include "mv_tools.h"
#include "debug_tools.h"
#include "2d_interpolation.cu.h"
#include "3d_interpolation.cu.h"
#include "jacobian_assembly.cu.h"
#include "Solver.h"

#define USE_CPU_SOLVER

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)



/**
* Check the return value of the CUDA runtime API call and exit
* the application if the call has failed.
*/
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
  if (err == cudaSuccess)
    return;
  std::cerr << statement << " returned " << cudaGetErrorString (err) << "(" << err << ") at " << file << ":" << line << std::endl;
  exit (1);
}

int
gpu_simulator::init (conn_mesh *_mesh, std::string table_base_name)
{

  //////////////////////////////////
  // CPU init
  //////////////////////////////////
  mesh = _mesh;
  acc1 = new interp_table (table_base_name + "_acc_0.txt");
  acc2 = new interp_table (table_base_name + "_acc_1.txt");
  flu1 = new interp_table (table_base_name + "_flu_0.txt");
  flu2 = new interp_table (table_base_name + "_flu_1.txt");

#ifdef USE_CPU_SOLVER
  cpu_solver = new gmres_solver2;
  cpu_preconditioner = new csr_ilu_prec;
  cpu_solver->set_prec (cpu_preconditioner);
#endif

  
  X.resize (2 * mesh->n_blocks);
  Xn.resize (2 * mesh->n_blocks);
  RHS.resize (2 * mesh->n_blocks);
  dX.resize (2 * mesh->n_blocks);
  
  X = mesh->initial_state;
  X.resize(2 * mesh->n_blocks);
  for (index_t i = 0; i < mesh->n_blocks; i++)
  {
    dX[i] = mesh->volume[i] * mesh->poro[i];
  }

  // allocate Jacobian
  index_t nnz = mesh->n_conns + mesh->n_blocks;
  Jacobian.init (mesh->n_blocks, mesh->n_blocks, 2, nnz);
  Jacobian.type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;

  // init Jacobian structure
  index_t *rows_ptr = Jacobian.get_rows_ptr ();
  index_t *diag_ind = Jacobian.get_diag_ind ();
  index_t *cols_ind = Jacobian.get_cols_ind ();
  index_t n_blocks = mesh->n_blocks;
  index_t n_conns = mesh->n_conns;
  std::vector <index_t> block_m = mesh->block_m;
  std::vector <index_t> block_p = mesh->block_p;

  index_t j = 0, k = 0;
  rows_ptr[0] = 0;
  memset (diag_ind, -1, n_blocks * sizeof (index_t)); // t_long <-----> index_t
  for (index_t i = 0; i < n_blocks; i++)
  {
    rows_ptr[i + 1] = rows_ptr[i];
    for (; j < n_conns && block_m[j] == i; j++)
    {
      rows_ptr[i + 1]++;
      if (diag_ind[i] < 0 && block_p[j] > i)
      {
        cols_ind[k] = i;
        diag_ind[i] = k++;
        rows_ptr[i + 1]++;
      }
      cols_ind[k++] = block_p[j];
    }
    if (diag_ind[i] < 0)
    {
      cols_ind[k] = i;
      diag_ind[i] = k++;
      rows_ptr[i + 1]++;
    }
  }


  // Check sorting

  for (index_t i = 0; i < n_blocks; i++)
    {
      index_t j1 = rows_ptr[i];
      index_t j2 = rows_ptr[i + 1] - 1;

      for (index_t j = j1; j < j2; j++)
      {
	  if (cols_ind[j] > cols_ind[j + 1])
	    printf("Cols index are not sorted in row %d!\n", i);
      }
    }


  //////////////////////////////////
  // GPU init
  //////////////////////////////////

#ifndef USE_CPU_SOLVER

#endif

  // interpolation
  index_t interp_data_size = 4 * (acc1->ax1_npoints - 1) * (acc1->ax2_npoints - 1);

  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_acc1_data, sizeof(value_t)*interp_data_size));
  CUDA_CHECK_RETURN(cudaMemcpy(gpu_acc1_data, acc1->data, sizeof(value_t)*interp_data_size, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_acc2_data, sizeof(value_t)*interp_data_size));
  CUDA_CHECK_RETURN(cudaMemcpy(gpu_acc2_data, acc2->data, sizeof(value_t)*interp_data_size, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_flu1_data, sizeof(value_t)*interp_data_size));
  CUDA_CHECK_RETURN(cudaMemcpy(gpu_flu1_data, flu1->data, sizeof(value_t)*interp_data_size, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_flu2_data, sizeof(value_t)*interp_data_size));
  CUDA_CHECK_RETURN(cudaMemcpy(gpu_flu2_data, flu2->data, sizeof(value_t)*interp_data_size, cudaMemcpyHostToDevice));


  // flux: 3 values per block (interpolated value + 2 derivatives)
  // acc:  4 values per block (interpolated value + 2 derivatives + interpolated value from previous timestep)
  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_acc1_res, sizeof(interp_value_t) * 4 * mesh->n_blocks));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_acc2_res, sizeof(interp_value_t) * 4 * mesh->n_blocks));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_flu1_res, sizeof(interp_value_t) * 4 * mesh->n_blocks));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_flu2_res, sizeof(interp_value_t) * 4 * mesh->n_blocks));

  // mesh

  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_block_m, sizeof(index_t)*mesh->n_conns));
  CUDA_CHECK_RETURN(cudaMemcpy(gpu_block_m, &mesh->block_m[0], sizeof(index_t)*mesh->n_conns, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_block_p, sizeof(index_t)*mesh->n_conns));
  CUDA_CHECK_RETURN(cudaMemcpy(gpu_block_p, &mesh->block_p[0], sizeof(index_t)*mesh->n_conns, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_tran, sizeof(value_t)*mesh->n_conns));
  CUDA_CHECK_RETURN(cudaMemcpy(gpu_tran, &mesh->tran[0], sizeof(value_t)*mesh->n_conns, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_PV, sizeof(value_t)*mesh->n_blocks));
  CUDA_CHECK_RETURN(cudaMemcpy(gpu_PV, &dX[0], sizeof(value_t)*mesh->n_blocks, cudaMemcpyHostToDevice));

  // initial solution

  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_x, sizeof(value_t)* 2 * mesh->n_blocks));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_xn, sizeof(value_t)* 2 * mesh->n_blocks));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_dx, sizeof(value_t)* 2 * mesh->n_blocks));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_rhs, sizeof(value_t)* 2 * mesh->n_blocks));

  CUDA_CHECK_RETURN(cudaMemcpy(gpu_x, &X[0], sizeof(value_t)* 2 * mesh->n_blocks, cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemcpy(gpu_xn, gpu_x, sizeof(value_t)* 2 * mesh->n_blocks, cudaMemcpyDeviceToDevice));

  // jacobian
  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_jac_rows_ptr, sizeof(index_t)* (mesh->n_blocks + 1)));
  CUDA_CHECK_RETURN(cudaMemcpy(gpu_jac_rows_ptr, Jacobian.get_rows_ptr(), sizeof(index_t)* (mesh->n_blocks + 1), cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_jac_cols_ind, sizeof(index_t)* nnz));
  CUDA_CHECK_RETURN(cudaMemcpy(gpu_jac_cols_ind, Jacobian.get_cols_ind(), sizeof(index_t) * nnz, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_jac_values, sizeof(value_t) * nnz * 4));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&gpu_jac_values_ilu, sizeof(value_t) * nnz * 4));

};


int gpu_simulator::assemble_jacobian (value_t dt, int is_first)
{
  const int interp_cuda_blocks = (mesh->n_blocks + INTERP_BLOCK_SIZE - 1)/INTERP_BLOCK_SIZE;
  const int assembly_cuda_blocks = (mesh->n_blocks + ASSEMBLY_BLOCK_SIZE - 1) / ASSEMBLY_BLOCK_SIZE;
  const int simple_ops_cuda_blocks = (mesh->n_blocks + SIMPLE_OPS_BLOCK_SIZE - 1) / SIMPLE_OPS_BLOCK_SIZE;
  float assemble_timer = 0.0f;

  // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    cudaError_t error;

    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
	    fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
	    exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
	    fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
	    exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
	    fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
	    exit(EXIT_FAILURE);
    }

  // Step 1. Calculate interpolations

  bilinear_interpolation_kernel<int, interp_value_t, INTERP_BLOCK_SIZE><<<interp_cuda_blocks, INTERP_BLOCK_SIZE>>>
  				(mesh->n_blocks,
  				 acc1->ax1_npoints, acc1->ax1_min, acc1->ax1_step_inv,
  				 acc1->ax2_npoints, acc1->ax2_min, acc1->ax2_step_inv,
  				 gpu_acc1_data, gpu_x, gpu_acc1_res);
  bilinear_interpolation_kernel<int, interp_value_t, INTERP_BLOCK_SIZE><<<interp_cuda_blocks, INTERP_BLOCK_SIZE>>>
  				(mesh->n_blocks,
  				 acc2->ax1_npoints, acc2->ax1_min, acc2->ax1_step_inv,
  				 acc2->ax2_npoints, acc2->ax2_min, acc2->ax2_step_inv,
  				 gpu_acc2_data, gpu_x, gpu_acc2_res);
  bilinear_interpolation_kernel<int, interp_value_t, INTERP_BLOCK_SIZE><<<interp_cuda_blocks, INTERP_BLOCK_SIZE>>>
  				(mesh->n_blocks,
  				 flu1->ax1_npoints, flu1->ax1_min, flu1->ax1_step_inv,
  				 flu1->ax2_npoints, flu1->ax2_min, flu1->ax2_step_inv,
  				 gpu_flu1_data, gpu_x, gpu_flu1_res);
  bilinear_interpolation_kernel<int, interp_value_t, INTERP_BLOCK_SIZE><<<interp_cuda_blocks, INTERP_BLOCK_SIZE>>>
  				(mesh->n_blocks,
  				 flu2->ax1_npoints, flu2->ax1_min, flu2->ax1_step_inv,
  				 flu2->ax2_npoints, flu2->ax2_min, flu2->ax2_step_inv,
  				 gpu_flu2_data, gpu_x, gpu_flu2_res);
  trilinear_interpolation_kernel<int, interp_value_t, INTERP_BLOCK_SIZE><<<interp_cuda_blocks, INTERP_BLOCK_SIZE>>>
  				(mesh->n_blocks,
  				 flu2->ax1_npoints, flu2->ax1_min, flu2->ax1_step_inv,
  				 flu2->ax2_npoints, flu2->ax2_min, flu2->ax2_step_inv,
  				 flu2->ax2_npoints, flu2->ax2_min, flu2->ax2_step_inv,
  				 gpu_flu2_data, gpu_x, gpu_flu2_res);

  if (is_first)
    copy_acc_interpolation<index_t, interp_value_t, SIMPLE_OPS_BLOCK_SIZE> <<<simple_ops_cuda_blocks, SIMPLE_OPS_BLOCK_SIZE>>>
	(mesh->n_blocks, gpu_acc1_res, gpu_acc2_res);


  // Step2. Assemble Jacobian

  jacobian_assembly_kernel<index_t, value_t, interp_value_t, ASSEMBLY_BLOCK_SIZE> << <simple_ops_cuda_blocks, ASSEMBLY_BLOCK_SIZE >> >
					      (mesh->n_blocks, dt, gpu_jac_rows_ptr, gpu_jac_cols_ind, gpu_jac_values,
						  gpu_rhs, gpu_x, gpu_tran, gpu_PV,
						  gpu_acc1_res, gpu_acc2_res, gpu_flu1_res, gpu_flu2_res);



  // Record the stop event
  error = cudaEventRecord(stop, NULL);

  if (error != cudaSuccess)
  {
	  fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
	  exit(EXIT_FAILURE);
  }

  // Wait for the stop event to complete
  error = cudaEventSynchronize(stop);

  if (error != cudaSuccess)
  {
	  fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
	  exit(EXIT_FAILURE);
  }

  error = cudaEventElapsedTime(&assemble_timer, start, stop);
  gpu_assemble_timer += assemble_timer;
  //printf ("Interpolations are built in %lf msec \n", interpolation_timer);



  //Jacobian.write_matrix_to_file ("gpu_jac_0.csr");
  ////write_vector_to_file ("gpu_jac_0.rhs", &RHS[0], 2 * mesh->n_blocks);

  //exit(0);

  return 0;
};

int gpu_simulator::run (sim_params *params)
{

  time_t rawtime;
  struct tm * timeinfo;
  char buffer[1024];

  time (&rawtime);
  timeinfo = localtime (&rawtime);
  //write_vector_to_file ("X_init", &X[0], 2 * mesh->n_blocks);

  strftime (buffer, 120, "DARTS-%Y-%m-%d-%H-%M.log", timeinfo);
  std::ofstream log (buffer);
  log << "GPU\nSim params: \n" << "\tFirst ts: \t" << params->first_ts << std::endl;
  log << "\tMax ts: \t" << params->max_ts << std::endl;
  log << "\tMult ts: \t" << params->mult_ts << std::endl;
  log << "\tTotal ts: \t" << params->total_time << std::endl;
  
  log << "\tMax i newton: \t" << params->max_i_newton << std::endl;
  log << "\tMax i linear: \t" << params->max_i_linear << std::endl;
  log << "\tTol newton: \t" << params->tolerance_newton << std::endl;
  log << "\tTol linear: \t" << params->tolerance_linear << std::endl;
  log << "\tResolution: \t" << acc1->ax1_npoints << std::endl;

  // initialize
  value_t t, dt, residual_newton;
  index_t i_newt, tot_newt = 0, tot_linear = 0, tot_linear_per_step;
  index_t r_code = 0, ts_num, last_ts;
  double assemble_timer = 0, linear_setup_timer = 0, linear_solve_timer = 0, total_timer = 0;
  double communication_timer = 0;
  gpu_assemble_timer = 0;
  const int simple_ops_cuda_blocks = (mesh->n_blocks + SIMPLE_OPS_BLOCK_SIZE - 1) / SIMPLE_OPS_BLOCK_SIZE;
  index_t nnz =  mesh->n_conns + mesh->n_blocks;
  
  Xn = X;
  dt = params->first_ts;

#ifdef USE_CPU_SOLVER
  cpu_solver->prop.set_max_iters (params->max_i_linear);
  cpu_solver->prop.set_tolerance (params->tolerance_linear);
  ((gmres_solver2 *)cpu_solver)->m = params->max_i_linear;
  ((csr_ilu_prec*)cpu_preconditioner)->set_diag_ind (Jacobian.get_diag_ind (), mesh->n_blocks);
#else
  phaseDim = 2;
  maxIt = params->max_i_linear;
  iigmres = params->max_i_linear;
  tol = params->tolerance_linear;
  logInfo = new LogInfo(maxIt);
  logInfo->Reset();

  gpu_solver = caseLinearSolver(SOLVER_fgmres);
  gpu_preconditioner = casePreconditioner();
  gpu_solver->setPreconditioner(gpu_preconditioner);
#endif

  total_timer -= clock ();

  for (t = dt, ts_num = 1, last_ts = 0; ; t += dt, ts_num++)
  {
    sprintf (buffer, "%d t = %.3lf: dt %.3lf\n", ts_num, t, dt);
    printf (buffer);
    log << buffer << std::flush;

	tot_linear_per_step = 0;
    for (i_newt = 0; i_newt < params->max_i_newton; i_newt++)
    {
      assemble_timer -= clock ();
      assemble_jacobian (dt, ts_num == 1 && i_newt == 0);
      assemble_timer += clock ();


      communication_timer -= clock ();
#ifdef USE_CPU_SOLVER
      CUDA_CHECK_RETURN(cudaMemcpy(Jacobian.get_values(), gpu_jac_values, sizeof(value_t) * 4 * nnz, cudaMemcpyDeviceToHost));
      CUDA_CHECK_RETURN(cudaMemcpy(&RHS[0], gpu_rhs, sizeof(value_t) * 2 * mesh->n_blocks, cudaMemcpyDeviceToHost));
      //Jacobian.write_matrix_to_file ("jac_11.csr");
      //Jacobian.write_matrix_to_file_ij("jac_ij");
      ////write_vector_to_file ("rhs", &RHS[0], 2 * mesh->n_blocks);
      //exit(0);
#else
      logInfo->Reset();
      CUDA_CHECK_RETURN(cudaMemcpy(gpu_jac_values_ilu, gpu_jac_values, sizeof(value_t) * 4 * nnz, cudaMemcpyDeviceToDevice));
      CUDA_CHECK_RETURN(cudaMemcpy(&RHS[0], gpu_rhs, sizeof(value_t) * 2 * mesh->n_blocks, cudaMemcpyDeviceToHost));
#endif
      communication_timer += clock ();


      residual_newton = sqrt (mv_vector_inner_product (&RHS[0], &RHS[0], mesh->n_blocks, 2));

      // exit if target tolerance reached
      if (residual_newton < params->tolerance_newton)
        break;

      
      linear_setup_timer -= clock ();

#ifdef USE_CPU_SOLVER
      r_code = cpu_solver->setup (&Jacobian);
      
      if (r_code)
      {
        printf ("ERROR: Linear solver setup returned %d\n", r_code);
        exit (1);
      }

#else
      gpu_preconditioner->setup(gpu_jac_rows_ptr, gpu_jac_cols_ind, gpu_jac_values_ilu, mesh->n_blocks, nnz, 2);
      gpu_solver->setMatrix(gpu_jac_rows_ptr, gpu_jac_cols_ind, gpu_jac_values, mesh->n_blocks, nnz, 2);
#endif
      linear_setup_timer += clock ();


      linear_solve_timer -= clock ();
#ifdef USE_CPU_SOLVER
      r_code = cpu_solver->solve (&Jacobian, &RHS[0], &dX[0]);
#else
      cudaMemset(gpu_dx, 0, sizeof(value_t) * 2 * mesh->n_blocks);
      gpu_solver->solve(gpu_rhs, gpu_dx);
#endif
      linear_solve_timer += clock ();

#ifdef USE_CPU_SOLVER
      if (r_code)
	{
	  printf ("ERROR: Linear solver solve returned %d\n", r_code);
	  exit (1);
	}
	else
	{
	  sprintf (buffer, "\t newton %d (tol %.e): linear %d (tol %.2e)\n", i_newt + 1, residual_newton, cpu_solver->prop.get_iters (), cpu_solver->prop.final_resid);
      printf (buffer);
      log << buffer << std::flush;
	  tot_linear += cpu_solver->prop.get_iters ();
	  tot_linear_per_step += cpu_solver->prop.get_iters ();
	}
      // make newton update
      std::transform (X.begin (), X.end (), dX.begin (), X.begin (), std::minus<double> ());

      // check if solution is physical
      for (index_t i = 0; i < mesh->n_blocks; i++)
      	{
      	  if (X[2 * i + 1] < 0 || X[2 * i + 1] > 1)
      	  {
      	    printf ("!!!!!!!!!!!!!!!!!!!!!!! Error: non-physical composition solution\n");
      	    exit(1);
      	  }
      	}

      communication_timer -= clock ();
      CUDA_CHECK_RETURN(cudaMemcpy(gpu_x, &X[0], sizeof(value_t) * 2 * mesh->n_blocks, cudaMemcpyHostToDevice));
      communication_timer += clock ();
#else //USE_CPU_SOLVER
      sprintf (buffer, "\t newton %d (tol %.e): linear %d (tol %.2e)\n",  i_newt + 1, residual_newton, int(logInfo->lastIt), logInfo->lastrel);
      printf (buffer);
      log << buffer << std::flush;
	  tot_linear += int (logInfo->lastIt);
	  tot_linear_per_step += int (logInfo->lastIt);

      communication_timer -= clock ();

      CUDA_CHECK_RETURN(cudaMemcpy(&dX[0], gpu_dx, sizeof(value_t) * 2 * mesh->n_blocks, cudaMemcpyDeviceToHost));
      std::transform (X.begin (), X.end (), dX.begin (), X.begin (), std::minus<double> ());
      CUDA_CHECK_RETURN(cudaMemcpy(gpu_x, &X[0], sizeof(value_t) * 2 * mesh->n_blocks, cudaMemcpyHostToDevice));

      communication_timer += clock ();

      // check if solution is physical
	  r_code = 0;
      for (index_t i = 0; i < mesh->n_blocks; i++)
		{
		  if (X[2 * i + 1] < LOWER_LIMIT)
			{
				r_code = 1;
				X[2 * i + 1] = LOWER_LIMIT + LOWER_LIMIT / 10;
			}
			if (X[2 * i + 1] > UPPER_LIMIT)
			{
				r_code = 1;
				X[2 * i + 1] = UPPER_LIMIT - LOWER_LIMIT / 10;
			}
		}
		if (r_code)
        printf ("Error: non-physical composition solution. Fixed.\n");
#endif

    }
	sprintf (buffer, "\t total newton %d (tol %.2e) linear %d\n", i_newt, residual_newton, tot_linear_per_step);
    printf (buffer);
    log << buffer << std::flush;
    tot_newt += i_newt;

    if (i_newt == params->max_i_newton && residual_newton > params->tolerance_newton)
    {
	  t -= dt;
      dt /= params->mult_ts;
      X = Xn;
      printf ("\t RESTART\n");
      log << "\t RESTART\n" << std::flush;
      ts_num--;

      communication_timer -= clock ();
      CUDA_CHECK_RETURN(cudaMemcpy(gpu_x, &X[0], sizeof(value_t) * 2 * mesh->n_blocks, cudaMemcpyHostToDevice));
      communication_timer += clock ();

      }
    else
      {

      if (last_ts)
        break;

      dt *= params->mult_ts;
      if (dt > params->max_ts)
        dt = params->max_ts;

      if (!last_ts && (t + dt + 1e-10) >= params->total_time)
      {
        dt = params->total_time - t;
        last_ts = 1;
      }
      Xn = X;
      copy_acc_interpolation<index_t, interp_value_t, SIMPLE_OPS_BLOCK_SIZE> << <simple_ops_cuda_blocks, SIMPLE_OPS_BLOCK_SIZE >> >
      	(mesh->n_blocks, gpu_acc1_res, gpu_acc2_res);
    }
  }
  total_timer += clock ();
  //write_vector_to_file ("gpu_X_final", &X[0], 2 * mesh->n_blocks);
  assemble_timer /= CLOCKS_PER_SEC;
  linear_setup_timer /= CLOCKS_PER_SEC;
  linear_solve_timer /= CLOCKS_PER_SEC;
  total_timer /= CLOCKS_PER_SEC;
  communication_timer /= CLOCKS_PER_SEC;

  r_code += sprintf (buffer, "Total steps %d newton %d linear %d\n", ts_num, tot_newt, tot_linear);
  r_code += sprintf (buffer + r_code, "Total elapsed %.1lf sec:\n", total_timer);
  r_code += sprintf (buffer + r_code, "\t assemble %.4lf sec\n", assemble_timer);
  r_code += sprintf (buffer + r_code, "\t setup %.4lf sec\n", linear_setup_timer);
  r_code += sprintf (buffer + r_code, "\t solve %.4lf sec\n", linear_solve_timer);
  r_code += sprintf (buffer + r_code, "\t communication %.4lf sec\n", communication_timer);
  r_code += sprintf (buffer + r_code, "\t gpu assemble %.4lf sec\n", gpu_assemble_timer / 1000);

  printf (buffer);
  log << buffer << std::flush;

  return 0;
}

int gpu_simulator::gpu_test (int argc, char** argv)
{
	double *x = NULL, *b = NULL;

//чтение с диска
	coo* ilu = new coo;
	coo* A = new coo;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-A")) {
		    printf ("loading matrix...\n");
			if (read_coo_MM(argv[i + 1], A) != 0)
				exit(1);
		}

		if (!strcmp(argv[i], "-rhs")) {
		    printf ("loading vector...\n");
			if (read_coo_vector(argv[i + 1], &b) != 0)
				exit(1);
		}
	}


// глобальные переменные, объявлены в decl.h, позволяют регулировать параметры решателя, необходимо выставить до инициализации решателя и предобуславливателя.
// размер блока bcsr по умолчанию равен 3
	phaseDim = 2;
// максимальное число итераций решателя по умолчанию 100
	maxIt = 100;

//максимальное число внутренних итераций gmres - по умолчанию 30
	iigmres = 30;

//конвертации для работоспособности примера
	csr* A_mat = new csr(A);
	printf ("n nnz dim: %d %d %d\n", A_mat->n, A_mat->nnz, phaseDim);
	delete A;
	bsr* bsr_mat = new bsr(A_mat, phaseDim);
	printf ("n nnz dim: %d %d %d\n", bsr_mat->n, bsr_mat->nnz, phaseDim);
	printf ("Rows: %d %d %d\n", bsr_mat->ia[0], bsr_mat->ia[1000], bsr_mat->ia[1001]);
	printf ("Cols: %d %d %d %d \n", bsr_mat->ja[0], bsr_mat->ja[1], bsr_mat->ja[3002], bsr_mat->ja[3003], bsr_mat->ja[4]);

	//delete A_mat;
// обязательно необходимо проинициализировать logInfo (монитор сходимости, сохраняет промежуточные невязки) - он вшит в решатель.
	logInfo = new LogInfo(maxIt);

//  сброс монитора и таймеров перед началом итерационного процесса. Библиотека собрана со своими таймерами, будут доступны при выставлении -DTIMER при компиляции
	logInfo->Reset();
#ifdef TIMER
	//timersInfo.Reset();
#endif

	x = (double*) calloc(A_mat->n, sizeof(double));

//caseLinearSolver принимает enum: SOLVER_fgmres, SOLVER_gmres, SOLVER_bicgstab
	LinearSolver* solver = caseLinearSolver(SOLVER_fgmres);
	//GMRESGPU* solver = new GMRESGPU(1e-6);
//включили только BCSR ILU0, поэтому без аргументов
	Preconditioner* prec = casePreconditioner();
//в setup и setMatrix происходит копирование данных матрицы во внутренние структуры, независимо от того, где лежат данные: на GPU или CPU.
	prec->setup      (bsr_mat->ia, bsr_mat->ja, bsr_mat->val, bsr_mat->n, bsr_mat->nnz, phaseDim);
	solver->setMatrix(bsr_mat->ia, bsr_mat->ja, bsr_mat->val, bsr_mat->n, bsr_mat->nnz, phaseDim);

	//delete bsr_mat;

	solver->setPreconditioner(prec);
	solver->solve(b, x);


// печать невязок и таймеров
	logInfo->Print();
	logInfo->PrintLast();
#ifdef TIMER
	timersInfo.Print();
#endif
	delete logInfo;
	delete prec;
	delete solver;
	free(b);
	free(x);
}
