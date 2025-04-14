// Option 1: assemble in single pass, one block (row) per thread
template<typename IndexType, typename ValueType, typename InterpValueType,
    int _BLOCK_SIZE>
  __global__ void
  jacobian_assembly_kernel (const int n_blocks,
			    const ValueType dt, const IndexType *rows_ptr,
			    const IndexType *cols_ind, ValueType *Jac,
			    ValueType *RHS, const ValueType *X,
			    const ValueType *tran, const ValueType *PV,
			    const InterpValueType *acc1,
			    const InterpValueType *acc2,
			    const InterpValueType *flu1,
			    const InterpValueType *flu2)
  {
    const unsigned i = blockIdx.x * _BLOCK_SIZE + threadIdx.x; // global thread index
    if (i > n_blocks - 1)
      return;
    //IndexType start = rows_ptr[i];
    //IndexType end = rows_ptr[i + 1];
    IndexType jac_diag_idx;
    ValueType p_diff, gamma_p_diff, jtran_dt;

    //__shared__ ValueType wkspc[BLOCK_SIZE * 6];
    //ValueType *jac_d = wkspc + 6 * threadIdx.x;
    //ValueType *rhs = wkspc + 6 * threadIdx.x + 4;
    ValueType jac_d[4];
    ValueType rhs[2];

    jac_d[0] = 0;
    jac_d[1] = 0;
    jac_d[2] = 0;
    jac_d[3] = 0;
    rhs[0] = 0;
    rhs[1] = 0;

    for (IndexType j = rows_ptr[i]; j < rows_ptr[i + 1]; j++)
      {
	unsigned cl = cols_ind[j];

	if (i == cl)
	  {
	    // fill diagonal part
	    rhs[0] += PV[i] * (acc1[4 * i] - acc1[4 * i + 3]);
	    rhs[1] += PV[i] * (acc2[4 * i] - acc2[4 * i + 3]);

	    jac_diag_idx = j;
	    jac_d[0] += PV[i] * acc1[4 * i + 1];
	    jac_d[1] += PV[i] * acc1[4 * i + 2];
	    jac_d[2] += PV[i] * acc2[4 * i + 1];
	    jac_d[3] += PV[i] * acc2[4 * i + 2];
	  }
	else
	  {
	    // fill offdiagonal part + contribute to diagonal

	    p_diff = X[2 * cl] - X[2 * i];

	    if (cl < i)
	      jtran_dt = tran[j - i] * dt;
	    else
	      jtran_dt = tran[j - i - 1] * dt;
	    gamma_p_diff = jtran_dt * p_diff;

	    if (p_diff < 0)
	      {
		//outflow
		rhs[0] -= gamma_p_diff * flu1[4 * i];
		rhs[1] -= gamma_p_diff * flu2[4 * i];

		jac_d[0] -= gamma_p_diff * flu1[4 * i + 1];
		jac_d[1] -= gamma_p_diff * flu1[4 * i + 2];
		jac_d[2] -= gamma_p_diff * flu2[4 * i + 1];
		jac_d[3] -= gamma_p_diff * flu2[4 * i + 2];

		Jac[4 * j + 0] = -jtran_dt * flu1[4 * i];
		jac_d[0]      -= -jtran_dt * flu1[4 * i];

		Jac[4 * j + 2] = -jtran_dt * flu2[4 * i];
		jac_d[2]      -= -jtran_dt * flu2[4 * i];

		Jac[4 * j + 1] = 0;
		Jac[4 * j + 3] = 0;
	      }
	    else
	      {
		rhs[0] -= gamma_p_diff * flu1[4 * cl];
		rhs[1] -= gamma_p_diff * flu2[4 * cl];

		jac_d[0] += jtran_dt * flu1[4 * cl];
		jac_d[2] += jtran_dt * flu2[4 * cl];

		Jac[4 * j + 0] = -gamma_p_diff * flu1[4 * cl + 1] - jtran_dt * flu1[4 * cl];
		Jac[4 * j + 1] = -gamma_p_diff * flu1[4 * cl + 2];
		Jac[4 * j + 2] = -gamma_p_diff * flu2[4 * cl + 1] - jtran_dt * flu2[4 * cl];
		Jac[4 * j + 3] = -gamma_p_diff * flu2[4 * cl + 2];
	      }
	  }
      }

    RHS[2 * i + 0] = rhs[0];
    RHS[2 * i + 1] = rhs[1];
    Jac[4 * jac_diag_idx + 0] = jac_d[0];
    Jac[4 * jac_diag_idx + 1] = jac_d[1];
    Jac[4 * jac_diag_idx + 2] = jac_d[2];
    Jac[4 * jac_diag_idx + 3] = jac_d[3];

  }


template<typename IndexType, typename InterpValueType, int _BLOCK_SIZE>
  __global__ void
  copy_acc_interpolation (const int n_blocks, InterpValueType *acc1, InterpValueType *acc2)
  {
    const unsigned i = blockIdx.x * _BLOCK_SIZE + threadIdx.x; // global thread index
    if (i > n_blocks - 1)
      return;
    acc1[4 * i + 3] = acc1[4 * i];
    acc2[4 * i + 3] = acc2[4 * i];
  }

