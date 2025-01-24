// One thread per cell (per jacobian row)
template<typename IndexType, typename ValueType, typename InterpValueType,
  int _BLOCK_SIZE>
  __global__ void
  jacobian_nc3_assembly_kernel (const int n_blocks,
  const ValueType dt, const IndexType *rows_ptr,
  const IndexType *cols_ind, ValueType *Jac,
  ValueType *RHS, const ValueType *X,
  const ValueType *tran, const ValueType *PV,
  const InterpValueType *acc1,
  const InterpValueType *acc2,
  const InterpValueType *acc3,
  const InterpValueType *flu1,
  const InterpValueType *flu2,
  const InterpValueType *flu3,
  const InterpValueType *acc_n)
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
    ValueType jac_d[9];
    ValueType rhs[3];

    jac_d[0] = 0;
    jac_d[1] = 0;
    jac_d[2] = 0;
    jac_d[3] = 0;
    jac_d[4] = 0;
    jac_d[5] = 0;
    jac_d[6] = 0;
    jac_d[7] = 0;
    jac_d[8] = 0;

    rhs[0] = 0;
    rhs[1] = 0;
    rhs[2] = 0;

    for (IndexType j = rows_ptr[i]; j < rows_ptr[i + 1]; j++)
    {
      unsigned cl = cols_ind[j];

      if (i == cl)
      {
        // fill diagonal part
        rhs[0] += PV[i] * (acc1[4 * i] - acc_n[4 * i]);
        rhs[1] += PV[i] * (acc2[4 * i] - acc_n[4 * i + 1]);
        rhs[2] += PV[i] * (acc3[4 * i] - acc_n[4 * i + 2]);

        jac_diag_idx = j;
        jac_d[0] += PV[i] * acc1[4 * i + 1];
        jac_d[1] += PV[i] * acc1[4 * i + 2];
        jac_d[2] += PV[i] * acc1[4 * i + 3];
        jac_d[3] += PV[i] * acc2[4 * i + 1];
        jac_d[4] += PV[i] * acc2[4 * i + 2];
        jac_d[5] += PV[i] * acc2[4 * i + 3];
        jac_d[6] += PV[i] * acc3[4 * i + 1];
        jac_d[7] += PV[i] * acc3[4 * i + 2];
        jac_d[8] += PV[i] * acc3[4 * i + 3];
      }
      else
      {
        // fill offdiagonal part + contribute to diagonal

        p_diff = X[3 * cl] - X[3 * i];

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
          rhs[2] -= gamma_p_diff * flu3[4 * i];

          jac_d[0] -= gamma_p_diff * flu1[4 * i + 1];
          jac_d[1] -= gamma_p_diff * flu1[4 * i + 2];
          jac_d[2] -= gamma_p_diff * flu1[4 * i + 3];
          jac_d[3] -= gamma_p_diff * flu2[4 * i + 1];
          jac_d[4] -= gamma_p_diff * flu2[4 * i + 2];
          jac_d[5] -= gamma_p_diff * flu2[4 * i + 3];
          jac_d[6] -= gamma_p_diff * flu3[4 * i + 1];
          jac_d[7] -= gamma_p_diff * flu3[4 * i + 2];
          jac_d[8] -= gamma_p_diff * flu3[4 * i + 3];

          Jac[9 * j + 0] = -jtran_dt * flu1[4 * i];
          jac_d[0] -= -jtran_dt * flu1[4 * i];

          Jac[9 * j + 3] = -jtran_dt * flu2[4 * i];
          jac_d[3] -= -jtran_dt * flu2[4 * i];

          Jac[9 * j + 6] = -jtran_dt * flu3[4 * i];
          jac_d[6] -= -jtran_dt * flu3[4 * i];

          Jac[9 * j + 1] = 0;
          Jac[9 * j + 2] = 0;
          Jac[9 * j + 4] = 0;
          Jac[9 * j + 5] = 0;
          Jac[9 * j + 7] = 0;
          Jac[9 * j + 8] = 0;
        }
        else
        {
          rhs[0] -= gamma_p_diff * flu1[4 * cl];
          rhs[1] -= gamma_p_diff * flu2[4 * cl];
          rhs[2] -= gamma_p_diff * flu3[4 * cl];

          jac_d[0] += jtran_dt * flu1[4 * cl];
          jac_d[3] += jtran_dt * flu2[4 * cl];
          jac_d[6] += jtran_dt * flu3[4 * cl];

          Jac[9 * j + 0] = -gamma_p_diff * flu1[4 * cl + 1] - jtran_dt * flu1[4 * cl];
          Jac[9 * j + 1] = -gamma_p_diff * flu1[4 * cl + 2];
          Jac[9 * j + 2] = -gamma_p_diff * flu1[4 * cl + 3];
          Jac[9 * j + 3] = -gamma_p_diff * flu2[4 * cl + 1] - jtran_dt * flu2[4 * cl];
          Jac[9 * j + 4] = -gamma_p_diff * flu2[4 * cl + 2];
          Jac[9 * j + 5] = -gamma_p_diff * flu2[4 * cl + 3];
          Jac[9 * j + 6] = -gamma_p_diff * flu3[4 * cl + 1] - jtran_dt * flu3[4 * cl];
          Jac[9 * j + 7] = -gamma_p_diff * flu3[4 * cl + 2];
          Jac[9 * j + 8] = -gamma_p_diff * flu3[4 * cl + 3];
        }
      }
    }

    RHS[3 * i + 0] = rhs[0];
    RHS[3 * i + 1] = rhs[1];
    RHS[3 * i + 2] = rhs[2];

    Jac[9 * jac_diag_idx + 0] = jac_d[0];
    Jac[9 * jac_diag_idx + 1] = jac_d[1];
    Jac[9 * jac_diag_idx + 2] = jac_d[2];
    Jac[9 * jac_diag_idx + 3] = jac_d[3];
    Jac[9 * jac_diag_idx + 4] = jac_d[4];
    Jac[9 * jac_diag_idx + 5] = jac_d[5];
    Jac[9 * jac_diag_idx + 6] = jac_d[6];
    Jac[9 * jac_diag_idx + 7] = jac_d[7];
    Jac[9 * jac_diag_idx + 8] = jac_d[8];
  }

  // various helper kernels

  template<typename IndexType, typename InterpValueType, int _BLOCK_SIZE>
  __global__ void
    copy_acc_interpolation_nc3 (const int n_blocks, InterpValueType *acc1,
    InterpValueType *acc2, InterpValueType *acc3, InterpValueType *acc_n)
  {
    const unsigned i = blockIdx.x * _BLOCK_SIZE + threadIdx.x; // global thread index
    if (i > n_blocks - 1)
      return;
    acc_n[4 * i + 0] = acc1[4 * i];
    acc_n[4 * i + 1] = acc2[4 * i];
    acc_n[4 * i + 2] = acc3[4 * i];
  }

  template<typename IndexType, typename ValueType, int _BLOCK_SIZE>
  __global__ void
    newton_update_with_correction (const int n_blocks, ValueType *gpu_x, ValueType *gpu_dx)
  {
    const unsigned i = blockIdx.x * _BLOCK_SIZE + threadIdx.x; // global thread index
    if (i > n_blocks - 1)
      return;
    ValueType min_zc = 1e-12;
    ValueType xn[3];
    bool flag;
    double sum;

    flag = false;
    xn[2] = 1;
    sum = 0;
    for (int j = 1; j < 3; j++)
    {
      xn[j - 1] = gpu_x[i*3 + j] - gpu_dx[i*3 + j];
      xn[2] -= xn[j - 1];
      if (xn[j - 1] < 0)
      {
        xn[j - 1] = min_zc;
        flag = true;
      }
      sum += xn[j - 1];
    }

    if (xn[2] < 0)
    {
      xn[2] = min_zc;
      flag = true;
    }
    sum += xn[2];

    if (flag)
    {
      xn[0] = xn[0] / sum;
      xn[1] = xn[1] / sum;
    }
    gpu_x[i * 3] = gpu_x[i * 3] - gpu_dx[i * 3];
    gpu_x[i * 3 + 1] = xn[0];
    gpu_x[i * 3 + 2] = xn[1];
  }

  template<typename IndexType, typename ValueType, int _BLOCK_SIZE>
  __global__ void
    correct_solution_and_calc_ratios (const int n_blocks, ValueType *gpu_x, ValueType *gpu_dx, float *gpu_ratio)
  {
    const unsigned i = blockIdx.x * _BLOCK_SIZE + threadIdx.x; // global thread inde
    if (i > n_blocks - 1)
      return;
    ValueType min_zc = 1e-12;
    ValueType xn[3];
    bool flag;
    double sum;

    flag = false;
    xn[2] = 1;
    sum = 0;
    for (int j = 1; j < 3; j++)
    {
      xn[j - 1] = gpu_x[i * 3 + j] - gpu_dx[i * 3 + j];
      xn[2] -= xn[j - 1];
      if (xn[j - 1] < 0)
      {
        xn[j - 1] = min_zc;
        flag = true;
      }
      sum += xn[j - 1];
    }

    if (xn[2] < 0)
    {
      xn[2] = min_zc;
      flag = true;
    }
    sum += xn[2];

    if (flag)
    {
      xn[0] = xn[0] / sum;
      xn[1] = xn[1] / sum;

      gpu_dx[i * 3 + 1] = gpu_x[i * 3 + 1] - xn[0];
      gpu_dx[i * 3 + 2] = gpu_x[i * 3 + 2] - xn[1];
    }
    

    for (int j = 0; j < 3; j++)
    {
      if (fabs(gpu_x[i * 3 + j]) > 1e-4)
          gpu_ratio[i * 3 + j] = fabs (gpu_dx[i * 3 + j] / gpu_x[i * 3 + j]);
      else
          gpu_ratio[3 * i + j] = 0;
    }
    
  }


  // One block per cell (per jacobian row)
  template<typename IndexType, typename ValueType, typename InterpValueType,
    int _BLOCK_SIZE>
    __global__ void
    jacobian_nc3_assembly_kernel2 (const int n_blocks,
    const ValueType dt, const IndexType *rows_ptr,
    const IndexType *cols_ind, ValueType *Jac,
    ValueType *RHS, const ValueType *X,
    const ValueType *tran, const ValueType *PV,
    const InterpValueType *acc1,
    const InterpValueType *acc2,
    const InterpValueType *acc3,
    const InterpValueType *flu1,
    const InterpValueType *flu2,
    const InterpValueType *flu3,
    const InterpValueType *acc_n)
  {
    const unsigned i = blockIdx.x * _BLOCK_SIZE; // global thread index
    const unsigned j = threadIdx.x;              // local thread index
    if (i > n_blocks - 1)
      return;

    //IndexType start = rows_ptr[i];
    //IndexType end = rows_ptr[i + 1];
    IndexType jac_diag_idx;
    ValueType p_diff, gamma_p_diff, jtran_dt;

    //__shared__ ValueType wkspc[BLOCK_SIZE * 6];
    //ValueType *jac_d = wkspc + 6 * threadIdx.x;
    //ValueType *rhs = wkspc + 6 * threadIdx.x + 4;
    ValueType jac_d[9];
    ValueType rhs[3];

    jac_d[0] = 0;
    jac_d[1] = 0;
    jac_d[2] = 0;
    jac_d[3] = 0;
    jac_d[4] = 0;
    jac_d[5] = 0;
    jac_d[6] = 0;
    jac_d[7] = 0;
    jac_d[8] = 0;

    rhs[0] = 0;
    rhs[1] = 0;
    rhs[2] = 0;

    for (IndexType j = rows_ptr[i]; j < rows_ptr[i + 1]; j++)
    {
      unsigned cl = cols_ind[j];

      if (i == cl)
      {
        // fill diagonal part
        rhs[0] += PV[i] * (acc1[4 * i] - acc_n[4 * i]);
        rhs[1] += PV[i] * (acc2[4 * i] - acc_n[4 * i + 1]);
        rhs[2] += PV[i] * (acc3[4 * i] - acc_n[4 * i + 2]);

        jac_diag_idx = j;
        jac_d[0] += PV[i] * acc1[4 * i + 1];
        jac_d[1] += PV[i] * acc1[4 * i + 2];
        jac_d[2] += PV[i] * acc1[4 * i + 3];
        jac_d[3] += PV[i] * acc2[4 * i + 1];
        jac_d[4] += PV[i] * acc2[4 * i + 2];
        jac_d[5] += PV[i] * acc2[4 * i + 3];
        jac_d[6] += PV[i] * acc3[4 * i + 1];
        jac_d[7] += PV[i] * acc3[4 * i + 2];
        jac_d[8] += PV[i] * acc3[4 * i + 3];
      }
      else
      {
        // fill offdiagonal part + contribute to diagonal

        p_diff = X[3 * cl] - X[3 * i];

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
          rhs[2] -= gamma_p_diff * flu3[4 * i];

          jac_d[0] -= gamma_p_diff * flu1[4 * i + 1];
          jac_d[1] -= gamma_p_diff * flu1[4 * i + 2];
          jac_d[2] -= gamma_p_diff * flu1[4 * i + 3];
          jac_d[3] -= gamma_p_diff * flu2[4 * i + 1];
          jac_d[4] -= gamma_p_diff * flu2[4 * i + 2];
          jac_d[5] -= gamma_p_diff * flu2[4 * i + 3];
          jac_d[6] -= gamma_p_diff * flu3[4 * i + 1];
          jac_d[7] -= gamma_p_diff * flu3[4 * i + 2];
          jac_d[8] -= gamma_p_diff * flu3[4 * i + 3];

          Jac[9 * j + 0] = -jtran_dt * flu1[4 * i];
          jac_d[0] -= -jtran_dt * flu1[4 * i];

          Jac[9 * j + 3] = -jtran_dt * flu2[4 * i];
          jac_d[3] -= -jtran_dt * flu2[4 * i];

          Jac[9 * j + 6] = -jtran_dt * flu3[4 * i];
          jac_d[6] -= -jtran_dt * flu3[4 * i];

          Jac[9 * j + 1] = 0;
          Jac[9 * j + 2] = 0;
          Jac[9 * j + 4] = 0;
          Jac[9 * j + 5] = 0;
          Jac[9 * j + 7] = 0;
          Jac[9 * j + 8] = 0;
        }
        else
        {
          rhs[0] -= gamma_p_diff * flu1[4 * cl];
          rhs[1] -= gamma_p_diff * flu2[4 * cl];
          rhs[2] -= gamma_p_diff * flu3[4 * cl];

          jac_d[0] += jtran_dt * flu1[4 * cl];
          jac_d[3] += jtran_dt * flu2[4 * cl];
          jac_d[6] += jtran_dt * flu3[4 * cl];

          Jac[9 * j + 0] = -gamma_p_diff * flu1[4 * cl + 1] - jtran_dt * flu1[4 * cl];
          Jac[9 * j + 1] = -gamma_p_diff * flu1[4 * cl + 2];
          Jac[9 * j + 2] = -gamma_p_diff * flu1[4 * cl + 3];
          Jac[9 * j + 3] = -gamma_p_diff * flu2[4 * cl + 1] - jtran_dt * flu2[4 * cl];
          Jac[9 * j + 4] = -gamma_p_diff * flu2[4 * cl + 2];
          Jac[9 * j + 5] = -gamma_p_diff * flu2[4 * cl + 3];
          Jac[9 * j + 6] = -gamma_p_diff * flu3[4 * cl + 1] - jtran_dt * flu3[4 * cl];
          Jac[9 * j + 7] = -gamma_p_diff * flu3[4 * cl + 2];
          Jac[9 * j + 8] = -gamma_p_diff * flu3[4 * cl + 3];
        }
      }
    }

    RHS[3 * i + 0] = rhs[0];
    RHS[3 * i + 1] = rhs[1];
    RHS[3 * i + 2] = rhs[2];

    Jac[9 * jac_diag_idx + 0] = jac_d[0];
    Jac[9 * jac_diag_idx + 1] = jac_d[1];
    Jac[9 * jac_diag_idx + 2] = jac_d[2];
    Jac[9 * jac_diag_idx + 3] = jac_d[3];
    Jac[9 * jac_diag_idx + 4] = jac_d[4];
    Jac[9 * jac_diag_idx + 5] = jac_d[5];
    Jac[9 * jac_diag_idx + 6] = jac_d[6];
    Jac[9 * jac_diag_idx + 7] = jac_d[7];
    Jac[9 * jac_diag_idx + 8] = jac_d[8];
  }
