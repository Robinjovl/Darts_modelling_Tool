#include "evaluator_iface.h"
#include "thrust/device_vector.h"
//#include "thrust/host_ptr.h"

#ifdef WITH_GPU
int operator_set_gradient_evaluator_cpu::evaluate_with_derivatives_d(int n_states_idxs, double *state_d, int *states_idxs_d,
                                                                     double *values_d, double *derivatives_d)
{
  //TODO: to determine sizes of arrays to get them copied without changing the arguments,
  //         we need to introduce get_n_dims() and get_n_ops() methods on this level
  printf("Not yet implemented!\n");
  return 0;
}

int operator_set_gradient_evaluator_cpu::evaluate_d(double *state_d, double *values_d)
{
  //TODO: to determine sizes of arrays to get them copied without changing the arguments,
  //         we need to introduce get_n_dims() and get_n_ops() methods on this level
  printf("Not yet implemented!\n");
  return 0;
}

int operator_set_gradient_evaluator_gpu::evaluate(const std::vector<double> &state, std::vector<double> &values)
{
  // Send input data to device
  timer->start();
  timer->node["copy to device"].start();
  thrust::device_vector<double> state_d(state);

  // Allocate device output data
  thrust::device_vector<double> values_d;
  values_d.resize(values.size());

  timer->node["copy to device"].stop();
  timer->stop();
  int ret = evaluate_d(thrust::raw_pointer_cast(state_d.data()),
                       thrust::raw_pointer_cast(values_d.data()));
  timer->start();
  timer->node["copy to host"].start();

  // Send output data to host
  thrust::copy(values_d.begin(), values_d.end(), values.begin());

  timer->node["copy to host"].stop();
  timer->stop();
  return ret;
}

int operator_set_gradient_evaluator_gpu::evaluate_with_derivatives(const std::vector<double> &states,
                                                                   const std::vector<int> &states_idxs,
                                                                   std::vector<double> &values,
                                                                   std::vector<double> &derivatives)
{
  // Send input data to device
  timer->start();
  timer->node["copy to device"].start();
  thrust::device_vector<double> states_d(states);
  thrust::device_vector<int> states_idxs_d(states_idxs);

  // Allocate device output data
  thrust::device_vector<double> values_d;
  thrust::device_vector<double> derivatives_d;
  values_d.resize(values.size());
  derivatives_d.resize(derivatives.size());

  timer->node["copy to device"].stop();
  timer->stop();
  int ret = evaluate_with_derivatives_d(states_idxs.size(), thrust::raw_pointer_cast(states_d.data()),
                                        thrust::raw_pointer_cast(states_idxs_d.data()),
                                        thrust::raw_pointer_cast(values_d.data()),
                                        thrust::raw_pointer_cast(derivatives_d.data()));
  timer->start();
  timer->node["copy to host"].start();

  // Send output data to host
  thrust::copy(values_d.begin(), values_d.end(), values.begin());
  thrust::copy(derivatives_d.begin(), derivatives_d.end(), derivatives.begin());

  timer->node["copy to host"].stop();
  timer->stop();
  return ret;
}

#endif
