//--------------------------------------------------------------------------
#ifndef OPENDARTS_AUXILIARY_TIMER_NODE_H
#define OPENDARTS_AUXILIARY_TIMER_NODE_H
//--------------------------------------------------------------------------

#include <map>
#include <string>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef WITH_GPU
#include <cuda_runtime.h>
#endif

namespace opendarts
{
  namespace auxiliary
  {
    class timer_node
    {
    public:
      timer_node();
      timer_node(const opendarts::auxiliary::timer_node &a);
      void start();
      void stop();
      double get_timer();

#ifdef WITH_GPU
      void start_gpu(cudaStream_t stream = 0);
      void stop_gpu(cudaStream_t stream = 0);
      double get_timer_gpu();
      bool node_name_ends_with_gpu(std::string const &node_name);
#endif // WITH_GPU

      void reset_recursive();
      std::string print(std::string offset, std::string &result);

      double timer;
      // This is added to allow for sub-timers in the timer and so forth
      std::map<std::string, opendarts::auxiliary::timer_node> node;

#ifdef WITH_GPU
      bool is_gpu_timer = false;
      cudaEvent_t event_start, event_stop;
#endif // WITH_GPU
    };

  } // namespace auxiliary
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_AUXILIARY_TIMER_NODE_H
//--------------------------------------------------------------------------
