#include <map>
#include <stdexcept>
#include <string>
#include <time.h>

#include "openDARTS/auxiliary/timer_node.hpp"

#ifdef _OPENMP
#include <omp.h>
#define clock_func omp_get_wtime
#define clock_norm 1
#else
#define clock_func clock
#define clock_norm CLOCKS_PER_SEC
#endif

#ifdef WITH_GPU
#include <cuda_runtime.h>
#define clock_sync cudaDeviceSynchronize()
#else
#define clock_sync
#endif

#define TIMER_START(timer)                                                                                             \
  clock_sync;                                                                                                          \
  timer -= clock_func();

#define TIMER_STOP(timer)                                                                                              \
  clock_sync;                                                                                                          \
  timer += clock_func();

#define TIMER_GET(timer, result)                                                                                       \
  result = timer;                                                                                                      \
  if (result < 0)                                                                                                      \
  {                                                                                                                    \
    TIMER_STOP(result)                                                                                                 \
  }                                                                                                                    \
  result /= clock_norm;

namespace opendarts
{
  namespace auxiliary
  {
    timer_node::timer_node() : timer(0)
    {
#ifdef WITH_GPU
      cudaEventCreate(&(this->event_start));
      cudaEventCreate(&(this->event_stop));
#endif // WITH_GPU
      ;
    }

    timer_node::timer_node(const opendarts::auxiliary::timer_node &a) : timer(a.timer)
    {
#ifdef WITH_GPU
      cudaEventCreate(&(this->event_start));
      cudaEventCreate(&(this->event_stop));
#endif // WITH_GPU
      ;
    }

    void timer_node::start() { TIMER_START(this->timer) }

    void timer_node::stop(){TIMER_STOP(this->timer)};

    double timer_node::get_timer()
    {
#ifdef WITH_GPU
      if (this->is_gpu_timer)
        return get_timer_gpu();
#endif // WITH_GPU

      double local_timer;
      TIMER_GET(this->timer, local_timer)
      return local_timer;
    }

#ifdef WITH_GPU

    void timer_node::start_gpu(cudaStream_t stream = 0)
    {
      this->is_gpu_timer = true;
      cudaEventRecord(this->event_start, stream);
    }

    void timer_node::stop_gpu(cudaStream_t stream = 0)
    {
      float local_timer;

      cudaEventRecord(this->event_stop, stream);
      cudaEventSynchronize(this->event_stop);
      cudaEventElapsedTime(&local_timer, this->event_start, this->event_stop);
      this->timer += local_timer;
    }

    double timer_node::get_timer_gpu() { return this->timer / 1000; };

    bool timer_node::node_name_ends_with_gpu(std::string const &node_name)
    {
      std::string const &ending = "_gpu";
      if (node_name.length() >= ending.length())
      {
        return (0 == node_name.compare(node_name.length() - ending.length(), ending.length(), ending));
      }
      else
      {
        return false;
      }
    }

#endif // WITH_GPU

    void timer_node::reset_recursive()
    {
      this->timer = 0;
      for (auto &n : this->node)
      {
        n.second.reset_recursive();
      }
    }

    std::string timer_node::print(std::string offset, std::string &result)
    {
      if (offset == "")
      {
        result += "Total elapsed " + std::to_string(this->get_timer()) + " sec\n";
        offset = "\t";
      }

      for (auto &n : this->node)
      {
        result += offset + n.first + " " + std::to_string(n.second.get_timer()) + " sec\n";
        n.second.print(offset + '\t', result);
      }
      return result;
    }
  } // namespace auxiliary
} // namespace opendarts
