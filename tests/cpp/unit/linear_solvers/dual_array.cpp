// Unit test for dual_array<T> -- the host/device storage primitive of the
// unified matrix layout (SOLVER_REFACTORING_PLAN.md section 12.5).
//
// The host-side behaviour is exercised in every build. The device-side
// behaviour is additionally exercised under WITH_GPU.

#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#include "dual_array.hpp"

using opendarts::linear_solvers::dual_array;

namespace
{
  int report(const char *name, bool ok)
  {
    if (!ok)
      std::cout << "dual_array: " << name << " FAILED" << std::endl;
    return ok ? 0 : 1;
  }

  // Default construction yields an empty array.
  int test_default_construct()
  {
    dual_array<double> a;
    return report("default construct empty", a.empty() && a.size() == 0);
  }

  // Sized construction allocates a value-initialised host buffer.
  int test_sized_construct()
  {
    const std::size_t n = 7;
    dual_array<double> a(n);
    bool ok = (a.size() == n) && !a.empty() && (a.host_data() != nullptr);
    const double *h = a.host_data();
    for (std::size_t i = 0; i < n && ok; ++i)
      ok = (h[i] == 0.0);
    return report("sized construct value-initialised", ok);
  }

  // resize preserves existing content and grows with value-initialised tail.
  int test_resize()
  {
    dual_array<int> a(3);
    int *h = a.host_data();
    h[0] = 10;
    h[1] = 11;
    h[2] = 12;
    a.resize(5);
    const int *r = a.host_data();
    bool ok = (a.size() == 5) && r[0] == 10 && r[1] == 11 && r[2] == 12 && r[3] == 0 && r[4] == 0;
    return report("resize preserves content", ok);
  }

  // Host writes are read back through the const accessor.
  int test_host_round_trip()
  {
    const std::size_t n = 16;
    dual_array<double> a(n);
    double *w = a.host_data();
    for (std::size_t i = 0; i < n; ++i)
      w[i] = 2.0 * static_cast<double>(i) + 1.0;
    const dual_array<double> &ca = a;
    const double *r = ca.host_data();
    bool ok = true;
    for (std::size_t i = 0; i < n && ok; ++i)
      ok = (r[i] == 2.0 * static_cast<double>(i) + 1.0);
    return report("host write/read round trip", ok);
  }

  // clone is an independent deep copy.
  int test_clone()
  {
    dual_array<int> a(4);
    int *h = a.host_data();
    std::iota(h, h + 4, 100);
    dual_array<int> b = a.clone();
    a.host_data()[0] = -1; // mutate the original
    const int *rb = b.host_data();
    bool ok = (b.size() == 4) && rb[0] == 100 && rb[1] == 101 && rb[2] == 102 && rb[3] == 103;
    return report("clone is independent", ok);
  }

  // Move construction transfers ownership and empties the source.
  int test_move()
  {
    dual_array<int> a(3);
    a.host_data()[2] = 42;
    dual_array<int> b(std::move(a));
    bool ok = (b.size() == 3) && (b.host_data()[2] == 42) && a.empty();

    dual_array<int> c;
    c = std::move(b);
    ok = ok && (c.size() == 3) && (c.host_data()[2] == 42) && b.empty();
    return report("move transfers ownership", ok);
  }

#ifdef WITH_GPU
  // Device buffer is allocated lazily and a host->device->host round trip
  // through cudaMemcpy preserves the data.
  int test_device_round_trip()
  {
    const std::size_t n = 32;
    dual_array<double> a(n);
    double *w = a.host_data();
    for (std::size_t i = 0; i < n; ++i)
      w[i] = 3.0 * static_cast<double>(i) - 5.0;

    bool ok = !a.device_allocated();
    a.sync_to_device(); // host -> device, allocates the device buffer
    ok = ok && a.device_allocated();

    // Touch the device side, then pull it back: the device holds the pattern
    // copied above, so the host buffer must come back unchanged.
    (void)a.device_data();
    a.sync_to_host(); // device -> host
    const double *r = a.host_data();
    for (std::size_t i = 0; i < n && ok; ++i)
      ok = (r[i] == 3.0 * static_cast<double>(i) - 5.0);

    return report("device host<->device round trip", ok);
  }
#endif // WITH_GPU
} // namespace

int main()
{
  int error_output = 0;

  error_output += test_default_construct();
  error_output += test_sized_construct();
  error_output += test_resize();
  error_output += test_host_round_trip();
  error_output += test_clone();
  error_output += test_move();
#ifdef WITH_GPU
  error_output += test_device_round_trip();
#endif

  if (error_output == 0)
    std::cout << "dual_array: all tests passed" << std::endl;

  return error_output;
}
