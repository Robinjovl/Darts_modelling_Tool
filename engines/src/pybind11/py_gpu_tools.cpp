#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

namespace py = pybind11;
#include "gpu_tools.h"

void pybind_gpu_tools(py::module &m)
{
    m.def("copy_data_to_device", [](std::vector<double>& host_data, py::capsule ptr)
    {
        if (!std::strcmp(ptr.name(), "double_ptr")) {
            double* device_ptr = static_cast<double*>(ptr.get_pointer());
            copy_data_to_device<double>(host_data, device_ptr);
        } else {
            throw std::runtime_error("Invalid capsule name!");
        }
    });
    m.def("copy_data_to_host", [](std::vector<double>& host_data, py::capsule ptr) 
    {
        if (!std::strcmp(ptr.name(), "double_ptr")) {
            double* device_ptr = static_cast<double*>(ptr.get_pointer());
            copy_data_to_host<double>(host_data, device_ptr);
        } else {
            throw std::runtime_error("Invalid capsule name!");
        }
    });
}

#endif //PYBIND11_ENABLED