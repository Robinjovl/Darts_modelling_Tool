#include "py_globals.h"
#include <pybind11/numpy.h>
namespace py = pybind11;

void pybind_pm_discretizer(py::module &);
void pybind_mesh_conn(py::module &);
void pybind_globals(py::module &);
void pybind_gpu_tools(py::module &);
void pybind_engine_base(py::module &);
void pybind_engine_nc_nl_cpu(py::module &);
void pybind_engine_elasticity_cpu(py::module &);
void pybind_engine_pm_cpu(py::module &);
void pybind_mech_operators(py::module &);
void pybind_contact(py::module &);
void pybind_engine_nce_g_cpu(py::module &);
void pybind_engine_nc_cg_cpu(py::module &);
void pybind_engine_nc_cg_gpu(py::module &);
void pybind_engine_super_cpu(py::module &);
void pybind_engine_super_mp_cpu(py::module &);
void pybind_engine_super_elastic_cpu(py::module &);

void pybind_engine_super_gpu(py::module &);
void pybind_well_controls(py::module &);
void pybind_ms_well(py::module &);
void pybind_evaluator_iface(py::module &);
void pybind_operator_set_from_files(py::module &);

void pybind_operator_set_interpolator_rates(py::module &);
void pybind_operator_set_interpolator_super(py::module &);
void pybind_operator_set_interpolator_pze_gra(py::module &);
void pybind_operator_set_interpolator_pz_cap_gra(py::module &);
void pybind_operator_set_interpolator_super_elastic(py::module &);

class ms_well;
class operator_set_gradient_evaluator_iface;

template <typename T>
py::array to_numpy(std::vector<T>& vec)
{
  // Create a Python array object that wraps the data in the vector
  return py::array(
    py::buffer_info(
      vec.data(),   // Pointer to the data (const_cast is needed because NumPy allows mutation)
      sizeof(T),                    // Size of one scalar
      py::format_descriptor<T>::format(),  // NumPy data type format descriptor
      1,                            // Number of dimensions
      { vec.size() },               // Shape of the array (vector size in this case)
      { sizeof(T) }                 // Strides (number of bytes to skip to the next element)
    ),
    py::cast(&vec)
    );
}

PYBIND11_MODULE(engines, m)
{
  m.doc() = "Delft Advanced Research Terra Simulator";
  //auto m1 = m.def_submodule("engines", "Collection of DARTS simulators based on OBL approach");
  py::bind_vector<std::vector<index_t>>(m, "index_vector", py::module_local(true), py::buffer_protocol())
      .def(py::pickle(
          [](const std::vector<index_t>& p) { // __getstate__
              py::tuple t(p.size());
              for (int i = 0; i < p.size(); i++)
                  t[i] = p[i];

              return t;
          },
          [](py::tuple t) { // __setstate__
              std::vector<index_t> p(t.size());

              for (int i = 0; i < p.size(); i++)
                  p[i] = t[i].cast<index_t>();

              //p.setExtra(t[1].cast<int>());

              return p;
          })) \
      .def("resize",
          (void (std::vector<index_t>::*) (size_t count)) & std::vector<index_t>::resize,
          "changes the number of elements stored") \
      .def("to_numpy", [](std::vector<index_t>& vec) {
            return to_numpy(vec);  // Call the conversion function
      }, "Converts the vector to a NumPy array");
  py::bind_vector<std::vector<value_t>>(m, "value_vector", py::module_local(true), py::buffer_protocol())
      .def(py::pickle(
          [](const std::vector<value_t> &p) { // __getstate__
            py::tuple t(p.size());
            for (int i = 0; i < p.size(); i++)
              t[i] = p[i];

            return t;
          },
          [](py::tuple t) { // __setstate__
            std::vector<value_t> p(t.size());

            for (int i = 0; i < p.size(); i++)
              p[i] = t[i].cast<value_t>();

            //p.setExtra(t[1].cast<int>());

            return p;
          })) \
      .def("resize",
          (void (std::vector<value_t>::*) (size_t count)) &std::vector<value_t>::resize,
          "changes the number of elements stored") \
      .def("to_numpy", [](std::vector<value_t>& vec) {
          return to_numpy(vec);  // Call the conversion function
      }, "Converts the vector to a NumPy array");
  py::bind_vector<std::vector<ms_well *>>(m, "ms_well_vector");
  py::bind_vector<std::vector<operator_set_gradient_evaluator_iface *>>(m, "op_vector");
  py::bind_map<std::map<std::string, timer_node>>(m, "timer_map");

  pybind_pm_discretizer(m);
  pybind_mesh_conn(m);
  pybind_globals(m);
  pybind_engine_base(m);
  pybind_engine_nc_nl_cpu(m);
  pybind_engine_elasticity_cpu(m);
  pybind_engine_pm_cpu(m);
  pybind_mech_operators(m);
  pybind_contact(m);
  pybind_engine_nce_g_cpu(m);
  pybind_engine_super_cpu(m);
  pybind_engine_super_mp_cpu(m);
  pybind_engine_super_elastic_cpu(m);

  pybind_well_controls(m);
  pybind_ms_well(m);
  pybind_evaluator_iface(m);

  pybind_operator_set_interpolator_rates(m);
  pybind_operator_set_interpolator_super(m);
  pybind_operator_set_interpolator_pze_gra(m);
  pybind_operator_set_interpolator_pz_cap_gra(m);

#ifdef WITH_GPU
  pybind_engine_nc_cg_gpu(m);
  pybind_engine_super_gpu(m);
  pybind_gpu_tools(m);
#endif
}
