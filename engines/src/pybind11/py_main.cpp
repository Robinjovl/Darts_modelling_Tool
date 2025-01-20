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
	  
  
  // Logging related bindings
  py::module_ m_logging = m.def_submodule(
    "logging", 
    "A submodule for logging related functionalities."
  );
  py::enum_<logging::LoggingLevel>(m_logging, "LoggingLevel")
    .value("DEBUG", logging::LoggingLevel::DEBUG)
    .value("INFO", logging::LoggingLevel::INFO)
    .value("WARNING", logging::LoggingLevel::WARNING)
    .value("ERROR", logging::LoggingLevel::ERROR)
    .value("CRITICAL", logging::LoggingLevel::CRITICAL)
    .export_values();
  m_logging.def(
    "log", 
    py::overload_cast<const std::string&, logging::LoggingLevel>(
      &logging::log
    ),  
    "Adds a message to logs.",
    py::arg("message"),
    py::arg("level") = logging::LoggingLevel::INFO
  );
  m_logging.def(
    "set_logging_level",
    &logging::set_logging_level,
    "Sets logging verbosity level.",
    py::arg("level")
  );
  m_logging.def(
    "duplicate_output_to_file", 
    &logging::duplicate_output_to_file, 
    "Duplicates outputs to a file.",
    py::arg("file_path"));
  m_logging.def("flush", &logging::flush, "Flushes output streams.");
  m_logging.def(
    "debug", 
    &logging::debug, 
    "Detailed information, typically only of interest to a developer " 
    "trying to diagnose a problem.",
    py::arg("message")
  );
  m_logging.def(
    "info", 
    &logging::info, 
    "Confirmation that things are working as expected.",
    py::arg("message")
  );
  m_logging.def(
    "warning", 
    &logging::warning, 
    "An indication that something unexpected happened, or that a problem might "
    "occur in the near future (e.g. ‘disk space low’). The software is still "
    "working as expected.",
    py::arg("message")
  );
  m_logging.def(
    "error", 
    &logging::error, 
    "Due to a more serious problem, the software has not been able to perform "
    "some function.",
    py::arg("message")
  );
  m_logging.def(
    "critical", 
    &logging::critical, 
    "A serious error, indicating that the program itself may be unable to "
    "continue running.",
    py::arg("message")
  );
  // end pybind logging

	  
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
