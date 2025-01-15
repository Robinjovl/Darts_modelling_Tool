#ifdef PYBIND11_ENABLED
#include "py_logging.hpp"
#include "logging.h"
#include <pybind11/stl.h>

namespace py = pybind11;

// Logging related bindings
using namespace logging;

void pybind_logging(py::module &m) {

  py::module_ m_logging = m.def_submodule(
      "logging", "A submodule for logging related functionalities.");

  py::enum_<LoggingLevel>(m_logging, "LoggingLevel")
      .value("DEBUG", LoggingLevel::DEBUG)
      .value("INFO", LoggingLevel::INFO)
      .value("WARNING", LoggingLevel::WARNING)
      .value("ERROR", LoggingLevel::ERROR)
      .value("CRITICAL", LoggingLevel::CRITICAL)
      .export_values();

  m_logging.def("log", static_cast<void (*)(const std::string &)>(&log),
                "Adds a message to logs.");
  m_logging.def("set_verbosity", &set_verbosity,
                "Sets logging verbosity level.", py::arg("level"));

  m_logging.def("flush", &flush, "Flushes output streams.");

  m_logging.def("set_file", &set_file, "Sets a file to write the logs to.");

  m_logging.def("enable_screen_output", &enable_screen_output, "Enables or disables screen output.", py::arg("enabled")); 

#define PYBIND_LOG(level, name, description)                                   \
  m_logging.def(#name, &name, description, py::arg("message"));

  LEVELS(PYBIND_LOG)

  py::class_<Logger>(m_logging, "Logger");
}

#endif
