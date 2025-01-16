#include <optional>
#include <pybind11/detail/common.h>
#ifdef PYBIND11_ENABLED
#include "logging.h"
#include "py_logging.hpp"
#include <pybind11/stl.h>

namespace py = pybind11;

// Logging related bindings
using namespace logging;
using namespace std;

void pybind_logging(py::module &m) {

  py::module_ m_logging = m.def_submodule(
      "logging", "A submodule for logging related functionalities.");

#define PYBIND_LEVEL(level, name, description)                                 \
  .value(#level, level, description)

  py::enum_<LoggingLevel>(m_logging, "LoggingLevel") LEVELS(PYBIND_LEVEL)
      .export_values();

  m_logging.def("log", static_cast<void (*)(const std::string &)>(&log),
                "Adds a message to the logs.");
  m_logging.def("set_verbosity", &set_verbosity,
                "Sets logging verbosity level.", py::arg("level"));

  m_logging.def("flush", &logging::flush, "Flushes the output streams.");

  m_logging.def("set_file", &set_file, "Sets a file to write the logs to.");

  m_logging.def("enable_screen_output", &enable_screen_output,
                "Enables or disables screen output.", py::arg("enabled"));

  m_logging.def("get_logger", &get_logger, "Creates a new logger.",
                py::arg("name"), py::arg("file") = std::nullopt,
                py::arg("screen") = true,
                py::return_value_policy::reference);

#define PYBIND_LOG(level, name, description)                                   \
  m_logging.def(#name, &name, description, py::arg("message"));

  LEVELS(PYBIND_LOG)

#define PYBIND_CLASS_LOG(level, name, description)                             \
  .def(#name, &Logger::name, description, py::arg("message"))

  py::class_<Logger>(m_logging, "Logger") LEVELS(PYBIND_CLASS_LOG)
      .def(
          "log",
          [](Logger &self, const std::string &message) { self.log(message); },
          "Logs a message.", py::arg("message"))
      .def("set_file", &Logger::set_file,
           "Sets a file destination for the log messages.", py::arg("file"))
      .def("enable_screen_output", &Logger::enable_screen_output,
           "Enables or disables the logging of messages to the screen.",
           py::arg("enabled"))
      .def("flush", &Logger::flush, "Flushes this logger and its children.")
      .def("set_verbosity", &Logger::set_verbosity,
           "Sets the verbosity level of the logger.", py::arg("level"))
      .def("get_logger",
           py::overload_cast<const string &>(&Logger::get_logger),
           py::arg("name"),
           py::return_value_policy::reference)
    .def("get_visual_repr", &Logger::get_visual_repr, "Returns a tree view of this logger and its children.", py::arg("prefix") = "")
      .def("print_loggers", &Logger::print_loggers, "Prints the tree view of this logger and its children.");

  m_logging.def("get_visual_repr", &get_visual_repr, "Returns a tree view of the loggers' hierarchy.", py::arg("prefix") = "");
m_logging.def("print_loggers", &print_loggers, "Prints the tree view of all loggers and their children.", py::arg("prefix") = "");
}

#endif
