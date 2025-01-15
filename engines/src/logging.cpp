#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>

#include "logging.h"

using namespace std;

namespace logging {
Logger Logger::s_root_logger;

shared_ptr<Logger> get_logger(const string &name) {
  return Logger::s_root_logger.get_logger(name);
}

shared_ptr<Logger> get_logger(const string &name, const string &filename) {
  return Logger::s_root_logger.get_logger(name, filename);
}

shared_ptr<Logger> get_logger(const string &name, const string &filename,
                              bool stdout) {
  return Logger::s_root_logger.get_logger(name, filename, stdout);
}

shared_ptr<Logger> logger = logging::get_logger("logging");

/** Basic wrapper around c++ std::cout object to expose it to python. */
void log(const string &msg) { Logger::s_root_logger.log(msg); }

template <LoggingLevel level> void log(const string &message) {
  Logger::s_root_logger.log<level>(message);
}

void log(const string &msg, LoggingLevel level) {
  Logger::s_root_logger.log(msg, level);
}

void set_verbosity(LoggingLevel level) {
  Logger::s_root_logger.set_verbosity(level);
}

void set_file(const string &filename) {
  Logger::s_root_logger.set_file(filename);
}
void flush() { Logger::s_root_logger.flush(); }

Logger::Logger() { m_name = "root"; }
Logger::Logger(const Logger &other)
    : m_level(other.m_level), m_stdout(other.m_stdout) {
  set_file(other.m_file);
}

shared_ptr<Logger> Logger::get_logger(const string &name) {
  auto child_logger = make_shared<Logger>(*this);
  child_logger->m_name = name;
  child_logger->m_parent_logger = this;
  logging::logger->debug("Creating child logger '" + name + "' from '" +
                         m_name + "'.");
  m_child_loggers.push_back(child_logger);
  return child_logger;
}

shared_ptr<Logger> Logger::get_logger(const string &name,
                                      const string &filename) {
  auto child_logger = get_logger(name);
  child_logger->set_file(filename);

  return child_logger;
}

shared_ptr<Logger> Logger::get_logger(const string &name,
                                      const string &filename, bool stdout) {
  auto child_logger = get_logger(name, filename);
  child_logger->m_stdout = stdout;
  return child_logger;
}

void Logger::set_verbosity(LoggingLevel level) {
  m_level = level;
  for (auto &child : m_child_loggers) {
    if (child.expired()) {
      continue;
    }
    child.lock()->set_verbosity(level);
  }
}

void Logger::log(const string &message) {
  if (m_fstream) {
    std::lock_guard<std::mutex> lock(m_fstream->mutex);
    m_fstream->stream << message << "\n";
  }

  if (m_stdout) {
    cout << message + "\n";
  }
}

template <LoggingLevel level> void Logger::log(const string &message) {
  if (level < m_level) {
    return;
  }

  log(message);
}

void Logger::log(const string &message, const LoggingLevel &level) {
  if (level < m_level) {
    return;
  }
  log(message);
}

void Logger::flush() {
  logger->debug("Flushing " + m_name);
  if (m_fstream) {
    std::lock_guard<std::mutex> lock(m_fstream->mutex);
    m_fstream->stream.flush();
  }

  if (m_stdout) {
    cout.flush();
  }

  for (auto &child : m_child_loggers) {
    if (child.expired()) {
      continue;
    }

    child.lock()->flush();
  }
}

void Logger::set_file(const optional<string> &file) {
  for (auto &child : m_child_loggers) {
    if (child.expired()) {
      continue;
    }
    child.lock()->set_file(file);
  }
  // Lock file streams
  std::lock_guard<std::mutex> lock(s_file_streams_mutex);

  // Early return if same file
  if (file == m_file)
    return;

  m_file = file;

  if (!m_file.has_value()) {
    // Reset shared_ptr for the file stream
    m_fstream.reset();
    return;
  }

  // Access the weak pointer associated with the file
  auto &fstream = s_file_streams[m_file.value()];

  if (fstream.expired()) {
    // Create a new shared_ptr for the file stream and
    // associate it with this logger
    m_fstream = std::make_shared<FStreamWithMutex>(m_file.value());
    fstream = m_fstream;

    // Handle file errors
    if (!m_fstream->stream.is_open()) {
      logger->error("Failed to open file: " + m_file.value());
      m_fstream.reset();
      m_file = std::nullopt;
    } else {
      logger->debug("New file stream created for file: " + m_file.value());
    }

  } else {
    m_fstream = fstream.lock();
  }
}

void Logger::enable_screen_output(bool display_on_screen) {
  m_stdout = display_on_screen;
  for (auto &child : m_child_loggers) {
    if (child.expired()) {
      continue;
    }
    child.lock()->enable_screen_output(display_on_screen);
  }
}

std::unordered_map<string, std::weak_ptr<FStreamWithMutex>>
    Logger::s_file_streams;
std::mutex Logger::s_file_streams_mutex;
/*Logger::get_root_logger() {*/
/*  static Logger Logger::s_root_logger;*/
/*  s_root_logger*/
/**/
/*}*/

} // namespace logging

string read_file(const string &filename) {
  ifstream ifs(filename);
  return string(std::istreambuf_iterator<char>{ifs}, {});
}

template <typename T>
bool assert_equal(T actual, T expected, const string &test_name) {
  if (expected != actual) {
    std::cerr << "Test failed: " << test_name << "\n\nExpected:\n"
              << expected << "\nActual:\n"
              << actual << "\n";
    return false;
  } else {
    std::cout << "Test passed: " << test_name << "\n";
    return true;
  }
}

bool test_logging(bool debug = false) {
  cout << "Testing logging." << "\n";

  auto orig_count_buffer = std::cout.rdbuf();
  bool res = true;

  if (debug)
    logging::logger->set_verbosity(logging::DEBUG);

  // Create loggers
  // Both logger A and logger B write to the same file :
  // logA.log, and to screen
  // Logger B only logs to logB.log
  auto loggerA = logging::get_logger("A", "logA.log");
  auto loggerB = logging::get_logger("B", "logB.log");
  auto loggerABis = loggerA->get_logger("A-bis");
  loggerABis->enable_screen_output(false);
  auto stdoutLogger = logging::get_logger("stdout");

  // Create a string stream to capture the output
  std::ostringstream stdout_stream;
  std::cout.rdbuf(stdout_stream.rdbuf());

  loggerA->error("Error A");
  loggerB->error("Error B");
  loggerABis->error("Error A bis");
  stdoutLogger->log("stdout");
  loggerA->debug("Unlogged debug because default verbosity "
                 "level is : INFO.");
  loggerA->set_verbosity(logging::DEBUG);
  loggerA->enable_screen_output(false);
  loggerA->debug("Screen only debug");

  // Restore the original cout buffer
  cout.rdbuf(orig_count_buffer);

  logging::flush();

  res = assert_equal(stdout_stream.str(),
                     string("Error A\n"
                            "Error B\n"
                            "stdout\n"),
                     "stdout") &&
        res;
  res = assert_equal(read_file("logA.log"),
                     string("Error A\n"
                            "Error A bis\n"
                            "Screen only debug\n"),
                     "log A") &&
        res;
  res =
      assert_equal(read_file("logB.log"), string("Error B\n"), "log B") && res;

  // Remove test log files
  remove("logA.log");
  remove("logB.log");

  return res;
}

int main(int argc, char *argv[]) {
  bool debug = argc >= 2 && string(argv[1]) == "-v";

  // Save original buffer of std::cout
  return !test_logging(debug);
}
