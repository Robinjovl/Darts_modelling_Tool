#include <cassert>
#include <csignal>
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
// ANSI escape sequences for colors

Logger Logger::s_root_logger;

Logger &logger = logging::get_logger("logging");

/** Basic wrapper around c++ std::cout object to expose it to python. */
void log(const string &msg) { Logger::s_root_logger.log(msg); }

/*template <LoggingLevel level> void log(const string &message) {*/
/*  Logger::s_root_logger.log<level>(message);*/
/*}*/

/*#define INSTANTIATE_GLOBAL_LOG(level, name, description) \*/
/*  template void log<level>(const string &);*/
/**/
/*LEVELS(INSTANTIATE_GLOBAL_LOG)*/

void set_verbosity(LoggingLevel level) {
  Logger::s_root_logger.set_verbosity(level);
}

void set_file(const string &filename) {
  Logger::s_root_logger.set_file(filename);
}
void flush() { Logger::s_root_logger.flush(); }

void signalHandler(int signum) {
  if (signum == SIGINT) {
    cout << "\n";
  }
  flush();
  std::exit(signum);
}

int root_logger_creations = 0;

Logger::Logger() {
  assert(root_logger_creations++ == 0);
  m_name = "root";
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);
}

Logger::Logger(const string &name) : m_name(name) {}

Logger &Logger::get_logger(const string &name) {
  lock_guard<mutex> lock(s_loggers_mutex);

  auto it = m_child_loggers.find(name);
  // Return logger if it already exists
  if (it != m_child_loggers.end()) {
    return *it->second;
  }

  // Create a new logger
  auto &child_logger =
      *m_child_loggers.emplace(name, make_unique<Logger>(name)).first->second;
  child_logger.m_level = m_level;
  child_logger.m_stdout = m_stdout;
  child_logger.set_file(m_file);
  child_logger.m_parent_logger = this;

  // Trick for using the logging logger even when it is being created
  (name == "logging" && m_name == "root" ? child_logger : logger)
      .debug("Creating child logger '{}' from '{}'.", name, m_name);
  return child_logger;
}

Logger &Logger::get_logger(const string &name,
                           const optional<string> &filename) {
  auto &child_logger = get_logger(name);
  child_logger.set_file(filename);

  return child_logger;
}

Logger &Logger::get_logger(const string &name, const optional<string> &filename,
                           const bool screen_output) {
  auto &child_logger = get_logger(name, filename);
  child_logger.m_stdout = screen_output;
  return child_logger;
}

/*Logger &get_logger(const std::string &name,*/
/*                   const std::optional<std::string> &file = std::nullopt,*/
/*                   const bool enable_screen_output = true);*/
// Root get_logger
Logger &get_logger(const string &name, const optional<string> &file,
                   const bool enable_screen_output) {
  return Logger::s_root_logger.get_logger(name, file, enable_screen_output);
}

void Logger::set_verbosity(LoggingLevel level) {
  m_level = level;
  for (auto &[_, child] : m_child_loggers) {
    child->set_verbosity(level);
  }
}

void Logger::log(const string &message) {
  if (m_fstream) {
    std::lock_guard<std::mutex> lock(m_fstream->mutex);
    m_fstream->stream << message << "\n";
  }

  if (m_stdout) {
    cout << message << "\n";
  }
}

/*template <LoggingLevel level> void Logger::log(const string &message) {*/
/*  if (level < m_level) {*/
/*    return;*/
/*  }*/
/**/
/*  log(message);*/
/*}*/

/*#define INSTANTIATE_LOG(level, name, description) \*/
/*  template void Logger::log<level>(const string &);*/
/**/
/*LEVELS(INSTANTIATE_LOG)*/

/*void Logger::log(const string &message, const LoggingLevel &level) {*/
/*  if (level < m_level) {*/
/*    return;*/
/*  }*/
/*  log(message);*/
/*}*/

void Logger::flush() {
  logger.debug("Flushing {}.", m_name);
  if (m_fstream) {
    std::lock_guard<std::mutex> lock(m_fstream->mutex);
    m_fstream->stream.flush();
  }

  if (m_stdout) {
    cout.flush();
  }

  for (auto &[_, child] : m_child_loggers) {
    child->flush();
  }
}

void Logger::set_file(const optional<string> &file) {
  for (auto &[_, child] : m_child_loggers) {
    child->set_file(file);
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
      logger.error("Failed to open file: {}", m_file.value());
      m_fstream.reset();
      m_file = std::nullopt;
    } else {
      logger.debug("New file stream created for file: {}", m_file.value());
    }

  } else {
    m_fstream = fstream.lock();
  }
}

void Logger::enable_screen_output(bool enabled) {
  m_stdout = enabled;
  for (auto &[_, child] : m_child_loggers) {
    child->enable_screen_output(enabled);
  }
}

void enable_screen_output(bool enabled) {
  Logger::s_root_logger.enable_screen_output(enabled);
}

std::unordered_map<string, std::weak_ptr<FStreamWithMutex>>
    Logger::s_file_streams;
std::mutex Logger::s_file_streams_mutex;
mutex Logger::s_loggers_mutex;

string Logger::get_visual_repr(const std::string &prefix) const {
  string res;
  res += m_name + "\n";

  auto end = m_child_loggers.end();
  for (auto it = m_child_loggers.begin(); it != end; ++it) {
    bool is_last = (std::next(it) == end);
    res += prefix + (!is_last ? "├" : "└") + " ";
    res += it->second->get_visual_repr(prefix + "│ ");
  }
  return res;
}
string get_visual_repr(const string &prefix) {
  return Logger::s_root_logger.get_visual_repr(prefix);
}

void Logger::print_loggers(const string &prefix) const {
  cout << this->get_visual_repr(prefix);
}

void print_loggers(const string &prefix) {
  Logger::s_root_logger.print_loggers(prefix);
}

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

  logging::logger.set_file("logging.log");
  logging::logger.set_verbosity(logging::DEBUG);
  logging::logger.enable_screen_output(debug);

  // Create loggers
  // Both logger A and logger B write to the same file :
  // logA.log, and to screen
  // Logger B only logs to logB.log
  auto &loggerA = logging::get_logger("A", "logA.log");
  auto &loggerB = logging::get_logger("B", "logB.log");
  auto &loggerABis = loggerA.get_logger("A-bis");
  loggerABis.enable_screen_output(false);
  auto &stdoutLogger = logging::get_logger("stdout");

  // Create a string stream to capture the output
  std::ostringstream stdout_stream;
  std::cout.rdbuf(stdout_stream.rdbuf());

  loggerA.error("Error A");
  loggerB.error("Error B");
  loggerABis.error("Error A bis");
  stdoutLogger.log("stdout");
  loggerA.debug("Unlogged debug because default verbosity "
                "level is : INFO.");
  loggerA.set_verbosity(logging::DEBUG);
  loggerA.enable_screen_output(false);
  loggerA.debug("Screen only debug");

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

  res = assert_equal(read_file("logging.log"),
                     string(
                         R"(Creating child logger 'A' from 'root'.
New file stream created for file: logA.log
Creating child logger 'B' from 'root'.
New file stream created for file: logB.log
Creating child logger 'A-bis' from 'A'.
Creating child logger 'stdout' from 'root'.
Flushing root.
Flushing B.
Flushing A.
Flushing A-bis.
Flushing stdout.
Flushing logging.
)"),
                     "logging");
  // Remove test log files
  remove("logA.log");
  remove("logB.log");
  remove("logging.log");

  res = assert_equal(logging::get_visual_repr(), string(R"(root
├ B
├ A
│ └ A-bis
├ stdout
└ logging
)"),
                     "visual representation") &&
        res;

  if (debug)
    logging::print_loggers();

  return res;
}

int main(int argc, char *argv[]) {
  bool debug = argc >= 2 && string(argv[1]) == "-v";

  // Save original buffer of std::cout
  return !test_logging(debug);
}
