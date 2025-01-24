#pragma once

#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

/**
 * @namespace logging
 *
 * A logging utility designed to provide various levels of logging capabilities
 * similar to the Python logging module. This namespace facilitates logging
 * messages at different levels of verbosity, such as DEBUG, INFO, WARNING,
 * ERROR, and CRITICAL, allowing developers to control the detail of log
 * information. It also provides functions to configure logging behavior,
 * including setting the logging level, duplicating output to a file, and
 * ensuring buffered data is flushed. The namespace aims to enhance debugging
 * and monitoring of applications by providing a flexible and detailed logging
 * framework.
 *
 * # Usage
 *
 * ## Library code
 *
 * ```
 * #include "logging.h"
 *
 * logging::debug("display debug information");
 * logging::error("dsplay error");
 * ```
 *
 * ## User code
 *
 * ```
 * #include "logging.h"
 *
 * logging::set_logging_level(logging::LoggingLevel::DEBUG);  // All messages
 * are written as debug is the highest verbosity level.
 *
 * logging::set_logging_level(logging::LoggingLevel::ERROR);  // Only errors and
 * critical errors messages are written.
 * ```
 *
 */
namespace logging {
/**
 * Logging levels.
 *
 * Copied from python for a better compatibility.
 *
 * TODO: Maybe add verbosity controls more specific to the needs of darts.
 */
enum class LoggingLevel {
  // Detailed information, typically only of interest to a developer trying to
  // diagnose a problem.
  DEBUG = 10,

  // Confirmation that things are working as expected.
  INFO = 20,

  // An indication that something unexpected happened, or that a problem might
  // occur in the near future (e.g. ‘disk space low’). The software is still
  // working as expected.
  WARNING = 30,

  // Due to a more serious problem, the software has not been able to perform
  // some function.
  ERROR = 40,

  // A serious error, indicating that the program itself may be unable to
  // continue running.
  CRITICAL = 50
};

#define LEVELS(X)                                                              \
  X(DEBUG, debug,                                                              \
    "Detailed information, typically only of interest to a "                   \
    "developer trying to diagnose a problem.")                                 \
  X(INFO, info, "Confirmation that things are working as expected.")           \
  X(WARNING, warning,                                                          \
    "An indication that something unexpected happened, or that a"              \
    "problem might occur in the near future (e.g. ‘disk space low’). "         \
    "The software is still working as expected.")                              \
  X(ERROR, error,                                                              \
    "Due to a more serious problem, the software has not been "                \
    "able to perform some function.")                                          \
  X(CRITICAL, critical,                                                        \
    "A serious error, indicating that the program itself may be "              \
    "unable to continue running.")

using enum LoggingLevel;

/**
 * The default log verbosity level.
 *
 * The default verbosity level is information because users are not interested
 * in debug information.
 */
constexpr LoggingLevel DEFAULT_LOGGING_LEVEL = LoggingLevel::INFO;

/** Basic wrapper around c++ std::cout object to expose it to python. */
void log(const std::string &msg);

void set_file(const std::string &filename);

/**
 * Logs a message with a specified logging level.
 *
 * @param level The desired logging level (e.g., DEBUG, INFO, WARNING, ERROR,
 * CRITICAL).
 *
 * @param msg The message to log
 */
void log(LoggingLevel level, const std::string &msg);

// Log function with different argument order for python
void log(const std::string &msg, LoggingLevel level);

template <LoggingLevel level> void log(const std::string &msg);

// Macro to generate log functions for each level
#define ROOT_LOG(level, name, description)                                     \
  /* Logs a message with level verbosity. */                                   \
  inline void name(const std::string &msg) { log<LoggingLevel::level>(msg); }

LEVELS(ROOT_LOG);

/** Sets the logging level, which determines the verbosity of log messages.
 *
 * Higher levels produce more detailed logs, while lower levels may only show
 * critical messages. Use this to control the amount of log information based
 * on your needs.
 *
 * @param level The desired logging level (e.g., DEBUG, INFO, WARNING, ERROR,
 * CRITICAL).
 */
void set_verbosity(LoggingLevel level);

/**
 * Flushes output buffers, ensuring all data is written to the underlying
 * streams.
 *
 * This is particularly useful in case of program crashes or unexpected
 * interruptions, where buffered data might otherwise be lost.
 *
 * Note: Avoid flushing frequently, as it can significantly reduce
 * performance.
 */
void flush();

struct FStreamWithMutex {
  FStreamWithMutex() {}
  FStreamWithMutex(const std::string &filename) : stream(filename) {}
  std::ofstream stream;
  std::mutex mutex;
};

/** Logger class.
 */
class Logger {
public:
  Logger(const std::string &name);
  std::string m_name;

  Logger &get_logger(const std::string &name);
  Logger &get_logger(const std::string &name,
                     const std::optional<std::string> &filename);
  Logger &get_logger(const std::string &name,
                     const std::optional<std::string> &filename,
                     const bool screen_output);

  void log(const std::string &message);
  template <LoggingLevel level> void log(const std::string &message);
  void log(const std::string &message, const LoggingLevel &level);

  // Macro to generate log functions for each level
#define LOG(level, name, description)                                          \
  /** Logs a message with level verbosity. */                                  \
  inline void name(const std::string &message) {                               \
    log<LoggingLevel::level>(message);                                         \
  }

  LEVELS(LOG);

  void flush();

  void set_verbosity(LoggingLevel level);

  void set_file(const std::optional<std::string> &file);
  void enable_screen_output(bool enabled);
  static Logger s_root_logger;

  std::string get_visual_repr(const std::string &prefix = "") const;
  void print_loggers(const std::string &prefix = "") const;

  Logger(const Logger &other) = delete; // Disable copy constructor
  Logger(Logger &&) = default;          // Enable move constructor

private:
  // Make default constructor private because
  // it is only used to create the root logger
  Logger();

  LoggingLevel m_level = DEFAULT_LOGGING_LEVEL;
  std::unordered_map<std::string, std::unique_ptr<Logger>> m_child_loggers;
  std::optional<std::string> m_file = std::nullopt; // Log file name, optional
  std::shared_ptr<FStreamWithMutex> m_fstream; // Output file stream, with mutex
  bool m_stdout = true;                        // Output to screen
  Logger *m_parent_logger;

  static std::unordered_map<std::string, std::weak_ptr<FStreamWithMutex>>
      s_file_streams;
  static std::mutex s_file_streams_mutex;
  static std::mutex s_loggers_mutex;
};

Logger &get_logger(const std::string &name,
                   const std::optional<std::string> &file = std::nullopt,
                   const bool enable_screen_output = true);

// Logging logger.
//
// You can use this logger to control logging related logs.
extern Logger &logger;

// BEGIN GLOBAL LOGGING FUNCTIONS

void enable_screen_output(bool enabled);
std::string get_visual_repr(const std::string &prefix = "");
void print_loggers(const std::string &prefix = "");

// END GLOBAL LOGGING FUNCTIONS

} // namespace logging
