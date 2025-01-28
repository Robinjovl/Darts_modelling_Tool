#pragma once

#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

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

  TIMER = 15,

  // Confirmation that things are working as expected.
  INFO = 20,

  // An indication that something unexpected happened, or that a problem might
  // occur in the near future (e.g. ‘disk space low’). The software is still
  // working as expected.
  WARNING = 30,

  SUCCESS = 35,

  // Due to a more serious problem, the software has not been able to perform
  // some function.
  ERROR = 40,

  // A serious error, indicating that the program itself may be unable to
  // continue running.
  CRITICAL = 50
};

using enum LoggingLevel;

constexpr auto RESET = "\033[0m";

// Primary template
template <LoggingLevel level> constexpr const char *getLogLevelColor() {
  return RESET; // Default to reset
}

// Template specializations for each logging level
template <> constexpr const char *getLogLevelColor<LoggingLevel::DEBUG>() {
  return "\033[36m"; // Cyan
}

template <> constexpr const char *getLogLevelColor<LoggingLevel::TIMER>() {
  return "\033[35m"; // Magenta
}

template <> constexpr const char *getLogLevelColor<LoggingLevel::INFO>() {
  return "\033[37m"; // White/Light Gray
}

template <> constexpr const char *getLogLevelColor<LoggingLevel::WARNING>() {
  return "\033[33m"; // Yellow
}

template <> constexpr const char *getLogLevelColor<LoggingLevel::SUCCESS>() {
  return "\033[32m"; // Green
}

template <> constexpr const char *getLogLevelColor<LoggingLevel::ERROR>() {
  return "\033[31m"; // Red
}

template <> constexpr const char *getLogLevelColor<LoggingLevel::CRITICAL>() {
  return "\033[1;31m"; // Bold Red
}

#define LEVELS(X)                                                              \
  X(DEBUG, debug,                                                              \
    "Detailed information, typically only of interest to a "                   \
    "developer trying to diagnose a problem.")                                 \
  X(TIMER, timer,                                                              \
    "Timer information, how long it took to complete an operation.")           \
  X(INFO, info, "Confirmation that things are working as expected.")           \
  X(WARNING, warning,                                                          \
    "An indication that something unexpected happened, or that a"              \
    "problem might occur in the near future (e.g. ‘disk space low’). "         \
    "The software is still working as expected.")                              \
  X(SUCCESS, success, "An operation was succesful.")                           \
  X(ERROR, error,                                                              \
    "Due to a more serious problem, the software has not been "                \
    "able to perform some function.")                                          \
  X(CRITICAL, critical,                                                        \
    "A serious error, indicating that the program itself may be "              \
    "unable to continue running.")

/**
 * The default log verbosity level.
 *
 * The default verbosity level is information because users are not interested
 * in debug information.
 */
constexpr LoggingLevel DEFAULT_LOGGING_LEVEL = INFO;

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
  static Logger s_root_logger;
  std::string m_name;

  Logger(const std::string &name);
  Logger(Logger &&) = default;          // Enable move constructor
  Logger(const Logger &other) = delete; // Disable copy constructor

  Logger &get_logger(const std::string &name);
  Logger &get_logger(const std::string &name,
                     const std::optional<std::string> &filename);
  Logger &get_logger(const std::string &name,
                     const std::optional<std::string> &filename,
                     const bool screen_output);

  /*template <typename T> Logger &operator<<(const T &value) {*/
  /*  if (m_fstream) {*/
  /*    std::lock_guard<std::mutex> lock(m_fstream->mutex);*/
  /*    m_fstream->stream << value;*/
  /*  }*/
  /**/
  /*  if (m_stdout) {*/
  /*    std::cout << value;*/
  /*  }*/
  /*  return *this; // Return the current object to allow for method chaining*/
  /*}*/

  void log(const std::string &message);
  template <LoggingLevel level> void log(const std::string &message) {
    if (level < m_level) {
      return;
    }
    log(std::format("{}{}{}", getLogLevelColor<level>(), message, RESET));
  }

  template <typename... Args>
  void log(const std::format_string<Args...> &fmt, Args &&...args) {
    log(std::format(fmt, std::forward<Args>(args)...));
  }

  template <LoggingLevel level, typename... Args>
  void log(const std::format_string<Args...> &message, Args &&...args) {
    if (level < m_level) {
      return;
    }

    log(std::format("{}{}{}", getLogLevelColor<level>(),
                    std::format(message, std::forward<Args>(args)...), RESET));
  }

  // Macro to generate log functions for each level
#define LOG(level, name, description)                                          \
  /** Logs a message with this verbosity. */                                   \
  inline void name(const std::string &message) { log<level>(message); }        \
  /** Logs a message with this verbosity. */                                   \
  template <typename... Args>                                                  \
  inline void name(const std::format_string<Args...> &message,                 \
                   Args &&...args) {                                           \
    log<LoggingLevel::level, Args...>(message, std::forward<Args>(args)...);   \
  }

  LEVELS(LOG);

  void set_verbosity(LoggingLevel level);
  void enable_screen_output(bool enabled);
  void set_file(const std::optional<std::string> &file);

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

  // Debugging logging
  std::string get_visual_repr(const std::string &prefix = "") const;
  void print_loggers(const std::string &prefix = "") const;

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

// Logging logger.
//
// You can use this logger to control logging related logs.
extern Logger &logger;

// BEGIN GLOBAL LOGGING FUNCTIONS

Logger &get_logger(const std::string &name,
                   const std::optional<std::string> &file = std::nullopt,
                   const bool enable_screen_output = true);

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

void log(const std::string &message);

template <LoggingLevel level> void log(const std::string &message) {
  Logger::s_root_logger.log<level>(message);
}

/**
 * Logs a message with a specified logging level.
 *
 * @param level The desired logging level (e.g., DEBUG, INFO, WARNING, ERROR,
 * CRITICAL).
 *
 * @param msg The message to log
 */
template <LoggingLevel level, typename... Args>
void log(const std::format_string<Args...> &msg, Args &&...args) {
  Logger::s_root_logger.log<level, Args...>(msg, std::forward<Args>(args)...);
}

// Macro to generate log functions for each level
#define ROOT_LOG(level, name, description)                                     \
  inline void name(const std::string &message) { log<level>(message); }        \
  /* Logs a message with level verbosity. */                                   \
  template <typename... Args>                                                  \
  inline void name(const std::format_string<Args...> &msg, Args &&...args) {   \
    log<LoggingLevel::level, Args...>(msg, std::forward<Args>(args)...);       \
  }

LEVELS(ROOT_LOG);

void enable_screen_output(bool enabled);
std::string get_visual_repr(const std::string &prefix = "");
void print_loggers(const std::string &prefix = "");
void set_file(const std::string &filename);

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

// END GLOBAL LOGGING FUNCTIONS

} // namespace logging
