#pragma once

#include <string>


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
 * logging::set_logging_level(logging::LoggingLevel::DEBUG);  // All messages are written as debug is the highest verbosity level.
 *
 * logging::set_logging_level(logging::LoggingLevel::ERROR);  // Only errors and critical errors messages are written.
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

/**
 * The default log verbosity level.
 *
 * The default verbosity level is information because users are not interested in debug
 * information.
 */
constexpr LoggingLevel DEFAULT_LOGGING_LEVEL = LoggingLevel::INFO;

/** Basic wrapper around c++ std::cout object to expose it to python. */
void log(const std::string &msg);

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

// Macro to generate log functions for each level
#define LOG_FUNCTION(level, name)                                              \
  /** Logs a message with level verbosity. */                                                                    \
  inline void name(const std::string &msg) { log(LoggingLevel::level, msg); }

LOG_FUNCTION(DEBUG, debug);
LOG_FUNCTION(INFO, info);
LOG_FUNCTION(WARNING, warning);
LOG_FUNCTION(ERROR, error);
LOG_FUNCTION(CRITICAL, critical);

/** Sets the logging level, which determines the verbosity of log messages.
 *
 * Higher levels produce more detailed logs, while lower levels may only show
 * critical messages. Use this to control the amount of log information based on
 * your needs.
 *
 * @param level The desired logging level (e.g., DEBUG, INFO, WARNING, ERROR,
 * CRITICAL).
 */
void set_logging_level(LoggingLevel level);

/**
 * Configures standard output with std::cout to be duplicated to a specified
 * file.
 */
void duplicate_output_to_file(const std::string &file);

/**
 * Flushes output buffers, ensuring all data is written to the underlying
 * streams.
 *
 * This is particularly useful in case of program crashes or unexpected
 * interruptions, where buffered data might otherwise be lost.
 *
 * Note: Avoid flushing frequently, as it can significantly reduce performance.
 */
void flush();

} // namespace logging
