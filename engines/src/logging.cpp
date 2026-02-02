#include <fstream>
#include <iostream>
#include <ostream>
#include <vector>

#include "logging.h"

using namespace std;

/**
 * This class allows to combine several output streams into one.
 */
class CombinedOutputs : public streambuf {
public:
  /**
   * Constructor of CombinedOutputs which takes any number of output streams as
   * arguments.
   */
  template <typename... Streams> CombinedOutputs(Streams &...outputStreams) {
    addStream(outputStreams...);
  }

public:
  /** All the output streams that are combined. */
  vector<ostream *> streams;

  // Called when buffer is full and requires flushing
  int overflow(int c) override {
    if (c == EOF)
      return EOF;

    for (auto &stream : streams) {
      if (!(stream->put(c))) {
        return EOF;
      }
    }
    return c;
  }

  // Synchronize is called to flush the stream buffers.
  int sync() override {
    for (auto &stream : streams) {
      if (!(stream->flush())) {
        return -1;
      }
    }
    return 0;
  }

private:
  /** Recursive variadic method to support any number of output streams. */
  template <typename Stream, typename... Rest>
  void addStream(Stream &stream, Rest &...rest) {
    streams.push_back(&stream);
    addStream(rest...);
  }

  // Stop case
  void addStream() {}
};

/**
 * Singleton class which contains all the output streams used for logging.
 *
 * Standard output with std::cout can be duplicated to a file with
 * `LoggingManagement::duplicate_output_to_file.`
 */
class LoggingManagement {
public:
  /**
   * Duplicates standard output with std::cout to a specified file.
   */
  void duplicate_output_to_file(const string &filepath) {
    logFile.close();
    logFile.open(filepath);
    if (!logFile.is_open()) {
      cerr << "Log: Failed to open file" << endl;
      combined = CombinedOutputs(stdioStream);
      return;
    }
    combined = CombinedOutputs(logFile, stdioStream);
    cout.rdbuf(&combined);
  }

  /**
   * Flushes stream buffers.
   */
  void flush() { LoggingManagement::instance().combined.sync(); }

  /**
   * Getter for the singleton instance.
   */
  static LoggingManagement &instance() {
    static LoggingManagement instance;
    return instance;
  }

private:
  /// Optional file to duplicate output to
  ofstream logFile;

  /// Standard output stream
  ostream stdioStream;

  /// Combined output streams
  CombinedOutputs combined;

  /// Private singleton constructor
  LoggingManagement() : stdioStream(cout.rdbuf()) {}

  /// Private singleton desctructor which closes the output file
  ~LoggingManagement() { logFile.close(); }
};

namespace logging {
/** Current logging level. */
LoggingLevel logging_level = DEFAULT_LOGGING_LEVEL;

/** Sets the logging level. */
void set_logging_level(LoggingLevel level) { logging_level = level; }

/** Basic wrapper around c++ std::cout object to expose it to python. */
void log(const string &msg) { cout << msg << "\n"; }

void log(LoggingLevel level, const std::string &msg) {
  if (level < logging_level)
    return;
  log(msg);
}

void log(const std::string &msg, LoggingLevel level) {
  log(level, msg);
}

void flush() { LoggingManagement::instance().flush(); }

void duplicate_output_to_file(const string &file) {
  LoggingManagement::instance().duplicate_output_to_file(file);
}

} // namespace logging
