#pragma once

#include <string>

namespace logging {
/** Basic wrapper around c++ std::cout object to expose it to python. */
void log(const std::string &msg);

void duplicate_output_to_file(const std::string &file);

void flush();

} // namespace logging

