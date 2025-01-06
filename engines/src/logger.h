#include <string>

using namespace std;

/** Basic wrapper around c++ std::cout object to expose it to python. */
void print(const string &msg);

void duplicate_output_to_file(const string &file);


class LoggingManagement {
public:
  static void flush();
};
