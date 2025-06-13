#include "globals.h"
#include <iomanip>

#ifdef WITH_GPU
int device_num = 0;
#endif

namespace std
{
  std::string to_string(const __uint128_t& value)
  {
    return std::to_string(static_cast<double>(value));
  };
};

void write_vector_to_file(std::string file_name, std::vector<value_t> &v) 
{
  std::ofstream outFile(file_name);
  for (const auto &e : v) outFile << std::scientific << std::setprecision(5) << e << "\n";
}

