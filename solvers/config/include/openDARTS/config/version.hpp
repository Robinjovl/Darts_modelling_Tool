//--------------------------------------------------------------------------
#ifndef OPENDARTS_CONFIG_VERSION_H
#define OPENDARTS_CONFIG_VERSION_H
//--------------------------------------------------------------------------

#include <string>

namespace opendarts
{
  namespace config
  {
    // Variables for compatibility with older interface 
    extern const char *LINSOLV_BUILD_DATE;
    extern const char *LINSOLV_BUILD_MACHINE;
    extern const char *LINSOLV_BUILD_GIT_HASH;
    
    // Return the major version of openDARTS
    std::string get_version_major();

    // Return the minor version of openDARTS
    std::string get_version_minor();

    // Return the absolute path of the openDARTS source directory
    std::string get_cmake_openDARTS_source_dir();

    // The Git ref at compile time
    std::string get_git_Ref();

    // The raw git hash at compile time
    std::string get_git_hash();
    
    // The build date of this version of the code
    std::string get_build_date();
    
    // The name of the machine where the code was built
    std::string get_build_machine();

    // Whether the working directory was clean at compile time
    bool is_git_clean();

  } // namespace config
} // namespace opendarts
//--------------------------------------------------------------------------
#endif // OPENDARTS_CONFIG_VERSION_H
//--------------------------------------------------------------------------
