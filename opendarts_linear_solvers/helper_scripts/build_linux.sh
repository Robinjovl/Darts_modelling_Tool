#!/bin/bash


# Setup shell script run -------------------------------------------------------
# Exit when any command fails
set -e
# ------------------------------------------------------------------------------

################################################################################
# Help info                                                                    #
################################################################################
Help_Info()
{
  # Help information ---------------------------------------------------------
  echo "$(basename "$0") [-hc]"
  echo "   Script to install opendarts-linear_solvers on macOS."
  echo "USAGE: "
  echo "   -h : displays this help menu."
  echo "   -c : cleans up build to prepare a new fresh build."
  # ------------------------------------------------------------------------------
}

################################################################################
################################################################################
# Main program                                                                 #
################################################################################
################################################################################

# Read input arguments ---------------------------------------------------------
clean_mode=false  # set mode to clean up, cleans build to prepare for fresh new build

while getopts :ch option;
do
    case "$option" in
        h) # Display help 
           Help_Info
           exit;;
        c) # Clean mode
           clean_mode=true;;
    esac
done
# ------------------------------------------------------------------------------

# Build loop -------------------------------------------------------------------
if [[ "$clean_mode" == true ]]; then
    # Cleaning build to prepare a fresh build 
    
    # Cleaning thirdparty libs 
    echo 'Time for some cleaning!'
    echo -e '   Cleaning thirdparty libs\n'
    cd ../thirdparty/SuperLU_5.2.1
    make clean 
    cd ../../helper_scripts
    
    # Clearning opendart-linear-solvers
    echo '\n   Cleaning opendarts-linear-solvers'
    rm -r ../build_make
else
  # Build 
  
  # Startup information ----------------------------------------------------------
  echo -e "\n========================================================================"
  echo "| Building opendarts-linear-solvers: START"
  echo -e "========================================================================\n"
  # ------------------------------------------------------------------------------
  
  # Build thirparty libraries ----------------------------------------------------
  echo -e "\n- Building thirdparty libs: START\n"
  cd ../thirdparty/
  
  # -- Build SuperLU -------------------------------------------------------------
  echo -e "\n--- Building SuperLU: START\n"
  cd SuperLU_5.2.1

  # Setup Makefile include files to macOS 
  cp conf_gcc_linux.mk conf.mk
  cp make_gcc_linux.inc make.inc 

  # Build SuperLU 
  make 

  # Install SuperLU (to the current directory) 
  make install

  # Return to start directory 
  cd ../
  echo -e "\n--- Building SuperLU: DONE!\n"
  # ------------------------------------------------------------------------------

  cd ../helper_scripts
  echo -e "\n- Building thirdparty libs: DONE!\n"
  # ------------------------------------------------------------------------------
  
  # Build opendarts-linear_solvers -----------------------------------------------
  echo -e "\n- Building opendarts-linear-solvers: START\n"
  # Setup build folder 
  cd ..
  mkdir -p build_make
  cd build_make

  # Setup install folder 
  mkdir -p ../../darts-engines/lib/opendarts_linear_solvers

  # Setup build with cmake 
  cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=../../darts-engines/lib/opendarts_linear_solvers -D SET_CXX11_ABI_0=TRUE -D ENABLE_TESTING=TRUE ../

  # Build 
  make

  # Test it
  ctest 

  # Install it
  make install
  
  cd ../helper_scripts
  echo -e "\n- Building opendarts-linear-solvers: DONE!\n"
  # ------------------------------------------------------------------------------
  
  # Close up information ---------------------------------------------------------
  echo -e "\n========================================================================"
  echo -e "| Building opendarts-linear-solvers: DONE!"
  echo -e "========================================================================\n"
  # ------------------------------------------------------------------------------
fi
# ------------------------------------------------------------------------------
