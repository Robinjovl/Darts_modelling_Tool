#!/bin/zsh


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
    echo '\n   Cleaning thirdparty libs\n'
    cd ../../thirdparty/SuperLU_5.2.1
    make clean 
    cd ../../solvers/helper_scripts
    
    # Clearning solvers
    echo '\n   Cleaning opendarts-solvers'
    rm -r ../build_make
else
  # Build 
  
  # Startup information ----------------------------------------------------------
  echo "\n========================================================================"
  echo "| Building opendarts-solvers: START"
  echo "========================================================================\n"
  # ------------------------------------------------------------------------------
  
  # Build thirparty libraries ----------------------------------------------------
  echo "\n- Building thirdparty libs: START\n"
  cd ../../thirdparty/
  
  # -- Build SuperLU -------------------------------------------------------------
  echo "\n--- Building SuperLU: START\n"
  cd SuperLU_5.2.1

  # Setup Makefile include files to macOS 
  cp conf_gcc-11_macOS_m1.mk conf.mk
  cp make_gcc-11_macOS_m1.inc make.inc 

  # Build SuperLU 
  make 

  # Install SuperLU (to the current directory) 
  make install

  # Return to start directory 
  cd ../
  echo "\n--- Building SuperLU: DONE!\n"
  # ------------------------------------------------------------------------------

  cd ../solvers/helper_scripts
  echo "\n- Building thirdparty libs: DONE!\n"
  # ------------------------------------------------------------------------------
  
  # Build opendarts-linear_solvers -----------------------------------------------
  echo "\n- Building opendarts-solvers: START\n"
  # Setup build folder 
  cd ..
  mkdir -p build_make
  cd build_make

  # Setup install folder 
  mkdir -p ../../engines/lib/solvers

  # Setup build with cmake 
  cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=../../engines/lib/solvers -D SET_CXX11_ABI_0=TRUE -D ENABLE_TESTING=TRUE ../

  # Build 
  make

  # Test it
  ctest 

  # Install it
  make install
  
  cd ../helper_scripts
  echo "\n- Building opendarts-solvers: DONE!\n"
  # ------------------------------------------------------------------------------
  
  # Close up information ---------------------------------------------------------
  echo "\n========================================================================"
  echo "| Building opendarts-solvers: DONE!"
  echo "========================================================================\n"
  # ------------------------------------------------------------------------------
fi
# ------------------------------------------------------------------------------
