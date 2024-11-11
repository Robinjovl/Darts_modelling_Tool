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
  # Help information -----------------------------------------------------------
  echo "$(basename "$0") [-hcmtj]"
  echo "   Script to install opendarts-linear_solvers on macOS."
  echo "USAGE: "
  echo "   -h : displays this help menu."
  echo "   -c : cleans up build to prepare a new fresh build."
  echo "   -t : Enable testing: ctest of solvers."
  echo "   -m : Mode in which solvers are build [Release, Debug]."
  echo "   -j : Number of threads for compilation."
  # ----------------------------------------------------------------------------
}


################################################################################
################################################################################
# Main program                                                                 #
################################################################################
################################################################################

# Read input arguments ---------------------------------------------------------
clean_mode=false  # set mode to clean up, cleans build to prepare for fresh new build
testing=false     # Whether to enable the testing (ctest) of solvers.
config="Release"  # default configuration (install).
NT=8              # Number of threads by default 8

while getopts ":chtm:j:" option;
do
    case "$option" in
        h) # Display help 
           Help_Info
           exit;;
        c) # Clean mode
           clean_mode=true;;
        t) # Testing
           testing=true;; 
        m) # Select a mode
           config=${OPTARG};;
        j) # Number of threads
           NT=${OPTARG};;   
    esac
done
# ------------------------------------------------------------------------------

# Build loop -------------------------------------------------------------------
if [[ "$clean_mode" == true ]]; then
    # Cleaning build to prepare a fresh build 
    
    # Cleaning thirdparty libs 
    echo 'Time for some cleaning!'
    echo -e '   Cleaning thirdparty libs\n'
    cd ../../thirdparty/SuperLU_5.2.1
    make clean 
    cd ../../solvers/helper_scripts
    
    # Cleaning solvers
    echo '\n   Cleaning solvers'
    rm -r ../../build
    rm -r ../../engines/lib/solvers
else
  # Build 
  
  # Startup information ----------------------------------------------------------
  echo -e "\n========================================================================"
  echo "| Building opendarts-solvers: START"
  echo -e "========================================================================\n"
  # ------------------------------------------------------------------------------
  
   # Build thirparty libraries ----------------------------------------------------
  echo -e "\n- Building thirdparty libs: START\n"
  cd ../../thirdparty/
  
  # -- Build Hypre ---------------------------------------------------------------
  echo -e "\n--- Building Hypre: START\n"
  cd hypre/src/cmbuild
  # Setup hypre build with no MPI support (we only use single processor)
  # Request build of tests and examples just to be sure everything is fine in the build 
  cmake -D HYPRE_BUILD_TESTS=ON -D HYPRE_BUILD_EXAMPLES=ON -D HYPRE_WITH_MPI=OFF -D CMAKE_INSTALL_PREFIX=../../../install ..
  
  # Build hypre 
  make 
  make install
  
  # Return to start directory 
  cd ../../../
  echo -e "\n--- Building Hypre: DONE!\n"
  # ------------------------------------------------------------------------------
  
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

  cd ../solvers/helper_scripts
  echo -e "\n- Building thirdparty libs: DONE!\n"
  # ------------------------------------------------------------------------------

  # Setup install folder 
  mkdir -p ../../engines/lib/solvers
  
  # Setup build folder
  rm -rf ../../build # Deletes previous build version incase it has not been cleaned up
  mkdir -p ../../build
  cd ../../build

  # Setup build with cmake
  if [[ "$testing" == true ]]; then
    cmake -D CMAKE_BUILD_TYPE=$config -D SET_CXX11_ABI_0=TRUE -D CMAKE_INSTALL_PREFIX=../engines/lib/solvers -D ONLY_SOLVERS=ON -D ENABLE_TESTING=ON ../
  else
    cmake -D CMAKE_BUILD_TYPE=$config -D SET_CXX11_ABI_0=TRUE -D CMAKE_INSTALL_PREFIX=../engines/lib/solvers -D ONLY_SOLVERS=ON ../
  fi
  
  # Build
  make -j $NT

  # Test it
  if [[ "$testing" == true ]]; then
    ctest
  fi

  # Install it
  make install -j $NT
  
  # Return to root
  cd ..
  # ------------------------------------------------------------------------------
  
  # Close up information ---------------------------------------------------------
  echo -e "\n========================================================================"
  echo -e "| Building opendarts-solvers: DONE!"
  echo -e "========================================================================\n"
  # ------------------------------------------------------------------------------
fi
# --------------------------------------------------------------------------------
