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
  echo "$(basename "$0") [-hcmgjt]"
  echo "   Script to install opendarts-linear_solvers on macOS."
  echo "USAGE: "
  echo "   -h : displays this help menu."
  echo "   -c : cleans up build to prepare a new fresh build."
  echo "   -t : Enable testing: ctest of solvers."
  echo "   -m : Mode in which solvers are build [Release, Debug]."
  echo "   -j : Number of threads for compilation."
  echo "   -g : g++ version, example: g++-13"
  # ------------------------------------------------------------------------------
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
gpp_version=g++-13 # Version of g++ 

while getopts ":chtm:j:g:" option;
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
        g) # gpp version
           gpp_version=${OPTARG};;
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
    rm -r ../../build
    rm -r ../../engines/lib/solvers
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
  
  # -- Build Hypre ---------------------------------------------------------------
  echo "\n--- Building Hypre: START\n"
  cd hypre/src/cmbuild
  # Setup hypre build with no MPI support (we only use single processor)
  # Request build of tests and examples just to be sure everything is fine in the build 
  cmake -D CMAKE_C_COMPILER=gcc-13 -D HYPRE_BUILD_TESTS=ON -D HYPRE_BUILD_EXAMPLES=ON -D HYPRE_WITH_MPI=OFF -D CMAKE_INSTALL_PREFIX=../../../install ..
  
  # Build hypre 
  make 
  make install
  
  # Return to start directory 
  cd ../../../
  echo "\n--- Building Hypre: DONE!\n"
  # ------------------------------------------------------------------------------
  
  # -- Build SuperLU -------------------------------------------------------------
  echo "\n--- Building SuperLU: START\n"
  cd SuperLU_5.2.1

  # Setup Makefile include files to macOS 
  cp conf_gcc-13_macOS_m1.mk conf.mk
  cp make_gcc-13_macOS_m1.inc make.inc 

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

  # Setup install folder 
  mkdir -p ../../engines/lib/solvers
  
  # Setup build folder
  rm -rf ../../build # Deletes previous build version incase it has not been cleaned up
  mkdir -p ../../build
  cd ../../build

  # Setup build with cmake
  if [[ "$testing" == true ]]; then
   cmake -D CMAKE_BUILD_TYPE=$config -D CMAKE_INSTALL_PREFIX=../engines/lib/solvers -D ONLY_SOLVERS=ON -D CMAKE_CXX_COMPILER=$gpp_version -D ENABLE_TESTING=ON ../
  else
   cmake -D CMAKE_BUILD_TYPE=$config -D CMAKE_INSTALL_PREFIX=../engines/lib/solvers -D ONLY_SOLVERS=ON -D CMAKE_CXX_COMPILER=$gpp_version ../
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
  echo "\n========================================================================"
  echo "| Building opendarts-solvers: DONE!"
  echo "========================================================================\n"
  # ------------------------------------------------------------------------------
fi
# --------------------------------------------------------------------------------