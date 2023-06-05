# to use the opendarts version of linear solvers (opensource) add -D OPENDARTS_LINEAR_SOLVERS
# to use bos version of linear solver (not opensource) remove those parts
# This applies to CXXFLAGS and CXXFLAGS_DEBUG
CXX         	= g++
CXXFLAGS    	= -O3 -fPIC -march=native -std=c++14 -Wno-deprecated-declarations -Wno-attributes
CXXFLAGS_DEBUG	=  -O0 -g -rdynamic -fPIC -march=native -std=c++14 -Wno-deprecated-declarations -Wno-attributes

CC          	= gcc
CFLAGS      	= -O3 -march=native -fPIC 

ARCH        	= /usr/bin/ar
ARCHFLAGS   	= cr
RANLIB      	= /usr/bin/ranlib

OMP_FLAG    	= -fopenmp

# Setup build for using opendarts-linear-solvers or deprecated version of linear-solvers
ifndef $(USE_OPENDARTS_LINEAR_SOLVERS)
USE_OPENDARTS_LINEAR_SOLVERS = true
endif

ifeq ($(USE_OPENDARTS_LINEAR_SOLVERS), true)
$(info Building darts-engines with opendarts-linear-solvers)
CXXFLAGS += -D OPENDARTS_LINEAR_SOLVERS
CXXFLAGS_DEBUG += -D OPENDARTS_LINEAR_SOLVERS
LINEAR_SOLVERS_DIR = ./lib/opendarts_linear_solvers
else
$(info Building darts-engines with iterative linear-solvers)
LINEAR_SOLVERS_DIR = ./lib/darts_linear_solvers
endif
