include conf.mk

TARGET      = ./benchmark_linear_solvers
IFLAGS      += -I$(DARTS_LINEAR_SOLVERS)/include
IFLAGS      += -I/Users/apalha/Dropbox/work/dev/onset/linear_solvers/opendarts_linear_solvers/install/include

LFLAGS_BASE	+= -L$(DARTS_LINEAR_SOLVERS)/lib
LFLAGS_BASE	+= -L/Users/apalha/Dropbox/work/dev/onset/linear_solvers/opendarts_linear_solvers/install/lib/
LFLAGS_MT  	= $(LFLAGS_BASE) -ldarts_linear_solvers_mt -lopendarts_linear_solvers


default: $(TARGET)

$(TARGET): benchmark_linear_solvers.cpp
	$(CXX) $(CXXFLAGS) $(OMP_FLAG) $(IFLAGS) $(LFLAGS_MT) -o benchmark_linear_solvers $<

clean:
	rm -rf $(TARGET)

clean_target:
	rm -rf $(TARGET)
