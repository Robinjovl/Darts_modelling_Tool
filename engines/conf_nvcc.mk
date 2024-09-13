NVCC        		= nvcc
NVCCFLAGS   		= -ccbin=$(CXX) --compiler-options="$(CXXFLAGS)" -G -arch=sm_80 --ptxas-options=-v -allow-unsupported-compiler
NVCCFLAGS_DEBUG   	= -ccbin=$(CXX) --compiler-options="$(CXXFLAGS_DEBUG)" -G -arch=sm_80 
NVCCFLAGS_PROFILE  	= -ccbin=$(CXX) --compiler-options="$(CXXFLAGS_DEBUG)" -O3 -lineinfo) -arch=sm_80


