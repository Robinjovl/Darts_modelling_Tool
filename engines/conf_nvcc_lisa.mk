NVCC        = nvcc
NVCCFLAGS   = -ccbin=$(CXX) --compiler-options="$(CXXFLAGS)" -O3 -gencode=arch=compute_61,code=\"sm_61,compute_61\" -gencode=arch=compute_70,code=\"sm_70,compute_70\" -gencode=arch=compute_75,code=\"sm_75,compute_75\"
NVCCFLAGS_DEBUG   = -ccbin=$(CXX) --compiler-options="$(CXXFLAGS_DEBUG)" -O0 -g -G -arch=sm_61


