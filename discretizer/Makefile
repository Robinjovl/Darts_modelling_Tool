include conf_gcc.mk

#CPP = g++
#C = gcc
INCLUDEFLAGS = -I./src \
				-I../engines/src \
				-I../engines/src/interpolation \
				-I../thirdparty/pybind11/include \
				-I../thirdparty \
				-I../thirdparty/eigen \
				-I../thirdparty/MshIO/src \
				-I../thirdparty/MshIO/include \
				$(shell python3-config --includes)

#LFLAGS = $(shell python3-config --libs)

SOURCEDIR = ./src
MSHIODIR = ../thirdparty/MshIO/src

DFLAGS      = -DPYBIND11_ENABLED -D_GLIBCXX_USE_CXX11_ABI=0
CPPSOURCES = $(shell find $(SOURCEDIR) -name '*.cpp')
CPPSOURCES += $(shell find $(MSHIODIR) -name '*.cpp')
CPPOBJECTS = $(CPPSOURCES:%.cpp=%.o)

EXECUTABLE = ../darts/discretizer.so

BUILD_MACHINE := $(shell whoami)@$(shell hostname)
BUILD_DATE := $(shell TZ=Europe/Amsterdam date +"%d/%m/%Y, %H:%M:%S")
BUILD_GIT_HASH := $(shell git describe --always --dirty --match 'NOT A TAG')

all: $(EXECUTABLE) 

debug: CFLAGS += -O0 -g -fsanitize=address -fno-omit-frame-pointer
debug: all

release: CFLAGS += -O2
release: all

$(EXECUTABLE) : $(CPPOBJECTS) $(COBJECTS)
	$(CXX) -shared -o $@ $(COBJECTS) $(CPPOBJECTS) $(LFLAGS) 

%.o: %.cpp
	$(CXX) $(DFLAGS) $(CXXFLAGS) $(INCLUDEFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDEFLAGS) -c $< -o $@

clean:
	rm -f $(CPPOBJECTS) $(EXECUTABLE)


.PHONY: build_info

# Additional rule to force rebuild build_info.cpp file every time
src/discretizer_build_info.cpp: build_info
    $(shell echo "#include \"discretizer_build_info.h\""  > src/discretizer_build_info.cpp)
    $(shell echo "const char *DISCRETIZER_BUILD_DATE = \"$(BUILD_DATE)\";"  >> src/discretizer_build_info.cpp)
    $(shell echo "const char *DISCRETIZER_BUILD_MACHINE = \"$(BUILD_MACHINE)\";"  >> src/discretizer_build_info.cpp)
    $(shell echo "const char *DISCRETIZER_BUILD_GIT_HASH = \"$(BUILD_GIT_HASH)\";"  >> src/discretizer_build_info.cpp)
