# => Vulcan Makefile <= #

# The name of the vulcan plugin
NAME = vulcan_plugin

# C++ source files for your plugin. By default we grab all *.cpp files.
CXXSRC = $(notdir $(wildcard *.cpp))
# CUDA source files for your plugin. By default we grab all *.cu files.
CUDASRC = $(notdir $(wildcard *.cu))

# Complilers, flags, includes, and libraries
CXX = /usr/bin/g++
NVCC = /usr/local/cuda/bin/nvcc

CXXFLAGS = \
    -std=c++11 \
    -fopenmp \
    -fPIC \
    `python3 -m pybind11 --includes` \
    -O3 \
    -Wall \
    -Wuninitialized \
    -Wno-unknown-pragmas \

NVCCFLAGS = \
    -Xcompiler \
    -rdynamic \
    -lineinfo \
    -gencode=arch=compute_35,code=sm_35 \
    -gencode=arch=compute_52,code=sm_52 \
    -gencode=arch=compute_61,code=sm_61 \
    -gencode=arch=compute_70,code=sm_70 \
    --ptxas-options=-v \
    --std=c++11 \
    -O3 \
    --compiler-options '-fPIC -O3' \

INCLUDES = \
    -I/usr/local/cuda/include

LIBRARIES = \
    -L/usr/local/cuda/lib64 -lcudart

LDFLAGS = \

# Used to determine linking flags.
UNAME = $(shell uname)

# Add the flags needed for shared library creation
ifeq ($(UNAME), Linux)
    LDFLAGS += -shared
endif
ifeq ($(UNAME), Darwin)
    LDFLAGS += -shared -undefined dynamic_lookup
    CXXFLAGS += -fno-common
endif

TARGET = $(NAME).so

# Start the compilation rules
default:: $(TARGET)

# The object files
BINOBJ = $(CUDASRC:%.cu=%.o)
BINOBJ += $(CXXSRC:%.cpp=%.o)

%.o: %.cu
	$(NVCC) $(CXXDEFS) $(NVCCFLAGS) $(INCLUDES) -c $<

%.o: %.cpp
	$(CXX) $(CXXDEFS) $(CXXFLAGS) $(INCLUDES) -c $<

$(TARGET): $(BINOBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(CXXDEFS) $(LIBRARIES)

# Erase all compiled intermediate files
clean:
	rm -f $(BINOBJ) $(TARGET) *.d *.pyc 

