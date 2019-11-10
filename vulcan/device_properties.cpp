#include "device_properties.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include "sprintf2.hpp"

namespace vulcan {

std::string device_property_string(int device)
{
    int count;
    cudaGetDeviceCount(&count);
    if (device < 0 || device >= count) {
        throw std::runtime_error("Invalid device id");
    }
    
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device);

    std::string s;
    s += sprintf2("Name:                          %s\n",  devProp.name);
    s += sprintf2("Major revision number:         %d\n",  devProp.major);
    s += sprintf2("Minor revision number:         %d\n",  devProp.minor);
    s += sprintf2("Total global memory:           %zu\n",  devProp.totalGlobalMem);
    s += sprintf2("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    s += sprintf2("Total registers per block:     %d\n",  devProp.regsPerBlock);
    s += sprintf2("Warp size:                     %d\n",  devProp.warpSize);
    s += sprintf2("Maximum memory pitch:          %u\n",  devProp.memPitch);
    s += sprintf2("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        s += sprintf2("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        s += sprintf2("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    s += sprintf2("Clock rate:                    %d\n",  devProp.clockRate);
    s += sprintf2("Total constant memory:         %u\n",  devProp.totalConstMem);
    s += sprintf2("Texture alignment:             %u\n",  devProp.textureAlignment);
    s += sprintf2("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    s += sprintf2("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    s += sprintf2("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return s;
}

int ndevice()
{
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

} // namespace vulcan
