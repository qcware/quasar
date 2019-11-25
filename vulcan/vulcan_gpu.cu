#include "vulcan_gpu.hpp"
#include "vulcan_types.hpp"

namespace vulcan {

namespace gpu {

// => Utility Functions <= //

std::pair<int, int> cuda_grid_size(int nqubit, int block_size)
{
    int size = (1ULL << nqubit);
    if (size < block_size) {
        return std::pair<int, int>(1, size);
    } else {
        return std::pair<int, int>(size / block_size, block_size);
    }
}

// => Dot Kernel <= //

// > Warp and Block Binary Summation Functions < //

template <typename T>
__inline__ __device__
T shfl_xor(T val, int offset);

// Kepler shuffle: SM 3+
// val += __shfl_xor(val, offset);
// NOTE: __shfl_xor is actually deprecated as of CUDA 9 due to the
// possibility of intra-warp divergence in Volta. The solution is to
// use a drop-in replacement:
//  "__shfl_xx(val, ...)" -> "__shfl_xx_sync(0xFFFFFFFF, val, ...)"
// 
// So in CUDA 9+ we would use:
// val += __shfl_xor_sync(0xFFFFFFFF, val, offset);

template <typename T>
__inline__ __device__
scalar<T> shfl_xor(scalar<T> val, int offset) {
    return scalar<T>(
	__shfl_xor_sync(0xFFFFFFFF, val.real(), offset));
        // __shfl_xor(val.real(), offset));
}

template <typename T>
__inline__ __device__
complex<T> shfl_xor(complex<T> val, int offset) {
    return complex<T>(
	__shfl_xor_sync(0xFFFFFFFF, val.real(), offset),
	__shfl_xor_sync(0xFFFFFFFF, val.imag(), offset));
        // __shfl_xor(val.real(), offset),
       //  __shfl_xor(val.imag(), offset));
}

/**
 * Perform binary summation across all threads within a warp. Use of shared or
 * global memory is not allowed. Use of block-level synchronization is not
 * allowed. The allowed time complexity is O(log2(warpSize)). Note that you
 * should not assume that the warp is 32 wide, but should use the special
 * keyword warpSize to determine the size of the warp.
 * 
 * @param val the contribution to the sum from the current thread
 * @return val the sum of val across all threads within the warp. Only the 0-th
 *  thread within the warp needs to return the full sum (though there is no
 *  performance penalty for all the threads to return the full sum).
 **/
template <typename T>
__inline__ __device__
T warp_sum(T val)
{
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += vulcan::gpu::shfl_xor(val, offset);
    }
    return val;
}

/**
 * Perform binary summation across all threads within a 1D block. You may
 * assume that a maximum of 32 warps are present in the block, and you may use
 * up to 32 floats of shared memory. You may assume that there are at least 32
 * threads in a warp.  Use of global memory is not allowed. The allowed time
 * complexity is O(log2(blockDim.x)). You may use two __syncthreads calls. 
 * 
 * @param val the contribution to the sum from the current thread
 * @return val the sum of val across all threads within the warp. Only the 0-th
 *  thread within the block needs to return the full sum (getting all threads
 *  within the block to return the same sum requires three calls to
 *  __syncthreads).
 **/
template <typename T>
__inline__ __device__
T block_sum(T val)
{
    static __shared__ T shared[32];
    val = vulcan::gpu::warp_sum(val);
    if (threadIdx.x % warpSize == 0) shared[threadIdx.x / warpSize] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize ? shared[threadIdx.x] : T::zero());
    if (threadIdx.x / warpSize == 0) val = vulcan::gpu::warp_sum(val);
    __syncthreads(); // This syncthreads is so that nobody writes into shared before we are finished reading
    return val;
}

template <typename T>
__global__ void dot_kernel_1(
    int nqubit, 
    T* statevector1,
    T* statevector2,
    T* output)
{
    T val = T::zero();
    for (int index = threadIdx.x + blockIdx.x * blockDim.x; index < (1 << nqubit); index += blockDim.x * gridDim.x) {
        val += vulcan::conj(statevector1[index]) * statevector2[index];
    }

    val = vulcan::gpu::block_sum(val);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = val;   
    }
} 

template <typename T>
__global__ void dot_kernel_2(
    int size,
    T* array)
{
    T val = T::zero();
    for (int index = threadIdx.x + blockIdx.x * blockDim.x; index < size; index += blockDim.x * gridDim.x) {
        val += array[index];
    }

    val = vulcan::gpu::block_sum(val);

    if (threadIdx.x == 0) {
        array[blockIdx.x] = val;   
    }
} 

template <typename T>
T dot(
    int nqubit,
    T* statevector1,
    T* statevector2)
{
    int block_size = 512; // preferred block size
    int grid_size = 64; // max grid size
    
    if ((1ULL << nqubit) <= block_size) {
        T* output;
        cudaMalloc(&output, sizeof(T));
        // never launch less than one full warp
        int clamped_block_size = max((int) (1ULL << nqubit), 32); 
        dot_kernel_1<<<1, clamped_block_size>>>(
            nqubit,
            statevector1,
            statevector2,
            output);
        T val;
        cudaMemcpy(&val, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
        return val;
    } 
    
    grid_size = min(grid_size, (int) ((1ULL << nqubit) / block_size));
    T* output;
    cudaMalloc(&output, grid_size*sizeof(T));
    dot_kernel_1<<<grid_size, block_size>>>(
        nqubit,
        statevector1,
        statevector2,
        output);
    // never launch less than one full warp
    int clamped_block_size = max(grid_size, 32);
    dot_kernel_2<<<1, clamped_block_size>>>(
        grid_size,
        output);
    T val;
    cudaMemcpy(&val, output, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(output);
    return val;
}

template float64 dot<float64>(int, float64*, float64*);
template float32 dot<float32>(int, float32*, float32*);
template complex128 dot<complex128>(int, complex128*, complex128*);
template complex64 dot<complex64>(int, complex64*, complex64*);

// => AXPBY Kernel <= //

template <typename T>
__global__ void axpby_kernel(
    T* statevector1,
    T* statevector2,
    T a,
    T b)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    T val = statevector1[index];
    val *= a;
    
    if (b != T::zero()) {
        val += b * statevector2[index];
    }

    statevector2[index] = val;
}

template<typename T>
int axpby_block_size();

template<>
int axpby_block_size<float32>() { return 256; }

template<>
int axpby_block_size<float64>() { return 128; }

template<>
int axpby_block_size<complex64>() { return 64; }

template<>
int axpby_block_size<complex128>() { return 64; }

template <typename T>
void axpby(
    int nqubit,
    T* statevector1_d,
    T* statevector2_d,
    T a,
    T b)
{
    std::pair<int, int> grid_size = cuda_grid_size(nqubit, axpby_block_size<T>());

    axpby_kernel<<<std::get<0>(grid_size), std::get<1>(grid_size)>>>(
        statevector1_d,
        statevector2_d,
        a,  
        b);
}

template void axpby<float64>(int, float64*, float64*, float64, float64); 
template void axpby<float32>(int, float32*, float32*, float32, float32); 
template void axpby<complex128>(int, complex128*, complex128*, complex128, complex128); 
template void axpby<complex64>(int, complex64*, complex64*, complex64, complex64); 

// => 1-Qubit Kernel <= //

template <typename T>
__global__ void gate_1_kernel(
    T* statevector1,
    T* statevector2,
    int qubitA,
    T O00,
    T O01,
    T O10,
    T O11,
    T a,
    T b)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    int mask = (1 << qubitA) - 1;
    int index0 = (index << 1) - ((index & mask) << 1) + (index & mask);
    int index1 = index0 + (1 << qubitA);
    
    T val0 = statevector1[index0];
    T val1 = statevector1[index1];

    T res0 = O00 * val0 + O01 * val1;
    T res1 = O10 * val0 + O11 * val1;

    res0 *= a;
    res1 *= a;
    
    if (b != T::zero()) {
        res0 += b * statevector2[index0];
        res1 += b * statevector2[index1];
    }
    
    statevector2[index0] = res0;
    statevector2[index1] = res1;
}

template<typename T>
int apply_gate_1_block_size();

template<>
int apply_gate_1_block_size<float32>() { return 128; }

template<>
int apply_gate_1_block_size<float64>() { return 64; }

template<>
int apply_gate_1_block_size<complex64>() { return 32; }

template<>
int apply_gate_1_block_size<complex128>() { return 16; }

template <typename T>
void apply_gate_1(
    int nqubit,
    T* statevector1_d,
    T* statevector2_d,
    int qubitA,
    T O00,
    T O01,
    T O10,
    T O11,
    T a,
    T b)
{
    if (qubitA < 0 || qubitA >= nqubit) {
        throw std::runtime_error("qubitA is out of bounds");
    }

    std::pair<int, int> grid_size = cuda_grid_size(nqubit - 1, apply_gate_1_block_size<T>());

    gate_1_kernel<<<std::get<0>(grid_size), std::get<1>(grid_size)>>>(
        statevector1_d,
        statevector2_d,
        qubitA,
        O00,
        O01,
        O10,
        O11,
        a,
        b);
}
    
template void apply_gate_1<float64>(
    int, 
    float64*, 
    float64*, 
    int, 
    float64, 
    float64, 
    float64, 
    float64,
    float64,
    float64);

template void apply_gate_1<float32>(
    int, 
    float32*, 
    float32*, 
    int, 
    float32, 
    float32, 
    float32, 
    float32,
    float32,
    float32);

template void apply_gate_1<complex128>(
    int, 
    complex128*, 
    complex128*, 
    int, 
    complex128, 
    complex128, 
    complex128, 
    complex128,
    complex128,
    complex128);

template void apply_gate_1<complex64>(
    int, 
    complex64*, 
    complex64*, 
    int, 
    complex64, 
    complex64, 
    complex64, 
    complex64,
    complex64,
    complex64);

// => 2-Qubit Kernel <= //

template <typename T>
__global__ void gate_2_kernel(
    T* statevector1,
    T* statevector2,
    int qubitA,
    int qubitB,
    T O00,
    T O01,
    T O02,
    T O03,
    T O10,
    T O11,
    T O12,
    T O13,
    T O20,
    T O21,
    T O22,
    T O23,
    T O30,
    T O31,
    T O32,
    T O33,
    T a,
    T b)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    int max_qubit = max(qubitA, qubitB);
    int min_qubit = min(qubitA, qubitB);
    int max_mask = (1 << (max_qubit - 1)) - 1;
    int min_mask = (1 << min_qubit) - 1;
    int index0 = 0;
    index0 += (index << 2) - ((index & max_mask) << 2);
    index &= max_mask;
    index0 += (index << 1) - ((index & min_mask) << 1);
    index &= min_mask;
    index0 += index;
    int index1 = index0 + (1 << qubitB);
    int index2 = index0 + (1 << qubitA);
    int index3 = index0 + (1 << qubitA) + (1 << qubitB);
    
    T val0 = statevector1[index0];
    T val1 = statevector1[index1];
    T val2 = statevector1[index2];
    T val3 = statevector1[index3];

    T res0 = O00 * val0 + O01 * val1 + O02 * val2 + O03 * val3;
    T res1 = O10 * val0 + O11 * val1 + O12 * val2 + O13 * val3;
    T res2 = O20 * val0 + O21 * val1 + O22 * val2 + O23 * val3;
    T res3 = O30 * val0 + O31 * val1 + O32 * val2 + O33 * val3;

    res0 *= a;
    res1 *= a;
    res2 *= a;
    res3 *= a;
    
    if (b != T::zero()) {
        res0 += b * statevector2[index0];
        res1 += b * statevector2[index1];
        res2 += b * statevector2[index2];
        res3 += b * statevector2[index3];
    }
    
    statevector2[index0] = res0;
    statevector2[index1] = res1;
    statevector2[index2] = res2;
    statevector2[index3] = res3;
}

template<typename T>
int apply_gate_2_block_size();

template<>
int apply_gate_2_block_size<float32>() { return 128; }

template<>
int apply_gate_2_block_size<float64>() { return 64; }

template<>
int apply_gate_2_block_size<complex64>() { return 32; }

template<>
int apply_gate_2_block_size<complex128>() { return 16; }

template <typename T>
void apply_gate_2(
    int nqubit,
    T* statevector1_d,
    T* statevector2_d,
    int qubitA,
    int qubitB,
    T O00,
    T O01,
    T O02,
    T O03,
    T O10,
    T O11,
    T O12,
    T O13,
    T O20,
    T O21,
    T O22,
    T O23,
    T O30,
    T O31,
    T O32,
    T O33,
    T a,
    T b)
{
    if (qubitA < 0 || qubitA >= nqubit) {
        throw std::runtime_error("qubitA is out of bounds");
    }
    if (qubitB < 0 || qubitB >= nqubit) {
        throw std::runtime_error("qubitB is out of bounds");
    }
    if (qubitA == qubitB) {
        throw std::runtime_error("qubitA == qubitB");
    }

    std::pair<int, int> grid_size = cuda_grid_size(nqubit - 2, apply_gate_2_block_size<T>());

    gate_2_kernel<<<std::get<0>(grid_size), std::get<1>(grid_size)>>>(
        statevector1_d,
        statevector2_d,
        qubitA,
        qubitB,
        O00,
        O01,
        O02,
        O03,
        O10,
        O11,
        O12,
        O13,
        O20,
        O21,
        O22,
        O23,
        O30,
        O31,
        O32,
        O33,
        a,
        b);
}

template void apply_gate_2<float64>(
   int, 
   float64*, 
   float64*, 
   int, 
   int, 
   float64, 
   float64, 
   float64, 
   float64,
   float64, 
   float64, 
   float64, 
   float64,
   float64, 
   float64, 
   float64, 
   float64,
   float64, 
   float64, 
   float64, 
   float64, 
   float64, 
   float64);
    
template void apply_gate_2<float32>(
   int, 
   float32*, 
   float32*, 
   int, 
   int, 
   float32, 
   float32, 
   float32, 
   float32,
   float32, 
   float32, 
   float32, 
   float32,
   float32, 
   float32, 
   float32, 
   float32,
   float32, 
   float32, 
   float32, 
   float32, 
   float32, 
   float32);
    
template void apply_gate_2<complex128>(
   int, 
   complex128*, 
   complex128*, 
   int, 
   int, 
   complex128, 
   complex128, 
   complex128, 
   complex128,
   complex128, 
   complex128, 
   complex128, 
   complex128,
   complex128, 
   complex128, 
   complex128, 
   complex128,
   complex128, 
   complex128, 
   complex128, 
   complex128, 
   complex128, 
   complex128);
    
template void apply_gate_2<complex64>(
   int, 
   complex64*, 
   complex64*, 
   int, 
   int, 
   complex64, 
   complex64, 
   complex64, 
   complex64,
   complex64, 
   complex64, 
   complex64, 
   complex64,
   complex64, 
   complex64, 
   complex64, 
   complex64,
   complex64, 
   complex64, 
   complex64, 
   complex64, 
   complex64, 
   complex64);

// => Measurement Operations <= //

template <typename T, typename U>
__global__ void abs2_kernel(
    T* statevector1_d,
    U* statevector2_d)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    T val = statevector1_d[index];
    T abs2 = vulcan::abs2(val);
    statevector2_d[index] = abs2.real();
}

template<typename T>
int abs2_block_size();

template<>
int abs2_block_size<float32>() { return 256; }

template<>
int abs2_block_size<float64>() { return 128; }

template<>
int abs2_block_size<complex64>() { return 128; }

template<>
int abs2_block_size<complex128>() { return 64; }

template <typename T, typename U>
void abs2(
    int nqubit,
    T* statevector1_d,
    U* statevector2_d)
{
    std::pair<int, int> grid_size = cuda_grid_size(nqubit, abs2_block_size<T>());

    abs2_kernel<<<std::get<0>(grid_size), std::get<1>(grid_size)>>>(
        statevector1_d,
        statevector2_d);
}

template void abs2<float64, float64>(int, float64*, float64*);
template void abs2<float32, float32>(int, float32*, float32*);
template void abs2<complex128, float64>(int, complex128*, float64*);
template void abs2<complex64, float32>(int, complex64*, float32*);

template<typename T>
int sweep_block_size();

template<>
int sweep_block_size<float32>() { return 256; }

template<>
int sweep_block_size<float64>() { return 128; }

template <typename T>
__global__ void upsweep_kernel(
    T* statevector,
    int d)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    
    int k = index * (1 << (d+1));

    int index0 = k + (1 << (d)) - 1;
    int index1 = k + (1 << (d+1)) - 1;

    T a = statevector[index0];
    T b = statevector[index1];

    T c = a + b;

    statevector[index1] = c;
}

template <typename T>
__global__ void downsweep_kernel(
    T* statevector,
    int d)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    
    int k = index * (1 << (d+1));

    int index0 = k + (1 << (d)) - 1;
    int index1 = k + (1 << (d+1)) - 1;

    T a = statevector[index0];
    T b = statevector[index1];

    T c = a + b;

    statevector[index0] = b;
    statevector[index1] = c;
}

template <typename T>
T cumsum(
    int nqubit,
    T* statevector)
{
    for (int d = 0; d < nqubit; d++) {
        std::pair<int, int> grid_size = cuda_grid_size(nqubit - 1 - d, sweep_block_size<T>());
        upsweep_kernel<<<std::get<0>(grid_size), std::get<1>(grid_size)>>>(
            statevector, 
            d);
    }

    T val = get_statevector_element(statevector, (1 << nqubit) - 1);
    set_statevector_element(statevector, (1 << nqubit) - 1, T(0.0));

    for (int d = nqubit - 1; d >= 0; d--) {
        std::pair<int, int> grid_size = cuda_grid_size(nqubit - 1 - d, sweep_block_size<T>());
        downsweep_kernel<<<std::get<0>(grid_size), std::get<1>(grid_size)>>>(
            statevector, 
            d);
    }

    return val;
}

template float64 cumsum<float64>(int, float64*);
template float32 cumsum<float32>(int, float32*);

template <typename T>
__global__ void measurement_kernel(
    int nqubit,
    T* cumsum,
    T sum,
    int nmeasurement,
    T* randoms,
    int* measurements)
{
    int index = threadIdx.x  + blockDim.x * blockIdx.x;
    if (index >= nmeasurement) return;
    T random = randoms[index];
    random *= sum; // Now in the [0, sum) regime
    
    int ket = 0;
    int interval = (1 << nqubit);
    for (int d = 0; d < nqubit; d++) {
        interval >>= 1;
        int ket2 = ket + interval; 
        T pval = cumsum[ket2];
        if (pval.real() < random.real()) {
            ket = ket2;
        }
    }
     
    measurements[index] = ket; 
}

template <typename T>
void measure(
    int nqubit,
    T* cumsum,
    T sum,
    int nmeasurement,
    T* randoms,
    int* measurements)
{
    int block_size = 512;

    int grid_size = (nmeasurement - 1) / block_size + 1;
    
    measurement_kernel<<<grid_size, block_size>>>(
        nqubit,
        cumsum,
        sum,
        nmeasurement,
        randoms,
        measurements);
}

template void measure<float64>(int, float64*, float64, int, float64*, int*);
template void measure<float32>(int, float32*, float32, int, float32*, int*);

} // namespace vulcan::gpu

} // namespace vulcan
