namespace vulcan {

namespace x {

template typename<T>
__global__ void gate_kernel(
    T* statevector,
    int ngate,
    int* nqubits,
    T* matrices,
    )
{
    static __shared__ T statevector2; 

    int index = threadIdx.x + blockDim.x * blockIdx.x;
}

} // namespace vulcan::x

} // namespace vulcan
