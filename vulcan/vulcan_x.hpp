namespace vulcan {

namespace x {

template<typename T, int D, int B>
__device__ __forceinline__ void apply_gate_1(
    int nstatevector,
    T* statevector,
    int* qubits,
    T* matrices)
{
    int qubitA = qubits[0];
    T O00 = matrices[0];
    T O01 = matrices[1];
    T O10 = matrices[2];
    T O11 = matrices[3];

    for (int k = 0; k < D; k++) {
        int indexG = threadIdx.x + k * B + blockIdx.x * D * B;
        if (indexG >= nstatevector) continue;
        int index0 = threadsIdx.x + k * B;
        if (index0 & (1 << qubitA)) continue;
        int index1 = index0 + (1 << qubitA);
        T val0 = statevector[index0];
        T val1 = statevector[index1];
        T res0 = O00 * val0 + O01 * val1;
        T res1 = O10 * val0 + O11 * val1;
        statevector[index0] = res0;
        statevector[index1] = res1;
    }
}
    
template<typename T, int D, int B>
__device__ __forceinline__ void apply_gate_2(
    int nstatevector,
    T* statevector,
    int* qubits,
    T* matrices)
{
    int qubitA = qubits[0];
    int qubitB = qubits[1];
    T O00 = matrices[ 0];
    T O01 = matrices[ 1];
    T O02 = matrices[ 2];
    T O03 = matrices[ 3];
    T O10 = matrices[ 4];
    T O11 = matrices[ 5];
    T O12 = matrices[ 6];
    T O13 = matrices[ 7];
    T O20 = matrices[ 8];
    T O21 = matrices[ 9];
    T O22 = matrices[10];
    T O23 = matrices[11];
    T O30 = matrices[12];
    T O31 = matrices[13];
    T O32 = matrices[14];
    T O33 = matrices[15];

    for (int k = 0; k < D; k++) {
        int indexG = threadIdx.x + k * B + blockIdx.x * D * B;
        if (indexG >= nstatevector) continue;
        int index0 = threadsIdx.x + k * B;
        if (index0 & ((1 << qubitA) + (1 << qubitB))) continue;
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
        statevector[index0] = res0;
        statevector[index1] = res1;
        statevector[index2] = res2;
        statevector[index3] = res3;
    }
}
    

template <typename T, int D, int B>
__global__ void gate_kernel(
    int nstatevector,
    T* statevector,
    int ngate,
    int* nqubits,
    int* qubits,
    T* matrices)
{
    static __shared__ T statevectorS[D * B]; 

    for (int k = 0; k < D; k++) {
        int indexG = threadIdx.x + k * B + blockIdx.x * D * B;
        if (indexG >= nstatevector) continue;
        int indexS = threadIdx.x + k * B;
        statevectorS[indexS] = statevector[indexG];
    }

    for (int gindex = 0; gindex < ngate; gindex++) {
        int nqubit = nqubits[gindex]; 
        if (nqubit == 1) {
            vulcan::x::apply_gate_1<T, D, B>(nstatevector, statevectorS, qubits, matrices);
            __syncthreads();
            qubits += 1;
            matrices += 4;
        } else if (nqubit == 2) {
            vulcan::x::apply_gate_2<T, D, B>(nstatevector, statevectorS, qubits, matrices);
            __syncthreads();
            qubits += 2;
            matrices += 16;
        }
    }

    for (int k = 0; k < D; k++) {
        int indexG = threadIdx.x + k * B + blockIdx.x * D * B;
        if (indexG >= nstatevector) continue;
        int indexS = threadIdx.x + k * B;
        statevector[indexG] = statevectorS[indexS];
    }
}

} // namespace vulcan::x

} // namespace vulcan
