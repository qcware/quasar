#pragma once

#include <cuda_runtime.h>
#include <utility>
#include <stdexcept>

// ==> Project Vulcan: Low-Level C++11 GPU API <== // 

/**
 * Project Vulcan provides a C++11 API for CUDA-accelerated quantum circuit
 * simulations. This particular part of the API provides low-level BLAS1/2-like
 * API for manipulating statevectors that live in GPU device memory.
 *
 * Conventions:
 *  - All statevectors are of shape 2**nqubit.
 *  - Qubits are ordered *opposite* of Mike and Ike for convenience with bit
 *     fiddling. E.g., in ket |0001>, the 0th qubit is in the |1> state.
 *  - Signed integers are used to address statevectors, limiting the maximum
 *     possible number of qubits to 31 (MAX_NQUBIT).
 *  - All BLAS1/2-like functions support a/b scalar parameters like axpby,
 *     e.g., all operate as:
 *
 *         statevector2 = a * op(statevector1) + b * statevector2
 *     
 *     If b is 0.0, statevector2 is not loaded, providing a modest performance
 *     improvement.
 *  - Unless otherwise noted, all BLAS1/2-like functions support self
 *      assignment, e.g., the function must return correctly if statevector1
 *      and statevector2 are identical.  This can be useful for in-place gate
 *      applications, saving one statevector worth of device memory.
 *  - All functions are templated for type T. If real types are used any
 *      imaginary components of the computation will be truncated to zero
 *      throughout, possibly leading to formally incorrect results. The user is
 *      responsible for verifying formal correctness when using real types.
 *  - Special numerical types are necessary for interoperability between host
 *      and device code. These are defined in vulcan_types.hpp. Unless
 *      otherwise indicated, the templated functions below are instantiated for
 *      the following types T:
 *
 *           vulcan::float32
 *           vulcan::float64
 *           vulcan::complex64
 *           vulcan::complex128
 **/
    
namespace vulcan {

namespace gpu {

// => Utility <= //

// Maximum number of qubits the library can handle
#define MAX_NQUBIT 31

// => BLAS 1 Operations <= //

/**
 * Malloc (but do not initialize) a device statevector of 2**nqubit T elements.
 *
 * Params:
 *  nqubit (int) - number of qubits in statevector
 * Returns:
 *  (T*) - device pointer to allocated statevector
 **/
template <typename T>
T* malloc_statevector(int nqubit)
{
    if (nqubit > MAX_NQUBIT) {
        throw std::runtime_error("nqubit too large");
    }
    T* statevector_d;
    cudaError_t error = cudaMalloc(&statevector_d, (1ULL << nqubit) * sizeof(T));
    if (error != cudaSuccess) {
        throw std::runtime_error("Failure in cudaMalloc");
    }
    return statevector_d;
}

/**
 * Free a device statevector.
 *
 * Params:
 *  statevector_d (T*) - device pointer to allocated statevector
 * Result:
 *  The pointer is freed
 **/
template <typename T>
T* free_statevector(T* statevector_d)
{
    cudaFree(statevector_d);
    return statevector_d;
}

/**
 * Set all entries of a statevector to 0.0 (the null state |>).
 *
 * Params:
 *  nqubit (int) - number of qubits in statevector
 *  statevector_d (T*) - device pointer to allocated statevector
 * Result:
 *  The device statevector is filled with 0.0 of type T
 **/
template <typename T>
void zero_statevector(
    int nqubit,
    T* statevector_d)
{
    cudaMemset(statevector_d, '\0', (1ULL << nqubit) * sizeof(T));
}

/**
 * Set a given ket (a single configuration) in a statevector to value.
 *
 * E.g., setting a previously zeroed statevector's 0-th ket to 1.0 prepares the
 * all-zeros state |000...>.
 *
 * Params:
 *  statevector_d (T*) - device pointer to allocated statevector
 *  ket (int) - index of ket to set value for
 *  value (T) - value to set
 * Result:
 *  The statevector entry at the ket-th index is set to value.
 **/
template <typename T>
void set_statevector_element(
    T* statevector_d,
    int ket,
    T value)
{
    cudaMemcpy(statevector_d + ket, &value, sizeof(T), cudaMemcpyHostToDevice);
}

/**
 * Get the value of a given ket (a single configuration) in a statevector.
 *
 * Params:
 *  statevector_d (T*) - device pointer to allocated statevector
 *  ket (int) - index of ket to set value for
 * Returns:
 *  (T) - the value of the statevector at the ket index 
 *
 **/
template <typename T>
T get_statevector_element(
    T* statevector_d,
    int ket)
{
    T value;
    cudaMemcpy(&value, statevector_d + ket, sizeof(T), cudaMemcpyDeviceToHost);
    return value;
}

/**
 * Initialize a statevector to the all-zeros state |000....>, corresponding to
 * a statevector of [1.0, 0.0, 0.0, ...].
 *
 * Params:
 *  nqubit (int) - number of qubits in statevector
 *  statevector_d (T*) - device pointer to allocated statevector
 * Result:
 *  The statevector is initialized to [1.0, 0.0, 0.0, ...]
 **/
template <typename T>
void initialize_zero_ket(
    int nqubit,
    T* statevector_d)
{
    zero_statevector(nqubit, statevector_d);
    set_statevector_element(statevector_d, 0, T(1.0));
}

/**
 * Copy a statevector from host memory to device memory.
 *
 * Params:
 *  nqubit (int) - number of qubits in statevector
 *  statevector_d (T*) - device pointer to allocated statevector
 *  statevector_h (T*) - host pointer to allocated statevector
 * Result:
 *  statevector_d is overwritten with the contents of statevector_h
 **/
template <typename T>
void copy_statevector_to_device(
    int nqubit,
    T* statevector_d,
    const T* statevector_h)
{
    cudaMemcpy(statevector_d, statevector_h, (1ULL << nqubit) * sizeof(T), cudaMemcpyHostToDevice);
}

/**
 * Copy a statevector from device memory to host memory.
 *
 * Params:
 *  nqubit (int) - number of qubits in statevector
 *  statevector_d (T*) - device pointer to allocated statevector
 *  statevector_h (T*) - host pointer to allocated statevector
 * Result:
 *  statevector_h is overwritten with the contents of statevector_d
 **/
template <typename T>
void copy_statevector_to_host(
    int nqubit,
    T* statevector_d,
    T* statevector_h)
{
    cudaMemcpy(statevector_h, statevector_d, (1ULL << nqubit) * sizeof(T), cudaMemcpyDeviceToHost);
}

/**
 * Malloc and initialize a device statevector of 2**nqubit T elements.
 *
 * Params:
 *  nqubit (int) - number of qubits in statevector
 *  statevector_h (T*) - point to host statevector if an input statevector is
 *   to be copied, or nullptr if the |0000...> state is to be initialized.
 * Returns:
 *  (T*) - device pointer to allocated statevector
 **/
template <typename T>
T* malloc_and_initialize_statevector(
    int nqubit,
    const T* statevector_h)
{
    T* statevector_d = vulcan::gpu::malloc_statevector<T>(nqubit);
    if (statevector_h == nullptr) {
	vulcan::gpu::initialize_zero_ket(nqubit, statevector_d);
    } else {
        vulcan::gpu::copy_statevector_to_device(nqubit, statevector_d, statevector_h);
    }

    return statevector_d;
}

/**
 * Compute the dot product <statevector1|statevector2>:
 *
 * D = \sum_I conj(statevector1_d_I) * statevector2_d_I
 *
 * Params:
 *  nqubit (int) - number of qubits in statevector
 *  statevector1_d (T*) - device pointer to allocated (bra) statevector
 *  statevector2_d (T*) - device pointer to allocated (ket) statevector
 * Returns:
 *  (T) - resultant dot product
 **/
template <typename T>
T dot(
    int nqubit,
    T* statevector1_d,
    T* statevector2_d);

/**
 * Compute the weighted linear combination of statevectors, storing the result
 * in statevector2:
 *
 *    statevector2 = a * statevector1 + b * statevector2
 *  
 * Note that axpby(nqubit, statevector1, statevector2) copies statevector1 to
 * statevector2.
 *
 * Params:
 *  nqubit (int) - number of qubits in statevector
 *  statevector1_d (T*) - device pointer to allocated input statevector
 *  statevector2_d (T*) - device pointer to allocated output statevector
 *  a (T) - multiplier of statevector1
 *  b (T) - multiplier of statevector2. If b is 0.0, statevector2 is not read,
 *   moderately increasing performance.
 * Result:
 *  statevector2_d is overwritten with the result of the operation
 **/
template <typename T>
void axpby(
    int nqubit,
    T* statevector1_d,
    T* statevector2_d,
    T a = T(1.0),
    T b = T(0.0));

// => Gate Operations <= //

/**
 * Apply a 1-qubit gate to statevector1, storing the result in statevector2:
 *
 *    statevector2 = a * G1(statevector1) + b * statevector2
 *
 * Params:
 *  nqubit (int) - number of qubits in statevector
 *  statevector1_d (T*) - device pointer to allocated input statevector
 *  statevector2_d (T*) - device pointer to allocated output statevector
 *  qubitA (int) - the index of the qubit to apply the gate at
 *  O00 (T) - element of the gate operator
 *  O01 (T) - element of the gate operator
 *  O10 (T) - element of the gate operator
 *  O11 (T) - element of the gate operator
 *  a (T) - multiplier of statevector1
 *  b (T) - multiplier of statevector2. If b is 0.0, statevector2 is not read,
 *   moderately increasing performance.
 * Result:
 *  statevector2_d is overwritten with the result of the operation
 **/
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
    T a = T(1.0),
    T b = T(0.0));

/**
 * Apply a 2-qubit gate to statevector1, storing the result in statevector2:
 *
 *    statevector2 = a * G2(statevector1) + b * statevector2
 *
 * Params:
 *  nqubit (int) - number of qubits in statevector
 *  statevector1_d (T*) - device pointer to allocated input statevector
 *  statevector2_d (T*) - device pointer to allocated output statevector
 *  qubitA (int) - the first index of the qubit to apply the gate at
 *  qubitB (int) - the second index of the qubit to apply the gate at
 *  O00 (T) - element of the gate operator
 *  O01 (T) - element of the gate operator
 *  O02 (T) - element of the gate operator
 *  O03 (T) - element of the gate operator
 *  O10 (T) - element of the gate operator
 *  O11 (T) - element of the gate operator
 *  O12 (T) - element of the gate operator
 *  O13 (T) - element of the gate operator
 *  O20 (T) - element of the gate operator
 *  O21 (T) - element of the gate operator
 *  O22 (T) - element of the gate operator
 *  O23 (T) - element of the gate operator
 *  O30 (T) - element of the gate operator
 *  O31 (T) - element of the gate operator
 *  O32 (T) - element of the gate operator
 *  O33 (T) - element of the gate operator
 *  a (T) - multiplier of statevector1
 *  b (T) - multiplier of statevector2. If b is 0.0, statevector2 is not read,
 *   moderately increasing performance.
 * Result:
 *  statevector2_d is overwritten with the result of the operation
 **/
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
    T a = T(1.0),
    T b = T(0.0));

// => Specialized Pauli Operators <= //

template <typename T>
void apply_pauli(
    int nqubit,
    T* statevector1_d,
    T* statevector2_d,
    int Xmask,
    int Ymask,
    int Zmask,
    T a = T(1.0),
    T b = T(0.0));

// => Random Quantum Measurement Operations <= //

/**
 * Place the real absolute square of statevector1_d in statevector2_d:
 *  statevector2_d = real(abs**2(statevector1_d))
 * 
 * Separate types T and U are used to permit downcasting from complex
 * statevector1_d to real statevector2_d.
 * 
 * Note that self-assignment is supported if T and U are the same real type.
 * 
 * The following combinations of types are instantiated:
 *   abs2<float32, float32> (supports self-assignment)
 *   abs2<float64, float64> (supports self-assignment)
 *   abs2<complex64, float32>
 *   abs2<complex128, float64>
 * 
 * Params:
 *  nqubit (int) - number of qubits in statevector
 *  statevector1_d (T*) - device pointer to allocated input statevector
 *  statevector2_d (U*) - device pointer to allocated output statevector
 * Result:
 *  statevector2_d is overwritten with the result of the operation
 **/
template <typename T, typename U>
void abs2(
    int nqubit,
    T* statevector1_d,
    U* statevector2_d);

/**
 * Form the exclusive cumulative sum of statevector_d in place. 
 * 
 * (statevector_d)_I = sum_{J = 0}^{I-1} (statevector_d)_J
 * 
 * Note that the final value of (statevector_d)_0 is 0.0.
 * The total sum is returned to the host to preserve the full information
 * content of statevector.
 * 
 * Butterfly summation is used for speed and for enhanced numerical stability
 * 
 * The following combinations of types are instantiated:
 *   cumsum<float32> 
 *   cumsum<float64> 
 *
 * Params:
 *  nqubit (int) - number of qubits in statevector
 *  statevector_d (T*) - device pointer to allocated input/output statevector
 * Result:
 *  statevector_d is updated with the cumulative sum.
 * Returns:
 *  (T) the total sum of all elements in statevector_d
 **/ 
template <typename T>
T cumsum(
    int nqubit,
    T* statevector_d);

/**
 * Transformation from (pseudo)random numbers uniformly sampled in [0.0, 1.0)
 * to random configurations (kets) sampled from the discrete probability
 * distribution of a statevector.
 * 
 * The algorithm works by determining, for each random, the ket I for which:
 *  
 *  (cumsum_d)_I <= random * sum < cumsum_(I + 1)
 * 
 * Note that we do not generate the uniform random values within this function.
 * These are specified by the user as an input argument to this function, and
 * may be generated as desired on the CPU or GPU and then placed in global GPU
 * memory.
 * 
 * Note that the user assumes the burden of checking the numerical quality of
 * the cover of the random sampling, particularly with large statevectors with
 * nearly uniform probability amplitudes in low precisions. For instance,
 * sampling from the uniform |+> = \prod_I (H_I) |0> state with 32-bit
 * precision floating point types in statevectors with >~ 22 qubits is somewhat
 * numerically dicey: the returned samples will nicely extend across the domain
 * from |000...> to |111...>, but not all kets will be reachable by this random
 * sampling function. This is because there is not enough numerical precision
 * in the mantissa of 32-bit precision floating point to distinguish the
 * least-significant bits of the kets (a problem in both the uniqueness of the
 * cumsum values and the entropy of the uniform random number). Sampling from
 * the uniform |+> state is a worst-case example - quantum algorithms that
 * concentrate amplitudes are practically found to perform quite well in
 * float32 across all qubit sizes permitted within the library.
 *
 * The following combinations of types are instantiated:
 *   measure<float32> 
 *   measure<float64> 
 *
 * Params:
 *  nqubit (int) - number of qubits in statevector
 *  cumsum_d (T*) - device pointer to input statevector holding cumulative sum
 *   of abs2 of statevector (the cumulative probability distribution)
 *  sum (T) - value of total sum of abs2 of statevector returned by cumsum.
 *   This is usually very close to 1.0, and is used here to mitigate minor
 *   roundoff error that may have occurred in cumsum.
 *  nmeasurement (int) - number of measurements to perform
 *  randoms (T*) - device pointer of input uniform random numbers in the range
 *   [0, 1). Size nmeasurement.
 *  measurements (int) - device pointer to the output register of random ket
 *   indices. Size nmeasurement.
 * Result:
 *  measurements is overwritten with the nmeasurement indices of the random
 *   kets.
 **/
template <typename T>
void measure(
    int nqubit,
    T* cumsum_d,
    T sum,
    int nmeasurement,
    T* randoms,
    int* measurements);

} // namespace vulcan::gpu

} // namespace vulcan
