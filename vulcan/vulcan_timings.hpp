#pragma once

#include "vulcan_gpu.hpp"
#include "vulcan_types.hpp"
#include <cuda_runtime.h>
#include <cstring>

namespace vulcan {

template <typename T>
double time_malloc_statevector(
    int nqubit)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    T* statevector = vulcan::malloc_statevector<T>(nqubit);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    vulcan::free_statevector(statevector);
    
    return 1.0E-3 * time_ms;
}

template <typename T>
double time_free_statevector(
    int nqubit)
{
    T* statevector = vulcan::malloc_statevector<T>(nqubit);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vulcan::free_statevector(statevector);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    return 1.0E-3 * time_ms;
}

template <typename T>
double time_zero_statevector(
    int nqubit)
{
    T* statevector = vulcan::malloc_statevector<T>(nqubit);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vulcan::zero_statevector<T>(nqubit, statevector);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    vulcan::free_statevector(statevector);

    return 1.0E-3 * time_ms;
}

template <typename T>
double time_set_statevector_element(
    int nqubit)
{
    T* statevector = vulcan::malloc_statevector<T>(nqubit);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vulcan::set_statevector_element(statevector, 0, T(1.0));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    vulcan::free_statevector(statevector);

    return 1.0E-3 * time_ms;
}

template <typename T>
double time_copy_statevector_to_device(
    int nqubit)
{
    T* statevector = vulcan::malloc_statevector<T>(nqubit);
    T* statevector_h = new T[(1ULL << nqubit)];
    vulcan::zero_statevector<T>(nqubit, statevector);
    std::memset(statevector_h, '\0', (1ULL << nqubit) * sizeof(T));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vulcan::copy_statevector_to_device(nqubit, statevector, statevector_h);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    vulcan::free_statevector(statevector);
    delete[] statevector_h;

    return 1.0E-3 * time_ms;
}

template <typename T>
double time_copy_statevector_to_host(
    int nqubit)
{
    T* statevector = vulcan::malloc_statevector<T>(nqubit);
    T* statevector_h = new T[(1ULL << nqubit)];
    vulcan::zero_statevector<T>(nqubit, statevector);
    std::memset(statevector_h, '\0', (1ULL << nqubit) * sizeof(T));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vulcan::copy_statevector_to_host(nqubit, statevector, statevector_h);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    vulcan::free_statevector(statevector);
    delete[] statevector_h;

    return 1.0E-3 * time_ms;
}

template <typename T>
double time_dot(
    int nqubit,
    bool same)
{
    T* statevector1 = vulcan::malloc_statevector<T>(nqubit);
    T* statevector2 = vulcan::malloc_statevector<T>(nqubit);
    vulcan::zero_statevector<T>(nqubit, statevector1);
    vulcan::zero_statevector<T>(nqubit, statevector2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vulcan::dot<T>(nqubit, statevector1, (same ? statevector1 : statevector2));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    vulcan::free_statevector(statevector1);
    vulcan::free_statevector(statevector2);

    return 1.0E-3 * time_ms;
}

template <typename T>
double time_axpby(
    int nqubit,
    T a,
    T b)
{
    T* statevector1 = vulcan::malloc_statevector<T>(nqubit);
    T* statevector2 = vulcan::malloc_statevector<T>(nqubit);
    vulcan::zero_statevector<T>(nqubit, statevector1);
    vulcan::zero_statevector<T>(nqubit, statevector2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vulcan::axpby<T>(nqubit, statevector1, statevector2, a, b);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    vulcan::free_statevector(statevector1);
    vulcan::free_statevector(statevector2);

    return 1.0E-3 * time_ms;
}

void run_timings_1(
    int min_nqubit,
    int max_nqubit)
{
    printf("==> 1D Timings <==\n\n");
    
    printf("T in [s]\n");
    printf("B in [GiB / s]\n");
    printf("\n");

    size_t multiplier = 0;

    printf("=> Malloc Statevector <=\n\n");
    multiplier = 1;
    printf("Multiplier = %zu\n", multiplier);
    printf("\n");
    printf("%2s: %12s %12s %12s %12s %12s %12s %12s %12s\n",
        "N",
        "T float32",
        "T float64",
        "T complex64",
        "T complex128",
        "B float32",
        "B float64",
        "B complex64",
        "B complex128"
        );
    for (int nqubit = min_nqubit; nqubit <= max_nqubit; nqubit++) {
        double T_float32 = time_malloc_statevector<float32>(nqubit);
        double T_float64 = time_malloc_statevector<float64>(nqubit);
        double T_complex64 = time_malloc_statevector<complex64>(nqubit);
        double T_complex128 = time_malloc_statevector<complex128>(nqubit);
        size_t ndata = (1ULL << nqubit) * multiplier;
        double B_float32 = ndata * sizeof(float32) / T_float32 / 1E9;
        double B_float64 = ndata * sizeof(float64) / T_float64 / 1E9;
        double B_complex64 = ndata * sizeof(complex64) / T_complex64 / 1E9;
        double B_complex128 = ndata * sizeof(complex128) / T_complex128 / 1E9;
        printf("%2d: %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E\n",
            nqubit,
            T_float32,
            T_float64,
            T_complex64,
            T_complex128,
            B_float32,
            B_float64,
            B_complex64,
            B_complex128
            );
    }
    printf("\n");

    printf("=> Free Statevector <=\n\n");
    multiplier = 1;
    printf("Multiplier = %zu\n", multiplier);
    printf("\n");
    printf("%2s: %12s %12s %12s %12s %12s %12s %12s %12s\n",
        "N",
        "T float32",
        "T float64",
        "T complex64",
        "T complex128",
        "B float32",
        "B float64",
        "B complex64",
        "B complex128"
        );
    for (int nqubit = min_nqubit; nqubit <= max_nqubit; nqubit++) {
        double T_float32 = time_free_statevector<float32>(nqubit);
        double T_float64 = time_free_statevector<float64>(nqubit);
        double T_complex64 = time_free_statevector<complex64>(nqubit);
        double T_complex128 = time_free_statevector<complex128>(nqubit);
        size_t ndata = (1ULL << nqubit) * multiplier;
        double B_float32 = ndata * sizeof(float32) / T_float32 / 1E9;
        double B_float64 = ndata * sizeof(float64) / T_float64 / 1E9;
        double B_complex64 = ndata * sizeof(complex64) / T_complex64 / 1E9;
        double B_complex128 = ndata * sizeof(complex128) / T_complex128 / 1E9;
        printf("%2d: %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E\n",
            nqubit,
            T_float32,
            T_float64,
            T_complex64,
            T_complex128,
            B_float32,
            B_float64,
            B_complex64,
            B_complex128
            );
    }
    printf("\n");

    printf("=> Zero Statevector <=\n\n");
    multiplier = 1;
    printf("Multiplier = %zu\n", multiplier);
    printf("\n");
    printf("%2s: %12s %12s %12s %12s %12s %12s %12s %12s\n",
        "N",
        "T float32",
        "T float64",
        "T complex64",
        "T complex128",
        "B float32",
        "B float64",
        "B complex64",
        "B complex128"
        );
    for (int nqubit = min_nqubit; nqubit <= max_nqubit; nqubit++) {
        double T_float32 = time_zero_statevector<float32>(nqubit);
        double T_float64 = time_zero_statevector<float64>(nqubit);
        double T_complex64 = time_zero_statevector<complex64>(nqubit);
        double T_complex128 = time_zero_statevector<complex128>(nqubit);
        size_t ndata = (1ULL << nqubit) * multiplier;
        double B_float32 = ndata * sizeof(float32) / T_float32 / 1E9;
        double B_float64 = ndata * sizeof(float64) / T_float64 / 1E9;
        double B_complex64 = ndata * sizeof(complex64) / T_complex64 / 1E9;
        double B_complex128 = ndata * sizeof(complex128) / T_complex128 / 1E9;
        printf("%2d: %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E\n",
            nqubit,
            T_float32,
            T_float64,
            T_complex64,
            T_complex128,
            B_float32,
            B_float64,
            B_complex64,
            B_complex128
            );
    }
    printf("\n");

    printf("=> Set Statevector Element <=\n\n");
    multiplier = 1;
    printf("Multiplier = %zu\n", multiplier);
    printf("\n");
    printf("%2s: %12s %12s %12s %12s %12s %12s %12s %12s\n",
        "N",
        "T float32",
        "T float64",
        "T complex64",
        "T complex128",
        "B float32",
        "B float64",
        "B complex64",
        "B complex128"
        );
    for (int nqubit = min_nqubit; nqubit <= max_nqubit; nqubit++) {
        double T_float32 = time_set_statevector_element<float32>(nqubit);
        double T_float64 = time_set_statevector_element<float64>(nqubit);
        double T_complex64 = time_set_statevector_element<complex64>(nqubit);
        double T_complex128 = time_set_statevector_element<complex128>(nqubit);
        size_t ndata = (1ULL << nqubit) * multiplier;
        double B_float32 = ndata * sizeof(float32) / T_float32 / 1E9;
        double B_float64 = ndata * sizeof(float64) / T_float64 / 1E9;
        double B_complex64 = ndata * sizeof(complex64) / T_complex64 / 1E9;
        double B_complex128 = ndata * sizeof(complex128) / T_complex128 / 1E9;
        printf("%2d: %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E\n",
            nqubit,
            T_float32,
            T_float64,
            T_complex64,
            T_complex128,
            B_float32,
            B_float64,
            B_complex64,
            B_complex128
            );
    }
    printf("\n");

    printf("=> Copy Statevector to Device <=\n\n");
    multiplier = 1;
    printf("Multiplier = %zu\n", multiplier);
    printf("\n");
    printf("%2s: %12s %12s %12s %12s %12s %12s %12s %12s\n",
        "N",
        "T float32",
        "T float64",
        "T complex64",
        "T complex128",
        "B float32",
        "B float64",
        "B complex64",
        "B complex128"
        );
    for (int nqubit = min_nqubit; nqubit <= max_nqubit; nqubit++) {
        double T_float32 = time_copy_statevector_to_device<float32>(nqubit);
        double T_float64 = time_copy_statevector_to_device<float64>(nqubit);
        double T_complex64 = time_copy_statevector_to_device<complex64>(nqubit);
        double T_complex128 = time_copy_statevector_to_device<complex128>(nqubit);
        size_t ndata = (1ULL << nqubit) * multiplier;
        double B_float32 = ndata * sizeof(float32) / T_float32 / 1E9;
        double B_float64 = ndata * sizeof(float64) / T_float64 / 1E9;
        double B_complex64 = ndata * sizeof(complex64) / T_complex64 / 1E9;
        double B_complex128 = ndata * sizeof(complex128) / T_complex128 / 1E9;
        printf("%2d: %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E\n",
            nqubit,
            T_float32,
            T_float64,
            T_complex64,
            T_complex128,
            B_float32,
            B_float64,
            B_complex64,
            B_complex128
            );
    }
    printf("\n");

    printf("=> Copy Statevector to Host <=\n\n");
    multiplier = 1;
    printf("Multiplier = %zu\n", multiplier);
    printf("\n");
    printf("%2s: %12s %12s %12s %12s %12s %12s %12s %12s\n",
        "N",
        "T float32",
        "T float64",
        "T complex64",
        "T complex128",
        "B float32",
        "B float64",
        "B complex64",
        "B complex128"
        );
    for (int nqubit = min_nqubit; nqubit <= max_nqubit; nqubit++) {
        double T_float32 = time_copy_statevector_to_host<float32>(nqubit);
        double T_float64 = time_copy_statevector_to_host<float64>(nqubit);
        double T_complex64 = time_copy_statevector_to_host<complex64>(nqubit);
        double T_complex128 = time_copy_statevector_to_host<complex128>(nqubit);
        size_t ndata = (1ULL << nqubit) * multiplier;
        double B_float32 = ndata * sizeof(float32) / T_float32 / 1E9;
        double B_float64 = ndata * sizeof(float64) / T_float64 / 1E9;
        double B_complex64 = ndata * sizeof(complex64) / T_complex64 / 1E9;
        double B_complex128 = ndata * sizeof(complex128) / T_complex128 / 1E9;
        printf("%2d: %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E\n",
            nqubit,
            T_float32,
            T_float64,
            T_complex64,
            T_complex128,
            B_float32,
            B_float64,
            B_complex64,
            B_complex128
            );
    }
    printf("\n");

    printf("=> Dot <A|A> <=\n\n");
    multiplier = 1;
    printf("Multiplier = %zu\n", multiplier);
    printf("\n");
    printf("%2s: %12s %12s %12s %12s %12s %12s %12s %12s\n",
        "N",
        "T float32",
        "T float64",
        "T complex64",
        "T complex128",
        "B float32",
        "B float64",
        "B complex64",
        "B complex128"
        );
    for (int nqubit = min_nqubit; nqubit <= max_nqubit; nqubit++) {
        double T_float32 = time_dot<float32>(nqubit, true);
        double T_float64 = time_dot<float64>(nqubit, true);
        double T_complex64 = time_dot<complex64>(nqubit, true);
        double T_complex128 = time_dot<complex128>(nqubit, true);
        size_t ndata = (1ULL << nqubit) * multiplier;
        double B_float32 = ndata * sizeof(float32) / T_float32 / 1E9;
        double B_float64 = ndata * sizeof(float64) / T_float64 / 1E9;
        double B_complex64 = ndata * sizeof(complex64) / T_complex64 / 1E9;
        double B_complex128 = ndata * sizeof(complex128) / T_complex128 / 1E9;
        printf("%2d: %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E\n",
            nqubit,
            T_float32,
            T_float64,
            T_complex64,
            T_complex128,
            B_float32,
            B_float64,
            B_complex64,
            B_complex128
            );
    }
    printf("\n");
    
    printf("=> Dot <A|B> <=\n\n");
    multiplier = 2;
    printf("Multiplier = %zu\n", multiplier);
    printf("\n");
    printf("%2s: %12s %12s %12s %12s %12s %12s %12s %12s\n",
        "N",
        "T float32",
        "T float64",
        "T complex64",
        "T complex128",
        "B float32",
        "B float64",
        "B complex64",
        "B complex128"
        );
    for (int nqubit = min_nqubit; nqubit <= max_nqubit; nqubit++) {
        double T_float32 = time_dot<float32>(nqubit, false);
        double T_float64 = time_dot<float64>(nqubit, false);
        double T_complex64 = time_dot<complex64>(nqubit, false);
        double T_complex128 = time_dot<complex128>(nqubit, false);
        size_t ndata = (1ULL << nqubit) * multiplier;
        double B_float32 = ndata * sizeof(float32) / T_float32 / 1E9;
        double B_float64 = ndata * sizeof(float64) / T_float64 / 1E9;
        double B_complex64 = ndata * sizeof(complex64) / T_complex64 / 1E9;
        double B_complex128 = ndata * sizeof(complex128) / T_complex128 / 1E9;
        printf("%2d: %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E\n",
            nqubit,
            T_float32,
            T_float64,
            T_complex64,
            T_complex128,
            B_float32,
            B_float64,
            B_complex64,
            B_complex128
            );
    }
    printf("\n");
    
    printf("=> AXPBY (a=1.0, b=1.0) <=\n\n");
    multiplier = 3;
    printf("Multiplier = %zu\n", multiplier);
    printf("\n");
    printf("%2s: %12s %12s %12s %12s %12s %12s %12s %12s\n",
        "N",
        "T float32",
        "T float64",
        "T complex64",
        "T complex128",
        "B float32",
        "B float64",
        "B complex64",
        "B complex128"
        );
    for (int nqubit = min_nqubit; nqubit <= max_nqubit; nqubit++) {
        double T_float32 = time_axpby<float32>(nqubit, float32(1.0), float32(1.0));
        double T_float64 = time_axpby<float64>(nqubit, float64(1.0), float64(1.0));
        double T_complex64 = time_axpby<complex64>(nqubit, complex64(1.0), complex64(1.0));
        double T_complex128 = time_axpby<complex128>(nqubit, complex128(1.0), complex128(1.0));
        size_t ndata = (1ULL << nqubit) * multiplier;
        double B_float32 = ndata * sizeof(float32) / T_float32 / 1E9;
        double B_float64 = ndata * sizeof(float64) / T_float64 / 1E9;
        double B_complex64 = ndata * sizeof(complex64) / T_complex64 / 1E9;
        double B_complex128 = ndata * sizeof(complex128) / T_complex128 / 1E9;
        printf("%2d: %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E\n",
            nqubit,
            T_float32,
            T_float64,
            T_complex64,
            T_complex128,
            B_float32,
            B_float64,
            B_complex64,
            B_complex128
            );
    }
    printf("\n");
    
    printf("=> AXPBY (a=1.0, b=0.0) <=\n\n");
    multiplier = 2;
    printf("Multiplier = %zu\n", multiplier);
    printf("\n");
    printf("%2s: %12s %12s %12s %12s %12s %12s %12s %12s\n",
        "N",
        "T float32",
        "T float64",
        "T complex64",
        "T complex128",
        "B float32",
        "B float64",
        "B complex64",
        "B complex128"
        );
    for (int nqubit = min_nqubit; nqubit <= max_nqubit; nqubit++) {
        double T_float32 = time_axpby<float32>(nqubit, float32(1.0), float32(0.0));
        double T_float64 = time_axpby<float64>(nqubit, float64(1.0), float64(0.0));
        double T_complex64 = time_axpby<complex64>(nqubit, complex64(1.0), complex64(0.0));
        double T_complex128 = time_axpby<complex128>(nqubit, complex128(1.0), complex128(0.0));
        size_t ndata = (1ULL << nqubit) * multiplier;
        double B_float32 = ndata * sizeof(float32) / T_float32 / 1E9;
        double B_float64 = ndata * sizeof(float64) / T_float64 / 1E9;
        double B_complex64 = ndata * sizeof(complex64) / T_complex64 / 1E9;
        double B_complex128 = ndata * sizeof(complex128) / T_complex128 / 1E9;
        printf("%2d: %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E %12.3E\n",
            nqubit,
            T_float32,
            T_float64,
            T_complex64,
            T_complex128,
            B_float32,
            B_float64,
            B_complex64,
            B_complex128
            );
    }
    printf("\n");
}

} // namespace vulcan
