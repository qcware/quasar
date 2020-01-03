#pragma once

// ==> Vulcan Numerical Types Library <== //

/**
 * To avoid excessive code duplication, we need a set of data types which:
 *  (1) represent scalar (real) and complex types on the same footing
 *  (2) represent float or double precision on the same footing
 *  (3) have a unified set of arithmethic operations in host or device code
 *  (4) are type-punnable to real primitive data types (float or double) or
 *      complex primitive data types (std::complex<float> or
 *      std::complex<double>) for zero overhead conversion.
 * 
 * (1) and (3) preclude the direct use of std::complex, so we have elected to
 * write a simple header-only library to accomplish this objective.
 *  
 * These types support the following standard arithmetic/logic operations:
 *  ==
 *  !=
 *  +
 *  -
 *  *
 *  +=
 *  *= 
 *  conj (a - bj)
 *  abs2 (a * a + b * b)
 **/

namespace vulcan {

// => Scalar (Real) Types <= //

/**
 * Class scalar represents a real scalar type which mimics a complex type, but
 * which always has zero imaginary part. Objects of this class are fully
 * data-equivalent (type punnable) to real primitive objects of type T, e.g.,
 * if T are float or double.
 * 
 * Note that any input arguments involving imaginary components are ignored and
 * clamped to zero.
 **/ 
template <typename T>
class scalar {

public:

/**
 * Default constructor. NOTE: does *not* initialize the value to zero, as might
 * be expected. This is due to the inability to provide a non-trivial default
 * constructor for shared device memory, coupled with the requirement to define
 * a default constructor for host vector operations.
 **/
__host__ __device__ __forceinline__
scalar() {}

/**
 * Complex type constructor.
 * 
 * Params:
 *  (T) real - real part of scalar
 *  (T) imag - imaginary part of scalar (clamped to zero)
 **/ 
__host__ __device__ __forceinline__
scalar(
    T real,
    T imag) :
    real_(real)
    {} 

/**
 * Real type constructor (Imaginary part interpreted to be zero)
 * 
 * Params:
 *  (T) real - real part of scalar
 **/ 
__host__ __device__ __forceinline__
scalar(
    T real) :
    real_(real)
    {}

__host__ __device__ __forceinline__
void operator+=(const scalar<T>& other) {
    real_ += other.real();
}

__host__ __device__ __forceinline__
void operator*=(const scalar<T>& other) {
    real_ = real() * other.real();
}

/// Real part of number
__host__ __device__ __forceinline__
T real() const { return real_; }
/// Imaginary part of number (always 0.0)
__host__ __device__ __forceinline__
T imag() const { return static_cast<T>(0.0); }

/// Uniform "zero" (0.0 + 0.0j) for floating point and scalar/complex type.
static
__host__ __device__ __forceinline__
scalar<T> zero() { return scalar<T>(static_cast<T>(0.0)); }

protected:

/// Type Punnable data layout

/// Real part of number
T real_;

};

template <typename T>
__host__ __device__ __forceinline__
bool operator==(const scalar<T>& a, const scalar<T>& b) {
    return a.real() == b.real();
}

template <typename T>
__host__ __device__ __forceinline__
bool operator!=(const scalar<T>& a, const scalar<T>& b) {
    return a.real() != b.real();
}

template <typename T>
__host__ __device__ __forceinline__
scalar<T> operator+(const scalar<T>& a, const scalar<T>& b) {
    return scalar<T>(
        a.real() + b.real());
}

template <typename T>
__host__ __device__ __forceinline__
scalar<T> operator-(const scalar<T>& a, const scalar<T>& b) {
    return scalar<T>(
        a.real() - b.real());
}

template <typename T>
__host__ __device__ __forceinline__
scalar<T> operator*(const scalar<T>& a, const scalar<T>& b) {
    return scalar<T>(
        a.real() * b.real());
}

template <typename T>
__host__ __device__ __forceinline__
scalar<T> conj(const scalar<T>& a) {
    return scalar<T>(
        a.real());
}

template <typename T>
__host__ __device__ __forceinline__
scalar<T> abs2(const scalar<T>& a) {
    return scalar<T>(a.real() * a.real());
}

// => Complex Types <= //

/**
 * Class complex represents a complex type with explicit real and imaginary
 * components. This type is laid out in memory as a T representing the real
 * part of the number followed by a second T representing the imaginary part of
 * the number.  which always has zero imaginary part. Objects of this class are
 * fully data-equivalent (type punnable) to std::complex objects templated to
 * T, e.g., if T are float or double.
 **/ 
template <typename T>
class complex {

public:

/**
 * Default constructor. NOTE: does *not* initialize the value to zero, as might
 * be expected. This is due to the inability to provide a non-trivial default
 * constructor for shared device memory, coupled with the requirement to define
 * a default constructor for host vector operations.
 **/
__host__ __device__ __forceinline__
complex() {}

/**
 * Complex type constructor.
 * 
 * Params:
 *  (T) real - real part of scalar
 *  (T) imag - imaginary part of scalar 
 **/ 
__host__ __device__ __forceinline__
complex(
    T real,
    T imag) :
    real_(real),
    imag_(imag)
    {} 

/**
 * Real type constructor (Imaginary part interpreted to be zero)
 * 
 * Params:
 *  (T) real - real part of scalar
 **/ 
__host__ __device__ __forceinline__
complex(
    T real) :
    real_(real),
    imag_(static_cast<T>(0.0))
    {}

__host__ __device__ __forceinline__
void operator+=(const complex<T>& other) {
    real_ += other.real();
    imag_ += other.imag();
}

__host__ __device__ __forceinline__
void operator*=(const complex<T>& other) {
    T a = real() * other.real() - imag() * other.imag();
    T b = real() * other.imag() + imag() * other.real();
    real_ = a;
    imag_ = b;
}

/// Real part of number
__host__ __device__ __forceinline__
T real() const { return real_; }
/// Imaginary part of number
__host__ __device__ __forceinline__
T imag() const { return imag_; }

/// Uniform "zero" (0.0 + 0.0j) for floating point and scalar/complex type.
static
__host__ __device__ __forceinline__
complex<T> zero() { return complex<T>(static_cast<T>(0.0)); }

protected:

/// Type Punnable data layout

/// Real part of number
T real_;
/// Imaginary part of number
T imag_;

};

template <typename T>
__host__ __device__ __forceinline__
bool operator==(const complex<T>& a, const complex<T>& b) {
    return a.real() == b.real() && a.imag() == b.imag();
}

template <typename T>
__host__ __device__ __forceinline__
bool operator!=(const complex<T>& a, const complex<T>& b) {
    return a.real() != b.real() || a.imag() != b.imag();
}

template <typename T>
__host__ __device__ __forceinline__
complex<T> operator+(const complex<T>& a, const complex<T>& b) {
    return complex<T>(
        a.real() + b.real(),
        a.imag() + b.imag());
}

template <typename T>
__host__ __device__ __forceinline__
complex<T> operator-(const complex<T>& a, const complex<T>& b) {
    return complex<T>(
        a.real() - b.real(),
        a.imag() - b.imag());
}

template <typename T>
__host__ __device__ __forceinline__
complex<T> operator*(const complex<T>& a, const complex<T>& b) {
    return complex<T>(
        a.real() * b.real() - a.imag() * b.imag(),
        a.real() * b.imag() + a.imag() * b.real()); 
}

template <typename T>
__host__ __device__ __forceinline__
complex<T> conj(const complex<T>& a) {
    return complex<T>(
        a.real(),
        -a.imag());
}

template <typename T>
__host__ __device__ __forceinline__
complex<T> abs2(const complex<T>& a) {
    return complex<T>(a.real() * a.real() + a.imag() * a.imag());
}

// => Shorthand Notation <= //

/**
 * The following typedefs provide a shorthand for the standard numerical types
 * used in the Vulcan library. These follow the python numpy naming convention,
 * e.g., complex128 represents a complex type with a total width of 128 bits,
 * e.g., a real and imag part each of float64 (C++ double).
 **/

/// Scalar float32 (C++ float) type. Type punnable to float
typedef scalar<float> float32;
/// Scalar float64 (C++ double) type. Type punnable to double
typedef scalar<double> float64;
/// Complex float32 (C++ float) type. Type punnable to std::complex<float>
typedef complex<float> complex64;
/// Complex float64 (C++ double) type. Type punnable to std::complex<double>
typedef complex<double> complex128;
        
} // namespace vulcan
