#pragma once

namespace vulcan {

template <typename T>
class scalar {

public:

__host__ __device__ __forceinline__
scalar() {}

__host__ __device__ __forceinline__
scalar(
    T real,
    T imag) :
    real_(real)
    {} 

__host__ __device__ __forceinline__
scalar(
    T real) :
    real_(real)
    {}

__host__ __device__ __forceinline__
void operator=(const T& x) {
    real_ = x;
}

__host__ __device__ __forceinline__
void operator+=(const scalar<T>& other) {
    real_ += other.real();
}

__host__ __device__ __forceinline__
void operator*=(const scalar<T>& other) {
    real_ = real() * other.real();
}

__host__ __device__ __forceinline__
T real() const { return real_; }
__host__ __device__ __forceinline__
T imag() const { return static_cast<T>(0.0); }

static
__host__ __device__ __forceinline__
scalar<T> zero() { return scalar<T>(static_cast<T>(0.0)); }

protected:

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
scalar<T> conj(const scalar<T>& a) {
    return scalar<T>(
        a.real());
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
scalar<T> abs2(const scalar<T>& a) {
    return scalar<T>(a.real() * a.real());
}


template <typename T>
class complex {

public:
__host__ __device__ __forceinline__
complex() {}

__host__ __device__ __forceinline__
complex(
    T real,
    T imag) :
    real_(real),
    imag_(imag)
    {} 

__host__ __device__ __forceinline__
complex(
    T real) :
    real_(real),
    imag_(static_cast<T>(0.0))
    {}

__host__ __device__ __forceinline__
void operator=(const T& x) {
    real_ = x;
    imag_ = static_cast<T>(0.0);
}

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

__host__ __device__ __forceinline__
T real() const { return real_; }
__host__ __device__ __forceinline__
T imag() const { return imag_; }

static
__host__ __device__ __forceinline__
complex<T> zero() { return complex<T>(static_cast<T>(0.0)); }

protected:

T real_;
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
complex<T> conj(const complex<T>& a) {
    return complex<T>(
        a.real(),
        -a.imag());
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
complex<T> abs2(const complex<T>& a) {
    return complex<T>(a.real() * a.real() + a.imag() * a.imag());
}

typedef scalar<float> float32;
typedef scalar<double> float64;
typedef complex<float> complex64;
typedef complex<double> complex128;
        
} // namespace vulcan
