#include "vulcan.hpp"
#include "vulcan_types.hpp"
#include "vulcan_timings.hpp"
#include "device_properties.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <complex>
#include <cstdio>

namespace py = pybind11;

namespace vulcan {

typedef py::array_t<float, py::array::c_style | py::array::forcecast> np_float32;
typedef py::array_t<double, py::array::c_style | py::array::forcecast> np_float64;
typedef py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> np_complex64;
typedef py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> np_complex128;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> np_int32;

template <typename U>
U* np_pointer(
    int nqubit,
    const py::array_t<U>& statevector)
{
    if (statevector.size() == 0) return nullptr;
    if (statevector.ndim() != 1) throw std::runtime_error("statevector must have ndim == 1");
    if (statevector.shape(0) != (1ULL << nqubit)) throw std::runtime_error("statevector must be shape (2**nqubit,)");
    
    py::buffer_info buffer = statevector.request();
    U* ptr = (U*) buffer.ptr;
    return ptr;
}

template <typename T, typename U> 
py::array_t<U> py_run_statevector(
    const Circuit<T>& circuit,
    const py::array_t<U>& statevector,
    bool compressed)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");

    py::array_t<U> result = py::array_t<U>(1ULL << circuit.nqubit());
    py::buffer_info buffer = result.request();
    U* ptr = (U*) buffer.ptr;

    U* ptr2 = np_pointer(circuit.nqubit(), statevector);

    vulcan::run_statevector<T>(circuit, (T*) ptr2, (T*) ptr, compressed);

    return result;
}

np_float32 py_run_statevector_float32(
    const Circuit<float32>& circuit,
    const np_float32& statevector,
    bool compressed)
{
    return py_run_statevector<float32, float>(circuit, statevector, compressed);
}

np_float64 py_run_statevector_float64(
    const Circuit<float64>& circuit,
    const np_float64& statevector,
    bool compressed)
{
    return py_run_statevector<float64, double>(circuit, statevector, compressed);
}

np_complex64 py_run_statevector_complex64(
    const Circuit<complex64>& circuit,
    const np_complex64& statevector,
    bool compressed)
{
    return py_run_statevector<complex64, std::complex<float>>(circuit, statevector, compressed);
}

np_complex128 py_run_statevector_complex128(
    const Circuit<complex128>& circuit,
    const np_complex128& statevector,
    bool compressed)
{
    return py_run_statevector<complex128, std::complex<double>>(circuit, statevector, compressed);
}

template <typename T, typename U> 
py::array_t<U> py_run_pauli_sigma(
    const Pauli<T>& pauli,
    const py::array_t<U>& statevector,
    bool compressed)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");

    if (statevector.ndim() != 1) throw std::runtime_error("statevector must have ndim == 1");
    if (statevector.shape(0) != (1ULL << pauli.nqubit())) throw std::runtime_error("statevector must be shape (2**nqubit,)");
    
    py::buffer_info buffer1 = statevector.request();
    U* ptr1 = (U*) buffer1.ptr;

    py::array_t<U> result = py::array_t<U>(1ULL << pauli.nqubit());
    py::buffer_info buffer2 = result.request();
    U* ptr2 = (U*) buffer2.ptr;

    vulcan::run_pauli_sigma<T>(pauli, (T*) ptr1, (T*) ptr2, compressed);

    return result;
}

np_float32 py_run_pauli_sigma_float32(
    const Pauli<float32>& pauli,
    const np_float32& statevector,
    bool compressed)
{
    return py_run_pauli_sigma<float32, float>(pauli, statevector, compressed);
}

np_float64 py_run_pauli_sigma_float64(
    const Pauli<float64>& pauli,
    const np_float64& statevector,
    bool compressed)
{
    return py_run_pauli_sigma<float64, double>(pauli, statevector, compressed);
}

np_complex64 py_run_pauli_sigma_complex64(
    const Pauli<complex64>& pauli,
    const np_complex64& statevector,
    bool compressed)
{
    return py_run_pauli_sigma<complex64, std::complex<float>>(pauli, statevector, compressed);
}

np_complex128 py_run_pauli_sigma_complex128(
    const Pauli<complex128>& pauli,
    const np_complex128& statevector,
    bool compressed)
{
    return py_run_pauli_sigma<complex128, std::complex<double>>(pauli, statevector, compressed);
}

template <typename T, typename U>
Pauli<T> py_run_pauli_expectation(
    const Circuit<T>& circuit,
    const Pauli<T>& pauli,
    const py::array_t<U>& statevector,
    bool compressed)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");

    U* ptr2 = np_pointer(circuit.nqubit(), statevector);

    return vulcan::run_pauli_expectation<T>(circuit, pauli, (T*) ptr2, compressed);
}

Pauli<float32> py_run_pauli_expectation_float32(
    const Circuit<float32>& circuit,
    const Pauli<float32>& pauli,
    const np_float32& statevector,
    bool compressed)
{
    return py_run_pauli_expectation<float32, float>(circuit, pauli, statevector, compressed);
}

Pauli<float64> py_run_pauli_expectation_float64(
    const Circuit<float64>& circuit,
    const Pauli<float64>& pauli,
    const np_float64& statevector,
    bool compressed)
{
    return py_run_pauli_expectation<float64, double>(circuit, pauli, statevector, compressed);
}

Pauli<complex64> py_run_pauli_expectation_complex64(
    const Circuit<complex64>& circuit,
    const Pauli<complex64>& pauli,
    const np_complex64& statevector,
    bool compressed)
{
    return py_run_pauli_expectation<complex64, std::complex<float>>(circuit, pauli, statevector, compressed);
}

Pauli<complex128> py_run_pauli_expectation_complex128(
    const Circuit<complex128>& circuit,
    const Pauli<complex128>& pauli,
    const np_complex128& statevector,
    bool compressed)
{
    return py_run_pauli_expectation<complex128, std::complex<double>>(circuit, pauli, statevector, compressed);
}

template <typename T, typename U> 
U py_run_pauli_expectation_value(
    const Circuit<T>& circuit,
    const Pauli<T>& pauli,
    const py::array_t<U>& statevector,
    bool compressed)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");

    U* ptr2 = np_pointer(circuit.nqubit(), statevector);

    T val = vulcan::run_pauli_expectation_value<T>(circuit, pauli, (T*) ptr2, compressed);

    U val2;
    std::memcpy(&val2, &val, sizeof(T));

    return val2;
}

float py_run_pauli_expectation_value_float32(
    const Circuit<float32>& circuit,
    const Pauli<float32>& pauli,
    const np_float32& statevector,
    bool compressed)
{
    return py_run_pauli_expectation_value<float32, float>(circuit, pauli, statevector, compressed);
}

double py_run_pauli_expectation_value_float64(
    const Circuit<float64>& circuit,
    const Pauli<float64>& pauli,
    const np_float64& statevector,
    bool compressed)
{
    return py_run_pauli_expectation_value<float64, double>(circuit, pauli, statevector, compressed);
}

std::complex<float> py_run_pauli_expectation_value_complex64(
    const Circuit<complex64>& circuit,
    const Pauli<complex64>& pauli,
    const np_complex64& statevector,
    bool compressed)
{
    return py_run_pauli_expectation_value<complex64, std::complex<float>>(circuit, pauli, statevector, compressed);
}

std::complex<double> py_run_pauli_expectation_value_complex128(
    const Circuit<complex128>& circuit,
    const Pauli<complex128>& pauli,
    const np_complex128& statevector,
    bool compressed)
{
    return py_run_pauli_expectation_value<complex128, std::complex<double>>(circuit, pauli, statevector, compressed);
}

template <typename T, typename U> 
py::array_t<U> py_run_pauli_expectation_value_gradient(
    const Circuit<T>& circuit,
    const Pauli<T>& pauli,
    const py::array_t<U>& statevector,
    bool compressed)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");

    U* ptr2 = np_pointer(circuit.nqubit(), statevector);

    std::vector<T> val = vulcan::run_pauli_expectation_value_gradient<T>(circuit, pauli, (T*) ptr2, compressed);

    py::array_t<U> result = py::array_t<U>(val.size());
    py::buffer_info buffer3 = result.request();
    U* ptr3 = (U*) buffer3.ptr;

    ::memcpy(ptr3, val.data(), val.size() * sizeof(T));

    return result;
}

np_float32 py_run_pauli_expectation_value_gradient_float32(
    const Circuit<float32>& circuit,
    const Pauli<float32>& pauli,
    const np_float32& statevector,
    bool compressed)
{
    return py_run_pauli_expectation_value_gradient<float32, float>(circuit, pauli, statevector, compressed);
}

np_float64 py_run_pauli_expectation_value_gradient_float64(
    const Circuit<float64>& circuit,
    const Pauli<float64>& pauli,
    const np_float64& statevector,
    bool compressed)
{
    return py_run_pauli_expectation_value_gradient<float64, double>(circuit, pauli, statevector, compressed);
}

np_complex64 py_run_pauli_expectation_value_gradient_complex64(
    const Circuit<complex64>& circuit,
    const Pauli<complex64>& pauli,
    const np_complex64& statevector,
    bool compressed)
{
    return py_run_pauli_expectation_value_gradient<complex64, std::complex<float>>(circuit, pauli, statevector, compressed);
}

np_complex128 py_run_pauli_expectation_value_gradient_complex128(
    const Circuit<complex128>& circuit,
    const Pauli<complex128>& pauli,
    const np_complex128& statevector,
    bool compressed)
{
    return py_run_pauli_expectation_value_gradient<complex128, std::complex<double>>(circuit, pauli, statevector, compressed);
}

template <typename T, typename U, typename V, typename W> 
py::array_t<int> py_run_measurement(
    const Circuit<T>& circuit,
    const py::array_t<U>& statevector,
    const py::array_t<W>& randoms,
    bool compressed)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");
    static_assert(sizeof(V) == sizeof(W), "V and W must be of equal size");

    if (randoms.ndim() != 1) {
        throw std::runtime_error("randoms must be (nmeasurement,) shape");
    }

    py::buffer_info randoms_buffer = randoms.request();
    W* randoms_ptr = (W*) randoms_buffer.ptr;

    py::array_t<int> result = py::array_t<int>(randoms.shape(0));
    py::buffer_info result_buffer = result.request();
    int* result_ptr = (int*) result_buffer.ptr;

    U* statevector_ptr = np_pointer(circuit.nqubit(), statevector);

    vulcan::run_measurement<T, V>(circuit, (T*) statevector_ptr, randoms.shape(0), (V*) randoms_ptr, result_ptr, compressed);

    return result;
}

np_int32 py_run_measurement_float32(
    const Circuit<float32>& circuit,
    const np_float32& statevector,
    const np_float32& randoms,
    bool compressed)
{
    return py_run_measurement<float32, float, float32, float>(circuit, statevector, randoms, compressed);
}

np_int32 py_run_measurement_float64(
    const Circuit<float64>& circuit,
    const np_float64& statevector,
    const np_float64& randoms,
    bool compressed)
{
    return py_run_measurement<float64, double, float64, double>(circuit, statevector, randoms, compressed);
}

np_int32 py_run_measurement_complex64(
    const Circuit<complex64>& circuit,
    const np_complex64& statevector,
    const np_float32& randoms,
    bool compressed)
{
    return py_run_measurement<complex64, std::complex<float>, float32, float>(circuit, statevector, randoms, compressed);
}

np_int32 py_run_measurement_complex128(
    const Circuit<complex128>& circuit,
    const np_complex128& statevector,
    const np_float64& randoms,
    bool compressed)
{
    return py_run_measurement<complex128, std::complex<double>, float64, double>(circuit, statevector, randoms, compressed);
}

template <typename T>
Circuit<T> py_build_circuit(
    int nqubit,
    const std::vector<int>& nqubits,
    const std::vector<std::string>& names,
    const std::vector<std::complex<double>>& matrices,
    const std::vector<int>& qubits)
{
    if (nqubits.size() != names.size()) {
        throw std::runtime_error("nqubits.size() != names.size()");
    }  

    size_t nmatrices = 0;
    for (int nqubit2 : nqubits) {
        nmatrices += (1 << 2*nqubit2);
    }
    if (matrices.size() != nmatrices) {
        throw std::runtime_error("matrices not correctly sized");
    }

    size_t nqubits2 = 0;
    for (int nqubit2 : nqubits) {
        nqubits2 += nqubit2;
    }
    if (qubits.size() != nqubits2) {
        throw std::runtime_error("qubits not correctly sized");
    }

    std::vector<Gate<T>> gates;
    std::vector<std::vector<int>> qubits2;
    size_t matrix_index = 0;
    size_t qubit_index = 0;
    for (size_t index = 0; index < nqubits.size(); index++) {
        int nqubit2 = nqubits[index];
        std::vector<T> matrix(1 << 2*nqubit2);
        for (size_t index2 = 0; index2 < (1 << 2*nqubit2); index2++) {
            std::complex<double> val = matrices[index2 + matrix_index];
            matrix[index2] = T(val.real(), val.imag());
        }
        matrix_index += (1 << 2*nqubit2);
        std::vector<int> qubits3(nqubit2);
        for (size_t index2 = 0; index2 < nqubit2; index2++) {
            qubits3[index2] = qubits[index2 + qubit_index];
        }
        qubit_index += nqubit2;
        gates.push_back(Gate<T>(
            nqubit2,
            names[index],
            matrix));
        qubits2.push_back(qubits3);
    }

    return Circuit<T>(
        nqubit,
        gates,
        qubits2);
}

Circuit<float32> py_build_circuit_float32(
    int nqubit,
    const std::vector<int>& nqubits,
    const std::vector<std::string>& names,
    const std::vector<std::complex<double>>& matrices,
    const std::vector<int>& qubits)
{
    return py_build_circuit<float32>(nqubit, nqubits, names, matrices, qubits);
}

Circuit<float64> py_build_circuit_float64(
    int nqubit,
    const std::vector<int>& nqubits,
    const std::vector<std::string>& names,
    const std::vector<std::complex<double>>& matrices,
    const std::vector<int>& qubits)
{
    return py_build_circuit<float64>(nqubit, nqubits, names, matrices, qubits);
}

Circuit<complex64> py_build_circuit_complex64(
    int nqubit,
    const std::vector<int>& nqubits,
    const std::vector<std::string>& names,
    const std::vector<std::complex<double>>& matrices,
    const std::vector<int>& qubits)
{
    return py_build_circuit<complex64>(nqubit, nqubits, names, matrices, qubits);
}

Circuit<complex128> py_build_circuit_complex128(
    int nqubit,
    const std::vector<int>& nqubits,
    const std::vector<std::string>& names,
    const std::vector<std::complex<double>>& matrices,
    const std::vector<int>& qubits)
{
    return py_build_circuit<complex128>(nqubit, nqubits, names, matrices, qubits);
}

template <typename T>
Pauli<T> py_build_pauli(
    int nqubit,
    const std::vector<int>& nterms,
    const std::vector<int>& types,
    const std::vector<int>& qubits,
    const std::vector<std::complex<double>>& values)
{
    if (nterms.size() != values.size()) {
        throw std::runtime_error("nterms.size() != values.size()");
    } 

    size_t nterm_total = 0;
    for (int nterm : nterms) {
        nterm_total += nterm;
    }
    if (types.size() != nterm_total) {
        throw std::runtime_error("types not correctly sized");
    }
    if (qubits.size() != nterm_total) {
        throw std::runtime_error("qubits not correctly sized");
    }

    std::vector<std::vector<int>> types2;
    std::vector<std::vector<int>> qubits2;
    std::vector<T> values2;
    size_t offset = 0;
    for (size_t index = 0; index < nterms.size(); index++) {
        std::vector<int> types3;
        std::vector<int> qubits3;
        for (size_t index2 = 0; index2 < nterms[index]; index2++) {
            types3.push_back(types[index2 + offset]);
            qubits3.push_back(qubits[index2 + offset]);
        }
        types2.push_back(types3);
        qubits2.push_back(qubits3);
        offset += nterms[index];
        std::complex<double> val = values[index];
        values2.push_back(T(val.real(), val.imag()));
    }

    return Pauli<T>(
        nqubit,
        types2,
        qubits2,
        values2);
}

Pauli<float32> py_build_pauli_float32(
    int nqubit,
    const std::vector<int>& nterms,
    const std::vector<int>& types,
    const std::vector<int>& qubits,
    const std::vector<std::complex<double>>& values)
{
    return py_build_pauli<float32>(nqubit, nterms, types, qubits, values);
}

Pauli<float64> py_build_pauli_float64(
    int nqubit,
    const std::vector<int>& nterms,
    const std::vector<int>& types,
    const std::vector<int>& qubits,
    const std::vector<std::complex<double>>& values)
{
    return py_build_pauli<float64>(nqubit, nterms, types, qubits, values);
}

Pauli<complex64> py_build_pauli_complex64(
    int nqubit,
    const std::vector<int>& nterms,
    const std::vector<int>& types,
    const std::vector<int>& qubits,
    const std::vector<std::complex<double>>& values)
{
    return py_build_pauli<complex64>(nqubit, nterms, types, qubits, values);
}

Pauli<complex128> py_build_pauli_complex128(
    int nqubit,
    const std::vector<int>& nterms,
    const std::vector<int>& types,
    const std::vector<int>& qubits,
    const std::vector<std::complex<double>>& values)
{
    return py_build_pauli<complex128>(nqubit, nterms, types, qubits, values);
}

PYBIND11_MODULE(vulcan_plugin, m) {

    m.def("ndevice", &vulcan::ndevice);
    m.def("device_property_string", &vulcan::device_property_string);

    m.def("run_timings_blas_1", &vulcan::run_timings_blas_1);
    m.def("run_timings_gate_1", &vulcan::run_timings_gate_1);
    m.def("run_timings_gate_2", &vulcan::run_timings_gate_2);
    m.def("run_timings_pauli", &vulcan::run_timings_pauli);
    m.def("run_timings_measurement", &vulcan::run_timings_measurement);

    // => float32 <= //

    py::class_<vulcan::float32>(m, "float32")
    .def(py::init<float, float>())
    .def("real", &vulcan::float32::real)
    .def("imag", &vulcan::float32::imag)
    ;

    py::class_<vulcan::Gate<vulcan::float32>>(m, "Gate_float32")
    .def(py::init<int, const std::string&, const std::vector<vulcan::float32>&>())
    .def("nqubit", &vulcan::Gate<vulcan::float32>::nqubit)
    .def("name", &vulcan::Gate<vulcan::float32>::name)
    .def("matrix", &vulcan::Gate<vulcan::float32>::matrix)
    .def("adjoint", &vulcan::Gate<vulcan::float32>::adjoint)
    ; 

    py::class_<vulcan::Circuit<vulcan::float32>>(m, "Circuit_float32")
    .def(py::init<int, const std::vector<vulcan::Gate<vulcan::float32>>&, const std::vector<std::vector<int>>&>())
    .def("nqubit", &vulcan::Circuit<vulcan::float32>::nqubit)
    .def("gates", &vulcan::Circuit<vulcan::float32>::gates)
    .def("qubits", &vulcan::Circuit<vulcan::float32>::qubits)
    .def("adjoint", &vulcan::Circuit<vulcan::float32>::adjoint)
    .def("bit_reversal", &vulcan::Circuit<vulcan::float32>::bit_reversal)
    ; 

    py::class_<vulcan::Pauli<vulcan::float32>>(m, "Pauli_float32")
    .def(py::init<int, const std::vector<std::vector<int>>&, const std::vector<std::vector<int>>&, const std::vector<vulcan::float32>&>())
    .def("nqubit", &vulcan::Pauli<vulcan::float32>::nqubit)
    .def("types", &vulcan::Pauli<vulcan::float32>::types)
    .def("qubits", &vulcan::Pauli<vulcan::float32>::qubits)
    .def("values", &vulcan::Pauli<vulcan::float32>::values)
    .def("bit_reversal", &vulcan::Pauli<vulcan::float32>::bit_reversal)
    ; 

    m.def("build_circuit_float32", py_build_circuit_float32);    
    m.def("build_pauli_float32", py_build_pauli_float32);    

    m.def("run_statevector_float32", py_run_statevector_float32);
    m.def("run_pauli_sigma_float32", py_run_pauli_sigma_float32);
    m.def("run_pauli_expectation_float32", py_run_pauli_expectation_float32);
    m.def("run_pauli_expectation_value_float32", py_run_pauli_expectation_value_float32);
    m.def("run_pauli_expectation_value_gradient_float32", py_run_pauli_expectation_value_gradient_float32);
    m.def("run_measurement_float32", py_run_measurement_float32);

    // => float64 <= //

    py::class_<vulcan::float64>(m, "float64")
    .def(py::init<double, double>())
    .def("real", &vulcan::float64::real)
    .def("imag", &vulcan::float64::imag)
    ;

    py::class_<vulcan::Gate<vulcan::float64>>(m, "Gate_float64")
    .def(py::init<int, const std::string&, const std::vector<vulcan::float64>&>())
    .def("nqubit", &vulcan::Gate<vulcan::float64>::nqubit)
    .def("name", &vulcan::Gate<vulcan::float64>::name)
    .def("matrix", &vulcan::Gate<vulcan::float64>::matrix)
    .def("adjoint", &vulcan::Gate<vulcan::float64>::adjoint)
    ; 

    py::class_<vulcan::Circuit<vulcan::float64>>(m, "Circuit_float64")
    .def(py::init<int, const std::vector<vulcan::Gate<vulcan::float64>>&, const std::vector<std::vector<int>>&>())
    .def("nqubit", &vulcan::Circuit<vulcan::float64>::nqubit)
    .def("gates", &vulcan::Circuit<vulcan::float64>::gates)
    .def("qubits", &vulcan::Circuit<vulcan::float64>::qubits)
    .def("adjoint", &vulcan::Circuit<vulcan::float64>::adjoint)
    .def("bit_reversal", &vulcan::Circuit<vulcan::float64>::bit_reversal)
    ; 

    py::class_<vulcan::Pauli<vulcan::float64>>(m, "Pauli_float64")
    .def(py::init<int, const std::vector<std::vector<int>>&, const std::vector<std::vector<int>>&, const std::vector<vulcan::float64>&>())
    .def("nqubit", &vulcan::Pauli<vulcan::float64>::nqubit)
    .def("types", &vulcan::Pauli<vulcan::float64>::types)
    .def("qubits", &vulcan::Pauli<vulcan::float64>::qubits)
    .def("values", &vulcan::Pauli<vulcan::float64>::values)
    .def("bit_reversal", &vulcan::Pauli<vulcan::float64>::bit_reversal)
    ; 

    m.def("build_circuit_float64", py_build_circuit_float64);    
    m.def("build_pauli_float64", py_build_pauli_float64);    

    m.def("run_statevector_float64", py_run_statevector_float64);
    m.def("run_pauli_sigma_float64", py_run_pauli_sigma_float64);
    m.def("run_pauli_expectation_float64", py_run_pauli_expectation_float64);
    m.def("run_pauli_expectation_value_float64", py_run_pauli_expectation_value_float64);
    m.def("run_pauli_expectation_value_gradient_float64", py_run_pauli_expectation_value_gradient_float64);
    m.def("run_measurement_float64", py_run_measurement_float64);

    // => complex64 <= //

    py::class_<vulcan::complex64>(m, "complex64")
    .def(py::init<float, float>())
    .def("real", &vulcan::complex64::real)
    .def("imag", &vulcan::complex64::imag)
    ;

    py::class_<vulcan::Gate<vulcan::complex64>>(m, "Gate_complex64")
    .def(py::init<int, const std::string&, const std::vector<vulcan::complex64>&>())
    .def("nqubit", &vulcan::Gate<vulcan::complex64>::nqubit)
    .def("name", &vulcan::Gate<vulcan::complex64>::name)
    .def("matrix", &vulcan::Gate<vulcan::complex64>::matrix)
    .def("adjoint", &vulcan::Gate<vulcan::complex64>::adjoint)
    ; 

    py::class_<vulcan::Circuit<vulcan::complex64>>(m, "Circuit_complex64")
    .def(py::init<int, const std::vector<vulcan::Gate<vulcan::complex64>>&, const std::vector<std::vector<int>>&>())
    .def("nqubit", &vulcan::Circuit<vulcan::complex64>::nqubit)
    .def("gates", &vulcan::Circuit<vulcan::complex64>::gates)
    .def("qubits", &vulcan::Circuit<vulcan::complex64>::qubits)
    .def("adjoint", &vulcan::Circuit<vulcan::complex64>::adjoint)
    .def("bit_reversal", &vulcan::Circuit<vulcan::complex64>::bit_reversal)
    ; 

    py::class_<vulcan::Pauli<vulcan::complex64>>(m, "Pauli_complex64")
    .def(py::init<int, const std::vector<std::vector<int>>&, const std::vector<std::vector<int>>&, const std::vector<vulcan::complex64>&>())
    .def("nqubit", &vulcan::Pauli<vulcan::complex64>::nqubit)
    .def("types", &vulcan::Pauli<vulcan::complex64>::types)
    .def("qubits", &vulcan::Pauli<vulcan::complex64>::qubits)
    .def("values", &vulcan::Pauli<vulcan::complex64>::values)
    .def("bit_reversal", &vulcan::Pauli<vulcan::complex64>::bit_reversal)
    ; 

    m.def("build_circuit_complex64", py_build_circuit_complex64);    
    m.def("build_pauli_complex64", py_build_pauli_complex64);    

    m.def("run_statevector_complex64", py_run_statevector_complex64);
    m.def("run_pauli_sigma_complex64", py_run_pauli_sigma_complex64);
    m.def("run_pauli_expectation_complex64", py_run_pauli_expectation_complex64);
    m.def("run_pauli_expectation_value_complex64", py_run_pauli_expectation_value_complex64);
    m.def("run_pauli_expectation_value_gradient_complex64", py_run_pauli_expectation_value_gradient_complex64);
    m.def("run_measurement_complex64", py_run_measurement_complex64);

    // => complex128 <= //

    py::class_<vulcan::complex128>(m, "complex128")
    .def(py::init<double, double>())
    .def("real", &vulcan::complex128::real)
    .def("imag", &vulcan::complex128::imag)
    ;

    py::class_<vulcan::Gate<vulcan::complex128>>(m, "Gate_complex128")
    .def(py::init<int, const std::string&, const std::vector<vulcan::complex128>&>())
    .def("nqubit", &vulcan::Gate<vulcan::complex128>::nqubit)
    .def("name", &vulcan::Gate<vulcan::complex128>::name)
    .def("matrix", &vulcan::Gate<vulcan::complex128>::matrix)
    .def("adjoint", &vulcan::Gate<vulcan::complex128>::adjoint)
    ; 

    py::class_<vulcan::Circuit<vulcan::complex128>>(m, "Circuit_complex128")
    .def(py::init<int, const std::vector<vulcan::Gate<vulcan::complex128>>&, const std::vector<std::vector<int>>&>())
    .def("nqubit", &vulcan::Circuit<vulcan::complex128>::nqubit)
    .def("gates", &vulcan::Circuit<vulcan::complex128>::gates)
    .def("qubits", &vulcan::Circuit<vulcan::complex128>::qubits)
    .def("adjoint", &vulcan::Circuit<vulcan::complex128>::adjoint)
    .def("bit_reversal", &vulcan::Circuit<vulcan::complex128>::bit_reversal)
    ; 

    py::class_<vulcan::Pauli<vulcan::complex128>>(m, "Pauli_complex128")
    .def(py::init<int, const std::vector<std::vector<int>>&, const std::vector<std::vector<int>>&, const std::vector<vulcan::complex128>&>())
    .def("nqubit", &vulcan::Pauli<vulcan::complex128>::nqubit)
    .def("types", &vulcan::Pauli<vulcan::complex128>::types)
    .def("qubits", &vulcan::Pauli<vulcan::complex128>::qubits)
    .def("values", &vulcan::Pauli<vulcan::complex128>::values)
    .def("bit_reversal", &vulcan::Pauli<vulcan::complex128>::bit_reversal)
    ; 

    m.def("build_circuit_complex128", py_build_circuit_complex128);    
    m.def("build_pauli_complex128", py_build_pauli_complex128);    

    m.def("run_statevector_complex128", py_run_statevector_complex128);
    m.def("run_pauli_sigma_complex128", py_run_pauli_sigma_complex128);
    m.def("run_pauli_expectation_complex128", py_run_pauli_expectation_complex128);
    m.def("run_pauli_expectation_value_complex128", py_run_pauli_expectation_value_complex128);
    m.def("run_pauli_expectation_value_gradient_complex128", py_run_pauli_expectation_value_gradient_complex128);
    m.def("run_measurement_complex128", py_run_measurement_complex128);

}

} // namespace vulcan
