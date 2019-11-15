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
    const py::array_t<U>& statevector)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");

    py::array_t<U> result = py::array_t<U>(1ULL << circuit.nqubit());
    py::buffer_info buffer = result.request();
    U* ptr = (U*) buffer.ptr;

    U* ptr2 = np_pointer(circuit.nqubit(), statevector);

    vulcan::run_statevector<T>(circuit, (T*) ptr2, (T*) ptr);

    return result;
}

np_float32 py_run_statevector_float32(
    const Circuit<float32>& circuit,
    const np_float32& statevector)
{
    return py_run_statevector<float32, float>(circuit, statevector);
}

np_float64 py_run_statevector_float64(
    const Circuit<float64>& circuit,
    const np_float64& statevector)
{
    return py_run_statevector<float64, double>(circuit, statevector);
}

np_complex64 py_run_statevector_complex64(
    const Circuit<complex64>& circuit,
    const np_complex64& statevector)
{
    return py_run_statevector<complex64, std::complex<float>>(circuit, statevector);
}

np_complex128 py_run_statevector_complex128(
    const Circuit<complex128>& circuit,
    const np_complex128& statevector)
{
    return py_run_statevector<complex128, std::complex<double>>(circuit, statevector);
}

template <typename T, typename U> 
py::array_t<U> py_run_pauli_sigma(
    const Pauli<T>& pauli,
    const py::array_t<U>& statevector)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");

    if (statevector.ndim() != 1) throw std::runtime_error("statevector must have ndim == 1");
    if (statevector.shape(0) != (1ULL << pauli.nqubit())) throw std::runtime_error("statevector must be shape (2**nqubit,)");
    
    py::buffer_info buffer1 = statevector.request();
    U* ptr1 = (U*) buffer1.ptr;

    py::array_t<U> result = py::array_t<U>(1ULL << pauli.nqubit());
    py::buffer_info buffer2 = result.request();
    U* ptr2 = (U*) buffer2.ptr;

    vulcan::run_pauli_sigma<T>(pauli, (T*) ptr1, (T*) ptr2);

    return result;
}

np_float32 py_run_pauli_sigma_float32(
    const Pauli<float32>& pauli,
    const np_float32& statevector)
{
    return py_run_pauli_sigma<float32, float>(pauli, statevector);
}

np_float64 py_run_pauli_sigma_float64(
    const Pauli<float64>& pauli,
    const np_float64& statevector)
{
    return py_run_pauli_sigma<float64, double>(pauli, statevector);
}

np_complex64 py_run_pauli_sigma_complex64(
    const Pauli<complex64>& pauli,
    const np_complex64& statevector)
{
    return py_run_pauli_sigma<complex64, std::complex<float>>(pauli, statevector);
}

np_complex128 py_run_pauli_sigma_complex128(
    const Pauli<complex128>& pauli,
    const np_complex128& statevector)
{
    return py_run_pauli_sigma<complex128, std::complex<double>>(pauli, statevector);
}

template <typename T, typename U>
Pauli<T> py_run_pauli_expectation(
    const Circuit<T>& circuit,
    const Pauli<T>& pauli,
    const py::array_t<U>& statevector)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");

    U* ptr2 = np_pointer(circuit.nqubit(), statevector);

    return vulcan::run_pauli_expectation<T>(circuit, pauli, (T*) ptr2);
}

Pauli<float32> py_run_pauli_expectation_float32(
    const Circuit<float32>& circuit,
    const Pauli<float32>& pauli,
    const np_float32& statevector)
{
    return py_run_pauli_expectation<float32, float>(circuit, pauli, statevector);
}

Pauli<float64> py_run_pauli_expectation_float64(
    const Circuit<float64>& circuit,
    const Pauli<float64>& pauli,
    const np_float64& statevector)
{
    return py_run_pauli_expectation<float64, double>(circuit, pauli, statevector);
}

Pauli<complex64> py_run_pauli_expectation_complex64(
    const Circuit<complex64>& circuit,
    const Pauli<complex64>& pauli,
    const np_complex64& statevector)
{
    return py_run_pauli_expectation<complex64, std::complex<float>>(circuit, pauli, statevector);
}

Pauli<complex128> py_run_pauli_expectation_complex128(
    const Circuit<complex128>& circuit,
    const Pauli<complex128>& pauli,
    const np_complex128& statevector)
{
    return py_run_pauli_expectation<complex128, std::complex<double>>(circuit, pauli, statevector);
}

template <typename T, typename U> 
U py_run_pauli_expectation_value(
    const Circuit<T>& circuit,
    const Pauli<T>& pauli,
    const py::array_t<U>& statevector)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");

    U* ptr2 = np_pointer(circuit.nqubit(), statevector);

    T val = vulcan::run_pauli_expectation_value<T>(circuit, pauli, (T*) ptr2);

    U val2;
    std::memcpy(&val2, &val, sizeof(T));

    return val2;
}

float py_run_pauli_expectation_value_float32(
    const Circuit<float32>& circuit,
    const Pauli<float32>& pauli,
    const np_float32& statevector)
{
    return py_run_pauli_expectation_value<float32, float>(circuit, pauli, statevector);
}

double py_run_pauli_expectation_value_float64(
    const Circuit<float64>& circuit,
    const Pauli<float64>& pauli,
    const np_float64& statevector)
{
    return py_run_pauli_expectation_value<float64, double>(circuit, pauli, statevector);
}

std::complex<float> py_run_pauli_expectation_value_complex64(
    const Circuit<complex64>& circuit,
    const Pauli<complex64>& pauli,
    const np_complex64& statevector)
{
    return py_run_pauli_expectation_value<complex64, std::complex<float>>(circuit, pauli, statevector);
}

std::complex<double> py_run_pauli_expectation_value_complex128(
    const Circuit<complex128>& circuit,
    const Pauli<complex128>& pauli,
    const np_complex128& statevector)
{
    return py_run_pauli_expectation_value<complex128, std::complex<double>>(circuit, pauli, statevector);
}

template <typename T, typename U> 
py::array_t<U> py_run_pauli_expectation_value_gradient(
    const Circuit<T>& circuit,
    const Pauli<T>& pauli,
    const py::array_t<U>& statevector)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");

    U* ptr2 = np_pointer(circuit.nqubit(), statevector);

    std::vector<T> val = vulcan::run_pauli_expectation_value_gradient<T>(circuit, pauli, (T*) ptr2);

    py::array_t<U> result = py::array_t<U>(val.size());
    py::buffer_info buffer3 = result.request();
    U* ptr3 = (U*) buffer3.ptr;

    ::memcpy(ptr3, val.data(), val.size() * sizeof(T));

    return result;
}

np_float32 py_run_pauli_expectation_value_gradient_float32(
    const Circuit<float32>& circuit,
    const Pauli<float32>& pauli,
    const np_float32& statevector)
{
    return py_run_pauli_expectation_value_gradient<float32, float>(circuit, pauli, statevector);
}

np_float64 py_run_pauli_expectation_value_gradient_float64(
    const Circuit<float64>& circuit,
    const Pauli<float64>& pauli,
    const np_float64& statevector)
{
    return py_run_pauli_expectation_value_gradient<float64, double>(circuit, pauli, statevector);
}

np_complex64 py_run_pauli_expectation_value_gradient_complex64(
    const Circuit<complex64>& circuit,
    const Pauli<complex64>& pauli,
    const np_complex64& statevector)
{
    return py_run_pauli_expectation_value_gradient<complex64, std::complex<float>>(circuit, pauli, statevector);
}

np_complex128 py_run_pauli_expectation_value_gradient_complex128(
    const Circuit<complex128>& circuit,
    const Pauli<complex128>& pauli,
    const np_complex128& statevector)
{
    return py_run_pauli_expectation_value_gradient<complex128, std::complex<double>>(circuit, pauli, statevector);
}

PYBIND11_MODULE(vulcan_plugin, m) {

    m.def("ndevice", &vulcan::ndevice);
    m.def("device_property_string", &vulcan::device_property_string);

    m.def("run_timings_blas_1", &vulcan::run_timings_blas_1);
    m.def("run_timings_gate_1", &vulcan::run_timings_gate_1);
    m.def("run_timings_gate_2", &vulcan::run_timings_gate_2);
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

    m.def("run_statevector_float32", py_run_statevector_float32);
    m.def("run_pauli_sigma_float32", py_run_pauli_sigma_float32);
    m.def("run_pauli_expectation_float32", py_run_pauli_expectation_float32);
    m.def("run_pauli_expectation_value_float32", py_run_pauli_expectation_value_float32);
    m.def("run_pauli_expectation_value_gradient_float32", py_run_pauli_expectation_value_gradient_float32);

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

    m.def("run_statevector_float64", py_run_statevector_float64);
    m.def("run_pauli_sigma_float64", py_run_pauli_sigma_float64);
    m.def("run_pauli_expectation_float64", py_run_pauli_expectation_float64);
    m.def("run_pauli_expectation_value_float64", py_run_pauli_expectation_value_float64);
    m.def("run_pauli_expectation_value_gradient_float64", py_run_pauli_expectation_value_gradient_float64);

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

    m.def("run_statevector_complex64", py_run_statevector_complex64);
    m.def("run_pauli_sigma_complex64", py_run_pauli_sigma_complex64);
    m.def("run_pauli_expectation_complex64", py_run_pauli_expectation_complex64);
    m.def("run_pauli_expectation_value_complex64", py_run_pauli_expectation_value_complex64);
    m.def("run_pauli_expectation_value_gradient_complex64", py_run_pauli_expectation_value_gradient_complex64);

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

    m.def("run_statevector_complex128", py_run_statevector_complex128);
    m.def("run_pauli_sigma_complex128", py_run_pauli_sigma_complex128);
    m.def("run_pauli_expectation_complex128", py_run_pauli_expectation_complex128);
    m.def("run_pauli_expectation_value_complex128", py_run_pauli_expectation_value_complex128);
    m.def("run_pauli_expectation_value_gradient_complex128", py_run_pauli_expectation_value_gradient_complex128);

}

} // namespace vulcan
