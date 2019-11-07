#include "vulcan.hpp"
#include "vulcan_types.hpp"
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

template <typename T, typename U> 
py::array_t<U> py_run_statevector(
    const Circuit<T>& circuit)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");

    py::array_t<U> result = py::array_t<U>(1ULL << circuit.nqubit());
    py::buffer_info buffer = result.request();
    U* ptr = (U*) buffer.ptr;

    vulcan::run_statevector<T>(circuit, nullptr, (T*) ptr);

    return result;
}

np_complex128 py_run_statevector_complex128(
    const Circuit<complex128>& circuit)
{
    return py_run_statevector<complex128, std::complex<double>>(circuit);
}

template <typename T, typename U> 
py::array_t<U> py_run_pauli_sigma(
    const Pauli<T>& pauli,
    const py::array_t<U>& statevector)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");

    py::buffer_info buffer1 = statevector.request();
    U* ptr1 = (U*) buffer1.ptr;

    py::array_t<U> result = py::array_t<U>(1ULL << pauli.nqubit());
    py::buffer_info buffer2 = result.request();
    U* ptr2 = (U*) buffer2.ptr;

    vulcan::run_pauli_sigma<T>(pauli, (T*) ptr1, (T*) ptr2);

    return result;
}

np_complex128 py_run_pauli_sigma_complex128(
    const Pauli<complex128>& pauli,
    const np_complex128 statevector)
{
    return py_run_pauli_sigma<complex128, std::complex<double>>(pauli, statevector);
}

template <typename T, typename U> 
U py_run_pauli_expectation_value(
    const Circuit<T>& circuit,
    const Pauli<T>& pauli)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");

    T val = vulcan::run_pauli_expectation_value<T>(circuit, pauli, nullptr);

    U val2;
    std::memcpy(&val2, &val, sizeof(T));

    return val2;
}

std::complex<double> py_run_pauli_expectation_value_complex128(
    const Circuit<complex128>& circuit,
    const Pauli<complex128>& pauli)
{
    return py_run_pauli_expectation_value<complex128, std::complex<double>>(circuit, pauli);
}

template <typename T, typename U> 
std::vector<U> py_run_pauli_expectation_value_gradient(
    const Circuit<T>& circuit,
    const Pauli<T>& pauli)
{
    static_assert(sizeof(T) == sizeof(U), "T and U must be of equal size");

    std::vector<T> val = vulcan::run_pauli_expectation_value_gradient<T>(circuit, pauli, nullptr);

    std::vector<U> val2(val.size());

    ::memcpy(val2.data(), val.data(), val.size() * sizeof(T));

    return val2;
}

std::vector<std::complex<double>> py_run_pauli_expectation_value_gradient_complex128(
    const Circuit<complex128>& circuit,
    const Pauli<complex128>& pauli)
{
    return py_run_pauli_expectation_value_gradient<complex128, std::complex<double>>(circuit, pauli);
}

PYBIND11_MODULE(vulcan_plugin, m) {

    m.def("ndevice", &vulcan::ndevice);
    m.def("device_property_string", &vulcan::device_property_string);

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
    m.def("run_pauli_expectation_value_complex128", py_run_pauli_expectation_value_complex128);
    m.def("run_pauli_expectation_value_gradient_complex128", py_run_pauli_expectation_value_gradient_complex128);

}

} // namespace vulcan
