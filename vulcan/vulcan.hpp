#pragma once

#include "vulcan_gpu.hpp"
#include "vulcan_types.hpp"
#include <stdexcept>
#include <vector>
#include <algorithm>
    
namespace vulcan {    

template <typename T>
class Gate {

public:
    
Gate(
    int nqubit,
    const std::string& name,
    const std::vector<T>& matrix) :
    nqubit_(nqubit),
    name_(name),
    matrix_(matrix) {
    if (matrix.size() != (1ULL << 2*nqubit)) {
        throw std::runtime_error("Gate size is not correct");
    }
}

int nqubit() const { return nqubit_; }
const std::string& name() const { return name_; }
const std::vector<T>& matrix() const { return matrix_; }

Gate<T> adjoint() const 
{
    std::vector<T> matrix(1ULL << 2*nqubit_);
    for (int i = 0; i < 1ULL << nqubit_; i++) {
        for (int j = 0; j < 1ULL << nqubit_; j++) {
            matrix[j * (1ULL << nqubit_) + i] = vulcan::conj(matrix_[i * (1ULL << nqubit_) + j]);
        }
    }
    return Gate(
        nqubit_,
	name_,
        matrix);
}

protected:

int nqubit_; 
std::string name_;
std::vector<T> matrix_;

};

template <typename T>
class Circuit {

public:

Circuit(
    int nqubit,
    const std::vector<Gate<T>>& gates,
    const std::vector<std::vector<int>>& qubits) :
    nqubit_(nqubit),
    gates_(gates),
    qubits_(qubits) {
    if (gates.size() != qubits.size()) {
        throw std::runtime_error("gates and qubits must be same size");
    } 

    for (size_t index = 0; index < gates.size(); index++) {
        if (gates[index].nqubit() != qubits[index].size()) {
            throw std::runtime_error("gates and qubits must have same index sizes");
        }
        for (int qubit : qubits[index]) {
            if (qubit < 0 || qubit >= nqubit_) {
                throw std::runtime_error("qubit out of position");
            }
        }
    }
}

int nqubit() const { return nqubit_; }
const std::vector<Gate<T>>& gates() const { return gates_; }
const std::vector<std::vector<int>>& qubits() const { return qubits_; }

Circuit<T> adjoint() const 
{
    std::vector<Gate<T>> gates;
    std::vector<std::vector<int>> qubits;
    for (int index = ((int) gates_.size()) - 1; index >= 0; index--) {
        gates.push_back(gates_[index].adjoint());
        qubits.push_back(qubits_[index]);
    }   
    return Circuit(
        nqubit_,
        gates,
        qubits);
}

Circuit<T> bit_reversal() const
{
    std::vector<std::vector<int>> qubits;
    for (size_t index = 0; index < qubits_.size(); index++) {
        qubits.push_back({});
        for (size_t index2 = 0; index2 < qubits_[index].size(); index2++) {
            qubits[index].push_back(nqubit_ - 1 - qubits_[index][index2]);
        }
    } 
    return Circuit(
        nqubit_,
        gates_,
        qubits);
}

protected:

int nqubit_;
std::vector<Gate<T>> gates_;
std::vector<std::vector<int>> qubits_;

};

template <typename T>
class Pauli {

public:

Pauli(
    int nqubit,
    const std::vector<std::vector<int>>& types,
    const std::vector<std::vector<int>>& qubits,
    const std::vector<T> values) :
    nqubit_(nqubit),
    types_(types),
    qubits_(qubits),
    values_(values) {
    
    if (types.size() != qubits.size()) {
        throw std::runtime_error("types and qubits must be same size");
    } 
    if (values.size() != qubits.size()) {
        throw std::runtime_error("values and qubits must be same size");
    }

    for (size_t index = 0; index < types.size(); index++) {
        if (types[index].size() != qubits[index].size()) {
            throw std::runtime_error("types and qubits must have same index sizes");
        }
        for (int qubit : qubits[index]) {
            if (qubit < 0 || qubit >= nqubit_) {
                throw std::runtime_error("qubit out of position");
            }
        }
        for (int type : types[index]) {
            if (type < 0 || type > 2) {
                throw std::runtime_error("type must be 0 (X), 1 (Y), or 2 (Z)");
            }
        }
    }
}

int nqubit() const { return nqubit_; }
const std::vector<std::vector<int>>& types() const { return types_; }
const std::vector<std::vector<int>>& qubits() const { return qubits_; }
const std::vector<T>& values() const { return values_; }

Pauli<T> bit_reversal() const
{
    std::vector<std::vector<int>> qubits;
    for (size_t index = 0; index < qubits_.size(); index++) {
        qubits.push_back({});
        for (size_t index2 = 0; index2 < qubits_[index].size(); index2++) {
            qubits[index].push_back(nqubit_ - 1 - qubits_[index][index2]);
        }
    } 
    return Pauli(
        nqubit_,
        types_,
        qubits,
        values_);
}

protected:

int nqubit_;
std::vector<std::vector<int>> types_;
std::vector<std::vector<int>> qubits_;
std::vector<T> values_;

};    

// => Internal C++ API <= //

namespace util {

template <typename T>
void run_statevector(
    const Circuit<T>& circuit,
    T* statevector_d)
{
    for (size_t index = 0; index < circuit.gates().size(); index++) {
        const Gate<T>& gate = circuit.gates()[index];
        const std::vector<int>& qubits = circuit.qubits()[index];
        const std::vector<T>& matrix = gate.matrix();
        if (gate.nqubit() == 1) {
            int qubitA = qubits[0];
            vulcan::apply_gate_1(
                circuit.nqubit(),
                statevector_d,
                statevector_d,
                qubitA,
                matrix[0],
                matrix[1],
                matrix[2],
                matrix[3]);
        } else if (gate.nqubit() == 2) {
            int qubitA = qubits[0];
            int qubitB = qubits[1];
            vulcan::apply_gate_2(
                circuit.nqubit(),
                statevector_d,
                statevector_d,
                qubitA,
                qubitB,
                matrix[ 0],
                matrix[ 1],
                matrix[ 2],
                matrix[ 3],
                matrix[ 4],
                matrix[ 5],
                matrix[ 6],
                matrix[ 7],
                matrix[ 8],
                matrix[ 9],
                matrix[10],
                matrix[11],
                matrix[12],
                matrix[13],
                matrix[14],
                matrix[15]);
        } else {
            throw std::runtime_error("Gate nqubit is invalid.");
        }
    }
}

template <typename T> 
void apply_pauli_1(
    int nqubit,
    T* statevector1_d,
    T* statevector2_d,
    int qubitA,
    int type,
    T a = T(1.0),
    T b = T(0.0))
{
    T O00;
    T O01;
    T O10;
    T O11;

    if (type == 0) {
        O00 = T(0.0, 0.0);
        O01 = T(1.0, 0.0);
        O10 = T(1.0, 0.0);
        O11 = T(0.0, 0.0);
    } else if (type == 1) {
        O00 = T(0.0, 0.0);
        O01 = T(0.0, -1.0);
        O10 = T(0.0, 1.0);
        O11 = T(0.0, 0.0);
    } else if (type == 2) {
        O00 = T(1.0, 0.0);
        O01 = T(0.0, 0.0);
        O10 = T(0.0, 0.0);
        O11 = T(-1.0, 0.0);
    } else {
        throw std::runtime_error("type must be 0 (X), 1 (Y), or 2 (Z)");
    }

    vulcan::apply_gate_1(
        nqubit,
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

template <typename T> 
void apply_pauli_1_mody(
    int nqubit,
    T* statevector1_d,
    T* statevector2_d,
    int qubitA,
    int type,
    T a = T(1.0),
    T b = T(0.0))
{
    T O00;
    T O01;
    T O10;
    T O11;

    if (type == 0) {
        O00 = T(0.0, 0.0);
        O01 = T(1.0, 0.0);
        O10 = T(1.0, 0.0);
        O11 = T(0.0, 0.0);
    } else if (type == 1) {
        O00 = T(0.0, 0.0);
        O01 = T(1.0, 0.0);
        O10 = T(-1.0, 0.0);
        O11 = T(0.0, 0.0);
    } else if (type == 2) {
        O00 = T(1.0, 0.0);
        O01 = T(0.0, 0.0);
        O10 = T(0.0, 0.0);
        O11 = T(-1.0, 0.0);
    } else {
        throw std::runtime_error("type must be 0 (X), 1 (Y), or 2 (Z)");
    }

    vulcan::apply_gate_1(
        nqubit,
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

template <typename T> 
void apply_ipauli_1(
    int nqubit,
    T* statevector1_d,
    T* statevector2_d,
    int qubitA,
    int type,
    T a = T(1.0),
    T b = T(0.0))
{
    T O00;
    T O01;
    T O10;
    T O11;

    if (type == 0) {
        O00 = T(0.0, 0.0);
        O01 = T(0.0, 1.0);
        O10 = T(0.0, 1.0);
        O11 = T(0.0, 0.0);
    } else if (type == 1) {
        O00 = T(0.0, 0.0);
        O01 = T(1.0,  0.0);
        O10 = T(-1.0, 0.0);
        O11 = T(0.0, 0.0);
    } else if (type == 2) {
        O00 = T(0.0, 1.0);
        O01 = T(0.0, 0.0);
        O10 = T(0.0, 0.0);
        O11 = T(0.0, -1.0);
    } else {
        throw std::runtime_error("type must be 0 (X), 1 (Y), or 2 (Z)");
    }

    vulcan::apply_gate_1(
        nqubit,
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

/**
 * (-i)**n. Returns correct result if T is a scalar quantity and n is even.
 *
 * Params:
 *  n (int) - desired integral power of (-i)
 * Returns:
 *  (T) - (-i)**n. 0.0 (incorrect) if T is scalar and n is odd.
 **/
template <typename T>
T mi_pow_n(int n)
{
    if (n % 2 == 0) {
        return T(((n / 2) % 2) == 0 ? 1.0 : -1.0, 0.0);
    } else {
        return T(0.0, (((n + 1) / 2) % 2) == 0 ? 1.0 : -1.0);
    }
}

template <typename T>
void apply_pauli(
    const Pauli<T>& pauli,
    T* statevector1_d,
    T* statevector2_d,
    T* statevector3_d)
{
    vulcan::zero_statevector(pauli.nqubit(), statevector2_d);
    for (size_t index = 0; index < pauli.types().size(); index++) {
        const std::vector<int>& types = pauli.types()[index];
        const std::vector<int>& qubits = pauli.qubits()[index];
        T val = pauli.values()[index];
        vulcan::axpby(pauli.nqubit(), statevector1_d, statevector3_d);
        int ny = 0;
        for (size_t index2 = 0; index2 < types.size(); index2++) {
            int type = types[index2];
            int qubit = qubits[index2];          
            if (type == 1) ny += 1;
            vulcan::util::apply_pauli_1_mody(pauli.nqubit(), statevector3_d, statevector3_d, qubit, type);
        }  
        // (-i)**ny
        T scal = mi_pow_n<T>(ny);
        vulcan::axpby(pauli.nqubit(), statevector3_d, statevector2_d, scal*val, T(1.0));
    }
}

template <typename T>
Pauli<T> pauli_expectation(
    const Pauli<T>& pauli,
    T* statevector1_d,
    T* statevector2_d)
{
    std::vector<T> values(pauli.types().size());
    for (size_t index = 0; index < pauli.types().size(); index++) {
        const std::vector<int>& types = pauli.types()[index];
        const std::vector<int>& qubits = pauli.qubits()[index];
        vulcan::axpby(pauli.nqubit(), statevector1_d, statevector2_d);
        int ny = 0;
        for (size_t index2 = 0; index2 < types.size(); index2++) {
            int type = types[index2];
            int qubit = qubits[index2];          
            if (type == 1) ny += 1;
            vulcan::util::apply_pauli_1_mody(pauli.nqubit(), statevector2_d, statevector2_d, qubit, type);
        }  
        // (-i)**ny
        T scal = mi_pow_n<T>(ny);
        values[index] = scal*vulcan::dot(pauli.nqubit(), statevector1_d, statevector2_d); 
    }
    
    return Pauli<T>(
        pauli.nqubit(),
        pauli.types(),
        pauli.qubits(),
        values);
}

} // namespace vulcan::util

// => External C++ API <= //

template <typename T>
void run_statevector(
    const Circuit<T>& circuit,
    T* statevector_h,
    T* result_h)
{
    T* statevector_d = vulcan::malloc_and_initialize_statevector(circuit.nqubit(), statevector_h);
    vulcan::util::run_statevector(circuit, statevector_d);
    vulcan::copy_statevector_to_host(circuit.nqubit(), statevector_d, result_h);
    vulcan::free_statevector(statevector_d);
}

template <typename T>
void run_pauli_sigma(
    const Pauli<T>& pauli,
    T* statevector_h,
    T* result_h)
{
    T* statevector1_d = vulcan::malloc_and_initialize_statevector(pauli.nqubit(), statevector_h);
    T* statevector2_d = vulcan::malloc_statevector<T>(pauli.nqubit());
    T* statevector3_d = vulcan::malloc_statevector<T>(pauli.nqubit());
    vulcan::util::apply_pauli(pauli, statevector1_d, statevector2_d, statevector3_d);
    vulcan::copy_statevector_to_host(pauli.nqubit(), statevector2_d, result_h);
    vulcan::free_statevector(statevector1_d);
    vulcan::free_statevector(statevector2_d);
    vulcan::free_statevector(statevector3_d);
}

template <typename T>
Pauli<T> run_pauli_expectation(
    const Circuit<T>& circuit,
    const Pauli<T>& pauli,
    const T* statevector_h)
{
    if (circuit.nqubit() != pauli.nqubit()) {
        throw std::runtime_error("circuit and pauli do not have same nqubit");
    }
    
    T* statevector1_d = vulcan::malloc_and_initialize_statevector(pauli.nqubit(), statevector_h);
    T* statevector2_d = vulcan::malloc_statevector<T>(pauli.nqubit());
    vulcan::util::run_statevector(circuit, statevector1_d);
    Pauli<T> expectation = vulcan::util::pauli_expectation(pauli, statevector1_d, statevector2_d);
    vulcan::free_statevector(statevector1_d);
    vulcan::free_statevector(statevector2_d);
    return expectation; 
}

template <typename T>
T run_pauli_expectation_value(
    const Circuit<T>& circuit,
    const Pauli<T>& pauli,
    const T* statevector_h)
{
    if (pauli.nqubit() != circuit.nqubit()) {
	throw std::runtime_error("pauli and circuit must have same nqubit");
    }

    T* statevector1_d = vulcan::malloc_and_initialize_statevector(pauli.nqubit(), statevector_h);
    T* statevector2_d = vulcan::malloc_statevector<T>(pauli.nqubit());
    T* statevector3_d = vulcan::malloc_statevector<T>(pauli.nqubit());
    vulcan::util::run_statevector(circuit, statevector1_d);
    vulcan::util::apply_pauli(pauli, statevector1_d, statevector2_d, statevector3_d);
    T val = vulcan::dot(circuit.nqubit(), statevector1_d, statevector2_d);
    vulcan::free_statevector(statevector1_d);
    vulcan::free_statevector(statevector2_d);
    vulcan::free_statevector(statevector3_d);
    return val;
}
    
template <typename T>
std::vector<T> run_pauli_expectation_value_gradient(
    const Circuit<T>& circuit,
    const Pauli<T>& pauli,
    const T* statevector_h)
{
    if (pauli.nqubit() != circuit.nqubit()) {
	throw std::runtime_error("pauli and circuit must have same nqubit");
    }

    T* statevector1_d = vulcan::malloc_and_initialize_statevector(pauli.nqubit(), statevector_h);
    T* statevector2_d = vulcan::malloc_statevector<T>(pauli.nqubit());
    T* statevector3_d = vulcan::malloc_statevector<T>(pauli.nqubit());
    vulcan::util::run_statevector(circuit, statevector1_d);
    vulcan::util::apply_pauli(pauli, statevector1_d, statevector2_d, statevector3_d);

    Circuit<T> circuit2 = circuit.adjoint();
     
    std::vector<T> gradient;
    std::vector<Gate<T>> gates;
    std::vector<std::vector<int>> qubits;
    for (size_t index = 0; index < circuit2.gates().size(); index++) {
        gates.push_back(circuit2.gates()[index]);
        qubits.push_back(circuit2.qubits()[index]);
        const std::string& name = circuit2.gates()[index].name();
        if (circuit2.gates()[index].nqubit() == 1 && (name == "Rx" || name == "Ry" || name == "Rz")) {
            int type = -1;
            if (name == "Rx") type = 0;
            else if (name == "Ry") type = 1;
            else type = 2;
            Circuit<T> circuit3(
                circuit2.nqubit(),
                gates,
                qubits);
	    gates.clear();
	    qubits.clear();
            vulcan::util::run_statevector(circuit3, statevector1_d);
            vulcan::util::run_statevector(circuit3, statevector2_d);
            vulcan::util::apply_ipauli_1(
                circuit2.nqubit(),
                statevector2_d,
                statevector3_d,
                circuit2.qubits()[index][0],
                type);
            gradient.push_back(T(2.0 * vulcan::dot(circuit.nqubit(), statevector1_d, statevector3_d).real()));
        }
    }

    std::reverse(gradient.begin(), gradient.end());

    vulcan::free_statevector(statevector1_d);
    vulcan::free_statevector(statevector2_d);
    vulcan::free_statevector(statevector3_d);
    return gradient;
}
    
} // namespace vulcan
