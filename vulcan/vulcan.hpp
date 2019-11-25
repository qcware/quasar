#pragma once

#include "vulcan_gpu.hpp"
#include "vulcan_util.hpp"
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <map>

namespace vulcan {    

// => Vulcan C++ Data Structures <= //

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

Circuit<T> compressed() const
{
    for (const Gate<T>& gate : gates_) {
        if (gate.nqubit() > 2) {
            throw std::runtime_error("compressed cannot handle nqubit > 2");
        }
    }

    const std::vector<Gate<T>>& gates1 = gates_;
    const std::vector<std::vector<int>>& qubits1 = qubits_;

    // Jam runs of 1-body gates together into single 1-body gates
    std::vector<Gate<T>> gates2;
    std::vector<std::vector<int>> qubits2;
    std::vector<std::vector<size_t>> gate_indices1(nqubit_);
    for (size_t index = 0; index < qubits1.size(); index++) {
        const Gate<T>& gate = gates1[index];
        const std::vector<int>& qubits = qubits1[index];
        if (qubits.size() == 1) {
            gate_indices1[qubits[0]].push_back(index);
        } else {
            for (int qubit : qubits) {
                if (gate_indices1[qubit].size()) {
                    std::vector<T> U = vulcan::util::identity2<T>();
                    for (size_t index2 : gate_indices1[qubit]) {
                        const Gate<T>& gate2 = gates1[index2];
                        U = vulcan::util::multiply2<T>(gate2.matrix(), U);
                    }
                    gates2.push_back(Gate<T>(
                        1,
                        "U1",
                        U));
                    qubits2.push_back({qubit});
                    gate_indices1[qubit].clear();
                }
            }
            gates2.push_back(gate);
            qubits2.push_back(qubits);
        }
    }
    for (int qubit = 0; qubit < nqubit_; qubit++) {
        if (gate_indices1[qubit].size()) {
            std::vector<T> U = vulcan::util::identity2<T>();
            for (size_t index2 : gate_indices1[qubit]) {
                const Gate<T>& gate2 = gates1[index2];
                U = vulcan::util::multiply2<T>(gate2.matrix(), U);
            }
            gates2.push_back(Gate<T>(
                1,
                "U1",
                U));
            qubits2.push_back({qubit});
        }
    }

    // Find and separate isolated 1-body gates
    std::vector<bool> has_2_body(nqubit_, false);
    for (size_t index = 0; index < gates2.size(); index++) {
        const std::vector<int>& qubits = qubits2[index];
        if (qubits.size() > 1) {
            for (int qubit : qubits) {
                has_2_body[qubit] = true;
            }
        }
    }
    std::vector<Gate<T>> gates0;
    std::vector<std::vector<int>> qubits0;
    std::vector<Gate<T>> gates3;
    std::vector<std::vector<int>> qubits3;
    for (size_t index = 0; index < gates2.size(); index++){
        const Gate<T>& gate = gates2[index];
        const std::vector<int>& qubits = qubits2[index];
        if (qubits.size() == 1 && !has_2_body[qubits[0]]) {
            gates0.push_back(gate);
            qubits0.push_back(qubits);
        } else {
            gates3.push_back(gate);
            qubits3.push_back(qubits);
        }
    }

    // Merge 1-body gates into neighboring 2-body gates
    std::map<size_t, std::vector<std::pair<size_t, int>>> merges;
    for (size_t index = 0; index < gates3.size(); index++){
        const std::vector<int>& qubits = qubits3[index];
        if (qubits.size() != 1) continue;
        bool found = false;
        for (ssize_t index2 = index + 1; index2 < gates3.size(); index2++) {
            const std::vector<int>& qubits2 = qubits3[index2];
            if (qubits2.size() != 2) continue;
            if (qubits[0] == qubits2[0]) {
                if (!merges.count(index2)) merges[index2] = {};
                merges[index2].push_back(std::pair<size_t, int>(index, 0));
                found = true;
                break;
            } else if (qubits[0] == qubits2[1]) {
                if (!merges.count(index2)) merges[index2] = {};
                merges[index2].push_back(std::pair<size_t, int>(index, 1));
                found = true;
                break;
            }
        }
        if (found) continue;
        for (ssize_t index2 = index - 1; index2 >= 0; index2--) {
            const std::vector<int>& qubits2 = qubits3[index2];
            if (qubits2.size() != 2) continue;
            if (qubits[0] == qubits2[0]) {
                if (!merges.count(index2)) merges[index2] = {};
                merges[index2].push_back(std::pair<size_t, int>(index, 0));
                found = true;
                break;
            } else if (qubits[0] == qubits2[1]) {
                if (!merges.count(index2)) merges[index2] = {};
                merges[index2].push_back(std::pair<size_t, int>(index, 1));
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("Impossible");
    }

    std::vector<Gate<T>> gates4;
    std::vector<std::vector<int>> qubits4;
    for (size_t index = 0; index < gates3.size(); index++){
        const Gate<T>& gate = gates3[index];
        const std::vector<int>& qubits = qubits3[index];
        if (qubits.size() == 1) continue;
        if (!merges.count(index)) {
            gates4.push_back(gate);
            qubits4.push_back(qubits);
            continue;
        }
        std::vector<T> U = util::identity4<T>();
        for (auto merge : merges[index]) {
            size_t index2 = std::get<0>(merge);
            if (index2 > index) continue;
            int wire = std::get<1>(merge);
            const Gate<T>& gate2 = gates3[index2]; 
            // [RMP: kron convention checked]
            std::vector<T> U2 = (wire == 0 ? util::kron2I2<T>(gate2.matrix()) : util::kron2I1<T>(gate2.matrix()));
            U = util::multiply4<T>(U2, U);
        }
        U = util::multiply4<T>(gate.matrix(), U);
        for (auto merge : merges[index]) {
            size_t index2 = std::get<0>(merge);
            if (index2 < index) continue;
            int wire = std::get<1>(merge);
            const Gate<T>& gate2 = gates3[index2]; 
            // [RMP: kron convention checked]
            std::vector<T> U2 = (wire == 0 ? util::kron2I2<T>(gate2.matrix()) : util::kron2I1<T>(gate2.matrix()));
            U = util::multiply4<T>(U2, U);
        }
        gates4.push_back(Gate<T>(
            2,
            "U2",
            U));
        qubits4.push_back(qubits);
    }

    // Merge neighboring 2-body gates
    std::vector<Gate<T>> gates5;
    std::vector<std::vector<int>> qubits5;
    std::vector<Gate<T>> gates_stack;
    std::vector<std::vector<int>> qubits_stack;
    for (size_t index = 0; index < gates4.size(); index++) {
        const Gate<T>& gate = gates4[index];
        const std::vector<int>& qubits = qubits4[index];
        bool found = false;
        std::vector<size_t> clashes;
        for (size_t index2 = 0; index2 < gates_stack.size(); index2++) {
            const Gate<T>& gate2 = gates_stack[index2];
            const std::vector<int>& qubits2 = qubits_stack[index2];
            if (qubits[0] == qubits2[0] && qubits[1] == qubits2[1]) {
                // Match [RMP: no bitflip required, as expected]
                std::vector<T> U = vulcan::util::multiply4<T>(gate.matrix(), gate2.matrix());
                gates_stack[index2] = Gate<T>(
                    2,
                    "U2",
                    U);
                found = true;
                break;
            } else if (qubits[0] == qubits2[1] && qubits[1] == qubits2[0]) {
                // Match (transposed) [RMP: bitflip required, as expected]
                std::vector<T> U = vulcan::util::multiply4<T>(vulcan::util::bitflip4<T>(gate.matrix()), gate2.matrix());
                gates_stack[index2] = Gate<T>(
                    2,
                    "U2",
                    U);
                found = true;
                break;
            } else if (qubits[0] == qubits2[0] || qubits[0] == qubits2[1] || qubits[1] == qubits2[0] || qubits[1] == qubits2[1]) {
                // Clash
                clashes.push_back(index2);
                found = true;
            }
        }

        // Retire clashes
        if (clashes.size()) {
            ssize_t swap_index = gates_stack.size() - 1;
            for (ssize_t index3 = clashes.size() - 1; index3 >= 0; index3--) {
                size_t index = clashes[index3];
                gates5.push_back(gates_stack[index]);
                qubits5.push_back(qubits_stack[index]);
                gates_stack[index] = gates_stack[swap_index];
                qubits_stack[index] = qubits_stack[swap_index];
                gates_stack.pop_back();
                qubits_stack.pop_back();
                swap_index--;
            }
        }

        // Not found or clashes
        if (!found || clashes.size()) {
            gates_stack.push_back(gate);
            qubits_stack.push_back(qubits);
        }
    }
    for (size_t index = 0; index < gates_stack.size(); index++) {
        gates5.push_back(gates_stack[index]);
        qubits5.push_back(qubits_stack[index]);
    }

    // Wedge isolated 1-body gates into 2-body gates
    std::vector<Gate<T>> gates0f;
    std::vector<std::vector<int>> qubits0f;
    for (size_t index = 0; index < gates0.size(); index+=2) {
        if (index + 1 == gates0.size()) {
            const Gate<T>& gate = gates0[index];
            const std::vector<int>& qubits = qubits0[index];
            gates0f.push_back(gate);
            qubits0f.push_back(qubits);
        } else {
            const Gate<T>& gate1 = gates0[index + 0];
            const std::vector<int>& qubits1 = qubits0[index + 0];
            const Gate<T>& gate2 = gates0[index + 1];
            const std::vector<int>& qubits2 = qubits0[index + 1];
            std::vector<T> U = vulcan::util::kron2<T>(gate1.matrix(), gate2.matrix());
            gates0f.push_back(Gate<T>(
                2,
                "U2",
                U));
            qubits0f.push_back({qubits1[0], qubits2[0]});
        }
    }

    gates0f.insert(gates0f.end(), gates5.begin(), gates5.end());
    qubits0f.insert(qubits0f.end(), qubits5.begin(), qubits5.end());

    return Circuit(
        nqubit_,
        gates0f,
        qubits0f);
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

// => Vulcan C++ API Functions <= //

template <typename T>
void run_statevector(
    const Circuit<T>& circuit,
    T* statevector_h,
    T* result_h,
    bool compressed)
{
    T* statevector_d = vulcan::malloc_and_initialize_statevector(circuit.nqubit(), statevector_h);
    vulcan::util::run_statevector(compressed ? circuit.compressed() : circuit, statevector_d);
    vulcan::copy_statevector_to_host(circuit.nqubit(), statevector_d, result_h);
    vulcan::free_statevector(statevector_d);
}

template <typename T>
void run_pauli_sigma(
    const Pauli<T>& pauli,
    T* statevector_h,
    T* result_h,
    bool compressed)
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
    const T* statevector_h,
    bool compressed)
{
    if (circuit.nqubit() != pauli.nqubit()) {
        throw std::runtime_error("circuit and pauli do not have same nqubit");
    }
    
    T* statevector1_d = vulcan::malloc_and_initialize_statevector(pauli.nqubit(), statevector_h);
    T* statevector2_d = vulcan::malloc_statevector<T>(pauli.nqubit());
    vulcan::util::run_statevector(compressed ? circuit.compressed() : circuit, statevector1_d);
    Pauli<T> expectation = vulcan::util::pauli_expectation(pauli, statevector1_d, statevector2_d);
    vulcan::free_statevector(statevector1_d);
    vulcan::free_statevector(statevector2_d);
    return expectation; 
}

template <typename T>
T run_pauli_expectation_value(
    const Circuit<T>& circuit,
    const Pauli<T>& pauli,
    const T* statevector_h,
    bool compressed)
{
    if (pauli.nqubit() != circuit.nqubit()) {
	throw std::runtime_error("pauli and circuit must have same nqubit");
    }

    T* statevector1_d = vulcan::malloc_and_initialize_statevector(pauli.nqubit(), statevector_h);
    T* statevector2_d = vulcan::malloc_statevector<T>(pauli.nqubit());
    T* statevector3_d = vulcan::malloc_statevector<T>(pauli.nqubit());
    vulcan::util::run_statevector(compressed ? circuit.compressed() : circuit, statevector1_d);
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
    const T* statevector_h, 
    bool compressed)
{
    if (pauli.nqubit() != circuit.nqubit()) {
	throw std::runtime_error("pauli and circuit must have same nqubit");
    }

    T* statevector1_d = vulcan::malloc_and_initialize_statevector(pauli.nqubit(), statevector_h);
    T* statevector2_d = vulcan::malloc_statevector<T>(pauli.nqubit());
    T* statevector3_d = vulcan::malloc_statevector<T>(pauli.nqubit());
    vulcan::util::run_statevector(compressed ? circuit.compressed() : circuit, statevector1_d);
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

template <typename T, typename U>
void run_measurement(
    const Circuit<T>& circuit,
    const T* statevector_h,
    int nmeasurement,
    const U* randoms_h,
    int* measurements_h,
    bool compressed)
{
    T* statevector_d = vulcan::malloc_and_initialize_statevector(circuit.nqubit(), statevector_h);
    vulcan::util::run_statevector(compressed ? circuit.compressed() : circuit, statevector_d);

    U* pvector_d;
    if (sizeof(T) == sizeof(U)) {
        pvector_d = (U*) statevector_d; // in-place with real types
    } else if (sizeof(T) == 2*sizeof(U)) {
        pvector_d = vulcan::malloc_statevector<U>(circuit.nqubit()); // Temporary with complex types
    } else {
        throw std::runtime_error("T/U sizes do not make sense");
    }

    vulcan::abs2<T, U>(circuit.nqubit(), statevector_d, pvector_d);

    U sum = vulcan::cumsum<U>(circuit.nqubit(), pvector_d);

    U* randoms_d;
    cudaMalloc(&randoms_d, sizeof(U)*nmeasurement);
    cudaMemcpy(randoms_d, randoms_h, sizeof(U)*nmeasurement, cudaMemcpyHostToDevice);

    int* measurements_d;
    cudaMalloc(&measurements_d, sizeof(int)*nmeasurement);

    vulcan::measure(circuit.nqubit(), pvector_d, sum, nmeasurement, randoms_d, measurements_d); 

    cudaMemcpy(measurements_h, measurements_d, sizeof(int)*nmeasurement, cudaMemcpyDeviceToHost);

    vulcan::free_statevector(statevector_d);
    if (sizeof(T) != sizeof(U)) {
        vulcan::free_statevector(pvector_d);
    }
    cudaFree(randoms_d);
    cudaFree(measurements_d);
}
    
} // namespace vulcan
